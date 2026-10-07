"""Associer chaque image acceptée à son patient.

Les chemins d'origine et les informations patients restent dans des rapports
privés. Aucun fichier destiné à Kaggle n'est produit par cette étape.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re
import uuid

from adhd.identity import (
    SITE_CODES, load_key, normalize_identifier, pseudonymize,
)
from adhd.intake.validate_dataset import fingerprint, save_json


# Source : clé officielle des informations patients ADHD-200.
DX_TARGETS = {"0": 0, "1": 1, "2": 1, "3": 1}
DX_SOURCE = (
    "https://fcon_1000.projects.nitrc.org/indi/adhd200/general/"
    "ADHD-200_PhenotypicKey.pdf"
)


def load_phenotypes(path):
    """Isoler les lignes invalides et les identifiants ambigus."""
    index = {}
    duplicates = set()
    issues = []

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"ScanDir ID", "Site", "DX", "Age", "Gender"}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError("Colonnes indispensables absentes du tableau.")

        for number, row in enumerate(reader, start=2):
            if not any(v.strip() for v in row.values() if isinstance(v, str)):
                continue

            try:
                patient = normalize_identifier(row["ScanDir ID"])
                site = normalize_identifier(row["Site"])
                if site not in SITE_CODES.values():
                    raise ValueError("Centre non reconnu.")
            except (ValueError, TypeError):
                issues.append({
                    "line_number": number,
                    "reason": "identifiant_ou_centre_invalide",
                })
                continue

            identity = (site, patient)

            if identity in index or identity in duplicates:
                index.pop(identity, None)
                duplicates.add(identity)
                issues.append({
                    "line_number": number,
                    "reason": "identifiant_patient_duplique",
                })
                continue

            index[identity] = row

    return index, issues


def run(validation_file, phenotype_file, key_file, report_dir, quarantine_dir):
    validation = json.loads(validation_file.read_text(encoding="utf-8"))
    if (
        validation.get("schema_version") != 1
        or validation.get("validation_version") != "validation_v1"
    ):
        raise RuntimeError("Format de liste de validation non pris en charge.")

    records = validation["accepted_records"]
    if not records:
        raise RuntimeError("La liste des images acceptées est vide.")

    key = load_key(key_file)
    # Référence de la clé utilisée, sans enregistrer la clé elle-même.
    key_reference = hashlib.sha256(key).hexdigest()[:16]
    patients, phenotype_issues = load_phenotypes(phenotype_file)

    labeled = []
    unlabeled = []
    rejected = []
    seen_paths = set()
    pseudo_identities = {}
    images_per_patient = Counter()
    distribution = defaultdict(Counter)

    for record in records:
        if record.get("status") != "accepted":
            raise RuntimeError("Une image non acceptée figure dans la liste.")

        relative = Path(record["source_relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Chemin source non autorisé.")
        if str(relative) in seen_paths:
            raise RuntimeError("Une image figure plusieurs fois dans la liste.")
        seen_paths.add(str(relative))

        site_name = relative.parts[0]
        site_code = SITE_CODES.get(site_name)
        reasons = []
        candidates = set()

        for part in relative.parts[1:-1]:
            match = re.fullmatch(r"(?:sub-)?([0-9]{5,7})", part)
            if match:
                candidates.add(normalize_identifier(match.group(1)))

        if site_code is None:
            reasons.append("centre_non_reconnu")
        if len(candidates) != 1:
            reasons.append("identifiant_absent_ou_ambigu")

        row = None
        if not reasons:
            patient = next(iter(candidates))
            row = patients.get((site_code, patient))
            if row is None:
                reasons.append("ligne_patient_absente_ou_ambigue")

        if reasons:
            rejected.append({
                "source_relative_path": str(relative),
                "source_sha256": record["source_sha256"],
                "reasons": reasons,
            })
            continue

        diagnosis = (row.get("DX") or "").strip().lower()
        if diagnosis not in DX_TARGETS and diagnosis not in {"", "pending"}:
            rejected.append({
                "source_relative_path": str(relative),
                "source_sha256": record["source_sha256"],
                "reasons": ["code_diagnostic_non_reconnu"],
            })
            continue

        pseudo_id = pseudonymize(site_code, patient, key)
        identity = (site_code, patient)

        # Une collision ne doit jamais associer deux patients différents.
        if (
            pseudo_id in pseudo_identities
            and pseudo_identities[pseudo_id] != identity
        ):
            raise RuntimeError("Collision entre deux codes patients protégés.")
        pseudo_identities[pseudo_id] = identity
        images_per_patient[pseudo_id] += 1

        target = DX_TARGETS.get(diagnosis)
        matched = {
            "pseudo_id": pseudo_id,
            "site": site_name,
            "site_code": site_code,
            "source_relative_path": str(relative),
            "source_sha256": record["source_sha256"],
            "dx_source_code": diagnosis or None,
            "target_binary": target,
            # Garder ces champs pour l'étude des biais.
            # Ils ne sont pas automatiquement des entrées du CNN.
            "age_source_value": (row.get("Age") or "").strip(),
            "gender_source_code": (row.get("Gender") or "").strip(),
        }

        if target is None:
            unlabeled.append(matched)
            distribution[site_name]["sans_cible"] += 1
        else:
            labeled.append(matched)
            distribution[site_name][f"cible_{target}"] += 1

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]

    for directory in (report_dir, quarantine_dir):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    quarantine_path = quarantine_dir / f"matching_quarantine_{run_id}.json"
    save_json(quarantine_path, {
        "schema_version": 1,
        "run_id": run_id,
        "rejected_images": rejected,
        "phenotype_row_issues": phenotype_issues,
    })

    destination = report_dir / f"matching_{run_id}.json"
    save_json(destination, {
        "schema_version": 1,
        "matching_version": "matching_v1",
        "run_id": run_id,
        "recorded_at": started.isoformat(),
        "source_validation_run_id": validation["run_id"],
        "validation_manifest_sha256": fingerprint(validation_file),
        "phenotype_file_sha256": fingerprint(phenotype_file),
        "pseudonymization_key_reference": key_reference,
        "diagnosis_code_source": DX_SOURCE,
        "quarantine_manifest": quarantine_path.name,
        "summary": {
            "input_images": len(records),
            "labeled_images": len(labeled),
            "unlabeled_images": len(unlabeled),
            "rejected_images": len(rejected),
            "distinct_matched_patients": len(pseudo_identities),
            "phenotype_row_issues": len(phenotype_issues),
        },
        "labeled_records": labeled,
        "unlabeled_records": unlabeled,
    })

    print("RAPPROCHEMENT ET PSEUDONYMISATION TERMINÉS")
    print(f"Images reçues : {len(records)}")
    print(f"Images avec cible renseignée : {len(labeled)}")
    print(f"Images sans cible : {len(unlabeled)}")
    print(f"Images rejetées au rapprochement : {len(rejected)}")
    print(f"Lignes patients présentant un problème : {len(phenotype_issues)}")
    print(f"Patients distincts associés : {len(pseudo_identities)}")
    print("Patients avec plusieurs images :", sum(
        count > 1 for count in images_per_patient.values()
    ))

    print("\nRÉPARTITION PAR CENTRE")
    for site, counts in sorted(distribution.items()):
        print(f"{site} : {dict(counts)}")

    print(f"\nRapport privé : {destination.name}")
    print(f"Rapport privé de quarantaine : {quarantine_path.name}")
    print("Aucune IRM ni clé modifiée.")
    print("Aucun découpage entraînement/validation/test effectué.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--phenotype-file", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--quarantine-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.validation_file, args.phenotype_file, args.key_file,
        args.report_dir, args.quarantine_dir,
    )


if __name__ == "__main__":
    main()
