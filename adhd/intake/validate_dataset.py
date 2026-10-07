"""Construire les listes de travail et de quarantaine.

Les fichiers bruts sont lus, sans être déplacés ou modifiés.
Les rapports détaillés contiennent des chemins : ils restent privés.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

from adhd.intake.filter_modalities import classify
from adhd.intake.technical_profile import discover_images, inspect_image


def fingerprint(path):
    """Calculer la signature du contenu d'un fichier."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, content):
    """Publier un nouveau rapport complet, accessible à son propriétaire."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        os.chmod(temporary, 0o600)
        json.dump(
            content, stream, ensure_ascii=False,
            indent=2, allow_nan=False,
        )
        stream.write("\n")
    temporary.replace(path)


def run(root, report_dir, quarantine_dir, review_file):
    if not root.is_dir():
        raise RuntimeError("Dossier brut introuvable.")

    decision = json.loads(review_file.read_text(encoding="utf-8"))
    if (
        decision.get("schema_version") != 1
        or decision.get("decision") != "exclude_until_review"
        or decision.get("review_status") != "pending"
    ):
        raise RuntimeError("Format ou état de la décision non pris en charge.")

    reviewed_path = Path(decision["source_relative_path"])
    if reviewed_path.is_absolute() or ".." in reviewed_path.parts:
        raise RuntimeError("Chemin de décision non autorisé.")
    expected_hash = decision["source_sha256"]
    if len(expected_hash) != 64:
        raise RuntimeError("Empreinte de décision invalide.")

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    accepted = []
    rejected = []
    excluded_rest = []
    counts = Counter()
    reviewed_found = False

    for path in discover_images(root):
        relative = path.relative_to(root)
        modality = classify(relative)

        if modality == "rest":
            counts["rest"] += 1
            excluded_rest.append(str(relative))
            continue

        reasons = []
        record = {
            "source_relative_path": str(relative),
            "site": relative.parts[0],
            "modality": modality,
        }

        try:
            record["source_size_bytes"] = path.stat().st_size
            record["source_sha256"] = fingerprint(path)

            if modality == "unsupported":
                reasons.append("type_image_non_pris_en_charge")
            else:
                technical = inspect_image(path)
                record["technical_check"] = technical
                reasons.extend(technical["reasons"])

                # Le prétraitement prévu utilise des distances en millimètres.
                # Une autre unité devra être traitée explicitement.
                if (
                    technical["status"] == "ok"
                    and technical.get("spatial_unit") != "mm"
                ):
                    reasons.append("unite_spatiale_a_verifier")

            if relative == reviewed_path:
                reviewed_found = True
                record["review_decision_id"] = decision["decision_id"]

                if record["source_sha256"] == expected_hash:
                    reasons.append(decision["reason"])
                else:
                    # Ne pas appliquer une ancienne décision à un
                    # fichier dont le contenu a changé.
                    reasons.append("fichier_modifie_decision_a_reexaminer")

        except Exception as error:
            reasons.append("lecture_ou_empreinte_impossible")
            record["error_type"] = type(error).__name__

        record["reasons"] = sorted(set(reasons))

        if reasons:
            record["status"] = "quarantined"
            rejected.append(record)
            counts["quarantined"] += 1
        else:
            record["status"] = "accepted"
            accepted.append(record)
            counts["accepted"] += 1

        counts["checked_non_rest"] += 1
        if counts["checked_non_rest"] % 25 == 0:
            print(
                f"Avancement : {counts['checked_non_rest']} fichiers contrôlés",
                flush=True,
            )

    if counts["checked_non_rest"] == 0:
        raise RuntimeError("Aucune image hors REST trouvée.")

    finished = datetime.now(timezone.utc)

    # Ces emplacements sont montés séparément de raw.
    for directory in (report_dir, quarantine_dir):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    quarantine_path = quarantine_dir / f"quarantine_{run_id}.json"
    save_json(quarantine_path, {
        "schema_version": 1,
        "run_id": run_id,
        "isolation_method": "exclusion_from_accepted_manifest",
        "raw_preserved": True,
        "records": rejected,
    })

    manifest_path = report_dir / f"validation_{run_id}.json"
    save_json(manifest_path, {
        "schema_version": 1,
        "validation_version": "validation_v1",
        "run_id": run_id,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "review_file": review_file.name,
        "review_file_sha256": fingerprint(review_file),
        "reviewed_source_found": reviewed_found,
        "quarantine_manifest": quarantine_path.name,
        "summary": dict(counts),
        "accepted_records": accepted,
        "excluded_rest_paths": excluded_rest,
    })

    print("\nVALIDATION TERMINÉE")
    print(f"Images acceptées : {counts['accepted']}")
    print(f"Images en quarantaine : {counts['quarantined']}")
    print(f"Images REST exclues : {counts['rest']}")
    print(f"Image de la décision OHSU retrouvée : {reviewed_found}")

    reasons = Counter(
        reason for record in rejected for reason in record["reasons"]
    )
    print(f"Motifs de quarantaine : {dict(reasons)}")
    print(f"Durée : {(finished - started).total_seconds():.1f} secondes")
    print(f"Liste privée des images acceptées : {manifest_path.name}")
    print(f"Liste privée de quarantaine : {quarantine_path.name}")
    print("Aucun fichier brut modifié.")

    if not reviewed_found:
        print(
            "ATTENTION : l'image concernée par la décision OHSU "
            "n'est plus présente dans les fichiers parcourus."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--quarantine-dir", type=Path, required=True)
    parser.add_argument("--review-file", type=Path, required=True)
    args = parser.parse_args()
    run(args.raw_root, args.report_dir, args.quarantine_dir, args.review_file)


if __name__ == "__main__":
    main()
