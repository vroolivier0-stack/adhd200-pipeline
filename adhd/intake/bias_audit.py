"""Décrire les déséquilibres des données avant leur répartition.

Cet audit ne décide pas automatiquement d'exclure un centre.
Les résultats décrivent les données ; ils ne mesurent pas encore
les différences de performance du modèle entre groupes.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import uuid

from adhd.intake.validate_dataset import fingerprint, save_json


def summarize_age(records):
    """Décrire les âges renseignés sans inventer les valeurs manquantes."""
    ages = []
    unavailable = 0

    for record in records:
        text = record.get("age_source_value", "").strip()
        try:
            value = float(text)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Âge inexploitable.")
            ages.append(value)
        except (ValueError, TypeError):
            unavailable += 1

    return {
        "usable_count": len(ages),
        "unavailable_or_invalid_count": unavailable,
        "minimum": min(ages) if ages else None,
        "median": statistics.median(ages) if ages else None,
        "maximum": max(ages) if ages else None,
    }


def summarize_group(records):
    classes = Counter(record["target_binary"] for record in records)
    total = len(records)
    genders = Counter()

    for record in records:
        code = record.get("gender_source_code", "").strip()
        genders[code if code in {"0", "1"} else "unavailable_or_invalid"] += 1

    return {
        "patients": total,
        "control_count": classes[0],
        "adhd_count": classes[1],
        "adhd_percentage": round(100 * classes[1] / total, 2) if total else None,
        "gender_codes": dict(genders),
        "age": summarize_age(records),
    }


def run(matching_file, output_directory):
    source = json.loads(matching_file.read_text(encoding="utf-8"))
    if (
        source.get("schema_version") != 1
        or source.get("matching_version") != "matching_v1"
    ):
        raise RuntimeError("Format du rapport de rapprochement non reconnu.")

    labeled = source["labeled_records"]
    unlabeled = source["unlabeled_records"]

    # Compter les patients, même si plusieurs images deviennent disponibles.
    patients = {}
    image_counts = Counter()
    content_patients = defaultdict(set)

    for record in labeled:
        pseudo = record["pseudo_id"]
        if record["target_binary"] not in (0, 1):
            raise RuntimeError("Une cible renseignée doit être 0 ou 1.")

        previous = patients.get(pseudo)
        if previous is not None:
            fields = (
                "site", "target_binary",
                "age_source_value", "gender_source_code",
            )
            if any(previous.get(field) != record.get(field) for field in fields):
                raise RuntimeError("Informations contradictoires pour un patient.")

        patients[pseudo] = record
        image_counts[pseudo] += 1
        content_patients[record["source_sha256"]].add(pseudo)

    labeled_ids = set(patients)
    unlabeled_ids = {record["pseudo_id"] for record in unlabeled}
    if labeled_ids & unlabeled_ids:
        raise RuntimeError("Un patient figure avec et sans cible.")

    # Inclure les images sans cible dans la recherche de contenus identiques.
    for record in unlabeled:
        content_patients[record["source_sha256"]].add(record["pseudo_id"])

    identical_content_groups = [
        group for group in content_patients.values() if len(group) > 1
    ]

    by_site = defaultdict(list)
    for record in patients.values():
        by_site[record["site"]].append(record)

    summaries = {
        site: summarize_group(records)
        for site, records in sorted(by_site.items())
    }

    observations = []
    for site, summary in summaries.items():
        controls = summary["control_count"]
        adhd = summary["adhd_count"]

        if controls == 0 or adhd == 0:
            observations.append({
                "site": site,
                "type": "single_class_site",
                "explanation": "Un seul groupe est présent dans ce centre.",
            })
        elif min(controls, adhd) / summary["patients"] < 0.10:
            observations.append({
                "site": site,
                "type": "minority_below_10_percent",
                "explanation": (
                    "Le groupe minoritaire représente moins de 10 % du centre. "
                    "Ce seuil est un repère descriptif, pas une règle médicale."
                ),
            })

    global_summary = summarize_group(list(patients.values()))
    recorded = datetime.now(timezone.utc)
    run_id = recorded.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]

    report = {
        "schema_version": 1,
        "audit_version": "bias_audit_v1",
        "run_id": run_id,
        "recorded_at": recorded.isoformat(),
        "source_matching_run_id": source["run_id"],
        "source_matching_sha256": fingerprint(matching_file),
        "labeled_image_count": len(labeled),
        "distinct_labeled_patients": len(patients),
        "distinct_unlabeled_patients": len(unlabeled_ids),
        "patients_with_multiple_labeled_images": sum(
            count > 1 for count in image_counts.values()
        ),
        "identical_content_groups_across_patients": len(identical_content_groups),
        "patients_in_identical_content_groups": len(
            set().union(*identical_content_groups)
        ) if identical_content_groups else 0,
        "global": global_summary,
        "by_site": summaries,
        "observations": observations,
        "limits": [
            "Les performances du modèle par groupe ne sont pas encore mesurées.",
            "Les fichiers sont comparés par leur empreinte binaire.",
            "Des conversions différentes d'une même acquisition peuvent échapper "
            "à cette comparaison.",
            "Aucun centre n'est exclu par cet audit.",
        ],
    }

    output_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = output_directory / f"bias_audit_{run_id}.json"
    save_json(destination, report)

    print("AUDIT DES DONNÉES TERMINÉ")
    print(f"Patients avec cible : {len(patients)}")
    print(f"Patients sans cible : {len(unlabeled_ids)}")
    print(
        f"Ensemble : {global_summary['control_count']} témoins"
        f" | {global_summary['adhd_count']} ADHD"
        f" | ADHD : {global_summary['adhd_percentage']} %"
    )

    print("\nRÉPARTITION PAR CENTRE")
    for site, summary in summaries.items():
        print(
            f"{site} : {summary['patients']} patients"
            f" | témoins : {summary['control_count']}"
            f" | ADHD : {summary['adhd_count']}"
            f" | ADHD : {summary['adhd_percentage']} %"
        )
        print(f"  Codes de sexe : {summary['gender_codes']}")
        print(f"  Âges : {summary['age']}")

    print("\nPOINTS À EXAMINER")
    for observation in observations:
        print(f"{observation['site']} : {observation['explanation']}")

    print(
        "Groupes de fichiers identiques associés à plusieurs patients :",
        len(identical_content_groups),
    )
    print(f"\nRapport agrégé : {destination.name}")
    print("Aucun identifiant patient ni chemin d'IRM dans ce rapport.")
    print("Aucun centre exclu ; aucune donnée modifiée.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.matching_file, args.output_dir)


if __name__ == "__main__":
    main()
