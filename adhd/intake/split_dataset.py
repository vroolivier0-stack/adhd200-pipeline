"""Séparer les patients selon une stratégie enregistrée.

Les listes publiques au sein du projet de données utilisent les codes
protégés. Les liens vers les fichiers bruts restent dans un rapport privé.
Toutes ces listes restent des données à accès limité.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import yaml

from adhd.intake.validate_dataset import fingerprint, save_json


def make_assignments(records, roles, seed, fraction, test_fraction):
    patients = {}
    groups = defaultdict(list)
    assignments = {}

    for record in records:
        pseudo = record["pseudo_id"]
        identity = (record["site"], record["target_binary"])

        if pseudo in patients and patients[pseudo] != identity:
            raise RuntimeError("Centre ou classe contradictoire pour un patient.")
        patients[pseudo] = identity

    for pseudo, (site, target) in sorted(patients.items()):
        role = roles.get(site)

        if role == "arrivals_reserve":
            if target is not None:
                raise RuntimeError("Un patient étiqueté figure dans le groupe sans cible.")
            assignments[pseudo] = "arrivals_reserve"

        elif role == "external_test":
            if target not in (0, 1):
                raise RuntimeError("Le test externe exige une cible connue.")
            assignments[pseudo] = "test_external"

        elif role == "external_controls":
            if target != 0:
                raise RuntimeError("Le groupe de témoins externes contient une autre cible.")
            assignments[pseudo] = "controls_external"

        elif role == "train_validation":
            if target not in (0, 1):
                raise RuntimeError("Apprentissage et validation exigent une cible connue.")
            groups[(site, target)].append(pseudo)

        else:
            raise RuntimeError("Centre absent ou rôle inconnu dans la configuration.")

    for (site, target), identities in sorted(groups.items()):
        if len(identities) < 3:
            raise RuntimeError(
                "Un groupe centre/classe contient moins de trois patients. "
                "La stratégie doit être réexaminée."
            )

        # Un classement déterministe évite de dépendre de l'ordre des fichiers.
        # La graine, le centre, la classe et le code patient fixent ce classement.
        def order(pseudo):
            message = f"split:v1:{seed}:{site}:{target}:{pseudo}".encode()
            return (hashlib.sha256(message).hexdigest(), pseudo)

        ordered = sorted(identities, key=order)
        validation_count = max(1, round(len(ordered) * fraction))
        test_count = max(1, round(len(ordered) * test_fraction))

        if validation_count + test_count >= len(ordered):
            raise RuntimeError(
                "Ce groupe centre/classe ne permet pas les trois usages. "
                "La stratégie doit être réexaminée."
            )

        for position, pseudo in enumerate(ordered):
            if position < validation_count:
                assignments[pseudo] = "validation"
            elif position < validation_count + test_count:
                assignments[pseudo] = "test_internal"
            else:
                assignments[pseudo] = "train"

    return assignments


def run(matching_file, config_file, private_dir, splits_dir, summary_dir):
    source = json.loads(matching_file.read_text(encoding="utf-8"))
    if source.get("matching_version") != "matching_v1":
        raise RuntimeError("Format de rapprochement non reconnu.")

    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise RuntimeError("Format de configuration non reconnu.")

    roles = config["roles"]
    seed = config["seed"]
    fraction = float(config["validation_fraction"])
    test_fraction = float(config["test_fraction"])

    if (
        not isinstance(seed, int)
        or not 0 < fraction < 1
        or not 0 < test_fraction < 1
        or fraction + test_fraction >= 1
    ):
        raise RuntimeError("Graine ou proportion invalide.")

    records = source["labeled_records"] + source["unlabeled_records"]
    assignments = make_assignments(records, roles, seed, fraction, test_fraction)

    # Vérifier que l'ordre d'entrée ne change pas les groupes.
    repeated = make_assignments(list(reversed(records)), roles, seed, fraction, test_fraction)
    if repeated != assignments:
        raise RuntimeError("La répartition dépend de l'ordre des images.")

    patient_sets = defaultdict(set)
    content_splits = defaultdict(set)
    private_records = []
    working_records = []

    for record in records:
        pseudo = record["pseudo_id"]
        split = assignments[pseudo]
        patient_sets[split].add(pseudo)
        content_splits[record["source_sha256"]].add(split)

        private_records.append({**record, "split": split})

        # Ne pas transmettre les chemins bruts dans la liste de travail.
        working_records.append({
            "pseudo_id": pseudo,
            "site": record["site"],
            "target_binary": record["target_binary"],
            "split": split,
        })

    names = sorted(patient_sets)
    for position, first in enumerate(names):
        for second in names[position + 1:]:
            if patient_sets[first] & patient_sets[second]:
                raise RuntimeError("Un patient apparaît dans plusieurs groupes.")

    if any(len(splits) > 1 for splits in content_splits.values()):
        raise RuntimeError("Un contenu identique apparaît dans plusieurs groupes.")

    summaries = {}
    for split in names:
        group = [
            record for record in records
            if assignments[record["pseudo_id"]] == split
        ]
        unique = {record["pseudo_id"]: record for record in group}
        classes = Counter(record["target_binary"] for record in unique.values())
        sites = Counter(record["site"] for record in unique.values())

        summaries[split] = {
            "patients": len(unique),
            "images": len(group),
            "controls": classes[0],
            "adhd": classes[1],
            "without_target": classes[None],
            "by_site": dict(sorted(sites.items())),
        }

    train = summaries.get("train", {})
    if train.get("controls", 0) == 0 or train.get("adhd", 0) == 0:
        raise RuntimeError("Les deux classes doivent être présentes dans l'apprentissage.")

    # Les groupes de réglage et de test doivent contenir les deux classes.
    for name in ("validation", "test_internal", "test_external"):
        group = summaries.get(name, {})
        if group.get("controls", 0) == 0 or group.get("adhd", 0) == 0:
            raise RuntimeError(
                "Un groupe d'évaluation ne contient pas les deux classes."
            )

    # Vérifier explicitement le respect des rôles de centres.
    for record in records:
        split = assignments[record["pseudo_id"]]
        role = roles[record["site"]]
        if split in {"train", "validation", "test_internal"}:
            if role != "train_validation":
                raise RuntimeError("Un centre réservé contamine l'apprentissage.")
        if role == "arrivals_reserve" and split != "arrivals_reserve":
            raise RuntimeError("Une arrivée réservée figure dans un autre groupe.")

    # Référence pour une éventuelle pondération de la classe ADHD.
    # Calculée exclusivement sur l'apprentissage, jamais sur le test.
    train_positive_weight = train["controls"] / train["adhd"]

    recorded = datetime.now(timezone.utc)
    run_id = recorded.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]

    for directory in (private_dir, splits_dir, summary_dir):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    metadata = {
        "schema_version": 1,
        "split_version": "split_v2",
        "run_id": run_id,
        "source_matching_run_id": source["run_id"],
        "source_matching_sha256": fingerprint(matching_file),
        "config_sha256": fingerprint(config_file),
        "configuration": config,
        "recorded_at": recorded.isoformat(),
    }

    private_path = private_dir / f"split_private_{run_id}.json"
    save_json(private_path, {**metadata, "records": private_records})

    working_path = splits_dir / f"splits_{run_id}.json"
    save_json(working_path, {**metadata, "records": working_records})

    summary_path = summary_dir / f"split_summary_{run_id}.json"
    save_json(summary_path, {
        **metadata,
        "groups": summaries,
        "train_positive_weight_reference": train_positive_weight,
        "checks": {
            "patient_groups_disjoint": True,
            "identical_content_groups_disjoint": True,
            "input_order_independent": True,
        },
        "limits": [
            "Le test externe combine différences de centre et de population.",
            "Les témoins externes ne permettent pas une évaluation des deux classes.",
            "Aucune garantie de test inédit pour les modèles précédemment entraînés.",
        ],
    })

    print("RÉPARTITION DES PATIENTS TERMINÉE")
    for split, summary in summaries.items():
        print(
            f"{split} : {summary['patients']} patients"
            f" | témoins : {summary['controls']}"
            f" | ADHD : {summary['adhd']}"
            f" | sans cible : {summary['without_target']}"
        )
        print(f"  Centres : {summary['by_site']}")

    print("\nContrôles réussis :")
    print("Un patient appartient à un seul groupe.")
    print("Aucun fichier identique partagé entre groupes.")
    print("La répartition ne dépend pas de l'ordre des images.")
    print(f"Référence de pondération ADHD, apprentissage seul : {train_positive_weight:.4f}")
    print(f"\nListe de travail protégée : {working_path.name}")
    print(f"Correspondances privées : {private_path.name}")
    print(f"Résumé agrégé : {summary_path.name}")
    print("Aucune IRM transformée ; aucun modèle entraîné.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching-file", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.matching_file, args.config_file, args.private_dir,
        args.splits_dir, args.summary_dir,
    )


if __name__ == "__main__":
    main()
