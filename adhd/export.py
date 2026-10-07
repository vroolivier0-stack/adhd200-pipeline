"""Créer localement un paquet d'entraînement pour Kaggle.

Seuls les groupes train et validation sont exportés.
Les correspondances avec les codes patients restent dans un rapport privé.
Les tableaux numériques peuvent encore contenir l'anatomie du visage :
cet export ne constitue pas une anonymisation complète.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import uuid

import nibabel as nib
import numpy as np

from adhd.utils import fingerprint, save_json


def safe_child(root, relative):
    """Empêcher qu'un chemin sorte du dossier autorisé."""
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("Chemin relatif non autorisé.")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise RuntimeError("Le chemin sort du dossier autorisé.")
    if path.is_symlink():
        raise RuntimeError("Un lien symbolique n'est pas autorisé.")
    return path


def run(args):
    os.umask(0o077)

    preparation = json.loads(
        args.preparation_report.read_text(encoding="utf-8")
    )
    if (
        preparation.get("preparation_version") != "historical_preparation_v1"
        or preparation.get("status") != "complete_for_selected_scope"
        or preparation.get("scope_limited") is not False
        or preparation.get("summary", {}).get("failed") != 0
    ):
        raise RuntimeError("Un rapport de préparation complet et réussi est requis.")

    prepared_root = safe_child(
        args.processed_root, preparation["output_directory"]
    )
    if not prepared_root.is_dir():
        raise RuntimeError("Dossier des volumes préparés introuvable.")

    records = sorted(
        (
            row for row in preparation["prepared_records"]
            if row["split"] in {"train", "validation"}
        ),
        key=lambda row: (
            row["split"], row["pseudo_id"], row["source_sha256"]
        ),
    )
    counts = Counter(row["split"] for row in records)
    if dict(counts) != {"train": 563, "validation": 119}:
        raise RuntimeError(
            "Ce premier export attend 563 images train et 119 validation."
        )

    seen_images = set()
    subjects = {}
    contents = {}
    for row in records:
        if row["site"] == "Brown" or row["target_binary"] not in (0, 1):
            raise RuntimeError("Image non autorisée pour cet export.")

        identity = (row["pseudo_id"], row["source_sha256"])
        if identity in seen_images:
            raise RuntimeError("Une image figure plusieurs fois.")
        seen_images.add(identity)

        previous_split = subjects.setdefault(row["pseudo_id"], row["split"])
        if previous_split != row["split"]:
            raise RuntimeError("Un patient apparaît dans les deux groupes.")

        previous_content_split = contents.setdefault(
            row["source_sha256"], row["split"]
        )
        if previous_content_split != row["split"]:
            raise RuntimeError("Un contenu identique apparaît dans les deux groupes.")

    args.export_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.private_report_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_" + uuid.uuid4().hex[:8]
    )
    package = args.export_root / f"kaggle_{run_id}"
    package.mkdir(mode=0o700)
    (package / "images").mkdir(mode=0o700)

    export_rows = []
    private_rows = []

    for number, row in enumerate(records, start=1):
        source = safe_child(prepared_root, row["prepared_relative_path"])
        if fingerprint(source) != row["prepared_sha256"]:
            raise RuntimeError("Un volume préparé a changé.")

        image = nib.load(str(source))
        try:
            values = image.get_fdata(dtype=np.float32, caching="unchanged")
            if values.shape != (128, 128, 128):
                raise RuntimeError("Dimensions inattendues.")
            if (
                not np.isfinite(values).all()
                or values.min() < -0.00001
                or values.max() > 1.00001
                or values.min() == values.max()
            ):
                raise RuntimeError("Valeurs numériques inattendues.")

            # Un numéro propre au paquet remplace le code patient interne.
            export_id = f"image_{number:06d}"
            filename = f"images/{export_id}.npz"
            destination = package / filename

            # Le fichier contient uniquement le tableau numérique.
            np.savez_compressed(destination, image=values)
        finally:
            image.uncache()
        del values, image

        exported_hash = fingerprint(destination)
        export_rows.append({
            "image_id": export_id,
            "file": filename,
            "split": row["split"],
            "target": row["target_binary"],
            "site": row["site"],
            "sha256": exported_hash,
        })
        private_rows.append({
            "image_id": export_id,
            "pseudo_id": row["pseudo_id"],
            "source_sha256": row["source_sha256"],
            "prepared_relative_path": row["prepared_relative_path"],
            "prepared_sha256": row["prepared_sha256"],
            "exported_sha256": exported_hash,
        })

        if number % 25 == 0 or number == len(records):
            print(f"Export local : {number}/{len(records)} images", flush=True)

    table = package / "dataset.csv"
    with table.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("image_id", "file", "split", "target", "site", "sha256"),
        )
        writer.writeheader()
        writer.writerows(export_rows)

    # Calculer la pondération sur les patients d'entraînement uniquement.
    train_patients = {}
    for row in records:
        if row["split"] == "train":
            previous = train_patients.setdefault(
                row["pseudo_id"], row["target_binary"]
            )
            if previous != row["target_binary"]:
                raise RuntimeError("Classes contradictoires pour un patient.")

    classes = Counter(train_patients.values())
    if classes[0] == 0 or classes[1] == 0:
        raise RuntimeError("Les deux classes doivent être présentes.")

    metadata = {
        "schema_version": 1,
        "export_id": run_id,
        "format": "npz",
        "array_key": "image",
        "array_dtype": "float32",
        "array_shape": [128, 128, 128],
        "target_definition": {
            "0": "temoin",
            "1": "ADHD_codes_source_1_2_3",
        },
        "groups": dict(counts),
        "train_patient_classes": {
            "controls": classes[0],
            "adhd": classes[1],
        },
        "train_positive_weight_reference": classes[0] / classes[1],
        "dataset_csv_sha256": fingerprint(table),
        "preprocessing": preparation["configuration"]["preprocessing"],
        "contains_test_data": False,
        "face_anonymization_verified": False,
        "remote_upload_performed": False,
    }
    save_json(package / "dataset_info.json", metadata)

    (package / "README.txt").write_text(
        "PAQUET LOCAL POUR L'ENTRAINEMENT ADHD-200\n\n"
        "dataset.csv décrit les images, groupes et classes.\n"
        "Chaque fichier NPZ contient un tableau nommé image.\n"
        "Lecture : numpy.load(chemin, allow_pickle=False)['image'].\n"
        "Une dimension de canal devra être ajoutée par le programme PyTorch.\n\n"
        "Les tests sont conservés hors de ce paquet.\n"
        "Le paquet peut contenir l'anatomie du visage.\n"
        "La protection avant transfert doit encore être vérifiée.\n",
        encoding="utf-8",
    )

    private_report = args.private_report_dir / f"kaggle_export_{run_id}.json"
    save_json(private_report, {
        "schema_version": 1,
        "export_id": run_id,
        "package_directory": package.name,
        "source_preparation_run_id": preparation["run_id"],
        "source_preparation_report_sha256": fingerprint(args.preparation_report),
        "export_code_sha256": fingerprint(Path(__file__)),
        "dataset_csv_sha256": fingerprint(table),
        "summary": metadata,
        "private_correspondences": private_rows,
    })

    # Présent uniquement lorsque tous les fichiers et rapports sont enregistrés.
    (package / "LOCAL_EXPORT_COMPLETE.txt").write_text(
        "Export local complet. Aucun transfert effectué.\n",
        encoding="utf-8",
    )

    total_bytes = sum(
        path.stat().st_size for path in package.rglob("*") if path.is_file()
    )
    print("\nEXPORT LOCAL TERMINÉ")
    print(f"Entraînement : {counts['train']}")
    print(f"Validation : {counts['validation']}")
    print("Images de test exportées : 0")
    print(f"Taille du paquet : {total_bytes / 1024**3:.2f} Gio")
    print(f"Dossier : {package.name}")
    print(f"Correspondances privées : {private_report.name}")
    print("Aucune clé ni code patient interne dans le paquet.")
    print("Anatomie du visage potentiellement présente.")
    print("Aucune donnée transférée ; aucun modèle entraîné.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-report", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--private-report-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
