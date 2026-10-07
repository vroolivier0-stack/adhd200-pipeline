"""Préparer les images historiques et reprendre un travail interrompu.

Chaque volume produit possède une fiche décrivant son origine et sa recette.
Brown reste réservé aux arrivées : ce module ne le prépare pas.
Les fichiers bruts ne sont jamais modifiés.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import time
import uuid

import nibabel as nib
import numpy as np
import torch
import yaml

from adhd.data import prepare_image
from adhd.utils import fingerprint, save_json


ALLOWED_SPLITS = (
    "train", "validation", "test_internal",
    "test_external", "controls_external",
)


def save_and_check_volume(prepared, path):
    """Enregistrer un en-tête neuf, puis relire réellement les valeurs."""
    image = nib.Nifti1Image(prepared["array"], prepared["affine"])
    image.header.set_xyzt_units("mm")
    image.set_sform(prepared["affine"], code=1)
    image.set_qform(prepared["affine"], code=1)
    nib.save(image, str(path))

    saved = nib.load(str(path))
    try:
        values = saved.get_fdata(dtype=np.float32)
        if not np.array_equal(values, prepared["array"]):
            raise RuntimeError("Valeurs différentes après enregistrement.")
        if not np.allclose(
            saved.affine, prepared["affine"], rtol=1e-5, atol=1e-4
        ):
            raise RuntimeError("Position spatiale différente après enregistrement.")
        if nib.aff2axcodes(saved.affine) != ("R", "A", "S"):
            raise RuntimeError("Orientation finale inattendue.")
        if saved.header.get_xyzt_units()[0] != "mm":
            raise RuntimeError("Unité finale inattendue.")
    finally:
        saved.uncache()


def check_existing(volume, receipt, expected):
    """Réutiliser uniquement un résultat accompagné d'une fiche compatible."""
    if not volume.is_file() or not receipt.is_file():
        return None
    try:
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in expected.items()):
            return None
        if previous.get("prepared_sha256") != fingerprint(volume):
            return None
        return previous
    except (OSError, ValueError):
        return None


def run(args):
    os.umask(0o077)
    torch.set_num_threads(2)

    if not args.raw_root.is_dir():
        raise RuntimeError("Dossier brut introuvable.")
    if args.limit is not None and args.limit < 1:
        raise RuntimeError("La limite doit être un entier positif.")

    source = json.loads(args.split_file.read_text(encoding="utf-8"))
    if source.get("split_version") != "split_v2":
        raise RuntimeError("La répartition version 2 est requise.")

    config = yaml.safe_load(args.config_file.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise RuntimeError("Configuration non reconnue.")

    # Identifier précisément les éléments qui déterminent la préparation.
    provenance = {
        "split_manifest_sha256": fingerprint(args.split_file),
        "config_sha256": fingerprint(args.config_file),
        "normalization_code_sha256": fingerprint(
            Path(__file__).with_name("data.py")
        ),
        "batch_code_sha256": fingerprint(Path(__file__)),
        "versions": {
            package: version(package)
            for package in ("monai", "torch", "numpy", "nibabel", "scipy")
        },
    }
    import hashlib
    signature = hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode("utf-8")
    ).hexdigest()

    records = sorted(
        (
            row for row in source["records"]
            if row["split"] in args.splits
        ),
        key=lambda row: (row["split"], row["pseudo_id"], row["source_sha256"]),
    )
    available_count = len(records)
    if args.limit is not None:
        records = records[:args.limit]
    if not records:
        raise RuntimeError("Aucune image sélectionnée.")

    # Contrôler toute la sélection avant de commencer.
    seen_sources = set()
    for row in records:
        relative = Path(row["source_relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Chemin source non autorisé.")
        if relative.parts[0] == "Brown" or row["site"] == "Brown":
            raise RuntimeError("Brown doit rester réservé aux arrivées.")
        if row["target_binary"] not in (0, 1):
            raise RuntimeError("Une cible connue est attendue.")
        if row["source_relative_path"] in seen_sources:
            raise RuntimeError("Une image apparaît plusieurs fois.")
        seen_sources.add(row["source_relative_path"])

    args.output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.report_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    output = args.output_root / f"prepared_{signature}"
    if output.is_symlink():
        raise RuntimeError("Dossier de résultats non autorisé.")
    output.mkdir(mode=0o700, exist_ok=True)

    started = time.monotonic()
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_" + uuid.uuid4().hex[:8]
    )
    counts = Counter()
    results = []
    failures = []

    # Empêcher deux programmes d'écrire simultanément dans ce même dossier.
    lock_path = output / ".preparation.lock"
    if lock_path.is_symlink():
        raise RuntimeError("Emplacement de verrou non autorisé.")

    with lock_path.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Une autre préparation utilise déjà ce dossier.")

        for number, row in enumerate(records, start=1):
            print(
                f"Image {number}/{len(records)} : "
                f"{row['site']}, groupe {row['split']}",
                flush=True,
            )

            try:
                original = args.raw_root / row["source_relative_path"]
                if original.is_symlink() or not original.resolve().is_relative_to(
                    args.raw_root.resolve()
                ):
                    raise RuntimeError("Fichier source non autorisé.")

                # Vérifier l'original même si un résultat existe déjà.
                if fingerprint(original) != row["source_sha256"]:
                    raise RuntimeError("L'original a changé depuis sa validation.")

                folder = output / row["split"]
                if folder.is_symlink():
                    raise RuntimeError("Sous-dossier de résultats non autorisé.")
                folder.mkdir(mode=0o700, exist_ok=True)

                stem = row["pseudo_id"] + "_" + row["source_sha256"]
                volume = folder / f"{stem}.nii.gz"
                receipt = folder / f"{stem}.json"
                pending = folder / f".{stem}.pending.nii.gz"
                receipt_pending = receipt.with_suffix(".tmp")

                for path in (volume, receipt, pending, receipt_pending):
                    if path.is_symlink():
                        raise RuntimeError("Emplacement de résultat non autorisé.")

                expected = {
                    "schema_version": 1,
                    "preparation_signature": signature,
                    "pseudo_id": row["pseudo_id"],
                    "site": row["site"],
                    "target_binary": row["target_binary"],
                    "split": row["split"],
                    "source_sha256": row["source_sha256"],
                }
                previous = check_existing(volume, receipt, expected)

                if previous is not None:
                    counts["reused"] += 1
                    print("  Résultat existant vérifié et réutilisé.", flush=True)
                    prepared_hash = previous["prepared_sha256"]
                else:
                    repairing = volume.exists() or receipt.exists()

                    # Nettoyer seulement les fichiers provisoires de cette image.
                    pending.unlink(missing_ok=True)
                    receipt_pending.unlink(missing_ok=True)
                    try:
                        prepared = prepare_image(original, config)
                        save_and_check_volume(prepared, pending)
                        prepared_hash = fingerprint(pending)
                        statistics = prepared["statistics"]
                        del prepared

                        # Rendre le volume disponible après sa vérification.
                        pending.replace(volume)
                        save_json(receipt, {
                            **expected,
                            "prepared_sha256": prepared_hash,
                            "prepared_filename": volume.name,
                            "statistics": statistics,
                            "recorded_at": datetime.now(timezone.utc).isoformat(),
                        })
                    finally:
                        pending.unlink(missing_ok=True)
                        receipt_pending.unlink(missing_ok=True)

                    counts["repaired" if repairing else "created"] += 1
                    print("  Volume enregistré et relu avec succès.", flush=True)

                results.append({
                    **expected,
                    "prepared_relative_path": str(volume.relative_to(output)),
                    "receipt_relative_path": str(receipt.relative_to(output)),
                    "prepared_sha256": prepared_hash,
                })

            except Exception as error:
                counts["failed"] += 1
                failures.append({
                    "source_relative_path": row["source_relative_path"],
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                })
                print(
                    f"  Échec enregistré dans le rapport privé : "
                    f"{type(error).__name__}",
                    flush=True,
                )

        report_path = args.report_dir / f"preparation_{run_id}.json"
        save_json(report_path, {
            "schema_version": 1,
            "preparation_version": "historical_preparation_v1",
            "run_id": run_id,
            "source_split_run_id": source["run_id"],
            "configuration": config,
            "provenance": provenance,
            "preparation_signature": signature,
            "output_directory": output.name,
            "requested_splits": args.splits,
            "available_images_in_requested_splits": available_count,
            "selected_images": len(records),
            "scope_limited": len(records) < available_count,
            "status": "failed" if failures else "complete_for_selected_scope",
            "duration_seconds": time.monotonic() - started,
            "summary": {
                name: counts[name]
                for name in ("created", "reused", "repaired", "failed")
            },
            "prepared_records": results,
            "failures": failures,
        })

    print("\nPRÉPARATION DU LOT TERMINÉE")
    print(f"Images sélectionnées : {len(records)} / {available_count}")
    print(f"Nouveaux résultats : {counts['created']}")
    print(f"Résultats réutilisés : {counts['reused']}")
    print(f"Résultats reconstruits : {counts['repaired']}")
    print(f"Échecs : {counts['failed']}")
    print(f"Dossier de résultats : {output.name}")
    print(f"Rapport privé : {report_path.name}")
    print("Originaux conservés ; Brown non préparé.")

    if failures:
        raise RuntimeError(
            "Le lot contient des échecs. Consulter le rapport privé."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=ALLOWED_SPLITS, default=["train"]
    )
    parser.add_argument("--limit", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
