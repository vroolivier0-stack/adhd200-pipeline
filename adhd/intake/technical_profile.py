"""Contrôler les fichiers IRM sans les modifier.

Un fichier lisible n'est pas nécessairement un examen médical de bonne qualité.
Les contrôles réalisés ici concernent le format et les valeurs numériques.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

import nibabel as nib
import numpy as np


def discover_images(root):
    """Parcourir les sous-dossiers sans suivre les liens symboliques."""
    def stop_on_error(error):
        raise error

    for directory, folders, filenames in os.walk(
        root, followlinks=False, onerror=stop_on_error
    ):
        directory = Path(directory)
        folders[:] = sorted(
            name for name in folders
            if not (directory / name).is_symlink()
        )
        for filename in sorted(filenames):
            path = directory / filename
            if not path.is_symlink():
                if filename.lower().endswith((".nii", ".nii.gz")):
                    yield path


def inspect_image(path):
    """Lire une image entière, puis libérer sa mémoire avant la suivante."""
    record = {"status": "ok", "reasons": []}
    image = None

    try:
        image = nib.load(str(path), mmap=False)
        shape = tuple(int(size) for size in image.shape)
        record["shape"] = list(shape)

        if len(shape) != 3 or any(size <= 1 for size in shape):
            record["reasons"].append("volume_3d_attendu")
            record["status"] = "a_isoler"
            return record

        # L'affine décrit la position du volume dans l'espace.
        affine = np.asarray(image.affine, dtype=np.float64)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            record["reasons"].append("position_spatiale_invalide")
        elif np.linalg.matrix_rank(affine[:3, :3]) < 3:
            record["reasons"].append("position_spatiale_non_inversible")

        # Les espacements indiquent la taille des pixels en volume.
        spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
        if not np.isfinite(spacing).all() or np.any(spacing <= 0):
            record["reasons"].append("espacement_invalide")
        else:
            record["spacing"] = spacing.tolist()
        record["spatial_unit"] = image.header.get_xyzt_units()[0]

        # Charger réellement toutes les valeurs : lire uniquement
        # l'en-tête ne suffit pas à détecter un fichier tronqué.
        values = image.get_fdata(dtype=np.float32, caching="unchanged")

        if not np.isfinite(values).all():
            record["reasons"].append("valeurs_nan_ou_infinies")
        else:
            minimum = float(values.min())
            maximum = float(values.max())
            record["minimum"] = minimum
            record["maximum"] = maximum
            if minimum == maximum:
                record["reasons"].append("image_uniforme")

        del values

    except Exception as error:
        record["reasons"].append("lecture_ou_controle_impossible")
        # Le message détaillé peut contenir un chemin :
        # il sera conservé uniquement dans le rapport privé.
        record["error_type"] = type(error).__name__
        record["error_message"] = str(error)

    finally:
        if image is not None:
            image.uncache()

    if record["reasons"]:
        record["status"] = "a_isoler"

    return record


def run(root, report_directory):
    if not root.is_dir():
        raise RuntimeError("Dossier brut introuvable.")

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]

    records = []
    totals = defaultdict(Counter)
    shapes = defaultdict(Counter)
    skipped_rest = 0
    unknown_type = 0

    for path in discover_images(root):
        relative = path.relative_to(root)
        label = str(relative).lower()

        # Ces règles reposent sur les noms observés lors de l'inventaire.
        # Le filtrage définitif sera placé dans son module dédié.
        if "rest" in label:
            skipped_rest += 1
            continue
        if not any(word in label for word in ("anat", "t1w", "mprage")):
            unknown_type += 1
            continue

        site = relative.parts[0]
        record = inspect_image(path)
        record["source_relative_path"] = str(relative)
        record["site"] = site
        record["source_size_bytes"] = path.stat().st_size
        records.append(record)

        totals[site][record["status"]] += 1
        for reason in record["reasons"]:
            totals[site]["raison:" + reason] += 1
        if "shape" in record:
            shapes[site][str(tuple(record["shape"]))] += 1

        if len(records) % 25 == 0:
            print(f"Avancement : {len(records)} images contrôlées", flush=True)

    if not records:
        raise RuntimeError("Aucune image anatomique candidate trouvée.")

    finished = datetime.now(timezone.utc)
    report = {
        "schema_version": 1,
        "check_version": "technical_profile_v1",
        "run_id": run_id,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "anatomical_candidates_checked": len(records),
        "rest_skipped": skipped_rest,
        "unknown_type_skipped": unknown_type,
        "summary_by_site": {
            site: dict(counts) for site, counts in sorted(totals.items())
        },
        "records": records,
    }

    report_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = report_directory / f"technical_profile_{run_id}.json"
    temporary = destination.with_suffix(".tmp")

    # Écrire un nouveau rapport complet, puis le rendre disponible.
    # Cela évite qu'un autre programme lise un rapport à moitié écrit.
    with temporary.open("x", encoding="utf-8") as stream:
        os.chmod(temporary, 0o600)
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(destination)

    print("\nCONTRÔLE TECHNIQUE TERMINÉ")
    print(f"Images anatomiques contrôlées : {len(records)}")
    print(f"Images REST laissées hors de ce contrôle : {skipped_rest}")
    print(f"Images de type à examiner : {unknown_type}")

    for site, counts in sorted(totals.items()):
        print(f"\n{site} :")
        print(f"  Contrôles réussis : {counts['ok']}")
        print(f"  Fichiers à isoler : {counts['a_isoler']}")
        for reason, count in sorted(counts.items()):
            if reason.startswith("raison:"):
                print(f"  {reason} : {count}")
        print(f"  Dimensions observées : {dict(shapes[site])}")

    print(f"\nDurée : {report['duration_seconds']:.1f} secondes")
    print(f"Rapport privé enregistré : {destination.name}")
    print("Aucune image déplacée, supprimée ou modifiée.")
    print("Un contrôle réussi ne constitue pas une validation médicale.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.raw_root, arguments.report_dir)


if __name__ == "__main__":
    main()
