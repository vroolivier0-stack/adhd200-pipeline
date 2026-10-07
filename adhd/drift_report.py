"""Rapport de dérive avec Evidently.

Question posée : les images reçues ressemblent-elles encore à celles qui ont
servi à entraîner le modèle ?

Evidently compare des tableaux de chiffres, pas des images. On lui fournit donc,
pour chaque image, quatre mesures simples (les mêmes que la surveillance du
pipeline) :
  - luminosite_moyenne : la luminosité générale ;
  - contraste          : l'écart-type des intensités ;
  - zones_claires_p90  : le niveau des zones claires (90e centile) ;
  - part_occupee       : la place que prend la tête dans le cube.

Référence : les images d'entraînement du paquet envoyé à Kaggle.
Courant   : les images reçues et prédites, lues dans la base de données.

Ce programme tourne dans sa propre boîte Docker (voir Dockerfile.evidently),
pour que les outils d'Evidently ne se mélangent pas avec ceux du pipeline :
    docker compose --profile monitoring run --rm drift-report

Une dérive signale que les DONNÉES ont changé. Elle ne prouve pas que le modèle
se trompe : pour cela, il faut le vrai diagnostic.
"""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["luminosite_moyenne", "contraste", "zones_claires_p90", "part_occupee"]
KEYS = ["mean", "std", "p90", "occupied"]  # noms utilisés par le pipeline


def root():
    return Path(os.environ.get("DATA_ROOT", "/data"))


def features(array):
    """Les quatre mesures d'une image préparée (mêmes calculs que adhd/flow.py)."""
    return [float(array.mean()), float(array.std()),
            float(np.percentile(array, 90)), float(np.count_nonzero(array) / array.size)]


def latest_package():
    """Le dernier paquet d'export complet (celui qui a servi à l'entraînement)."""
    folders = [
        p for p in (root() / "export_kaggle").iterdir()
        if p.is_dir() and (p / "dataset.csv").is_file() and (p / "LOCAL_EXPORT_COMPLETE.txt").is_file()
    ]
    if not folders:
        raise RuntimeError("Aucun paquet d'export complet dans export_kaggle.")
    return max(folders, key=lambda p: p.stat().st_mtime)


def reference_table():
    """Mesures des images d'entraînement. Calculées une fois, puis relues."""
    package = latest_package()
    cache = root() / "reference" / "compact" / f"evidently_train_features_{package.name}.csv"
    if cache.is_file():
        return pd.read_csv(cache)
    rows = []
    with (package / "dataset.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] != "train":
                continue
            with np.load(package / row["file"], allow_pickle=False) as archive:
                rows.append([row["site"], *features(archive["image"])])
    table = pd.DataFrame(rows, columns=["hopital", *COLUMNS])
    cache.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache, index=False)  # ni nom de fichier ni identifiant : hôpital et mesures seulement
    return table


def current_table():
    """Mesures des images reçues : une ligne par patient, la plus récente."""
    import psycopg
    from psycopg.rows import dict_row

    dsn = Path(os.environ.get("APP_DSN_FILE", "/run/secrets/app_dsn")).read_text().strip()
    query = """
        SELECT DISTINCT ON (i.subject) i.site, i.features
        FROM items i JOIN batches b ON b.id = i.batch_id
        WHERE i.status = 'predicted' AND i.cohort != 'synthetic'
        ORDER BY i.subject, b.created_at DESC
    """
    with psycopg.connect(dsn, connect_timeout=10, row_factory=dict_row) as connection:
        rows = connection.execute(query).fetchall()
    return pd.DataFrame(
        [[r["site"], *[float(r["features"][k]) for k in KEYS]] for r in rows],
        columns=["hopital", *COLUMNS],
    )


def drifted_share(report):
    """Lire dans le rapport la part de mesures en dérive (comme dans l'exemple du cours)."""
    metrics = report.get("metrics", [])
    for metric in metrics:
        name = str(metric.get("metric_id") or metric.get("metric_name") or "")
        if "DriftedColumns" in name:
            value = metric.get("value", {})
            return float(value.get("share", 0.0)), int(value.get("count", 0))
    value = metrics[0].get("value", {}) if metrics else {}
    return float(value.get("share", 0.0)), int(value.get("count", 0))


def main():
    import evidently
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    threshold = float(os.environ.get("DRIFT_THRESHOLD", "0.5"))
    reference, current = reference_table(), current_table()
    if len(current) < 5:
        raise SystemExit(f"Pas assez d'images reçues pour un rapport ({len(current)}).")

    result = Report([DataDriftPreset()]).run(
        current_data=Dataset.from_pandas(current[COLUMNS], data_definition=DataDefinition()),
        reference_data=Dataset.from_pandas(reference[COLUMNS], data_definition=DataDefinition()),
    )
    folder = root() / "reports" / "compact" / "evidently"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    html = folder / f"drift_{stamp}.html"
    result.save_html(str(html))

    share, count = drifted_share(result.dict())
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool": "evidently " + evidently.__version__,
        "reference_images": int(len(reference)),
        "current_images": int(len(current)),
        "current_by_hospital": {k: int(v) for k, v in current["hopital"].value_counts().items()},
        "columns": COLUMNS,
        "drifted_columns": count,
        "drift_share": share,
        "threshold": threshold,
        "drift_detected": bool(share >= threshold),
        "html": html.name,
        "limit": "Dérive des données d'entrée, distincte de la performance du modèle.",
    }
    (folder / "latest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
