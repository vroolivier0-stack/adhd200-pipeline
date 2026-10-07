"""Estimer ce que coûte le pipeline, à partir de ses propres mesures.

Chaque tâche exécutée enregistre sa durée (événements « resources »).
Ce module additionne ces durées par type de tâche et les convertit en énergie
et en euros avec les tarifs de configs/service.yaml (rubrique « costs »).
Ces tarifs sont des HYPOTHÈSES réglables : le résultat est un ordre de
grandeur, pas une facture.
"""
import os

MEASURED = """
    SELECT j.kind, count(*) AS runs,
           sum((e.body->>'seconds')::float8) AS seconds,
           max((e.body->>'peak_process_mib')::float8) AS peak_mib
    FROM events e JOIN jobs j ON j.id = e.body->>'job'
    WHERE e.kind = 'resources'
    GROUP BY j.kind ORDER BY j.kind
"""
STORED = ("raw", "historical", "arrivals", "export_kaggle", "models")


def estimate(tasks, rates, storage_gib=None):
    """Convertir des durées mesurées en coûts. Fonction pure, testable sans base."""
    hours = sum(float(t["seconds"] or 0) for t in tasks) / 3600
    energy = hours * rates["local_power_watts"] / 1000
    gpu_hours = float(rates.get("gpu_hours_training", 0))
    result = {
        "tasks": [
            {"task": t["kind"], "runs": int(t["runs"]),
             "seconds_total": round(float(t["seconds"] or 0), 2),
             "seconds_mean": round(float(t["seconds"] or 0) / max(int(t["runs"]), 1), 3),
             "peak_memory_mib": round(float(t["peak_mib"] or 0), 1)}
            for t in tasks
        ],
        "compute_hours": round(hours, 4),
        "local_energy_kwh": round(energy, 5),
        "local_cost_eur": round(energy * rates["electricity_eur_per_kwh"], 4),
        "local_co2_g": round(energy * rates["grid_gco2_per_kwh"], 2),
        "cloud_cpu_equivalent_eur": round(hours * rates["cloud_cpu_eur_per_hour"], 4),
        "training_gpu_hours": gpu_hours,
        "training_cloud_equivalent_eur": round(gpu_hours * rates["cloud_gpu_eur_per_hour"], 2),
        "rates": rates,
        "note": "Durées mesurées par le pipeline ; tarifs = hypothèses de configs/service.yaml.",
    }
    if storage_gib is not None:
        total = sum(storage_gib.values())
        result["storage_gib"] = {k: round(v, 2) for k, v in storage_gib.items()}
        result["storage_cloud_equivalent_eur_per_month"] = round(
            total * rates["storage_eur_per_gb_month"], 2)
    return result


def summary():
    """Coûts de calcul d'après la base (rapide : utilisé par le tableau de bord)."""
    from .utils import db, settings
    rates = settings().get("costs")
    if not rates:
        return None
    with db() as conn:
        tasks = conn.execute(MEASURED).fetchall()
    return estimate(tasks, rates)


def report():
    """Rapport complet, taille des dossiers comprise (commande : adhd.cli costs)."""
    from .utils import db, root, settings
    sizes = {}
    for name in STORED:
        total = 0
        for directory, _, files in os.walk(root() / name):
            for file in files:
                try:
                    total += os.path.getsize(os.path.join(directory, file))
                except OSError:
                    pass
        sizes[name] = total / 1024 ** 3
    with db() as conn:
        tasks = conn.execute(MEASURED).fetchall()
    return estimate(tasks, settings()["costs"], sizes)
