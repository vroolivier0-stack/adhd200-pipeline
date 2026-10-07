"""Alias MLflow comme état souhaité ; cache local vérifié pour l'inférence."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
import uuid

from .utils import root, read, save_json, fingerprint, secret, http, event, db

NAME = "ADHD200_ANATOMICAL"
URI = "https://dagshub.com/vroolivier0/adhd200-pipeline.mlflow"


def client():
    import mlflow
    os.environ["MLFLOW_TRACKING_USERNAME"] = "vroolivier0"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = secret("MLFLOW_PASSWORD")
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "15"
    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"
    return mlflow.MlflowClient(tracking_uri=URI, registry_uri=URI)


def validate_manifest(manifest, version):
    if (str(manifest.get("mlflow_version")) != str(version.version)
        or manifest.get("mlflow_run") != version.run_id
        or manifest.get("synthetic") is not False
        or manifest.get("schema_version") != 1):
        raise ValueError("Manifeste incompatible avec la version du registre")
    if str(version.status) != "READY":
        raise ValueError("Version du registre non prête")


def desired_versions(c):
    return {alias: c.get_model_version_by_alias(NAME, alias)
            for alias in ("champion", "challenger")}


def fetch_release(c, version):
    from .cli import import_model
    digest = version.tags.get("deployment_manifest_sha256")
    if not digest or len(digest) != 64:
        raise ValueError("Paquet de déploiement non publié pour cette version")
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(c.download_artifacts(version.run_id, "deployment", directory))
        files = list(folder.rglob("*"))
        if (any(path.is_symlink() for path in files)
            or {path.relative_to(folder).as_posix() for path in files if path.is_file()}
               != {"release.json", "best.pt", "model_card.md"}):
            raise ValueError("Paquet distant inattendu")
        if fingerprint(folder / "release.json") != digest:
            raise ValueError("Empreinte du manifeste distant différente")
        manifest = read(folder / "release.json")
        validate_manifest(manifest, version)
        if fingerprint(folder / "best.pt") != manifest["weights_sha256"]:
            raise ValueError("Empreinte des poids distante différente")
        return import_model(folder)["version"]


def deploy_from_registry():
    from .service import state, lock, load, validate_candidate
    from .utils import settings
    c = client()
    versions = desired_versions(c)
    # Préparer toutes les copies avant d'actualiser l'état servi.
    installed = {alias: fetch_release(c, version) for alias, version in versions.items()}
    for identifier in installed.values():
        validate_candidate(load(identifier)[1]["validation"], settings())
    # Refuser une modification des alias pendant le téléchargement.
    latest = desired_versions(c)
    if any((latest[a].version, latest[a].run_id) != (versions[a].version, versions[a].run_id)
           for a in versions):
        raise ValueError("Alias modifiés pendant le déploiement : état servi conservé")
    with lock:
        old = state()
        new = {"champion": installed["champion"], "challenger": None, "fraction": 0,
               "previous": old.get("champion") if old.get("champion") != installed["champion"]
                           else old.get("previous"),
               "catalog_challenger": installed["challenger"],
               "registry_versions": {alias: str(value.version) for alias, value in versions.items()},
               "registry_runs": {alias: value.run_id for alias, value in versions.items()},
               "registry_uri": URI, "registry_checked_at": datetime.now(timezone.utc).isoformat()}
        save_json(root() / "models/compact/deployment.json", new)
    event("deployment", {"source": "registry_aliases", **new})
    return new


def promote_registry(identifier):
    from .service import state
    c = client()
    desired = desired_versions(c)
    old = state()
    candidate = read(root() / "models/compact" / identifier / "release.json")
    if (candidate["mlflow_run"] != desired["challenger"].run_id
        or str(candidate["mlflow_version"]) != str(desired["challenger"].version)
        or str(desired["champion"].version) != old["registry_versions"]["champion"]):
        raise ValueError("Alias changés depuis la préparation du canary")
    c.set_registered_model_alias(NAME, "champion", desired["challenger"].version)
    c.set_registered_model_alias(NAME, "challenger", desired["champion"].version)
    return deploy_from_registry()


def check_registry():
    from .service import state
    current = state()
    if "registry_versions" not in current:
        result = {"status": "not_configured", "served_model_preserved": True}
    else:
        try:
            wanted = desired_versions(client())
        except Exception as error:
            result = {"status": "registry_unavailable", "error_type": type(error).__name__,
                      "served_model_preserved": True}
        else:
            manifest = read(root() / "models/compact" / current["champion"] / "release.json")
            mismatches = []
            if (str(manifest["mlflow_version"]), manifest["mlflow_run"]) != (
                    str(wanted["champion"].version), wanted["champion"].run_id):
                mismatches.append("champion")
            candidate = read(root() / "models/compact" / current["catalog_challenger"] / "release.json")
            if (str(candidate["mlflow_version"]), candidate["mlflow_run"]) != (
                    str(wanted["challenger"].version), wanted["challenger"].run_id):
                mismatches.append("challenger_catalog")
            result = {"status": "mismatch" if mismatches else "aligned",
                      "mismatches": mismatches, "served_model_preserved": True}
    save_json(root() / "reports/compact/registry_alignment.json", result)
    if result["status"] in {"mismatch", "registry_unavailable"}:
        event("alert", {"code": "registry_" + result["status"], **result})
    event("registry_check", result)
    return result


def bootstrap():
    from .service import load
    c = client()
    versions = {}
    for number in ("3", "4"):
        folders = [folder for folder in (root() / "models/compact").iterdir()
                   if folder.is_dir() and (folder / "release.json").is_file()
                   and str(read(folder / "release.json").get("mlflow_version")) == number]
        if len(folders) != 1:
            raise ValueError("Une copie locale unique est requise pour la version " + number)
        folder = folders[0]
        load(folder.name)
        version = c.get_model_version(NAME, number)
        validate_manifest(read(folder / "release.json"), version)
        digest = fingerprint(folder / "release.json")
        existing = version.tags.get("deployment_manifest_sha256")
        if existing and existing != digest:
            raise ValueError("Un autre paquet est déjà publié pour cette version")
        versions[number] = (folder, version, digest)
    existing_aliases = c.get_registered_model(NAME).aliases
    if existing_aliases.get("champion") not in (None, "3", 3):
        raise ValueError("Un autre champion distant existe : bootstrap arrêté")
    for number, (folder, version, digest) in versions.items():
        if not version.tags.get("deployment_manifest_sha256"):
            with tempfile.TemporaryDirectory() as directory:
                import shutil
                for name in ("release.json", "best.pt", "model_card.md"):
                    shutil.copyfile(folder / name, Path(directory) / name)
                c.log_artifacts(version.run_id, directory, artifact_path="deployment")
            c.set_model_version_tag(NAME, number, "deployment_manifest_sha256", digest)
    c.set_registered_model_alias(NAME, "champion", "3")
    c.set_registered_model_alias(NAME, "challenger", "4")
    return http("/deployment", {"action": "registry-deploy"}, admin=True)


def restoration_exercise():
    from .service import state
    c = client()
    before = desired_versions(c)
    if str(before["champion"].version) != "3" or str(before["challenger"].version) != "4":
        raise ValueError("L'exercice exige champion 3 et challenger 4")
    if state().get("registry_versions", {}).get("champion") != "3":
        raise ValueError("Déployer d'abord depuis le registre")
    with db() as conn:
        rows = conn.execute("SELECT id FROM items WHERE site='Brown' AND cohort='arrivals_reserve' AND status='predicted' ORDER BY id LIMIT 1").fetchall()
    if not rows:
        raise ValueError("IRM Brown préparée requise")
    identifier = rows[0]["id"]
    report = {"scope": "Exercice technique de restauration ; aucune promotion scientifique",
              "started_at": datetime.now(timezone.utc).isoformat(), "steps": []}
    path = root() / "reports/compact" / ("registry_restore_" + uuid.uuid4().hex + ".json")
    changed = False
    try:
        # Prévalider le téléchargement avant de déplacer le pointeur distant.
        fetch_release(c, before["challenger"])
        c.set_model_version_tag(NAME, "4", "restoration_exercise", "technical_only_not_scientific_promotion")
        changed = True
        c.set_registered_model_alias(NAME, "champion", "4")
        mismatch = check_registry()
        if mismatch["status"] != "mismatch":
            raise AssertionError("Désaccord transitoire non détecté")
        applied = http("/deployment", {"action": "registry-deploy"}, admin=True)
        prediction = http("/predict", {"ids": [identifier]})["predictions"][0]
        if prediction["version"] != applied["champion"] or applied["registry_versions"]["champion"] != "4":
            raise AssertionError("L'API ne sert pas la version 4")
        report["steps"].append({"registry_champion": "4", "served_version": prediction["version"], "score": prediction["score"]})
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        if changed:
            try:
                c.set_registered_model_alias(NAME, "champion", "3")
                applied = http("/deployment", {"action": "registry-deploy"}, admin=True)
                prediction = http("/predict", {"ids": [identifier]})["predictions"][0]
                if prediction["version"] != applied["champion"] or applied["registry_versions"]["champion"] != "3":
                    raise AssertionError("Restauration de la version 3 non vérifiée")
                report["steps"].append({"registry_champion": "3", "served_version": prediction["version"], "score": prediction["score"]})
                report["alignment"] = check_registry()
            except Exception as error:
                report.update(status="restore_failed", restore_error_type=type(error).__name__)
                save_json(path, report)
                print("RESTAURATION NON CONFIRMÉE :", path, flush=True)
                raise
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_json(path, report)
    report["status"] = "checks_passed"
    save_json(path, report)
    return {"report": str(path), "champion": "3", "challenger_catalog": "4", "canary_fraction": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bootstrap", "deploy", "check", "restore-exercise"))
    args = parser.parse_args()
    commands = {"bootstrap": bootstrap, "deploy": lambda: http("/deployment", {"action": "registry-deploy"}, admin=True),
                "check": check_registry, "restore-exercise": restoration_exercise}
    import json
    print(json.dumps(commands[args.command](), ensure_ascii=False))


if __name__ == "__main__":
    main()
