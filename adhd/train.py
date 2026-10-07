"""Organiser un entraînement et reprendre une exécution interrompue.

Un entraînement réel exige le GPU CUDA.
--inspect contrôle la configuration et le tableau sans entraîner les modèles.
Les groupes de test ne sont jamais ouverts par ce programme.
"""

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid
import warnings

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from adhd.learning import ADHD200Dataset, fingerprint
from adhd.learning import (
    make_optimizer, select_precision, train_epoch, evaluate_epoch,
    save_checkpoint, load_checkpoint,
)
from adhd.learning import create_model, MODEL_CODE_VERSION


def atomic_json(path, value):
    """Rendre disponible un fichier JSON seulement après son écriture complète."""
    path = Path(path)
    if path.is_symlink():
        raise RuntimeError("Emplacement de rapport non autorisé.")
    temporary = path.parent / (path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def advance_stopping(state, score, min_delta):
    """Compter les passages sans progrès suffisamment important."""
    reference = state["reference"]
    if reference is None or score > reference + min_delta:
        return {"reference": float(score), "bad_epochs": 0}
    return {
        "reference": reference,
        "bad_epochs": state["bad_epochs"] + 1,
    }


def capture_rng(train_generator, validation_generator, device):
    """Conserver l'état des tirages aléatoires, pas seulement leur graine."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        "train_generator": train_generator.get_state(),
        "validation_generator": validation_generator.get_state(),
    }


def restore_rng(state, train_generator, validation_generator, device):
    """Replacer les tirages dans l'état de la dernière époque terminée."""
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["algorithm"],
        np.asarray(numpy_state["keys"], dtype=np.uint32),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        if len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("Le nombre de GPU diffère de la sauvegarde.")
        torch.cuda.set_rng_state_all(state["cuda"])
    train_generator.set_state(state["train_generator"])
    validation_generator.set_state(state["validation_generator"])


def seed_worker(worker_id):
    """Fixer les tirages des programmes qui lisent les images."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def prepare_config(args):
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != 1
        or config.get("protocol") != "comparable_training_v1"
    ):
        raise RuntimeError("Configuration d'entraînement non reconnue.")
    config = copy.deepcopy(config)

    if args.learning_rate is not None:
        config["optimizer"]["learning_rate"] = args.learning_rate
    if args.seed is not None:
        config["seed"] = args.seed

    if args.model not in config["models"]:
        raise RuntimeError("Le modèle manque dans la configuration.")
    if not isinstance(config["seed"], int) or not 0 <= config["seed"] < 2**32:
        raise RuntimeError("Graine invalide.")

    training = config["training"]
    for name in ("max_epochs", "batch_size", "accumulation_steps"):
        if not isinstance(training[name], int) or training[name] < 1:
            raise RuntimeError(f"Paramètre invalide : {name}")
    if not isinstance(training["num_workers"], int) or training["num_workers"] < 0:
        raise RuntimeError("Nombre de lecteurs invalide.")
    if training["require_cuda"] is not True:
        raise RuntimeError("Cette version exige CUDA pour les vrais entraînements.")
    if config["augmentation"]["enabled"]:
        raise RuntimeError("L'augmentation n'est pas encore implémentée.")
    if (
        config["loss"]["name"] != "BCEWithLogitsLoss"
        or config["loss"]["positive_weight_source"] != "train_patient_counts"
    ):
        raise RuntimeError("Calcul d'erreur non pris en charge.")

    scheduler = config["scheduler"]
    stopping = config["early_stopping"]
    if (
        scheduler["name"] != "ReduceLROnPlateau"
        or scheduler["monitor"] != "roc_auc"
        or scheduler["mode"] != "max"
        or stopping["monitor"] != "roc_auc"
        or stopping["mode"] != "max"
        or config["evaluation"]["selection_metric"] != "roc_auc"
    ):
        raise RuntimeError("La sélection ROC-AUC prévue est requise.")
    if not 1 <= stopping["min_epochs"] <= training["max_epochs"]:
        raise RuntimeError("Nombre minimal d'époques invalide.")
    if stopping["patience"] < 1 or stopping["min_delta"] < 0:
        raise RuntimeError("Paramètres d'arrêt invalides.")
    if not 0 <= config["evaluation"]["threshold"] <= 1:
        raise RuntimeError("Seuil d'évaluation invalide.")
    return config


def run(args):
    os.umask(0o077)
    config = prepare_config(args)
    from adhd.utils import read, recipe_hash
    compatibility = read(args.package / "dataset_info.json").get("compatibility", {})
    if not args.inspect and (compatibility.get("compatible") is not True or compatibility.get("recipe_hash") != recipe_hash() or compatibility.get("data_code_sha256") != fingerprint(Path(__file__).with_name("data.py"))):
        raise RuntimeError("Contrat de préparation non vérifié pour cet environnement")
    train_data = ADHD200Dataset(args.package, "train")
    validation_data = ADHD200Dataset(args.package, "validation")

    classes = Counter(int(row["target"]) for row in train_data.rows)
    if classes[0] == 0 or classes[1] == 0:
        raise RuntimeError("Deux classes sont nécessaires à l'entraînement.")
    validation_classes = Counter(
        int(row["target"]) for row in validation_data.rows
    )
    if validation_classes[0] == 0 or validation_classes[1] == 0:
        raise RuntimeError("Deux classes sont nécessaires à la validation.")
    if (
        {row["image_id"] for row in train_data.rows}
        & {row["image_id"] for row in validation_data.rows}
    ):
        raise RuntimeError("Une image figure dans les deux groupes.")

    weight = classes[0] / classes[1]
    if not np.isclose(weight, train_data.info["train_positive_weight_reference"]):
        raise RuntimeError("La pondération ne correspond pas au paquet.")

    # Contrôler également les paramètres de construction et d'AdamW.
    model = create_model(args.model, **config["models"][args.model])
    make_optimizer(model, config["optimizer"])

    if args.inspect:
        print("PROGRAMME D'ENTRAÎNEMENT : CONFIGURATION CONTRÔLÉE")
        print(f"Modèle : {args.model}")
        print(f"Entraînement : {len(train_data)} ; validation : {len(validation_data)}")
        print(f"Pondération ADHD calculée sur train : {weight:.4f}")
        print(f"Vitesse initiale : {config['optimizer']['learning_rate']}")
        print("Aucune image ouverte ; aucun entraînement lancé.")
        return

    if args.output_root is None and args.resume is None:
        raise RuntimeError("--output-root est requis pour un nouvel entraînement.")

    # Définir ce réglage avant le démarrage des calculs CUDA.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU CUDA absent. Les entraînements réels doivent être lancés sur Kaggle."
        )
    device = torch.device("cuda:0")
    precision = select_precision(device, config["training"]["mixed_precision"])
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    deterministic = bool(config["training"]["deterministic"])
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic

    # Certaines opérations 3D peuvent ne pas avoir d'implémentation strictement
    # déterministe selon la version. Les avertissements seront enregistrés.
    torch.use_deterministic_algorithms(deterministic, warn_only=True)

    # Reconstruire après fixation des tirages, puis placer sur le GPU.
    model = create_model(args.model, **config["models"][args.model]).to(device)
    optimizer = make_optimizer(model, config["optimizer"])
    scheduler_config = config["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_config["mode"],
        factor=scheduler_config["factor"],
        patience=scheduler_config["patience"],
        threshold=scheduler_config["threshold"],
        threshold_mode=scheduler_config["threshold_mode"],
        min_lr=scheduler_config["min_learning_rate"],
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(precision == torch.float16)
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(weight, device=device), reduction="sum"
    )

    train_generator = torch.Generator().manual_seed(seed)
    validation_generator = torch.Generator().manual_seed(seed + 1)
    loader_settings = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": config["training"]["num_workers"],
        "pin_memory": True,
        "worker_init_fn": seed_worker,
        "persistent_workers": False,
        "drop_last": False,
    }
    train_loader = DataLoader(
        train_data, shuffle=True, generator=train_generator, **loader_settings
    )
    validation_loader = DataLoader(
        validation_data, shuffle=False,
        generator=validation_generator, **loader_settings
    )

    folder = Path(__file__).parent
    profile = {
        "config": config,
        "model_name": args.model,
        "model_code_version": MODEL_CODE_VERSION,
        "dataset_csv_sha256": fingerprint(args.package / "dataset.csv"),
        "dataset_info_sha256": fingerprint(args.package / "dataset_info.json"),
        "code_sha256": {
            name: fingerprint(folder / name)
            for name in ("learning.py", "train.py", "data.py", "cli.py")
        },
        "environment": {
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "yaml": str(yaml.__version__),
            "cuda": str(torch.version.cuda),
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "precision": str(precision) if precision is not None else "float32",
            "deterministic_requested": deterministic,
            "deterministic_strict": False,
        },
    }

    loaded = None
    if args.resume is not None:
        loaded = load_checkpoint(args.resume)
        if loaded.get("kind") != "training_resume" or loaded.get("profile") != profile:
            raise RuntimeError(
                "Reprise refusée : données, configuration, code ou environnement différents."
            )
        output = args.resume.resolve().parent
    else:
        args.output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        name = (
            args.model + "_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_" + uuid.uuid4().hex[:8]
        )
        output = args.output_root / name
        output.mkdir(mode=0o700)

    lock_path = output / ".training.lock"
    if lock_path.is_symlink():
        raise RuntimeError("Emplacement de verrou non autorisé.")

    with lock_path.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Un entraînement utilise déjà ce dossier.")

        history = []
        best = None
        stopping = {"reference": None, "bad_epochs": 0}
        start_epoch = 1
        finished = False

        if loaded is not None:
            model.load_state_dict(loaded["model_state"])
            optimizer.load_state_dict(loaded["optimizer_state"])
            scheduler.load_state_dict(loaded["scheduler_state"])
            scaler.load_state_dict(loaded["scaler_state"])
            history = loaded["history"]
            best = loaded["best"]
            stopping = loaded["stopping"]
            start_epoch = loaded["epoch"] + 1
            finished = loaded["finished"]

            # Restaurer les tirages après toutes les constructions d'objets.
            restore_rng(
                loaded["rng"], train_generator, validation_generator, device
            )
            print(f"Reprise après l'époque {loaded['epoch']}.", flush=True)

        atomic_json(output / "run_info.json", {
            "profile": profile,
            "train_positive_weight": weight,
            "parameters": sum(p.numel() for p in model.parameters()),
        })

        current_epoch = start_epoch
        try:
            if not finished:
                for epoch in range(start_epoch, config["training"]["max_epochs"] + 1):
                    current_epoch = epoch
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(device)
                    started = time.monotonic()
                    rate_used = optimizer.param_groups[0]["lr"]

                    print(
                        f"Époque {epoch}/{config['training']['max_epochs']}"
                        f" | {args.model} | taux {rate_used:.6g}",
                        flush=True,
                    )
                    with warnings.catch_warnings(record=True) as recorded:
                        warnings.simplefilter("always")
                        training_result = train_epoch(
                            model, train_loader, optimizer, criterion, device, scaler,
                            accumulation_steps=config["training"]["accumulation_steps"],
                            clip_norm=config["training"]["gradient_clip_norm"],
                            precision=precision,
                        )
                        validation_result = evaluate_epoch(
                            model, validation_loader, criterion, device,
                            threshold=config["evaluation"]["threshold"],
                            precision=precision,
                        )

                    score = validation_result["metrics"]["global"]["roc_auc"]
                    if score is None or not np.isfinite(score):
                        raise RuntimeError("ROC-AUC de validation inexploitable.")

                    # Conserver la meilleure valeur, même si son progrès est petit.
                    if best is None or score > best["score"]:
                        best = {
                            "epoch": epoch,
                            "score": float(score),
                            "model_state": {
                                name: value.detach().cpu().clone()
                                for name, value in model.state_dict().items()
                            },
                            "validation": validation_result,
                        }

                    stopping = advance_stopping(
                        stopping, score, config["early_stopping"]["min_delta"]
                    )
                    scheduler.step(score)
                    finished = (
                        epoch == config["training"]["max_epochs"]
                        or (
                            epoch >= config["early_stopping"]["min_epochs"]
                            and stopping["bad_epochs"] >= config["early_stopping"]["patience"]
                        )
                    )

                    torch.cuda.synchronize()
                    history.append({
                        "epoch": epoch,
                        "train": training_result,
                        "validation_loss": validation_result["loss"],
                        "validation_metrics": validation_result["metrics"],
                        "learning_rate_used": rate_used,
                        "learning_rate_next": optimizer.param_groups[0]["lr"],
                        "duration_seconds": time.monotonic() - started,
                        "peak_gpu_allocated_mib": (
                            torch.cuda.max_memory_allocated(device) / 1024**2
                        ),
                        "peak_gpu_reserved_mib": (
                            torch.cuda.max_memory_reserved(device) / 1024**2
                        ),
                        "warnings": sorted(set(str(item.message) for item in recorded)),
                    })

                    # Cette sauvegarde contient aussi la meilleure version :
                    # elle suffit à reconstruire les rapports après une interruption.
                    save_checkpoint(output / "last.pt", {
                        "schema_version": 1,
                        "kind": "training_resume",
                        "profile": profile,
                        "epoch": epoch,
                        "finished": finished,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "rng": capture_rng(train_generator, validation_generator, device),
                        "history": history,
                        "stopping": stopping,
                        "best": best,
                    })
                    atomic_json(output / "history.json", history)
                    save_checkpoint(output / "best.pt", {
                        "schema_version": 1,
                        "kind": "inference_model",
                        "model_name": args.model,
                        "model_parameters": config["models"][args.model],
                        "model_code_version": MODEL_CODE_VERSION,
                        "model_state": best["model_state"],
                        "epoch": best["epoch"],
                        "validation_roc_auc": best["score"],
                        "threshold": config["evaluation"]["threshold"],
                        "profile": profile,
                    })
                    atomic_json(
                        output / "best_validation.json", best["validation"]
                    )
                    print(
                        f"  ROC-AUC validation : {score:.4f}"
                        f" | meilleure : {best['score']:.4f}",
                        flush=True,
                    )
                    if finished:
                        break

            if best is None:
                raise RuntimeError("Aucune meilleure version disponible.")

            # Reconstituer les fichiers dérivés si la dernière exécution
            # s'était interrompue juste après l'écriture de last.pt.
            atomic_json(output / "history.json", history)
            atomic_json(output / "best_validation.json", best["validation"])
            save_checkpoint(output / "best.pt", {
                "schema_version": 1,
                "kind": "inference_model",
                "model_name": args.model,
                "model_parameters": config["models"][args.model],
                "model_code_version": MODEL_CODE_VERSION,
                "model_state": best["model_state"],
                "epoch": best["epoch"],
                "validation_roc_auc": best["score"],
                "threshold": config["evaluation"]["threshold"],
                "profile": profile,
            })
            atomic_json(output / "summary.json", {
                "status": "completed",
                "model": args.model,
                "completed_epochs": len(history),
                "best_epoch": best["epoch"],
                "best_validation_roc_auc": best["score"],
                "early_stopped": len(history) < config["training"]["max_epochs"],
                "test_data_used": False,
                "remote_mlflow_logging_implemented": True,
            })

        except (Exception, KeyboardInterrupt) as error:
            atomic_json(output / "interruption.json", {
                "epoch_in_progress": current_epoch,
                "error_type": type(error).__name__,
                "message": str(error),
                "resume_file": "last.pt" if (output / "last.pt").is_file() else None,
            })
            raise

    from adhd.cli import publish
    publish(output, args.package)
    print(f"ENTRAÎNEMENT TERMINÉ : {output}")
    print(f"Meilleure époque : {best['epoch']} ; ROC-AUC : {best['score']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--model", choices=("simplecnn3d", "resnet3d"), default="simplecnn3d"
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--inspect", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
