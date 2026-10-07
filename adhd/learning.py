"""Mesurer les résultats sans entraîner le modèle.

La classe 1 représente ADHD ; la classe 0 représente les témoins.
Les scores fournis sont compris entre 0 et 1.

Un score provenant du modèle n'est pas automatiquement une probabilité
médicale fiable. Sa calibration devra être étudiée séparément.
"""

import numpy as np


METRICS_VERSION = "binary_metrics_v1"


def validate_inputs(targets, scores, threshold):
    """Refuser les données vides, contradictoires ou non numériques."""
    targets = np.asarray(targets, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    if (
        targets.ndim != 1 or scores.ndim != 1
        or targets.size == 0 or targets.shape != scores.shape
    ):
        raise ValueError("Deux listes non vides de même longueur sont attendues.")
    if not np.isfinite(targets).all() or not np.isin(targets, [0, 1]).all():
        raise ValueError("Les classes doivent être 0 ou 1.")
    if (
        not np.isfinite(scores).all()
        or np.any(scores < 0) or np.any(scores > 1)
    ):
        raise ValueError("Les scores doivent être finis et compris entre 0 et 1.")

    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Seuil de décision invalide.")

    return targets.astype(np.int64), scores, threshold


def ranking_metrics(targets, scores):
    """Comparer le classement des deux classes, en tenant compte des égalités."""
    positives = scores[targets == 1]
    negatives = scores[targets == 0]
    if positives.size == 0 or negatives.size == 0:
        return None, None

    # Un couple correctement classé vaut 1 ; une égalité vaut 0,5.
    # Le calcul se fait par petits groupes pour limiter la mémoire.
    correct_pairs = 0.0
    for start in range(0, positives.size, 256):
        differences = positives[start:start + 256, None] - negatives[None, :]
        correct_pairs += float(np.count_nonzero(differences > 0))
        correct_pairs += 0.5 * float(np.count_nonzero(differences == 0))
    roc_auc = correct_pairs / (positives.size * negatives.size)

    # Mesurer la précision à mesure que l'on inclut des scores plus faibles.
    # Les scores égaux sont inclus ensemble, sans ordre arbitraire.
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_targets = targets[order]
    cumulative_positives = np.cumsum(sorted_targets)

    ends = np.flatnonzero(
        np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    )
    recall = cumulative_positives[ends] / positives.size
    precision = cumulative_positives[ends] / (ends + 1)
    average_precision = np.sum(
        np.diff(np.r_[0.0, recall]) * precision
    )
    return float(roc_auc), float(average_precision)


def binary_metrics(targets, scores, threshold=0.5):
    """Compter les erreurs et calculer les mesures globales.

    None signifie qu'une mesure n'est pas calculable dans ce groupe.
    La précision est notamment indéfinie si aucune image n'est prédite ADHD.
    """
    targets, scores, threshold = validate_inputs(targets, scores, threshold)
    predictions = (scores >= threshold).astype(np.int64)

    tn = int(np.sum((targets == 0) & (predictions == 0)))
    fp = int(np.sum((targets == 0) & (predictions == 1)))
    fn = int(np.sum((targets == 1) & (predictions == 0)))
    tp = int(np.sum((targets == 1) & (predictions == 1)))

    controls = tn + fp
    adhd = tp + fn
    sensitivity = tp / adhd if adhd else None
    specificity = tn / controls if controls else None
    precision = tp / (tp + fp) if tp + fp else None
    balanced_accuracy = (
        (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None
        else None
    )
    f1 = (
        2 * tp / (2 * tp + fp + fn)
        if adhd else None
    )

    roc_auc, average_precision = ranking_metrics(targets, scores)

    return {
        "metrics_version": METRICS_VERSION,
        "sample_count": int(targets.size),
        "control_count": controls,
        "adhd_count": adhd,
        "threshold": threshold,
        "confusion": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        },
        "accuracy": float((tn + tp) / targets.size),
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "adhd_prevalence": float(adhd / targets.size),
        "brier_score": float(np.mean((scores - targets) ** 2)),
        "both_classes_present": bool(controls and adhd),
    }


def grouped_metrics(targets, scores, sites, threshold=0.5):
    """Produire les mesures globales et par centre.

    Les effectifs accompagnent les scores : quelques patients ne suffisent
    pas à établir une conclusion robuste sur un centre.
    """
    targets, scores, threshold = validate_inputs(targets, scores, threshold)
    sites = np.asarray(sites, dtype=str)
    if sites.ndim != 1 or sites.shape != targets.shape:
        raise ValueError("Un centre par image est attendu.")
    if any(not site.strip() for site in sites):
        raise ValueError("Un nom de centre non vide est attendu.")

    return {
        "global": binary_metrics(targets, scores, threshold),
        "by_site": {
            site: binary_metrics(
                targets[sites == site], scores[sites == site], threshold
            )
            for site in sorted(set(sites.tolist()))
        },
    }


"""Définir deux modèles 3D pour les nouveaux entraînements.

Les modèles produisent un score brut par image, appelé logit.
Ce score sera utilisé par le calcul d'erreur pendant l'apprentissage.
Une conversion en probabilité sera effectuée lors de l'évaluation.

Ces modèles ne constituent pas des outils de diagnostic médical validés.
"""

from math import gcd

import torch
from torch import nn


MODEL_CODE_VERSION = "adhd3d_v1"


def normalization(channels):
    """Normaliser les valeurs par groupes de cartes de caractéristiques."""
    return nn.GroupNorm(gcd(8, channels), channels)


def check_parameters(base_channels, dropout):
    if (
        not isinstance(base_channels, int)
        or isinstance(base_channels, bool)
        or base_channels < 4
    ):
        raise ValueError("Au moins quatre canaux de base sont attendus.")
    if not 0 <= dropout < 1:
        raise ValueError("Le dropout doit être compris entre 0 inclus et 1 exclu.")


def check_input(image):
    # Format d'un lot : images × canaux × profondeur × hauteur × largeur.
    if (
        image.ndim != 5
        or image.shape[0] < 1
        or image.shape[1] != 1
        or min(image.shape[2:]) < 16
    ):
        raise ValueError("Un lot de volumes 3D à un canal est attendu.")


class SimpleCNN3D(nn.Module):
    """Quatre étapes successives pour extraire des motifs du volume."""

    def __init__(self, base_channels=8, dropout=0.2):
        super().__init__()
        check_parameters(base_channels, dropout)
        layers = []
        incoming = 1

        for outgoing in (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ):
            layers.extend([
                # Chercher des motifs dans les trois directions du volume.
                nn.Conv3d(
                    incoming, outgoing, kernel_size=3,
                    stride=2, padding=1, bias=False,
                ),
                normalization(outgoing),
                nn.ReLU(inplace=True),
            ])
            incoming = outgoing

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(incoming, 1)

    def forward(self, image):
        check_input(image)
        features = self.features(image)
        features = self.pool(features).flatten(1)
        return self.classifier(self.dropout(features)).squeeze(1)


class ResidualBlock3D(nn.Module):
    """Ajouter le résultat d'un traitement à une branche de raccourci."""

    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv3d(
                incoming, outgoing, 3, stride=stride, padding=1, bias=False
            ),
            normalization(outgoing),
            nn.ReLU(inplace=True),
            nn.Conv3d(outgoing, outgoing, 3, padding=1, bias=False),
            normalization(outgoing),
        )
        self.shortcut = (
            nn.Identity()
            if incoming == outgoing and stride == 1
            else nn.Sequential(
                nn.Conv3d(
                    incoming, outgoing, 1, stride=stride, bias=False
                ),
                normalization(outgoing),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, image):
        return self.activation(self.branch(image) + self.shortcut(image))


class ResNet3D(nn.Module):
    """Variante résiduelle 3D avec une profondeur réglable.

    Un bloc par étape donne une variante de type ResNet-10.
    Deux blocs par étape donnent une variante de type ResNet-18.
    La normalisation utilisée ici est GroupNorm.
    """

    def __init__(
        self, base_channels=8, dropout=0.2,
        blocks_per_stage=(1, 1, 1, 1),
    ):
        super().__init__()
        check_parameters(base_channels, dropout)
        blocks_per_stage = tuple(blocks_per_stage)
        if len(blocks_per_stage) != 4 or any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 1
            for count in blocks_per_stage
        ):
            raise ValueError("Quatre nombres positifs de blocs sont attendus.")

        self.stem = nn.Sequential(
            nn.Conv3d(
                1, base_channels, 7, stride=2, padding=3, bias=False
            ),
            normalization(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1),
        )

        layers = []
        incoming = base_channels
        for stage, count in enumerate(blocks_per_stage):
            outgoing = base_channels * (2 ** stage)
            for position in range(count):
                stride = 2 if stage > 0 and position == 0 else 1
                layers.append(ResidualBlock3D(incoming, outgoing, stride))
                incoming = outgoing

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(incoming, 1)

    def forward(self, image):
        check_input(image)
        features = self.features(self.stem(image))
        features = self.pool(features).flatten(1)
        return self.classifier(self.dropout(features)).squeeze(1)


def create_model(
    name, base_channels=8, dropout=0.2, blocks_per_stage=None,
):
    """Construire un modèle avec les paramètres enregistrés pour un essai."""
    if name == "simplecnn3d":
        if blocks_per_stage is not None:
            raise ValueError("La profondeur résiduelle concerne seulement ResNet3D.")
        return SimpleCNN3D(base_channels, dropout)

    if name == "resnet3d":
        blocks = (
            (1, 1, 1, 1)
            if blocks_per_stage is None
            else blocks_per_stage
        )
        return ResNet3D(base_channels, dropout, blocks)

    raise ValueError("Modèle attendu : simplecnn3d ou resnet3d.")


"""Lire les images préparées pour les présenter à PyTorch.

Une image est ouverte à la demande : le jeu complet n'est pas chargé en mémoire.
Les centres servent aux analyses ; ils ne sont pas fournis comme entrée au CNN.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ADHD200Dataset(Dataset):
    """Fournir une image, sa classe et ses références de suivi."""

    def __init__(self, root, split, transform=None):
        self.root = Path(root).resolve()
        if split not in {"train", "validation"}:
            raise ValueError("Groupe train ou validation attendu.")
        if transform is not None and split != "train":
            raise ValueError("Les transformations aléatoires sont réservées au train.")

        marker = self.root / "LOCAL_EXPORT_COMPLETE.txt"
        if not marker.is_file():
            raise RuntimeError("Un export local complet est requis.")

        info = json.loads(
            (self.root / "dataset_info.json").read_text(encoding="utf-8")
        )
        if (
            info.get("schema_version") != 1
            or info.get("array_shape") != [128, 128, 128]
            or info.get("array_dtype") != "float32"
            or info.get("contains_test_data") is not False
        ):
            raise RuntimeError("Description du jeu de données inattendue.")

        table = self.root / "dataset.csv"
        if fingerprint(table) != info["dataset_csv_sha256"]:
            raise RuntimeError("Le tableau ne correspond plus à l'export.")

        with table.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != [
                "image_id", "file", "split", "target", "site", "sha256"
            ]:
                raise RuntimeError("Colonnes du tableau inattendues.")
            all_rows = list(reader)

        identifiers = set()
        for row in all_rows:
            if None in row or any(v is None for v in row.values()):
                raise RuntimeError("Ligne du tableau mal formée.")
            if row["image_id"] in identifiers:
                raise RuntimeError("Identifiant d'image dupliqué.")
            identifiers.add(row["image_id"])
            if row["split"] not in {"train", "validation"}:
                raise RuntimeError("Groupe inattendu.")
            if row["target"] not in {"0", "1"}:
                raise RuntimeError("Classe binaire attendue.")

        self.rows = [row for row in all_rows if row["split"] == split]
        if len(self.rows) != info["groups"].get(split):
            raise RuntimeError("Effectif du groupe incohérent.")
        if not self.rows:
            raise RuntimeError("Le groupe sélectionné est vide.")

        self.split = split
        self.transform = transform
        self.info = info
        self._verified = set()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        relative = Path(row["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Chemin d'image non autorisé.")

        path = self.root / relative
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(self.root)
            or not path.is_file()
        ):
            raise RuntimeError("Fichier d'image absent ou non autorisé.")

        # Vérifier l'empreinte à la première lecture par ce lecteur.
        if index not in self._verified:
            if fingerprint(path) != row["sha256"]:
                raise RuntimeError("Une image a changé depuis son export.")
            self._verified.add(index)

        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["image"]:
                raise RuntimeError("Contenu du fichier inattendu.")
            array = archive["image"]

        if array.shape != (128, 128, 128) or array.dtype != np.float32:
            raise RuntimeError("Dimensions ou type numérique inattendus.")
        if not np.isfinite(array).all():
            raise RuntimeError("L'image contient des valeurs non finies.")

        # Ajouter une dimension pour l'unique canal d'intensité.
        # Format d'une image : canal × profondeur × hauteur × largeur.
        image = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)

        if self.transform is not None:
            image = self.transform(image)
            if (
                not isinstance(image, torch.Tensor)
                or tuple(image.shape) != (1, 128, 128, 128)
                or image.dtype != torch.float32
                or not torch.isfinite(image).all().item()
            ):
                raise RuntimeError("Résultat de transformation inattendu.")

        return {
            "image": image.contiguous(),
            "target": torch.tensor(float(row["target"]), dtype=torch.float32),
            "image_id": row["image_id"],
            "site": row["site"],
        }


"""Exécuter un passage d'apprentissage ou de validation.

Le programme principal décidera du nombre de passages, de l'arrêt
et des sauvegardes à conserver. Ce module exécute les calculs.
"""

from contextlib import nullcontext
import math
import os
from pathlib import Path
import uuid

import torch



def make_optimizer(model, config):
    """Construire AdamW en séparant les poids à pénaliser des autres paramètres."""
    if config["name"] != "AdamW":
        raise ValueError("AdamW est attendu.")

    rate = float(config["learning_rate"])
    decay = float(config["weight_decay"])
    if (
        not math.isfinite(rate) or rate <= 0
        or not math.isfinite(decay) or decay < 0
    ):
        raise ValueError("Vitesse d'apprentissage ou pénalisation invalide.")

    penalized = []
    unpenalized = []
    exclude = config["exclude_normalization_and_bias_from_decay"]

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if exclude and (parameter.ndim <= 1 or name.endswith(".bias")):
            unpenalized.append(parameter)
        else:
            penalized.append(parameter)

    groups = []
    if penalized:
        groups.append({"params": penalized, "weight_decay": decay})
    if unpenalized:
        groups.append({"params": unpenalized, "weight_decay": 0.0})
    if not groups:
        raise ValueError("Le modèle ne contient aucun paramètre ajustable.")

    return torch.optim.AdamW(groups, lr=rate)


def select_precision(device, mode="auto"):
    """Choisir le format des calculs selon les capacités du GPU."""
    if mode not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError("Mode de précision non reconnu.")
    if device.type != "cuda":
        if mode not in {"auto", "float32"}:
            raise ValueError("La précision réduite est réservée ici au GPU.")
        return None

    if mode == "float32":
        return None
    if mode == "auto":
        return (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    if mode == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Ce GPU ne prend pas en charge bfloat16.")
    return torch.bfloat16 if mode == "bfloat16" else torch.float16


def precision_context(device, precision):
    """Activer la précision mixte seulement lorsqu'elle est demandée."""
    if precision is None:
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("Un GPU CUDA est attendu pour la précision mixte.")
    return torch.autocast(device_type="cuda", dtype=precision)


def train_epoch(
    model, loader, optimizer, criterion, device, scaler,
    accumulation_steps=1, clip_norm=1.0, precision=None,
):
    """Ajuster les poids pendant un passage sur l'entraînement.

    Le calcul d'erreur doit utiliser reduction='sum'.
    Les gradients cumulés sont ensuite divisés par le nombre réel d'images.
    Cela traite aussi correctement un dernier lot incomplet.
    """
    if criterion.reduction != "sum":
        raise ValueError("Le calcul d'erreur doit utiliser reduction='sum'.")
    if (
        not isinstance(accumulation_steps, int)
        or accumulation_steps < 1
        or not math.isfinite(float(clip_norm))
        or clip_norm <= 0
    ):
        raise ValueError("Accumulation ou limite de gradients invalide.")
    if len(loader) == 0:
        raise ValueError("Le lecteur d'entraînement est vide.")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_images = 0
    accumulated_images = 0
    updates = 0
    skipped_updates = 0

    for position, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with precision_context(device, precision):
            logits = model(images)
            if logits.shape != targets.shape:
                raise RuntimeError("Dimensions des scores et classes incompatibles.")
            loss = criterion(logits, targets)

        if not torch.isfinite(loss).item():
            raise RuntimeError("Erreur d'apprentissage non finie.")

        count = targets.numel()
        if count == 0:
            raise RuntimeError("Un lot vide a été rencontré.")
        total_loss += float(loss.detach().cpu())
        total_images += count
        accumulated_images += count

        # Cumuler les ajustements calculés, sans modifier encore les poids.
        scaler.scale(loss).backward()

        boundary = (
            position % accumulation_steps == 0
            or position == len(loader)
        )
        if not boundary:
            continue

        # Revenir à l'échelle réelle avant de normaliser et limiter les gradients.
        scaler.unscale_(optimizer)
        parameters = [
            parameter for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not parameters:
            raise RuntimeError("Aucun gradient n'a été calculé.")

        finite = all(
            torch.isfinite(parameter.grad).all().item()
            for parameter in parameters
        )

        if not finite:
            # En float16, le mécanisme de mise à l'échelle peut ignorer
            # un ajustement qui déborde, puis réduire son échelle.
            if not scaler.is_enabled():
                raise RuntimeError("Gradients non finis sans correction de précision.")
            scaler.step(optimizer)
            scaler.update()
            skipped_updates += 1
        else:
            for parameter in parameters:
                parameter.grad.div_(accumulated_images)

            torch.nn.utils.clip_grad_norm_(
                parameters, max_norm=clip_norm, error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            updates += 1

        optimizer.zero_grad(set_to_none=True)
        accumulated_images = 0

    if updates == 0:
        raise RuntimeError("Aucun ajustement valide des poids n'a été effectué.")

    return {
        "loss": total_loss / total_images,
        "images": total_images,
        "optimizer_updates": updates,
        "skipped_updates": skipped_updates,
    }


@torch.inference_mode()
def evaluate_epoch(
    model, loader, criterion, device, threshold=0.5, precision=None,
):
    """Mesurer les résultats sans calculer de gradients ni modifier les poids."""
    if criterion.reduction != "sum":
        raise ValueError("Le calcul d'erreur doit utiliser reduction='sum'.")

    model.eval()
    total_loss = 0.0
    targets_all = []
    scores_all = []
    sites_all = []
    predictions = []
    seen_identifiers = set()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with precision_context(device, precision):
            logits = model(images)
            if logits.shape != targets.shape:
                raise RuntimeError("Dimensions des scores et classes incompatibles.")
            loss = criterion(logits, targets)

        if not torch.isfinite(loss).item():
            raise RuntimeError("Erreur de validation non finie.")

        # Convertir le score brut en un score compris entre 0 et 1.
        scores = torch.sigmoid(logits.float()).cpu().tolist()
        labels = targets.cpu().tolist()
        identifiers = list(batch["image_id"])
        sites = list(batch["site"])

        if not (
            len(scores) == len(labels) == len(identifiers) == len(sites)
        ):
            raise RuntimeError("Références du lot incompatibles.")

        total_loss += float(loss.cpu())
        targets_all.extend(labels)
        scores_all.extend(scores)
        sites_all.extend(sites)

        for identifier, site, label, score in zip(
            identifiers, sites, labels, scores, strict=True
        ):
            if identifier in seen_identifiers:
                raise RuntimeError("Une image apparaît deux fois dans la validation.")
            seen_identifiers.add(identifier)
            predictions.append({
                "image_id": identifier,
                "site": site,
                "target": int(label),
                "score": float(score),
            })

    if not targets_all:
        raise RuntimeError("La validation est vide.")

    return {
        "loss": total_loss / len(targets_all),
        "metrics": grouped_metrics(
            targets_all, scores_all, sites_all, threshold
        ),
        "predictions": predictions,
    }


def save_checkpoint(path, state):
    """Publier une sauvegarde seulement après son écriture complète."""
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ValueError("Une sauvegarde de version 1 est attendue.")

    path = Path(path)
    if path.is_symlink():
        raise RuntimeError("Emplacement de sauvegarde non autorisé.")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.parent / (path.name + "." + uuid.uuid4().hex + ".tmp")

    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path):
    """Lire les tenseurs et valeurs simples d'une sauvegarde du projet."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Sauvegarde absente ou non autorisée.")

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise RuntimeError("Format de sauvegarde non reconnu.")
    return state
