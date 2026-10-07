"""Préparer une image avec une recette déterministe et mesurable.

Cette première recette est en cours de vérification.
Elle ne réalise ni extraction du cerveau ni anonymisation du visage.
"""

import numpy as np
import torch
from monai.transforms import (
    LoadImage, EnsureChannelFirst, Orientation, Spacing,
    CropForeground, ScaleIntensityRange, Resize, SpatialPad,
)


def prepare_image(path, config, with_preview=False):
    """Retourner les valeurs, leur position spatiale et les mesures."""

    import nibabel as nib
    native = nib.load(str(path))
    if len(native.shape) != 3 or np.prod(native.shape) > 100_000_000 or native.header.get_xyzt_units()[0] != "mm" or not np.isfinite(native.affine).all():
        raise ValueError("Volume 3D en millimètres requis")
    extent = (np.asarray(native.shape)-1)*np.linalg.norm(native.affine[:3,:3],axis=0)
    if np.max(extent)>500 or abs(np.linalg.det(native.affine[:3,:3]))<1e-8:
        raise ValueError("Géométrie atypique à revoir")
    recipe = config["preprocessing"]
    edge = recipe["output_edge"]
    percentiles = recipe["intensity_percentiles"]
    spacing = recipe["intermediate_spacing_mm"]

    if (
        not isinstance(edge, int) or edge < 2
        or len(percentiles) != 2
        or not 0 <= percentiles[0] < percentiles[1] <= 100
        or len(spacing) != 3
        or any(float(value) <= 0 for value in spacing)
        or recipe["orientation"] != "RAS"
    ):
        raise ValueError("Paramètres de préparation invalides.")

    image = LoadImage(image_only=True, dtype=np.float32)(str(path))
    if image.ndim != 3 or not torch.isfinite(image).all().item():
        raise ValueError("Un volume 3D aux valeurs finies est attendu.")

    native_shape = list(image.shape)
    image = EnsureChannelFirst(channel_dim="no_channel")(image)
    image = Orientation(axcodes="RAS", labels=(("L", "R"), ("P", "A"), ("I", "S")))(image)
    canonical_shape = list(image.shape[1:])

    preview_before = (
        image.detach().cpu().numpy()[0].copy() if with_preview else None
    )

    image = Spacing(
        pixdim=tuple(spacing), mode="bilinear",
    )(image)
    spaced_shape = list(image.shape[1:])

    image = CropForeground(
        select_fn=lambda values: values != 0,
        allow_smaller=True,
    )(image)
    cropped_shape = list(image.shape[1:])

    values = image.detach().cpu().numpy()
    nonzero = values[values != 0]
    if nonzero.size == 0:
        raise ValueError("Le volume ne contient aucune valeur non nulle.")

    low, high = np.percentile(nonzero, percentiles)
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("Bornes d'intensité inexploitables.")

    background = image == 0
    image = ScaleIntensityRange(
        a_min=float(low), a_max=float(high),
        b_min=0, b_max=1, clip=True,
    )(image)
    image[background] = 0

    nominal_resize_factor = edge / max(cropped_shape)
    image = Resize(
        spatial_size=edge, size_mode="longest",
        mode="trilinear", align_corners=False, anti_aliasing=True,
    )(image)
    resized_shape = list(image.shape[1:])

    image = SpatialPad(
        spatial_size=(edge, edge, edge), mode="constant",
    )(image)

    array = image.detach().cpu().numpy()[0].astype(np.float32)
    affine = image.affine.detach().cpu().numpy().copy()

    if array.shape != (edge, edge, edge):
        raise RuntimeError("Dimensions finales inattendues.")
    if not np.isfinite(array).all() or not np.isfinite(affine).all():
        raise RuntimeError("Valeurs ou position spatiale invalides.")
    if array.max() <= array.min():
        raise RuntimeError("Image uniforme après préparation.")
    if array.min() < -0.00001 or array.max() > 1.00001:
        raise RuntimeError("Intensités finales hors de l'intervalle attendu.")

    return {
        "array": array,
        "affine": affine,
        "preview_before": preview_before,
        "statistics": {
            "recipe_version": recipe["recipe_version"],
            "native_shape": native_shape,
            "canonical_shape": canonical_shape,
            "intermediate_shape": spaced_shape,
            "cropped_shape": cropped_shape,
            "resized_shape": resized_shape,
            "final_shape": list(array.shape),
            "crop_removed_fraction": float(
                1 - np.prod(cropped_shape) / np.prod(spaced_shape)
            ),
            "nonzero_fraction_after_crop": float(
                nonzero.size / values.size
            ),
            "nominal_resize_factor": float(nominal_resize_factor),
            "intensity_lower_bound": float(low),
            "intensity_upper_bound": float(high),
            "final_spacing_mm": np.linalg.norm(
                affine[:3, :3], axis=0
            ).tolist(),
            "final_minimum": float(array.min()),
            "final_maximum": float(array.max()),
        },
    }
