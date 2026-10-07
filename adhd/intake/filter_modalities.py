"""Reconnaître les types d'images à partir des noms observés."""

def classify(relative):
    """Renvoyer anatomical, rest ou unsupported.

    Cette reconnaissance ne remplace pas le contrôle du contenu.
    """
    parts = [part.lower() for part in relative.parts]
    name = parts[-1]
    folders = parts[:-1]

    # Une indication REST est prioritaire sur une indication anatomique.
    if (
        any(part == "rest" or part.startswith("rest_") for part in folders)
        or name in {"rest.nii", "rest.nii.gz"}
        or "_task-rest_" in name
    ):
        return "rest"

    anatomy = any(
        part == "anat" or part.startswith("anat_") for part in folders
    )
    t1 = (
        name in {"mprage.nii", "mprage.nii.gz"}
        or name.endswith(("_t1w.nii", "_t1w.nii.gz"))
    )
    return "anatomical" if anatomy and t1 else "unsupported"
