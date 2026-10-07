"""Brouiller l'identité des patients (pseudonymisation).

Principe : le numéro de dossier est remplacé par un code calculé avec une clé
secrète (HMAC-SHA256). Sans la clé, on ne peut pas retrouver le numéro, même
en essayant toutes les combinaisons. Avec la même clé, un même patient reçoit
toujours le même code : c'est ce qui permet de le suivre sans le nommer.

La clé reste hors du code et hors du dépôt Git. Ce mécanisme protège les
identifiants ; il ne rend pas les images anonymes.
"""
import hashlib
import hmac
import os
from pathlib import Path
import re
import stat

# Code numérique de chaque hôpital dans la collection ADHD-200.
SITE_CODES = {
    "PEK": "1", "Brown": "2", "KKI": "3", "NeuroIMAGE": "4",
    "NYU": "5", "OHSU": "6", "Pittsburgh": "7", "WashU": "8",
}
PSEUDONYM = re.compile(r"sub_[0-9a-f]{32}")
PATIENT_FOLDER = re.compile(r"(?:sub-)?([0-9]{5,7})")
KEY_FILE = "/run/secrets/pseudonymization_key"


def normalize_identifier(value):
    """Reconnaître un même numéro malgré des zéros initiaux."""
    text = str(value).strip()
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError("Un identifiant numérique est attendu.")
    return text.lstrip("0") or "0"


def load_key(path=None):
    """Lire la clé secrète (32 octets, lisible par son seul propriétaire)."""
    path = Path(path or os.environ.get("PSEUDONYMIZATION_KEY_FILE", KEY_FILE))
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Fichier de clé absent ou non autorisé.")
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise RuntimeError("La clé doit être accessible à son propriétaire seul.")
    key = path.read_bytes()
    if len(key) != 32:
        raise RuntimeError("La clé doit contenir exactement 32 octets.")
    return key


def pseudonymize(site_code, patient_identifier, key):
    """Calculer le code d'un patient à partir de son hôpital et de son numéro."""
    if not isinstance(key, bytes) or len(key) != 32:
        raise ValueError("Clé invalide.")
    site = normalize_identifier(site_code)
    patient = normalize_identifier(patient_identifier)
    message = f"adhd200:subject:v1:{site}:{patient}".encode("utf-8")
    return "sub_" + hmac.new(key, message, hashlib.sha256).hexdigest()[:32]


def subject_for(row, key_loader):
    """Renvoyer le code brouillé d'une image annoncée dans un manifeste de lot.

    - le lot fournit un numéro de dossier (patient_id) : le pipeline le brouille ;
    - le lot fournit un code déjà brouillé (subject) : il est accepté tel quel ;
    - tout autre identifiant est REFUSÉ : aucun numéro en clair n'entre en base.
    """
    has_raw, has_code = "patient_id" in row, "subject" in row
    if has_raw == has_code:
        raise ValueError("Fournir patient_id ou subject, pas les deux.")
    if has_raw:
        return pseudonymize(SITE_CODES[row["site"]], row["patient_id"], key_loader())
    subject = str(row["subject"])
    if PSEUDONYM.fullmatch(subject):
        return subject
    if row.get("cohort") == "synthetic" and 1 <= len(subject) <= 100:
        return subject  # fichiers fabriqués pour les tests : aucun patient réel
    raise ValueError("Identifiant non pseudonymisé refusé.")


def patient_from_path(relative):
    """Retrouver le numéro de dossier dans un chemin du dépôt brut,
    par exemple Brown/0026001/session_1/anat_1/mprage.nii.gz."""
    found = {
        match.group(1) for part in Path(relative).parts[1:-1]
        if (match := PATIENT_FOLDER.fullmatch(part))
    }
    if len(found) != 1:
        raise ValueError("Numéro de dossier introuvable ou ambigu.")
    return found.pop()
