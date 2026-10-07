"""Étapes amont du trajet historique : du dépôt brut à la répartition des patients.

Ordre d'exécution (chaque étape lit le rapport de la précédente) :
  1. technical_profile  contrôle technique de chaque IRM (lisible, 3D, géométrie)
  2. validate_dataset   liste des images acceptées ; les autres sont mises à l'écart
  3. match_phenotypes   rapprochement avec les diagnostics + pseudonymisation
  4. bias_audit         effectifs, âge et sexe par hôpital
  5. split_dataset      répartition des patients (entraînement, validation, tests)

Lancement :  python -m adhd.cli intake <étape> -- <arguments de l'étape>
"""

STEPS = ("technical_profile", "validate_dataset", "match_phenotypes",
         "bias_audit", "split_dataset")
