# Bloc 3 — matrice des exigences et des preuves

Consolidation du 7 octobre 2026. Sources : référentiel Architecte en intelligence artificielle, bloc 3 ; modalités de plateforme ; trois transcrits COACHING fournis. Les transcrits portent sur les blocs 2/3 et ne constituent pas une source d'exigences du bloc 4.

Le référentiel demande un plan du pipeline, le code hébergé sur GitHub et une capture vidéo du pipeline en fonctionnement. La plateforme demande aussi un support de présentation. La présentation orale est de cinq minutes, suivie de quinze minutes de questions. Aucune durée obligatoire de vidéo n'est déduite de ces cinq minutes.

## Preuves observées

| Contrôle | Résultat | Rapport sous ADHD200_DATA/reports/compact |
| --- | --- | --- |
| Premier lot Brown | Trois IRM réelles prédites, un fichier artificiellement corrompu isolé ; deux passages suivants sans nouvelles prédictions | first_real_batch_verified.json |
| Panne temporaire de l'API | collect en up_for_retry, seconde tentative réussie, puis prepare et predict ; même exécution Airflow | b3_retry_20261006T235837Z_c1da532d.json |
| Nouvelle livraison d'un doublon | Nouveau manifeste et nouveau lot, contenu déjà présent ; zéro nouvel élément et zéro nouvelle prédiction | b3_duplicate_fe5407d58efb4b09b46ec45dca865c40.json |

Le premier passage était planifié ; l'exercice de panne a été déclenché manuellement une seule fois pour contrôler la relance automatique. Ces deux propriétés sont démontrées séparément. La déduplication concerne les octets identiques d'un fichier, pas toutes les conversions d'un même examen. Les rapports restent datés : une prédiction ultérieure avec une autre version du modèle ne rend pas invalide le contrôle antérieur, mais exige de comparer les doublons par image et version.

## Couverture : 21 indicateurs

| Repère | Attente | État et travail à clôturer |
| --- | --- | --- |
| 1.1 | Choix du mode de traitement | Batch de fichiers lourds ; pas de besoin clinique instantané ; planification cinq minutes. Mesurer le délai dépôt/résultat sur les journaux. |
| 1.2 | Collecte sans perte | Reçus, empreintes et lot réel vérifiés ; doublon explicite neutralisé. |
| 1.3 | Diagramme avec entrées, sorties et stockage | Schéma fourni dans le README ; le reporter sur le support et vérifier les montages du compose actuel. |
| 2.1 | Transformations modulaires | Modules livrés ; contrôler le code actuel et sa réutilisation historique/arrivées dans la revue. |
| 2.2 | Préparation adaptée au modèle | Volumes 128³ ; compatibilité contrôlée sur cinq centres ; calcul CPU/GPU vérifié. |
| 2.3 | Retraitement après changement des règles | Mécanisme de signature livré ; démonstration de retraitement isolé avec ancienne sortie conservée encore à confirmer. |
| 3.1 | Ordonnancement automatique | Premier lot planifié ; filmer une exécution de type scheduled, et non uniquement l'exercice manuel. |
| 3.2 | Dépendances sur réussite | Pendant la panne, prepare/predict non lancées ; exécution réussie après retry. |
| 3.3 | Reproductibilité dans un autre environnement | WSL CPU et Kaggle GPU utilisés ; cela ne prouve pas à lui seul la réexécution identique du même traitement. Compléter une preuve sur environnement propre/CI. |
| 4.1 | Validation des formats/types | Tests Docker réussis et contrôles NIfTI/NPZ livrés ; citer les tests utiles, conserver leur sortie. |
| 4.2 | Isolation des anomalies | Fichier corrompu isolé ; trois IRM valides traitées. |
| 4.3 | Relance automatique | Panne API, état up_for_retry, seconde tentative et même DAG run réussis. |
| 5.1 | Masquage des champs sensibles | Pseudonymisation HMAC dans le dépôt (`adhd/identity.py`), appliquée par l'étape de collecte ; identifiant en clair refusé ; tests ajoutés. À confirmer sur WSL : tests Docker, `identity-check`, lot réel avec numéros de dossier. Ne pas appeler cela une anonymisation certifiée. |
| 5.2 | Secrets externalisés | Secrets montés dans les conteneurs ; auditer la sélection Git avant publication. |
| 5.3 | Transferts sécurisés | Services publiés sur loopback, HTTPS distant ; réseau Docker local distinct d'un transit inter-sites chiffré. Documenter la limite et la destination Kaggle privée. |
| 6.1 | Alertes panne/délai | Code et test watchdog présents ; vérifier l'alerte visible et la cadence réelle. Ne pas promettre une alerte instantanée avec un contrôle périodique. |
| 6.2 | Coûts calcul/transfert | Estimation produite par le pipeline (`adhd/costs.py`) : durées mesurées × tarifs déclarés comme hypothèses dans `configs/service.yaml` ; affichée dans le tableau de bord ; rapport complet par `adhd.cli costs`. Heures de GPU à renseigner. |
| 6.3 | Journaux d'audit | Journaux Airflow et événements persistants ; montrer l'erreur et sa reprise avec le même run ID. |
| 7.1 | Lignage | source SHA → volume préparé/reçu → version → prédiction ; montrer un trajet vérifié, sans numéro patient brut. |
| 7.2 | Backlog/priorisation | Priorité actuelle : terminer B3 avant nouvelle surveillance B4 ; statuts ci-dessous. |
| 7.3 | Documentation accessible/reprise | README, opérations et décisions disponibles ; audit du code actuel, commandes et installation par un tiers à consolider. |

## Backlog de clôture

1. Acquis : premier lot, corruption, persistance de la file, retry Airflow, doublon explicite.
2. En cours : audit de la version installée, documentation, lignage et mesures.
3. À confirmer : retraitement historique isolé, réexécution sur environnement propre, visibilité des alertes.
4. À produire : schéma/support final, publication GitHub contrôlée, répétition et vidéo B3.
5. Différé : Evidently et suite de surveillance B4. Les trois lots Brown 9/9/5 ont été déposés ; vérifier leur état avant d'annoncer 26 IRM traitées.

Aucune ligne de cette matrice n'attribue une note ou ne certifie une conformité juridique ou médicale.
