# Fiche d'identité du modèle — `adhd200` version 3 (champion)

> **Outil d'aide à la recherche. Ce modèle ne pose aucun diagnostic.**
> Fiche établie le 7 octobre 2026 à partir des rapports d'entraînement et d'évaluation du projet.

## 1. En bref

| | |
| --- | --- |
| Rôle | Donner, pour une IRM anatomique du cerveau, un score entre 0 et 1 : ressemblance avec les images de patients ayant un TDAH |
| Architecture | ResNet3D : réseau à convolutions 3D avec raccourcis, 4 étages de 8, 16, 32 et 64 filtres |
| Taille | Environ 227 000 paramètres |
| Entrée | Un volume de 128 × 128 × 128 voxels, intensités entre 0 et 1 |
| Sortie | Un score ; au-dessus du seuil de 0,5, l'image est rangée du côté « TDAH » |
| Version du registre | 3, étiquette `champion` (MLflow sur DagsHub) |
| Usage prévu | Démonstration d'un pipeline complet ; exploration de recherche |
| Usages exclus | Diagnostic, dépistage, toute décision concernant une personne |

## 2. Données

Collection publique ADHD-200, IRM anatomiques. 960 images acceptées, dont 934 avec diagnostic connu (575 témoins, 359 TDAH).

| Groupe | Images | Témoins | TDAH | Rôle |
| --- | --- | --- | --- | --- |
| Entraînement | 563 | 336 | 227 | Apprendre |
| Validation | 119 | 71 | 48 | Choisir le modèle |
| Test interne | 119 | 71 | 48 | Noter, mêmes hôpitaux, patients jamais vus |
| Test externe (NeuroIMAGE) | 73 | 37 | 36 | Noter, hôpital jamais vu |
| Témoins externes (WashU) | 60 | 60 | 0 | Noter la reconnaissance des témoins ailleurs |

Hôpitaux d'entraînement : KKI, NYU, OHSU, PEK, Pittsburgh. Un patient n'appartient qu'à un seul groupe. Cible : 0 pour un témoin, 1 pour l'un des trois sous-types de TDAH, regroupés.

Préparation : même orientation, voxels de 1 mm, suppression des bords vides, intensités ramenées entre 0 et 1, réduction à 128³ sans déformation. Ni retrait du crâne, ni effacement du visage.

## 3. Entraînement

| Réglage | Valeur |
| --- | --- |
| Lieu | Kaggle, sur carte graphique |
| Optimiseur | AdamW, pénalisation des poids 0,0001 |
| Vitesse d'apprentissage | 0,001, divisée par 2 quand la validation stagne 3 époques |
| Durée | 40 époques au maximum, arrêt anticipé après 8 époques sans progrès |
| Lot | 2 images, cumulées par 4 |
| Fonction d'erreur | `BCEWithLogitsLoss`, classe TDAH pondérée (336 / 227 ≈ 1,48) |
| Graine | 42 |
| Images modifiées au hasard | Aucune |

Quatre entraînements ont été comparés : deux architectures (SimpleCNN3D, ResNet3D) et deux vitesses d'apprentissage (0,001 et 0,0003). Tout le reste était identique.

## 4. Résultats

### Choix du modèle, sur la validation (119 images)

| Modèle | Vitesse | AUC | Sensibilité | Spécificité | Exactitude équilibrée | Décision |
| --- | --- | --- | --- | --- | --- | --- |
| SimpleCNN3D | 0,001 | 0,589 | 0 | 1 | 0,500 | Refusé : répond toujours « témoin » |
| SimpleCNN3D | 0,0003 | 0,616 | 0 | 1 | 0,500 | Refusé : répond toujours « témoin » |
| **ResNet3D** | **0,001** | **0,704** | **0,667** | **0,535** | **0,601** | **Champion (version 3)** |
| ResNet3D | 0,0003 | 0,628 | 0,813 | 0,380 | 0,596 | Challenger (version 4) |

### Évaluation finale du champion

Calculée une seule fois, après le choix du champion. Aucun réglage n'a été modifié ensuite.

| Groupe | Images | AUC | Sensibilité | Spécificité | Exactitude équilibrée |
| --- | --- | --- | --- | --- | --- |
| Test interne | 119 | 0,774 | 0,750 (36 sur 48) | 0,676 (48 sur 71) | 0,713 |
| Test externe (NeuroIMAGE) | 73 | 0,615 | 0,694 (25 sur 36) | 0,595 (22 sur 37) | 0,645 |
| Témoins externes (WashU) | 60 | sans objet | sans objet | 0,433 (26 sur 60) | sans objet |

### Test interne, par hôpital

| Hôpital | Images | TDAH | AUC |
| --- | --- | --- | --- |
| OHSU | 16 | 6 | 0,817 |
| NYU | 40 | 23 | 0,660 |
| PEK | 36 | 15 | 0,644 |
| KKI | 12 | 3 | 0,556 |
| Pittsburgh | 15 | 1 | 1,000 (sans valeur : un seul TDAH) |

## 5. Lecture des résultats

- **Le modèle classe mieux que le hasard** sur les hôpitaux qu'il connaît (AUC 0,77 en test interne), sans être fiable.
- **Il se dégrade sur un hôpital jamais vu** : l'AUC tombe à 0,61 sur NeuroIMAGE, dont les patients sont aussi plus âgés.
- **Sur WashU, il donne 34 fausses alertes pour 60 témoins.** Sa spécificité y passe sous 0,5 : sur un appareil inconnu, il range la majorité des témoins du côté « TDAH ».
- **Les résultats varient fortement d'un hôpital à l'autre.** Une partie du score global peut venir de la reconnaissance de l'appareil plutôt que du trouble.
- **Les effectifs sont faibles** : sur 119 images, une AUC a une marge d'environ ± 0,10.

## 6. Limites et risques

- Peu d'images pour un réseau 3D : risque d'apprentissage par cœur.
- Proportions de TDAH très différentes selon les hôpitaux (de 4 % à 58 %), ainsi que l'âge et le sexe.
- Un seul entraînement par réglage : la variabilité entre deux tirages n'est pas mesurée.
- Scores non calibrés : un score de 0,9 ne signifie pas 90 % de chances.
- Crâne et visage présents dans les images.
- Le contrôle de dérive du pipeline n'a pas signalé WashU, alors que le modèle y est dégradé : une dérive mesurée sur des indicateurs simples ne remplace pas la mesure de la performance.

## 7. Garde-fous en service

- Un modèle qui répond toujours la même chose est refusé (`configs/service.yaml`).
- Score de classement minimal de 0,55 et exactitude équilibrée supérieure à 0,50 pour être mis en service.
- Mise en service progressive : 10 % des demandes au challenger, promotion seulement s'il égale le champion.
- Retour à la version précédente par déplacement de l'étiquette.
- Chaque prédiction enregistre la version du modèle ; un avertissement « pas un diagnostic » figure dans l'API et le tableau de bord.
- Une carte de sensibilité montre les zones de l'image qui ont pesé dans le score. Elle n'indique aucune cause médicale.

## 8. Surveillance

- Contrôle de dérive toutes les 5 minutes sur quatre mesures des images ; rapport Evidently à la demande (`adhd/drift_report.py`).
- Mesure de la performance réelle quand le diagnostic est connu après coup.
- Demande de réentraînement préparée quand assez de nouvelles images étiquetées sont réunies ; jamais de mise en service automatique.
