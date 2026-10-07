# Historique des décisions

## 15. Registre des décisions d’architecture

Ce registre décrit les choix du protocole actuel, leurs motivations et
leurs limites. Il sera enrichi à chaque décision structurante.

Un choix initial n’est pas nécessairement optimal. Les changements futurs
doivent préciser leur raison, les éléments concernés et les conséquences
sur la comparabilité des résultats.

### 15.1. Signification des exclusions

| Type de décision | Application | Conséquence |
| --- | --- | --- |
| Exclusion de modalité | 293 images REST | Hors du périmètre du modèle anatomique ; sources conservées |
| Quarantaine de qualité | Une image OHSU | Exclusion provisoire des listes acceptées ; original conservé |
| Réservation de centre | NeuroIMAGE, WashU, Brown | Rôle distinct de l’apprentissage ; données conservées |
| Exclusion du paquet Kaggle | Tests et réserve Brown | Absents du paquet d’apprentissage/validation |
| Exclusion Git | Données, secrets, poids et résultats locaux | Absents du dépôt de code |
| Report du versionnement | Fichiers de squelette encore vides | Conservés localement ; ajoutés lorsqu’ils seront renseignés |

La réservation d’un centre n’est pas une déclaration d’inutilité de ses
données. Une exclusion Git n’est pas une suppression sur le disque.

### 15.2. Projet, environnements et stockage

| Référence | Décision | Justification | Conséquence ou limite |
| --- | --- | --- | --- |
| ADR-01 | Nouveau projet et nouveau dépôt DagsHub | Construire une base identifiable sans modifier l’ancien travail | Les anciens résultats ne deviennent pas automatiquement des résultats du nouveau protocole |
| ADR-02 | Séparer `ADHD200_PIPELINE` et `ADHD200_DATA` | Versionner le code sans incorporer les IRM et correspondances privées | Les chemins et procédures de raccordement doivent être documentés |
| ADR-03 | Dépôt DagsHub privé | Limiter l’accès au code et aux futurs résultats partagés | La visibilité privée ne remplace pas la gestion des secrets |
| ADR-04 | Initialiser Git dans le projet existant | Éviter un second dossier issu d’un clone vide | Une seule copie locale de référence pour ce nouveau projet |
| ADR-05 | Docker Python 3.12 pour les traitements locaux | Isoler les bibliothèques du Python de l’hôte | La construction doit enregistrer les versions effectivement résolues |
| ADR-06 | PyTorch CPU local et PyTorch GPU Kaggle | Réserver le calcul GPU à l’environnement d’entraînement prévu | La disponibilité CUDA locale n’est pas attendue ; elle est exigée pour les entraînements réels |
| ADR-07 | Utilisateur Docker non administrateur aligné sur l’UID/GID WSL | Permettre les accès aux montages et fournir un utilisateur reconnu par MONAI | L’image doit être construite avec les numéros de l’utilisateur concerné |
| ADR-08 | Montages bruts en lecture seule et réseau désactivé pour les contrôles locaux | Limiter les modifications accidentelles et les communications inutiles pendant ces contrôles | Les services distants nécessiteront un fonctionnement réseau distinct et documenté |
| ADR-09 | Dépendances initialement non figées | Construire d’abord un environnement fonctionnel et relever ses versions | Une reconstruction ultérieure peut résoudre d’autres versions ; le verrouillage reste à réaliser |

### 15.3. Validation, protection et traçabilité

| Référence | Décision | Justification | Conséquence ou limite |
| --- | --- | --- | --- |
| ADR-10 | Modèle centré sur les IRM anatomiques, sans REST | Garder un périmètre cohérent de classification sur volumes 3D | Les informations fonctionnelles des séquences REST ne sont pas exploitées |
| ADR-11 | Reconnaissance des modalités par règles explicites de noms et dossiers | Adapter le filtrage aux organisations effectivement inventoriées | Les noms ne prouvent pas à eux seuls la nature médicale d’une acquisition |
| ADR-12 | Contrôler les valeurs de l’image entière | Détecter notamment les lectures incomplètes et valeurs non finies | Un contrôle technique réussi ne valide pas la qualité médicale |
| ADR-13 | Exclure provisoirement la géométrie OHSU atypique sans corriger l’en-tête | Ne pas inventer une réparation dont la cause n’est pas établie | Une revue documentée est nécessaire avant une éventuelle réintégration |
| ADR-14 | Quarantaine logique par manifestes | Empêcher l’utilisation des fichiers concernés tout en préservant les sources | Les traitements suivants doivent consommer les listes validées |
| ADR-15 | Comparer les identifiants numériques après normalisation des zéros initiaux | Rapprocher les représentations différentes d’un même numéro | Les duplications ou ambiguïtés après normalisation doivent être refusées |
| ADR-16 | Vérifier conjointement le patient et le centre lors du rapprochement | Éviter une association sur le seul numéro patient | Une incompatibilité conduit à un rejet, pas à une correction arbitraire |
| ADR-17 | Pseudonymes HMAC avec clé extérieure au code | Produire des codes stables sans publier les numéros patients | La stabilité dépend de la conservation sécurisée de la clé ; ce n’est pas une anonymisation |
| ADR-18 | Rapports détaillés privés et sorties agrégées séparées | Conserver les preuves et correspondances sans les mélanger aux livrables diffusables | Un rapport agrégé doit encore être examiné avant diffusion |
| ADR-19 | Empreintes SHA-256 et écritures temporaires puis remplacement | Vérifier l’intégrité et éviter la lecture de fichiers partiellement écrits | Les empreintes ne prouvent ni la qualité médicale ni l’absence de doublons après conversion |

### 15.4. Protocole expérimental et rôles des centres

| Référence | Décision | Justification | Conséquence ou limite |
| --- | --- | --- | --- |
| ADR-20 | Regrouper les codes DX 1, 2 et 3 en cible ADHD | Définir une première tâche binaire avec la nomenclature source | Les sous-types ne sont pas distingués par ce modèle |
| ADR-21 | Conserver les diagnostics inconnus hors de l’apprentissage supervisé | Ne pas fabriquer de cible | Brown ne peut pas fournir une mesure de performance diagnostique |
| ADR-22 | Réserver Brown aux arrivées simulées Airflow | Démontrer des arrivées sans retirer de patients étiquetés à l’apprentissage | L’arrivée est simulée et doit être annoncée au jury |
| ADR-23 | Réserver NeuroIMAGE au test externe | Mesurer le transfert vers un centre absent du développement | Les différences d’âge et de centre sont confondues dans cette évaluation |
| ADR-24 | Réserver WashU comme témoins externes | Étudier les faux positifs hors du développement sur un centre à une seule classe | Sensibilité et ROC-AUC binaire ne sont pas calculables sur ce groupe |
| ADR-25 | Maintenir Pittsburgh dans le développement | Conserver ses données malgré un déséquilibre important et la présence de quatre patients ADHD | Choix discuté, distinct de la proposition d’exclusion de Claude ; risque d’apprentissage de signatures de centre à examiner |
| ADR-26 | Ajouter un test interne dans le protocole v2 | Séparer la sélection du modèle de son évaluation sur les centres de développement | Moins de patients servent à ajuster les poids |
| ADR-27 | Répartir par patient et stratifier par centre/classe | Éviter le partage d’un patient entre groupes et maintenir les deux classes dans le développement | Les petits groupes imposent des arrondis ; la stratification ne supprime pas les biais |
| ADR-28 | Fixer la graine et un classement déterministe | Reproduire les affectations indépendamment de l’ordre des fichiers | Une modification de clé, protocole ou population peut changer les affectations |
| ADR-29 | Conserver les anciennes répartitions et identifier explicitement la v2 | Préserver l’historique des décisions | Les listes anciennes ne doivent pas être sélectionnées implicitement |

Une revue externe a recommandé d’écarter Pittsburgh et WashU de
l’apprentissage, d’ajouter un test interne et de réserver Brown aux
arrivées. Le protocole actuel retient le test interne et Brown comme
réserve ; WashU reste externe et Pittsburgh reste dans le développement.

La pertinence du maintien de Pittsburgh n’est pas démontrée par les seuls
effectifs. Une étude de sensibilité avec et sans ce centre peut être
envisagée sur les données de développement. Elle doit être définie sans
utiliser les tests finaux pour choisir la stratégie.

### 15.5. Préparation et transfert

| Référence | Décision | Justification | Conséquence ou limite |
| --- | --- | --- | --- |
| ADR-30 | Réorienter RAS puis rééchantillonner à 1 mm intermédiaire | Harmoniser les axes et les étapes de traitement | Cela ne réalise pas de recalage anatomique entre patients |
| ADR-31 | Recadrer les valeurs non nulles et normaliser par percentiles propres à chaque image | Réduire le fond et limiter l’influence des intensités extrêmes | Le bruit influence le recadrage ; aucune extraction du cerveau n’est réalisée |
| ADR-32 | Redimensionner le plus grand axe puis compléter à 128³ | Fournir une entrée commune aux CNN en conservant approximativement les proportions | L’échelle physique finale varie et les informations fines sont réduites |
| ADR-33 | Tester la recette uniquement sur des images d’apprentissage | Développer le prétraitement sans utiliser les tests pour ses réglages | Les pilotes ne garantissent pas la qualité de toutes les images |
| ADR-34 | Couvrir les formats supplémentaires avant le traitement historique | Examiner les dimensions natives absentes du premier pilote | La couverture porte sur centre/dimensions, pas sur toutes les variations d’acquisition |
| ADR-35 | Réutiliser les sorties préparées dont les reçus et empreintes sont valides | Limiter les recalculs et faciliter les reprises | Une reprise après interruption forcée doit encore être démontrée |
| ADR-36 | Exporter seulement train et validation en NPZ | Alimenter Kaggle sans transmettre les tests ni les en-têtes NIfTI | L’anatomie faciale peut rester présente ; les NPZ ne conservent pas l’affine |
| ADR-37 | Remplacer les pseudonymes internes par des numéros propres au paquet | Réduire les correspondances directement transportées | Le lien de traçabilité reste dans un rapport privé séparé |
| ADR-38 | Vérifier intégralement les 682 fichiers exportés | Contrôler le paquet réellement produit avant son utilisation | Ce contrôle d’intégrité ne valide pas la protection avant transfert |

### 15.6. Modèles, optimisation et évaluation

| Référence | Décision | Justification | Conséquence ou limite |
| --- | --- | --- | --- |
| ADR-39 | Comparer un CNN simple et une architecture résiduelle compacte | Disposer de deux capacités de modèle dans un budget maîtrisable | Leur pertinence doit être mesurée ; aucun champion n’est encore sélectionné |
| ADR-40 | Utiliser GroupNorm | Éviter de dépendre des statistiques d’un petit lot comme avec BatchNorm | Ce choix peut être réexaminé dans une comparaison contrôlée |
| ADR-41 | Ne pas fournir le centre comme entrée au CNN | Évaluer un modèle fondé sur les volumes | Le centre peut néanmoins rester identifiable dans l’apparence des images |
| ADR-42 | AdamW avec decay séparé pour poids, biais et normalisation | Régulariser les poids sans appliquer automatiquement la même pénalisation à tous les paramètres | Les valeurs retenues sont initiales, pas déjà optimales |
| ADR-43 | Pondérer la perte selon les seuls effectifs d’apprentissage | Traiter le déséquilibre sans utiliser les classes du test | La pondération peut affecter la calibration ; elle ne supprime pas les biais de centre |
| ADR-44 | Petits lots avec accumulation normalisée par le nombre réel d’images | Limiter la mémoire GPU et traiter correctement les lots incomplets | L’efficacité réelle doit être mesurée sur le GPU disponible |
| ADR-45 | Précision mixte adaptée au GPU, limitation des gradients et surveillance des valeurs non finies | Améliorer l’utilisation des ressources et contrôler les débordements | Le chemin CUDA doit être validé par un véritable calcul d’apprentissage |
| ADR-46 | Première comparaison sans augmentation, deux vitesses d’apprentissage et configurations communes | Obtenir une référence lisible avant une recherche plus large | Une comparaison commune n’assure pas que chaque architecture dispose de ses paramètres optimaux |
| ADR-47 | Sélection et arrêt pilotés par la ROC-AUC de validation | Utiliser une mesure de classement adaptée à une tâche déséquilibrée | Les essais répétés peuvent suradapter les choix à cette validation |
| ADR-48 | Seuil initial 0,5 et métriques complémentaires globales/par centre | Ne pas réduire l’évaluation à l’exactitude globale | Calibration, intervalles d’incertitude et robustesse restent à étudier |
| ADR-49 | Sauvegardes locales complètes et objectif de suivi MLflow distant | Permettre la reprise et conserver les preuves d’expérience | Le suivi distant et la reprise GPU doivent encore être vérifiés |
| ADR-50 | Déterminisme demandé avec avertissements autorisés | Favoriser la reproductibilité malgré les limites de certaines opérations CUDA | Ce mode ne garantit pas une identité bit à bit |

### 15.7. Versionnement et documentation

| Référence | Décision | Justification | Conséquence ou limite |
| --- | --- | --- | --- |
| ADR-51 | Exclure de Git secrets, IRM, tableaux numériques, archives, poids et résultats locaux | Maintenir un dépôt de code sans données sensibles ou volumineuses | Le `.gitignore` n’empêche pas un ajout forcé ; la sélection doit être contrôlée |
| ADR-52 | Premier commit limité aux fichiers renseignés et aux initialisateurs de packages sélectionnés | Publier une base identifiable sans présenter des fichiers vides comme des services réalisés | Le squelette restant demeure local jusqu’à son implémentation |
| ADR-53 | Rédiger le README technique dès maintenant et le mettre à jour | Conserver les décisions, commandes et limites pendant le développement | La documentation doit être relue à chaque évolution du comportement |
| ADR-54 | Séparer documentation complète et présentation vulgarisée | Servir les besoins techniques et ceux d’un lecteur non initié | Les deux versions devront rester cohérentes |

Les décisions nouvelles seront ajoutées avec leur motivation, leurs
alternatives pertinentes, leurs effets sur les données ou les résultats,
et leur état de vérification. Une modification d’un choix existant devra
conserver la trace du choix précédent.


## Livraison compacte du 6 octobre 2026 — décisions complémentaires

| Décision | Motif | Effet et vérification |
| --- | --- | --- |
| Remplacer les fichiers vides par dix modules fonctionnels dont un export, et un initialisateur | Simplifier sans présenter du squelette comme un service réalisé | Ancien code archivé ; syntaxe et tests avant remplacement |
| Fusionner lecteur, modèles, métriques et moteur dans `learning.py` | Garder une définition unique de l'apprentissage | Calcul d'une époque et sérialisation testés dans Docker |
| Conserver le protocole v2 et ses patients | Éviter une modification expérimentale implicite pendant la restructuration | Rôles et effectifs dans README ; étude Pittsburgh séparée |
| Contrôler la compatibilité avant réutilisation des NPZ | La nouvelle pile numérique peut changer la préparation | Cinq échantillons train, code/recette liés au paquet et modèle |
| Refuser géométries invalides et volumes natifs >100 millions de voxels | Borner les entrées non fiables | Garde-fous techniques, pas validation médicale |
| Lots atomiques avec READY, copie brute et accusé | Détecter une publication complète et conserver les sources | Copie vérifiée avant transaction ; répétition idempotente |
| File de tâches PostgreSQL et worker unique | Ne pas charger MONAI dans Airflow ni transmettre les volumes dans XCom | Trois essais, lease récupérable ; volumes privés séparés |
| Deux DAGs toutes les cinq minutes | Automatiser arrivées et supervision sur un besoin de démonstration | Freshness à mesurer, seuil d'alerte choisi vingt minutes |
| Bases séparées pour prédictions, tests, Airflow ; MLflow distant | Séparer stockage métier et métadonnées | Utilisateurs/DSN distincts, port SQL non exposé |
| Modèle absent tant qu'aucun entraînement réel n'est importé | Éviter toute simulation de résultats présentée comme un champion | `/health` disponible ; `/ready` reste 503 |
| Cache de deux modèles, lots de quatre et concurrence bornée | Maîtriser la mémoire CPU et le service | Tests de charge et p95 à mesurer sur WSL |
| Canary 10 % stable par image, promotion conditionnée et rollback | Faire un déploiement progressif réel | Les alias du registre sont synchronisés explicitement |
| Gradient×entrée | Fournir une explicabilité intégrée peu coûteuse | Sensibilité locale non causale, contrôlée sur calcul fictif |
| Dérive descriptive sur quatre caractéristiques et sujets distincts | Surveiller les IRM sans afficher leurs données brutes | Ne remplace pas les métriques supervisées ; labels réels nécessaires |
| Aucun apprentissage sur Brown ou les patients déjà réservés | Empêcher labels inventés et fuite des tests | `new_training` et snapshot protègent les groupes initiaux |
| Réentraînement automatique désactivé initialement | Les comptes, quotas, protection faciale et nouveaux labels ne sont pas encore vérifiés | Chaîne de soumission écrite ; activation après démonstration des prérequis |
| GitHub pour le code/CI ; DagsHub pour MLflow et même historique | Répondre au libellé du référentiel et au coaching | Accès du jury et runner privé à organiser |
| Airflow standalone lié au loopback | Limiter les services pour la démonstration locale | Ne pas confondre avec une installation publique de production |
| Une commande d'installation avec tests avant remplacement | Éviter les substitutions module par module et garder une issue de reprise | Ancien dossier conservé ; échec de démarrage post-remplacement explicite |

Les cinquante-quatre décisions précédentes sont conservées comme historique. Une affirmation de visibilité privée du dépôt dans cet historique doit être vérifiée dans les paramètres du compte : sa présence dans le texte n'en est pas une preuve. Les paramètres d'optimisation restent des hypothèses, pas des résultats. Le dossier de formation applicable et l'accord propre au sujet doivent être conservés comme preuves indépendantes ; les échanges attribués au formateur dans le prompt de passation sont des indications rapportées, pas une confirmation obtenue par ce logiciel.

## Suivi de la campagne initiale et politique MLflow YAML — 6 octobre 2026

Quatre essais Kaggle, graine 42, sont conservés avec leurs poids, reprises, profils et releases. La copie `models/experiments/initial_campaign_*/code` garde le code et les configurations utilisés avant cette correction. Les originaux et la répartition ne sont pas modifiés.

Les métriques globales/par centre, artefacts et échantillonnage des époques sont désormais sélectionnés par `configs/training.yaml`, section `tracking.mlflow`. Publication en fin d’essai : `log_every_epochs` règle les points conservés dans les courbes, pas un envoi en temps réel. Une métrique indéfinie reste absente de MLflow et conserve sa valeur null dans le rapport JSON.

L’export utilise le format pickle explicitement lorsque la version de MLflow offre ce réglage, embarque le code du modèle, vérifie son rechargement puis enregistre l’artefact du run. Les futurs kernels installent MLflow/PyYAML sans version imposée ; les environnements effectifs restent enregistrés. L’image CPU locale conserve ses dépendances testées.

Cette correction de suivi ne demande aucun réentraînement. Aucun champion n’est promu ; Evidently et les vidéos ne sont pas déclarés réalisés. La revue faciale est une précaution du projet, pas une exigence explicite identifiée du jury ni une certification d’anonymisation.

## Décision : arrêt de la campagne initiale après quatre essais

Deux architectures et deux taux d’apprentissage ont été comparés avec
la graine 42. Nous ne réalisons pas de répétitions multi-graines à ce stade :
la priorité est la validation du pipeline complet et les preuves de certification.

La validation comprend 119 patients, avec de faibles effectifs dans certains
centres. Changer la graine ne corrigerait ni cette limitation ni les déséquilibres
entre centres. Des répétitions permettraient néanmoins d’étudier la variabilité
de l’apprentissage ; celle-ci reste non mesurée.

ResNet3D à 0,001, version MLflow 3, est retenu pour la démonstration.
ResNet3D à 0,0003, version 4, est retenu comme challenger actif potentiel.
SimpleCNN3D à 0,0003 reste une référence comparée, exclue de la mise en
service car ses prédictions sont toutes « témoin » au seuil 0,5.

Les seuils d’éligibilité sont des choix du projet, pas des exigences du jury.
Le refus des prédictions constantes est ajouté après analyse de la campagne :
il ne s’agit pas d’une règle expérimentale annoncée avant les entraînements.

Aucune supériorité statistique ni validation clinique n’est revendiquée.
Le seuil reste fixé à 0,5. Les choix sont gelés avant l’évaluation finale
sur les tests réservés ; leurs résultats ne serviront pas à régler le modèle.

## Vocation architecturale et portée scientifique

Ce projet vise principalement à démontrer, pour la certification des blocs
3 et 4, une architecture MLOps fonctionnelle et traçable : préparation des
données, comparaison de modèles, versionnement, déploiement, orchestration,
surveillance et reprise après incident. Ces capacités doivent être étayées
par des preuves d’exécution ; leur présence dans le code ne suffit pas.

Sur le plan scientifique, répéter les entraînements avec plusieurs graines
permettrait de mesurer la variabilité liée à l’initialisation et aux tirages
aléatoires. Cette démarche conserve son intérêt, même avec un petit
échantillon ; elle ne compense toutefois ni le faible nombre de patients
de validation ni les déséquilibres entre centres.

Nous avons arrêté la campagne à quatre essais avec la graine 42, afin de
consacrer les ressources restantes à la vérification de l’architecture
complète. Nous avons renoncé aux répétitions multi-graines à ce stade,
pas à l’utilisation d’une graine fixée pour la reproductibilité.

Ce choix de périmètre laisse la variabilité de l’apprentissage non mesurée.
Les résultats constituent une comparaison exploratoire pour sélectionner
un modèle de démonstration, sans établir une supériorité statistique,
une robustesse clinique ou une aptitude au diagnostic.

## Déploiement piloté par les alias du registre

Les alias DagsHub/MLflow `champion` et `challenger` désignent les versions
souhaitées. Le déploiement les lit, télécharge les paquets `deployment`,
contrôle manifeste, poids, recette et éligibilité, puis publie atomiquement
l’état local utilisé par l’API. Aucun téléchargement n’a lieu par prédiction.

La copie locale permet de continuer l’inférence si le registre est indisponible.
Le contrôle planifié distingue `aligned`, `mismatch` et `registry_unavailable`.
Les deux derniers états produisent des alertes ; ils ne remplacent pas le modèle.
Un changement d’alias doit être suivi de la commande de déploiement : il n’est
pas pris en compte instantanément par l’API.

`catalog_challenger` conserve le candidat du registre. `challenger` et `fraction`
décrivent seulement le trafic canary actif. Un candidat peut donc être présent
avec une fraction de zéro. Une promotion normale conserve les barrières de
validation et de canary, puis actualise le registre avant de déployer.

Commandes :
- `python3 deploy.py cli` reste disponible pour les autres fonctions.
- `docker compose run --rm --no-deps worker python -m adhd.registry_deployment deploy`
- `docker compose run --rm --no-deps worker python -m adhd.registry_deployment check`
- `docker compose run --rm --no-deps worker python -m adhd.registry_deployment restore-exercise`

L’exercice temporaire version 4 puis version 3 est déclaré technique. Il ne
constitue ni une promotion scientifique ni un réglage après lecture des tests.
Il vérifie deux inférences et la restauration de l’alias et du modèle servi.
Le jeton local demeure hors du dépôt, sans affichage dans les journaux.
