# Exploitation, sécurité et démonstration

Cette livraison remplace le squelette de code après ses précontrôles. Elle conserve l'ancien projet dans `~/ADHD200_PIPELINE_ARCHIVE_<date>` et le même dossier de données. Aucun secret, jeu de données, poids ou historique Git n'est intégré à la livraison. L'installateur conserve la copie Git locale, sélectionne le nouveau code et ne fait ni commit ni push. L'ancienne sélection Git reste dans l'archive.

## Installation

Prérequis : WSL avec Python 3, Docker Desktop intégré à WSL, Docker Compose v2, accès aux registres et paquets pendant la construction. Prévoir environ 12 Gio de RAM accessibles à Docker, et de l'espace pour images, nouvelles sorties et copies de démonstration. Les limites Compose sont des plafonds par conteneur, pas des mesures de consommation.

Extraire le ZIP dans un dossier distinct, puis `python3 deploy.py install`. La première construction peut durer plusieurs dizaines de minutes. Syntaxe, configuration Compose, tests CPU/PostgreSQL/API puis import des deux DAGs sont contrôlés avant le remplacement. Un échec avant ce remplacement conserve le projet original ; PostgreSQL de test et les secrets persistants nouvellement créés peuvent rester présents pour la reprise. Une panne après le remplacement exige de consulter les journaux ; l'installateur ne prétend pas que le démarrage des services est atomique.

Depuis le projet installé :

```bash
python3 deploy.py status
python3 deploy.py test
docker compose logs --tail=100 api worker airflow
python3 deploy.py airflow-password
```

Les interfaces sont exclusivement liées au poste : API `localhost:8000/docs`, dashboard `localhost:8501`, Airflow `localhost:8080`. Airflow emploie son mode standalone et SimpleAuthManager : adapté à cette démonstration locale, pas un déploiement public de production.

## Secrets

Les secrets sont générés une fois dans `ADHD200_DATA/secrets/compact_runtime/` (répertoire 700, fichiers 600) et raccordés par `.runtime`, exclu de Git. La clé HMAC historique reste inchangée dans `ADHD200_DATA/secrets/pseudonymization.key`. Ni cette clé ni les identifiants patients ne sont utilisés comme entrées du CNN.

Le token DagsHub se renseigne dans le fichier privé `mlflow_password` du dossier persistant, puis `python3 deploy.py prepare --data "$HOME/ADHD200_DATA"` et `docker compose up -d --force-recreate api worker` actualisent les raccordements. Aucun token ne doit être collé dans le README, un notebook publié ou une commande enregistrée. Pour Kaggle : fichiers privés `kaggle_username`, `kaggle_key`, et variables non secrètes `KAGGLE_DATASET`, `KAGGLE_KERNEL` dans `.env`. Adapter la licence de réutilisation au dataset ; `other` n'est pas une autorisation de transfert.

Les deux clés HTTP ont des droits séparés : lecture/prédiction et administration. Elles ne sont pas affichées au démarrage. Pour la documentation interactive de l'API, fournir l'en-tête `X-API-Key` à chaque appel. Le dashboard et les DAGs les lisent depuis leurs montages de secrets. PostgreSQL n'expose aucun port à l'hôte ; ses bases `predictions`, `predictions_test`, `orchestration` sont distinctes du backend MLflow géré par DagsHub.

HTTP est limité au loopback et au réseau Compose pour cette démo. Les transferts Kaggle/DagsHub sont HTTPS. Un accès distant aux interfaces nécessiterait TLS, contrôle d'accès renforcé et revue du déploiement. La pseudonymisation et le format NPZ ne retirent pas le visage. La revue préalable de protection, de droits de réutilisation, de destination privée et de durée de conservation bloque les transferts automatiques tant qu'elle est incomplète.

## Vérifier la compatibilité avant Kaggle

La pile compacte utilise des versions différentes du premier Docker. Avant de réutiliser l'export, recalculer une image d'entraînement par centre et comparer les tableaux :

```bash
python3 deploy.py cli compatibility \
  /data/export_kaggle/kaggle_20261005T203927Z_b2f6a09d \
  /data/reports/private/split_private_20261005T191224Z_fc7dc57c.json \
  /data/reports/private/kaggle_export_20261005T203927Z_b2f6a09d.json
```

Cette commande préserve les NPZ mais enrichit `dataset_info.json` et produit un rapport dans `reference/compact`. Cinq échantillons ne prouvent pas l'identité des 682 transformations ; ils détectent un écart manifeste. L'empreinte du code de préparation et la recette sont ensuite exigées par l'entraînement et liées à la version déployée.

Si le contrôle échoue, recréer les volumes avec `adhd.historical`, puis leur export avec `adhd.export` ; les anciennes sorties sont conservées. Ne pas contourner le contrôle en éditant la preuve. Exemple de retraitement :

```bash
docker compose run --rm worker python -m adhd.historical \
  --raw-root /data/raw \
  --split-file /data/reports/private/split_private_20261005T191224Z_fc7dc57c.json \
  --config-file /app/configs/pipeline.yaml \
  --output-root /data/historical/processed/compact \
  --report-dir /data/reports/compact/private \
  --splits train validation
```

Le rapport imprimé donne l'entrée d'`adhd.export --preparation-report ... --processed-root /data/historical/processed/compact --export-root /data/export_kaggle --private-report-dir /data/reports/compact/private`. Adapter les chemins au nouveau rapport et paquet, puis refaire la compatibilité avec son rapport privé d'export. Aucun groupe de test ne doit entrer dans l'export d'entraînement.

## Entraînement Kaggle et retour du modèle

Ne pas remplacer le PyTorch GPU Kaggle par celui de l'image Docker CPU. Générer le script autonome :

```bash
python3 deploy.py cli kernel /data/reference/compact/kaggle \
  --id COMPTE/adhd200-training --dataset COMPTE/adhd200-private
```

Avant un transfert, créer une revue privée liée au SHA-256 du CSV : champs `csv_sha256`, `face_protection_verified`, `reuse_reviewed`, `private_destination_verified`, `access_retention_defined`, `reviewer`, `evidence`. Les booléens doivent correspondre à une vérification réelle, pas être cochés pour débloquer une commande. Le code ne réalise pas de defacing.

Configurer dans Kaggle un dataset **privé**, un notebook **privé**, un GPU et un secret `DAGSHUB_TOKEN` accessible au notebook. Importer le script `kernel.py`. Il contrôle un calcul avant/arrière GPU puis lance quatre essais comparables. Les résultats sont dans `/kaggle/working/runs` ; les modèles sont enregistrés sous le même nom `ADHD200_ANATOMICAL` dans MLflow. Récupérer le dossier `release` d'un essai dans `ADHD200_DATA/models/imported/<essai>/`, puis :

```bash
python3 deploy.py cli import-model /data/models/imported/ESSAI/release
```

Cette étape vérifie les poids, la recette et un calcul CPU. Elle retourne une version immuable. Via `/deployment` avec la clé administrateur, demander `candidate` et cette version. Le premier candidat admissible devient champion ; le suivant reçoit 10 % du routage. Aucun score clinique minimum n'est garanti. Un modèle qui ne passe pas les seuils reste une expérience.

Les quatre essais ne suffisent pas à annoncer une optimisation exhaustive. La campagne reste arrêtée à quatre essais, graine 42. Les répétitions multi-graines et les autres recherches de paramètres ne sont pas prévues pour clôturer la certification. Elles auraient un intérêt scientifique pour quantifier la variabilité ; celle-ci n’est pas mesurée ici. Aucun choix n’est révisé à partir des tests finaux. Le moteur `python -m adhd.train --help` permet les variations de taux/graine et la reprise via `last.pt`. Les derniers poids, optimiseur, scheduler, scaler et RNG sont sauvegardés après chaque époque complète. La reprise GPU réelle reste à démontrer.

## Arrivées et incidents

Après installation d'un modèle, déposer trois copies Brown, avec un fichier corrompu supplémentaire :

```bash
python3 deploy.py cli simulate \
  /data/reports/private/split_private_20261005T191224Z_fc7dc57c.json \
  --size 3 --corrupt
```

Attendre le DAG planifié toutes les cinq minutes. Le manifeste et `READY` ne deviennent visibles qu'après publication du dossier complet. La collecte copie les sources dans `arrivals/raw`, enregistre un accusé, puis prépare les valides, isole la corruption, prédit et stocke. Relancer ne crée pas de doublons. Les sources et rejets restent disponibles.

Scénarios à filmer : panne temporaire (`docker compose stop api`, puis `docker compose start api`), tâche dépendante bloquée, reprise ; corruption avec valides ; double collecte ; arrêt du worker et récupération de lease ; modification d'une recette et nouvelle signature ; dépassement de délai et alerte dans le dashboard. Les tests unitaires ne remplacent pas ces exécutions visibles.

## Modèles, labels et surveillance

`/labels` reçoit une cible réelle et sa source. Les fichiers synthétiques sont exclus. Brown peut recevoir une cible d'évaluation uniquement si elle devient réellement disponible. `new_training` refuse Brown, les centres réservés et tous les patients du protocole initial. Une cohorte `evaluation_only` peut démontrer des labels différés après gel des modèles ; elle ne devient jamais une donnée de réentraînement.

Une fenêtre compte les sujets distincts, pas les requêtes répétées. La dérive compare moyenne/std/p90/occupation à la référence train, minimum 20 sujets. Les métriques supervisées exigent cinq sujets de chaque classe avec vraie cible. Ces seuils sont des repères de démonstration ; l'âge et le centre peuvent expliquer les écarts. Une baisse de ROC-AUC de 0,10 alerte, elle ne diagnostique pas une cause.

Une dérive et vingt véritables nouveaux sujets étiquetés admissibles produisent un snapshot. `automatic_retraining` peut ensuite soumettre un dataset versionné et le kernel Kaggle, après revue de transfert. La soumission peut aussi passer par le workflow GitHub dédié. Le passage réel avec accès distant et quotas reste à contrôler. Aucun réentraînement supervisé n'est possible à partir des seuls 26 Brown sans diagnostics.

La promotion requiert validation suffisante, au moins dix requêtes canary, erreurs ≤5 %, p95 ≤10 s et ROC-AUC validation au moins égale au champion. Ce sont des critères opérationnels/expérimentaux, pas une validation médicale. `/deployment rollback` retire un canary actif. Dans le fonctionnement piloté par le registre, restaurer le champion exige de restaurer son alias DagsHub, puis de lancer `python -m adhd.registry_deployment deploy`. Les alias font foi ; le modèle local constitue la copie servie et vérifiée. `registry-deploy` (et son nom de compatibilité `sync`) lit les alias pour déployer ; il ne les réécrit pas à partir d’un état local indépendant. L’exercice `restore-exercise` a confirmé le service de la version 4 puis le retour à la version 3.

## Reprise, sauvegarde et coûts

Sauvegarder PostgreSQL avec `pg_dump` sans afficher ses secrets, les volumes de données, les modèles immuables et les secrets séparément. Tester une restauration dans une base isolée avant l'oral. Ne pas lancer `docker compose down -v` sur le projet actif : les volumes contiennent les preuves. Un retour à l'ancien code utilise son dossier archivé ; il ne garantit pas à lui seul la compatibilité de la base.

Les ressources par tâche sont enregistrées (durée et pic du processus), celles des entraînements par époque (CUDA allouée/réservée). La mémoire RSS est un maximum du processus, pas une mesure exacte par tâche. Collecter aussi `docker stats --no-stream` et les tailles de `ADHD200_DATA`. Répéter des requêtes sert à la mesure de charge et n'ajoute aucun patient indépendant.

Hypothèse énergétique GPU : 100 W × durée, fichier `resources.json`, **estimation**, pas mesure électrique. Coût énergétique = kWh × tarif réel ; utiliser 0,25 €/kWh uniquement comme scénario annoncé. Coût monétaire GPU dépend du compte/quota Kaggle ; un quota gratuit n'est pas une ressource illimitée. Transfert initial : 2,66 Gio, auquel s'ajoutent sauvegardes et modèles. Mettre les prix/quota observés et la date dans le rapport final ; ne pas inventer un tarif contractuel.
