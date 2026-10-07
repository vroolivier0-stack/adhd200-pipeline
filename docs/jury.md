# Couverture des blocs 3 et 4 et preuves à présenter

Sources examinées : référentiel « Architecte en intelligence artificielle » (bloc 3 pages 10–14, bloc 4 pages 15–20), énoncés Automatic Fraud Detection et Final Project, captures de plateforme, trois TXT COACHING, schémas pédagogiques et prompt/avis de Claude fournis par Olivier. Ordre d'autorité : référentiel, modalités et énoncés, formateur, exemples, avis d'assistants.

La colonne « livraison » désigne du code/documentation présent. La colonne « preuve » désigne une exécution ou un élément à obtenir. Elle n'accorde aucune note et ne prétend pas remplacer l'analyse du jury. Le contrat de formation et l'accord individuel du sujet restent des pièces à conserver. Le README ne peut pas certifier leur existence.

## Bloc 3 — 21 indicateurs

| Repère | Attente résumée | Livraison | Preuve à recueillir |
| --- | --- | --- | --- |
| 1.1 | Justifier batch/streaming et fraîcheur | README, lots de cinq minutes, alerte vingt minutes | Temps dépôt→résultat et justification de l'usage recherche |
| 1.2 | Collecte adaptée sans perte | Collecte, READY, copie brute, hash et transaction | Lot compté, accusé, panne/reprise, aucun doublon |
| 1.3 | Diagramme des flux et stockages | Schéma README et contrats operations | Captures des services réellement actifs |
| 2.1 | Transformations modulaires | data, historical, flow | Même préparation appelée dans les deux trajets |
| 2.2 | Adaptation au modèle | Contrat 128³, recipe/code/model | Compatibilité puis calcul CPU/GPU |
| 2.3 | Retraiter l'historique | Signatures code/config/versions et reçus | Nouvelle sortie après modification, ancienne conservée |
| 3.1 | Orchestrateur planifié | DAG arrivals et monitoring | Déclenchement programmé filmé, sans lancer chaque tâche |
| 3.2 | Dépendances sur réussite | Chaîne collect→prepare→predict | Échec provoqué, étape suivante bloquée puis reprise |
| 3.3 | Autre environnement | Docker, versions principales, freeze résolu | Build/CI sur une seconde machine ; versions/digests conservés |
| 4.1 | Validation des types/formats | Hashes, NIfTI/NPZ, API et contraintes SQL | Tests Docker + fichiers invalides |
| 4.2 | Corruption isolée | Quarantaine par image | Lot mêlant valide et corrompu, valide arrivé en base |
| 4.3 | Reprises réseau | HTTP borné, Airflow retries, jobs persistants | Panne API puis reprise sans duplication |
| 5.1 | Masquage des données | Pseudonymes historiques, références opaques | Absence d'IDs bruts dans UI/logs ; revue du visage |
| 5.2 | Identifiants externalisés | Secrets privés, deux clés API, DSN distincts | Audit Git et journaux sans tokens |
| 5.3 | Transferts sécurisés | Local loopback ; HTTPS distant ; gate | Destination privée et autorisation de transfert documentées |
| 6.1 | Alertes panne/délai | Events persistants, supervision et logs Airflow | Retard/échec visible dans dashboard ; panne indépendante examinée |
| 6.2 | Coûts calcul/transfert | Ressources et scénarios operations | Mesures WSL/GPU, quotas et hypothèses tarifaires datées |
| 6.3 | Journaux exploitables | JSONL, SQL events, logs Airflow | Retrouver une tâche échouée et sa référence |
| 7.1 | Traçabilité des données | Source→reçu→volume→version→prédiction | Reconstituer le trajet d'une IRM sans exposer son identité |
| 7.2 | Backlog et priorisation | Itérations et critères ci-dessous | Statuts datés et liens vers preuves GitHub |
| 7.3 | Documentation de reprise | README, opérations et décisions | Installation par une autre personne, restauration testée |

## Bloc 4 — 20 indicateurs

| Repère | Attente résumée | Livraison | Preuve à recueillir |
| --- | --- | --- | --- |
| 1.1 | Entraînement/validation automatisés | Campagne quatre essais + kernel + chaîne CT | Exécution Kaggle et soumission traçable |
| 1.2 | Versions/restauration modèle et données | Checkpoints, profiles, releases, MLflow, hashes | Reprise GPU et restauration d'une version avec ses données |
| 1.3 | Tests avant mise à jour | CI et barrières de candidat/promotion | Échec de test ou mauvais candidat effectivement refusé |
| 2.1 | Mode d'inférence adapté | API unitaire et lot, worker CPU | Justification volume/latence du besoin |
| 2.2 | Disponibilité sous charge | Cache, limites, santé et redémarrage | Benchmark et panne/reprise sur WSL |
| 2.3 | Optimisation d'exécution | Modèle en mémoire, inférence sans gradients, lots | Comparaison latence/ressources avant/après réglage |
| 3.1 | Déploiement progressif | Canary réel 10 %, promotion, rollback | Requêtes sur deux versions et retour arrière |
| 3.2 | Dérive/baisse de performance | Caractéristiques train, vraies cibles, alertes | Fenêtres distinctes, effectifs et rapports |
| 3.3 | Déclenchement réentraînement | Snapshot admissible + gate + Kaggle/Actions | Conditions remplies et version de dataset/kernel soumise |
| 4.1 | Ressources suivies | Temps, RSS, CUDA allouée/réservée | Tableau d'essais et docker stats |
| 4.2 | Efficience énergétique | Modèles compacts, AMP/arrêt/reprise, estimation | Compromis temps/mémoire/score ; énergie annoncée estimée |
| 5.1 | Explicabilité | Gradient×entrée dans API et dashboard | Carte sur une prédiction réelle et limites expliquées |
| 5.2 | Garde-fous | Auth, validation, bornes, labels protégés | Refus des entrées/cibles/versions invalides |
| 5.3 | Transparence/sécurité/réglementaire | Usage recherche, limites et protection | Analyse liée à l'usage réel ; aucun certificat juridique inventé |
| 6.1 | Choix justifiés au cahier des charges | Besoin README et registre de décisions | Argumenter les alternatives et les critères retenus |
| 6.2 | Cohérence métier/infrastructure/sécurité | Local CPU/Kaggle GPU/MLflow séparés | Contrats et flux effectivement observés |
| 7.1 | Architecture transmissible | Schéma, installation, incidents, rollback | Déploiement/restauration par un tiers |
| 7.2 | Model cards | Fiche générée par essai | Métriques/versions/usages interdits renseignés réellement |
| 8.1 | Veille analysée/testée | Sources et hypothèses ci-dessous | Résultats de comparaison et date de lecture |
| 8.2 | Recommandations de veille | Choix conditionnés aux mesures | Recommandation motivée après essais, pas promesse |

## Itérations et critères d'acceptation

| Priorité | Itération | Critère de clôture |
| --- | --- | --- |
| P0 | Déploiement et contrat | Tests Docker+DAGs réussis, interfaces actives, export compatible, aucune donnée brute modifiée |
| P0 | Protection/transfert et code | Revue de protection réelle ; dépôt GitHub privé accessible au jury ; premier commit vérifié |
| P1 | Kaggle et référence | Deux architectures, deux taux, environnement et résultats MLflow enregistrés ; modèle importé |
| P1 | Démonstration bloc 3 | Planification, corruption, panne/reprise, SQL/UI et vidéo complète |
| P2 | Cycle modèle bloc 4 | Canary, refus/promotion, rollback, benchmark, explication et model cards |
| P2 | Surveillance/CT | Fenêtres suffisantes, vrais labels ; soumission démontrée ou limites documentées honnêtement |
| P2 | Évaluation gelée | Confirmations, test interne/externe/contrôles, incertitude et biais discutés |
| P2 | Dossier final | Preuves datées, coûts, veille, sauvegarde/restauration et supports oraux |

L'implémentation est livrée ensemble ; ces itérations concernent l'exécution, la vérification et les réglages. Remplir un fichier n'est pas un critère de clôture.

## Vidéos et oral

Bloc 3 : cinq minutes de présentation puis quinze de questions ; filmer le trajet d'un petit lot, un rejet et la reprise. Montrer schéma, DAG vert, lignes SQL et dashboard. Dire explicitement « arrivées Brown simulées, diagnostics inconnus ».

Bloc 4 : cinq minutes puis dix de questions ; montrer comparaison MLflow, version servie, canary et rollback, CI, monitoring et une explication. Exposer les limites de labels et de quotas au lieu d'inventer un réentraînement réussi. Préparer une vidéo même si une démonstration en direct est prévue.

## Veille et décisions d'expérimentation

Sources primaires à conserver avec date de consultation (6 octobre 2026 pour la livraison) :

- [Airflow 3.3.2 : authentification simple](https://airflow.apache.org/docs/apache-airflow/3.3.2/core-concepts/auth-manager/simple/index.html) : développement/test ; décision : loopback, pas exposition publique.
- [MONAI 1.5.2 publié](https://pypi.org/project/monai/1.5.2/) et [documentation](https://docs.monai.io/) : pile fixée pour la construction ; décision : contrôler les différences avec l'ancien prétraitement.
- [PyTorch : précision mixte](https://docs.pytorch.org/docs/stable/amp.html) : hypothèse de réduction mémoire/temps ; décider après mesure CUDA et stabilité.
- [MLflow : registre](https://mlflow.org/docs/latest/ml/model-registry/) : nom commun et aliases ; décision : conserver aussi un routage réel dans l'API.
- [GitHub Actions](https://docs.github.com/en/actions) : tests sur push/PR, déploiement sur runner privé ; ne jamais exécuter un PR non fiable sur le runner WSL avec données/secrets.

Comparer initialement des modèles compacts, pas une grande ResNet par défaut. Étudier ensuite augmentation, capacité, régularisation, précision et taille de lot. Ne pas annoncer AMP, GroupNorm ou AdamW comme automatiquement supérieurs. La recommandation finale devra relier score, variance, biais, temps et mémoire mesurés.
