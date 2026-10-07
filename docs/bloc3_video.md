# Vidéo bloc 3 — préparation, pas encore enregistrement final

Ce document est une proposition de déroulé. Le référentiel demande une vidéo ; la durée indicative ci-dessous n'est pas une exigence du jury. L'oral de cinq minutes est distinct de la vidéo. Le scénario final sera confirmé après clôture de la matrice B3.

## Préparation

- Fermer les terminaux qui montrent des tokens, mots de passe, identifiants bruts ou correspondances privées.
- Ouvrir le schéma, Airflow http://localhost:8080 (connexion hors enregistrement), Streamlit http://localhost:8501 et un terminal WSL.
- Agrandir la police et utiliser des noms de scènes lisibles. Masquer toute notification personnelle.
- Identifier les rapports de preuve et les exécutions Airflow existantes. Ne pas supprimer de lignes de base pour fabriquer une démonstration.
- La réserve Brown ayant été déposée, ne pas relancer simulate en supposant qu'il reste des patients. Préparer un scénario de répétition explicite avant tournage.

## Scènes proposées — environ six à huit minutes, à adapter

| Scène | Écran et action | Commentaire à dire | Preuve |
| --- | --- | --- | --- |
| 1 — Contexte et schéma | Support : entrée, stockage brut, préparation/quarantaine, API, PostgreSQL, Streamlit ; Airflow orchestre | « Il s'agit d'arrivées simulées d'IRM Brown réelles, utilisées pour une démonstration technique, sans diagnostic connu. Le traitement par lots convient à ces fichiers lourds. » | Diagramme et décision batch |
| 2 — Déclenchement planifié | Airflow, DAG adhd_arrivals, sélectionner un run de type scheduled ; afficher la date et le graphe | « Airflow déclenche le traitement toutes les cinq minutes. Chaque tâche attend le succès de la précédente. » | Run scheduled et chaîne collect/prepare/predict |
| 3 — Qualité et résultats | Streamlit : lot initial, trois predicted, un quarantined, version et scores | « Le fichier artificiellement corrompu est isolé ; les autres IRM continuent. Ces scores ne sont pas des diagnostics. » | SQL/dashboard et rapport premier lot |
| 4 — Panne et reprise | Airflow, run b3_retry_20261006T235837Z_c1da532d ; tâche collect et journaux des deux essais ; rapport JSON | « Cet exercice a été déclenché une fois. L'API était arrêtée. Airflow a relancé automatiquement collect, puis poursuivi les dépendances. » | up_for_retry, tentative 2, même run successful |
| 5 — Doublon | Reçu du nouveau lot c90b5e45e1e70f5f14b2005159524eb3e8212a8053f1dc3ecaf84e9fab3a7aea et rapport b3_duplicate | « Une nouvelle livraison contenait les mêmes octets. L'empreinte a évité un second élément et une nouvelle prédiction. » | Zéro nouvel élément/prédiction, état initial conservé |
| 6 — Traçabilité, protection et limites | Code/README, table de preuves ; sans afficher de fichiers de secrets | « Les références permettent de relier source, préparation, version et résultat. Le code est séparé des données et des secrets. La déduplication binaire ne détecte pas toute reconversion. » | Lignage et sélection Git contrôlée |

Utiliser des preuves existantes est autorisé comme démonstration du fonctionnement observé ; annoncer clairement toute séquence enregistrée à un autre moment. Ne pas présenter une lecture de rapport comme une nouvelle exécution en direct. Une séquence attendue plusieurs minutes peut être accélérée avec l'indication « attente accélérée ».

Les scènes retraitement, reproductibilité, alertes et coûts seront intégrées ou liées au support après vérification. La vidéo ne remplace ni le code GitHub ni le schéma ni le support.
