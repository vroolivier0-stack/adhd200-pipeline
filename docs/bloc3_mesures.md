# Bloc 3 — mesures et contrôles complémentaires

Rapport privé : `b3_final_checks_20261007T001922Z_16ffc4cb.json`.

Retraitement démontré sur une image train, réutilisation sans recalcul, nouvelle recette isolée avec ancienne sortie conservée, valeurs identiques dans deux conteneurs neufs de même image sur WSL. Pas de preuve d’exécution sur une deuxième machine ; GitHub Actions reste à exécuter.

Une alerte de délai a été créée par le vrai watchdog à partir d’un lot artificiellement ancien, puis vérifiée en base et dans la réponse fournie au dashboard. Aucun délai réel de vingt minutes n’est revendiqué. Le lot synthétique a été clôturé et l’événement de preuve conservé.

| Type de tâche | Exécutions enregistrées | Somme des durées écoulées (s) | Maximum du pic RSS du processus (Mio) |
| --- | ---: | ---: | ---: |
| collect | 88 | 3.20 | 1226.28 |
| prepare | 88 | 59.76 | 1226.28 |
| predict | 88 | 5.28 | 1226.28 |

Les exécutions comptées incluent les passages sans nouveau travail. Les durées ne sont ni une facture ni une mesure CPU. Les pics RSS appartiennent au processus worker, pas exclusivement à chaque tâche. Le détail et un instantané docker stats sont conservés dans le rapport privé.

Pour l’analyse de coût : infrastructure WSL existante ; stockage local et électricité restent à estimer selon le matériel et le tarif réels. Le paquet initial Kaggle faisait environ 2,66 Gio ; cela ne mesure pas tous les octets réellement transférés. La disponibilité d’un quota gratuit n’est pas une garantie de ressources illimitées. Renseigner les dépenses constatées séparément des scénarios d’estimation, sans tarifs inventés.
