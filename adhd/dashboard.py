"""Tableau de bord : suivre le trajet des IRM et la vie du modèle.

Il lit uniquement l'API (/summary). Il n'affiche ni numéro de dossier ni IRM
brute : seulement des compteurs, des empreintes raccourcies et des scores.
"""
import numpy as np
import pandas as pd
import streamlit as st
from adhd.utils import http

REASONS = {
    'ImageFileError': 'fichier illisible ou corrompu',
    'ValueError': 'contenu non conforme (format, géométrie ou valeurs)',
    'OSError': 'fichier illisible',
    'EOFError': 'fichier tronqué',
    'BadGzipFile': 'archive compressée invalide',
}
ALERTS = {
    'invalid_batch': 'lot refusé à la collecte (manifeste, empreinte ou identifiant non conforme)',
    'batch_delay': 'lot non terminé dans le délai prévu',
    'failed_job': 'tâche en échec après ses relances',
    'drift_or_performance_drop': 'dérive des images ou baisse de performance',
}
ROLES = (('champion', 'Champion (en service)'), ('catalog_challenger', 'Challenger (inscrit au registre)'), ('previous', 'Version précédente'))


def short(value):
    return str(value)[:8] if value else '—'


def when(value):
    try:
        return pd.to_datetime(value, utc=True).tz_convert('Europe/Paris').strftime('%d/%m %H:%M')
    except Exception:
        return str(value)


def model_name(version, versions):
    info = versions.get(version) or {}
    number = info.get('registry_version')
    label = f"version {number}" if number else short(version)
    return f"{label} · {info['architecture']}" if info.get('architecture') else label


def overlay(anatomy, heat):
    """Superposer la sensibilité (en rouge) sur la coupe du cerveau (en gris)."""
    grey = np.clip(np.asarray(anatomy, dtype=float), 0, 1)
    heat = np.asarray(heat, dtype=float)
    top = np.percentile(heat, 99.5) if heat.size else 0
    heat = np.clip(heat / top, 0, 1) ** 0.5 if top > 0 else np.zeros_like(heat)
    image = np.stack([np.clip(grey + 0.9 * heat, 0, 1), grey * (1 - 0.6 * heat), grey * (1 - 0.6 * heat)], axis=-1)
    image = np.rot90(image)
    return np.kron(image, np.ones((6, 6, 1)))  # agrandissement : 64 points -> 384 pixels


st.set_page_config(page_title='ADHD200 — démonstration', layout='wide')
st.title('ADHD200 — pipeline de recherche')
st.warning('Démonstration technique : les scores ne sont pas des diagnostics médicaux.')
if st.button('Actualiser'):
    st.rerun()
try:
    report = http('/summary')
except Exception:
    st.error('API indisponible : vérifier les services et les journaux.')
    st.stop()

versions = report.get('versions') or {}
deployment = report.get('deployment') or {}
predictions = report.get('predictions') or []
status = {row['status']: row['n'] for row in report.get('items') or []}
journey, results, lineage, models, supervision, explanation = st.tabs(
    ['Trajet des IRM', 'Résultats', 'Suivre une image', 'Modèles', 'Supervision', 'Explication'])

with journey:
    st.caption('Boîte de réception → collecte → préparation → prédiction → base de données. Lancé par Airflow toutes les 5 minutes.')
    a, b, c, d, e = st.columns(5)
    a.metric('Lots reçus', report.get('batch_count', '—'))
    b.metric('Images reçues', sum(status.values()))
    c.metric('Prédites', status.get('predicted', 0))
    d.metric('En quarantaine', status.get('quarantined', 0))
    e.metric('En cours', status.get('received', 0) + status.get('prepared', 0))
    if report.get('sites'):
        st.subheader('Par hôpital')
        table = pd.DataFrame(report['sites']).pivot_table(index='site', columns='status', values='n', aggfunc='sum', fill_value=0)
        table = table.rename(columns={'predicted': 'Prédites', 'quarantined': 'En quarantaine', 'received': 'Reçues', 'prepared': 'Préparées'})
        table.index.name = 'Hôpital'
        st.dataframe(table.reset_index(), hide_index=True)
    if report.get('batches'):
        st.subheader('Derniers lots')
        st.dataframe(pd.DataFrame([{
            'Lot': short(row['id']), 'Reçu le': when(row['created_at']), 'Hôpital': row.get('site') or '—',
            'Images': row['images'], 'Prédites': row['predicted'], 'En quarantaine': row['quarantined'],
            'État': {'complete': 'terminé', 'received': 'en cours'}.get(row['status'], row['status']),
            'Arrivée simulée': 'oui' if row.get('simulation') == 'true' else 'non',
        } for row in report['batches']]), hide_index=True)
    st.subheader('Fichiers écartés (quarantaine)')
    if report.get('quarantine'):
        st.dataframe(pd.DataFrame([{
            'Fichier': short(row['id']), 'Lot': short(row['batch_id']), 'Reçu le': when(row['created_at']),
            'Hôpital': row['site'], 'Motif': REASONS.get(row['reason'], row['reason'] or 'non précisé'),
        } for row in report['quarantine']]), hide_index=True)
        st.caption('Un fichier écarté ne bloque pas les autres images de son lot.')
    else:
        st.info('Aucun fichier en quarantaine.')
    st.caption('Arrivées simulées à partir d’IRM réelles. Brown : diagnostics inconnus. Aucune performance inventée.')

with results:
    if predictions:
        st.dataframe(pd.DataFrame([{
            'Date': when(row['created_at']), 'Hôpital': row['site'], 'Image': short(row['item_id']),
            'Score': round(row['score'], 3),
            'Position': 'au-dessus du seuil' if row['score'] >= ((versions.get(row['version']) or {}).get('threshold') or 0.5) else 'sous le seuil',
            'Modèle': model_name(row['version'], versions),
            'Parcours': {'champion': 'champion', 'canary': 'challenger (essai progressif)'}.get(row['route'], row['route']),
        } for row in predictions]), hide_index=True)
        st.subheader('Répartition des scores')
        scores = pd.Series([row['score'] for row in predictions])
        bins = pd.cut(scores, bins=[i / 10 for i in range(11)], include_lowest=True).value_counts().sort_index()
        st.bar_chart(pd.DataFrame({'Nombre d’images': bins.values}, index=[f"{i / 10:.1f} à {(i + 1) / 10:.1f}" for i in range(10)]))
        st.caption('Score entre 0 et 1. Des scores très tranchés sur un hôpital jamais vu signalent un modèle trop sûr de lui, pas une certitude.')
    else:
        st.info('Aucune prédiction pour l’instant.')

with lineage:
    st.caption('Retrouver l’origine d’un résultat : du fichier reçu jusqu’au score.')
    if predictions:
        labels = [f"{short(row['item_id'])} · {row['site']} · {when(row['created_at'])}" for row in predictions]
        row = predictions[labels.index(st.selectbox('Image', labels))]
        st.dataframe(pd.DataFrame([
            ('1. Lot d’arrivée', short(row.get('batch_id'))),
            ('2. Fichier d’origine (empreinte)', short(row['item_id'])),
            ('3. Patient (code brouillé)', short(row.get('subject'))),
            ('4. Recette de préparation (version)', short(row.get('recipe'))),
            ('5. Volume préparé (empreinte)', short(row.get('prepared_sha'))),
            ('6. Modèle', model_name(row['version'], versions)),
            ('7. Score', f"{row['score']:.3f}"),
            ('8. Enregistré le', when(row['created_at'])),
        ], columns=['Étape', 'Valeur']), hide_index=True)
        st.caption('Les empreintes sont raccourcies à 8 caractères. Aucun numéro de dossier n’est conservé en base.')
    else:
        st.info('Aucune prédiction pour l’instant.')

with models:
    rows = []
    for key, label in ROLES:
        version = deployment.get(key)
        if not version:
            continue
        info = versions.get(version) or {}
        auc = info.get('validation_roc_auc')
        rows.append({'Rôle': label, 'Version du registre': info.get('registry_version') or '—',
                     'Architecture': info.get('architecture') or '—',
                     'AUC de validation': round(auc, 3) if isinstance(auc, (int, float)) else '—',
                     'Seuil': info.get('threshold', '—'), 'Empreinte': short(version)})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.info('Aucun modèle déployé.')
    fraction = float(deployment.get('fraction') or 0)
    if deployment.get('challenger'):
        st.write(f"Mise en service progressive : **{fraction:.0%}** des demandes sont confiées au challenger.")
    else:
        st.write('Mise en service progressive : aucun challenger ne reçoit de demandes actuellement (0 %).')
    if deployment.get('registry_checked_at'):
        st.caption(f"Étiquettes lues dans le registre MLflow (DagsHub) le {when(deployment['registry_checked_at'])}.")
    if report.get('deployments'):
        st.subheader('Journal des changements de modèle')
        st.dataframe(pd.DataFrame([{
            'Date': when(row['created_at']),
            'Champion': model_name(row['body'].get('champion'), versions) if row['body'].get('champion') else '—',
            'Challenger actif': model_name(row['body'].get('challenger'), versions) if row['body'].get('challenger') else 'aucun',
            'Part du challenger': f"{float(row['body'].get('fraction') or 0):.0%}",
            'Origine': 'étiquettes du registre' if row['body'].get('source') == 'registry_aliases' else 'action locale',
        } for row in report['deployments']]), hide_index=True)
    st.caption('Champion et challenger sont de véritables versions entraînées. Aucun modèle n’est créé au démarrage.')

with supervision:
    st.subheader('Alertes')
    if report.get('alerts'):
        st.dataframe(pd.DataFrame([{
            'Date': when(row['created_at']),
            'Alerte': ALERTS.get(row['body'].get('code'), row['body'].get('code')),
            'Objet': short(row['body'].get('batch') or row['body'].get('job') or ''),
        } for row in report['alerts']]), hide_index=True)
    else:
        st.info('Aucune alerte.')
    st.subheader('Dernier contrôle de dérive')
    last = report.get('monitoring')
    if last:
        body = last['body']
        minimum = report.get('monitoring_minimum_subjects')
        if body.get('input_drift') is None:
            st.info(f"Contrôle du {when(last['created_at'])} : {body.get('subjects', 0)} patient(s) comparé(s). "
                    f"Pas assez pour conclure (minimum : {minimum}).")
        elif body['input_drift']:
            st.error(f"Contrôle du {when(last['created_at'])} : les images reçues diffèrent de celles de l’entraînement.")
        else:
            st.success(f"Contrôle du {when(last['created_at'])} : pas de dérive détectée sur {body.get('subjects')} patient(s).")
        if body.get('limit'):
            st.caption(body['limit'])
    else:
        st.info('Aucun contrôle enregistré.')
    costs = report.get('costs')
    if costs:
        st.subheader('Durées mesurées et coûts estimés')
        st.dataframe(pd.DataFrame([{
            'Tâche': row['task'], 'Exécutions': row['runs'], 'Durée totale (s)': row['seconds_total'],
            'Durée moyenne (s)': row['seconds_mean'], 'Mémoire max (Mio)': row['peak_memory_mib'],
        } for row in costs['tasks']]), hide_index=True)
        a, b, c, d = st.columns(4)
        a.metric('Calcul local', f"{costs['compute_hours']} h")
        b.metric('Coût local estimé', f"{costs['local_cost_eur']} €")
        c.metric('Équivalent cloud', f"{costs['cloud_cpu_equivalent_eur']} €")
        d.metric('CO2 estimé', f"{costs['local_co2_g']} g")
        st.caption(f"Entraînement : {costs['training_gpu_hours']} h de carte graphique (équivalent cloud {costs['training_cloud_equivalent_eur']} €). "
                   'Durées mesurées par le pipeline ; tarifs = hypothèses réglables dans configs/service.yaml.')

with explanation:
    st.caption('Quelles zones de l’image ont le plus pesé dans le score ? En rouge : les plus influentes.')
    ids = list(dict.fromkeys(row['item_id'] for row in predictions))
    if ids:
        labels = {f"{short(i)} · {next(r['site'] for r in predictions if r['item_id'] == i)}": i for i in ids}
        selected = labels[st.selectbox('Volume préparé', list(labels))]
        if st.button('Calculer la sensibilité'):
            try:
                result = http('/explain', {'ids': [selected]})
                names = ('Vue de côté', 'Vue de face', 'Vue de dessus')
                if result.get('anatomy') and result.get('heat'):
                    for column, name, anatomy, heat in zip(st.columns(3), names, result['anatomy'], result['heat']):
                        column.image(overlay(anatomy, heat), caption=name, clamp=True)
                else:  # ancienne API : carte seule, agrandie
                    for column, name, plane in zip(st.columns(3), names, result['planes']):
                        column.image(np.kron(np.asarray(plane, dtype=float) ** 0.5, np.ones((12, 12))), caption=name, clamp=True)
                st.caption(result['limit'])
            except Exception:
                st.error('Explication indisponible. Consulter les journaux.')
    else:
        st.info('Une prédiction réelle est nécessaire pour afficher une explication.')
