"""Dashboard métier et supervision : aucun identifiant brut affiché."""
import numpy as np
import streamlit as st
from adhd.utils import http
st.set_page_config(page_title='ADHD200 — démonstration',layout='wide')
st.title('ADHD200 — pipeline de recherche')
st.warning('Démonstration technique : les scores ne sont pas des diagnostics médicaux.')
if st.button('Actualiser'):st.rerun()
try:report=http('/summary')
except Exception:st.error('API indisponible : vérifier les services et les journaux.');st.stop()
data,models,supervision,explanation=st.tabs(['Données et résultats','Modèles','Supervision','Explication'])
with data:
    st.dataframe(report['items']);st.dataframe(report['predictions'])
    st.caption('Brown : arrivées simulées, diagnostics inconnus. Aucune performance inventée.')
with models:st.json(report['deployment']);st.caption('Champion et candidat sont de véritables versions entraînées. Aucun modèle n’est créé au démarrage.')
with supervision:
    for value in report['events']:st.write(value['created_at'],value['kind']);st.json(value['body'])
with explanation:
    ids=list(dict.fromkeys(r['item_id'] for r in report['predictions']))
    if ids:
        selected=st.selectbox('Volume préparé',ids)
        if st.button('Calculer une sensibilité locale'):
            try:
                result=http('/explain',{'ids':[selected]});st.write(result['limit'])
                for column,plane in zip(st.columns(3),result['planes']):column.image(np.asarray(plane),clamp=True)
            except Exception:st.error('Explication indisponible. Consulter les journaux.')
    else:st.info('Une prédiction réelle est nécessaire pour afficher une explication.')
