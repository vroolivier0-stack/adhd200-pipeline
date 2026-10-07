"""API de recherche, modèles immuables, vrai canary et retour arrière."""
import hashlib
import hmac
import math
import threading
import time
from pathlib import Path
from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from .utils import root,safe,read,save_json,secret,db,volume,recipe_hash,settings,event,fingerprint

app=FastAPI(title='ADHD200 recherche — non clinique',version='compact-1')
lock=threading.RLock();pending=threading.BoundedSemaphore(2);cache={}
@app.middleware('http')
async def bounded_body(request,call_next):
    from fastapi.responses import JSONResponse
    try:
        if int(request.headers.get('content-length','0'))>16384:return JSONResponse(status_code=413,content={'detail':'Requête trop grande'})
    except ValueError:return JSONResponse(status_code=400,content={'detail':'Longueur invalide'})
    return await call_next(request)

def auth(value,name):
    if not value or not hmac.compare_digest(value,secret(name)):raise HTTPException(401,'Accès refusé')
def reader(x_api_key:str=Header(default='')):auth(x_api_key,'API_KEY')
def admin(x_api_key:str=Header(default='')):auth(x_api_key,'ADMIN_KEY')
def state():
    path=root()/'models/compact/deployment.json'
    return read(path) if path.exists() else {'champion':None,'challenger':None,'fraction':0,'previous':None}
def route(identifier,s):
    if not s['champion']:raise HTTPException(503,'Modèle entraîné absent')
    value=int(hashlib.sha256((identifier+(s['challenger'] or '')).encode()).hexdigest()[:16],16)/2**64
    return (s['challenger'],'canary') if s['challenger'] and value<s['fraction'] else (s['champion'],'champion')
def load(version):
    import torch
    from .learning import create_model,load_checkpoint
    if len(version)!=64 or any(c not in '0123456789abcdef' for c in version):raise ValueError('Version invalide')
    with lock:
        if version in cache:return cache[version]
        folder=safe(root()/'models/compact',version);manifest=read(folder/'release.json')
        if fingerprint(folder/'best.pt')!=manifest['weights_sha256'] or manifest['synthetic'] or recipe_hash(manifest['recipe'])!=recipe_hash() or manifest['profile']['code_sha256'].get('data.py')!=fingerprint(Path(__file__).with_name('data.py')):raise ValueError('Contrat de modèle invalide')
        parameters=manifest['parameters']
        if parameters['base_channels']>32:raise ValueError('Capacité excessive')
        model=create_model(manifest['architecture'],**parameters).eval();checkpoint=load_checkpoint(folder/'best.pt')
        if not all(torch.isfinite(value).all() for value in checkpoint['model_state'].values()):raise ValueError('Poids non finis')
        model.load_state_dict(checkpoint['model_state'],strict=True);torch.set_num_threads(2)
        with torch.inference_mode():
            result=model(torch.zeros(1,1,128,128,128))
            if result.shape!=(1,) or not torch.isfinite(result).all():raise ValueError('Modèle inexploitable')
        if len(cache)>=2:cache.pop(next(iter(cache)))
        cache[version]=(model,manifest);return cache[version]
class Strict(BaseModel):model_config=ConfigDict(extra='forbid')
class Predict(Strict):
    ids:list[str]=Field(min_length=1,max_length=4)
class Job(Strict):
    key:str=Field(min_length=1,max_length=150)
    kind:Literal['collect','prepare','predict','monitor']
    payload:dict=Field(default_factory=dict)
class Label(Strict):
    item_id:str=Field(pattern='^[0-9a-f]{64}$')
    target:Literal[0,1]
    purpose:Literal['evaluation_only','new_training']
    source:str=Field(min_length=5,max_length=200)
class Deploy(Strict):
    action:Literal['candidate','promote','rollback','sync','registry-deploy']
    version:str|None=None
@app.get('/health')
def health():return {'service':'up','research_only':True}
@app.get('/ready')
def ready():
    try:
        with db() as conn:conn.execute('SELECT 1')
        if not state()['champion']:raise ValueError('Absent')
        load(state()['champion']);return {'ready':True}
    except Exception:raise HTTPException(503,'Base ou modèle pas encore prêt')
def get_item(identifier):
    if len(identifier)!=64 or any(c not in '0123456789abcdef' for c in identifier):raise HTTPException(422,'Identifiant invalide')
    with db() as conn:item=conn.execute("SELECT * FROM items WHERE id=%s AND status IN ('prepared','predicted')",(identifier,)).fetchone()
    if not item:raise HTTPException(404,'Volume préparé absent')
    path=safe(root(),item['prepared_ref'])
    if fingerprint(path)!=item['prepared_sha'] or item['recipe']!=recipe_hash():raise HTTPException(409,'Volume ou recette modifié')
    return item,volume(path)
@app.post('/predict-batch',dependencies=[Depends(reader)])
def predict(body:Predict):
    if len(set(body.ids))!=len(body.ids):raise HTTPException(422,'Identifiants répétés')
    if not pending.acquire(False):raise HTTPException(429,'Capacité dépassée')
    try:
        import numpy as np
        import torch
        results=[];groups={};s=state()
        for identifier in body.ids:
            item,array=get_item(identifier);version,way=route(identifier,s);groups.setdefault((version,way),[]).append((item,array))
        with lock:
            for (version,way),items in groups.items():
                start=time.monotonic()
                try:
                    model,manifest=load(version)
                    with torch.inference_mode():scores=torch.sigmoid(model(torch.from_numpy(np.stack([a for _,a in items])).unsqueeze(1))).tolist()
                    if not all(math.isfinite(v) and 0<=v<=1 for v in scores):raise ValueError('Score invalide')
                    with db() as conn:
                        for (item,_),score in zip(items,scores,strict=True):
                            conn.execute('INSERT INTO predictions(item_id,version,score,route) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING',(item['id'],version,score,way))
                            results.append({'item_id':item['id'],'version':version,'score':score,'decision':int(score>=manifest['threshold']),'route':way,'research_only':True})
                    event('serving',{'version':version,'route':way,'success':True,'seconds':time.monotonic()-start})
                except Exception:
                    event('serving',{'version':version,'route':way,'success':False,'seconds':time.monotonic()-start});raise
        return {'predictions':results}
    except HTTPException:raise
    except Exception:raise HTTPException(503,'Inférence indisponible')
    finally:pending.release()
@app.post('/predict',dependencies=[Depends(reader)])
def predict_one(body:Predict):
    if len(body.ids)!=1:raise HTTPException(422,'Une seule image attendue')
    return predict(body)
@app.post('/explain',dependencies=[Depends(reader)])
def explain(body:Predict):
    if len(body.ids)!=1:raise HTTPException(422,'Une image attendue')
    if not pending.acquire(False):raise HTTPException(429,'Capacité dépassée')
    try:
        import numpy as np
        import torch
        _,array=get_item(body.ids[0]);version,_=route(body.ids[0],state())
        with lock:
            model,_=load(version);image=torch.from_numpy(array.copy()).unsqueeze(0).unsqueeze(0).requires_grad_(True)
            grad=torch.autograd.grad(model(image).sum(),image)[0];importance=(grad*image).abs().detach().numpy()[0,0]
            if not np.isfinite(importance).all():raise ValueError('Explication invalide')
            importance/=max(float(importance.max()),1e-12)
        return {'version':version,'method':'absolute_gradient_times_input','planes':[np.take(importance,64,axis=i)[::4,::4].tolist() for i in range(3)],
                # Pour l'affichage : la coupe du cerveau et la sensibilité au même format (64 x 64), à superposer.
                'anatomy':[np.take(array,64,axis=i)[::2,::2].tolist() for i in range(3)],'heat':[np.take(importance,64,axis=i)[::2,::2].tolist() for i in range(3)],
                'limit':'Sensibilité locale, pas de preuve causale ou médicale.'}
    finally:pending.release()
@app.post('/jobs',dependencies=[Depends(admin)])
def enqueue(body:Job):
    import json
    from psycopg.types.json import Jsonb
    identifier=hashlib.sha256(json.dumps(body.model_dump(),sort_keys=True).encode()).hexdigest()
    with db() as conn:conn.execute('INSERT INTO jobs(id,kind,payload) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING',(identifier,body.kind,Jsonb(body.payload)))
    return {'id':identifier}
@app.get('/jobs/{identifier}',dependencies=[Depends(admin)])
def job(identifier:str):
    with db() as conn:result=conn.execute('SELECT id,status,result,error,attempts FROM jobs WHERE id=%s',(identifier,)).fetchone()
    if not result:raise HTTPException(404,'Tâche absente')
    return result
@app.post('/labels',dependencies=[Depends(admin)])
def label(body:Label):
    with db() as conn:
        item=conn.execute('SELECT * FROM items WHERE id=%s',(body.item_id,)).fetchone()
        if not item:raise HTTPException(404,'Image absente')
        if item['cohort']=='synthetic':raise HTTPException(409,'Synthétique exclu')
        if body.purpose=='new_training':
            protected=read(safe(root(),'reports/private/split_private_20261005T191224Z_fc7dc57c.json'))
            if item['cohort']!='new_labeled' or item['site'] not in {'KKI','NYU','OHSU','PEK','Pittsburgh'} or item['subject'] in {r['pseudo_id'] for r in protected['records']}:raise HTTPException(409,'Patient ou centre protégé')
        old=conn.execute('SELECT * FROM labels WHERE item_id=%s',(body.item_id,)).fetchone()
        if old and (old['target']!=body.target or old['purpose']!=body.purpose):raise HTTPException(409,'Cible contradictoire')
        conn.execute('INSERT INTO labels(item_id,target,purpose,source) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING',(body.item_id,body.target,body.purpose,body.source))
    return {'recorded':True}

def validate_candidate(metrics, policy):
    for name in ("roc_auc", "balanced_accuracy"):
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Mesure de validation absente ou invalide")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Mesure de validation non finie ou hors limites")
    if (metrics["roc_auc"] < policy["minimum_auc"]
        or metrics["balanced_accuracy"] < policy["minimum_balanced_accuracy"]):
        raise ValueError("Validation insuffisante")
    confusion = metrics.get("confusion", {})
    keys = ("true_negatives", "false_positives",
            "false_negatives", "true_positives")
    counts = [confusion.get(key) for key in keys]
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("Matrice de confusion invalide")
    tn, fp, fn, tp = counts
    if tn + fp == 0 or fn + tp == 0:
        raise ValueError("Deux classes réelles requises pour la validation")
    if tp + fp == 0 or tn + fn == 0:
        raise ValueError("Toutes les prédictions appartiennent à la même classe")

@app.post('/deployment',dependencies=[Depends(admin)])
def deploy(body:Deploy):
    if body.action in ('sync', 'registry-deploy'):
        try:
            from .registry_deployment import deploy_from_registry
            return deploy_from_registry()
        except Exception:
            raise HTTPException(503, 'Registre indisponible ou paquet invalide : modèle servi conservé')
    try:
        with lock:
            s=state();policy=settings()
            if body.action=='candidate':
                _,m=load(body.version);metrics=m['validation']
                validate_candidate(metrics, policy)
                if s.get('registry_versions') and body.version!=s.get('catalog_challenger'):raise ValueError('Candidat différent de l’alias challenger')
                if not s['champion']:s['champion']=body.version
                elif s['champion']!=body.version:s.update(challenger=body.version,fraction=policy['canary_fraction'],started_at=time.time())
                else:raise ValueError('Déjà champion')
            elif body.action=='promote':
                if not s['challenger']:raise ValueError('Candidat absent')
                with db() as conn:rows=conn.execute("SELECT body FROM events WHERE kind='serving' AND body->>'version'=%s AND extract(epoch from created_at)>=%s",(s['challenger'],s['started_at'])).fetchall()
                import numpy as np
                if len(rows)<policy['minimum_canary_requests'] or sum(not r['body']['success'] for r in rows)/len(rows)>policy['maximum_error_fraction'] or np.percentile([r['body']['seconds'] for r in rows],95)>policy['maximum_p95_seconds']:raise ValueError('Canary insuffisant')
                if load(s['challenger'])[1]['validation']['roc_auc']<load(s['champion'])[1]['validation']['roc_auc']:raise ValueError('Validation inférieure')
                if s.get('registry_versions'):
                    from .registry_deployment import promote_registry
                    return promote_registry(s['challenger'])
                s.update(previous=s['champion'],champion=s['challenger'],challenger=None,fraction=0)
            elif body.action=='rollback':
                if s['challenger']:s.update(challenger=None,fraction=0)
                elif s.get('registry_versions'):raise ValueError('Restaurer l’alias puis déployer depuis le registre')
                elif s['previous']:s['champion'],s['previous']=s['previous'],s['champion']
                else:raise ValueError('Version précédente absente')
            else:raise ValueError('Action inconnue')
            save_json(root()/'models/compact/deployment.json',s);event('deployment',s);return s
    except (ValueError,RuntimeError,OSError,TypeError):raise HTTPException(409,'Action refusée : vérifier les critères et la version')
def describe_version(version):
    """Fiche lisible d'une version de modèle installée (architecture, score de validation, seuil)."""
    try:
        manifest=read(safe(root()/'models/compact',version)/'release.json');validation=manifest.get('validation') or {}
        return {'version':version,'registry_version':str(manifest.get('mlflow_version') or ''),'architecture':manifest.get('architecture'),'threshold':manifest.get('threshold'),
                'validation_roc_auc':validation.get('roc_auc'),'validation_balanced_accuracy':validation.get('balanced_accuracy')}
    except Exception:return {'version':version}
@app.get('/summary',dependencies=[Depends(reader)])
def summary():
    with db() as conn:
        counts=conn.execute('SELECT status,count(*) AS n FROM items GROUP BY status').fetchall()
        predictions=conn.execute('SELECT p.*,i.site,i.batch_id,i.subject,i.prepared_sha,i.recipe FROM predictions p JOIN items i ON i.id=p.item_id ORDER BY p.created_at DESC LIMIT 200').fetchall()
        events=conn.execute("SELECT kind,body,created_at FROM events WHERE kind IN ('alert','monitoring','resources','retraining','registry_check') ORDER BY created_at DESC LIMIT 20").fetchall()
        sites=conn.execute('SELECT site,status,count(*) AS n FROM items GROUP BY site,status ORDER BY site,status').fetchall()
        batches=conn.execute("SELECT b.id,b.created_at,b.status,b.manifest->>'simulation' AS simulation,min(i.site) AS site,count(i.id) AS images,count(*) FILTER (WHERE i.status='predicted') AS predicted,count(*) FILTER (WHERE i.status='quarantined') AS quarantined FROM batches b LEFT JOIN items i ON i.batch_id=b.id GROUP BY b.id ORDER BY b.created_at DESC LIMIT 50").fetchall()
        batch_count=conn.execute('SELECT count(*) AS n FROM batches').fetchone()['n']
        quarantine=conn.execute("SELECT i.id,i.batch_id,i.site,i.reason,b.created_at FROM items i JOIN batches b ON b.id=i.batch_id WHERE i.status='quarantined' ORDER BY b.created_at DESC LIMIT 50").fetchall()
        alerts=conn.execute("SELECT body,created_at FROM events WHERE kind='alert' ORDER BY created_at DESC LIMIT 30").fetchall()
        monitoring=conn.execute("SELECT body,created_at FROM events WHERE kind='monitoring' ORDER BY created_at DESC LIMIT 1").fetchone()
        deployments=conn.execute("SELECT body,created_at FROM events WHERE kind='deployment' ORDER BY created_at DESC LIMIT 15").fetchall()
    # Coûts estimés à partir des durées mesurées (voir adhd/costs.py) ; absents si non configurés.
    try:
        from .costs import summary as cost_summary
        costs=cost_summary()
    except Exception:costs=None
    deployment=state()
    known={p['version'] for p in predictions}|{deployment.get(k) for k in ('champion','challenger','previous','catalog_challenger')}
    versions={v:describe_version(v) for v in known if v}
    return {'deployment':deployment,'items':counts,'predictions':predictions,'events':events,'costs':costs,
            'sites':sites,'batches':batches,'batch_count':batch_count,'quarantine':quarantine,'alerts':alerts,'monitoring':monitoring,'deployments':deployments,'versions':versions,
            'monitoring_minimum_subjects':settings().get('minimum_monitoring_subjects')}

@app.post('/watchdog',dependencies=[Depends(admin)])
def watchdog():
    with db() as conn:
        late=conn.execute("SELECT id FROM batches WHERE status!='complete' AND created_at<now()-(%s * interval '1 second')",(settings()['batch_delay_seconds'],)).fetchall()
        failures=conn.execute("SELECT id FROM jobs WHERE status='failed' OR (status='running' AND lease_at<now()-interval '30 minutes')").fetchall()
    for row in late:event('alert',{'code':'batch_delay','batch':row['id']},'delay_'+row['id'])
    for row in failures:event('alert',{'code':'failed_or_stalled_job','job':row['id']},'failed_'+row['id'])
    return {'late_batches':len(late),'failed_jobs':len(failures)}
