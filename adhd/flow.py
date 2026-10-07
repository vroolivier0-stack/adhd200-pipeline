"""Collecte, traitement, supervision et tâches persistantes avec trois essais."""
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time
import uuid
import numpy as np
from .utils import root,safe,read,save_json,db,event,fingerprint,recipe,recipe_hash,settings,http,volume

def collect():
    batches=[];incoming=root()/'incoming';incoming.mkdir(parents=True,exist_ok=True,mode=0o700)
    for folder in sorted(incoming.iterdir()):
        if not folder.is_dir() or folder.is_symlink() or folder.name.startswith('.') or not (folder/'READY').is_file():continue
        try:
            manifest=read(safe(folder,'manifest.json'));rows=manifest['files'];names=set()
            if manifest.get('schema_version')!=1 or not 1<=len(rows)<=10:raise ValueError('Manifeste invalide')
            for row in rows:
                source=safe(folder,row['file'])
                if row['file'] in names or not str(source).endswith(('.nii','.nii.gz')) or source.stat().st_size>300*1024**2 or fingerprint(source)!=row['sha256']:raise ValueError('Fichier invalide')
                names.add(row['file'])
                if row['site'] not in {'KKI','NYU','OHSU','PEK','Pittsburgh','Brown','WashU','NeuroIMAGE'} or row['cohort'] not in {'arrivals_reserve','evaluation_only','new_labeled','synthetic'} or not 1<=len(row['subject'])<=100:raise ValueError('Provenance invalide')
            identifier=fingerprint(folder/'manifest.json');backup=root()/'arrivals/raw'/identifier
            backup.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            if not backup.exists():
                staging=backup.parent/('.'+uuid.uuid4().hex)
                shutil.copytree(folder,staging)
                for row in rows:
                    if fingerprint(safe(staging,row['file']))!=row['sha256']:raise ValueError('Copie différente')
                staging.rename(backup)
            from psycopg.types.json import Jsonb
            with db() as conn:
                conn.execute('INSERT INTO batches(id,manifest) VALUES (%s,%s) ON CONFLICT DO NOTHING',(identifier,Jsonb(manifest)))
                for row in rows:
                    conn.execute('INSERT INTO items(id,batch_id,subject,site,cohort,raw_ref) VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING',(row['sha256'],identifier,row['subject'],row['site'],row['cohort'],(backup/row['file']).relative_to(root()).as_posix()))
                    old=conn.execute('SELECT subject,site,cohort FROM items WHERE id=%s',(row['sha256'],)).fetchone()
                    if old['subject']!=row['subject'] or old['site']!=row['site'] or old['cohort']!=row['cohort']:raise ValueError('Association contradictoire')
            save_json(folder/'RECEIVED.json',{'id':identifier,'raw_preserved':True});batches.append(identifier)
        except (OSError,ValueError,KeyError) as error:
            key=hashlib.sha256(folder.name.encode()).hexdigest()
            save_json(root()/'quarantine/compact'/(key+'.json'),{'stage':'collect','error_type':type(error).__name__})
            event('alert',{'code':'invalid_batch','batch':key},key)
    return {'batches':batches}

def features(array):return {'mean':float(array.mean()),'std':float(array.std()),'p90':float(np.percentile(array,90)),'occupied':float(np.count_nonzero(array)/array.size)}
def prepare(batch_ids):
    from .data import prepare_image
    from importlib.metadata import version
    code=fingerprint(Path(__file__).with_name('data.py'))
    signature=hashlib.sha256((recipe_hash()+code+version('monai')+version('torch')).encode()).hexdigest();prepared=bad=0
    with db() as conn:rows=conn.execute("SELECT * FROM items WHERE batch_id=ANY(%s) AND status='received'",(batch_ids,)).fetchall()
    for row in rows:
        try:
            source=safe(root(),row['raw_ref'])
            if fingerprint(source)!=row['id']:raise ValueError('Original modifié')
            path=root()/'arrivals/processed'/signature/(row['id']+'.npz');receipt=path.with_suffix('.json')
            if not (path.is_file() and receipt.is_file() and read(receipt)['sha256']==fingerprint(path)):
                result=prepare_image(source,{'preprocessing':recipe()});path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
                pending=path.with_suffix('.tmp')
                with pending.open('wb') as stream:np.savez_compressed(stream,image=result['array'])
                if not np.array_equal(volume(pending),result['array']):raise ValueError('Relecture différente')
                pending.replace(path);save_json(receipt,{'source_sha256':row['id'],'sha256':fingerprint(path),'statistics':result['statistics'],'recipe_hash':signature,'code_sha256':fingerprint(Path(__file__).with_name('data.py'))})
            from psycopg.types.json import Jsonb
            with db() as conn:conn.execute("UPDATE items SET status='prepared',prepared_ref=%s,prepared_sha=%s,recipe=%s,features=%s WHERE id=%s",(path.relative_to(root()).as_posix(),fingerprint(path),recipe_hash(),Jsonb(features(volume(path))),row['id']))
            prepared+=1
        except Exception as error:
            with db() as conn:conn.execute("UPDATE items SET status='quarantined',reason=%s WHERE id=%s",(type(error).__name__,row['id']))
            save_json(root()/'quarantine/compact'/(row['id']+'.json'),{'stage':'prepare','error_type':type(error).__name__})
            bad+=1
    return {'batches':batch_ids,'prepared':prepared,'quarantined':bad}

def predict(batch_ids):
    with db() as conn:rows=conn.execute("SELECT id FROM items WHERE batch_id=ANY(%s) AND status='prepared' ORDER BY id",(batch_ids,)).fetchall()
    for start in range(0,len(rows),4):
        ids=[r['id'] for r in rows[start:start+4]];http('/predict-batch',{'ids':ids})
        with db() as conn:conn.execute("UPDATE items SET status='predicted' WHERE id=ANY(%s)",(ids,))
    with db() as conn:conn.execute("UPDATE batches SET status='complete' WHERE id=ANY(%s)",(batch_ids,))
    return {'predicted':len(rows),'batches':batch_ids}

def monitor():
    from .service import state
    from .registry_deployment import check_registry
    registry_alignment = check_registry()
    current=state();report={'status':'waiting_for_model'};policy=settings()
    with db() as conn:
        late=conn.execute("SELECT id FROM batches WHERE status!='complete' AND created_at<now()-(%s * interval '1 second')",(policy['batch_delay_seconds'],)).fetchall()
        failures=conn.execute("SELECT id FROM jobs WHERE status='failed'").fetchall()
    for row in late:event('alert',{'code':'batch_delay','batch':row['id']},'delay_'+row['id'])
    for row in failures:event('alert',{'code':'failed_job','job':row['id']},'failed_'+row['id'])
    if current['champion']:
        manifest=read(root()/'models/compact'/current['champion']/'release.json');reference=manifest['reference'];keys=reference['keys']
        with db() as conn:
            rows=conn.execute("SELECT * FROM (SELECT DISTINCT ON (i.subject) i.subject,i.features,b.created_at FROM items i JOIN batches b ON b.id=i.batch_id WHERE i.status='predicted' AND i.cohort!='synthetic' ORDER BY i.subject,b.created_at DESC) s ORDER BY created_at DESC LIMIT 100").fetchall()
            labels=conn.execute("SELECT * FROM (SELECT DISTINCT ON (i.subject) i.subject,i.site,l.target,p.score,p.created_at FROM predictions p JOIN items i ON i.id=p.item_id JOIN labels l ON l.item_id=i.id WHERE p.version=%s AND i.cohort!='synthetic' ORDER BY i.subject,p.created_at DESC) s ORDER BY created_at DESC LIMIT 100",(current['champion'],)).fetchall()
            new=conn.execute("SELECT DISTINCT ON (i.subject) i.subject,i.site,i.prepared_ref,i.prepared_sha,l.target FROM items i JOIN labels l ON i.id=l.item_id WHERE l.purpose='new_training' AND i.cohort='new_labeled' ORDER BY i.subject,i.id").fetchall()
        report={'version':current['champion'],'subjects':len(rows),'input_drift':None,'performance':None};detected=False
        if len(rows)>=policy['minimum_monitoring_subjects']:
            values=np.asarray([[r['features'][k] for k in keys] for r in rows]);shift=np.abs(values.mean(0)-reference['mean'])/np.maximum(reference['std'],1e-4)
            detected=bool(np.any(shift>policy['standardized_shift_threshold']));report.update(input_drift=detected,standardized_shifts=shift.tolist())
        if all(sum(r['target']==c for r in labels)>=policy['minimum_labeled_each_class'] for c in (0,1)):
            from .learning import grouped_metrics
            report['performance']=grouped_metrics([r['target'] for r in labels],[r['score'] for r in labels],[r['site'] for r in labels])
            detected=detected or report['performance']['global']['roc_auc']<manifest['validation']['roc_auc']-policy['performance_auc_drop']
        if detected:
            event('alert',{'code':'drift_or_performance_drop','limit':'Changement de centre/population et faible effectif à examiner.'},'drift_'+current['champion'])
            if len(new)>=policy['minimum_new_training_subjects']:
                snapshot={'rows':new,'champion':current['champion'],'test_used':False};identifier=hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest();path=root()/'reference/compact/retraining'/(identifier+'.json')
                save_json(path,snapshot);event('retraining',{'snapshot':path.relative_to(root()).as_posix(),'subjects':len(new),'submitted':False},identifier)
                if policy['automatic_retraining']:
                    from .cli import submit_retraining
                    submit_retraining(path)
        report['limit']='Dérive descriptive distincte des performances. Brown sans cibles réelles ne fournit pas de métriques supervisées.'
    event('monitoring',report);return report

def worker():
    from psycopg.types.json import Jsonb
    while True:
        with db() as conn:
            conn.execute("UPDATE jobs SET status=CASE WHEN attempts>=3 THEN 'failed' ELSE 'queued' END WHERE status='running' AND lease_at<now()-interval '30 minutes'")
            job=conn.execute("SELECT * FROM jobs WHERE status='queued' AND attempts<3 ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1").fetchone()
            if job:conn.execute("UPDATE jobs SET status='running',attempts=attempts+1,lease_at=now() WHERE id=%s",(job['id'],))
        if not job:time.sleep(2);continue
        start=time.monotonic()
        try:
            result={'collect':collect,'monitor':monitor}.get(job['kind'])
            result=result() if result else {'prepare':prepare,'predict':predict}[job['kind']](job['payload']['batches'])
            with db() as conn:conn.execute("UPDATE jobs SET status='succeeded',result=%s WHERE id=%s",(Jsonb(result),job['id']))
            event('resources',{'job':job['id'],'seconds':time.monotonic()-start,'peak_process_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
        except Exception as error:
            with db() as conn:conn.execute('UPDATE jobs SET status=%s,error=%s WHERE id=%s',('failed' if job['attempts']+1>=3 else 'queued',type(error).__name__,job['id']))
            time.sleep(2**min(job['attempts'],3))
