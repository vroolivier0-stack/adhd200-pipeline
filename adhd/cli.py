"""Commandes intégrées : préparation, Kaggle, publication et démonstration."""
import argparse
import base64
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid
import zipfile
import numpy as np
from .utils import root,read,save_json,safe,fingerprint,recipe,recipe_hash,volume,secret

def verify(package):
    package=Path(package);info=read(safe(package,'dataset_info.json'));table=safe(package,'dataset.csv')
    if info.get('contains_test_data') is not False or fingerprint(table)!=info['dataset_csv_sha256'] or not safe(package,'LOCAL_EXPORT_COMPLETE.txt').is_file():raise ValueError('Paquet incomplet ou modifié')
    with table.open(newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=['image_id','file','split','target','site','sha256']:raise ValueError('Colonnes inconnues')
        rows=list(reader)
    ids=set();files=set();contents={};counts=Counter()
    for row in rows:
        if None in row or any(v is None for v in row.values()) or row['split'] not in {'train','validation'} or row['target'] not in {'0','1'} or row['site'] not in {'KKI','NYU','OHSU','PEK','Pittsburgh'}:raise ValueError('Image non autorisée')
        if row['image_id'] in ids or row['file'] in files:raise ValueError('Image répétée')
        ids.add(row['image_id']);files.add(row['file']);path=safe(package,row['file'])
        if fingerprint(path)!=row['sha256']:raise ValueError('Empreinte incorrecte')
        array=volume(path);content=hashlib.sha256(array.astype('<f4',copy=False).tobytes()).hexdigest()
        if contents.setdefault(content,row['split'])!=row['split']:raise ValueError('Fuite entre groupes')
        counts[row['split']]+=1
    if dict(counts)!=info['groups'] or any(not any(r['split']==s and r['target']==c for r in rows) for s in ('train','validation') for c in ('0','1')):raise ValueError('Effectifs incohérents')
    allowed=files|{'dataset.csv','dataset_info.json','README.txt','LOCAL_EXPORT_COMPLETE.txt'}
    actual={p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()}
    if any(p.is_symlink() for p in package.rglob('*')) or actual!=allowed:raise ValueError('Fichier inattendu')
    return info,rows

def compatibility(package,split_file,export_report):
    from .data import prepare_image
    info,rows=verify(package);split=read(split_file);private=read(export_report)
    mapping={r['image_id']:r for r in private['private_correspondences']};sources={r['source_sha256']:r for r in split['records']};results=[];sites=set()
    for row in rows:
        if row['split']!='train' or row['site'] in sites:continue
        sites.add(row['site']);source=sources[mapping[row['image_id']]['source_sha256']]
        original=safe(root()/'raw',source['source_relative_path'])
        if fingerprint(original)!=source['source_sha256']:raise ValueError('Original modifié')
        prepared=prepare_image(original,{'preprocessing':recipe()})['array'];exported=volume(safe(package,row['file']))
        same=bool(np.allclose(prepared,exported,atol=1e-5,rtol=1e-4));results.append({'site':row['site'],'compatible':same,'max_abs_difference':float(np.max(np.abs(prepared-exported)))})
    decision={'csv_sha256':fingerprint(Path(package)/'dataset.csv'),'recipe_hash':recipe_hash(),'data_code_sha256':fingerprint(Path(__file__).with_name('data.py')),'samples':results,'compatible':len(results)==5 and all(r['compatible'] for r in results)}
    save_json(root()/'reference/compact/compatibility.json',decision)
    if not decision['compatible']:raise ValueError('Prétraitement différent : préparer un nouveau paquet avant Kaggle')
    info['compatibility']=decision;save_json(Path(package)/'dataset_info.json',info);return decision


def _validate_mlflow_policy(policy):
    epoch_names = {
        "train_loss", "validation_loss", "learning_rate", "epoch_duration_seconds",
        "peak_gpu_memory_mib", "optimizer_updates", "skipped_updates",
    }
    validation_names = {
        "roc_auc", "average_precision", "accuracy", "balanced_accuracy",
        "sensitivity", "specificity", "precision", "f1", "brier_score",
    }
    if not isinstance(policy, dict) or policy.get("policy_version") != 1:
        raise ValueError("Politique MLflow version 1 attendue")
    if policy.get("publication") != "end_of_run":
        raise ValueError("Seule la publication en fin d'essai est implémentée")
    every = policy.get("log_every_epochs")
    if isinstance(every, bool) or not isinstance(every, int) or every < 1:
        raise ValueError("Fréquence d'échantillonnage des époques invalide")
    if not isinstance(policy.get("experiment"), str) or not policy["experiment"].strip():
        raise ValueError("Nom d'expérience MLflow manquant")
    if not isinstance(policy.get("report_by_site"), bool):
        raise ValueError("report_by_site doit être booléen")
    for group, allowed in (("epoch", epoch_names), ("validation", validation_names)):
        names = policy.get("metrics", {}).get(group)
        if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or name not in allowed for name in names)
            or len(set(names)) != len(names)):
            raise ValueError(f"Sélection de métriques MLflow invalide : {group}")
    artifacts = policy.get("artifacts")
    allowed_artifacts = {
        "run_info.json", "summary.json", "history.json",
        "best_validation.json", "best.pt", "last.pt",
    }
    if (not isinstance(artifacts, list) or not artifacts
        or any(not isinstance(name, str) or name not in allowed_artifacts for name in artifacts)
        or len(set(artifacts)) != len(artifacts)):
        raise ValueError("Sélection d'artefacts MLflow invalide")
    return policy


def _mlflow_policy():
    import yaml
    path = Path(__file__).resolve().parents[1] / "configs/training.yaml"
    return _validate_mlflow_policy(yaml.safe_load(path.read_text())["tracking"]["mlflow"])


def log_training_details(output, checkpoint):
    import math
    import re
    import mlflow
    policy = _mlflow_policy()
    config = checkpoint["profile"]["config"]
    mlflow.log_params({
        "initial_learning_rate": config["optimizer"]["learning_rate"],
        "weight_decay": config["optimizer"]["weight_decay"],
        "batch_size": config["training"]["batch_size"],
        "accumulation_steps": config["training"]["accumulation_steps"],
        "max_epochs": config["training"]["max_epochs"],
        "dropout": config["models"][checkpoint["model_name"]]["dropout"],
        "base_channels": config["models"][checkpoint["model_name"]]["base_channels"],
        "precision": checkpoint["profile"]["environment"]["precision"],
    })
    mlflow.set_tag("mlflow.runName",
        f'{checkpoint["model_name"]}_lr{config["optimizer"]["learning_rate"]}_seed{config["seed"]}')
    mlflow.log_dict({"scope": "publication_policy", "policy": policy}, "tracking_policy.json")
    history = read(Path(output) / "history.json")
    for index, record in enumerate(history):
        epoch = int(record["epoch"])
        if epoch % policy["log_every_epochs"] and index != len(history) - 1:
            continue
        values = {
            "train_loss": record["train"]["loss"],
            "validation_loss": record["validation_loss"],
            "learning_rate": record["learning_rate_used"],
            "epoch_duration_seconds": record["duration_seconds"],
            "peak_gpu_memory_mib": record["peak_gpu_allocated_mib"],
            "optimizer_updates": record["train"]["optimizer_updates"],
            "skipped_updates": record["train"]["skipped_updates"],
        }
        selected = {name: values[name] for name in policy["metrics"]["epoch"]}
        groups = [("validation_", record["validation_metrics"]["global"])]
        if policy["report_by_site"]:
            groups.extend(("validation_site_" + re.sub(r"[^A-Za-z0-9_]", "_", site) + "_", metrics)
                for site, metrics in record["validation_metrics"]["by_site"].items())
        for prefix, metrics in groups:
            for name in policy["metrics"]["validation"]:
                value = metrics[name]
                if value is not None:
                    selected[prefix + name] = value
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in selected.values()):
            raise ValueError("Métrique MLflow non numérique ou non finie")
        mlflow.log_metrics(selected, step=epoch)


def publish(output,package):
    import torch
    import mlflow
    import mlflow.pytorch
    from .learning import load_checkpoint,create_model
    from .flow import features
    info,rows=verify(package);checkpoint=load_checkpoint(Path(output)/'best.pt');validation=read(Path(output)/'best_validation.json')['metrics']['global']
    model=create_model(checkpoint['model_name'],**checkpoint['model_parameters']).eval();model.load_state_dict(checkpoint['model_state'])
    values=np.asarray([list(features(volume(safe(package,r['file']))).values()) for r in rows if r['split']=='train'])
    reference={'keys':['mean','std','p90','occupied'],'mean':values.mean(0).tolist(),'std':values.std(0).tolist(),'count':len(values)}
    uri=os.environ.get('MLFLOW_TRACKING_URI');version=None;run_id=None
    if uri:
        if not uri.startswith('https://'):raise ValueError('MLflow distant HTTPS requis')
        if Path('/run/secrets/mlflow_password').exists():os.environ['MLFLOW_TRACKING_PASSWORD']=secret('MLFLOW_PASSWORD')
        policy = _mlflow_policy();mlflow.set_tracking_uri(uri);mlflow.set_experiment(policy["experiment"])
        with mlflow.start_run(run_id=os.environ.get('ADHD_RECOVERY_RUN_ID')) as run:
            run_id=run.info.run_id;mlflow.log_params({'architecture':checkpoint['model_name'],'seed':checkpoint['profile']['config']['seed'],'data_csv_sha256':fingerprint(Path(package)/'dataset.csv')})
            log_training_details(output, checkpoint)
            for k in policy["metrics"]["validation"]:
                v = validation[k]
                if v is not None:mlflow.log_metric(k,v)
            with tempfile.TemporaryDirectory() as directory:
                local_model = Path(directory) / 'model'
                mlflow.pytorch.save_model(
                    model,
                    path=str(local_model),
                    **({"serialization_format": "pickle"} if "serialization_format" in __import__("inspect").signature(mlflow.pytorch.save_model).parameters else {}),
                    code_paths=[str(Path(__file__).parent)],
                )
                restored = mlflow.pytorch.load_model(str(local_model))
                with torch.inference_mode():
                    example = torch.zeros(1, 1, 128, 128, 128)
                    torch.testing.assert_close(
                        model(example), restored(example)
                    )
                del restored, example
                mlflow.log_artifacts(str(local_model), artifact_path='model')

            existing = [
                item for item in mlflow.MlflowClient().search_model_versions(
                    "name='ADHD200_ANATOMICAL'"
                )
                if item.run_id == run_id
            ]
            if not existing:
                mlflow.register_model(
                    f'runs:/{run_id}/model', 'ADHD200_ANATOMICAL'
                )

            versions=mlflow.MlflowClient().search_model_versions("name='ADHD200_ANATOMICAL'")
            version=max((int(v.version) for v in versions if v.run_id==run_id),default=None)
            for name in policy["artifacts"]:
                if (Path(output)/name).exists():mlflow.log_artifact(str(Path(output)/name))
    release=Path(output)/'release';release.mkdir(exist_ok=True,mode=0o700);shutil.copyfile(Path(output)/'best.pt',release/'best.pt')
    manifest={'schema_version':1,'architecture':checkpoint['model_name'],'parameters':checkpoint['model_parameters'],'weights_sha256':fingerprint(release/'best.pt'),'recipe':info['preprocessing'],'reference':reference,'threshold':checkpoint['threshold'],'validation':validation,'mlflow_version':version,'mlflow_run':run_id,'synthetic':False,'test_used':False,'profile':checkpoint['profile']}
    save_json(release/'release.json',manifest)
    (release/'model_card.md').write_text('# Modèle de recherche ADHD200\n\n'+json.dumps(manifest,indent=2)+'\n\nTests non utilisés. Scores non calibrés. Biais d’âge/centre et résolution physique variable. Pas d’usage clinique. Le prétraitement n’ajoute pas d’effacement du visage ; le contrôle des sources reste limité.\n')
    history=read(Path(output)/'history.json');seconds=sum(r['duration_seconds'] for r in history)
    save_json(Path(output)/'resources.json',{'seconds':seconds,'gpu_energy_estimated_kwh':seconds*100/3600000,'assumed_gpu_watts':100,'energy_measured':False})
    return str(release)

def import_model(folder):
    folder=Path(folder);manifest=read(folder/'release.json')
    if manifest['synthetic'] or recipe_hash(manifest['recipe'])!=recipe_hash() or fingerprint(folder/'best.pt')!=manifest['weights_sha256']:raise ValueError('Version non autorisée')
    identifier=fingerprint(folder/'release.json');destination=root()/'models/compact'/identifier;destination.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    if not destination.exists():
        with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
            staging=Path(directory)/'release';staging.mkdir()
            for name in ('release.json','best.pt','model_card.md'):shutil.copyfile(safe(folder,name),staging/name)
            staging.rename(destination)
    from .service import load
    load(identifier);return {'version':identifier}

def simulate(split_file,size=3,corrupt=False):
    split=read(split_file)
    if split.get('split_version')!='split_v2' or not 1<=size<=9:raise ValueError('Manifeste v2 et taille 1 à 9 requis')
    incoming=root()/'incoming';incoming.mkdir(parents=True,exist_ok=True,mode=0o700);used=set()
    for p in incoming.glob('*/manifest.json'):used.update(r['sha256'] for r in read(p)['files'])
    rows=[r for r in split['records'] if r['split']=='arrivals_reserve' and r['site']=='Brown' and r['source_sha256'] not in used][:size]
    if not rows:raise ValueError('Réserve épuisée')
    with tempfile.TemporaryDirectory(prefix='.staging_',dir=incoming) as directory:
        staging=Path(directory);records=[]
        for i,row in enumerate(rows):
            source=safe(root()/'raw',row['source_relative_path'])
            if fingerprint(source)!=row['source_sha256']:raise ValueError('Source modifiée')
            name=f'image_{i}.nii.gz';shutil.copyfile(source,staging/name);records.append({'file':name,'sha256':fingerprint(staging/name),'subject':row['pseudo_id'],'site':'Brown','cohort':'arrivals_reserve'})
        if corrupt:
            (staging/'corrupt.nii.gz').write_bytes(b'corrupt synthetic test');records.append({'file':'corrupt.nii.gz','sha256':fingerprint(staging/'corrupt.nii.gz'),'subject':'synthetic_test','site':'Brown','cohort':'synthetic'})
        save_json(staging/'manifest.json',{'schema_version':1,'files':records,'simulation':True});(staging/'READY').write_text('complete\n');identifier=fingerprint(staging/'manifest.json');staging.rename(incoming/identifier)
    return {'batch':identifier,'count':len(records),'simulation':True}

def gate(package,review):
    decision=read(review)
    if decision.get('csv_sha256')!=fingerprint(Path(package)/'dataset.csv') or not all(decision.get(k) is True for k in ('face_protection_verified','reuse_reviewed','private_destination_verified','access_retention_defined')) or not decision.get('reviewer') or not decision.get('evidence'):raise ValueError('Revue de transfert incomplète')
    return True

def kernel(destination,kernel_id,dataset):
    if '/' not in kernel_id or '/' not in dataset:raise ValueError('Identifiants propriétaire/nom requis')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True,mode=0o700);project=Path(__file__).resolve().parents[1];buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in list((project/'adhd').glob('*.py'))+list((project/'configs').glob('*.yaml')):archive.write(path,path.relative_to(project))
    payload=base64.b64encode(buffer.getvalue()).decode()
    code=f'''import pathlib,os,base64,io,zipfile,subprocess,sys
work=pathlib.Path('/kaggle/working/code');work.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))) as z:z.extractall(work)
sys.path.insert(0,str(work));os.chdir(work)
os.environ['PIPELINE_CONFIG']=str(work/'configs/pipeline.yaml')
subprocess.run([sys.executable,'-m','pip','install','mlflow','PyYAML'],check=True)
from kaggle_secrets import UserSecretsClient
os.environ['MLFLOW_TRACKING_URI']='https://dagshub.com/vroolivier0/adhd200-pipeline.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME']='vroolivier0'
os.environ['MLFLOW_TRACKING_PASSWORD']=UserSecretsClient().get_secret('DAGSHUB_TOKEN')
import torch
if not torch.cuda.is_available():raise RuntimeError('Activer CUDA Kaggle')
from adhd.learning import create_model,select_precision
for name in ('simplecnn3d','resnet3d'):
    model=create_model(name).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    precision=select_precision(torch.device('cuda'),'auto');scaler=torch.amp.GradScaler('cuda',enabled=precision==torch.float16)
    from adhd.learning import precision_context
    with precision_context(torch.device('cuda'),precision):
        logits=model(torch.rand(1,1,128,128,128,device='cuda'));loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([1.],device='cuda'))
    scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);scaler.step(optimizer);scaler.update()
    if not torch.isfinite(loss):raise RuntimeError('Échec GPU/AMP')
    del model,optimizer
files=list(pathlib.Path('/kaggle/input').rglob('dataset.csv'))
if len(files)!=1:raise RuntimeError('Un seul paquet requis')
from adhd.cli import campaign
campaign(files[0].parent,pathlib.Path('/kaggle/working/runs'))
'''
    (destination/'kernel.py').write_text(code);save_json(destination/'kernel-metadata.json',{'id':kernel_id,'title':'ADHD200 compact training','code_file':'kernel.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_internet':True,'dataset_sources':[dataset],'competition_sources':[],'kernel_sources':[]});return str(destination)

def campaign(package,output):
    from types import SimpleNamespace
    from .train import run
    import yaml
    info,_=verify(package);compatible=info.get('compatibility',{})
    if compatible.get('recipe_hash')!=recipe_hash() or compatible.get('data_code_sha256')!=fingerprint(Path(__file__).with_name('data.py')) or compatible.get('compatible') is not True:raise ValueError('Vérifier le contrat de préparation avant la campagne')
    cfg=Path(__file__).resolve().parents[1]/'configs/training.yaml';settings=yaml.safe_load(cfg.read_text());count=0
    for model in ('simplecnn3d','resnet3d'):
        for rate in settings['experiments']['initial_learning_rates']:
            count+=1
            if count>settings['experiments']['maximum_full_runs']:raise ValueError('Budget dépassé')
            run(SimpleNamespace(package=Path(package),config=cfg,model=model,output_root=Path(output),resume=None,learning_rate=rate,seed=42,inspect=False))
    return {'runs':count,'test_used':False}

def submit_retraining(snapshot_file):
    """Construire un nouveau paquet, validation inchangée, puis soumettre Kaggle."""
    snapshot=read(snapshot_file);base=safe(root(),os.environ.get('TRAINING_PACKAGE','export_kaggle/kaggle_20261005T203927Z_b2f6a09d'));info,rows=verify(base)
    protected={r['pseudo_id'] for r in read(root()/'reports/private/split_private_20261005T191224Z_fc7dc57c.json')['records']}
    if len(snapshot['rows'])<20:raise ValueError('Au moins vingt nouveaux sujets étiquetés requis')
    identifier=fingerprint(snapshot_file);folder=root()/'export_kaggle/compact_retraining'/identifier
    if not folder.exists():
        folder.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        with tempfile.TemporaryDirectory(dir=folder.parent) as directory:
            staging=Path(directory)/'package';shutil.copytree(base,staging);seen=set()
            for row in snapshot['rows']:
                if row['subject'] in protected|seen or row['site'] not in {'KKI','NYU','OHSU','PEK','Pittsburgh'} or row['target'] not in (0,1):raise ValueError('Patient protégé, répété ou invalide')
                seen.add(row['subject']);path=safe(root(),row['prepared_ref'])
                if fingerprint(path)!=row['prepared_sha']:raise ValueError('Volume changé')
                volume(path);name=f'images/image_{len(rows)+1:06d}.npz';shutil.copyfile(path,staging/name);rows.append({'image_id':Path(name).stem,'file':name,'split':'train','target':str(row['target']),'site':row['site'],'sha256':fingerprint(staging/name)})
            with (staging/'dataset.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['image_id','file','split','target','site','sha256']);writer.writeheader();writer.writerows(rows)
            classes=Counter(r['target'] for r in rows if r['split']=='train');info.update(groups=dict(Counter(r['split'] for r in rows)),dataset_csv_sha256=fingerprint(staging/'dataset.csv'),train_positive_weight_reference=classes['0']/classes['1'],train_patient_classes={'controls':classes['0'],'adhd':classes['1']})
            save_json(staging/'dataset_info.json',info);verify(staging);staging.rename(folder)
    review=root()/'reference/compact/transfer_reviews'/(fingerprint(folder/'dataset.csv')+'.json');gate(folder,review)
    os.environ['KAGGLE_USERNAME']=secret('KAGGLE_USERNAME');os.environ['KAGGLE_KEY']=secret('KAGGLE_KEY');slug=os.environ['KAGGLE_DATASET'];kernel_id=os.environ['KAGGLE_KERNEL']
    if slug.split('/')[0]!=os.environ['KAGGLE_USERNAME'] or kernel_id.split('/')[0]!=os.environ['KAGGLE_USERNAME']:raise ValueError('Compte Kaggle différent')
    with tempfile.TemporaryDirectory() as directory:
        upload=Path(directory)/'package';shutil.copytree(folder,upload);save_json(upload/'dataset-metadata.json',{'id':slug,'title':'ADHD200 private training','licenses':[{'name':os.environ.get('KAGGLE_LICENSE','other')}]})
        subprocess.run(['kaggle','datasets','version','-p',str(upload),'-m',identifier,'--dir-mode','zip'],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    code=kernel(root()/'reference/compact/kernels'/identifier,kernel_id,slug);subprocess.run(['kaggle','kernels','push','-p',code],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    from .utils import event
    event('retraining',{'snapshot':str(Path(snapshot_file).relative_to(root())),'submitted':True},'submitted_'+identifier);return {'submitted':True,'kernel':kernel_id}

def holdout(report_file, processed_root, release_folder, output):
    import torch
    import nibabel as nib
    from .learning import create_model, load_checkpoint, grouped_metrics
    report=read(report_file);manifest=read(Path(release_folder)/'release.json')
    if report.get('summary',{}).get('failed')!=0 or recipe_hash(report['configuration']['preprocessing'])!=recipe_hash(manifest['recipe']):raise ValueError('Préparation incompatible')
    base=safe(processed_root,report['output_directory']);model=create_model(manifest['architecture'],**manifest['parameters']).eval();weights=load_checkpoint(Path(release_folder)/'best.pt')
    if fingerprint(Path(release_folder)/'best.pt')!=manifest['weights_sha256']:raise ValueError('Poids modifiés')
    model.load_state_dict(weights['model_state']);torch.set_num_threads(2);groups={};seen=set()
    for row in report['prepared_records']:
        if row['split'] not in {'test_internal','test_external','controls_external'}:continue
        if row['pseudo_id'] in seen:raise ValueError('Sujet répété dans le test')
        seen.add(row['pseudo_id']);path=safe(base,row['prepared_relative_path'])
        if fingerprint(path)!=row['prepared_sha256']:raise ValueError('Volume modifié')
        array=nib.load(path).get_fdata(dtype=np.float32)
        if array.shape!=(128,128,128) or not np.isfinite(array).all():raise ValueError('Volume invalide')
        with torch.inference_mode():score=float(torch.sigmoid(model(torch.from_numpy(array).unsqueeze(0).unsqueeze(0)))[0])
        groups.setdefault(row['split'],[]).append((row['target_binary'],score,row['site']))
    if not groups:raise ValueError('Aucun groupe de test')
    results={name:grouped_metrics([r[0] for r in rows],[r[1] for r in rows],[r[2] for r in rows],manifest['threshold']) for name,rows in groups.items()}
    save_json(output,{'model_sha256':manifest['weights_sha256'],'preparation_report_sha256':fingerprint(report_file),'groups':results,'model_selection_performed':False});return {'report':str(output),'groups':list(groups)}

def benchmark(ids,repeats):
    from .utils import http
    import time
    if not 1<=len(ids)<=4 or not 1<=repeats<=100:raise ValueError('Charge bornée')
    durations=[]
    for _ in range(repeats):
        start=time.monotonic();http('/predict-batch',{'ids':ids});durations.append(time.monotonic()-start)
    report={'requests':repeats,'images_per_request':len(ids),'median_seconds':float(np.median(durations)),'p95_seconds':float(np.percentile(durations,95)),'independent_patients_added':0}
    save_json(root()/'reports/compact/benchmark.json',report);return report

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('db-init');sub.add_parser('worker')
    p=sub.add_parser('verify');p.add_argument('package',type=Path)
    p=sub.add_parser('compatibility');p.add_argument('package',type=Path);p.add_argument('split',type=Path);p.add_argument('export_report',type=Path)
    p=sub.add_parser('simulate');p.add_argument('split',type=Path);p.add_argument('--size',type=int,default=3);p.add_argument('--corrupt',action='store_true')
    p=sub.add_parser('import-model');p.add_argument('folder',type=Path)
    p=sub.add_parser('kernel');p.add_argument('output',type=Path);p.add_argument('--id',required=True);p.add_argument('--dataset',required=True)
    p=sub.add_parser('campaign');p.add_argument('package',type=Path);p.add_argument('output',type=Path)
    p=sub.add_parser('retrain');p.add_argument('snapshot',type=Path)
    p=sub.add_parser('transfer-check');p.add_argument('package',type=Path);p.add_argument('review',type=Path)
    p=sub.add_parser('holdout');p.add_argument('report',type=Path);p.add_argument('processed_root',type=Path);p.add_argument('release',type=Path);p.add_argument('output',type=Path)
    p=sub.add_parser('benchmark');p.add_argument('ids',nargs='+');p.add_argument('--repeats',type=int,default=10)
    args=parser.parse_args()
    if args.command=='db-init':
        from .utils import db
        with db() as conn:conn.execute((Path(__file__).resolve().parents[1]/'sql.sql').read_text())
        result={'database_initialized':True}
    elif args.command=='worker':
        from .flow import worker
        return worker()
    elif args.command=='verify':result={'groups':verify(args.package)[0]['groups']}
    elif args.command=='compatibility':result=compatibility(args.package,args.split,args.export_report)
    elif args.command=='simulate':result=simulate(args.split,args.size,args.corrupt)
    elif args.command=='import-model':result=import_model(args.folder)
    elif args.command=='kernel':result={'directory':kernel(args.output,args.id,args.dataset)}
    elif args.command=='campaign':result=campaign(args.package,args.output)
    elif args.command=='retrain':result=submit_retraining(args.snapshot)
    elif args.command=='holdout':result=holdout(args.report,args.processed_root,args.release,args.output)
    elif args.command=='benchmark':result=benchmark(args.ids,args.repeats)
    else:result={'reviewed':gate(args.package,args.review)}
    print(json.dumps(result,allow_nan=False))
if __name__=='__main__':main()
