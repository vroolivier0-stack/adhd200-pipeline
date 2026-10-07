"""Installer après contrôles, archiver l'ancien code, conserver les données."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
def compose(project,*args):return subprocess.run(['docker','compose','--project-directory',str(project),'-f',str(project/'compose.yaml'),*args],check=True)
def write_private(path,text,mode=0o600):
    if path.is_symlink():raise ValueError('Lien non autorisé')
    if path.exists():return
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700);fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,mode)
    with os.fdopen(fd,'w') as stream:stream.write(text)
def prepare(project,data):
    data=Path(data).resolve();runtime=project/'.runtime';persistent=data/'secrets/compact_runtime'
    if data.is_symlink() or persistent.is_symlink():raise ValueError('Lien refusé')
    for name in ('raw','reports/private','reports/compact','historical','export_kaggle','incoming','arrivals','quarantine','reference','models','logs'):
        path=data/name
        if path.is_symlink():raise ValueError('Dossier symbolique refusé')
        path.mkdir(parents=True,exist_ok=True,mode=0o700)
    persistent.mkdir(parents=True,exist_ok=True,mode=0o700);runtime.mkdir(exist_ok=True,mode=0o700)
    # Clé de pseudonymisation : créée UNIQUEMENT si elle n'existe pas (environnement neuf, tests).
    # Une clé existante n'est jamais remplacée : les codes patients en dépendent.
    key=data/'secrets/pseudonymization.key'
    if key.is_symlink():raise ValueError('Lien refusé')
    if not key.exists():
        descriptor=os.open(key,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'wb') as stream:stream.write(secrets.token_bytes(32))
    if persistent.stat().st_mode&0o077 or runtime.is_symlink():raise ValueError('Secrets non privés')
    for name in ('postgres_password','app_password','airflow_password','api_key','admin_key','airflow_jwt'):write_private(persistent/name,secrets.token_hex(32)+'\n')
    app=(persistent/'app_password').read_text().strip();airflow=(persistent/'airflow_password').read_text().strip()
    write_private(persistent/'app_dsn',f'postgresql://compact_app:{app}@postgres/predictions\n');write_private(persistent/'test_dsn',f'postgresql://compact_app:{app}@postgres/predictions_test\n');write_private(persistent/'airflow_dsn',f'postgresql+psycopg2://compact_airflow:{airflow}@postgres/orchestration\n')
    write_private(persistent/'airflow_users.json',json.dumps({'jury':secrets.token_urlsafe(24)}))
    for name in ('mlflow_password','kaggle_username','kaggle_key'):write_private(persistent/name,'')
    sql=f"CREATE USER compact_app PASSWORD '{app}';\nCREATE USER compact_airflow PASSWORD '{airflow}';\nCREATE DATABASE predictions OWNER compact_app;\nCREATE DATABASE predictions_test OWNER compact_app;\nCREATE DATABASE orchestration OWNER compact_airflow;\nREVOKE CONNECT ON DATABASE predictions FROM PUBLIC;\nREVOKE CONNECT ON DATABASE predictions_test FROM PUBLIC;\nREVOKE CONNECT ON DATABASE orchestration FROM PUBLIC;\nGRANT CONNECT ON DATABASE predictions TO compact_app;\nGRANT CONNECT ON DATABASE predictions_test TO compact_app;\nGRANT CONNECT ON DATABASE orchestration TO compact_airflow;\n"
    write_private(persistent/'init.sql',sql)
    for path in persistent.iterdir():
        if path.is_file() and not path.is_symlink():
            target=runtime/path.name
            if target.exists():target.unlink()
            shutil.copyfile(path,target);target.chmod(0o644 if path.name=='init.sql' else 0o600)
    write_private(project/'.env',f'ADHD_DATA={data}\nPIPELINE_UID={os.getuid()}\nPIPELINE_GID={os.getgid()}\n')
    (runtime/'testdata').mkdir(exist_ok=True,mode=0o700)
def install(target,data):
    target=Path(target)
    if os.getuid()==0:raise ValueError('Utilisateur WSL habituel, sans sudo')
    if target.is_symlink() or target.resolve()==HERE:raise ValueError('Extraire dans un dossier distinct')
    for path in HERE.rglob('*'):
        if path.is_symlink():raise ValueError('Lien dans la livraison')
        if path.suffix=='.py':compile(path.read_text(),str(path),'exec')
    archive=None
    with tempfile.TemporaryDirectory(prefix='.adhd_compact_',dir=target.parent) as directory:
        candidate=Path(directory)/'project';shutil.copytree(HERE,candidate,ignore=shutil.ignore_patterns('.runtime','.env','__pycache__','.git'))
        prepare(candidate,data);compose(candidate,'config','--quiet');compose(candidate,'build','api');compose(candidate,'up','-d','postgres');compose(candidate,'--profile','test','run','--rm','tests');compose(candidate,'--profile','test','run','--rm','dag-check')
        # L'ancien projet est remplacé seulement après les tests CPU+API+DB.
        if target.exists():
            if (target/'.git').is_dir():shutil.copytree(target/'.git',candidate/'.git')
            elif (target/'.git').exists():raise ValueError('Worktree externe non pris en charge')
            archive=target.parent/(target.name+'_ARCHIVE_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));target.rename(archive);archive.chmod(0o700)
        try:candidate.rename(target)
        except BaseException:
            if archive:archive.rename(target)
            raise
    if (target/'.git').is_dir():
        subprocess.run(['git','read-tree','--empty'],cwd=target,check=True);subprocess.run(['git','add','-A'],cwd=target,check=True)
    compose(target,'up','-d');compose(target,'ps')
    print('Ancien projet archivé :',archive);print('API http://localhost:8000/docs | Streamlit http://localhost:8501 | Airflow http://localhost:8080')
    print('Services lancés ; modèle réel absent avant Kaggle. Aucun transfert ni push réalisé.')
def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('install');p.add_argument('--target',type=Path,default=Path.home()/'ADHD200_PIPELINE');p.add_argument('--data',type=Path,default=Path.home()/'ADHD200_DATA')
    p=sub.add_parser('prepare');p.add_argument('--data',type=Path,required=True)
    p=sub.add_parser('cli');p.add_argument('arguments',nargs=argparse.REMAINDER)
    sub.add_parser('test');sub.add_parser('status');sub.add_parser('airflow-password')
    args=parser.parse_args()
    if args.command=='install':install(args.target,args.data)
    elif args.command=='prepare':prepare(HERE,args.data)
    elif args.command=='cli':compose(HERE,'run','--rm','--no-deps','worker','python','-m','adhd.cli',*args.arguments)
    elif args.command=='test':compose(HERE,'--profile','test','run','--rm','tests')
    elif args.command=='airflow-password':print((HERE/'.runtime/airflow_users.json').read_text())
    else:compose(HERE,'ps')
if __name__=='__main__':main()
