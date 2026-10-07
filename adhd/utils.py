"""Entrées/sorties atomiques, secrets externes et base applicative."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid

def root(): return Path(os.environ.get('DATA_ROOT','/data'))
def fingerprint(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def safe(base,name):
    base=Path(base).resolve(strict=True);relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:raise ValueError('Chemin refusé')
    path=base
    for part in relative.parts:
        path=path/part
        if path.is_symlink():raise ValueError('Lien refusé')
    if not path.resolve().is_relative_to(base):raise ValueError('Chemin hors périmètre')
    return path

def save_json(path,value):
    path=Path(path);payload=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    if path.is_symlink():raise ValueError('Lien refusé')
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    temporary=path.parent/(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'wb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)
def read(path):return json.loads(Path(path).read_text())
def settings():
    import yaml
    return yaml.safe_load(Path(os.environ.get('SERVICE_CONFIG','/app/configs/service.yaml')).read_text())
def recipe():
    import yaml
    return yaml.safe_load(Path(os.environ.get('PIPELINE_CONFIG','/app/configs/pipeline.yaml')).read_text())['preprocessing']
def recipe_hash(value=None):
    value=value or recipe();keys=('recipe_version','orientation','intermediate_spacing_mm','output_edge','intensity_percentiles')
    return hashlib.sha256(json.dumps({k:value[k] for k in keys},sort_keys=True).encode()).hexdigest()
def secret(name):
    value=Path(os.environ.get(name+'_FILE','/run/secrets/'+name.lower())).read_text().strip()
    if not value:raise RuntimeError('Secret absent: '+name)
    return value
@contextmanager
def db():
    import psycopg
    from psycopg.rows import dict_row
    with psycopg.connect(secret('APP_DSN'),connect_timeout=10,row_factory=dict_row) as conn:yield conn

def event(kind,body,key=None):
    from psycopg.types.json import Jsonb
    key=key or uuid.uuid4().hex
    with db() as conn:conn.execute('INSERT INTO events(id,kind,body) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING',(key,kind,Jsonb(body)))
    folder=root()/'logs/compact';folder.mkdir(parents=True,exist_ok=True,mode=0o700)
    value={'time':datetime.now(timezone.utc).isoformat(),'id':key,'kind':kind,'body':body}
    with (folder/'events.jsonl').open('a') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n')
    print(json.dumps(value,allow_nan=False),flush=True)

def http(path,payload=None,admin=False):
    key=secret('ADMIN_KEY' if admin else 'API_KEY');data=json.dumps(payload).encode() if payload is not None else None
    url=os.environ.get('API_URL','http://api:8000')+path
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json','X-API-Key':key})
            with urllib.request.urlopen(req,timeout=30) as response:return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code<500 and error.code!=429 or attempt==2:raise
        except (urllib.error.URLError,TimeoutError):
            if attempt==2:raise
        time.sleep(2**attempt)

def volume(path):
    import numpy as np
    import zipfile
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist())!=1 or archive.infolist()[0].file_size>9*1024**2:raise ValueError('NPZ non conforme')
    with np.load(path,allow_pickle=False) as archive:
        if archive.files!=['image']:raise ValueError('Tableau unique attendu')
        array=archive['image']
    if array.shape!=(128,128,128) or array.dtype!=np.float32 or not np.isfinite(array).all() or array.min()<-1e-5 or array.max()>1.00001 or array.max()<=array.min():raise ValueError('Volume non conforme')
    return array
