"""DAGs planifiés : références seulement dans XCom, pas de volumes."""
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from airflow.sdk import DAG,task,get_current_context

def call(path,body=None):
    data=json.dumps(body).encode() if body is not None else None
    request=urllib.request.Request('http://api:8000'+path,data=data,headers={'Content-Type':'application/json','X-API-Key':Path('/run/secrets/admin_key').read_text().strip()})
    with urllib.request.urlopen(request,timeout=30) as response:return json.load(response)
def execute(kind,payload):
    key=hashlib.sha256((get_current_context()['run_id']+kind).encode()).hexdigest();job=call('/jobs',{'key':key,'kind':kind,'payload':payload});deadline=time.monotonic()+1800
    while time.monotonic()<deadline:
        state=call('/jobs/'+job['id'])
        if state['status']=='succeeded':return state['result']
        if state['status']=='failed':raise RuntimeError('Tâche en échec '+job['id'])
        time.sleep(5)
    raise TimeoutError('Délai dépassé')
def failure(context):
    import logging
    logging.error('pipeline_failure dag=%s task=%s',context['dag'].dag_id,context['task_instance'].task_id)
    try:call('/watchdog',{})
    except Exception:logging.error('monitoring_unavailable')
arguments={'retries':2,'retry_delay':timedelta(seconds=30),'execution_timeout':timedelta(minutes=35),'on_failure_callback':failure}
with DAG('adhd_arrivals',schedule='*/5 * * * *',start_date=datetime(2026,1,1,tzinfo=timezone.utc),catchup=False,max_active_runs=1,default_args=arguments) as arrivals:
    @task
    def collect():return execute('collect',{})['batches']
    @task
    def prepare(batches):return execute('prepare',{'batches':batches})['batches']
    @task
    def predict(batches):return execute('predict',{'batches':batches})
    predict(prepare(collect()))
with DAG('adhd_monitoring',schedule='*/5 * * * *',start_date=datetime(2026,1,1,tzinfo=timezone.utc),catchup=False,max_active_runs=1,default_args=arguments) as monitoring:
    @task
    def monitor():
        call('/watchdog',{})
        return execute('monitor',{})
    monitor()
