"""Intégration réelle CPU/API/PostgreSQL avec données fictives isolées."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from fastapi.testclient import TestClient
from psycopg.types.json import Jsonb
from adhd.utils import root,db,save_json,secret,fingerprint,recipe_hash
from adhd.learning import create_model,binary_metrics,train_epoch,evaluate_epoch,save_checkpoint,load_checkpoint
from adhd.flow import collect,prepare
from adhd.service import app,route,deploy,Deploy

class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        with db() as conn:
            if conn.execute('SELECT current_database() AS name').fetchone()['name']!='predictions_test':raise RuntimeError('Base de test isolée obligatoire')
            conn.execute((Path(__file__).resolve().parents[1]/'sql.sql').read_text())
    def setUp(self):
        with db() as conn:conn.execute('TRUNCATE batches,items,predictions,labels,jobs,events CASCADE')
        (root()/'models/compact/deployment.json').unlink(missing_ok=True)
        self.client=TestClient(app);self.reader={'X-API-Key':secret('API_KEY')};self.admin={'X-API-Key':secret('ADMIN_KEY')}
    def test_metrics_and_architectures(self):
        self.assertEqual(binary_metrics([0,1],[0.5,0.5])['roc_auc'],0.5)
        self.assertIsNone(binary_metrics([0,0],[0.2,0.8])['roc_auc'])
        self.assertAlmostEqual(binary_metrics([0,1,0,1],[0.1,0.8,0.6,0.4])['average_precision'],5/6)
        import io
        for name in ('simplecnn3d','resnet3d'):
            model=create_model(name).eval()
            with torch.inference_mode():result=model(torch.zeros(1,1,128,128,128))
            self.assertEqual(tuple(result.shape),(1,));self.assertTrue(torch.isfinite(result).all())
            buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0);restored=create_model(name).eval();restored.load_state_dict(torch.load(buffer,weights_only=True))
            with torch.inference_mode():self.assertTrue(torch.allclose(result,restored(torch.zeros(1,1,128,128,128))))
    def test_accumulation_and_checkpoint(self):
        class Tiny(nn.Module):
            def __init__(self):super().__init__();self.linear=nn.Linear(4,1)
            def forward(self,x):return self.linear(x).squeeze(1)
        torch.manual_seed(42);rows=[{'image':torch.randn(4),'target':torch.tensor(float(i%2)),'image_id':str(i),'site':'test'} for i in range(5)];model=Tiny();reference=copy.deepcopy(model);loss=nn.BCEWithLogitsLoss(reduction='sum');opt=torch.optim.SGD(model.parameters(),lr=.01);other=torch.optim.SGD(reference.parameters(),lr=.01)
        (loss(reference(torch.stack([r['image'] for r in rows])),torch.stack([r['target'] for r in rows]))/5).backward();other.step()
        result=train_epoch(model,DataLoader(rows,batch_size=2),opt,loss,torch.device('cpu'),torch.amp.GradScaler('cuda',enabled=False),4,1000.)
        self.assertEqual(result['optimizer_updates'],1)
        for a,b in zip(model.parameters(),reference.parameters()):self.assertTrue(torch.allclose(a,b,atol=1e-7))
        before=copy.deepcopy(model.state_dict());evaluation=evaluate_epoch(model,DataLoader(rows,batch_size=2),loss,torch.device('cpu'));self.assertEqual(len(evaluation['predictions']),5)
        for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,before[key]))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'save.pt';save_checkpoint(path,{'schema_version':1,'weights':before,'optimizer':opt.state_dict()});self.assertEqual(load_checkpoint(path)['schema_version'],1)
    def test_valid_and_corrupt_volume_continue(self):
        incoming=root()/'incoming'
        if incoming.exists():shutil.rmtree(incoming)
        folder=incoming/'test';folder.mkdir(parents=True);array=np.zeros((24,24,24),np.float32);array[3:20,3:20,3:20]=np.linspace(.1,1,17**3).reshape(17,17,17)
        image=nib.Nifti1Image(array,np.eye(4));image.header.set_xyzt_units('mm');nib.save(image,folder/'good.nii.gz');(folder/'bad.nii.gz').write_bytes(b'corrupt')
        records=[{'file':name,'sha256':fingerprint(folder/name),'subject':'test_'+name,'site':'Brown','cohort':'synthetic'} for name in ('good.nii.gz','bad.nii.gz')]
        save_json(folder/'manifest.json',{'schema_version':1,'files':records});(folder/'READY').write_text('ready')
        first=collect();self.assertEqual(first,collect())
        with db() as conn:self.assertEqual(conn.execute('SELECT count(*) AS n FROM items').fetchone()['n'],2)
        result=prepare(first['batches']);self.assertEqual(result['prepared'],1);self.assertEqual(result['quarantined'],1);self.assertEqual(prepare(first['batches'])['prepared'],0)
        self.assertTrue((folder/'good.nii.gz').exists())
    def test_collect_pseudonymizes_patient_identifier(self):
        from adhd.identity import pseudonymize,SITE_CODES
        incoming=root()/'incoming'
        if incoming.exists():shutil.rmtree(incoming)
        key=bytes(range(32));array=np.zeros((24,24,24),np.float32);array[3:20,3:20,3:20]=np.linspace(.1,1,17**3).reshape(17,17,17)
        try:
            with tempfile.TemporaryDirectory() as directory:
                key_file=Path(directory)/'pseudonymization.key';key_file.write_bytes(key);key_file.chmod(0o600)
                # Lot 1 : numéro de dossier fourni, le pipeline doit le brouiller. Lot 2 : identifiant en clair, à refuser.
                for name,scale,identity in (('with_patient_id',1.0,{'patient_id':'0026001'}),('clear_identifier',0.5,{'subject':'0026002'})):
                    folder=incoming/name;folder.mkdir(parents=True);image=nib.Nifti1Image(array*scale,np.eye(4));image.header.set_xyzt_units('mm');nib.save(image,folder/'scan.nii.gz')
                    save_json(folder/'manifest.json',{'schema_version':1,'files':[{'file':'scan.nii.gz','sha256':fingerprint(folder/'scan.nii.gz'),'site':'Brown','cohort':'arrivals_reserve',**identity}]});(folder/'READY').write_text('ready')
                with patch.dict(os.environ,{'PSEUDONYMIZATION_KEY_FILE':str(key_file)}):result=collect()
            self.assertEqual(len(result['batches']),1)
            with db() as conn:
                subjects=[r['subject'] for r in conn.execute('SELECT subject FROM items').fetchall()]
                stored=json.dumps(conn.execute('SELECT manifest FROM batches').fetchone()['manifest'])
            self.assertEqual(subjects,[pseudonymize(SITE_CODES['Brown'],'26001',key)])
            self.assertNotIn('0026001',stored);self.assertNotIn('patient_id',stored);self.assertNotIn('0026002',stored)
        finally:shutil.rmtree(incoming,ignore_errors=True)
    def test_api_auth_queue_and_no_fake_model(self):
        self.assertEqual(self.client.get('/health').status_code,200);self.assertEqual(self.client.get('/summary').status_code,401);self.assertEqual(self.client.get('/ready').status_code,503)
        payload={'key':'test','kind':'collect'};self.assertEqual(self.client.post('/jobs',headers=self.reader,json=payload).status_code,401)
        self.assertEqual(self.client.post('/jobs',headers=self.admin,json=payload).json(),self.client.post('/jobs',headers=self.admin,json=payload).json())
        self.assertEqual(self.client.post('/predict',headers=self.reader,json={'ids':['../escape']}).status_code,422)
    def test_canary_and_promotion_refusal(self):
        current={'champion':'a'*64,'challenger':'b'*64,'fraction':0.1};results=[route(str(i),current)[1] for i in range(1000)]
        self.assertEqual(results,[route(str(i),current)[1] for i in range(1000)]);self.assertTrue(60<=results.count('canary')<=140)
        with patch('adhd.service.load',return_value=(None,{'validation':{'roc_auc':0.5,'balanced_accuracy':.5}})):
            self.assertEqual(self.client.post('/deployment',headers=self.admin,json={'action':'candidate','version':'a'*64}).status_code,409)
    def test_actual_prediction_and_explanation_synthetic_data(self):
        path=root()/'arrivals/test.npz';path.parent.mkdir(parents=True,exist_ok=True);array=np.zeros((128,128,128),np.float32);array[10:70]=.6;np.savez_compressed(path,image=array)
        with db() as conn:
            conn.execute('INSERT INTO batches(id,manifest) VALUES (%s,%s)',('b'*64,Jsonb({})))
            conn.execute("INSERT INTO items(id,batch_id,subject,site,cohort,raw_ref,prepared_ref,prepared_sha,recipe,status) VALUES (%s,%s,'test','Brown','synthetic','none',%s,%s,%s,'prepared')",('a'*64,'b'*64,path.relative_to(root()).as_posix(),fingerprint(path),recipe_hash()))
        class Toy(nn.Module):
            def forward(self,x):return x.mean((1,2,3,4))
        with patch('adhd.service.state',return_value={'champion':'f'*64,'challenger':None,'fraction':0}),patch('adhd.service.load',return_value=(Toy(),{'threshold':.5})):
            for _ in range(2):self.assertEqual(self.client.post('/predict',headers=self.reader,json={'ids':['a'*64]}).status_code,200)
            self.assertEqual(self.client.post('/explain',headers=self.reader,json={'ids':['a'*64]}).status_code,200)
        with db() as conn:self.assertEqual(conn.execute('SELECT count(*) AS n FROM predictions').fetchone()['n'],1)
        self.assertEqual(self.client.post('/labels',headers=self.admin,json={'item_id':'a'*64,'target':1,'purpose':'new_training','source':'synthetic not allowed'}).status_code,409)

    def test_watchdog_without_worker(self):
        with db() as conn:conn.execute("INSERT INTO batches(id,manifest,created_at) VALUES (%s,%s,now()-interval '1 hour')",('e'*64,Jsonb({})))
        self.assertEqual(self.client.post('/watchdog',headers=self.admin,json={}).status_code,200)
        with db() as conn:
            alerts=conn.execute("SELECT body FROM events WHERE kind='alert'").fetchall()
        self.assertEqual(alerts[0]['body']['code'],'batch_delay')

    def test_candidate_rejects_constant_predictions(self):
        from adhd.service import validate_candidate
        policy = {"minimum_auc": .55, "minimum_balanced_accuracy": .50}
        valid = {
            "roc_auc": .70, "balanced_accuracy": .60,
            "confusion": {
                "true_negatives": 38, "false_positives": 33,
                "false_negatives": 16, "true_positives": 32,
            },
        }
        validate_candidate(valid, policy)
        for confusion in (
            {"true_negatives": 71, "false_positives": 0,
             "false_negatives": 48, "true_positives": 0},
            {"true_negatives": 0, "false_positives": 71,
             "false_negatives": 0, "true_positives": 48},
        ):
            rejected = dict(valid, confusion=confusion)
            with self.assertRaises(ValueError):
                validate_candidate(rejected, policy)
        with self.assertRaises(ValueError):
            validate_candidate(dict(valid, roc_auc=float("nan")), policy)
