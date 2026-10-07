import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import numpy as np
from adhd.utils import safe,save_json,fingerprint,volume,http
from adhd.cli import verify,kernel,gate

class CoreTests(unittest.TestCase):
    def test_atomic_io_and_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);save_json(base/'value.json',{'complete':True})
            self.assertEqual(json.loads((base/'value.json').read_text()),{'complete':True})
            self.assertEqual((base/'value.json').stat().st_mode&0o777,0o600)
            for name in ('../escape','/absolute'):
                with self.assertRaises(ValueError):safe(base,name)
            (base/'link').symlink_to(base/'value.json')
            with self.assertRaises(ValueError):safe(base,'link')
            with self.assertRaises(ValueError):save_json(base/'bad.json',{'bad':float('nan')})
            self.assertFalse((base/'bad.json').exists())
    def test_volume_rejects_uniform_extra_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'image.npz';array=np.zeros((128,128,128),np.float32);array[20:50]=0.8
            np.savez_compressed(path,image=array);self.assertTrue(np.array_equal(volume(path),array))
            np.savez_compressed(path,image=np.ones_like(array))
            with self.assertRaises(ValueError):volume(path)
            np.savez_compressed(path,image=array,extra=np.array([1]))
            with self.assertRaises(ValueError):volume(path)
    def test_retry_only_transient(self):
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return b'{"ok":true}'
        error=urllib.error.HTTPError('http://example',503,'down',{},None)
        with patch('adhd.utils.secret',return_value='test'),patch('adhd.utils.time.sleep'),patch('adhd.utils.urllib.request.urlopen',side_effect=[error,Response()]) as call:
            self.assertTrue(http('/test')['ok']);self.assertEqual(call.call_count,2)
        error=urllib.error.HTTPError('http://example',401,'forbidden',{},None)
        with patch('adhd.utils.secret',return_value='test'),patch('adhd.utils.urllib.request.urlopen',side_effect=error) as call:
            with self.assertRaises(urllib.error.HTTPError):http('/test')
            self.assertEqual(call.call_count,1)
    def test_package_numeric_duplicate_cross_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            package=Path(directory);(package/'images').mkdir();rows=[]
            for i,(split,target) in enumerate([('train','0'),('train','1'),('validation','0'),('validation','1')]):
                array=np.zeros((128,128,128),np.float32);array[20:50]=(i+1)/5
                name=f'images/image_{i}.npz';np.savez_compressed(package/name,image=array)
                rows.append({'image_id':str(i),'file':name,'split':split,'target':target,'site':'KKI','sha256':fingerprint(package/name)})
            def publish():
                with (package/'dataset.csv').open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                save_json(package/'dataset_info.json',{'contains_test_data':False,'dataset_csv_sha256':fingerprint(package/'dataset.csv'),'groups':{'train':2,'validation':2}})
            (package/'LOCAL_EXPORT_COMPLETE.txt').write_text('complete');(package/'README.txt').write_text('test');publish()
            self.assertEqual(len(verify(package)[1]),4)
            import shutil
            shutil.copyfile(package/rows[0]['file'],package/rows[2]['file']);rows[2]['sha256']=fingerprint(package/rows[2]['file']);publish()
            with self.assertRaises(ValueError):verify(package)
    def test_kernel_self_contained_and_transfer_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(kernel(directory,'owner/kernel','owner/dataset'));source=(target/'kernel.py').read_text();compile(source,'kernel.py','exec')
            import ast,base64,io,zipfile
            payload=max((node.value for node in ast.walk(ast.parse(source)) if isinstance(node,ast.Constant) and isinstance(node.value,str)),key=len)
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
                self.assertIn('adhd/train.py',archive.namelist());self.assertIn('adhd/cli.py',archive.namelist())
                for name in archive.namelist():
                    if name.endswith('.py'):compile(archive.read(name),name,'exec')
            (target/'dataset.csv').write_text('test');save_json(target/'review.json',{'csv_sha256':fingerprint(target/'dataset.csv')})
            with self.assertRaises(ValueError):gate(target,target/'review.json')
    def test_identity_is_stable_keyed_and_refuses_clear_identifiers(self):
        from adhd.identity import pseudonymize,subject_for,patient_from_path,SITE_CODES
        key=bytes(range(32));other=bytes(range(1,33));code=pseudonymize(SITE_CODES['Brown'],'0026001',key)
        self.assertEqual(code,pseudonymize(SITE_CODES['Brown'],'26001',key))          # zéros initiaux
        self.assertNotEqual(code,pseudonymize(SITE_CODES['Brown'],'0026001',other))   # autre clé
        self.assertNotEqual(code,pseudonymize(SITE_CODES['KKI'],'0026001',key))       # autre hôpital
        self.assertRegex(code,'^sub_[0-9a-f]{32}$');self.assertNotIn('26001',code)
        self.assertEqual(patient_from_path('Brown/0026001/session_1/anat_1/mprage.nii.gz'),'0026001')
        self.assertEqual(subject_for({'patient_id':'0026001','site':'Brown','cohort':'arrivals_reserve'},lambda:key),code)
        self.assertEqual(subject_for({'subject':code,'site':'Brown','cohort':'arrivals_reserve'},lambda:key),code)
        with self.assertRaises(ValueError):subject_for({'subject':'0026001','site':'Brown','cohort':'arrivals_reserve'},lambda:key)
        with self.assertRaises(ValueError):subject_for({'subject':code,'patient_id':'1','site':'Brown','cohort':'arrivals_reserve'},lambda:key)
    def test_cost_estimate_uses_measured_durations(self):
        from adhd.costs import estimate
        rates={'local_power_watts':100,'electricity_eur_per_kwh':0.25,'grid_gco2_per_kwh':60,'cloud_cpu_eur_per_hour':0.10,'cloud_gpu_eur_per_hour':0.60,'storage_eur_per_gb_month':0.02,'gpu_hours_training':2}
        result=estimate([{'kind':'prepare','runs':2,'seconds':3600,'peak_mib':500},{'kind':'predict','runs':1,'seconds':3600,'peak_mib':700}],rates,{'raw':10.0})
        self.assertEqual(result['compute_hours'],2.0);self.assertEqual(result['local_energy_kwh'],0.2);self.assertEqual(result['local_cost_eur'],0.05)
        self.assertEqual(result['tasks'][0]['seconds_mean'],1800.0);self.assertEqual(result['training_cloud_equivalent_eur'],1.2);self.assertEqual(result['storage_cloud_equivalent_eur_per_month'],0.2)

