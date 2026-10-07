"""Tests du contrat distant et de la disponibilité du cache, sans réseau."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from adhd.registry_deployment import validate_manifest, check_registry
from adhd.utils import save_json

class RegistryDeploymentTests(unittest.TestCase):
    def test_manifest_must_match_registered_version(self):
        version = SimpleNamespace(version='3', run_id='run3', status='READY')
        manifest = {'schema_version':1,'mlflow_version':3,'mlflow_run':'run3','synthetic':False}
        validate_manifest(manifest, version)
        for change in ({'mlflow_version':4}, {'mlflow_run':'other'}, {'synthetic':True}):
            with self.assertRaises(ValueError):
                validate_manifest(dict(manifest, **change), version)
        with self.assertRaises(ValueError):
            validate_manifest(manifest, SimpleNamespace(version='3',run_id='run3',status='PENDING_REGISTRATION'))

    def test_unavailable_registry_keeps_local_deployment(self):
        original = {'champion':'local3','catalog_challenger':'local4','registry_versions':{'champion':'3','challenger':'4'}}
        with tempfile.TemporaryDirectory() as directory:
            with patch('adhd.service.state', return_value=original), patch('adhd.registry_deployment.root', return_value=Path(directory)), patch('adhd.registry_deployment.client', side_effect=RuntimeError('offline')), patch('adhd.registry_deployment.event'):
                result = check_registry()
            self.assertEqual(result['status'], 'registry_unavailable')
            self.assertEqual(original['champion'], 'local3')
            self.assertFalse((Path(directory)/'models/compact/deployment.json').exists())

    def test_alignment_and_mismatch_are_distinct(self):
        state = {'champion':'local3','catalog_challenger':'local4','registry_versions':{'champion':'3','challenger':'4'}}
        wanted = {'champion':SimpleNamespace(version='3',run_id='run3'), 'challenger':SimpleNamespace(version='4',run_id='run4')}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for identifier, version in (('local3','3'),('local4','4')):
                save_json(base/'models/compact'/identifier/'release.json', {'mlflow_version':version,'mlflow_run':'run'+version})
            with patch('adhd.service.state',return_value=state), patch('adhd.registry_deployment.root',return_value=base), patch('adhd.registry_deployment.client'), patch('adhd.registry_deployment.desired_versions',return_value=wanted), patch('adhd.registry_deployment.event'):
                self.assertEqual(check_registry()['status'], 'aligned')
                wanted['champion'] = SimpleNamespace(version='4',run_id='run4')
                self.assertEqual(check_registry()['status'], 'mismatch')
            self.assertEqual(state['champion'],'local3')
