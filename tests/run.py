"""Tous les tests requis doivent réussir ; aucun test Docker sauté."""
import json
import os
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
pattern='test_*.py' if os.environ.get('RUN_RUNTIME_TESTS')=='1' else 'test_core.py'
suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent),pattern=pattern)
result=unittest.TextTestRunner(verbosity=2).run(suite)
report={'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'skipped':len(result.skipped),'scope':'docker_cpu_database' if os.environ.get('RUN_RUNTIME_TESTS')=='1' else 'numpy_stdlib'}
print(json.dumps(report),flush=True)
if os.environ.get('DATA_ROOT'):
    path=Path(os.environ['DATA_ROOT'])/'logs/compact/tests.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')
sys.exit(0 if result.wasSuccessful() and result.testsRun>0 and not result.skipped else 1)
