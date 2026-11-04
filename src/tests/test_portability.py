import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
import yaml
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
from app.config import validate
spec = importlib.util.spec_from_file_location('k8s_helper', ROOT/'src/k8s.py')
k8s = importlib.util.module_from_spec(spec); spec.loader.exec_module(k8s)


class PortabilityTests(unittest.TestCase):
    def test_profile_cannot_exclude_probes(self):
        with self.assertRaises(ValueError): validate({'max_frames': 512})
        self.assertEqual(validate({'max_frames': 512, 'probe_frames': [128,512]})['max_frames'], 512)

    def test_concurrency_and_unknown_settings_fail_early(self):
        for value in [0, -1, 65, True]:
            with self.assertRaises(ValueError): validate({'concurrencies': [value]})
        with self.assertRaises(ValueError): validate({'typo': 1})

    def test_standard_gpu_allocation_and_persistent_data(self):
        with patch.dict(os.environ, {}, clear=True):
            resources = list(yaml.safe_load_all(k8s.render()))
        job = resources[-1]; pod = job['spec']['template']['spec']
        self.assertEqual(pod['containers'][0]['resources']['limits']['nvidia.com/gpu'], 1)
        self.assertFalse(pod['automountServiceAccountToken'])
        self.assertNotIn('hostNetwork', pod)
        self.assertNotIn('nodeSelector', pod)
        self.assertFalse(any('hostPath' in volume for volume in pod['volumes']))
        self.assertNotIn('storageClassName', resources[1]['spec'])
        env = {entry['name']: entry for entry in pod['containers'][0]['env']}
        self.assertIn('secretKeyRef', env['HF_TOKEN']['valueFrom'])
        self.assertNotIn('CUDA_VISIBLE_DEVICES', env)
        self.assertNotIn('NVIDIA_VISIBLE_DEVICES', env)

    def test_user_cluster_overrides(self):
        with patch.dict(os.environ, {'RUNTIME_CLASS':'gpu-runtime','STORAGE_CLASS':'fast-storage',
                                     'NODE_SELECTOR':'{"pool":"gpu"}'}, clear=True):
            resources = list(yaml.safe_load_all(k8s.render()))
        pod = resources[-1]['spec']['template']['spec']
        self.assertEqual(pod['runtimeClassName'], 'gpu-runtime')
        self.assertEqual(pod['nodeSelector'], {'pool':'gpu'})
        self.assertEqual(resources[1]['spec']['storageClassName'], 'fast-storage')

    def test_all_python_sources_parse(self):
        for path in (ROOT/'src').rglob('*.py'):
            ast.parse(path.read_text(), filename=str(path))

    def test_run_creates_secret_once_and_applies_the_yaml(self):
        for existing in ('', 'secret/indicf5-hf'):
            def reply(*args, **kwargs):
                if args[:2] == ('get', 'secret'): return existing
                if args[:2] == ('get', 'job'): return ''
            with patch.dict(os.environ, {'IMAGE':'registry.example/indicf5:v1'}, clear=True), \
                 patch.object(sys, 'argv', ['k8s.py', 'run']), \
                 patch.object(k8s, 'kubectl', side_effect=reply) as kubectl, \
                 patch.object(k8s, 'token') as token:
                k8s.main()
                self.assertEqual(token.call_count, 0 if existing else 1)
                applied = list(yaml.safe_load_all(kubectl.call_args.kwargs['data']))
                self.assertEqual(applied[-1]['kind'], 'Job')
                self.assertEqual(applied[-1]['spec']['template']['spec']['containers'][0]['image'],
                                 'registry.example/indicf5:v1')


if __name__ == '__main__': unittest.main()
