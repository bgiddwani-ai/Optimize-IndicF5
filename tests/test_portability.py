import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import validate
spec = importlib.util.spec_from_file_location('k8s_helper', ROOT/'scripts/k8s.py')
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
            resources = k8s.resources()
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
            resources = k8s.resources()
        pod = resources[-1]['spec']['template']['spec']
        self.assertEqual(pod['runtimeClassName'], 'gpu-runtime')
        self.assertEqual(pod['nodeSelector'], {'pool':'gpu'})
        self.assertEqual(resources[1]['spec']['storageClassName'], 'fast-storage')

    def test_serving_is_private_and_serialized(self):
        with patch.dict(os.environ, {}, clear=True): resources = k8s.resources('serve')
        self.assertEqual(resources[-1]['spec']['type'], 'ClusterIP')
        self.assertEqual(resources[-2]['spec']['replicas'], 1)
        self.assertEqual(resources[-2]['spec']['strategy']['type'], 'Recreate')

    def test_all_python_sources_parse(self):
        for directory in ('app','benchmark','scripts','tools'):
            for path in (ROOT/directory).glob('*.py'): ast.parse(path.read_text(), filename=str(path))


if __name__ == '__main__': unittest.main()
