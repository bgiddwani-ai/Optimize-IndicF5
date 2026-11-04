#!/usr/bin/env python3
"""Dependency-free Kubernetes helper. Credentials travel over stdin, not arguments."""
import argparse
import base64
import getpass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import validate


def yaml_text(value, indent=0):
    pad = ' '*indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            key = json.dumps(key)
            if isinstance(item, (dict, list)) and item:
                lines.append(f'{pad}{key}:\n{yaml_text(item, indent+2)}')
            else:
                lines.append(f'{pad}{key}: {json.dumps(item, ensure_ascii=False)}')
        return '\n'.join(lines)
    if isinstance(value, list):
        return '\n'.join(f'{pad}-\n{yaml_text(item, indent+2)}' if isinstance(item, (dict, list))
                         else f'{pad}- {json.dumps(item, ensure_ascii=False)}' for item in value)
    raise TypeError(type(value))


def options():
    run_id = os.environ.get('RUN_ID', 'run-001')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,49}', run_id):
        raise ValueError('RUN_ID must contain 1..50 lowercase letters, digits or hyphens')
    result = {
        'namespace': os.environ.get('NAMESPACE', 'indicf5'),
        'run_id': run_id, 'job': os.environ.get('JOB_NAME') or 'indicf5-'+run_id,
        'image': os.environ.get('IMAGE', 'ghcr.io/your-account/indicf5-aoti:latest'),
        'secret': os.environ.get('HF_SECRET', 'indicf5-hf'),
    }
    for key in ('namespace', 'job', 'secret'):
        if not re.fullmatch(r'[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?', result[key]):
            raise ValueError(f'Invalid Kubernetes name: {key}')
    return result


def resources(mode='benchmark'):
    opt = options()
    path = Path(os.environ.get('CONFIG_FILE', 'config.json'))
    if not path.is_absolute():
        path = ROOT/path
    cfg = validate(json.loads(path.read_text()))
    namespace = {'apiVersion': 'v1', 'kind': 'Namespace', 'metadata': {'name': opt['namespace']}}
    meta = {'namespace': opt['namespace']}
    pvc = {'apiVersion': 'v1', 'kind': 'PersistentVolumeClaim', 'metadata': {**meta, 'name': 'indicf5-data'},
           'spec': {'accessModes': ['ReadWriteOnce'], 'resources': {'requests': {'storage': os.environ.get('STORAGE_SIZE', '40Gi')}}}}
    if os.environ.get('STORAGE_CLASS'):
        pvc['spec']['storageClassName'] = os.environ['STORAGE_CLASS']
    config = {'apiVersion': 'v1', 'kind': 'ConfigMap', 'metadata': {**meta, 'name': 'cfg-'+opt['run_id']},
              'data': {'config.json': json.dumps(cfg, indent=2)}}
    labels = {'app.kubernetes.io/name': 'indicf5', 'indicf5/run': opt['run_id'], 'indicf5/mode': mode}
    container = {
        'name': 'app', 'image': opt['image'], 'imagePullPolicy': os.environ.get('IMAGE_PULL_POLICY', 'IfNotPresent'),
        'args': ['--serve'] if mode == 'serve' else [],
        'env': [{'name': 'CONFIG_FILE', 'value': '/etc/indicf5/config.json'},
                {'name': 'RUN_ID', 'value': opt['run_id']},
                {'name': 'HF_TOKEN', 'valueFrom': {'secretKeyRef': {'name': opt['secret'], 'key': 'HF_TOKEN'}}}],
        'resources': {
            'requests': {'cpu': os.environ.get('CPU_REQUEST', '4'), 'memory': os.environ.get('MEMORY_REQUEST', '16Gi'), 'nvidia.com/gpu': 1},
            'limits': {'cpu': os.environ.get('CPU_LIMIT', '8'), 'memory': os.environ.get('MEMORY_LIMIT', '32Gi'), 'nvidia.com/gpu': 1}},
        'securityContext': {'allowPrivilegeEscalation': False, 'readOnlyRootFilesystem': True, 'capabilities': {'drop': ['ALL']}},
        'volumeMounts': [{'name': 'data', 'mountPath': '/data'}, {'name': 'config', 'mountPath': '/etc/indicf5', 'readOnly': True},
                         {'name': 'tmp', 'mountPath': '/tmp'}, {'name': 'shm', 'mountPath': '/dev/shm'}],
    }
    spec = {
        'automountServiceAccountToken': False,
        'securityContext': {'runAsUser': 1000, 'runAsGroup': 1000, 'fsGroup': 1000, 'fsGroupChangePolicy': 'OnRootMismatch', 'runAsNonRoot': True, 'seccompProfile': {'type': 'RuntimeDefault'}},
        'restartPolicy': 'Never' if mode == 'benchmark' else 'Always',
        'terminationGracePeriodSeconds': 45,
        'dnsPolicy': os.environ.get('DNS_POLICY', 'ClusterFirst'),
        'containers': [container],
        'volumes': [{'name': 'data', 'persistentVolumeClaim': {'claimName': 'indicf5-data'}},
                    {'name': 'config', 'configMap': {'name': 'cfg-'+opt['run_id']}},
                    {'name': 'tmp', 'emptyDir': {'sizeLimit': '8Gi'}},
                    {'name': 'shm', 'emptyDir': {'medium': 'Memory', 'sizeLimit': '2Gi'}}],
    }
    if os.environ.get('RUNTIME_CLASS'):
        spec['runtimeClassName'] = os.environ['RUNTIME_CLASS']
    if os.environ.get('NODE_SELECTOR'):
        selector = json.loads(os.environ['NODE_SELECTOR'])
        if not isinstance(selector, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in selector.items()):
            raise ValueError('NODE_SELECTOR must be a JSON object of string labels')
        spec['nodeSelector'] = selector
    if os.environ.get('PULL_SECRET'):
        spec['imagePullSecrets'] = [{'name': os.environ['PULL_SECRET']}]
    template = {'metadata': {'labels': labels}, 'spec': spec}
    if mode == 'benchmark':
        workload = {'apiVersion': 'batch/v1', 'kind': 'Job', 'metadata': {**meta, 'name': opt['job'], 'labels': labels},
            'spec': {'backoffLimit': 0, 'activeDeadlineSeconds': int(os.environ.get('WAIT_TIMEOUT', '14400')),
                     'ttlSecondsAfterFinished': 604800, 'template': template}}
        return [namespace, pvc, config, workload]
    container.update(ports=[{'name': 'http', 'containerPort': 8000}],
        readinessProbe={'httpGet': {'path': '/healthz', 'port': 'http'}, 'periodSeconds': 10},
        startupProbe={'httpGet': {'path': '/healthz', 'port': 'http'}, 'periodSeconds': 10, 'failureThreshold': 180})
    workload = {'apiVersion': 'apps/v1', 'kind': 'Deployment', 'metadata': {**meta, 'name': 'indicf5'},
        'spec': {'replicas': 1, 'strategy': {'type': 'Recreate'},
                 'selector': {'matchLabels': {'app.kubernetes.io/name': 'indicf5', 'indicf5/mode': 'serve'}}, 'template': template}}
    service = {'apiVersion': 'v1', 'kind': 'Service', 'metadata': {**meta, 'name': 'indicf5'},
        'spec': {'type': 'ClusterIP', 'selector': {'app.kubernetes.io/name': 'indicf5', 'indicf5/mode': 'serve'},
                 'ports': [{'name': 'http', 'port': 8000, 'targetPort': 'http'}]}}
    return [namespace, pvc, config, workload, service]


def kubectl(*arguments, data=None, capture=False, namespaced=True):
    command = ['kubectl']
    if namespaced:
        command += ['-n', options()['namespace']]
    command += list(arguments)
    return subprocess.run(command, input=data, text=True, check=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def namespace():
    manifest = {'apiVersion': 'v1', 'kind': 'Namespace', 'metadata': {'name': options()['namespace']}}
    kubectl('apply', '-f', '-', data=json.dumps(manifest), namespaced=False)


def token():
    namespace()
    secret = getpass.getpass('Approved Hugging Face read token (hidden): ').strip()
    if not secret:
        raise ValueError('Token must not be empty')
    manifest = {'apiVersion': 'v1', 'kind': 'Secret', 'metadata': {'name': options()['secret'], 'namespace': options()['namespace']},
                'type': 'Opaque', 'data': {'HF_TOKEN': base64.b64encode(secret.encode()).decode()}}
    kubectl('apply', '--server-side', '--field-manager=indicf5-helper', '-f', '-', data=json.dumps(manifest))


def launch(mode):
    opt = options()
    if mode == 'benchmark':
        existing = kubectl('get', 'job', opt['job'], '--ignore-not-found', '-o', 'name', capture=True)
        if existing.strip():
            raise RuntimeError('Job already exists. Choose a new RUN_ID, or explicitly run ./run.sh stop first.')
    kubectl('apply', '-f', '-', data=json.dumps({'apiVersion': 'v1', 'kind': 'List', 'items': resources(mode)}), namespaced=False)


def wait():
    deadline = time.monotonic()+int(os.environ.get('WAIT_TIMEOUT', '14400'))
    previous = None
    while time.monotonic() < deadline:
        job = json.loads(kubectl('get', 'job', options()['job'], '-o', 'json', capture=True))
        status = job.get('status', {})
        if status.get('succeeded', 0):
            print('Benchmark complete. Run ./run.sh download, then inspect qualification.json separately.'); return
        if any(c.get('type') == 'Failed' and c.get('status') == 'True' for c in status.get('conditions', [])):
            raise RuntimeError('Benchmark Job failed. Inspect ./run.sh logs and download its evidence.')
        message = (status.get('active', 0), status.get('failed', 0))
        if message != previous:
            print(f'Waiting: active={message[0]}, failed={message[1]}', flush=True); previous = message
        time.sleep(5)
    raise TimeoutError('Wait timed out; the Job may still be running. Inspect status before retrying.')


def download():
    opt = options(); reader = 'reports-'+opt['run_id']; created = False
    pods = json.loads(kubectl('get', 'pods', '-l', 'app.kubernetes.io/name=indicf5,indicf5/run='+opt['run_id'], '-o', 'json', capture=True))['items']
    live = [p for p in pods if p.get('status', {}).get('phase') == 'Running']
    if live:
        pod = live[0]['metadata']['name']; container = 'app'
    else:
        pod = reader; container = 'reader'
        manifest = {'apiVersion': 'v1', 'kind': 'Pod', 'metadata': {'name': reader, 'namespace': opt['namespace']},
            'spec': {'automountServiceAccountToken': False, 'restartPolicy': 'Never', 'activeDeadlineSeconds': 3600,
                'securityContext': {'runAsUser': 1000, 'runAsGroup': 1000, 'fsGroup': 1000, 'runAsNonRoot': True, 'seccompProfile': {'type': 'RuntimeDefault'}},
                'containers': [{'name': 'reader', 'image': 'busybox:1.37.0', 'command': ['sleep', '3600'],
                    'securityContext': {'allowPrivilegeEscalation': False, 'readOnlyRootFilesystem': True, 'capabilities': {'drop': ['ALL']}},
                    'resources': {'requests': {'cpu': '25m', 'memory': '32Mi'}, 'limits': {'cpu': '250m', 'memory': '128Mi'}},
                    'volumeMounts': [{'name': 'data', 'mountPath': '/data', 'readOnly': True}]}],
                'volumes': [{'name': 'data', 'persistentVolumeClaim': {'claimName': 'indicf5-data'}}]}}
        kubectl('apply', '-f', '-', data=json.dumps(manifest)); created = True
    target = ROOT/'outputs'/opt['run_id']; target.mkdir(parents=True, exist_ok=True)
    try:
        if created:
            kubectl('wait', '--for=condition=Ready', 'pod/'+reader, '--timeout=300s')
        for directory in ('reports', 'evidence'):
            kubectl('cp', '-c', container, f'{pod}:/data/runs/{opt["run_id"]}/{directory}', str(target/directory))
        print(f'Results: outputs/{opt["run_id"]}/reports/report.html')
    finally:
        if created:
            kubectl('delete', 'pod', reader, '--ignore-not-found', '--wait=false')


def build():
    image = options()['image']
    if 'your-account' in image:
        raise ValueError('Set IMAGE in .env to a registry/repository you can push.')
    subprocess.run(['docker', 'build', '--platform', 'linux/amd64', '-t', image, str(ROOT)], check=True)
    subprocess.run(['docker', 'push', image], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['render','build','token','run','up','logs','wait','status','download','serve','forward','stop'])
    parser.add_argument('--serve', action='store_true', help='Render the optional serving Deployment')
    args = parser.parse_args()
    if args.command == 'render':
        print('\n---\n'.join(yaml_text(x) for x in resources('serve' if args.serve else 'benchmark'))); return
    if args.command in ('build', 'up'):
        build()
    if args.command in ('token', 'up'):
        if args.command == 'token': token()
        else:
            namespace()
            if not kubectl('get', 'secret', options()['secret'], '--ignore-not-found', '-o', 'name', capture=True).strip(): token()
    if args.command in ('run', 'up'): namespace(); launch('benchmark')
    elif args.command == 'serve': namespace(); launch('serve')
    elif args.command == 'wait': wait()
    elif args.command == 'logs': kubectl('logs', '-f', 'job/'+options()['job'], '--all-containers=true')
    elif args.command == 'status': kubectl('get', 'jobs,pods,pvc')
    elif args.command == 'download': download()
    elif args.command == 'forward': kubectl('port-forward', 'service/indicf5', '8000:8000')
    elif args.command == 'stop':
        kubectl('delete', 'job/'+options()['job'], 'deployment/indicf5', '--ignore-not-found', '--wait=true')
        print('GPU workloads stopped; PVC data and authentication Secret retained.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, TimeoutError, subprocess.CalledProcessError) as exc:
        print(f'Error: {exc}', file=sys.stderr); sys.exit(1)
