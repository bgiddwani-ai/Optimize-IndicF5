#!/usr/bin/env python3
"""Dependency-free Kubernetes helper. Credentials travel over stdin, not arguments."""
import argparse
import base64
import getpass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from string import Template

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from app.config import validate


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


def render():
    opt = options()
    path = Path(os.environ.get('CONFIG_FILE', 'config.json'))
    if not path.is_absolute():
        path = ROOT/path
    cfg = validate(json.loads(path.read_text()))
    optional = {}
    for name, field in (('STORAGE_CLASS', 'storageClassName'), ('RUNTIME_CLASS', 'runtimeClassName')):
        optional[name] = f'{field}: {json.dumps(os.environ[name])}' if os.environ.get(name) else ''
    optional['PULL_SECRET'] = ('imagePullSecrets: [{name: '+json.dumps(os.environ['PULL_SECRET'])+'}]') if os.environ.get('PULL_SECRET') else ''
    selector = json.loads(os.environ.get('NODE_SELECTOR') or '{}')
    if not isinstance(selector, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in selector.items()):
        raise ValueError('NODE_SELECTOR must be a JSON object of string labels')
    optional['NODE_SELECTOR'] = 'nodeSelector: '+json.dumps(selector) if selector else ''
    values = {**optional, 'NAMESPACE': opt['namespace'], 'RUN_ID': opt['run_id'],
              'JOB_NAME': opt['job'], 'IMAGE': json.dumps(opt['image']), 'HF_SECRET': opt['secret'],
              'CONFIG_JSON': json.dumps(cfg), 'WAIT_TIMEOUT': str(int(os.environ.get('WAIT_TIMEOUT', '14400')))}
    return Template((ROOT/'k8s.yaml').read_text()).substitute(values)


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


def launch():
    opt = options()
    if 'your-account' in opt['image']:
        raise ValueError('Export IMAGE with the container image you built and pushed.')
    existing = kubectl('get', 'job', opt['job'], '--ignore-not-found', '-o', 'name', capture=True)
    if existing.strip():
        raise RuntimeError('Job already exists. Export a new RUN_ID before rerunning.')
    kubectl('apply', '-f', '-', data=render(), namespaced=False)


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
        for name in ('reports', 'evidence', 'config.json'):
            kubectl('cp', '-c', container, f'{pod}:/data/runs/{opt["run_id"]}/{name}', str(target/name))
        print(f'Results: outputs/{opt["run_id"]}/reports/report.html')
    finally:
        if created:
            kubectl('delete', 'pod', reader, '--ignore-not-found', '--wait=false')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['render', 'token', 'run', 'logs', 'wait', 'download', 'stop'])
    args = parser.parse_args()
    if args.command == 'render':
        print(render()); return
    if args.command == 'token':
        token()
    elif args.command == 'run':
        if 'your-account' in options()['image']:
            raise ValueError('Export IMAGE with the container image you built and pushed.')
        namespace()
        if not kubectl('get', 'secret', options()['secret'], '--ignore-not-found', '-o', 'name', capture=True).strip():
            token()
        launch()
    elif args.command == 'wait': wait()
    elif args.command == 'logs': kubectl('logs', '-f', 'job/'+options()['job'], '--all-containers=true')
    elif args.command == 'download': download()
    elif args.command == 'stop':
        kubectl('delete', 'job/'+options()['job'], '--ignore-not-found', '--wait=true')
        print('Benchmark Job stopped; PVC results and authentication Secret retained.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, TimeoutError, subprocess.CalledProcessError) as exc:
        print(f'Error: {exc}', file=sys.stderr); sys.exit(1)
