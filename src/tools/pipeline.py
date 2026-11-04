"""One allocated GPU: build, validate, serve, measure, save, and release it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

from app.config import settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', action='store_true')
    args = parser.parse_args()
    cfg = settings()
    run_id = os.environ.get('RUN_ID', 'default')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,49}', run_id):
        raise ValueError('RUN_ID must be 1..50 lowercase letters, digits or hyphens')
    source = Path(__file__).resolve().parents[1]
    run = Path(os.environ.get('DATA_DIR', '/data'))/'runs'/run_id
    run.mkdir(parents=True, exist_ok=True)
    # The run owns its lock; parallel Jobs must use different run IDs.
    import fcntl
    lock = (run/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run/'evidence/status.json').exists() and not args.serve:
        raise RuntimeError('This run ID already has evidence. Choose a new RUN_ID instead of overwriting it.')
    for name in ('app', 'benchmark'):
        shutil.copytree(source/name, run/name, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('evidence', 'artifacts', 'reports'):
        (run/name).mkdir(exist_ok=True)
    os.chdir(run)
    Path('config.json').write_text(json.dumps(cfg, indent=2))
    os.environ['CONFIG_FILE'] = str(run/'config.json')
    os.environ['PYTHONPATH'] = str(run) + os.pathsep + os.environ.get('PYTHONPATH', '')
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = str(cfg['compile_threads'])
    os.environ['OMP_NUM_THREADS'] = str(cfg['cpu_threads'])
    phase = 'preflight'
    server = None

    def state(value):
        Path('evidence/status.json').write_text(json.dumps({'phase': phase, 'state': value,
            'utc': datetime.now(timezone.utc).isoformat()}, indent=2))
        print(f'{phase}: {value}', flush=True)

    def command(module, logfile, *arguments):
        with open('evidence/'+logfile, 'w') as output:
            result = subprocess.run([sys.executable, '-m', module, *arguments], stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            tail = '\n'.join(Path('evidence/'+logfile).read_text(errors='replace').splitlines()[-30:])
            print(tail, flush=True)
            raise RuntimeError(f'{module} failed; full details in evidence/{logfile}')

    def stop_signal(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_signal)
    try:
        state('running')
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('The Pod must receive exactly one NVIDIA GPU from Kubernetes.')
        if not args.serve or cfg['serve_backend'] == 'bf16_aoti' or cfg['allow_unqualified_aoti']:
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError('BF16 benchmarking requires a BF16-capable NVIDIA GPU.')
            if args.serve and not cfg['allow_unqualified_aoti']:
                raise RuntimeError('Review the report, then explicitly enable allow_unqualified_aoti for experimental AOTI serving.')
            phase = 'build'; state('running')
            command('app.build', 'build.log')
            phase = 'numerical'; state('running')
            command('benchmark.validate', 'numerical.log')
        with open('evidence/packages.txt', 'w') as output:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=output, check=True)
        phase = 'server'; state('running')
        if not args.serve:
            os.environ['BENCHMARK_MODE'] = '1'
        with open('evidence/server.log', 'w') as output:
            server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app.server:app', '--host', '0.0.0.0',
                                       '--port', '8000', '--workers', '1'], stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(300):
            if server.poll() is not None:
                raise RuntimeError('HTTP worker exited; inspect evidence/server.log')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2) as response:
                    Path('evidence/server-start.json').write_bytes(response.read())
                break
            except OSError:
                time.sleep(2)
        else:
            raise TimeoutError('HTTP worker did not become ready')
        if args.serve:
            phase = 'serving'; state('running')
            result = server.wait()
            if result:
                raise RuntimeError(f'HTTP worker exited with status {result}')
            return
        phase = 'quality'; state('running')
        if cfg['quality_enabled']:
            command('benchmark.quality', 'quality.log')
        else:
            Path('evidence/quality').mkdir(exist_ok=True)
            Path('evidence/quality/asr.json').write_text(json.dumps({'passed': False, 'skipped': True,
                'reason': 'quality_enabled=false; this run cannot qualify a candidate'}))
        phase = 'benchmark'; state('running')
        command('benchmark.client', 'benchmark.log', '--endpoint', 'http://127.0.0.1:8000')
        with urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=5) as response:
            Path('evidence/server-final.json').write_bytes(response.read())
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                  for prefix in ('app', 'benchmark') for p in Path(prefix).glob('*.py')}
        Path('evidence/source-hashes.json').write_text(json.dumps(hashes, indent=2))
        phase = 'report'; state('running')
        command('benchmark.report', 'report.log')
        phase = 'complete'; state('passed')
    except BaseException:
        state('failed')
        raise
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()


if __name__ == '__main__':
    main()
