"""Serial, bounded P14 runner (derived from p13_run.py). Remote execution only; never invokes remote git.

Start with setsid nohup, --phase serving (P1 capacity upper bound + P2 1k cold chunk-2048). Each child owns a process
session; cleanup only addresses those sessions. A shared lock prevents overlap.
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT = Path('/root/moe-server-s4000')
OLD = Path('/root/moe-server-p6d')
MODEL = '/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8'
DATA = Path('/root/p14_data')
PY = '/usr/bin/python3'
CHILDREN = {}
DEADLINE = 0  # Required CLI timestamp, set before any child is launched


def log(message):
    print(time.strftime('%Y-%m-%dT%H:%M:%S%z'), message, flush=True)


def clean(proc):
    # Includes multiprocessing children even when their parent has exited.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()
    CHILDREN.pop(proc.pid, None)


def launch(name, argv, cwd=ROOT, env_extra=None):
    if time.time() >= DEADLINE:
        raise RuntimeError('measurement deadline reached')
    env = dict(os.environ, PYTHONPATH=str(cwd), NANOVLLM_SEGMENT_GRAPH='0', NANOVLLM_TRACE_INIT='1')
    env.pop('MUSA_LAUNCH_BLOCKING', None)
    env.pop('CUDA_LAUNCH_BLOCKING', None)
    env.update(env_extra or {})
    meta = {'argv': argv, 'cwd': str(cwd), 'started': time.time(),
            'revision': os.environ.get('P14_REVISION', 'unknown'),
            'env': {key: value for key, value in env.items() if key.startswith('NANOVLLM_') or key in ('PYTHONPATH', 'MUSA_LAUNCH_BLOCKING', 'CUDA_LAUNCH_BLOCKING')},
            'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in list((cwd/'nanovllm').rglob('*.py')) + list((ROOT/'scripts').glob('p1[34]*.py')) + [ROOT/'scripts/benchmark_online.py', ROOT/'tests/p11_step_ab.py']}}
    with (DATA/f'{name}.log').open('w') as stream:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    CHILDREN[proc.pid] = proc
    meta['pid'] = proc.pid
    (DATA/f'{name}.meta.json').write_text(json.dumps(meta, indent=2))
    log(f'START {name} pid={proc.pid}')
    return proc


def run(name, argv, cwd=ROOT, timeout=2700, env_extra=None):
    proc = launch(name, argv, cwd, env_extra)
    try:
        rc = proc.wait(timeout=min(timeout, max(1, DEADLINE-time.time())))
        (DATA/f'{name}.exit.json').write_text(json.dumps({'rc': rc, 'finished': time.time()}))
        if rc:
            raise RuntimeError(f'{name} rc={rc}; see full log')
        log(f'PASS {name}')
    finally:
        clean(proc)


def idle():
    out = subprocess.check_output(['mthreads-gmi'], text=True)
    if len(re.findall(r'\|\s*\d+%\s+0MiB\(49152MiB\)', out)) != 2:
        raise RuntimeError('GPU not idle; no unknown processes will be killed')
    leftovers = list(Path('/dev/shm').glob('nanovllm*'))
    if leftovers:
        raise RuntimeError(f'project shm remains: {leftovers}; inspect ownership before cleanup')


def service(name, old=False, chunk=256, max_seqs=64):
    idle()
    argv = [PY, 'nanovllm/entrypoints/openai/api_server.py', '--model', MODEL,
            '--decode-moe-backend', 'bmm_dense', '--attention-backend', 'triton_paged',
            '--scheduler-policy', 'stall_free', '--prefill-chunk-size', str(chunk),
            '--max-model-len', '4096']
    if not old:
        argv += ['--max-num-seqs', str(max_seqs), '--enforce-eager']
    proc = launch(name, argv, OLD if old else ROOT)
    end = min(time.time()+1100, DEADLINE)
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError(f'{name} init exited rc={proc.returncode}')
        if 'handshake_done' in (DATA/f'{name}.log').read_text(errors='replace'):
            try:
                with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3) as response:
                    if response.status == 200:
                        log(f'READY {name}')
                        return proc
            except Exception:
                pass
        time.sleep(5)
    raise RuntimeError(f'{name} init timeout')


def stop_service(proc):
    owned_shm = set()
    for fd in Path(f'/proc/{proc.pid}/fd').glob('*'):
        try:
            target = Path(os.readlink(fd))
            if target.parent == Path('/dev/shm') and target.name.startswith('nanovllm'):
                owned_shm.add(target)
        except OSError:
            pass
    clean(proc)
    time.sleep(3)
    # Only this engine was allowed to start after idle(); its group is now dead.
    for path in owned_shm:
        path.unlink(missing_ok=True)
    idle()


def bench(name, fractions, long=False):
    argv = [PY, 'scripts/benchmark_online.py', '--model', MODEL.rsplit('/', 1)[-1],
            '--tokenizer', MODEL, '--mode', 'poisson', '--max-qps', '.87',
            '--fractions', fractions, '--requests', '128', '--repeats', '3',
            '--seed', '20261026', '--request-timeout-s', '120', '--output', str(DATA/f'{name}.json')]
    if long:
        # P14: every repeat gets a fresh prompt seed -> genuinely cold prefix cache each repeat.
        argv += ['--prompt-tokens', '1024', '--prompt-mode', 'unique', '--max-tokens', '64', '--prompt-seed-per-repeat']
    else:
        # Same workload as P13-capacity (shared prompt, 128 words, 128 tokens) so 1.8/2.0 extend 1.0-1.6.
        argv += ['--prompt-words', '128', '--prompt-mode', 'shared', '--max-tokens', '128']
    run(name, argv, timeout=4500)


def serving():
    # P1: capacity upper bound, production stack chunk-256.
    proc = service('api_cap256', chunk=256)
    try:
        bench('capacity_hi', '1.8,2.0')
    finally:
        stop_service(proc)
    # P2: 1k cold, chunk-2048 single arm, seed-per-repeat.
    proc = service('api_long2048', chunk=2048)
    try:
        bench('long2048_cold', '1.0', long=True)
    finally:
        stop_service(proc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=['serving'])
    parser.add_argument('--deadline', type=float, required=True, help='UTC Unix timestamp; reserve cleanup time')
    args = parser.parse_args()
    globals()['DEADLINE'] = args.deadline
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    DATA.mkdir(exist_ok=True)
    with open('/tmp/nanovllm-p14.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(str(os.getpid())); lock.flush()
        try:
            idle()
            serving()
        finally:
            for proc in list(CHILDREN.values()):
                clean(proc)
            log('runner finished; verify GPUs and shm')


if __name__ == '__main__':
    main()