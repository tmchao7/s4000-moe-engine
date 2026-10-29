"""Serial, bounded P13 runner. Remote execution only; never invokes remote git.

Start with setsid nohup, --phase prereq|aba|serving. Each child owns a process
session; cleanup only addresses those sessions. A shared lock prevents overlap.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT = Path('/root/moe-server-s4000')
OLD = Path('/root/moe-server-p6d')
MODEL = '/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8'
DATA = Path('/root/p13_data')
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
            'revision': os.environ.get('P13_REVISION', 'unknown'),
            'env': {key: env.get(key) for key in ('PYTHONPATH', 'NANOVLLM_SEGMENT_GRAPH', 'NANOVLLM_TRACE_INIT', 'MUSA_LAUNCH_BLOCKING')},
            'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in list((cwd/'nanovllm').rglob('*.py')) + list((ROOT/'scripts').glob('p13*.py')) + [ROOT/'scripts/benchmark_online.py', ROOT/'tests/p11_step_ab.py']}}
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
    if out.count('0MiB(49152MiB)') != 2:
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
    clean(proc)
    time.sleep(3)
    # Only this engine was allowed to start after idle(); its group is now dead.
    for path in Path('/dev/shm').glob('nanovllm*'):
        path.unlink()
    idle()


def bench(name, fractions, long=False):
    argv = [PY, 'scripts/benchmark_online.py', '--model', MODEL.rsplit('/', 1)[-1],
            '--tokenizer', MODEL, '--mode', 'poisson', '--max-qps', '.87',
            '--fractions', fractions, '--requests', '128', '--repeats', '3',
            '--seed', '20261026', '--request-timeout-s', '120', '--output', str(DATA/f'{name}.json')]
    argv += ['--prompt-tokens', '1024', '--prompt-mode', 'unique', '--max-tokens', '64'] if long else ['--prompt-words', '128', '--prompt-mode', 'shared', '--max-tokens', '128']
    run(name, argv, timeout=4500)


def prereq():
    for name, command in (
        ('unit', [PY, '-m', 'pytest', 'tests/test_benchmarking_v3.py', '-q']),
        ('scheduler', [PY, 'tests/p8_scheduler_gaps.py']),
        ('grouped', [PY, 'tests/p2_grouped_mm.py']),
        ('bmm', [PY, 'tests/p6_bmm_dense_tiny.py']),
    ):
        idle()
        run(name, command)
    for mode in ('eager', 'segment'):
        idle()
        name = f'tiny_{mode}'
        run(name, [PY, 'tests/p11_step_ab.py', '--mode', mode, '--model', '/tmp/tiny_moe',
            '--attention-backend', 'sdpa', '--batches', '16,17', '--prompt-tokens', '32',
            '--decode-tokens', '16', '--rounds', '3', '--output', str(DATA/f'{name}.json'),
            '--code-revision', os.environ.get('P13_REVISION', 'unknown')])
    for repeat in range(1, 6):
        for mode in ('eager', 'segment'):
            idle()
            name = f'padding_{repeat}_{mode}'
            run(name, [PY, 'tests/p11_step_ab.py', '--mode', mode, '--model',
                '/root/autodl-tmp/models/Qwen3-0.6B', '--dense', '--batches', '6,8,9,12,15,16,17,20,24,32',
                '--rounds', '3', '--prompt-tokens', '128', '--decode-tokens', '80',
                '--output', str(DATA/f'{name}.json'), '--code-revision', os.environ.get('P13_REVISION', 'unknown')])
    run('padding_verdict', [PY, 'scripts/p13_decide.py', 'padding',
        *[str(p) for p in sorted(DATA.glob('padding_[1-5]_*.json')) if '.meta.' not in p.name and '.exit.' not in p.name],
        '--output', str(DATA/'padding_verdict.json')])


def aba_phase():
    for name, old in (('aba_A1', True), ('aba_B', False), ('aba_A2', True)):
        proc = service('api_'+name, old=old, max_seqs=512)
        try:
            bench(name, '0.9,1.0')
        finally:
            stop_service(proc)
    run('aba_verdict', [PY, 'scripts/p13_decide.py', 'aba', *[str(DATA/f'aba_{n}.json') for n in ('A1', 'B', 'A2')], '--output', str(DATA/'aba_verdict.json')])


def serving():
    proc = service('api_serving256', chunk=256)
    try:
        bench('capacity', '1.0,1.2,1.4,1.6')
        bench('long256', '0.5,0.75,1.0', long=True)
    finally:
        stop_service(proc)
    proc = service('api_serving1024', chunk=1024)
    try:
        bench('long1024', '0.5,0.75,1.0', long=True)
    finally:
        stop_service(proc)
    proc = service('api_long256_repeat', chunk=256)
    try:
        bench('long256_repeat', '0.5', long=True)
    finally:
        stop_service(proc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=['prereq', 'aba', 'serving'])
    parser.add_argument('--deadline', type=float, required=True, help='UTC Unix timestamp; reserve cleanup time')
    args = parser.parse_args()
    globals()['DEADLINE'] = args.deadline
    DATA.mkdir(exist_ok=True)
    with open('/tmp/nanovllm-p13.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(str(os.getpid())); lock.flush()
        try:
            idle()
            {'prereq': prereq, 'aba': aba_phase, 'serving': serving}[args.phase]()
        finally:
            for proc in list(CHILDREN.values()):
                clean(proc)
            log('runner finished; verify GPUs and shm')


if __name__ == '__main__':
    main()
