"""P13 fix chain (I33): validate dense-ladder + pad-fallback, then 30B online A/B.

Runs AFTER the aba/serving master chain (waits for /root/p13_data/chain_master.log).
Phases: tiny smoke -> 0.6B tokens/steps probe -> 30B eager arm -> 30B segment+fix arm
-> verdict via scripts/p12_serving_ab.py.

Pre-frozen gates for the 30B A/B (written before any run, 2026-10-29 night):
  F1 SLO not worse: per tier, slo_standard & slo_strict medians of fix arm
     >= eager median - 1/128.
  F2 recovery: tier 1.0 TPOT p95 fix median <= eager median * 1.02
     (P12 unfixed arm was +5.9%; recovery to within +2% = "harmless-ized").
  F3 revival bonus: tier 1.0 fix <= 0.97 * eager AND tier 0.5 fix <= 0.97 * eager.
  F1&F2 -> fix validated (segment graph becomes an inert-by-default switch);
  F3 additionally -> promotion discussion opens (defaults stay unchanged tonight).
"""
import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path('/root/moe-server-s4000')
DATA = Path('/root/p13_data')
MODEL = '/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8'
PY = '/usr/bin/python3'
CHILDREN = {}
DEADLINE = 0

FIX_ENV = {
    'NANOVLLM_SEGMENT_GRAPH': '1',
    'NANOVLLM_SEGMENT_GRAPH_BUCKETS': 'dense',
    'NANOVLLM_SEGMENT_GRAPH_PAD_RATIO': '0.15',
}


def log(msg):
    print(time.strftime('%Y-%m-%dT%H:%M:%S%z'), msg, flush=True)


def clean(proc):
    import signal
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        pass
    proc.wait()
    CHILDREN.pop(proc.pid, None)


def launch(name, argv, env_extra=None):
    assert time.time() < DEADLINE, 'deadline reached'
    env = dict(os.environ, PYTHONPATH=str(ROOT), NANOVLLM_TRACE_INIT='1')
    env.pop('MUSA_LAUNCH_BLOCKING', None)
    env.pop('CUDA_LAUNCH_BLOCKING', None)
    env.update(env_extra or {})
    with (DATA / f'{name}.log').open('w') as stream:
        proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
    CHILDREN[proc.pid] = proc
    log(f'START {name} pid={proc.pid}')
    return proc


def run(name, argv, timeout=2700, env_extra=None):
    proc = launch(name, argv, env_extra)
    try:
        rc = proc.wait(timeout=min(timeout, max(1, DEADLINE - time.time())))
        (DATA / f'{name}.exit.json').write_text(json.dumps({'rc': rc, 'finished': time.time()}))
        if rc:
            raise RuntimeError(f'{name} rc={rc}')
        log(f'PASS {name}')
    finally:
        clean(proc)


def idle():
    import re
    import subprocess as sp
    out = sp.check_output(['mthreads-gmi'], text=True)
    if len(re.findall(r'\|\s*\d+%\s+0MiB\(49152MiB\)', out)) != 2:
        raise RuntimeError('GPU not idle')
    leftovers = list(Path('/dev/shm').glob('nanovllm*'))
    if leftovers:
        raise RuntimeError(f'project shm remains: {leftovers}')


def service(name, segment_fix=False):
    idle()
    env_extra = dict(FIX_ENV) if segment_fix else {}
    argv = [PY, 'nanovllm/entrypoints/openai/api_server.py', '--model', MODEL,
            '--decode-moe-backend', 'bmm_dense', '--max-num-seqs', '64',
            '--max-model-len', '4096']
    if segment_fix:
        argv.append('--no-enforce-eager')
    else:
        argv.append('--enforce-eager')
    proc = launch(name, argv, env_extra)
    end = min(time.time() + 1100, DEADLINE)
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError(f'{name} init exited rc={proc.returncode}')
        text = (DATA / f'{name}.log').read_text(errors='replace')
        if 'handshake_done' in text:
            try:
                with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3) as r:
                    if r.status == 200:
                        log(f'READY {name}')
                        return proc
            except Exception:
                pass
        time.sleep(5)
    raise RuntimeError(f'{name} init timeout')


def stop_service(proc):
    import signal
    clean(proc)
    time.sleep(5)
    idle()


def bench(name, fractions):
    run(name, [PY, 'scripts/benchmark_online.py', '--model', MODEL.rsplit('/', 1)[-1],
               '--tokenizer', MODEL, '--mode', 'poisson', '--max-qps', '.87',
               '--fractions', fractions, '--requests', '128', '--repeats', '3',
               '--seed', '20261026', '--request-timeout-s', '120',
               '--output', str(DATA / f'{name}.json')], timeout=5400)


def main():
    global DEADLINE
    ap = argparse.ArgumentParser()
    ap.add_argument('--deadline', type=float, required=True)
    ap.add_argument('--skip-wait', action='store_true')
    args = ap.parse_args()
    DEADLINE = args.deadline
    if not args.skip_wait:
        marker = DATA / 'chain_master.log'
        log('waiting for master chain...')
        while not marker.exists():
            time.sleep(60)
        log('master chain done; starting fix chain')
    import fcntl
    with open('/tmp/nanovllm-p13.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(str(os.getpid())); lock.flush()
        try:
            # Phase 1: tiny smoke (capture + replay + token self-consistency).
            run('fix_tiny_smoke', [PY, 'tests/p13_segment_fix_smoke.py'], timeout=1500)
            # Phase 2: 0.6B tokens/steps with dense+fallback vs eager reference.
            batches = '6,8,9,12,15,16,17,20,24,32'
            common = ['--model', '/root/autodl-tmp/models/Qwen3-0.6B', '--dense',
                      '--batches', batches, '--rounds', '3',
                      '--prompt-tokens', '128', '--decode-tokens', '80']
            for i in (1, 2):
                idle()
                run(f'fixprobe_eager_{i}', [PY, 'tests/p11_step_ab.py', '--mode', 'eager',
                    *common, '--output', str(DATA / f'fixprobe_eager_{i}.json')], timeout=1200)
                idle()
                run(f'fixprobe_segment_{i}', [PY, 'tests/p11_step_ab.py', '--mode', 'segment',
                    *common, '--output', str(DATA / f'fixprobe_segment_{i}.json')],
                    timeout=1200, env_extra=dict(FIX_ENV))
            run('fixprobe_verdict', [PY, 'scripts/p13_decide.py', 'padding',
                *[str(DATA / f'fixprobe_{m}_{i}.json') for i in (1, 2) for m in ('eager', 'segment')],
                '--output', str(DATA / 'fixprobe_verdict.json')], timeout=600)
            # Phase 3: 30B eager arm (frozen matrix).
            proc = service('api_fix_eager')
            try:
                bench('fix_eager', '0.5,0.75,0.9,1.0')
            finally:
                stop_service(proc)
            # Phase 4: 30B segment+fix arm (frozen matrix + saturation).
            proc = service('api_fix_segment', segment_fix=True)
            try:
                bench('fix_segment', '0.5,0.75,0.9,1.0,1.2,1.4')
            finally:
                stop_service(proc)
            # Phase 5: verdict.
            run('fix_verdict', [PY, 'scripts/p12_serving_ab.py',
                str(DATA / 'fix_eager.json'), str(DATA / 'fix_segment.json'),
                '--verdict-out', str(DATA / 'fix_verdict.json')], timeout=600)
            log('fix chain DONE')
        finally:
            for proc in list(CHILDREN.values()):
                clean(proc)
            log('fix runner finished; verify GPUs and shm')


if __name__ == '__main__':
    main()
