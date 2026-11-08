"""I12 teardown watchdog test (P14). Run on the server:
    PYTHONPATH=. python3 tests/p14_exit_watchdog.py     # rc=0 == pass

Uses LLMEngine.__new__ with stub model_runner/ps, so no GPU and no model needed.
Case 1: exit() hangs forever -> watchdog must kill ps, unlink shm, exit rc=0 fast.
Case 2: exit() completes normally -> watchdog must NOT fire, rc=0, no message.
Case 3: NANOVLLM_EXIT_TIMEOUT_S=0 disables the watchdog (slow-but-finite exit completes).
"""
import os
import subprocess
import sys
import textwrap
import time

CHILD = textwrap.dedent('''
    import sys, time, os, atexit
    from nanovllm.engine.llm_engine import LLMEngine
    mode = sys.argv[1]

    class Runner:
        def call(self, name, *a):
            if mode == "hang":
                time.sleep(3600)
            if mode == "slow":
                time.sleep(2)

    class P:
        killed = False
        def kill(self): P.killed = True; open("/tmp/p14_killed", "w").write("1")
        def join(self): pass

    e = LLMEngine.__new__(LLMEngine)
    e.model_runner = Runner()
    e.ps = [P()]
    e._shm_name = "nanovllm_p14_test"
    open("/dev/shm/nanovllm_p14_test", "w").write("x")
    e.exit()
    print("EXIT_RETURNED", flush=True)
''')


def run(mode, timeout_env):
    env = dict(os.environ, NANOVLLM_EXIT_TIMEOUT_S=str(timeout_env), PYTHONPATH=".")
    for f in ("/tmp/p14_killed", "/dev/shm/nanovllm_p14_test"):
        if os.path.exists(f):
            os.unlink(f)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-c", CHILD, mode], env=env,
                       capture_output=True, text=True, timeout=60)
    return r, time.time() - t0, os.path.exists("/tmp/p14_killed"), os.path.exists("/dev/shm/nanovllm_p14_test")


def main():
    ok = True
    r, dt, killed, shm = run("hang", 3)
    c1 = r.returncode == 0 and "watchdog fired" in r.stderr and killed and not shm and "EXIT_RETURNED" not in r.stdout and dt < 30
    print(f"[hang] rc={r.returncode} dt={dt:.1f}s killed={killed} shm_left={shm} fired={'watchdog fired' in r.stderr} -> {'PASS' if c1 else 'FAIL'}")
    r, dt, killed, shm = run("ok", 60)
    c2 = r.returncode == 0 and "watchdog fired" not in r.stderr and "EXIT_RETURNED" in r.stdout and not killed
    print(f"[normal] rc={r.returncode} dt={dt:.1f}s killed={killed} -> {'PASS' if c2 else 'FAIL'}")
    r, dt, killed, shm = run("slow", 0)
    c3 = r.returncode == 0 and "watchdog fired" not in r.stderr and "EXIT_RETURNED" in r.stdout
    print(f"[disabled] rc={r.returncode} dt={dt:.1f}s -> {'PASS' if c3 else 'FAIL'}")
    for f in ("/tmp/p14_killed", "/dev/shm/nanovllm_p14_test"):
        if os.path.exists(f):
            os.unlink(f)
    ok = c1 and c2 and c3
    print("RESULT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
