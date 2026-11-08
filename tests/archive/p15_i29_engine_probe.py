"""P15-B: I29 engine-level probe on the new seetacloud stack.

P9's minimal repro: 0.6B TP2 + triton_paged + whole-graph capture wedges the
device on the FIRST replay (3/3 on the old stack).  The pure-tensor ladder
(p9_mccl_graph_probe.py) is green on BOTH stacks, so only the engine-level
replay path can clear or re-confirm I29 here.

Protocol: N independent child processes (fresh engine each — same-process
dual engines are a known failure mode), each builds 0.6B TP2 triton_paged
with enforce_eager=False (graph capture on) and runs three generates.  The
parent watches each child with a hard timeout, kills the process group on
hang, and counts wedges.  3/3 wedges = I29 stands; 0/3 = I29 cleared on this
stack (escalate to 8B/30B per PLAN.md P15-B).

Run:  PYTHONPATH=. python3 tests/p15_i29_engine_probe.py --repeats 3
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "/root/autodl-tmp/models/Qwen3-0.6B"

CHILD = r"""
import sys, time
sys.path.insert(0, %(repo)r)
print("[child] import", flush=True)
import torch, torch_musa
from nanovllm import LLM, SamplingParams
print("[child] engine-init-start", flush=True)
llm = LLM(%(model)r, enforce_eager=False, tensor_parallel_size=2,
          attention_backend="triton_paged", max_model_len=1024,
          max_num_seqs=8, gpu_memory_utilization=0.5)
print("[child] engine-ready", flush=True)
for i, prompt in enumerate(["解释一下 KV cache。", "Explain paged attention.", "写一首关于 GPU 的诗。"]):
    t0 = time.perf_counter()
    outs = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=16, ignore_eos=True), use_tqdm=False)
    print(f"[child] gen{i} done in {time.perf_counter()-t0:.2f}s tokens={outs[0]['token_ids'][:6]}", flush=True)
llm.exit()
print("[child] clean-exit", flush=True)
"""


def run_child(timeout_s: float) -> dict:
    code = CHILD % {"repo": REPO, "model": MODEL}
    proc = subprocess.Popen(
        [sys.executable, "-c", code], cwd=REPO,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    )
    lines = []
    start = time.perf_counter()
    wedged = False
    while True:
        line = proc.stdout.readline()
        if line:
            lines.append(line.strip())
            print(f"  {line.strip()}", flush=True)
            if line.startswith("[child] clean-exit"):
                break
        elif proc.poll() is not None:
            break
        if time.perf_counter() - start > timeout_s:
            wedged = True
            break
    if wedged:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
    return {"wedge": wedged or proc.returncode not in (0, None),
            "rc": proc.returncode, "phases": lines}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=420.0)
    parser.add_argument("--out", default="/root/p15_data/i29_engine_probe.json")
    args = parser.parse_args()

    runs = []
    for i in range(args.repeats):
        print(f"[i29] repeat {i + 1}/{args.repeats}", flush=True)
        res = run_child(args.timeout_s)
        runs.append(res)
        print(f"[i29] repeat {i + 1}: {'WEDGE' if res['wedge'] else 'clean'}", flush=True)
        time.sleep(5)
        os.system("rm -f /dev/shm/nanovllm*")

    wedges = sum(1 for r in runs if r["wedge"])
    verdict = {
        "repeats": args.repeats, "wedges": wedges,
        "i29_stands": wedges == args.repeats,
        "i29_cleared": wedges == 0,
        "runs": runs,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=1)
    print(json.dumps({k: verdict[k] for k in ("repeats", "wedges", "i29_stands", "i29_cleared")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
