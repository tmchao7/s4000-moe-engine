"""Bounded TP2 graph controls for I1/I2/I3 without a 30B model load.

The parent invokes every attempt in a fresh process, so a graph/device failure
cannot poison the next sample.  It compares the existing tiny MoE structural
control with Qwen3-8B dense, always using the production-eligible SDPA plus
bmm_dense decode stack.  The script is intentionally a diagnostic; it never
changes graph or attention defaults.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_MODEL = "/tmp/tiny_moe"
DENSE_MODEL = "/root/models/Qwen3-8B"
PROMPTS = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12], [42, 42, 42]]
RESULT_PREFIX = "P6_GRAPH_ISOLATION_RESULT="


def worker(args):
    sys.path.insert(0, str(REPO_ROOT))
    from nanovllm import LLM, SamplingParams

    config = {
        "model": args.model,
        "control": args.control,
        "attention_backend": "sdpa",
        "decode_moe_backend": "bmm_dense",
        "tensor_parallel_size": 2,
        "expert_parallel_size": 2,
        "enforce_eager": False,
        "max_num_seqs": 8,
        "max_model_len": 2048 if args.dense else 256,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if not args.dense:
        config["moe_backend"] = "grouped_mm"

    llm = None
    result = {"control": args.control, "model": args.model, "status": "error"}
    try:
        llm = LLM(**config)
        result["graph_report"] = llm.model_runner.call("graph_report")
        params = SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True)

        def generate(label, clear_prefix=False):
            if clear_prefix:
                llm.scheduler.block_manager.clear_prefix_cache()
            print(f"[graph-isolation] {args.control} {label}", flush=True)
            outputs = llm.generate(PROMPTS, params, use_tqdm=False)
            return [output["token_ids"] for output in outputs]

        graph_1 = generate("graph-1")
        graph_2 = generate("graph-2")
        graph_3 = generate("graph-3")
        llm.model_runner.call("set_enforce_eager", True)
        eager = generate("eager")
        llm.model_runner.call("set_enforce_eager", False)
        graph_after_eager = generate("graph-after-eager")
        graph_after_clear = generate("graph-after-clear", clear_prefix=True)
        result.update({
            "status": "ok",
            "graph_stable": graph_1 == graph_2 == graph_3,
            "graph_after_eager_stable": graph_1 == graph_after_eager,
            "graph_after_clear_stable": graph_1 == graph_after_clear,
            "eager_matches_graph": graph_1 == eager,
            "tokens": {
                "graph_1": graph_1,
                "graph_2": graph_2,
                "graph_3": graph_3,
                "eager": eager,
                "graph_after_eager": graph_after_eager,
                "graph_after_clear": graph_after_clear,
            },
        })
    except BaseException as exc:  # diagnostic evidence must include device failures
        result.update({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
    finally:
        if llm is not None:
            try:
                llm.exit()
            except BaseException as exc:  # parent watchdog remains the last resort
                result["exit_error"] = f"{type(exc).__name__}: {exc}"
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "ok" else 1


def parse_result(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    return None


def run_attempt(args, control, attempt, launch_blocking):
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--control", control["name"], "--model", control["model"],
        "--max-tokens", str(args.max_tokens),
        "--gpu-memory-utilization", str(control["gpu_memory_utilization"]),
    ]
    if control["dense"]:
        command.append("--dense")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    if launch_blocking:
        environment["MUSA_LAUNCH_BLOCKING"] = "1"
    started = time.monotonic()
    process = subprocess.Popen(
        command, cwd=REPO_ROOT, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=args.watchdog_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    result = parse_result(stdout)
    return {
        "kind": "blocking_reproduction" if launch_blocking else "attempt",
        "control": control["name"],
        "attempt": attempt,
        "launch_blocking": launch_blocking,
        "elapsed_s": round(time.monotonic() - started, 3),
        "returncode": process.returncode,
        "timed_out": timed_out,
        "worker": result,
        "stdout_tail": stdout[-8000:],
        "stderr_tail": stderr[-8000:],
        "passed": bool(
            not timed_out and result and result.get("status") == "ok"
            and result.get("graph_stable")
            and result.get("graph_after_eager_stable")
            and result.get("graph_after_clear_stable")
            and result.get("eager_matches_graph")
        ),
    }


def parent(args):
    controls = [
        {"name": "tiny_moe_tp2", "model": args.tiny_model, "dense": False,
         "gpu_memory_utilization": 0.30},
        {"name": "qwen3_8b_dense_tp2", "model": args.dense_model, "dense": True,
         "gpu_memory_utilization": 0.30},
    ]
    output = Path(args.output)
    jsonl_path = output.with_suffix(".jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for control in controls:
            blocking_ran = False
            for attempt in range(1, args.attempts + 1):
                row = run_attempt(args, control, attempt, launch_blocking=False)
                rows.append(row)
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                print("[graph-isolation]", json.dumps({
                    "control": control["name"], "attempt": attempt,
                    "passed": row["passed"], "timed_out": row["timed_out"],
                }, sort_keys=True), flush=True)
                if not row["passed"] and not blocking_ran:
                    blocking_ran = True
                    diagnostic = run_attempt(args, control, attempt, launch_blocking=True)
                    rows.append(diagnostic)
                    stream.write(json.dumps(diagnostic, sort_keys=True) + "\n")
                    stream.flush()
    summary = {
        "schema_version": 1,
        "kind": "p6_graph_isolation",
        "attempts_requested": args.attempts,
        "watchdog_s": args.watchdog_s,
        "controls": {
            control["name"]: {
                "passed": sum(
                    row["passed"] for row in rows
                    if row["kind"] == "attempt" and row["control"] == control["name"]
                ),
                "total": sum(
                    row["kind"] == "attempt" and row["control"] == control["name"]
                    for row in rows
                ),
            }
            for control in controls
        },
        "records": rows,
    }
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[graph-isolation] wrote {output} and {jsonl_path}", flush=True)
    return 0 if all(value["passed"] == value["total"] for value in summary["controls"].values()) else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--control")
    parser.add_argument("--model")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--watchdog-s", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--tiny-model", default=TINY_MODEL)
    parser.add_argument("--dense-model", default=DENSE_MODEL)
    parser.add_argument("--output", default="/root/p6_graph_isolation.json")
    args = parser.parse_args()
    if args.worker:
        if not args.control or not args.model:
            parser.error("--worker requires --control and --model")
        raise SystemExit(worker(args))
    raise SystemExit(parent(args))


if __name__ == "__main__":
    main()
