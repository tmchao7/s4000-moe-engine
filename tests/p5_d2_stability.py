"""Load-D2 stability probe: b32 reproducibility + b64 crash, in one model load.

Two open questions block the attention decision (``docs/P5-attention-ab.md``):

1. batch 32 decode was **non-reproducible across identical repeats for both**
   attention backends, so the Triton paged kernel's 19-59% throughput win cannot
   be adopted yet.  ``--bisect`` runs the full 2x2 (MoE backend x attention
   backend) switch matrix to localise the unstable component.
2. the final matrix crashed on the **third b64 repeat** with
   ``RuntimeError: MUSA error: unknown error``; this script re-runs b64 alone and
   records whether the failure reproduces.

Everything streams to stdout as ``[d2] {...}`` JSON lines so a crash keeps the
finished evidence.

Run:  PYTHONPATH=<repo> python3 tests/p5_d2_stability.py --repeats 10 --bisect
"""
import argparse
import json
import random
import sys
import traceback

import torch

sys.path.insert(0, "/root/moe-server-s4000")

MODEL = "/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
SEED = 20261026


def first_divergence(reference, candidate):
    for sequence_index, (want, actual) in enumerate(zip(reference, candidate)):
        for position, (expected, got) in enumerate(zip(want, actual)):
            if expected != got:
                return {"sequence": sequence_index, "position": position,
                        "got": got, "expected": expected}
    return None


def log(record):
    print("[d2]", json.dumps(record, sort_keys=True), flush=True)


def run_series(llm, params, prompts, repeats, moe_backend, attention_backend, label):
    llm.model_runner.call("set_moe_backend", moe_backend)
    llm.model_runner.call("set_attention_backend", attention_backend)
    runs = []
    for _ in range(repeats):
        llm.scheduler.block_manager.clear_prefix_cache()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        runs.append([output["token_ids"] for output in outputs])
    divergences = [first_divergence(runs[0], run) for run in runs[1:]]
    stable = all(divergence is None for divergence in divergences)
    record = {
        "case": label,
        "moe_backend": moe_backend,
        "attention_backend": attention_backend,
        "repeats": repeats,
        "stable": stable,
        "divergences": divergences,
        "first_run_head": [row[:6] for row in runs[0][:4]] if runs else None,
    }
    log(record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--bisect", action="store_true", help="run the 2x2 switch matrix")
    parser.add_argument("--b64-repeats", type=int, default=3)
    parser.add_argument("--skip-b64", action="store_true")
    args = parser.parse_args()

    from nanovllm import LLM, SamplingParams

    llm = LLM(
        MODEL,
        enforce_eager=True,
        tensor_parallel_size=2,
        expert_parallel_size=2,
        moe_backend="grouped_mm",
        attention_backend="sdpa",
        scheduler_policy="fcfs",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )
    vocab = llm.model_runner.config.hf_config.vocab_size
    rng = random.Random(SEED)
    prompts = [
        [rng.randrange(vocab) for _ in range(args.context)] for _ in range(args.batch)
    ]
    params = SamplingParams(temperature=0, max_tokens=args.output_len, ignore_eos=True)
    results = {"b32": None, "bisect": [], "b64": None}
    try:
        results["b32"] = run_series(
            llm, params, prompts, args.repeats, "grouped_mm", "sdpa", "b32-default"
        )
        if args.bisect and not results["b32"]["stable"]:
            for moe_backend in ("grouped_mm", "loop"):
                for attention_backend in ("sdpa", "triton_paged"):
                    label = f"b32-{moe_backend}-{attention_backend}"
                    results["bisect"].append(run_series(
                        llm, params, prompts, min(args.repeats, 3),
                        moe_backend, attention_backend, label,
                    ))
        if not args.skip_b64:
            # The matrix uses prompt 128 / output 32 at b64; reproduce that shape.
            b64_prompts = [
                [rng.randrange(vocab) for _ in range(128)] for _ in range(64)
            ]
            b64_params = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
            llm.model_runner.call("set_moe_backend", "grouped_mm")
            llm.model_runner.call("set_attention_backend", "sdpa")
            b64 = {"repeats_requested": args.b64_repeats, "runs": [], "errors": []}
            for repeat in range(args.b64_repeats):
                try:
                    llm.scheduler.block_manager.clear_prefix_cache()
                    llm.generate(b64_prompts, b64_params, use_tqdm=False)
                    b64["runs"].append(repeat)
                    log({"case": "b64", "repeat": repeat, "status": "ok"})
                except Exception as exc:  # noqa: BLE001 - the crash is the evidence
                    error = f"{type(exc).__name__}: {exc}"
                    b64["errors"].append({"repeat": repeat, "error": error,
                                          "traceback": traceback.format_exc(limit=8)})
                    log({"case": "b64", "repeat": repeat, "status": "error",
                         "error": error})
                    break
            b64["reproduced"] = bool(b64["errors"])
            results["b64"] = b64
    finally:
        try:
            llm.exit()
        except Exception:  # noqa: BLE001 - process may already be wedged
            pass

    print("[d2] summary=" + json.dumps(results, sort_keys=True), flush=True)
    if results["b32"] is not None:
        print(
            f"[d2] b32 stable across {results['b32']['repeats']} repeats: "
            f"{results['b32']['stable']}",
            flush=True,
        )
    for row in results["bisect"]:
        print(f"[d2] bisect {row['case']}: stable={row['stable']}", flush=True)
    if results["b64"] is not None:
        print(
            f"[d2] b64 crash reproduced: {results['b64']['reproduced']} "
            f"(ok repeats={results['b64']['runs']})",
            flush=True,
        )


if __name__ == "__main__":
    main()
