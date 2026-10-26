"""Isolate the batch-32 decode non-determinism discovered by the attention A/B.

The A/B showed that at batch 32 the generated tokens differ across identical
repeats for BOTH attention backends (first divergence at the 2nd token, i.e. the
first decode step).  This script, in one 30B load, discriminates the candidates:

  A. ``grouped_mm``        -> baseline candidate (torch._grouped_mm / MCCL)
  B. ``loop``              -> per-expert Python loop with unique indices; if it
                              is stable while grouped_mm is not, the grouped GEMM
                              (or its packing) is the culprit
  C. ``grouped_mm`` + KV zeroed before each repeat -> if stable, stale KV data
                              is being read (slot/tail bug)
  D. batch 16 control      -> expected stable (matches the A/B result)

Each case runs three times and reports cross-repeat first divergence.

Run:  PYTHONPATH=<repo> python3 tests/p5_determinism_b32.py
"""
import argparse
import random
import sys

import torch

sys.path.insert(0, "/root/moe-server-s4000")

MODEL = "/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
BATCH = 32
CONTEXT = 512
OUTPUT_LEN = 8
REPEATS = 3
SEED = 20261026


def first_divergence(reference, candidate):
    for sequence_index, (want, actual) in enumerate(zip(reference, candidate)):
        for position, (expected, got) in enumerate(zip(want, actual)):
            if expected != got:
                return sequence_index, position, got, expected
    return None


def zero_kv_cache(llm):
    cache = getattr(llm.model_runner, "kv_cache", None)
    if cache is None:
        return False
    cache.zero_()
    torch.musa.synchronize()
    return True


def run_case(llm, params, prompts, moe_backend, zero_kv, label):
    llm.model_runner.call("set_moe_backend", moe_backend)
    runs = []
    for repeat in range(REPEATS):
        llm.scheduler.block_manager.clear_prefix_cache()
        if zero_kv:
            zero_kv_cache(llm)
        outputs = llm.generate(prompts, params, use_tqdm=False)
        runs.append([output["token_ids"] for output in outputs])
    divergences = [
        first_divergence(runs[0], run) for run in runs[1:]
    ]
    stable = all(divergence is None for divergence in divergences)
    print(
        f"[b32-det] {label:34s} stable={stable} divergences={divergences}",
        flush=True,
    )
    return stable


def main():
    global BATCH, CONTEXT, OUTPUT_LEN, REPEATS
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--context", type=int, default=CONTEXT)
    parser.add_argument("--output-len", type=int, default=OUTPUT_LEN)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args()
    BATCH, CONTEXT, OUTPUT_LEN, REPEATS = (
        args.batch, args.context, args.output_len, args.repeats
    )

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
    rng = random.Random(SEED)
    vocab = llm.model_runner.config.hf_config.vocab_size
    prompts32 = [[rng.randrange(vocab) for _ in range(CONTEXT)] for _ in range(BATCH)]
    prompts16 = prompts32[:16]
    params = SamplingParams(temperature=0, max_tokens=OUTPUT_LEN, ignore_eos=True)
    results = {}
    try:
        results["grouped_b32"] = run_case(
            llm, params, prompts32, "grouped_mm", False, "A grouped_mm b32"
        )
        results["loop_b32"] = run_case(
            llm, params, prompts32, "loop", False, "B loop b32"
        )
        results["grouped_b32_zero_kv"] = run_case(
            llm, params, prompts32, "grouped_mm", True, "C grouped_mm b32 + zero KV"
        )
        results["grouped_b16"] = run_case(
            llm, params, prompts16, "grouped_mm", False, "D grouped_mm b16 (control)"
        )
    finally:
        llm.exit()

    print(f"[b32-det] summary={results}", flush=True)
    if not results["grouped_b16"]:
        print("[b32-det] control is unstable -> problem is not batch-specific", flush=True)
    elif results["loop_b32"] and not results["grouped_b32"]:
        print("[b32-det] ROOT CAUSE: grouped GEMM path (loop is stable)", flush=True)
    elif results["grouped_b32_zero_kv"] and not results["grouped_b32"]:
        print("[b32-det] ROOT CAUSE: stale KV data (zeroing fixes it)", flush=True)
    elif not results["loop_b32"] and not results["grouped_b32"]:
        print(
            "[b32-det] shared component unstable (MCCL all-reduce or SDPA decode)",
            flush=True,
        )
    else:
        print("[b32-det] inconclusive; see per-case output", flush=True)


if __name__ == "__main__":
    main()
