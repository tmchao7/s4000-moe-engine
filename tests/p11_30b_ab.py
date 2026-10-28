"""P4-30B (P11): dual-track gate — eager vs segment graph on 30B TP2 paged.

Frozen gate (benchmarks/P11-graph.md P4-30B, P9 dual-track口径):
  token equality b1/b8/b32 (greedy, 64 decode, 512-token chat prompts) +
  within-arm CV<=5% + segment extra memory <=2 GiB vs eager +
  b1 decode median >=+15% faster -> adopt.
One mode per process; the caller alternates eager/segment x3 and compares
across processes (tokens bitwise, medians, memory).
"""
import argparse
import json
import statistics
import sys
import time

import torch

sys.path.insert(0, "/root/moe-server-s4000")
sys.path.insert(0, "/root/moe-server-s4000/tests")

from nanovllm import LLM, SamplingParams  # noqa: E402
from p8_moe_30b import build_prompts, drive_batch  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eager", "segment"], required=True)
    parser.add_argument("--model",
                        default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--output", default="/root/p11_30b.json")
    args = parser.parse_args()

    started = time.time()
    llm = LLM(
        args.model,
        enforce_eager=(args.mode == "eager"),
        tensor_parallel_size=2,
        expert_parallel_size=2,
        attention_backend="triton_paged",
        scheduler_policy="fcfs",
        moe_backend="grouped_mm",
        decode_moe_backend="bmm_dense",
        max_num_seqs=64,
        num_kvcache_blocks=730,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
    )
    runner = llm.model_runner
    payload = {
        "kind": "p11_30b_ab",
        "mode": args.mode,
        "model": args.model,
        "started": started,
        "batches": {},
    }
    tokenizer = llm.tokenizer
    prompts = build_prompts(tokenizer, 512)

    for batch in (1, 8, 32):
        subset = prompts[:batch]
        walls, prefills, token_runs = [], [], []
        for trial in range(args.trials):
            tokens, decode_wall, prefill_wall = drive_batch(
                llm, subset, args.decode_tokens
            )
            walls.append(decode_wall * 1000.0)
            prefills.append(prefill_wall)
            token_runs.append(tokens)
            print(f"[p4-30b] mode={args.mode} b{batch} trial={trial} "
                  f"decode_median={decode_wall * 1000.0:.2f}ms", flush=True)
        mean = statistics.fmean(walls)
        payload["batches"][f"b{batch}"] = {
            "decode_ms_per_trial": walls,
            "decode_median_ms": statistics.median(walls),
            "cv_pct": statistics.pstdev(walls) / mean * 100.0 if mean else None,
            "prefill_wall_s": prefills,
            "tokens_trial0": token_runs[0],
            "tokens_consistent": all(run == token_runs[0] for run in token_runs),
        }

    payload["memory"] = runner.memory_report()
    payload["finished"] = time.time()
    payload["wall_s"] = payload["finished"] - started
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=str)
    print(f"[p4-30b] mode={args.mode} wrote {args.output} "
          f"b1={payload['batches']['b1']['decode_median_ms']:.2f}ms "
          f"b8={payload['batches']['b8']['decode_median_ms']:.2f}ms "
          f"b32={payload['batches']['b32']['decode_median_ms']:.2f}ms "
          f"alloc={payload['memory']['allocated_gib']:.2f}GiB "
          f"wall={payload['wall_s']:.0f}s", flush=True)
    llm.exit()


if __name__ == "__main__":
    main()
