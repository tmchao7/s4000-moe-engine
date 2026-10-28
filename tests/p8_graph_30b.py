"""30B graph A/B (P2.4, L1): eager vs replay decode throughput and identity.

Loads Qwen3-30B-A3B TP2 with enforce_eager=False (graphs captured at init,
triton_paged + bmm_dense), then interleaves eager/replay decode per batch size
by flipping ``runner.enforce_eager`` at runtime (capture happened at init; the
flag only gates which path run() takes).

Frozen gates (docs/NIGHTLY-LOG.md P2, dual-track):
  hard: graph tokens == eager tokens per batch; throughput CV <= 5%;
        graph memory delta <= 2 GiB
  perf (b1): >= +15% adopt; +8..15% record "correct, pending decision";
        < +8% "graph not worth it on this stack"
  b8 >= +5% and b32 >= 0.98 report-only.

Run:
  PYTHONPATH=/root/moe-server-s4000 python3 tests/p8_graph_30b.py \
      --model /root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8 \
      --output /root/p8_graph_30b.json
"""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/root/moe-server-s4000")

from nanovllm import LLM, SamplingParams  # noqa: E402
from nanovllm.benchmarking import coefficient_of_variation  # noqa: E402

SEED = 20261028
TRIALS = 5
DECODE_STEPS = 16


def drive_batch(llm, prompts, max_tokens, use_graph):
    """Add a batch, prefill, then time decode steps under the given path.

    Returns (tokens_by_seq_sorted_by_seq_id, median_decode_wall_s).
    """
    runner = llm.model_runner
    # Both ranks must flip in lockstep: the MCCL collectives inside the replay
    # only match if both sides take the same path (rank1 flipping too requires
    # the shm command channel, not a local attribute write).
    runner.call("set_enforce_eager", not use_graph)
    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True))
    walls = []
    first_decode = False
    tokens = {}
    while not llm.is_finished():
        started = time.perf_counter()
        outputs, _ = llm.step()
        wall = time.perf_counter() - started
        if llm.last_step_metrics["mode"] == "decode":
            if first_decode:
                walls.append(wall)
            first_decode = True
        for seq_id, token_ids, _metrics in outputs:
            tokens[seq_id] = list(token_ids)
    ordered = [tokens[key] for key in sorted(tokens)]
    return ordered, statistics.median(walls) if walls else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--kv-blocks", type=int, default=640,
                        help="explicit KV blocks to leave graph-capture headroom")
    parser.add_argument("--decode-steps", type=int, default=DECODE_STEPS)
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--output", default="/root/p8_graph_30b.json")
    args = parser.parse_args()

    started_at = time.time()
    payload = {
        "schema_version": 1,
        "kind": "graph_30b_ab",
        "model": args.model,
        "gates_frozen": "docs/NIGHTLY-LOG.md P2 dual-track",
        "stack": {"attention": "triton_paged", "decode_moe": "bmm_dense",
                  "prefill_moe": "grouped_mm", "tp": 2, "enforce_eager_at_init": False},
    }
    rng = random.Random(SEED)
    vocab = 151936

    llm = LLM(
        args.model,
        enforce_eager=False,
        tensor_parallel_size=2,
        expert_parallel_size=2,
        attention_backend="triton_paged",
        scheduler_policy="fcfs",
        moe_backend="grouped_mm",
        decode_moe_backend="bmm_dense",
        max_num_seqs=32,
        num_kvcache_blocks=args.kv_blocks,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
    )
    runner = llm.model_runner
    payload["graph_memory"] = {
        "allocated_delta_gib": getattr(runner, "graph_memory_delta_gib", None),
        "reserved_delta_gib": getattr(runner, "graph_memory_reserved_delta_gib", None),
        "captured_moe_backend": getattr(runner, "captured_moe_backend", None),
        "kv_blocks": runner.config.num_kvcache_blocks,
    }
    # Warmup both paths once per batch size (allocator + bucket JIT).
    for use_graph in (False, True):
        drive_batch(llm, [[rng.randrange(vocab) for _ in range(16)]], 2, use_graph)

    results = {}
    for batch in (1, 8, 32):
        entries = {"eager": [], "graph": []}
        tokens = {"eager": [], "graph": []}
        for trial in range(args.trials * 2):
            use_graph = trial % 2 == 1
            mode = "graph" if use_graph else "eager"
            prompts = [
                [rng.randrange(vocab) for _ in range(16)]
                for _ in range(batch)
            ]
            ordered, median_wall = drive_batch(llm, prompts, args.decode_steps, use_graph)
            entries[mode].append(median_wall)
            tokens[mode].append(ordered)
            print(f"[graph30b] b{batch} trial {trial} mode={mode} "
                  f"decode_median={median_wall * 1000:.2f} ms", flush=True)
        graph_med = statistics.median(entries["graph"])
        eager_med = statistics.median(entries["eager"])
        identity = all(g == e for g, e in zip(tokens["graph"], tokens["eager"]))
        results[f"b{batch}"] = {
            "eager_walls_ms": [round(w * 1000, 2) for w in entries["eager"]],
            "graph_walls_ms": [round(w * 1000, 2) for w in entries["graph"]],
            "eager_median_ms": eager_med * 1000,
            "graph_median_ms": graph_med * 1000,
            "speedup": eager_med / graph_med if graph_med else None,
            "eager_cv": coefficient_of_variation(entries["eager"]),
            "graph_cv": coefficient_of_variation(entries["graph"]),
            "tokens_identical": identity,
        }
        print(f"[graph30b] b{batch}: eager={eager_med * 1000:.2f}ms graph={graph_med * 1000:.2f}ms "
              f"speedup={eager_med / graph_med if graph_med else 0:.3f} identical={identity}", flush=True)

    llm.exit()
    payload["results"] = results

    mem_delta = max(
        payload["graph_memory"]["allocated_delta_gib"] or 0.0,
        payload["graph_memory"]["reserved_delta_gib"] or 0.0,
    )
    hard = (
        all(entry["tokens_identical"] for entry in results.values())
        and all(entry["eager_cv"] <= 0.05 and entry["graph_cv"] <= 0.05 for entry in results.values())
        and mem_delta <= 2.0
    )
    b1 = results["b1"]["speedup"] or 0.0
    if b1 >= 1.15:
        perf = "ADOPT"
    elif b1 >= 1.08:
        perf = "RECORD_PENDING_DECISION"
    else:
        perf = "NOT_WORTH_IT"
    payload["verdict"] = {
        "hard_gate": "PASS" if hard else "FAIL",
        "b1_speedup": round(b1, 4),
        "performance_track": perf,
        "graph_memory_delta_gib": mem_delta,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[graph30b] verdict={json.dumps(payload['verdict'])}", flush=True)
    print(f"[graph30b] wrote {out} wall={time.time() - started_at:.0f}s", flush=True)


if __name__ == "__main__":
    main()
