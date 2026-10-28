"""P9 P2: profiled expert placement A/B (cross-load, one arm per load).

Placement is fixed at weight-load time, so the A/B cannot interleave inside
one process; each arm is one 30B load running the SAME frozen workload (the
P8 real-load chat set: 32 unique zh/en prompts, 512 tokens, decode 64):

  decode b32 x3   -> per-step decode wall median + CV   (primary gate)
  prefill-512 b32 x3 -> prefill wall median + CV        (secondary)
  expert counts reset before the A/B workloads; the imbalance of each arm is
  computed analytically from the SAME global routing counts under that arm's
  ownership (routing is placement-independent; ownership is not).

The linear arm with --make-map additionally builds the balanced profiled map
from this workload's counts (scripts/profile_experts.balanced_partition) and
writes it with the model-config sha256, for the profiled arm's load.

Frozen gates (docs/NIGHTLY-LOG.md P2.1): decode b32 median profiled <= 0.95x
linear (throughput +5%) OR mean rank imbalance <= 0.90x linear (-10%); within
arm CV <= 5% on both measurements (else extend to 5 trials, then INVALID);
token streams must match across arms (report-only sanity).

Run (two loads, sequential):
  PYTHONPATH=/root/moe-server-s4000 python3 tests/p9_placement_ab.py \
      --arm linear --make-map --output /root/p9_placement_linear.json
  PYTHONPATH=/root/moe-server-s4000 python3 tests/p9_placement_ab.py \
      --arm profiled --map /root/p9_expert_map.json \
      --output /root/p9_placement_profiled.json
"""
import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/root/moe-server-s4000")
sys.path.insert(0, "/root/moe-server-s4000/tests")
sys.path.insert(0, "/root/moe-server-s4000/scripts")

from nanovllm import LLM, SamplingParams  # noqa: E402
from nanovllm.benchmarking import coefficient_of_variation  # noqa: E402
from p8_moe_30b import build_prompts, drive_batch  # noqa: E402
from profile_experts import balanced_partition  # noqa: E402

SEED = 20261028
TRIALS = 3
TRIALS_EXT = 5
DECODE_TOKENS = 64


def mean_rank_imbalance(counts, assignments):
    """counts [layers, E]; assignments = {rank: [expert_ids]} -> mean |r0-r1|/(r0+r1)."""
    rank_loads = []
    for rank in sorted(assignments):
        idx = torch.tensor(assignments[rank], dtype=torch.long)
        rank_loads.append(counts[:, idx].sum(dim=1).float())
    total = sum(rank_loads).clamp(min=1.0)
    worst = None
    for i in range(len(rank_loads)):
        for j in range(i + 1, len(rank_loads)):
            imb = ((rank_loads[i] - rank_loads[j]).abs() / total).mean()
            worst = imb if worst is None else torch.maximum(worst, imb)
    return float(worst)


def run_arm(args):
    started_at = time.time()
    payload = {
        "schema_version": 1,
        "kind": "placement_arm",
        "arm": args.arm,
        "model": args.model,
        "gates_frozen": "docs/NIGHTLY-LOG.md P2.1",
        "workload": {"prompts": 32, "prompt_tokens": 512,
                     "decode_tokens": DECODE_TOKENS, "seed": SEED},
    }
    kwargs = {}
    if args.arm == "profiled":
        kwargs = {"expert_placement": "profiled", "expert_map_path": args.map}
    llm = LLM(
        args.model,
        enforce_eager=True,
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
        collect_expert_stats=True,
        **kwargs,
    )
    tokenizer = llm.tokenizer
    prompts = build_prompts(tokenizer, 512)
    model = llm.model_runner.model

    # Warmup (allocator + kernel JIT for both prefill and decode paths).
    drive_batch(llm, prompts[:2], 2)

    def trials(decode_tokens):
        walls = []
        toks = []
        count = TRIALS
        while True:
            walls, toks = [], []
            for _ in range(count):
                ordered, median_wall, _pw = drive_batch(llm, prompts, decode_tokens)
                walls.append(median_wall)
                toks.append(ordered)
                print(f"[p9place] {args.arm} decode_tok={decode_tokens} "
                      f"trial={len(walls)} wall={median_wall * 1000:.2f} ms", flush=True)
            if coefficient_of_variation(walls) <= 0.05 or count >= TRIALS_EXT:
                break
            count = TRIALS_EXT
        return walls, toks

    print(f"[p9place] {args.arm}: decode b32 x{TRIALS}", flush=True)
    model.reset_expert_stats()
    decode_walls, decode_toks = trials(DECODE_TOKENS)
    print(f"[p9place] {args.arm}: prefill-512 b32 x{TRIALS}", flush=True)
    prefill_walls = []
    for _ in range(TRIALS):
        for prompt in prompts:
            llm.add_request(prompt, SamplingParams(temperature=0, max_tokens=1, ignore_eos=True))
        trial_prefill = 0.0
        while not llm.is_finished():
            started = time.perf_counter()
            llm.step()
            wall = time.perf_counter() - started
            if llm.last_step_metrics["mode"] == "prefill":
                trial_prefill += wall
        prefill_walls.append(trial_prefill)
        print(f"[p9place] {args.arm} prefill trial={len(prefill_walls)} "
              f"wall={trial_prefill * 1000:.2f} ms", flush=True)

    counts = model.expert_stats().contiguous().cpu()  # [layers, E] global routing
    layers = counts.shape[0]
    half = counts.shape[1] // 2
    linear_assign = {0: list(range(0, half)), 1: list(range(half, counts.shape[1]))}
    imbalance_linear = mean_rank_imbalance(counts, linear_assign)

    if args.arm == "profiled":
        # The gate number must reflect the OWNERSHIP ACTUALLY LOADED (the map,
        # which is per-layer), not a fresh repartition of this arm's counts.
        mapping = json.loads(Path(args.map).read_text(encoding="utf-8"))
        layers_map = mapping.get("layers", mapping)
        per_layer_imb = []
        for layer_id in range(layers):
            loaded_assign = {
                rank: [int(e) for e in layers_map[str(layer_id)][str(rank)]]
                for rank in (0, 1)
            }
            per_layer_imb.append(
                mean_rank_imbalance(counts[layer_id:layer_id + 1], loaded_assign))
        imbalance_arm = statistics.mean(per_layer_imb)
    else:
        imbalance_arm = imbalance_linear

    if args.arm == "linear" and args.make_map:
        all_assign = {}
        for layer_id in range(layers):
            assignments, _loads = balanced_partition(counts[layer_id].tolist())
            all_assign[str(layer_id)] = {
                str(rank): experts for rank, experts in enumerate(assignments)
            }
        map_payload = {
            "model": args.model,
            "model_config_sha256": hashlib.sha256(
                (Path(args.model) / "config.json").read_bytes()
            ).hexdigest(),
            "workload": payload["workload"],
            "note": "balanced partition of the P8 chat workload counts; generated by tests/p9_placement_ab.py",
            "layers": all_assign,
            "counts": counts.tolist(),
        }
        Path(args.map).write_text(json.dumps(map_payload), encoding="utf-8")
        print(f"[p9place] wrote map {args.map}", flush=True)

    payload["decode_walls_ms"] = [round(w * 1000, 2) for w in decode_walls]
    payload["decode_median_ms"] = statistics.median(decode_walls) * 1000
    payload["decode_cv"] = coefficient_of_variation(decode_walls)
    payload["prefill_walls_ms"] = [round(w * 1000, 2) for w in prefill_walls]
    payload["prefill_median_ms"] = statistics.median(prefill_walls) * 1000
    payload["prefill_cv"] = coefficient_of_variation(prefill_walls)
    payload["imbalance_linear_ownership"] = imbalance_linear
    payload["imbalance_arm"] = imbalance_arm
    payload["tokens_first_trial"] = decode_toks[0]
    payload["tokens_deterministic"] = all(t == decode_toks[0] for t in decode_toks)
    payload["wall_seconds"] = round(time.time() - started_at, 1)
    # Persist before teardown: a wedged llm.exit() (I12) must not lose results.
    out = Path(args.output)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[p9place] {args.arm} verdict decode_median={payload['decode_median_ms']:.2f}ms "
          f"cv={payload['decode_cv']:.4f} imb_lin={imbalance_linear:.4f} "
          f"imb_arm={imbalance_arm:.4f}", flush=True)
    print(f"[p9place] wrote {out} wall={payload['wall_seconds']}s", flush=True)
    llm.exit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--arm", choices=("linear", "profiled"), required=True)
    parser.add_argument("--make-map", action="store_true")
    parser.add_argument("--map", default="/root/p9_expert_map.json")
    parser.add_argument("--output", default="/root/p9_placement_arm.json")
    args = parser.parse_args()
    run_arm(args)


if __name__ == "__main__":
    main()
