"""P29-A: production-stack topology / EP-imbalance profile (30B, one load).

Run on the S4000 box:
    PYTHONPATH=. python3 scripts/p29_probe_topology.py --output /root/p29_data/topology.json

This intentionally does NOT compare backends or gate anything; it records the
current production stack's all-reduce / expert-imbalance / step-time baseline.
"""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

from nanovllm import LLM, SamplingParams


def parse_workloads(value):
    out = []
    for item in value.split(","):
        batch, context = item.lower().split("x")
        out.append((int(batch), int(context)))
    return out


def _seg(profiles, name):
    values = [p.get("segments", {}).get(name, {}).get("total_ms", 0.0)
              for p in profiles]
    return sum(values) / len(values) if values else 0.0


def _sum_seg(profiles, prefix):
    names = set()
    for p in profiles:
        names.update(p.get("segments", {}))
    return sum(_seg(profiles, n) for n in names if n.startswith(prefix))


def summarize(profiles):
    forward = _seg(profiles, "runner.forward")
    attention = _sum_seg(profiles, "attention.")
    moe_compute = _sum_seg(profiles, "moe.")
    ep_allreduce = _seg(profiles, "moe.ep_all_reduce")
    # moe.* above includes ep_all_reduce; keep the fraction explicit.
    moe_compute -= ep_allreduce
    totals = []
    for p in profiles:
        totals.append(sum(row["value"] for row in p.get("observations", [])
                          if row["name"] == "moe.local_pairs"))
    avg = sum(totals) / len(totals) if totals else 0.0
    imbalance = (max(totals) / avg - 1.0) if avg else 0.0
    return {
        "forward_ms": round(forward, 4),
        "attention_ms": round(attention, 4),
        "moe_compute_ms": round(moe_compute, 4),
        "ep_allreduce_ms": round(ep_allreduce, 4),
        "attention_fraction": round(attention / forward, 4) if forward else 0.0,
        "moe_compute_fraction": round(moe_compute / forward, 4) if forward else 0.0,
        "ep_allreduce_fraction": round(ep_allreduce / forward, 4) if forward else 0.0,
        "rank_pair_imbalance": round(imbalance, 4),
        "local_pairs": [round(x, 2) for x in totals],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--workloads", default="1x128,8x128,32x128")
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--clean-steps", type=int, default=4)
    parser.add_argument("--output", default="/root/p29_data/topology.json")
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    workloads = parse_workloads(args.workloads)
    max_context = max(c for _, c in workloads)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=2,
        expert_parallel_size=2,
        moe_backend="grouped_mm",
        decode_moe_backend="bmm_dense",
        attention_backend="triton_paged",
        scheduler_policy="stall_free",
        prefill_chunk_size=256,
        enable_profiling=True,
        max_model_len=max_context + args.decode_steps + 8,
        gpu_memory_utilization=0.85,
    )
    rng = random.Random(args.seed)
    vocab = llm.model_runner.config.hf_config.vocab_size
    records = []
    try:
        for batch, context in workloads:
            prompts = [[rng.randrange(vocab) for _ in range(context)]
                       for _ in range(batch)]
            params = SamplingParams(temperature=0,
                                    max_tokens=args.decode_steps + args.clean_steps + 2,
                                    ignore_eos=True)
            ids = [llm.add_request(p, params) for p in prompts]

            llm.model_runner.call("start_profiling")
            while llm.scheduler.waiting:
                llm.step()
            prefill = llm.model_runner.call("profile_snapshot")
            records.append({
                "phase": "prefill", "batch": batch, "context": context,
                "summary": summarize(prefill),
            })

            llm.model_runner.call("start_profiling")
            step_ms = []
            for _ in range(args.decode_steps):
                if llm.is_finished():
                    break
                t0 = time.perf_counter()
                llm.step()
                step_ms.append((time.perf_counter() - t0) * 1e3)
            decode = llm.model_runner.call("profile_snapshot")
            # Clean (non-profiled) decode steps: the device-event profiler
            # adds per-layer events, so this is the step-time baseline.
            llm.model_runner.call("stop_profiling")
            clean_ms = []
            for _ in range(args.clean_steps):
                if llm.is_finished():
                    break
                t0 = time.perf_counter()
                llm.step()
                clean_ms.append((time.perf_counter() - t0) * 1e3)
            records.append({
                "phase": "decode", "batch": batch, "context": context,
                "step_ms_median": round(statistics.median(step_ms), 4) if step_ms else None,
                "clean_step_ms_median": round(statistics.median(clean_ms), 4) if clean_ms else None,
                "summary": summarize(decode),
            })
            print(f"[p29-topo] batch={batch} context={context} "
                  f"step_ms={records[-1]['step_ms_median']} "
                  f"ep_frac={records[-1]['summary']['ep_allreduce_fraction']} "
                  f"imbalance={records[-1]['summary']['rank_pair_imbalance']}",
                  flush=True)
            for seq_id in ids:
                llm.cancel_request(seq_id)
            llm.scheduler.block_manager.clear_prefix_cache()
    finally:
        llm.exit()

    payload = {
        "schema_version": 1,
        "model": args.model,
        "config": {
            "tp": 2, "ep": 2, "moe_backend": "grouped_mm",
            "decode_moe_backend": "bmm_dense",
            "attention_backend": "triton_paged",
            "scheduler_policy": "stall_free", "prefill_chunk_size": 256,
            "enforce_eager": True,
        },
        "workloads": [{"batch": b, "context": c} for b, c in workloads],
        "decode_steps": args.decode_steps,
        "records": records,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[p29-topo] wrote", args.output)


if __name__ == "__main__":
    main()
