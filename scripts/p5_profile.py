"""One-load 30B profiling matrix and evidence-based optimization decision."""

import argparse
import json
import random
from pathlib import Path


DEFAULT_WORKLOADS = "1x128,8x128,16x128,32x128,16x512,32x512,16x2048,32x2048"
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_workloads(value):
    workloads = []
    for item in value.split(","):
        batch, context = item.lower().split("x")
        workloads.append((int(batch), int(context)))
    return workloads


def segment_total(profiles, name):
    return max(
        (profile.get("segments", {}).get(name, {}).get("total_ms", 0.0) for profile in profiles),
        default=0.0,
    )


def local_pair_imbalance(profiles):
    totals = []
    for profile in profiles:
        totals.append(sum(
            row["value"] for row in profile.get("observations", [])
            if row["name"] == "moe.local_pairs"
        ))
    average = sum(totals) / len(totals) if totals else 0.0
    return (max(totals) / average - 1.0) if average else 0.0


def decide(records):
    decode = [row for row in records if row["phase"] == "decode"]
    worst_attention = 0.0
    worst_moe = 0.0
    worst_route_combine_of_moe = 0.0
    worst_collective = 0.0
    worst_imbalance = 0.0
    for row in decode:
        profiles = row["profiles"]
        forward = segment_total(profiles, "runner.forward")
        attention = (
            segment_total(profiles, "attention.gather_mask")
            + segment_total(profiles, "attention.sdpa")
        )
        moe_compute = sum(segment_total(profiles, name) for name in (
            "moe.router_topk", "moe.local_map", "moe.route_pack",
            "moe.grouped_mm_1", "moe.activation", "moe.grouped_mm_2",
            "moe.weighted_combine",
        ))
        route_combine = sum(segment_total(profiles, name) for name in (
            "moe.local_map", "moe.route_pack", "moe.weighted_combine",
        ))
        collective = segment_total(profiles, "moe.ep_all_reduce")
        if forward:
            worst_attention = max(worst_attention, attention / forward)
            worst_moe = max(worst_moe, moe_compute / forward)
            worst_collective = max(worst_collective, collective / forward)
        if moe_compute:
            worst_route_combine_of_moe = max(
                worst_route_combine_of_moe, route_combine / moe_compute
            )
        worst_imbalance = max(worst_imbalance, row["rank_pair_imbalance"])
    candidates = []
    if worst_attention >= 0.30:
        candidates.append("triton_paged")
    if worst_moe >= 0.35 or worst_route_combine_of_moe >= 0.10:
        candidates.append("fused_grouped_mm")
    if worst_collective >= 0.20:
        candidates.append("collective_overlap")
    if worst_imbalance >= 0.10:
        candidates.append("profiled_placement")
    priority = next(
        (name for name in ("triton_paged", "fused_grouped_mm", "collective_overlap", "profiled_placement") if name in candidates),
        "graph_or_launch_overhead",
    )
    return {
        "max_attention_fraction": worst_attention,
        "max_moe_compute_fraction": worst_moe,
        "max_route_combine_fraction_of_moe": worst_route_combine_of_moe,
        "max_collective_fraction": worst_collective,
        "max_rank_pair_imbalance": worst_imbalance,
        "triggered_candidates": candidates,
        "priority": priority,
    }


def correctness_gate(llm, sampling_params_cls, reference_path):
    reference_rows = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    prompts = [row["prompt"] for row in reference_rows]
    expected = [row["token_ids"] for row in reference_rows]
    results = {}
    params = sampling_params_cls(temperature=0, max_tokens=8)
    for backend in ("loop", "grouped_mm"):
        llm.model_runner.call("set_moe_backend", backend)
        runs = []
        for _ in range(2):
            llm.scheduler.block_manager.clear_prefix_cache()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            runs.append([output["token_ids"] for output in outputs])
        if runs[0] != expected or runs[1] != expected:
            raise RuntimeError(f"30B correctness gate failed for {backend}: {runs}")
        results[backend] = runs[0]
    if results["loop"] != results["grouped_mm"]:
        raise RuntimeError("loop/grouped_mm correctness gate mismatch")
    llm.model_runner.call("set_moe_backend", "grouped_mm")
    print("[p5-profile] 30B correctness gate PASS", flush=True)
    return {"reference": str(reference_path), "backends": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--workloads", default=DEFAULT_WORKLOADS)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--output", default="/root/p5_profile.json")
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument(
        "--reference",
        default=str(REPO_ROOT / "benchmarks" / "p2-reference-8tok.json"),
    )
    parser.add_argument("--skip-correctness-gate", action="store_true")
    args = parser.parse_args()
    workloads = parse_workloads(args.workloads)
    assert args.decode_steps > 0

    from nanovllm import LLM, SamplingParams

    max_context = max(context for _, context in workloads)
    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=2,
        expert_parallel_size=2,
        moe_backend="grouped_mm",
        attention_backend="sdpa",
        scheduler_policy="fcfs",
        collect_expert_stats=True,
        max_model_len=max_context + args.decode_steps + 8,
        gpu_memory_utilization=0.85,
    )
    rng = random.Random(args.seed)
    vocab_size = llm.model_runner.config.hf_config.vocab_size
    records = []
    verification = None
    try:
        if not args.skip_correctness_gate:
            verification = correctness_gate(llm, SamplingParams, args.reference)
        for batch, context in workloads:
            prompts = [
                [rng.randrange(vocab_size) for _ in range(context)]
                for _ in range(batch)
            ]
            params = SamplingParams(
                temperature=0,
                max_tokens=args.decode_steps + 2,
                ignore_eos=True,
            )
            llm.model_runner.model.reset_expert_stats()
            seq_ids = [llm.add_request(prompt, params) for prompt in prompts]

            llm.model_runner.call("start_profiling")
            while llm.scheduler.waiting:
                llm.step()
            prefill_profiles = llm.model_runner.call("profile_snapshot")
            records.append({
                "phase": "prefill",
                "batch": batch,
                "context": context,
                "profiles": prefill_profiles,
                "rank_pair_imbalance": local_pair_imbalance(prefill_profiles),
            })

            llm.model_runner.call("start_profiling")
            for _ in range(args.decode_steps):
                if llm.is_finished():
                    break
                llm.step()
            decode_profiles = llm.model_runner.call("profile_snapshot")
            records.append({
                "phase": "decode",
                "batch": batch,
                "context": context,
                "profiles": decode_profiles,
                "rank_pair_imbalance": local_pair_imbalance(decode_profiles),
            })

            for seq_id in seq_ids:
                llm.cancel_request(seq_id)
            llm.scheduler.block_manager.clear_prefix_cache()
            print(f"[p5-profile] batch={batch} context={context} captured", flush=True)
    finally:
        llm.exit()

    payload = {
        "schema_version": 1,
        "model": args.model,
        "decode_steps": args.decode_steps,
        "verification": verification,
        "records": records,
        "decision": decide(records),
    }
    output = Path(args.output)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[p5-profile] decision", json.dumps(payload["decision"], sort_keys=True), flush=True)
    print(f"[p5-profile] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
