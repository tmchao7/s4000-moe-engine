"""Collect Qwen3-MoE routing counts and build a static balanced expert map."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def balanced_partition(counts, ranks=2):
    capacity = len(counts) // ranks
    assignments = [[] for _ in range(ranks)]
    loads = [0 for _ in range(ranks)]
    for expert_id in sorted(range(len(counts)), key=lambda idx: counts[idx], reverse=True):
        candidates = [rank for rank in range(ranks) if len(assignments[rank]) < capacity]
        rank = min(candidates, key=lambda candidate: (loads[candidate], candidate))
        assignments[rank].append(expert_id)
        loads[rank] += counts[expert_id]
    for assignment in assignments:
        assignment.sort()
    return assignments, loads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--output", default="/root/p2_expert_map.json")
    args = parser.parse_args()

    from nanovllm import LLM, SamplingParams

    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=2,
        expert_parallel_size=2,
        moe_backend="grouped_mm",
        expert_placement="linear",
        collect_expert_stats=True,
        max_model_len=max(2048, args.prompt_len + args.output_len),
        gpu_memory_utilization=0.85,
    )
    rng = random.Random(20261026)
    vocab_size = llm.model_runner.config.hf_config.vocab_size
    prompts = [
        [rng.randrange(vocab_size) for _ in range(args.prompt_len)]
        for _ in range(args.batch)
    ]
    model = llm.model_runner.model
    model.reset_expert_stats()
    try:
        llm.generate(
            prompts,
            SamplingParams(temperature=0, max_tokens=args.output_len, ignore_eos=True),
            use_tqdm=False,
        )
        llm.model_runner.acc.synchronize()
        counts = model.expert_stats().contiguous().cpu().tolist()
    finally:
        llm.exit()

    layers = {}
    balancedness = []
    for layer_id, layer_counts in enumerate(counts):
        assignments, loads = balanced_partition(layer_counts)
        layers[str(layer_id)] = {
            str(rank): expert_ids for rank, expert_ids in enumerate(assignments)
        }
        average = sum(loads) / len(loads)
        balancedness.append(average / max(loads) if max(loads) else 1.0)
    payload = {
        "model": args.model,
        "model_config_sha256": hashlib.sha256(
            (Path(args.model) / "config.json").read_bytes()
        ).hexdigest(),
        "batch": args.batch,
        "prompt_len": args.prompt_len,
        "output_len": args.output_len,
        "mean_balancedness": sum(balancedness) / len(balancedness),
        "counts": counts,
        "layers": layers,
    }
    output = Path(args.output)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[profile] mean balancedness={payload['mean_balancedness']:.4f}")
    print(f"[profile] wrote {output}")


if __name__ == "__main__":
    main()
