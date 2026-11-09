"""P29-E: linear vs round_robin expert placement A/B + 30B greedy token gate.

One process per placement (placement is fixed at load time). Each process:
1. greedy 8-token generation on the frozen 4-prompt reference;
2. repeated clean decode-step timing at b1/b8/b32, 128-token prompts.

No serving default is changed. Run:
    PYTHONPATH=. python3 scripts/p29_rr_ab.py --placement linear --output /root/p29_data/rr_ab_linear.json
    PYTHONPATH=. python3 scripts/p29_rr_ab.py --placement round_robin --output /root/p29_data/rr_ab_rr.json
"""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

from nanovllm import LLM, SamplingParams

REFERENCE = Path(__file__).resolve().parents[1] / "benchmarks" / "p2-reference-8tok.json"


def parse_workloads(value):
    out = []
    for item in value.split(","):
        batch, context = item.lower().split("x")
        out.append((int(batch), int(context)))
    return out


def token_gate(llm, reference):
    llm.scheduler.block_manager.clear_prefix_cache()
    params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    outputs = llm.generate([row["prompt"] for row in reference], params,
                           use_tqdm=False)
    got = [out["token_ids"] for out in outputs]
    expected = [row["token_ids"] for row in reference]
    return {
        "pass": got == expected,
        "got": got,
        "expected": expected,
    }


def measure(llm, workloads, decode_steps, repeats, seed):
    rng = random.Random(seed)
    vocab = llm.model_runner.config.hf_config.vocab_size
    records = []
    for batch, context in workloads:
        per_repeat = []
        for rep in range(repeats):
            prompts = [[rng.randrange(vocab) for _ in range(context)]
                       for _ in range(batch)]
            params = SamplingParams(temperature=0, max_tokens=decode_steps + 4,
                                    ignore_eos=True)
            ids = [llm.add_request(p, params) for p in prompts]
            while llm.scheduler.waiting:
                llm.step()
            clean = []
            for _ in range(decode_steps):
                if llm.is_finished():
                    break
                t0 = time.perf_counter()
                llm.step()
                clean.append((time.perf_counter() - t0) * 1e3)
            per_repeat.append({
                "median_step_ms": round(statistics.median(clean), 4) if clean else None,
                "steps_ms": [round(x, 4) for x in clean],
            })
            for seq_id in ids:
                llm.cancel_request(seq_id)
            llm.scheduler.block_manager.clear_prefix_cache()
        medians = [r["median_step_ms"] for r in per_repeat
                   if r["median_step_ms"] is not None]
        records.append({
            "batch": batch, "context": context,
            "repeats": per_repeat,
            "median_of_medians_ms": round(statistics.median(medians), 4)
            if medians else None,
        })
        print(f"[p29-rr-ab] {batch}x{context} median_of_medians="
              f"{records[-1]['median_of_medians_ms']}", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--placement", required=True, choices=("linear", "round_robin"))
    parser.add_argument("--workloads", default="1x128,8x128,32x128")
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    workloads = parse_workloads(args.workloads)

    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=2,
        expert_parallel_size=2,
        expert_placement=args.placement,
        moe_backend="grouped_mm",
        decode_moe_backend="bmm_dense",
        attention_backend="triton_paged",
        scheduler_policy="stall_free",
        prefill_chunk_size=256,
        max_model_len=max(c for _, c in workloads) + args.decode_steps + 8,
        gpu_memory_utilization=0.85,
    )
    try:
        gate = token_gate(llm, reference)
        print(f"[p29-rr-ab] token_gate={gate['pass']}", flush=True)
        perf = measure(llm, workloads, args.decode_steps, args.repeats, args.seed)
    finally:
        llm.exit()

    payload = {
        "schema_version": 1,
        "placement": args.placement,
        "config": {
            "tp": 2, "ep": 2, "expert_placement": args.placement,
            "moe_backend": "grouped_mm", "decode_moe_backend": "bmm_dense",
            "attention_backend": "triton_paged", "scheduler_policy": "stall_free",
            "prefill_chunk_size": 256, "enforce_eager": True,
        },
        "token_gate": gate,
        "perf": perf,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[p29-rr-ab] wrote", args.output)


if __name__ == "__main__":
    main()
