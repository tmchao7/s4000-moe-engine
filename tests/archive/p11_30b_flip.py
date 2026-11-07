"""Flip-rate probe (P11 P4-30B numerics): 30B MoE greedy decode is not
bitwise run-to-run deterministic (observed: eager arm tokens_consistent=False
at b1/b8/b32, suspected atomic-add nondeterminism in MoE kernels).  The
frozen "token 全等" gate is therefore unachievable-as-written at 30B; this
probe quantifies flip rates so the numerics verdict becomes: cross-arm
(eager vs segment) flip rate ~= within-arm (eager vs eager) flip rate
-> numerics equivalent; cross >> within -> segment introduces drift.

One mode per process (same build as p11_30b_ab); the caller runs both arms
with the same build_prompts seed and compares offline.
"""
import argparse
import json
import sys
import time

sys.path.insert(0, "/root/moe-server-s4000")
sys.path.insert(0, "/root/moe-server-s4000/tests")

from nanovllm import LLM  # noqa: E402
from p8_moe_30b import build_prompts, drive_batch  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eager", "segment"], required=True)
    parser.add_argument("--model",
                        default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--output", default="/root/p11_30b_flip.json")
    args = parser.parse_args()

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
    prompts = build_prompts(llm.tokenizer, 512)
    payload = {"kind": "p11_30b_flip", "mode": args.mode, "batches": {}}

    for batch in [int(item) for item in args.batches.split(",")]:
        subset = prompts[:batch]
        runs = []
        for trial in range(args.trials):
            tokens, decode_wall, _prefill = drive_batch(llm, subset, 64)
            runs.append(tokens)
            print(f"[flip] mode={args.mode} b{batch} trial={trial} "
                  f"decode={decode_wall * 1000:.2f}ms", flush=True)
        flips_per_seq = []
        for seq_index in range(batch):
            reference = runs[0][seq_index]
            flips = []
            for trial in range(1, args.trials):
                other = runs[trial][seq_index]
                length = min(len(reference), len(other))
                flips.append(sum(
                    1 for pos in range(length) if reference[pos] != other[pos]
                ))
            flips_per_seq.append(flips)
        payload["batches"][f"b{batch}"] = {
            "token_runs": runs,
            "flips_vs_run0_per_seq": flips_per_seq,
        }

    payload["finished"] = time.time()
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    print(f"[flip] mode={args.mode} wrote {args.output}", flush=True)
    llm.exit()


if __name__ == "__main__":
    main()
