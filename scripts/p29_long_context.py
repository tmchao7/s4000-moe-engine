"""P29-H: long-context / KV-capacity probe on the 30B production stack.

Loads the model once with max_model_len large enough for the requested contexts,
then for each context: prefill one prompt, decode 8 steps, record wall time and
memory. No serving default is changed; this is a capability probe.
"""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

from nanovllm import LLM, SamplingParams


def run_context(llm, context, decode_steps, rng, vocab):
    prompt = [rng.randrange(vocab) for _ in range(context)]
    params = SamplingParams(temperature=0, max_tokens=decode_steps + 2,
                            ignore_eos=True)
    seq_id = llm.add_request(prompt, params)
    t0 = time.perf_counter()
    while llm.scheduler.waiting:
        llm.step()
    prefill_s = time.perf_counter() - t0
    clean = []
    for _ in range(decode_steps):
        if llm.is_finished():
            break
        t = time.perf_counter()
        llm.step()
        clean.append((time.perf_counter() - t) * 1e3)
    llm.cancel_request(seq_id)
    llm.scheduler.block_manager.clear_prefix_cache()
    return {
        "context": context,
        "prefill_s": round(prefill_s, 3),
        "prefill_tok_s": round(context / prefill_s, 2) if prefill_s else None,
        "decode_steps": len(clean),
        "decode_median_ms": round(statistics.median(clean), 3) if clean else None,
        "peak_mib": round(torch.musa.max_memory_allocated() / 2**20, 1)
        if hasattr(torch.musa, "max_memory_allocated") else None,
    }


def main():
    global torch
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--contexts", default="1024,4096,8192,16384")
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--output", default="/root/p29_data/long_context.json")
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    contexts = [int(x) for x in args.contexts.split(",")]
    max_context = max(contexts)
    max_model_len = max_context + args.decode_steps + 8

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
        max_model_len=max_model_len,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=0.85,
    )
    rng = random.Random(args.seed)
    vocab = llm.model_runner.config.hf_config.vocab_size
    records = []
    try:
        for context in contexts:
            print(f"[p29-longctx] context={context} start", flush=True)
            rec = run_context(llm, context, args.decode_steps, rng, vocab)
            records.append(rec)
            print(f"[p29-longctx] context={context} {rec}", flush=True)
    finally:
        llm.exit()

    payload = {
        "schema_version": 1,
        "config": {
            "tp": 2, "ep": 2, "moe_backend": "grouped_mm",
            "decode_moe_backend": "bmm_dense",
            "attention_backend": "triton_paged",
            "scheduler_policy": "stall_free",
            "prefill_chunk_size": 256, "enforce_eager": True,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": 16384,
        },
        "records": records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[p29-longctx] wrote", args.output)


if __name__ == "__main__":
    main()
