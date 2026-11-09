"""P30-A: 8k concurrency/SLO probe for the 30B production stack.

Loads 30B once, submits 18 independent ~7.6k-token prompts, and records TTFT/e2e/
aggregate throughput. Random token ids avoid prefix-cache reuse, so this is a
worst-case prefill/queue test rather than a document-QA test.
"""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

from nanovllm import LLM, SamplingParams
from nanovllm.benchmarking import percentile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--requests", type=int, default=18)
    parser.add_argument("--prompt-len", type=int, default=7600)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--output", default="/root/p30_data/longctx_slo.json")
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
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
        max_model_len=8192,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=0.85,
    )
    rng = random.Random(args.seed)
    try:
        vocab = llm.model_runner.config.hf_config.vocab_size
        prompts = [[rng.randrange(vocab) for _ in range(args.prompt_len)]
                   for _ in range(args.requests)]
        params = SamplingParams(temperature=0, max_tokens=args.max_tokens,
                                ignore_eos=True)
        t0 = time.perf_counter()
        outs = llm.generate(prompts, params, use_tqdm=False)
        wall = time.perf_counter() - t0
        ttfts = [o["metrics"]["ttft_s"] for o in outs if o.get("metrics")]
        e2es = [o["metrics"]["e2e_s"] for o in outs if o.get("metrics")]
        out_tokens = sum(len(o["token_ids"]) for o in outs)
        payload = {
            "schema_version": 1,
            "requests": args.requests,
            "prompt_len": args.prompt_len,
            "max_tokens": args.max_tokens,
            "wall_s": round(wall, 3),
            "output_tokens": out_tokens,
            "aggregate_output_tok_s": round(out_tokens / wall, 2),
            "ttft_s": {
                "p50": round(statistics.median(ttfts), 3),
                "p95": round(percentile(ttfts, 0.95), 3),
                "max": round(max(ttfts), 3),
            },
            "e2e_s": {
                "p50": round(statistics.median(e2es), 3),
                "p95": round(percentile(e2es, 0.95), 3),
                "max": round(max(e2es), 3),
            },
        }
    finally:
        llm.exit()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("[p30-longctx-slo] wrote", args.output)


if __name__ == "__main__":
    main()
