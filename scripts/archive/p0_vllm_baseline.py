"""P0 baseline: vllm-musa TP2 throughput on Qwen3-8B (dense, bf16).

Run on AutoDL S4000 instance:
    python3 /root/p0_vllm_baseline.py --model /root/models/Qwen3-8B \
        --concurrencies 1 8 32 128 --max-tokens 512

Metrics: total throughput + single-request latency; TTFT/TPOT via manual
timer for concurrency=1. Results appended to /root/p0_vllm_baseline.json.
"""
import argparse
import json
import random
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def rand_prompt(tokenizer, lo, hi, rng):
    n = rng.randint(lo, hi)
    ids = rng.sample(range(1000, tokenizer.vocab_size - 1), min(n, 30_000))
    return tokenizer.decode(ids[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/Qwen3-8B")
    ap.add_argument("--concurrencies", type=int, nargs="+", default=[1, 8, 32, 128])
    ap.add_argument("--input-lo", type=int, default=100)
    ap.add_argument("--input-hi", type=int, default=1024)
    ap.add_argument("--output-lo", type=int, default=100)
    ap.add_argument("--output-hi", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--enforce-eager", action="store_true", default=True)
    args = ap.parse_args()

    rng = random.Random(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=2,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
    )
    results = {"model": args.model, "tp": 2, "records": []}

    for c in args.concurrencies:
        prompts = [rand_prompt(tokenizer, args.input_lo, args.input_hi, rng) for _ in range(c)]
        outs_len = [rng.randint(args.output_lo, args.output_hi) for _ in range(c)]
        sps = [SamplingParams(temperature=0.6, max_tokens=m) for m in outs_len]
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sps)
        dt = time.perf_counter() - t0
        gen = sum(len(o.outputs[0].token_ids) for o in outs)
        rec = {
            "concurrency": c,
            "wall_s": round(dt, 2),
            "generated_tokens": gen,
            "throughput_tok_s": round(gen / dt, 1),
        }
        results["records"].append(rec)
        print("[P0]", json.dumps(rec), flush=True)

    with open("/root/p0_vllm_baseline.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("saved /root/p0_vllm_baseline.json")


if __name__ == "__main__":
    main()
