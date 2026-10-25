"""P2 full-model run: Qwen3-30B-A3B FP8, TP=2 on 2x S4000.

    PYTHONPATH=/root/moe-server-s4000 python3 tests/p2_full.py

1. Greedy generation on fixed prompts (coherence + math sanity).
2. Benchmark at 32-seq random load (mirrors bench.py methodology).
"""
import argparse
import time
from random import randint, seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    ap.add_argument("--bench-seqs", type=int, default=32)
    ap.add_argument("--bench-max-len", type=int, default=512)
    args = ap.parse_args()

    from nanovllm import LLM, SamplingParams
    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=2,
              max_model_len=2048, gpu_memory_utilization=0.85)

    prompts = [
        "简单介绍一下 MoE（Mixture of Experts）架构的核心思想。",
        "What is paged attention in LLM inference? Answer in two sentences.",
        "1 到 100 的整数中所有质数之和是多少？给出答案即可。",
        "请用 Python 写一个快速排序，附注释。",
    ]
    sp = SamplingParams(temperature=0, max_tokens=96)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    results = []
    for p, o in zip(prompts, outs):
        print(f"[p2] {p[:24]!r} -> {o['text'][:120]!r}", flush=True)
        results.append({"prompt": p, "output": o["text"], "token_ids": o["token_ids"]})
    import json
    with open("/root/p2_outputs.json", "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)

    # benchmark
    seed(0)
    prompt_ids = [[randint(0, 10000) for _ in range(randint(100, args.bench_max_len))]
                  for _ in range(args.bench_seqs)]
    sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, args.bench_max_len))
           for _ in range(args.bench_seqs)]
    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_ids, sps, use_tqdm=False)
    t = time.time() - t
    total = sum(s.max_tokens for s in sps)
    print(f"[p2] bench batch={args.bench_seqs}: total={total}tok time={t:.2f}s "
          f"throughput={total / t:.2f}tok/s", flush=True)
    with open("/root/p2_bench.json", "w") as f:
        json.dump({"batch": args.bench_seqs, "total_tokens": total, "time_s": t,
                   "throughput": total / t}, f, indent=1)


if __name__ == "__main__":
    main()
