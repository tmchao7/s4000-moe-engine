"""P1 benchmark: nanovllm (MUSA, single card) on Qwen3-8B dense.

    PYTHONPATH=/root/moe-server-s4000 python3 scripts/p1_bench.py --model /root/models/Qwen3-8B

Mirrors upstream bench.py methodology (random 100-1024 token prompts/outputs,
temperature=0.6, ignore_eos) at reduced sequence count for S4000.
"""
import argparse
import time
from random import randint, seed

from nanovllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/Qwen3-8B")
    ap.add_argument("--num-seqs", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    seed(0)
    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1,
              max_model_len=2048, gpu_memory_utilization=0.75)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, args.max_len))]
                        for _ in range(args.num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True,
                                      max_tokens=randint(100, args.max_len))
                       for _ in range(args.num_seqs)]

    llm.generate(["Benchmark: "], SamplingParams())  # warmup
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    print(f"[P1-bench] batch={args.num_seqs}: total={total_tokens}tok time={t:.2f}s "
          f"throughput={total_tokens / t:.2f}tok/s", flush=True)

    # single-stream decode
    sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=256)
    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(["Count from 1 to 500 in English words."], sp, use_tqdm=False)
    t = time.time() - t
    print(f"[P1-bench] single-stream: 256tok time={t:.2f}s decode={256 / t:.2f}tok/s", flush=True)


if __name__ == "__main__":
    main()
