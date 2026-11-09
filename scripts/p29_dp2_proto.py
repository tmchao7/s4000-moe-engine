"""P29-F: 0.6B two-replica (DP=2) vs one TP2 process prototype.

This is an architecture probe, not a 30B benchmark: it checks whether two
independent TP1 replicas on two S4000s can aggregate more decode throughput than
one TP2 process, and whether MUSA_VISIBLE_DEVICES isolation works. It uses Qwen3-0.6B
so the load is cheap; 30B feasibility is a paper calculation from Day2 memory data.
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import time
from pathlib import Path


def _worker(rank, mode, model, prompts, max_tokens, start, ready, q):
    os.environ["MUSA_VISIBLE_DEVICES"] = "0,1" if mode == "tp2" else str(rank)
    from nanovllm import LLM, SamplingParams

    tp = 2 if mode == "tp2" else 1
    llm = LLM(
        model,
        enforce_eager=True,
        tensor_parallel_size=tp,
        expert_parallel_size=tp,
        attention_backend="triton_paged",
        max_model_len=1024,
        gpu_memory_utilization=0.6,
    )
    try:
        params = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
        # Warmup, then clear the prefix cache so the timed run is not a cache hit.
        llm.generate(prompts[:1], SamplingParams(temperature=0, max_tokens=4,
                                                  ignore_eos=True), use_tqdm=False)
        llm.scheduler.block_manager.clear_prefix_cache()
        ready.set()
        start.wait()
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - t0
        q.put({
            "rank": rank,
            "elapsed_s": elapsed,
            "tokens": sum(len(o["token_ids"]) for o in outputs),
            "first_tokens": [o["token_ids"][:4] for o in outputs[:2]],
        })
    finally:
        llm.exit()


def run_mode(mode, args):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    start = ctx.Event()
    ready = [ctx.Event() for _ in range(2 if mode == "dp2" else 1)]
    rng = random.Random(args.seed)
    prompts = [[rng.randrange(1000, 150000) for _ in range(args.prompt_len)]
               for _ in range(args.requests)]
    workers = []
    if mode == "tp2":
        p = ctx.Process(target=_worker, args=(0, "tp2", args.model, prompts,
                                              args.max_tokens, start, ready[0], q))
        p.start(); workers.append(p)
    else:
        half = len(prompts) // 2
        for rank in range(2):
            shard = prompts[:half] if rank == 0 else prompts[half:]
            p = ctx.Process(target=_worker, args=(rank, "dp2", args.model, shard,
                                                  args.max_tokens, start, ready[rank], q))
            p.start(); workers.append(p)
    for ev in ready:
        if not ev.wait(timeout=600):
            raise RuntimeError(f"worker not ready for {mode}")
    t0 = time.perf_counter()
    start.set()
    results = [q.get(timeout=600) for _ in workers]
    wall = time.perf_counter() - t0
    for p in workers:
        p.join(timeout=60)
        if p.exitcode != 0:
            raise RuntimeError(f"worker exit {p.exitcode} for {mode}")
    total_tokens = sum(r["tokens"] for r in results)
    return {
        "mode": mode,
        "requests": len(prompts),
        "max_tokens": args.max_tokens,
        "prompt_len": args.prompt_len,
        "wall_s": round(wall, 4),
        "total_tokens": total_tokens,
        "aggregate_tok_s": round(total_tokens / wall, 2) if wall else None,
        "per_replica": [{**r, "elapsed_s": round(r["elapsed_s"], 4)}
                        for r in results],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    parser.add_argument("--mode", required=True, choices=("tp2", "dp2"))
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = run_mode(args.mode, args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
