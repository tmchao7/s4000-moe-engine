"""P7/B' — where does the 30B decode step go? (TP2/EP2, adopted stack, eager)

Split of one decode step into prepare / forward / sample on the production stack,
so the "per-step fixed cost" finding from the 0.6B probes (ISSUES I19) can be
quantified for the real model.  No graph capture (enforce_eager=True) — the graph
path is broken (I1) and must not be exercised on 30B.

Run:  PYTHONPATH=/root/moe-server-s4000 python3 tests/p7_step_split_30b.py
"""
import argparse
import importlib.util
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/root/moe-server-s4000")

REPO = Path("/root/moe-server-s4000")
MODEL = "/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
SEED = 20261027


def load_step_overhead():
    spec = importlib.util.spec_from_file_location("p6_oh", REPO / "tests" / "p6_step_overhead.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--batches", default="1,8,32")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output", default="/root/p7_step_split_30b.json")
    args = parser.parse_args()

    from nanovllm import LLM, SamplingParams

    oh = load_step_overhead()
    batches = [int(b) for b in args.batches.split(",")]

    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=2,
        expert_parallel_size=2,
        moe_backend="grouped_mm",
        decode_moe_backend="bmm_dense",
        attention_backend="sdpa",
        scheduler_policy="fcfs",
        max_num_seqs=max(batches),
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )
    runner = llm.model_runner
    install_timed_run = oh.install_timed_run
    phases = install_timed_run(runner)
    vocab = runner.config.hf_config.vocab_size
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True)

    records = []
    try:
        for batch in batches:
            rng = random.Random(SEED + batch)
            prompts = [[rng.randrange(vocab) for _ in range(args.prompt_len)] for _ in range(batch)]
            llm.generate(prompts, params, use_tqdm=False)  # warmup
            llm.generate(prompts, params, use_tqdm=False)
            # ``phases`` accumulates one entry per engine step (prefill + decode);
            # at 32 output tokens the decode steps dominate, so per-phase medians
            # over the whole run are a faithful decode-step split.
            prep = statistics.median(phases["prepare"]) if phases["prepare"] else None
            fwd = statistics.median(phases["forward"]) if phases["forward"] else None
            smp = statistics.median(phases["sample"]) if phases["sample"] else None
            total = (prep or 0) + (fwd or 0) + (smp or 0)
            record = {
                "batch": batch,
                "prepare_median_s": prep,
                "forward_median_s": fwd,
                "sample_median_s": smp,
                "step_total_median_s": total,
                "phase_share": {
                    "prepare": (prep or 0) / total if total else None,
                    "forward": (fwd or 0) / total if total else None,
                    "sample": (smp or 0) / total if total else None,
                },
                "steps_counted": len(phases["forward"]),
            }
            phases["prepare"].clear()
            phases["forward"].clear()
            phases["sample"].clear()
            records.append(record)
            print("[p7split]", json.dumps(record, sort_keys=True), flush=True)
    finally:
        llm.exit()

    Path(args.output).write_text(json.dumps({"records": records, "phases": phases}, indent=2), encoding="utf-8")
    print(f"[p7split] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
