"""P7 Phase 0a — draft-model decode step timing (no 30B load).

Measures per-step decode latency (itl_p50) for the speculative-decoding draft
candidates (Qwen3-0.6B / Qwen3-1.7B) on TP2, eager, greedy — the same
conditions under which a draft would run inside the engine.  Combined with the
frozen 30B target step times this yields the cost ratio
``c = draft_step / target_step`` used by the go/no-go gate in
docs/P7-speculative-decoding-survey.md §6.

Run:  PYTHONPATH=/root/moe-server-s4000 python3 tests/p7_draft_steps.py \
          --models /root/autodl-tmp/models/Qwen3-0.6B,/root/autodl-tmp/models/Qwen3-1.7B
"""
import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/root/moe-server-s4000")

from nanovllm.benchmarking import percentile  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="/root/autodl-tmp/models/Qwen3-0.6B,/root/autodl-tmp/models/Qwen3-1.7B")
    parser.add_argument("--batches", default="1,8,32")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--output", default="/root/p7_draft_steps.json")
    args = parser.parse_args()

    from nanovllm import LLM, SamplingParams
    from nanovllm.layers.rotary_embedding import get_rope

    batches = [int(b) for b in args.batches.split(",")]
    results = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        name = Path(model).name
        get_rope.cache_clear()
        llm = LLM(
            model,
            enforce_eager=True,
            tensor_parallel_size=2,
            attention_backend="sdpa",
            scheduler_policy="fcfs",
            max_model_len=args.prompt_len + args.max_tokens + 16,
            gpu_memory_utilization=0.5,
        )
        try:
            vocab = llm.model_runner.config.hf_config.vocab_size
            params = SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True)
            for batch in batches:
                rng = random.Random(args.seed + batch)
                prompts = [[rng.randrange(vocab) for _ in range(args.prompt_len)] for _ in range(batch)]
                llm.generate(prompts, params, use_tqdm=False)  # warmup
                itls, tpots = [], []
                for _ in range(3):
                    outputs = llm.generate(prompts, params, use_tqdm=False)
                    for output in outputs:
                        itls.extend(output["metrics"]["itls_s"])
                        tpots.append(output["metrics"]["tpot_s"])
                record = {
                    "model": name,
                    "batch": batch,
                    "prompt_len": args.prompt_len,
                    "itl_p50_s": percentile(itls, 0.50),
                    "itl_p95_s": percentile(itls, 0.95),
                    "tpot_p50_s": statistics.median(tpots),
                }
                results.append(record)
                print("[p7a]", json.dumps(record, sort_keys=True), flush=True)
        finally:
            llm.exit()
            get_rope.cache_clear()

    Path(args.output).write_text(json.dumps({"records": results}, indent=2), encoding="utf-8")
    print(f"[p7a] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
