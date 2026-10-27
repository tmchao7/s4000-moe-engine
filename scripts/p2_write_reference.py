"""Write a per-attention-backend greedy reference for p2_benchmark verification.

The committed ``benchmarks/p2-reference-8tok.json`` is only valid for the stack
it was generated on.  Switching attention backends changes bf16 reduction order,
so paged needs its own frozen reference before it can become the default
(docs/P6-numerics-policy.md §4).

Format is a bare list so ``scripts/p2_benchmark.py::run_verification`` can read
it unchanged: ``[{"prompt": ..., "token_ids": [...]}, ...]``.

Run (30B, one load):
  PYTHONPATH=/root/moe-server-s4000 python3 scripts/p2_write_reference.py \
      --attention-backend triton_paged \
      --output benchmarks/p2-reference-8tok-paged.json
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MODEL = "/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8"


def verify_prompts():
    spec = importlib.util.spec_from_file_location(
        "p2_benchmark", REPO / "scripts" / "p2_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VERIFY_PROMPTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--attention-backend", default="triton_paged",
                        choices=("sdpa", "triton_paged"))
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from nanovllm import LLM, SamplingParams

    prompts = verify_prompts()
    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=2,
        expert_parallel_size=2,
        moe_backend="grouped_mm",
        attention_backend=args.attention_backend,
        scheduler_policy="fcfs",
        max_model_len=1024,
        gpu_memory_utilization=0.85,
    )
    try:
        params = SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True)
        outputs = llm.generate(prompts, params, use_tqdm=False)
        rows = [
            {"prompt": prompt, "token_ids": output["token_ids"]}
            for prompt, output in zip(prompts, outputs)
        ]
    finally:
        llm.exit()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    print(f"[ref] attention={args.attention_backend} wrote {out}", flush=True)
    for row in rows:
        print(f"[ref] {row['token_ids']}", flush=True)


if __name__ == "__main__":
    main()
