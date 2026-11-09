"""P30-A: real 8k needle-in-a-haystack quality gate for the 30B server.

Pre-registered gate: 9/9 exact code match at context ~7.6k tokens, needle at
10/50/90% depth, 3 unique codes. Production stack, TP2/EP2, no serving default
change. This complements the Day8 random-token capability probe.
"""
import argparse
import json
import time
from pathlib import Path

from nanovllm import LLM, SamplingParams

BASE = (
    "The mixture-of-experts architecture routes each token to a small subset of "
    "experts. In a two-card deployment the experts are sharded across ranks, "
    "and the partial expert outputs are combined with an all-reduce. Paged "
    "attention stores the key and value cache in fixed-size blocks, which keeps "
    "the decode step independent of the sequence length. The scheduler fills a "
    "batch with prefill chunks and decode rows, while the KV cache grows by one "
    "token per decoding step. Consistent numerical behavior matters because "
    "greedy decoding can flip on a near tie. "
)
QUESTION = "\nQuestion: What is the secret code? Answer with the code only.\n"


def build_prompt(tok, code, depth, target_tokens):
    needle = tok.encode(f"\nThe secret code is {code}.\n", add_special_tokens=False)
    base = tok.encode(BASE, add_special_tokens=False)
    body = (base * ((target_tokens // len(base)) + 1))[:target_tokens - len(needle)]
    pos = int(len(body) * depth)
    body = body[:pos] + needle + body[pos:]
    question = tok.encode(QUESTION, add_special_tokens=False)
    return body + question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--target-context", type=int, default=7600)
    parser.add_argument("--output", default="/root/p30_data/longctx_quality.json")
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
    cases = []
    try:
        tok = llm.tokenizer
        for code in (4821, 7395, 1568):
            for depth in (0.1, 0.5, 0.9):
                prompt = build_prompt(tok, code, depth, args.target_context)
                cases.append({"code": code, "depth": depth, "prompt": prompt})
        params = SamplingParams(temperature=0, max_tokens=12, ignore_eos=True)
        t0 = time.perf_counter()
        outputs = llm.generate([c["prompt"] for c in cases], params, use_tqdm=False)
        elapsed = time.perf_counter() - t0
        records = []
        for case, out in zip(cases, outputs):
            text = out["text"]
            ok = str(case["code"]) in text
            records.append({
                "code": case["code"], "depth": case["depth"],
                "prompt_tokens": len(case["prompt"]),
                "output": text[:200], "pass": ok,
            })
        passed = sum(r["pass"] for r in records)
        payload = {
            "schema_version": 1,
            "gate": "9/9 exact code match",
            "target_context": args.target_context,
            "elapsed_s": round(elapsed, 3),
            "passed": passed, "total": len(records),
            "all_pass": passed == len(records),
            "records": records,
        }
    finally:
        llm.exit()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("[p30-longctx-quality] wrote", args.output)
    if not payload["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
