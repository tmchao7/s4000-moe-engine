"""P1 acceptance: nanovllm (MUSA) vs HF transformers greedy alignment on Qwen3-8B.

    python3 tests/p1_align.py --model /root/models/Qwen3-8B --max-tokens 64

Runs HF reference first (bf16, card 0), frees it, then runs nanovllm
(enforce_eager) and compares token ids. Metrics: first-token hit rate and
prefix hit rate over max_tokens.
"""
import argparse
import gc
import json

PROMPTS = [
    "简单介绍一下 MoE（Mixture of Experts）架构的核心思想。",
    "What is paged attention in LLM inference? Answer in two sentences.",
    "1 到 100 的整数中所有质数之和是多少？给出答案即可。",
    "请用 Python 写一个快速排序，附注释。",
    "The quick brown fox",
]


def hf_reference(model_path, max_tokens):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    outs = []
    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids.to(model.device)
            gen = model.generate(ids, max_new_tokens=max_tokens, do_sample=False)
            outs.append(gen[0][ids.shape[1]:].tolist())
    del model
    gc.collect()
    torch.musa.empty_cache() if torch.musa.is_available() else torch.cuda.empty_cache()
    return outs


def nanovllm_run(model_path, max_tokens):
    from nanovllm import LLM, SamplingParams
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1,
              max_model_len=2048, gpu_memory_utilization=0.75)
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)
    outs = llm.generate(PROMPTS, sp, use_tqdm=False)
    return [o["token_ids"] for o in outs], llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/Qwen3-8B")
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    print("[align] running HF reference...", flush=True)
    ref = hf_reference(args.model, args.max_tokens)
    print("[align] running nanovllm (musa)...", flush=True)
    ours, llm = nanovllm_run(args.model, args.max_tokens)
    llm.exit()

    rows = []
    for p, r, o in zip(PROMPTS, ref, ours):
        n = min(len(r), len(o))
        first = int(r[0] == o[0]) if n else 0
        prefix = 0
        for a, b in zip(r, o):
            if a != b:
                break
            prefix += 1
        rows.append({"prompt": p[:30], "first_token": bool(first),
                     "prefix_hit": prefix, "max_tokens": n,
                     "ref_head": r[:8], "ours_head": o[:8]})
        print(f"[align] {p[:24]!r}: first={bool(first)} prefix={prefix}/{n}", flush=True)

    first_rate = sum(r["first_token"] for r in rows) / len(rows)
    prefix_rate = sum(r["prefix_hit"] == r["max_tokens"] for r in rows) / len(rows)
    mean_prefix = sum(r["prefix_hit"] for r in rows) / sum(r["max_tokens"] for r in rows)
    result = {"model": args.model, "first_token_rate": first_rate,
              "full_match_rate": prefix_rate, "mean_prefix_hit": mean_prefix, "rows": rows}
    print("[align]", json.dumps({k: v for k, v in result.items() if k != "rows"}))
    with open("/root/p1_align.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("saved /root/p1_align.json")


if __name__ == "__main__":
    main()
