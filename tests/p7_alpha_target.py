"""P7 Phase 0b — 30B greedy reference outputs for acceptance-rate measurement.

Generates 128 greedy tokens for 32 diverse chat prompts on the ADOPTED stack
(sdpa + grouped_mm prefill + bmm_dense decode, TP2/EP2, eager) and saves the
token ids.  Phase 0c replays the same prompts on the draft candidates and
compares longest-common-prefix lengths against this reference.

Run:  PYTHONPATH=/root/moe-server-s4000 python3 tests/p7_alpha_target.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/moe-server-s4000")

MODEL = "/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8"

PROMPTS = [
    "简单介绍一下 MoE（Mixture of Experts）架构的核心思想。",
    "What is paged attention in LLM inference? Answer in two sentences.",
    "1 到 100 的整数中所有质数之和是多少？给出答案即可。",
    "请用 Python 写一个快速排序，附注释。",
    "解释一下 transformer 里的多头注意力为什么要除以 sqrt(d_k)。",
    "Write a SQL query that finds the top 3 customers by revenue per region.",
    "把下面这段话翻译成英文：大模型的推理成本主要来自显存带宽而不是计算。",
    "Explain the difference between throughput and latency in serving systems.",
    "用中文解释什么是 KV cache，以及为什么它在解码阶段如此重要。",
    "Debug this loop: for i in range(n): total += i*2 — why might it be slow for large n?",
    "简述 TCP 三次握手的过程。",
    "Give three practical tips for reducing GPU memory usage when fine-tuning LLMs.",
    "写一个 bash 一行命令，统计当前目录下所有 .py 文件的总行数。",
    "Explain what speculative decoding is and why it preserves output distribution.",
    "比较 RMSNorm 和 LayerNorm 的区别。",
    "Write a Python function to check if a string is a palindrome, ignoring punctuation.",
    "为什么 Claude/GPT 这类模型的首字延迟通常比后续 token 延迟高很多？",
    "Summarize the trade-offs between FP8 and BF16 quantization for inference.",
    "用冒泡排序解释算法的时间复杂度分析。",
    "What is the difference between tensor parallelism and pipeline parallelism?",
    "给我一个 Git rebase 和 merge 的使用场景对比。",
    "Explain what a mixture-of-experts router is and how top-k routing works.",
    "解释一下泊松到达过程在压测里的用途。",
    "Write a regex that matches an IPv4 address, and explain each part.",
    "中文分词为什么对早期 NLP 很重要？",
    "Explain how prefix caching helps an LLM serving system.",
    "用 Python 实现一个 LRU 缓存类，支持 get 和 put。",
    "What causes priority inversion, and how do mutex priority inheritance help?",
    "简述一下 SLO 在在线推理服务里的常见定义。",
    "Explain the observer pattern with a tiny Python example.",
    "把『工欲善其事，必先利其器』翻译成英文并解释含义。",
    "Why do MoE models have higher memory footprint than dense models of equal active parameters?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", default="/root/p7_target_ref.json")
    args = parser.parse_args()

    from nanovllm import LLM, SamplingParams
    from nanovllm.layers.rotary_embedding import get_rope

    get_rope.cache_clear()
    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=2,
        expert_parallel_size=2,
        moe_backend="grouped_mm",
        decode_moe_backend="bmm_dense",
        attention_backend="sdpa",
        scheduler_policy="fcfs",
        max_model_len=1024,
        gpu_memory_utilization=0.85,
    )
    try:
        params = SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True)
        outputs = llm.generate(PROMPTS, params, use_tqdm=False)
        rows = []
        for prompt, output in zip(PROMPTS, outputs):
            rows.append({
                "prompt": prompt,
                "prompt_tokens": output["metrics"]["prompt_tokens"],
                "token_ids": list(output["token_ids"]),
                "itl_p50_s": output["metrics"].get("itls_s", [None])[0] if output["metrics"].get("itls_s") else None,
                "itls_s": output["metrics"]["itls_s"],
            })
        payload = {
            "model": args.model,
            "stack": "sdpa+grouped_mm_prefill+bmm_dense_decode",
            "max_tokens": args.max_tokens,
            "rows": rows,
        }
        Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[p7b] wrote {args.output} ({len(rows)} prompts)", flush=True)
        for row in rows[:4]:
            head = row["token_ids"][:8]
            print(f"[p7b] {row['prompt'][:24]!r} -> head={head}", flush=True)
    finally:
        llm.exit()
        get_rope.cache_clear()


if __name__ == "__main__":
    main()
