"""P6/L3 — attention 后端输出质量等价性（100 条真实 prompt 双后端对拍）.

The frozen gate lives in ``scripts/p6_decide_attention_equivalence.py`` and is
derived from docs/P6-numerics-policy.md §3:

  L3  edit distance p95 <= 8% of characters; first-token agreement >= 99%;
      no degenerate (looping) outputs.
  L4  each backend deterministic across 3 repeats (token-identical).

Token-level agreement is REPORTED but never gated: it compounds over 128 decode
steps (0.21% per step -> ~6.5% per request), so it is not a quality measure.

One 30B load; both backends run on the same prompts via set_attention_backend.

Run:  PYTHONPATH=/root/moe-server-s4000 python3 tests/p6_attention_equivalence.py
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/moe-server-s4000")

MODEL = "/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
SEED = 20261027

# 100 prompts: 40 zh, 40 en, 10 code, 10 math/reasoning.
PROMPTS_ZH = [
    "简单介绍一下 MoE 架构的核心思想。",
    "解释一下 transformer 里的多头注意力为什么要除以 sqrt(d_k)。",
    "用中文解释什么是 KV cache，以及为什么它在解码阶段如此重要。",
    "简述 TCP 三次握手的过程。",
    "比较 RMSNorm 和 LayerNorm 的区别。",
    "为什么大模型的首字延迟通常比后续 token 延迟高很多？",
    "解释一下泊松到达过程在压测里的用途。",
    "中文分词为什么对早期 NLP 很重要？",
    "简述一下 SLO 在在线推理服务里的常见定义。",
    "把『工欲善其事，必先利其器』翻译成英文并解释含义。",
    "解释 paged attention 要解决什么问题。",
    "为什么 MoE 模型的显存占用比同等激活参数的稠密模型高？",
    "解释一下 prefix caching 在推理服务中的作用。",
    "简述张量并行和流水线并行的区别。",
    "为什么 bf16 训练比 fp32 更省显存？",
    "解释一下 top-k 路由在 MoE 里怎么工作。",
    "简述为什么要做 KV cache 量化。",
    "解释一下 continuous batching 相比静态 batching 的优势。",
    "为什么推理引擎要做 chunked prefill？",
    "解释一下专家并行（EP）和模型并行的区别。",
    "简述在线服务的准入控制（admission control）要解决什么问题。",
    "解释为什么 GPU 上的 kernel launch 开销对小 batch 影响更大。",
    "简述 roofline 模型怎么判断一个算子是访存瓶颈还是计算瓶颈。",
    "解释一下为什么 attention 在长上下文下会变成主要开销。",
    "简述投机解码为什么能在不改变输出分布的前提下加速。",
    "解释一下什么是 prefill 和 decode 阶段的分离部署。",
    "为什么说 decode 阶段通常是访存密集型？",
    "简述 prefix cache 的块哈希是怎么做的。",
    "解释一下 CUDA graph 为什么能降低启动开销。",
    "为什么小模型做草稿模型不一定划算？",
    "简述如何用 little 定律估算服务并发。",
    "解释一下 TTFT 和 TPOT 的区别以及各自的优化手段。",
    "为什么 NCCL all-reduce 的延迟在小张量下几乎是常数？",
    "简述模型量化里 per-channel 和 per-tensor 的取舍。",
    "解释一下为什么 top-p 采样比 top-k 更自适应。",
    "简述如何检测生成中的重复循环。",
    "解释一下什么是注意力掩码的 causal 约束。",
    "为什么同一个模型在不同 batch 大小下贪心输出可能不同？",
    "简述 FP8 训练/推理里 scale 的作用。",
    "解释一下为什么长上下文推理需要分块处理 KV。",
]
PROMPTS_EN = [
    "What is paged attention in LLM inference? Answer in two sentences.",
    "Explain the difference between throughput and latency in serving systems.",
    "Explain what speculative decoding is and why it preserves output distribution.",
    "Summarize the trade-offs between FP8 and BF16 quantization for inference.",
    "What is the difference between tensor parallelism and pipeline parallelism?",
    "Explain what a mixture-of-experts router is and how top-k routing works.",
    "Explain how prefix caching helps an LLM serving system.",
    "What causes priority inversion, and how does mutex priority inheritance help?",
    "Why do MoE models have a higher memory footprint than dense models of equal active parameters?",
    "Explain the roofline model in one paragraph.",
    "What is the difference between prefill and decode in transformer inference?",
    "Explain why kernel launch overhead matters more at small batch sizes.",
    "Describe how a paged KV cache avoids fragmentation.",
    "What is continuous batching and why does it improve GPU utilization?",
    "Explain the difference between TTFT and TPOT and how to optimize each.",
    "Why is greedy decoding not necessarily reproducible across batch sizes?",
    "Explain floating point non-associativity and its effect on LLM outputs.",
    "What does a scheduler do in an LLM serving engine?",
    "Explain chunked prefill and when it helps.",
    "What is expert parallelism and how does it differ from tensor parallelism?",
    "Describe admission control for an inference service under SLO constraints.",
    "Explain why decode is memory-bandwidth bound while prefill is compute bound.",
    "What is the purpose of a block table in paged attention?",
    "Explain how a draft model can accelerate generation without changing the output distribution.",
    "What are the trade-offs of running a larger batch size?",
    "Explain what a KV cache eviction policy needs to consider.",
    "Describe how CUDA graphs reduce per-step overhead.",
    "Why might a smaller draft model be slower than expected in speculative decoding?",
    "Explain Little's law and how to use it for capacity planning.",
    "What is the difference between per-tensor and per-channel quantization?",
    "Explain how top-p sampling differs from top-k sampling.",
    "Describe a method to detect repetition loops in generated text.",
    "What is a causal attention mask and why is it needed?",
    "Explain why tensor parallel all-reduce latency is nearly constant for small tensors.",
    "What is the role of scaling factors in FP8 inference?",
    "Explain why long-context inference needs chunked KV processing.",
    "Describe how to measure tail latency in a serving benchmark.",
    "What is the difference between offline batch inference and online serving?",
    "Explain why expert load imbalance hurts MoE throughput.",
    "Describe how an inference engine decides when to preempt a request.",
]
PROMPTS_CODE = [
    "请用 Python 写一个快速排序，附注释。",
    "Write a Python function to check if a string is a palindrome, ignoring punctuation.",
    "用 Python 实现一个 LRU 缓存类，支持 get 和 put。",
    "Write a SQL query that finds the top 3 customers by revenue per region.",
    "写一个 bash 一行命令，统计当前目录下所有 .py 文件的总行数。",
    "Debug this loop: for i in range(n): total += i*2 - why might it be slow for large n?",
    "Write a regex that matches an IPv4 address, and explain each part.",
    "用 Python 写一个函数，统计一段文本里每个单词出现的次数。",
    "Explain the observer pattern with a tiny Python example.",
    "Write a Python generator that yields Fibonacci numbers up to n.",
]
PROMPTS_MATH = [
    "1 到 100 的整数中所有质数之和是多少？给出答案即可。",
    "A train travels 120 km in 1.5 hours. What is its average speed?",
    "用冒泡排序解释算法的时间复杂度分析。",
    "What is the sum of the first 20 positive integers?",
    "若 x + 2y = 10 且 x - y = 1，求 x 和 y。",
    "Compute 17 * 23 and explain the steps.",
    "一个袋子有 3 个红球 2 个蓝球，随机取两个，都是红球的概率是多少？",
    "If a die is rolled twice, what is the probability of getting two sixes?",
    "3 的 5 次方是多少？用过程说明。",
    "A store offers 20% off, then an extra 10% off the discounted price. What is the total discount?",
]


def build_corpus():
    corpus = []
    for group, prompts in (("zh", PROMPTS_ZH), ("en", PROMPTS_EN),
                           ("code", PROMPTS_CODE), ("math", PROMPTS_MATH)):
        for i, prompt in enumerate(prompts):
            corpus.append({"id": f"{group}-{i:02d}", "group": group, "prompt": prompt})
    return corpus


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def lcp_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def repetition_rate(token_ids, n=8):
    """Fraction of n-grams that are repeats: a cheap degeneration detector."""
    if len(token_ids) < n + 1:
        return 0.0
    grams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def run_backend(llm, backend, corpus, max_tokens, params_cls):
    llm.model_runner.call("set_attention_backend", backend)
    llm.scheduler.block_manager.clear_prefix_cache()
    prompts = [row["prompt"] for row in corpus]
    params = params_cls(temperature=0, max_tokens=max_tokens, ignore_eos=True)
    started = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    elapsed = time.perf_counter() - started
    return [list(o["token_ids"]) for o in outputs], elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--determinism-repeats", type=int, default=3)
    parser.add_argument("--determinism-subset", type=int, default=10)
    parser.add_argument("--output", default="/root/p6_attention_equivalence.json")
    args = parser.parse_args()

    from nanovllm import LLM, SamplingParams

    corpus = build_corpus()
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
    results = {"backends": {}, "comparison": {}}
    try:
        tokens = {}
        timing = {}
        for backend in ("sdpa", "triton_paged"):
            ids, elapsed = run_backend(llm, backend, corpus, args.max_tokens, SamplingParams)
            tokens[backend] = ids
            timing[backend] = elapsed
            print(f"[equiv] {backend}: {len(ids)} prompts, {elapsed:.1f}s", flush=True)

        # L4: per-backend determinism on a subset
        det = {}
        subset = corpus[:args.determinism_subset]
        for backend in ("sdpa", "triton_paged"):
            llm.model_runner.call("set_attention_backend", backend)
            runs = []
            for _ in range(args.determinism_repeats):
                llm.scheduler.block_manager.clear_prefix_cache()
                outs = llm.generate(
                    [r["prompt"] for r in subset],
                    SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True),
                    use_tqdm=False,
                )
                runs.append([list(o["token_ids"]) for o in outs])
            det[backend] = all(r == runs[0] for r in runs)
            print(f"[equiv] determinism {backend}: {det[backend]}", flush=True)

        tokenizer = llm.tokenizer
        rows = []
        for row, a, b in zip(corpus, tokens["sdpa"], tokens["triton_paged"]):
            text_a = tokenizer.decode(a)
            text_b = tokenizer.decode(b)
            dist = levenshtein(text_a, text_b)
            denom = max(len(text_a), len(text_b), 1)
            agree = sum(1 for x, y in zip(a, b) if x == y) / max(len(a), 1)
            first_same = bool(a and b and a[0] == b[0])
            rows.append({
                "id": row["id"], "group": row["group"], "prompt": row["prompt"],
                "lcp_tokens": lcp_len(a, b),
                "text_sdpa": text_a[:400],
                "text_paged": text_b[:400],
                "token_agreement": agree,
                "first_token_agreement": first_same,
                "edit_distance_chars": dist,
                "edit_distance_ratio": dist / denom,
                "text_len_sdpa": len(text_a),
                "text_len_paged": len(text_b),
                "repetition_sdpa": repetition_rate(a),
                "repetition_paged": repetition_rate(b),
                "identical": a == b,
            })
            print("[equiv] %-8s agreement=%.3f ed=%.3f first=%s" % (
                row["id"], agree, dist / denom, first_same), flush=True)

        ratios = [r["edit_distance_ratio"] for r in rows]
        results["rows"] = rows
        results["comparison"] = {
            "requests": len(rows),
            "request_token_identical": sum(1 for r in rows if r["identical"]),
            "request_token_agreement_mean": statistics.mean(r["token_agreement"] for r in rows),
            "first_token_agreement_rate": statistics.mean(1.0 if r["first_token_agreement"] else 0.0 for r in rows),
            "edit_distance_ratio_p50": statistics.median(ratios),
            "edit_distance_ratio_p95": sorted(ratios)[max(0, int(0.95 * len(ratios)) - 1)],
            "edit_distance_ratio_max": max(ratios),
            "lcp_tokens_p50": statistics.median([r["lcp_tokens"] for r in rows]),
            "lcp_tokens_mean": statistics.mean([r["lcp_tokens"] for r in rows]),
            "degenerate_sdpa": sum(1 for r in rows if r["repetition_sdpa"] > 0.5),
            "degenerate_paged": sum(1 for r in rows if r["repetition_paged"] > 0.5),
        }
        results["determinism"] = det
        results["timing_s"] = timing
        results["max_tokens"] = args.max_tokens
        Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print("[equiv] comparison=" + json.dumps(results["comparison"], ensure_ascii=False), flush=True)
        print(f"[equiv] wrote {args.output}", flush=True)
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
