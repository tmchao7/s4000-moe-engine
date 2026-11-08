"""P15-A numeric gate: tuned paged-attention config vs the historical (32,4,1).

The autotuned config table (nanovllm/kernels/paged_attention.py) changes the
online-softmax tile size, so outputs are not guaranteed bitwise-identical.
Pre-registered gate (docs/PLAN.md P15-A, values from docs/numerics-policy.md):

  L2-style: per-position top-1 agreement >= 99.5% across all compared
            positions; first token 100% match; zero degenerate outputs.
  L4:       tuned config self-deterministic across two runs.

Protocol: the parent script spawns itself three times as a subprocess (one
process per engine — same-process dual engines are a known failure mode):
  arm A: NANOVLLM_PAGED_CONFIG=32,4,1  (historical baseline config)
  arm B1/B2: default env               (tuned per-batch table)
then compares the emitted token id streams.

Run:  PYTHONPATH=. python3 tests/p15_paged_gate.py
rc=0 gate passed; rc=1 gate failed; rc=2 harness error.
"""
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "/root/autodl-tmp/models/Qwen3-0.6B"
MAX_TOKENS = 32

PROMPTS = [
    # zh (12)
    "请解释什么是 MoE（混合专家）架构，以及它为什么能降低推理成本。",
    "用三句话介绍 KV cache 在自回归解码中的作用。",
    "写一句关于秋天的诗。",
    "把这句话翻译成英文：大模型推理的性能瓶颈往往在访存而不是算力。",
    "为什么 prefill 是计算密集型而 decode 是访存密集型？",
    "简述 PagedAttention 的核心思想。",
    "列出三个常见的 LLM 推理优化手段并各用一句话说明。",
    "用中文解释什么是 goodput。",
    "(chunked prefill)分块预填充解决什么问题？",
    "一句话说明 TP（张量并行）和 PP（流水线并行）的区别。",
    "什么是投机解码？它为什么对小模型草稿依赖接受率？",
    "简单介绍一下吞吐量与延迟的权衡。",
    # en (12)
    "Explain the difference between throughput and latency in LLM serving.",
    "Why does decode attention become memory-bandwidth bound at long context?",
    "Summarize how continuous batching works in one paragraph.",
    "What is prefill-decode disaggregation and when does it help?",
    "Give three reasons why a scheduling layer matters for inference engines.",
    "Explain online softmax in flash attention briefly.",
    "Why do FP8 weights need dequantization on hardware without FP8 compute?",
    "Describe what a paged KV cache block table is.",
    "What does TTFT stand for and why does it matter for user experience?",
    "How does speculative decoding use a draft model?",
    "Explain what an SLO-tiered serving system does.",
    "What is the memory cost of a KV cache entry per token?",
    # code (12)
    "Write a Python function that reverses a string.",
    "Write a Python function to check if a number is prime.",
    "Write a bash one-liner to find the 10 largest files under a directory.",
    "Write a SQL query selecting the top 5 customers by total order value.",
    "Write a Python generator that yields Fibonacci numbers.",
    "Write a regex that matches an IPv4 address.",
    "Write a Python function computing the moving average of a list.",
    "Write a git command to squash the last three commits, and explain it.",
    "Write a Python dataclass for a chat message with role and content.",
    "Write a short JSON schema for a user object with name and email.",
    "Write a C function that clamps an integer to [lo, hi].",
    "Write a Python snippet reading a JSONL file line by line.",
    # math (12)
    "计算 17 × 23 的结果，并给出中间步骤。",
    "一个袋子有 3 红 5 蓝球，取两个都是蓝的概率是多少？",
    "解方程 2x + 7 = 19。",
    "求 1 到 100 所有偶数之和。",
    "ax² + bx + c 的求根公式是什么？",
    "0.1 + 0.2 在浮点数里为什么等于 0.30000000000000004？",
    "一个圆柱半径 3 高 5，体积是多少？",
    "计算 log2(1024)。",
    "甲乙两地相距 240 公里，车速 80 km/h，几小时到达？",
    "掷两个骰子点数之和为 7 的概率？",
    "化简 (x² − 9)/(x − 3)。",
    "斐波那契数列第 10 项是多少？",
]


def run_arm(tag: str, out_path: str) -> None:
    env = dict(os.environ)
    if tag == "A":
        env["NANOVLLM_PAGED_CONFIG"] = "32,4,1"
    else:
        env.pop("NANOVLLM_PAGED_CONFIG", None)
    code = (
        "import json, sys; sys.path.insert(0, %r);"
        "from nanovllm import LLM, SamplingParams;"
        "llm = LLM(%r, enforce_eager=True, tensor_parallel_size=2,"
        " attention_backend='triton_paged', max_model_len=1024,"
        " max_num_seqs=64, gpu_memory_utilization=0.5);"
        "prompts = json.load(open(%r));"
        "outs = llm.generate(prompts, SamplingParams(temperature=0,"
        " max_tokens=%d, ignore_eos=False), use_tqdm=False);"
        "json.dump([o['token_ids'] for o in outs], open(%r, 'w'));"
        "llm.exit()"
    ) % (REPO, MODEL, sys.argv[-1], MAX_TOKENS, out_path)
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--arm":  # child arm process
        run_arm(sys.argv[2], sys.argv[3])
        return 0
    import tempfile

    tmp = tempfile.mkdtemp(prefix="p15_gate_")
    prompts_file = os.path.join(tmp, "prompts.json")
    with open(prompts_file, "w") as f:
        json.dump(PROMPTS, f, ensure_ascii=False)

    arms = {}
    for tag in ("A", "B1", "B2"):
        out = os.path.join(tmp, f"{tag}.json")
        print(f"[gate] running arm {tag} ...", flush=True)
        rc = subprocess.call(
            [sys.executable, __file__, "--arm", tag, out, prompts_file]
        )
        if rc != 0:
            print(f"[gate] arm {tag} failed rc={rc}", flush=True)
            return 2
        arms[tag] = json.load(open(out))
        print(f"[gate] arm {tag}: {len(arms[tag])} outputs", flush=True)

    if not (len(arms["A"]) == len(arms["B1"]) == len(arms["B2"]) == len(PROMPTS)):
        print("[gate] output count mismatch", flush=True)
        return 2

    # L4: tuned config self-determinism
    det = arms["B1"] == arms["B2"]
    # L2-style agreement + first token
    positions = 0
    mismatches = 0
    first_token_ok = 0
    degenerate = 0
    for a, b in zip(arms["A"], arms["B1"]):
        if b and len(set(b[:8])) == 1 and len(b) == MAX_TOKENS:
            degenerate += 1
        n = min(len(a), len(b))
        if n and a[0] == b[0]:
            first_token_ok += 1
        for i in range(n):
            positions += 1
            mismatches += a[i] != b[i]
    agree = 1 - mismatches / positions if positions else 0.0
    first_rate = first_token_ok / len(PROMPTS)

    verdict = {
        "gate": "p15_paged_config",
        "self_deterministic": det,
        "first_token_rate": round(first_rate, 4),
        "position_agreement": round(agree, 5),
        "positions_compared": positions,
        "degenerate_outputs": degenerate,
        "gates": {
            "determinism": det,
            "first_token_100pct": first_rate == 1.0,
            "agreement_ge_99.5pct": agree >= 0.995,
            "no_degenerate": degenerate == 0,
        },
    }
    print(json.dumps(verdict, indent=1), flush=True)
    passed = all(verdict["gates"].values())
    os.makedirs("/root/p15_data", exist_ok=True)
    with open("/root/p15_data/paged_gate.json", "w") as f:
        json.dump({"verdict": verdict, "armA": arms["A"], "armB1": arms["B1"]}, f)
    print(f"[gate] {'PASS' if passed else 'FAIL'}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
