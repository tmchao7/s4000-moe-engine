"""P22-A sampling-face: pre/post-change greedy baseline (bitwise anchor).

Greedy decode is deterministic for a fresh engine with the same prompt
history (I34/I31 discipline: discard nothing here — all prompts are fed
before any step, single burst, greedy).  Capture BEFORE the P22-A code
change, compare AFTER: token ids must be bitwise identical, and the mean
decode forward_s must stay within the CV gate.

Modes:
  capture  run and write /root/p22_data/baseline_greedy.json
  compare  run and diff against the capture; rc=0 iff bitwise equal

Run:  PYTHONPATH=. python3 tests/p22_baseline_greedy.py capture|compare
"""
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

PROMPTS = [
    "用一段话解释大模型推理中的 KV cache 与 paged attention。",
    "Write a Python function that computes moving averages.",
    "请比较 TCP 与 UDP 的区别，并给出各自适用场景。",
    "Summarize how continuous batching improves GPU utilization.",
    "计算 17 × 23 并给出中间步骤，然后用一句话解释竖式乘法。",
    "Explain why decode is memory-bandwidth bound in one paragraph.",
    "写一首关于秋天的四行诗。",
    "List three trade-offs between throughput and latency in serving.",
]
TIMING_PROMPTS = [f"背景{i}:" + "大模型推理系统需要在吞吐与延迟之间做精细权衡，prefill 计算密集而 decode 访存密集。" * 3
                  for i in range(16)]
OUT = "/root/p22_data/baseline_greedy.json"


def run_engine():
    from nanovllm import LLM, SamplingParams
    llm = LLM("/root/autodl-tmp/models/Qwen3-0.6B", enforce_eager=True,
              tensor_parallel_size=1, attention_backend="triton_paged",
              max_model_len=2048, max_num_seqs=32, gpu_memory_utilization=0.35,
              scheduler_policy="stall_free")
    sp = SamplingParams(temperature=0, max_tokens=48, ignore_eos=True)
    outs = llm.generate(PROMPTS, [sp] * len(PROMPTS), use_tqdm=False)
    tokens = [list(o["token_ids"]) for o in outs]

    sp2 = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True)
    for p in TIMING_PROMPTS:
        llm.add_request(p, sp2)
    dec_fw = []
    while not llm.is_finished():
        _, num = llm.step()
        m = llm.last_step_metrics
        if m["mode"] == "decode" and m["forward_s"] > 0:
            dec_fw.append(m["forward_s"])
    llm.exit()
    return tokens, dec_fw


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "capture"
    tokens, dec_fw = run_engine()
    timing = {"decode_fw_median_ms": round(statistics.median(dec_fw) * 1e3, 3),
              "decode_fw_mean_ms": round(statistics.mean(dec_fw) * 1e3, 3),
              "n_decode_steps": len(dec_fw)}
    print("[p22-baseline] timing:", timing, flush=True)
    if mode == "capture":
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as f:
            json.dump({"tokens": tokens, "timing": timing}, f, ensure_ascii=False)
        print(f"[p22-baseline] captured {len(tokens)} outputs -> {OUT}", flush=True)
        return 0
    with open(OUT) as f:
        base = json.load(f)
    same = base["tokens"] == tokens
    print(f"[p22-baseline] bitwise_equal={same}", flush=True)
    if not same:
        for i, (a, b) in enumerate(zip(base["tokens"], tokens)):
            if a != b:
                print(f"  diff at prompt {i}: first mismatch idx "
                      f"{next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), 'len')}",
                      flush=True)
                break
    ratio = timing["decode_fw_median_ms"] / max(base["timing"]["decode_fw_median_ms"], 1e-9)
    print(f"[p22-baseline] decode_fw ratio new/base = {ratio:.4f}", flush=True)
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
