"""P23/P24 30B final window (single load — 加载窗口只测量, 不二次加载).

Production stack (TP2 / triton_paged / bmm_dense / stall_free / eager).
Checks, in one load:
  1  defaults greedy self-consistency (two runs, cache cleared between)
  2  decode step time CV (3 segments, overlap OFF = production default)
  3  overlap ON vs OFF greedy: token-bitwise equal at 30B (same engine/history)
  4  guided_json sanity: schema-constrained output parses + finish stop
  5  top_p sanity (P22 carry)

Run:  PYTHONPATH=. python3 scripts/p24_30b_final.py --out /root/p24_data/final_30b.json
"""
import argparse
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

BODY = ("大模型推理系统需要在吞吐与延迟之间做精细权衡。prefill 阶段一次性处理整段输入,"
        "计算密集,占用大量算力;decode 阶段逐 token 生成,访存密集,每一步都要读取全部 "
        "KV cache。PagedAttention 把连续的 KV 存储切成页,连续批处理让新请求随时加入。")
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
SCHEMA = {"type": "object",
          "properties": {"name": {"type": "string", "maxLength": 12},
                         "age": {"type": "integer"}},
          "required": ["name", "age"], "additionalProperties": False}


def decode_segment(llm, sp, n=16):
    for i in range(n):
        llm.add_request(f"背景{i}: {BODY}", sp)
    fw = []
    while not llm.is_finished():
        llm.step()
        m = llm.last_step_metrics
        if m["mode"] == "decode" and m["forward_s"] > 0:
            fw.append(m["forward_s"])
    return statistics.median(fw) if fw else 0.0


def greedy_run(llm, prompts, sp):
    return [list(o["token_ids"]) for o in llm.generate(prompts, [sp] * len(prompts),
                                                       use_tqdm=False)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--out", default="/root/p24_data/final_30b.json")
    args = parser.parse_args()
    from nanovllm import LLM, SamplingParams

    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=2,
              attention_backend="triton_paged", max_model_len=2048,
              max_num_seqs=32, gpu_memory_utilization=0.9,
              scheduler_policy="stall_free", decode_moe_backend="bmm_dense",
              scheduler_overlap=True)
    verdict = {"model": args.model}
    ok = True

    # 1 defaults greedy self-consistency (overlap OFF = production default)
    sp0 = SamplingParams(temperature=0, max_tokens=48, ignore_eos=True)
    llm.set_scheduler_overlap(False)
    run1 = greedy_run(llm, PROMPTS, sp0)
    llm.scheduler.block_manager.clear_prefix_cache()
    run2 = greedy_run(llm, PROMPTS, sp0)
    verdict["greedy_self_consistent"] = run1 == run2
    ok &= verdict["greedy_self_consistent"]
    print(f"[30b-final] greedy self-consistent: {verdict['greedy_self_consistent']}",
          flush=True)

    # 2 decode step CV (overlap OFF)
    sp_dec = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True)
    decode_segment(llm, sp_dec)                       # warmup
    meds = [decode_segment(llm, sp_dec) for _ in range(3)]
    cv = statistics.pstdev(meds) / statistics.mean(meds)
    verdict["decode_step_median_ms"] = round(statistics.median(meds) * 1e3, 2)
    verdict["decode_step_cv"] = round(cv, 4)
    ok &= cv <= 0.05
    print(f"[30b-final] decode step {verdict['decode_step_median_ms']}ms "
          f"cv={cv:.4f}", flush=True)

    # 3 overlap ON vs OFF at 30B — token-bitwise equal
    llm.scheduler.block_manager.clear_prefix_cache()
    llm.set_scheduler_overlap(True)
    run_over = greedy_run(llm, PROMPTS, sp0)
    llm.set_scheduler_overlap(False)
    verdict["overlap_30b_bitwise_equal"] = run1 == run_over
    ok &= verdict["overlap_30b_bitwise_equal"]
    print(f"[30b-final] overlap@30B bitwise equal: "
          f"{verdict['overlap_30b_bitwise_equal']}", flush=True)

    # 4 guided_json sanity
    sp_g = SamplingParams(temperature=0.8, max_tokens=128,
                          guided_json=SCHEMA)
    outs = llm.generate(["给张三一张名片：", "生成一条记录："], [sp_g] * 2,
                        use_tqdm=False)
    g_ok = True
    for o in outs:
        text = llm.tokenizer.decode(o["token_ids"], skip_special_tokens=True)
        try:
            obj = json.loads(text.strip())
            g_ok &= isinstance(obj, dict) and "name" in obj and "age" in obj
        except Exception:
            g_ok = False
    g_ok &= all(o["metrics"]["finish_reason"] == "stop" for o in outs)
    verdict["guided_json_sanity"] = g_ok
    ok &= g_ok
    print(f"[30b-final] guided_json sanity: {g_ok}", flush=True)

    # 5 top_p sanity (P22 carry)
    sp_tp = SamplingParams(temperature=0.9, max_tokens=32, ignore_eos=True,
                           top_p=0.9)
    outs = llm.generate(PROMPTS[:2], [sp_tp] * 2, use_tqdm=False)
    good = all(len(o["token_ids"]) == 32 for o in outs)
    verdict["top_p_sanity"] = good
    ok &= good
    print(f"[30b-final] top_p sanity: {good}", flush=True)

    verdict["pass"] = ok
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=1)
    print(json.dumps(verdict, indent=1), flush=True)
    print(f"[30b-final] {'PASS' if ok else 'FAIL'}", flush=True)
    llm.exit()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
