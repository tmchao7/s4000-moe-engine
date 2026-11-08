"""P22-A 30B final window (single load — 加载窗口只测量, 不二次加载).

Production stack (TP2 / triton_paged / bmm_dense decode / stall_free / eager /
chunk-256 defaults).  Checks:
  1  defaults greedy self-consistency: two identical runs in one load produce
     bitwise-equal token ids (I34 discipline: same engine, same history;
     prefix cache cleared between)
  2  decode step time: 3 segments of 16 seqs x 64 tokens, per-segment median
     + cross-segment CV (gate <= 5%); absolute value recorded for the report
  3  top_p=0.9 sanity: finishes, 32 tokens
  4  engine stop_token_ids sanity: deterministic 2-token stop
  5  logprobs sanity: greedy top1 id == emitted token id

Run (detached per SOP):
  PYTHONPATH=. python3 scripts/p22_30b_final.py --out /root/p22_data/final_30b.json
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


def decode_segment(llm, sp, n=16):
    for i in range(n):
        llm.add_request(f"背景{i}: {BODY}", sp)
    fw = []
    finals = {}
    while not llm.is_finished():
        outputs, _ = llm.step()
        m = llm.last_step_metrics
        if m["mode"] == "decode":
            fw.append(m["forward_s"])
        for seq_id, toks, metrics in outputs:
            finals[seq_id] = list(toks)
    return fw, finals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--out", default="/root/p22_data/final_30b.json")
    args = parser.parse_args()
    from nanovllm import LLM, SamplingParams

    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=2,
              attention_backend="triton_paged", max_model_len=2048,
              max_num_seqs=32, gpu_memory_utilization=0.9,
              scheduler_policy="stall_free", decode_moe_backend="bmm_dense")
    verdict = {"model": args.model}
    ok = True

    # 1 defaults greedy self-consistency (bitwise, same load)
    sp0 = SamplingParams(temperature=0, max_tokens=48, ignore_eos=True)
    run1 = [list(o["token_ids"]) for o in llm.generate(PROMPTS, [sp0] * 8, use_tqdm=False)]
    llm.scheduler.block_manager.clear_prefix_cache()
    run2 = [list(o["token_ids"]) for o in llm.generate(PROMPTS, [sp0] * 8, use_tqdm=False)]
    verdict["greedy_self_consistent"] = run1 == run2
    ok &= verdict["greedy_self_consistent"]
    print(f"[30b-final] greedy self-consistent: {verdict['greedy_self_consistent']}",
          flush=True)

    # 2 decode step time CV (3 segments)
    sp_dec = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True)
    decode_segment(llm, sp_dec)                       # warmup
    meds = []
    for _ in range(3):
        fw, _ = decode_segment(llm, sp_dec)
        meds.append(statistics.median(fw))
    cv = statistics.pstdev(meds) / statistics.mean(meds)
    verdict["decode_step_median_ms"] = round(statistics.median(meds) * 1e3, 2)
    verdict["decode_step_segment_medians_ms"] = [round(m * 1e3, 2) for m in meds]
    verdict["decode_step_cv"] = round(cv, 4)
    ok &= cv <= 0.05
    print(f"[30b-final] decode step median {verdict['decode_step_median_ms']}ms "
          f"cv={cv:.4f}", flush=True)

    # 3 top_p sanity
    sp_tp = SamplingParams(temperature=0.9, max_tokens=32, ignore_eos=True, top_p=0.9)
    outs = llm.generate(PROMPTS[:2], [sp_tp] * 2, use_tqdm=False)
    good = all(len(o["token_ids"]) == 32 for o in outs)
    verdict["top_p_sanity"] = good
    ok &= good
    print(f"[30b-final] top_p sanity: {good}", flush=True)

    # 4 engine stop_token_ids (deterministic 2-token stop)
    ref = llm.generate(PROMPTS[:1], SamplingParams(temperature=0, max_tokens=16,
                                                   ignore_eos=True),
                       use_tqdm=False)[0]["token_ids"]
    out = llm.generate(PROMPTS[:1],
                       SamplingParams(temperature=0, max_tokens=16,
                                      stop_token_ids=[ref[1]]),
                       use_tqdm=False)[0]
    good = (out["metrics"]["finish_reason"] == "stop"
            and len(out["token_ids"]) == 2)
    verdict["stop_token_ids_sanity"] = good
    ok &= good
    print(f"[30b-final] stop_token_ids sanity: {good}", flush=True)

    # 5 logprobs sanity (greedy top1 == emitted)
    sp_lp = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True, logprobs=3)
    sid = llm.add_request(PROMPTS[0], sp_lp)
    events = []
    while not llm.is_finished():
        llm.step()
        events += llm.last_token_events
    good, n_lp = True, 0
    for seq_id, token_id, finished, metrics, lp in events:
        if seq_id != sid or lp is None:
            continue
        n_lp += 1
        good &= len(lp["top"]) == 3 and lp["top"][0][0] == token_id
    verdict["logprobs_sanity"] = good and n_lp == 8
    ok &= verdict["logprobs_sanity"]
    print(f"[30b-final] logprobs sanity: {verdict['logprobs_sanity']} (n={n_lp})",
          flush=True)

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
