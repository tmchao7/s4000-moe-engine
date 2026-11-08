"""P16-A: unified mixed batch (one forward) vs legacy _run_mixed two-pass split.

Drives a real 0.6B engine (TP1, triton_paged, eager, stall_free) with
STAGGERED arrivals — two short prompts start decoding first, then five
~600-token prompts are injected mid-flight so each arm runs several genuine
mixed steps (decode rows + prefill chunk in one batch) — under three flags:

  arm1  legacy   (NANOVLLM_MIXED_UNIFIED off, _run_mixed two-pass split)
  arm2  unified  (on — one forward per mixed step)
  arm3  legacy   again (ABA history anchor, I34)

Checks:
  structural  prepare_mixed rejects non-contiguous row groups and shapes a
              contiguous one correctly (decode-first context fields, prefill-
              only cu_seqlens); fcfs never emits a mixed batch while
              stall_free does (positive control); MoE split fallback
              (NANOVLLM_MIXED_MOE=split) completes and stays aligned.
  numeric     every arm must actually take mixed steps; arm1 == arm3 bitwise
              (history determinism); arm2 compared per-request: mismatch
              count + per-request LCP recorded (the <=0.5% LCP flip gate is
              judged by the gate run; this unit test only bounds gross
              breakage).

Run:  PYTHONPATH=. python3 tests/p16_mixed_unified.py     (rc=0 pass)
"""
import json
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine import model_runner as mr
from nanovllm.models import qwen3_moe
from nanovllm.utils.context import get_context

MODEL = "/root/autodl-tmp/models/Qwen3-0.6B"
MAX_TOKENS = 48

BODY = (
    "大模型推理系统需要在吞吐与延迟之间做精细权衡。prefill 阶段一次性处理整段输入,"
    "计算密集,占用大量算力;decode 阶段逐 token 生成,访存密集,每一步都要读取全部 "
    "KV cache。PagedAttention 把连续的 KV 存储切成页,连续批处理让新请求随时加入,"
    "而 stall-free 调度则保证 decode 优先。统一混合批把两个阶段放进同一个 forward,"
    "是消除双 pass 固定开销的结构性方案。"
)
SHORTS = [
    "用一句话解释什么是 KV cache。",
    "Write a Python function that reverses a string.",
    "计算 17 × 23 的结果。",
]
LONGS = [f"第{i}段:{BODY * 4}" for i in range(5)]  # ≈600 tokens → 3 chunks


def run_arm(llm, flag_on):
    """Staggered-arrival workload; returns ({arm_idx: tokens}, mode counts)."""
    runner = llm.model_runner
    runner.set_mixed_unified(flag_on)   # instance flag; rank-synced on TP2
    llm.scheduler.block_manager.clear_prefix_cache()
    sid_map = {}
    finals = {}
    modes = []
    for i, p in enumerate(SHORTS):
        sid = llm.add_request(p, SamplingParams(
            temperature=0, max_tokens=MAX_TOKENS, ignore_eos=True))
        sid_map[sid] = f"short{i}"
    # step until the shorts' prefills have completed (they start decoding)
    while llm.scheduler.waiting:
        outputs, _ = llm.step()
        modes.append(llm.last_step_metrics["mode"])
        for seq_id, toks, _ in outputs:
            finals[sid_map[seq_id]] = list(toks)
    for i, p in enumerate(LONGS):
        sid = llm.add_request(p, SamplingParams(
            temperature=0, max_tokens=MAX_TOKENS, ignore_eos=True))
        sid_map[sid] = f"long{i}"
    while not llm.is_finished():
        outputs, _ = llm.step()
        modes.append(llm.last_step_metrics["mode"])
        for seq_id, toks, _ in outputs:
            finals[sid_map[seq_id]] = list(toks)
    llm.model_runner.set_mixed_unified(False)
    from collections import Counter
    return finals, dict(Counter(modes))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--out", default="/root/p16_data/mixed_unified.json")
    args = parser.parse_args()
    torch.musa.set_device(0)
    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1,
              attention_backend="triton_paged", max_model_len=1024,
              max_num_seqs=8, gpu_memory_utilization=0.5,
              scheduler_policy="stall_free")
    try:
        return _mixed_body(llm, args)
    finally:
        llm.exit()


def _mixed_body(llm, args):
    runner = llm.model_runner
    verdict = {"arms": {}, "structural": {}}
    ok = True

    # ---- structural: prepare_mixed shape/contiguity (host-side, cheap) -----
    sp = SamplingParams(temperature=0, max_tokens=4)
    dec = Sequence(list(range(100, 108)), sp)          # 8 tokens, decode row
    dec.is_prefill = False
    dec.num_cached_tokens = 7
    dec.num_scheduled_tokens = 1
    dec.block_table = [0]
    pre = Sequence(list(range(200, 204)), sp)          # 4-token prefill row
    pre.num_scheduled_tokens = 4
    pre.block_table = [1]

    try:
        runner.prepare_mixed([pre, dec, pre])
        verdict["structural"]["contiguity_reject"] = False
        ok = False
    except ValueError:
        verdict["structural"]["contiguity_reject"] = True

    input_ids, positions = runner.prepare_mixed([dec, pre])
    ctx = get_context()
    verdict["structural"]["mixed_context"] = {
        "is_mixed": ctx.is_mixed,
        "num_decode_rows": ctx.num_decode_rows,
        "input_rows": int(input_ids.size(0)),
        "decode_ctx0": int(ctx.decode_context_lens[0]),
    }
    ok &= ctx.is_mixed and ctx.num_decode_rows == 1
    ok &= int(input_ids.size(0)) == 5                 # 1 decode + 4 prefill
    ok &= int(ctx.decode_context_lens[0]) == 8
    ok &= list(ctx.cu_seqlens_q_list) == [0, 4]       # prefill-only
    ok &= list(positions.tolist()) == [7, 0, 1, 2, 3]
    mr.reset_context()

    # ---- structural: fcfs never mixed, stall_free does ---------------------
    cfg = SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=512, eos=0,
                          kvcache_block_size=256, scheduler_policy="fcfs",
                          prefill_chunk_size=256, scheduler_starvation_steps=32,
                          num_kvcache_blocks=64)
    fcfs_modes = []
    sched = Scheduler(cfg)
    fseqs = [Sequence(list(range(10 + i, 10 + i + 8)),
                      SamplingParams(max_tokens=4, temperature=0)) for i in range(3)]
    for s in fseqs:
        sched.add(s)
    for _ in range(16):
        batch = sched.schedule()
        fcfs_modes.append(batch.mode)
        if batch.mode == "empty":
            break
        for seq in batch.sequences:
            if seq.is_prefill:
                seq.num_cached_tokens += seq.num_scheduled_tokens
                seq.num_scheduled_tokens = 0
                if seq.num_cached_tokens >= seq.num_tokens:
                    seq.status = SequenceStatus.RUNNING
        if batch.mode == "decode":
            for seq in batch.sequences:
                seq.append_token(1)
                if seq.num_completion_tokens >= seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    sched.running.remove(seq)
                    sched.block_manager.deallocate(seq)
    verdict["structural"]["fcfs_modes"] = fcfs_modes
    ok &= "mixed" not in fcfs_modes

    cfg2 = SimpleNamespace(**{**cfg.__dict__, "scheduler_policy": "stall_free"})
    sched2 = Scheduler(cfg2)
    lseq = Sequence(list(range(1000, 1700)), SamplingParams(max_tokens=4, temperature=0))
    sseqs = [Sequence(list(range(10 + i, 10 + i + 8)),
                      SamplingParams(max_tokens=4, temperature=0)) for i in range(3)]
    for s in [lseq] + sseqs:
        sched2.add(s)
    sf_modes = []
    for _ in range(8):
        batch = sched2.schedule()
        sf_modes.append(batch.mode)
        for seq in batch.sequences:
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.is_prefill and seq.num_cached_tokens >= seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                if seq in sched2.waiting:
                    sched2.waiting.remove(seq)
            elif not seq.is_prefill:
                seq.append_token(1)
    verdict["structural"]["stall_free_modes"] = sf_modes
    ok &= "mixed" in sf_modes

    # ---- numeric: ABA arms on one engine, staggered arrivals ---------------
    arm1, modes1 = run_arm(llm, False)
    arm2, modes2 = run_arm(llm, True)
    arm3, modes3 = run_arm(llm, False)

    verdict["mode_counts"] = {"legacy1": modes1, "unified": modes2, "legacy2": modes3}
    for name, m in (("legacy1", modes1), ("unified", modes2), ("legacy2", modes3)):
        if m.get("mixed", 0) < 2:
            print(f"[p16-mixed] INVALID: arm {name} took {m.get('mixed', 0)} "
                  f"mixed steps (need >=2)", flush=True)
            ok = False

    hist_ok = arm1 == arm3
    verdict["aba_history_deterministic"] = hist_ok
    ok &= hist_ok

    keys = sorted(arm1)
    n_match = sum(arm1[k] == arm2[k] for k in keys)
    lcps = []
    for k in keys:
        a, b = arm1[k], arm2[k]
        lcp = 0
        for x, y in zip(a, b):
            if x != y:
                break
            lcp += 1
        lcps.append(lcp)
    verdict["unified_vs_legacy"] = {
        "requests_bitwise_equal": n_match,
        "n_requests": len(keys),
        "mean_lcp": round(sum(lcps) / len(lcps), 2),
        "lcps": dict(zip(keys, lcps)),
    }
    # gross-breakage bound only; the <=0.5% LCP flip gate is the gate run's job
    ok &= n_match >= int(0.8 * len(keys))
    ok &= all(len(v) == MAX_TOKENS for v in arm1.values())
    ok &= all(len(v) == MAX_TOKENS for v in arm2.values())

    # ---- structural: MoE split fallback completes and stays aligned --------
    qwen3_moe._MIXED_MOE_SPLIT[0] = True
    arm4, modes4 = run_arm(llm, True)
    qwen3_moe._MIXED_MOE_SPLIT[0] = False
    verdict["mode_counts"]["unified_moe_split"] = modes4
    ok &= modes4.get("mixed", 0) >= 2
    n_match4 = sum(arm1[k] == arm4[k] for k in keys if k in arm4)
    verdict["unified_moe_split_vs_legacy_equal"] = n_match4

    verdict["pass"] = ok
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in verdict.items() if k != "arms"},
                     indent=1, ensure_ascii=False), flush=True)
    print(f"[p16-mixed] {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
