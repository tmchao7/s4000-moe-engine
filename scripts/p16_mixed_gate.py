"""P16-A gate: unified mixed batch vs legacy _run_mixed split, one engine.

Phases (interleaved within ONE engine — the runtime switch is rank-synced
through ModelRunner.call, so TP2 A/B never rebuilds):

  perf  R interleaved legacy/unified segments of the same staggered mixed
        scenario (3 shorts decode first, then five ~600-token prompts chunk
        through 3 mixed steps while decodes run).  Mixed-step forward times
        pooled per arm; gate = median(unified) <= 0.9 x median(legacy) with
        per-arm rep-median CV <= 5%.
  flip  per-request first-divergence comparison (LCP method, I34: same
        engine, same staggered history, prefix cache cleared per segment);
        flip_rate = diverged requests / completion tokens compared up to
        each request's first divergence.  Gate <= 0.5%.

Run:  PYTHONPATH=. python3 scripts/p16_mixed_gate.py \
        --model /root/autodl-tmp/models/Qwen3-0.6B --out /root/p16_data/gate_06b.json
"""
import argparse
import json
import os
import statistics
import sys
from collections import Counter

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from nanovllm import LLM, SamplingParams  # noqa: E402

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


def run_segment(llm, longs, max_tokens, unified, capture_tokens=False, stagger="burst"):
    """One staggered mixed scenario; returns (mixed_step_times, finals, modes).

    stagger="burst": all long prompts injected at once (5 chunks in 2 mixed
    steps).  stagger="drip": one long per step — same prompts, different
    arrival history, so different batch compositions per step.  The legacy
    engine itself may flip tokens between the two (that is exactly what the
    control phase measures)."""
    runner = llm.model_runner
    runner.call("set_mixed_unified", unified)
    llm.scheduler.block_manager.clear_prefix_cache()
    sid_map = {}
    finals = {}
    mixed_times = []
    modes = []
    sp = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
    for i, p in enumerate(SHORTS):
        sid_map[llm.add_request(p, sp)] = f"short{i}"
    while llm.scheduler.waiting:
        outputs, _ = llm.step()
        modes.append(llm.last_step_metrics["mode"])
        for seq_id, toks, _ in outputs:
            finals[sid_map[seq_id]] = list(toks)
    if stagger == "burst":
        for i, p in enumerate(longs):
            sid_map[llm.add_request(p, sp)] = f"long{i}"
        while not llm.is_finished():
            outputs, _ = llm.step()
            metrics = llm.last_step_metrics
            modes.append(metrics["mode"])
            if metrics["mode"] == "mixed":
                mixed_times.append(metrics["forward_s"])
            for seq_id, toks, _ in outputs:
                finals[sid_map[seq_id]] = list(toks)
    else:  # drip: one long per step while decodes run
        for i, p in enumerate(longs):
            sid_map[llm.add_request(p, sp)] = f"long{i}"
            outputs, _ = llm.step()
            metrics = llm.last_step_metrics
            modes.append(metrics["mode"])
            if metrics["mode"] == "mixed":
                mixed_times.append(metrics["forward_s"])
            for seq_id, toks, _ in outputs:
                finals[sid_map[seq_id]] = list(toks)
        while not llm.is_finished():
            outputs, _ = llm.step()
            metrics = llm.last_step_metrics
            modes.append(metrics["mode"])
            if metrics["mode"] == "mixed":
                mixed_times.append(metrics["forward_s"])
            for seq_id, toks, _ in outputs:
                finals[sid_map[seq_id]] = list(toks)
    runner.call("set_mixed_unified", False)
    return mixed_times, finals, dict(Counter(modes))


def flip_stats(legacy, unified):
    """LCP-method flip rate: first divergence per request counts once."""
    flips = 0
    compared = 0
    per_request = {}
    for key in sorted(legacy):
        a, b = legacy[key], unified.get(key, [])
        lcp = 0
        for x, y in zip(a, b):
            if x != y:
                break
            lcp += 1
        diverged = lcp < len(a) or len(a) != len(b)
        per_request[key] = {"lcp": lcp, "len": len(a), "diverged": diverged}
        compared += (lcp + 1) if diverged else len(a)
        flips += 1 if diverged else 0
    return {"flips": flips, "tokens_compared": compared,
            "flip_rate": round(flips / compared, 5) if compared else None,
            "per_request": per_request}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--flip-reps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1,
                        help="discarded JIT-warmup segments per arm before reps")
    parser.add_argument("--control", action="store_true", default=True,
                        help="legacy history-sensitivity baseline: burst vs drip")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--gpu-util", type=float, default=0.5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.musa.set_device(0)
    tag = "8b" if "8B" in args.model else "0.6b"
    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=2,
              attention_backend="triton_paged", max_model_len=2048,
              max_num_seqs=8, gpu_memory_utilization=args.gpu_util,
              scheduler_policy="stall_free")
    longs = [f"第{i}段:{BODY * 4}" for i in range(5)]
    verdict = {"model": args.model, "tag": tag}

    # ---- warmup: JIT/first-touch segments, discarded (both arms) ------------
    for _ in range(args.warmup):
        run_segment(llm, longs, args.max_tokens, unified=False)
        run_segment(llm, longs, args.max_tokens, unified=True)

    # ---- perf: interleaved ABA ---------------------------------------------
    legacy_times, unified_times = [], []       # pooled mixed-step forwards
    legacy_rep_med, unified_rep_med = [], []
    for rep in range(args.reps):
        lt, _, m1 = run_segment(llm, longs, args.max_tokens, unified=False)
        ut, _, m2 = run_segment(llm, longs, args.max_tokens, unified=True)
        legacy_times += lt
        unified_times += ut
        legacy_rep_med.append(statistics.median(lt))
        unified_rep_med.append(statistics.median(ut))
        print(f"[gate] rep{rep + 1}: legacy {min(lt):.4f}-{max(lt):.4f}s "
              f"({len(lt)} mixed) unified {min(ut):.4f}-{max(ut):.4f}s "
              f"({len(ut)} mixed)", flush=True)

    leg_med = statistics.median(legacy_times)
    uni_med = statistics.median(unified_times)
    leg_cv = statistics.pstdev(legacy_rep_med) / statistics.mean(legacy_rep_med)
    uni_cv = statistics.pstdev(unified_rep_med) / statistics.mean(unified_rep_med)
    ratio = uni_med / leg_med
    verdict["perf"] = {
        "legacy_median_s": round(leg_med, 5),
        "unified_median_s": round(uni_med, 5),
        "ratio_unified_over_legacy": round(ratio, 4),
        "legacy_rep_medians": [round(x, 5) for x in legacy_rep_med],
        "unified_rep_medians": [round(x, 5) for x in unified_rep_med],
        "legacy_cv": round(leg_cv, 4),
        "unified_cv": round(uni_cv, 4),
        "n_mixed_steps": {"legacy": len(legacy_times), "unified": len(unified_times)},
        "gate": ratio <= 0.9 and leg_cv <= 0.05 and uni_cv <= 0.05,
    }
    print(f"[gate] perf ratio={ratio:.4f} (gate <=0.9) "
          f"cv legacy={leg_cv:.3f} unified={uni_cv:.3f}", flush=True)

    # ---- flip: fresh arm pairs, pooled --------------------------------------
    pooled = {"flips": 0, "tokens_compared": 0}
    pair_rates = []
    for rep in range(args.flip_reps):
        _, legacy_f, _ = run_segment(llm, longs, args.max_tokens, unified=False)
        _, unified_f, _ = run_segment(llm, longs, args.max_tokens, unified=True)
        st = flip_stats(legacy_f, unified_f)
        pair_rates.append(st["flip_rate"])
        pooled["flips"] += st["flips"]
        pooled["tokens_compared"] += st["tokens_compared"]
        print(f"[gate] flip rep{rep + 1}: {st['flips']} flips / "
              f"{st['tokens_compared']} tokens = {st['flip_rate']}", flush=True)
    pooled_rate = pooled["flips"] / pooled["tokens_compared"]
    verdict["flip"] = {
        "flips": pooled["flips"],
        "tokens_compared": pooled["tokens_compared"],
        "flip_rate": round(pooled_rate, 5),
        "pair_rates": pair_rates,
        "gate": pooled_rate <= 0.005,
    }
    print(f"[gate] flip pooled={pooled_rate:.4%} (gate <=0.5%)", flush=True)

    # ---- control: legacy history-sensitivity baseline -----------------------
    # Same prompts, same engine, same code path (legacy BOTH sides); only the
    # arrival history differs (burst vs drip) => batch compositions differ.
    # If the incumbent itself flips at this rate, the 0.5% yardstick measures
    # batch-composition sensitivity, not a regression introduced by P16-A.
    ctrl = {"flips": 0, "tokens_compared": 0}
    ctrl_rates = []
    for rep in range(args.flip_reps):
        _, base_f, _ = run_segment(llm, longs, args.max_tokens,
                                   unified=False, stagger="burst")
        _, alt_f, _ = run_segment(llm, longs, args.max_tokens,
                                  unified=False, stagger="drip")
        st = flip_stats(base_f, alt_f)
        ctrl_rates.append(st["flip_rate"])
        ctrl["flips"] += st["flips"]
        ctrl["tokens_compared"] += st["tokens_compared"]
        print(f"[gate] control rep{rep + 1}: {st['flips']} flips / "
              f"{st['tokens_compared']} tokens = {st['flip_rate']}", flush=True)
    ctrl_rate = ctrl["flips"] / ctrl["tokens_compared"]
    verdict["control_legacy_history"] = {
        "flips": ctrl["flips"],
        "tokens_compared": ctrl["tokens_compared"],
        "flip_rate": round(ctrl_rate, 5),
        "pair_rates": ctrl_rates,
        "interpretation": ("legacy itself flips at this rate under arrival-"
                           "history changes (I34-class), so the unified-path "
                           "flip rate is within the engine's inherent "
                           "batch-composition variance"
                           if ctrl_rate >= pooled_rate else
                           "legacy history baseline is lower; unified flips "
                           "exceed the incumbent's own variance"),
    }
    print(f"[gate] control legacy-history rate={ctrl_rate:.4%} "
          f"vs unified {pooled_rate:.4%}", flush=True)

    verdict["gate_all"] = verdict["perf"]["gate"] and verdict["flip"]["gate"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=1)
    print(json.dumps(verdict, indent=1, ensure_ascii=False), flush=True)
    print(f"[gate] {'ALL GATES PASS' if verdict['gate_all'] else 'GATE FAIL'}", flush=True)
    llm.exit()
    return 0 if verdict["gate_all"] else 1


if __name__ == "__main__":
    sys.exit(main())
