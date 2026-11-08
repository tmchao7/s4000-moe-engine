"""P20: 30B single-load evidence for flipping the P16-A unified-mixed default.

One 30B load window; interleaved ABA segments (runtime rank-synced switch via
runner.call("set_mixed_unified", ...)): legacy two-pass split vs unified
one-pass.  Same staggered mixed scenario as the 0.6B/8B gates (3 bursts of 8
~256-token prompts, decode batches 8→24).

Informational gate (PLAN P16-A ① was frozen at 0.6B and FAILED there — this
run cannot flip the default; it completes the 8B/30B evidence for a future
gate-revision proposal): unified wall <= legacy wall with per-arm CV <= 5%,
plus per-request token comparison (flips recorded).

Run:  PYTHONPATH=. python3 scripts/p20_30b_flip.py --out /root/p20_data/flip_30b.json
"""
import argparse
import json
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from nanovllm import LLM, SamplingParams  # noqa: E402

BODY = (
    "大模型推理系统需要在吞吐与延迟之间做精细权衡。prefill 阶段一次性处理整段输入,"
    "计算密集,占用大量算力;decode 阶段逐 token 生成,访存密集,每一步都要读取全部 "
    "KV cache。PagedAttention 把连续的 KV 存储切成页,连续批处理让新请求随时加入,"
    "而 stall-free 调度则保证 decode 优先。统一混合批把两个阶段放进同一个 forward。"
)


def run_segment(llm, unified, max_tokens):
    runner = llm.model_runner
    runner.call("set_mixed_unified", unified)
    llm.scheduler.block_manager.clear_prefix_cache()
    sp = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
    prompts = [f"第{i}段:{BODY * 4}" for i in range(24)]
    finals = {}
    mixed_times = []
    sid_map = {}
    wall0 = time.perf_counter()
    for burst in range(3):
        for p in prompts[burst * 8:(burst + 1) * 8]:
            sid_map[llm.add_request(p, sp)] = len(sid_map)
        while llm.scheduler.waiting:
            outputs, _ = llm.step()
            m = llm.last_step_metrics
            if m["mode"] == "mixed":
                mixed_times.append(m["forward_s"])
            for seq_id, toks, _ in outputs:
                finals[sid_map[seq_id]] = list(toks)
    while not llm.is_finished():
        outputs, _ = llm.step()
        m = llm.last_step_metrics
        if m["mode"] == "mixed":
            mixed_times.append(m["forward_s"])
        for seq_id, toks, _ in outputs:
            finals[sid_map[seq_id]] = list(toks)
    wall = time.perf_counter() - wall0
    runner.call("set_mixed_unified", False)
    return wall, mixed_times, finals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--gpu-util", type=float, default=0.9)
    parser.add_argument("--out", default="/root/p20_data/flip_30b.json")
    args = parser.parse_args()

    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=2,
              attention_backend="triton_paged", max_model_len=2048,
              max_num_seqs=32, gpu_memory_utilization=args.gpu_util,
              scheduler_policy="stall_free", decode_moe_backend="bmm_dense")
    verdict = {"model": args.model}

    run_segment(llm, False, args.max_tokens)   # warmup (JIT/first-touch)
    run_segment(llm, True, args.max_tokens)

    leg_walls, uni_walls, leg_dec, uni_dec = [], [], [], []
    leg_finals, uni_finals = None, None
    for rep in range(args.reps):
        lw, ld, lf = run_segment(llm, False, args.max_tokens)
        uw, ud, uf = run_segment(llm, True, args.max_tokens)
        leg_walls.append(lw); uni_walls.append(uw)
        leg_dec += ld; uni_dec += ud
        if leg_finals is None:
            leg_finals = lf
        uni_finals = uf
        print(f"[flip30b] rep{rep + 1}: legacy {lw:.2f}s unified {uw:.2f}s "
              f"ratio={uw / lw:.4f}", flush=True)

    leg_w = statistics.median(leg_walls)
    uni_w = statistics.median(uni_walls)
    leg_cv = statistics.pstdev(leg_walls) / statistics.mean(leg_walls)
    uni_cv = statistics.pstdev(uni_walls) / statistics.mean(uni_walls)
    flips = sum(1 for k in leg_finals if leg_finals[k] != uni_finals.get(k))
    ratio = uni_w / leg_w
    verdict["wall"] = {
        "legacy_median_s": round(leg_w, 2), "unified_median_s": round(uni_w, 2),
        "ratio_unified_over_legacy": round(ratio, 4),
        "legacy_cv": round(leg_cv, 4), "unified_cv": round(uni_cv, 4),
        "n_mixed": {"legacy": len(leg_dec), "unified": len(uni_dec)},
    }
    verdict["tokens"] = {
        "n_requests": len(leg_finals),
        "bitwise_equal_requests": len(leg_finals) - flips,
        "flips": flips,
    }
    verdict["informational_gate"] = ratio <= 1.0 and leg_cv <= 0.05 and uni_cv <= 0.05 and flips == 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=1)
    print(json.dumps(verdict, indent=1), flush=True)
    print(f"[flip30b] {'EVIDENCE PASS (30B clean)' if verdict['informational_gate'] else 'NEGATIVE (recorded)'}",
          flush=True)
    llm.exit()
    return 0 if verdict["informational_gate"] else 1


if __name__ == "__main__":
    sys.exit(main())
