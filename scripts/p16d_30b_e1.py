"""P16-D: 30B single-load E1 confirmation — tuned paged table vs production.

ONE 30B load window (weights load ~7-11 min), then interleaved ABA segments
inside that engine: legacy (32,4,1) vs NANOVLLM_PAGED_CONFIG=auto tuned table,
flipped rank-synced via runner.call("set_paged_config", ...).  Each segment =
3 staggered bursts of 8 requests (256-token prompts, 128 gen tokens) so the
paged decode kernel sees batch sizes ~8 -> 24, where E1's kernel-level wins
were largest (b8/b16/b24).

Pre-registered gate (docs/PLAN.md P16-D): auto wall <= 0.95 x legacy wall with
per-arm CV <= 5%; negative result recorded as-is.  First segment per config is
JIT warmup and discarded (P16-A SOP #8).

Run:  PYTHONPATH=. python3 scripts/p16d_30b_e1.py --out /root/p16_data/e1_30b.json
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


def run_segment(llm, config_value):
    runner = llm.model_runner
    runner.call("set_paged_config", config_value)
    llm.scheduler.block_manager.clear_prefix_cache()
    sp = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True)
    prompts = [f"第{i}段:{BODY * 4}" for i in range(24)]   # ≈256 tokens each
    decode_times = []
    wall0 = time.perf_counter()
    for burst in range(3):
        for p in prompts[burst * 8:(burst + 1) * 8]:
            llm.add_request(p, sp)
        while llm.scheduler.waiting or llm.scheduler.running:
            outputs, _ = llm.step()
            m = llm.last_step_metrics
            if m["mode"] in ("decode", "mixed"):
                decode_times.append(m["forward_s"])
    wall = time.perf_counter() - wall0
    runner.call("set_paged_config", None)
    return wall, decode_times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--gpu-util", type=float, default=0.9)
    parser.add_argument("--out", default="/root/p16_data/e1_30b.json")
    args = parser.parse_args()

    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=2,
              attention_backend="triton_paged", max_model_len=2048,
              max_num_seqs=32, gpu_memory_utilization=args.gpu_util,
              scheduler_policy="stall_free", decode_moe_backend="bmm_dense")
    verdict = {"model": args.model}

    # warmup: one segment per config, discarded (JIT/first-touch)
    run_segment(llm, None)
    run_segment(llm, "auto")

    legacy_walls, auto_walls = [], []
    legacy_dec, auto_dec = [], []
    for rep in range(args.reps):
        lw, ld = run_segment(llm, None)
        aw, ad = run_segment(llm, "auto")
        legacy_walls.append(lw)
        auto_walls.append(aw)
        legacy_dec += ld
        auto_dec += ad
        print(f"[e1-30b] rep{rep + 1}: legacy wall={lw:.2f}s "
              f"auto wall={aw:.2f}s ratio={aw / lw:.4f}", flush=True)

    leg_w = statistics.median(legacy_walls)
    auto_w = statistics.median(auto_walls)
    leg_cv = statistics.pstdev(legacy_walls) / statistics.mean(legacy_walls)
    auto_cv = statistics.pstdev(auto_walls) / statistics.mean(auto_walls)
    ratio = auto_w / leg_w
    verdict["wall"] = {
        "legacy_median_s": round(leg_w, 2),
        "auto_median_s": round(auto_w, 2),
        "ratio_auto_over_legacy": round(ratio, 4),
        "legacy_walls": [round(x, 2) for x in legacy_walls],
        "auto_walls": [round(x, 2) for x in auto_walls],
        "legacy_cv": round(leg_cv, 4),
        "auto_cv": round(auto_cv, 4),
    }
    # decode-step pooled medians (the kernel-level signal)
    ld_med = statistics.median(legacy_dec)
    ad_med = statistics.median(auto_dec)
    verdict["decode_step"] = {
        "legacy_median_s": round(ld_med, 5),
        "auto_median_s": round(ad_med, 5),
        "ratio": round(ad_med / ld_med, 4),
        "n_steps": {"legacy": len(legacy_dec), "auto": len(auto_dec)},
    }
    verdict["gate"] = ratio <= 0.95 and leg_cv <= 0.05 and auto_cv <= 0.05
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=1)
    print(json.dumps(verdict, indent=1), flush=True)
    print(f"[e1-30b] {'GATE PASS' if verdict['gate'] else 'GATE FAIL (negative result recorded)'}",
          flush=True)
    llm.exit()
    return 0 if verdict["gate"] else 1


if __name__ == "__main__":
    sys.exit(main())
