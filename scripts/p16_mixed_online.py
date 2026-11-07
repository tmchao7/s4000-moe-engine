"""P16-A gate 3: online mixed-load A/B — legacy split vs unified mixed batch.

Spawns the P15 collocated framed server twice over the SAME closed-burst
mixed workload (6 long ~1k-token + 6 short, interleaved arrivals, so many
mixed steps), toggling NANOVLLM_MIXED_UNIFIED via the server env, and
compares client-boundary e2e + engine ITL percentiles.

Pre-registered gate (docs/PLAN.md P16-A ③): unified ITL p95 <= legacy ITL
p95 (improvement expected, recorded as-is); throughput drop <= 2%.

Run:  PYTHONPATH=. python3 scripts/p16_mixed_online.py \
        --model /root/autodl-tmp/models/Qwen3-8B --out /root/p16_data/online_8b.json
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from nanovllm.pd.protocol import JsonStream, connect_retry, send_json  # noqa: E402

LONG_PARAGRAPH = (
    "大模型推理系统需要在吞吐与延迟之间做精细权衡。prefill 阶段一次性处理整段输入,"
    "计算密集,占用大量算力;decode 阶段逐 token 生成,访存密集,每一步都要读取全部 "
    "KV cache。当两种阶段混杂在同一批 GPU 上时,一个长 prompt 的 prefill 分块会不断"
    "打断正在进行的 decode 批次,造成 token 间延迟的尖刺。PagedAttention 把连续的 "
    "KV 存储切成页,连续批处理让新请求随时加入,而 stall-free 调度则保证 decode 优先。"
    "这些技术的组合构成了现代推理引擎的底座,但它们解决的仍是合部部署下的干扰缓解,"
    "而不是彻底消除——统一混合批与 PD 分离是两条互补的结构性路线。"
)
SHORT_PROMPTS = [
    "用一句话解释什么是 KV cache。",
    "Write a Python function that reverses a string.",
    "计算 17 × 23 的结果。",
    "Why is decode memory-bandwidth bound?",
    "一句话介绍连续批处理。",
    "What does TTFT stand for?",
    "写一句关于秋天的诗。",
    "Explain online softmax briefly.",
]


def build_workload(n_long, n_short):
    long_prompt = LONG_PARAGRAPH * 6   # ≈1k tokens
    reqs = []
    li = si = 0
    while li < n_long or si < n_short:
        if li < n_long:
            reqs.append(("long", long_prompt + f"（第{li}段）"))
            li += 1
        if si < n_short:
            reqs.append(("short", SHORT_PROMPTS[si % len(SHORT_PROMPTS)]))
            si += 1
    return reqs


def run_arm(model, gpu_util, reqs, max_tokens, out_dir, tag, unified):
    env = dict(os.environ, PYTHONPATH=REPO,
               NANOVLLM_MIXED_UNIFIED="1" if unified else "0")
    log = open(f"{out_dir}/{tag}_{'unified' if unified else 'legacy'}.log", "w")
    proc = subprocess.Popen(
        [sys.executable, os.path.join(REPO, "scripts/p15_pd_serve.py"),
         "--role", "collocated", "--model", model,
         "--gpu-util", str(gpu_util), "--port", "29740"],
        env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        client = connect_retry("127.0.0.1", 29740)
        client.setblocking(True)
        t0 = time.perf_counter()
        for i, (_, prompt) in enumerate(reqs):
            send_json(client, {"req_id": i, "prompt": prompt,
                               "max_tokens": max_tokens})
        results = {}
        js = JsonStream()
        while len(results) < len(reqs):
            chunk = client.recv(1 << 20)
            if not chunk:
                raise ConnectionError("server closed before all results")
            for msg in js.feed(chunk):
                if msg.get("type") == "result":
                    results[msg["req_id"]] = msg
        wall = time.perf_counter() - t0
        send_json(client, {"type": "done"})
        client.close()
    finally:
        proc.wait(timeout=300)
    return [results[i] for i in range(len(reqs))], wall


def itl_percentiles(results):
    pooled = []
    for r in results:
        pooled.extend(r["metrics"].get("itls_s") or [])
    if not pooled:
        return {}
    pooled.sort()

    def pct(q):
        idx = min(len(pooled) - 1, max(0, int(q / 100 * len(pooled)) - 1))
        return round(pooled[idx], 4)
    return {"n_itls": len(pooled), "p50": pct(50), "p95": pct(95),
            "p99": pct(99), "mean": round(statistics.fmean(pooled), 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-8B")
    parser.add_argument("--gpu-util", type=float, default=0.5)
    parser.add_argument("--n-long", type=int, default=6)
    parser.add_argument("--n-short", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=3,
                        help="interleaved legacy/unified repeats; ITLs pooled")
    parser.add_argument("--out", default="/root/p16_data/online.json")
    args = parser.parse_args()

    reqs = build_workload(args.n_long, args.n_short)
    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)
    tag = "8b" if "8B" in args.model else "0.6b"

    print(f"[online] workload: {args.n_long} long + {args.n_short} short, "
          f"max_tokens={args.max_tokens}, repeats={args.repeats} (interleaved)",
          flush=True)
    leg_all, uni_all = [], []          # pooled results across repeats
    leg_p95s, uni_p95s, leg_walls, uni_walls = [], [], [], []
    for rep in range(args.repeats):
        leg_res, leg_wall = run_arm(args.model, args.gpu_util, reqs,
                                    args.max_tokens, out_dir,
                                    f"{tag}_r{rep}", unified=False)
        uni_res, uni_wall = run_arm(args.model, args.gpu_util, reqs,
                                    args.max_tokens, out_dir,
                                    f"{tag}_r{rep}", unified=True)
        leg_all += leg_res
        uni_all += uni_res
        leg_p95s.append(itl_percentiles(leg_res).get("p95"))
        uni_p95s.append(itl_percentiles(uni_res).get("p95"))
        leg_walls.append(leg_wall)
        uni_walls.append(uni_wall)
        print(f"[online] rep{rep + 1}: legacy wall={leg_wall:.2f}s "
              f"p95={leg_p95s[-1]} | unified wall={uni_wall:.2f}s "
              f"p95={uni_p95s[-1]}", flush=True)

    leg_itl, uni_itl = itl_percentiles(leg_all), itl_percentiles(uni_all)
    leg_e2e = [r["metrics"]["e2e_s"] for r in leg_all if r["metrics"]["e2e_s"]]
    uni_e2e = [r["metrics"]["e2e_s"] for r in uni_all if r["metrics"]["e2e_s"]]
    ratio = (round(uni_itl["p95"] / leg_itl["p95"], 3)
             if leg_itl.get("p95") and uni_itl.get("p95") else None)
    wall_leg = statistics.median(leg_walls)
    wall_uni = statistics.median(uni_walls)
    thr_ratio = round(wall_leg / wall_uni, 3)   # unified throughput / legacy
    verdict = {
        "model": args.model,
        "workload": {"n_long": args.n_long, "n_short": args.n_short,
                     "max_tokens": args.max_tokens, "repeats": args.repeats},
        "legacy": {"itl": leg_itl,
                   "client_e2e_mean_s": round(statistics.fmean(leg_e2e), 3),
                   "wall_median_s": round(wall_leg, 2),
                   "per_rep_p95": leg_p95s, "per_rep_wall": [round(w, 2) for w in leg_walls],
                   "throughput_rps": round(len(reqs) / wall_leg, 3)},
        "unified": {"itl": uni_itl,
                    "client_e2e_mean_s": round(statistics.fmean(uni_e2e), 3),
                    "wall_median_s": round(wall_uni, 2),
                    "per_rep_p95": uni_p95s, "per_rep_wall": [round(w, 2) for w in uni_walls],
                    "throughput_rps": round(len(reqs) / wall_uni, 3)},
        "itl_p95_ratio_unified_over_legacy": ratio,
        "throughput_ratio_unified_over_legacy": thr_ratio,
        "gate": ratio is not None and ratio <= 1.0 and thr_ratio >= 0.98,
    }
    with open(args.out, "w") as f:
        json.dump({"verdict": verdict,
                   "legacy_results": leg_all, "unified_results": uni_all},
                  f, indent=1)
    print(json.dumps(verdict, indent=1, ensure_ascii=False), flush=True)
    print(f"[online] {'PASS' if verdict['gate'] else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
