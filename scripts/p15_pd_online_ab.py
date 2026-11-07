"""P15-C-M3: collocated vs PD online A/B on identical hardware + workload.

Two arms, same 2-card budget, same closed-burst mixed workload:
  arm A "collocated": TP2 engine across both cards (standard production shape)
  arm B "pd":         P (card 0, TP1) + D (card 1, TP1) via the concurrent
                      control plane (scripts/p15_pd_serve.py)

Measurements at the client boundary (identical for both arms): per-request
e2e latency; engine-reported per-request decode ITLs (itls_s) pooled into
p50/p95/p99 per arm.  Step 0 established that collocated decode carries a
real interference tail (ITL p99/p50 2.3-6.3x); the question here is whether
PD removes it.

Pre-registered verdict (docs/PLAN.md P15-C-M3): pd_itl_p95 < collocated_itl_p95
=> positive result; otherwise negative, recorded as-is.

Run:  python3 scripts/p15_pd_online_ab.py --model .../Qwen3-8B --out /root/p15_data/pd_ab_8b.json
"""
import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from nanovllm.pd.protocol import connect_retry, recvn_blocking, send_json  # noqa: E402

LONG_PARAGRAPH = (
    "大模型推理系统需要在吞吐与延迟之间做精细权衡。prefill 阶段一次性处理整段输入,"
    "计算密集,占用大量算力;decode 阶段逐 token 生成,访存密集,每一步都要读取全部 "
    "KV cache。当两种阶段混杂在同一批 GPU 上时,一个长 prompt 的 prefill 分块会不断"
    "打断正在进行的 decode 批次,造成 token 间延迟的尖刺。PagedAttention 把连续的 "
    "KV 存储切成页,连续批处理让新请求随时加入,而 stall-free 调度则保证 decode 优先。"
    "这些技术的组合构成了现代推理引擎的底座,但它们解决的仍是合部部署下的干扰缓解,"
    "而不是彻底消除——PD 分离把两个阶段放到不同的硬件上,才是结构性的解法。"
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
    """Interleave long and short prompts; return list of prompt strings."""
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


def run_arm(role, model, gpu_util, reqs, max_tokens, out_dir, tag):
    """Spawn server role(s), drive reqs through the client protocol, return
    (results, wall_s)."""
    procs = []
    env = dict(os.environ, PYTHONPATH=REPO)
    try:
        if role == "collocated":
            procs.append(subprocess.Popen(
                [sys.executable, os.path.join(REPO, "scripts/p15_pd_serve.py"),
                 "--role", "collocated", "--model", model,
                 "--gpu-util", str(gpu_util), "--port", "29700"],
                env=env, stdout=open(f"{out_dir}/{tag}_collocated.log", "w"),
                stderr=subprocess.STDOUT))
            client_port = 29700
        else:
            procs.append(subprocess.Popen(
                [sys.executable, os.path.join(REPO, "scripts/p15_pd_serve.py"),
                 "--role", "decode", "--model", model,
                 "--gpu-util", str(gpu_util), "--port", "29720"],
                env={**env, "MUSA_VISIBLE_DEVICES": "1"},
                stdout=open(f"{out_dir}/{tag}_decode.log", "w"),
                stderr=subprocess.STDOUT))
            procs.append(subprocess.Popen(
                [sys.executable, os.path.join(REPO, "scripts/p15_pd_serve.py"),
                 "--role", "prefill", "--model", model,
                 "--gpu-util", str(gpu_util), "--port", "29710",
                 "--d-port", "29720"],
                env=env, stdout=open(f"{out_dir}/{tag}_prefill.log", "w"),
                stderr=subprocess.STDOUT))
            client_port = 29710

        client = connect_retry("127.0.0.1", client_port)
        client.setblocking(True)
        t0 = time.perf_counter()
        for i, (_, prompt) in enumerate(reqs):
            send_json(client, {"req_id": i, "prompt": prompt,
                               "max_tokens": max_tokens})
        results = {}
        from nanovllm.pd.protocol import JsonStream
        js = JsonStream()
        while len(results) < len(reqs):
            chunk = client.recv(1 << 20)   # blocking recv returns what is available
            if not chunk:
                raise ConnectionError("server closed before all results")
            for msg in js.feed(chunk):
                if msg.get("type") == "result":
                    results[msg["req_id"]] = msg
        wall = time.perf_counter() - t0
        send_json(client, {"type": "done"})
        client.close()
    finally:
        for proc in procs:
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
    parser.add_argument("--out", default="/root/p15_data/pd_ab.json")
    args = parser.parse_args()

    reqs = build_workload(args.n_long, args.n_short)
    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)
    tag = "8b" if "8B" in args.model else "0.6b"

    print(f"[ab] workload: {args.n_long} long + {args.n_short} short, "
          f"max_tokens={args.max_tokens}", flush=True)
    col_res, col_wall = run_arm("collocated", args.model, args.gpu_util,
                                reqs, args.max_tokens, out_dir, tag)
    print(f"[ab] collocated wall={col_wall:.2f}s", flush=True)
    pd_res, pd_wall = run_arm("pd", args.model, args.gpu_util,
                              reqs, args.max_tokens, out_dir, tag)
    print(f"[ab] pd wall={pd_wall:.2f}s", flush=True)

    col_itl, pd_itl = itl_percentiles(col_res), itl_percentiles(pd_res)
    col_e2e = [r["metrics"]["e2e_s"] for r in col_res if r["metrics"]["e2e_s"]]
    pd_e2e = [r["metrics"]["e2e_s"] for r in pd_res if r["metrics"]["e2e_s"]]
    ratio = (round(pd_itl["p95"] / col_itl["p95"], 3)
             if col_itl.get("p95") and pd_itl.get("p95") else None)
    verdict = {
        "model": args.model,
        "workload": {"n_long": args.n_long, "n_short": args.n_short,
                     "max_tokens": args.max_tokens},
        "collocated": {"itl": col_itl,
                       "client_e2e_mean_s": round(statistics.fmean(col_e2e), 3),
                       "wall_s": round(col_wall, 2),
                       "throughput_rps": round(len(reqs) / col_wall, 3)},
        "pd": {"itl": pd_itl,
               "client_e2e_mean_s": round(statistics.fmean(pd_e2e), 3),
               "wall_s": round(pd_wall, 2),
               "throughput_rps": round(len(reqs) / pd_wall, 3),
               "prefill_s_mean": round(statistics.fmean(
                   [r["prefill_s"] for r in pd_res if r.get("prefill_s")]), 4)},
        "itl_p95_ratio_pd_over_collocated": ratio,
        "pass": ratio is not None and ratio < 1.0,
    }
    with open(args.out, "w") as f:
        json.dump({"verdict": verdict,
                   "collocated_results": col_res, "pd_results": pd_res}, f, indent=1)
    print(json.dumps(verdict, indent=1, ensure_ascii=False), flush=True)
    print(f"[ab] {'POSITIVE (PD improves ITL p95)' if verdict['pass'] else 'NEGATIVE (recorded as-is)'}",
          flush=True)


if __name__ == "__main__":
    main()
