"""P18-A: PD vs collocated under Poisson (open-loop) arrivals.

P15-M3 judged PD vs collocated on a CLOSED burst; this re-runs the same
verdict under sustained Poisson load (client-side expovariate arrivals) —
the production-shaped question.  Arms (interleaved, poolable):
  collocated  TP2 engine, production stack (legacy mixed path)
  pd          1P1D: P card0 + D card1 via the P15 control plane

Pre-registered gate (docs/PLAN.md P18-A): pooled PD ITL p95 < collocated
ITL p95; TTFT compared and recorded.  Negative results recorded as-is.

Run:  PYTHONPATH=. python3 scripts/p18_pd_poisson.py \
        --model /root/autodl-tmp/models/Qwen3-0.6B --poisson-qps 0.5 \
        --out /root/p18_data/pd_poisson_06b.json
"""
import argparse
import importlib.util
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "p16mo", os.path.join(REPO, "scripts", "p16_mixed_online.py"))
p16mo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p16mo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    parser.add_argument("--gpu-util", type=float, default=0.5)
    parser.add_argument("--n-long", type=int, default=6)
    parser.add_argument("--n-short", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--poisson-qps", type=float, default=0.5)
    parser.add_argument("--out", default="/root/p18_data/pd_poisson.json")
    args = parser.parse_args()

    reqs = p16mo.build_workload(args.n_long, args.n_short)
    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)
    tag = "8b" if "8B" in args.model else "0.6b"

    print(f"[pd-poisson] qps={args.poisson_qps} repeats={args.repeats} "
          f"workload={len(reqs)} reqs", flush=True)
    coll_all, pd_all = [], []
    coll_p95s, pd_p95s, coll_ttfts, pd_ttfts = [], [], [], []
    for rep in range(args.repeats):
        c_res, c_wall = p16mo.run_arm(args.model, args.gpu_util, reqs,
                                      args.max_tokens, out_dir,
                                      f"{tag}_pdpo_r{rep}", unified=False,
                                      role="collocated",
                                      poisson_qps=args.poisson_qps)
        p_res, p_wall = p16mo.run_arm(args.model, args.gpu_util, reqs,
                                      args.max_tokens, out_dir,
                                      f"{tag}_pdpo_r{rep}", unified=False,
                                      role="pd",
                                      poisson_qps=args.poisson_qps)
        coll_all += c_res
        pd_all += p_res
        coll_p95s.append(p16mo.itl_percentiles(c_res).get("p95"))
        pd_p95s.append(p16mo.itl_percentiles(p_res).get("p95"))
        coll_ttfts.append(statistics.fmean(
            [r["metrics"]["ttft_s"] for r in c_res if r["metrics"].get("ttft_s")]))
        pd_ttfts.append(statistics.fmean(
            [r["metrics"]["ttft_s"] for r in p_res if r["metrics"].get("ttft_s")]))
        print(f"[pd-poisson] rep{rep + 1}: coll wall={c_wall:.2f}s p95={coll_p95s[-1]} "
              f"ttft={coll_ttfts[-1]:.3f} | pd wall={p_wall:.2f}s p95={pd_p95s[-1]} "
              f"ttft={pd_ttfts[-1]:.3f}", flush=True)

    coll_itl = p16mo.itl_percentiles(coll_all)
    pd_itl = p16mo.itl_percentiles(pd_all)
    ratio = (round(pd_itl["p95"] / coll_itl["p95"], 3)
             if coll_itl.get("p95") and pd_itl.get("p95") else None)
    verdict = {
        "model": args.model,
        "workload": {"qps": args.poisson_qps, "repeats": args.repeats,
                     "max_tokens": args.max_tokens},
        "collocated": {"itl": coll_itl, "ttft_mean_s": round(statistics.fmean(coll_ttfts), 3),
                       "per_rep_p95": coll_p95s},
        "pd": {"itl": pd_itl, "ttft_mean_s": round(statistics.fmean(pd_ttfts), 3),
               "per_rep_p95": pd_p95s},
        "itl_p95_ratio_pd_over_collocated": ratio,
        "pass": ratio is not None and ratio < 1.0,
    }
    with open(args.out, "w") as f:
        json.dump({"verdict": verdict, "collocated_results": coll_all,
                   "pd_results": pd_all}, f, indent=1)
    print(json.dumps(verdict, indent=1, ensure_ascii=False), flush=True)
    print(f"[pd-poisson] {'POSITIVE' if verdict['pass'] else 'NEGATIVE (recorded as-is)'}",
          flush=True)


if __name__ == "__main__":
    main()
