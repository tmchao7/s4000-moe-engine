"""P15-C Step 0: quantify decode ITL tail under collocated serving (zero GPU).

The PD survey (docs/PD-DISAGG-SURVEY.md §1.2) pre-registered the go/no-go:
PD disaggregation is only worth building if collocated (chunked-prefill)
serving shows a real decode-interference tail — ITL p95 / p50 >= 2x pooled
over a production-shaped online run.  Below that, chunked prefill already
absorbs the interference and the whole PD line archives as a recorded
negative result.

This reads existing local evidence JSONs (benchmark_online.py v3 records,
which carry per-request `itls_s`) — no server time needed.

Run:  python3 tests/p15_pd_step0.py            (from repo root)
"""
import json
import math
import os
import statistics

EVIDENCE = [
    # (label, path, note) — 30B production stack online runs
    ("eager_0.9x_1.0x", "benchmarks/data/p13/aba_A1.json", "arm A' 0.9/1.0 fractions"),
    ("eager_0.9x_1.0x_rep2", "benchmarks/data/p13/aba_A2.json", "arm A' repeat"),
    ("eager_12pt", "benchmarks/data/p12/p12_online_armA2.json", "12-point matrix"),
    ("long1024", "benchmarks/data/p13/long1024.json", "1k context, chunk-1024"),
    ("capacity_1.6x", "benchmarks/data/p13/capacity.json", "saturation bracketed"),
]


def pct(sorted_xs, q):
    if not sorted_xs:
        return float("nan")
    idx = min(len(sorted_xs) - 1, max(0, math.ceil(q / 100 * len(sorted_xs)) - 1))
    return sorted_xs[idx]


def analyze(path):
    d = json.load(open(path))
    recs = d.get("records") or d.get("requests") or []
    recs = [r for r in recs if r.get("ok") and r.get("itls_s")]
    all_itls = []
    per_req = []
    ttfts = []
    for r in recs:
        itls = [x for x in r["itls_s"] if x is not None and x > 0]
        if not itls:
            continue
        all_itls.extend(itls)
        itls.sort()
        per_req.append({"request_id": r.get("request_id"),
                        "n_itl": len(itls),
                        "p50": pct(itls, 50), "p95": pct(itls, 95),
                        "max": itls[-1]})
        ttfts.append(r["ttft_s"])
    all_itls.sort()
    ttfts.sort()
    if not all_itls:
        return None
    p50, p95 = pct(all_itls, 50), pct(all_itls, 95)
    p99 = pct(all_itls, 99)
    tail_reqs = sorted(per_req, key=lambda x: -x["p95"])
    return {
        "n_requests": len(per_req),
        "n_itls": len(all_itls),
        "itl_p50_s": round(p50, 4),
        "itl_p95_s": round(p95, 4),
        "itl_p99_s": round(p99, 4),
        "itl_max_s": round(all_itls[-1], 4),
        "ratio_p95_over_p50": round(p95 / p50, 2) if p50 else None,
        "ratio_p99_over_p50": round(p99 / p50, 2) if p50 else None,
        "ttft_p50_s": round(pct(ttfts, 50), 4),
        "ttft_p95_s": round(pct(ttfts, 95), 4),
        "worst5_request_p95": [round(x["p95"], 3) for x in tail_reqs[:5]],
        "mean_itl_s": round(statistics.fmean(all_itls), 4),
    }


def main():
    out = {"gate": "pd_step0_itl_tail", "threshold": "itl_p95/p50 >= 2.0", "sources": {}}
    for label, path, note in EVIDENCE:
        if not os.path.exists(path):
            out["sources"][label] = {"missing": path}
            continue
        res = analyze(path)
        if res is None:
            out["sources"][label] = {"note": note, "no_data": True}
            continue
        res["note"] = note
        out["sources"][label] = res
        print(f"[step0] {label:24s} n={res['n_requests']:4d} "
              f"ITL p50={res['itl_p50_s']:.3f} p95={res['itl_p95_s']:.3f} "
              f"p99={res['itl_p99_s']:.3f} max={res['itl_max_s']:.3f} "
              f"p95/p50={res['ratio_p95_over_p50']} "
              f"p99/p50={res['ratio_p99_over_p50']}", flush=True)

    ratios = [s["ratio_p99_over_p50"] for s in out["sources"].values()
              if isinstance(s, dict) and s.get("ratio_p99_over_p50")]
    ratios95 = [s["ratio_p95_over_p50"] for s in out["sources"].values()
                if isinstance(s, dict) and s.get("ratio_p95_over_p50")]
    out["verdict"] = {
        "max_p95_over_p50": max(ratios95) if ratios95 else None,
        "max_p99_over_p50": max(ratios) if ratios else None,
        "gate_2x_on_p99": bool(ratios) and max(ratios) >= 2.0,
        "gate_2x_on_p95": bool(ratios95) and max(ratios95) >= 2.0,
        "proceed": bool(ratios) and max(ratios) >= 2.0,
    }
    print(json.dumps(out["verdict"], indent=1), flush=True)
    os.makedirs("benchmarks/data/p15", exist_ok=True)
    with open("benchmarks/data/p15/step0_itl.json", "w") as f:
        json.dump(out, f, indent=1)
    print("[step0] wrote benchmarks/data/p15/step0_itl.json", flush=True)


if __name__ == "__main__":
    main()
