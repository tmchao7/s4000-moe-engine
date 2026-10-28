#!/usr/bin/env python3
"""P12 online A/B comparator: arm A (default stack) vs arm B (NANOVLLM_SEGMENT_GRAPH=1).

Gates are frozen in benchmarks/P12-serving.md (written before any run):
  1. per-tier slo_standard & slo_strict: B_median >= A_median - 1/requests
  2. tier 1.0 TPOT p95: B_median <= A_median (no tolerance)
  3. tier 0.5: TPOT p95 B <= 0.97*A  AND  TTFT p95 B <= 1.02*A
Medians over the 3 repeats; per-repeat rows printed for visibility.
"""
import argparse
import json
from pathlib import Path


def med(values):
    values = sorted(values)
    n = len(values)
    if n == 0:
        return float("nan")
    mid = n // 2
    return values[mid] if n % 2 else 0.5 * (values[mid - 1] + values[mid])


def load(path):
    payload = json.loads(Path(path).read_text())
    by_frac = {}
    for row in payload["summary"]:
        by_frac.setdefault(float(row["load_fraction"]), []).append(row)
    return payload, by_frac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm_a")
    parser.add_argument("arm_b")
    parser.add_argument("--verdict-out", default=None)
    args = parser.parse_args()

    payload_a, a = load(args.arm_a)
    payload_b, b = load(args.arm_b)
    # records carry per-request rows; request count per point = requests arg
    requests_per_point = max(
        sum(1 for r in payload_a["records"] if r["load_fraction"] == f and r["repeat"] == 1)
        for f in set(r["load_fraction"] for r in payload_a["records"])
    )
    tol = 1.0 / requests_per_point
    metrics = ("slo_standard_rate", "slo_strict_rate", "ttft_p50_s", "ttft_p95_s",
               "tpot_p95_s", "itl_p95_s", "e2e_p95_s", "request_throughput_per_s",
               "goodput_tokens_per_s")

    print(f"requests_per_point={requests_per_point} tolerance={tol:.4f}")
    table = {}
    for frac in sorted(a):
        print(f"\n== fraction {frac} ==")
        table[frac] = {}
        for metric in metrics:
            av = [row.get(metric) for row in a[frac] if row.get(metric) is not None]
            bv = [row.get(metric) for row in b[frac] if row.get(metric) is not None]
            am, bm = med(av), med(bv)
            delta = (bm - am) / am * 100 if am else float("nan")
            table[frac][metric] = {"a_repeats": av, "b_repeats": bv,
                                   "a_median": am, "b_median": bm, "delta_pct": delta}
            print(f"  {metric:26s} A={am:.4f}  B={bm:.4f}  delta={delta:+.1f}%"
                  f"   (A repeats {['%.4f' % v for v in av]})")

    gate1_fail = []
    for frac in sorted(a):
        for metric in ("slo_standard_rate", "slo_strict_rate"):
            bm = table[frac][metric]["b_median"]
            am = table[frac][metric]["a_median"]
            if bm < am - tol:
                gate1_fail.append({"fraction": frac, "metric": metric,
                                   "a_median": am, "b_median": bm, "tol": tol})
    tpot10_a = table[1.0]["tpot_p95_s"]["a_median"]
    tpot10_b = table[1.0]["tpot_p95_s"]["b_median"]
    tpot05_a = table[0.5]["tpot_p95_s"]["a_median"]
    tpot05_b = table[0.5]["tpot_p95_s"]["b_median"]
    ttft05_a = table[0.5]["ttft_p95_s"]["a_median"]
    ttft05_b = table[0.5]["ttft_p95_s"]["b_median"]
    gate2_pass = tpot10_b <= tpot10_a
    gate3_pass = (tpot05_b <= 0.97 * tpot05_a) and (ttft05_b <= 1.02 * ttft05_a)
    verdict = {
        "gate1_slo_not_worse": "PASS" if not gate1_fail else "FAIL",
        "gate1_failures": gate1_fail,
        "gate2_tpot_p95_1.0": "PASS" if gate2_pass else "FAIL",
        "gate2_numbers": {"a_median": tpot10_a, "b_median": tpot10_b},
        "gate3_low_load_gain": "PASS" if gate3_pass else "FAIL",
        "gate3_numbers": {"tpot_a": tpot05_a, "tpot_b": tpot05_b,
                          "tpot_gain_pct": (tpot05_a - tpot05_b) / tpot05_a * 100,
                          "ttft_a": ttft05_a, "ttft_b": ttft05_b},
        "overall": ("PASS" if not gate1_fail and gate2_pass and gate3_pass else "FAIL"),
    }
    print("\n== verdict ==")
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    if args.verdict_out:
        Path(args.verdict_out).write_text(json.dumps(verdict, indent=2, ensure_ascii=False))
        print(f"verdict written to {args.verdict_out}")


if __name__ == "__main__":
    main()
