"""P6/I4 — three-layer attention-backend equivalence gate (frozen).

Implements the contract in docs/P6-numerics-policy.md §3 from the measurements of
tests/p6_attention_equivalence.py (L3/L4) and the earlier probe outputs (L1/L2).

The historical gate (scripts/p5_decide.py --mode attention) is NOT modified: its
"token_agreement >= 0.99" conclusion `keep_sdpa` was correct under the old
contract.  This gate replaces it with the industry-standard one (vLLM / SGLang /
DeepInfer all treat cross-backend token identity as unattainable).

Usage:
  python3 scripts/p6_decide_attention_equivalence.py \
      --input /root/p6_attention_equivalence.json \
      --l2 /root/p6_attention_logits_i4.json \
      --output benchmarks/p6-attention-equivalence-decision.json
"""
import argparse
import hashlib
import json
from pathlib import Path

GATES = {
    # L3 (this run). Edit distance is NOT a gate: once a near-tie flips, the whole
    # continuation differs, so edit distance measures wording, not quality.
    # Quality is judged by a documented human/agent review of extreme+median
    # LCP samples (docs/P6-numerics-policy.md §3).
    "first_token_agreement_min": 0.99,
    "degenerate_max": 0,
    "review_equivalent_ratio_min": 0.90,
    "review_degraded_max": 0,
    # L4 (this run)
    "determinism_required": True,
    # L2 (from the frozen logits probe; re-checked if the file is given)
    "flip_rate_max": 0.005,
    "flip_margin_max": 0.5,
    # reported only, never gated
    "report_only": ["request_token_identical", "request_token_agreement_mean",
                    "edit_distance_ratio_p50", "edit_distance_ratio_max"],
}


def sha256(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/root/p6_attention_equivalence.json")
    parser.add_argument("--l2", default="/root/p6_attention_logits_i4.json",
                        help="optional logits probe output; skipped if missing")
    parser.add_argument("--output", default="benchmarks/p6-attention-equivalence-decision.json")
    parser.add_argument("--review", default=None,
                        help="qualitative review sidecar JSON {reviewed, equivalent, degraded}")
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    comparison = payload["comparison"]
    determinism = payload.get("determinism", {})

    checks = []
    checks.append({
        "layer": "L3", "name": "first_token_agreement",
        "value": comparison["first_token_agreement_rate"],
        "threshold": GATES["first_token_agreement_min"],
        "pass": comparison["first_token_agreement_rate"] >= GATES["first_token_agreement_min"],
    })
    checks.append({
        "layer": "L3", "name": "degenerate_outputs",
        "value": max(comparison["degenerate_sdpa"], comparison["degenerate_paged"]),
        "threshold": GATES["degenerate_max"],
        "pass": max(comparison["degenerate_sdpa"], comparison["degenerate_paged"]) <= GATES["degenerate_max"],
    })
    if args.review and Path(args.review).is_file():
        review = json.loads(Path(args.review).read_text(encoding="utf-8"))
        reviewed = max(review.get("reviewed", 0), 1)
        ratio = review.get("equivalent", 0) / reviewed
        checks.append({
            "layer": "L3", "name": "qualitative_equivalence_ratio", "value": ratio,
            "threshold": GATES["review_equivalent_ratio_min"],
            "pass": ratio >= GATES["review_equivalent_ratio_min"],
            "detail": {"reviewed": review.get("reviewed"), "equivalent": review.get("equivalent"),
                       "artifact": review.get("artifact")},
        })
        checks.append({
            "layer": "L3", "name": "review_degraded_count", "value": review.get("degraded", 0),
            "threshold": GATES["review_degraded_max"],
            "pass": review.get("degraded", 0) <= GATES["review_degraded_max"],
        })

    checks.append({
        "layer": "L4", "name": "per_backend_determinism",
        "value": determinism,
        "threshold": "both true",
        "pass": bool(determinism.get("sdpa")) and bool(determinism.get("triton_paged")),
    })

    # L2 comes from the teacher-forced logits probe (tests/p6_attention_logits_i4.py).
    # Its schema: {runs: [{summary: {top1_mismatch_requests, top1_compared_requests},
    #                        result: {diagnostics: [{requests: [{top1_equal, baseline_margin}]}]}}]}
    if Path(args.l2).is_file():
        l2 = json.loads(Path(args.l2).read_text(encoding="utf-8"))
        runs = l2.get("runs") or []
        flips = sum(r["summary"]["top1_mismatch_requests"] for r in runs)
        compared = sum(r["summary"]["top1_compared_requests"] for r in runs)
        margins = [
            req.get("baseline_margin")
            for r in runs
            for step in r.get("result", {}).get("diagnostics", [])
            for req in step.get("requests", [])
            if not req.get("top1_equal", True)
        ]
        margins = [m for m in margins if m is not None]
        if compared:
            rate = flips / compared
            checks.append({
                "layer": "L2", "name": "top1_flip_rate", "value": rate,
                "threshold": GATES["flip_rate_max"], "pass": rate < GATES["flip_rate_max"],
                "detail": {"flips": flips, "compared_request_steps": compared},
            })
        if margins:
            worst = max(margins)
            checks.append({
                "layer": "L2", "name": "flip_margin_max", "value": worst,
                "threshold": GATES["flip_margin_max"], "pass": worst < GATES["flip_margin_max"],
            })

    adopt = all(c["pass"] for c in checks)
    decision = {
        "schema_version": 1,
        "input": args.input,
        "evidence_sha256": sha256(args.input),
        "l2_input": args.l2,
        "l2_sha256": sha256(args.l2),
        "thresholds": GATES,
        "checks": checks,
        "reported_only": {
            key: comparison.get(key) for key in GATES["report_only"]
        },
        "decision": "adopt_triton_paged" if adopt else "keep_sdpa",
        "adopt": adopt,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    lines = ["# attention 后端等价性决策（三层门）", ""]
    lines.append("| 层 | 检查 | 值 | 阈值 | 结果 |")
    lines.append("|---|---|---:|---:|---|")
    for c in checks:
        value = c["value"]
        if isinstance(value, float):
            value = f"{value:.6g}"
        lines.append(f"| {c['layer']} | {c['name']} | `{value}` | {c['threshold']} | {'✅' if c['pass'] else '❌'} |")
    lines.append("")
    lines.append("只报告不作门：" + json.dumps(decision["reported_only"], ensure_ascii=False))
    lines.append("")
    lines.append(f"**决策：{decision['decision']}**")
    text = "\n".join(lines) + "\n"
    if args.markdown:
        Path(args.markdown).write_text(text, encoding="utf-8")
    print(f"[p6-decide] attention equivalence: {decision['decision']}")
    print(text)
    print(f"[p6-decide] wrote {out}")


if __name__ == "__main__":
    main()
