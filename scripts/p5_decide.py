"""Frozen acceptance gates for the P5/P6 A/B experiments.

The thresholds below are declared **before** the measurements run and are
committed with the code; they must not be tuned after seeing results.  Each
invocation writes an auditable decision JSON (thresholds / observed / decision
/ evidence sha256) plus a Markdown summary, mirroring the style of
``benchmarks/p5-profile-decision.json``.

    python3 scripts/p5_decide.py --mode attention \
        --input /root/p5_attention_ab.json \
        --output benchmarks/p5-attention-ab-decision.json \
        --markdown benchmarks/P5-attention-ab.md
"""
import argparse
import hashlib
import json
from pathlib import Path

ATTENTION_GATES = {
    "baseline": "sdpa",
    "candidate": "triton_paged",
    "long_context": 2048,
    "batches": [16, 32],
    "tpot_ratio_max": 0.90,
    "throughput_ratio_min": 1.00,
    "cv_max": 0.05,
}

MOE_GATES = {
    "baseline": "grouped_mm",
    "candidates": ["vllm_fused", "fused_grouped_mm"],
    "batches": [16, 32],
    "throughput_ratio_min": 1.08,
    "tpot_ratio_max": 1.00,
    "cv_max": 0.05,
}

GRAPH_GATES = {
    "b1_gain_min": 0.15,
    "b8_gain_min": 0.05,
    "regression_ratio_min": 0.98,
    "cv_max": 0.05,
    "memory_delta_max_gib": 2.0,
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ratio(numerator, denominator):
    if denominator in (None, 0):
        return None
    return numerator / denominator


def _decision_payload(mode, input_path, thresholds, observed):
    return {
        "schema_version": 1,
        "mode": mode,
        "input": str(input_path),
        "evidence_sha256": sha256_file(input_path),
        "thresholds": thresholds,
        "observed": observed,
    }


def decide_attention(payload):
    rows = payload["summary"]
    index = {
        (row["attention_backend"], int(row["batch"]), int(row["context"])): row
        for row in rows
    }
    checks = []
    for batch in ATTENTION_GATES["batches"]:
        context = ATTENTION_GATES["long_context"]
        baseline = index.get((ATTENTION_GATES["baseline"], batch, context))
        candidate = index.get((ATTENTION_GATES["candidate"], batch, context))
        if baseline is None or candidate is None:
            checks.append({
                "batch": batch, "context": context, "pass": False,
                "reason": "missing benchmark point",
            })
            continue
        tpot_ratio = _ratio(candidate["tpot_p95_median_s"], baseline["tpot_p95_median_s"])
        throughput_ratio = _ratio(candidate["throughput_median"], baseline["throughput_median"])
        passed = (
            tpot_ratio is not None
            and throughput_ratio is not None
            and tpot_ratio <= ATTENTION_GATES["tpot_ratio_max"]
            and throughput_ratio >= ATTENTION_GATES["throughput_ratio_min"]
            and candidate["throughput_cv"] <= ATTENTION_GATES["cv_max"]
        )
        checks.append({
            "batch": batch,
            "context": context,
            "tpot_ratio": tpot_ratio,
            "throughput_ratio": throughput_ratio,
            "candidate_cv": candidate["throughput_cv"],
            "candidate_token_mismatch_requests": candidate.get("token_mismatch_requests", 0),
            "pass": passed,
        })
    token_identity = all(row.get("token_identity", True) for row in rows)
    adopt = token_identity and bool(checks) and all(check["pass"] for check in checks)
    return {
        "decision": "adopt_triton_paged" if adopt else "keep_sdpa",
        "adopt": adopt,
        "token_identity": token_identity,
        "points": checks,
    }


def decide_moe(payload):
    records = [
        row for row in payload["records"] if row.get("cache_mode", "none") == "none"
    ]
    summary = [
        row for row in payload["summary"] if row.get("cache_mode", "none") == "none"
    ]
    backends = {row["backend"] for row in summary}
    candidate = next(
        (name for name in MOE_GATES["candidates"] if name in backends), None
    )
    if candidate is None:
        return {
            "decision": "keep_grouped_mm",
            "adopt": False,
            "reason": "candidate backend not present in benchmark",
            "candidates_found": sorted(backends),
        }

    def request_tokens(backend, batch, repeat):
        for row in records:
            if (
                row["backend"] == backend
                and int(row["batch"]) == batch
                and int(row["repeat"]) == repeat
            ):
                return [request["token_ids"] for request in row.get("requests", [])]
        return None

    token_mismatches = []
    for batch in MOE_GATES["batches"]:
        for repeat in range(int(payload.get("repeats", 3))):
            baseline_tokens = request_tokens(MOE_GATES["baseline"], batch, repeat)
            candidate_tokens = request_tokens(candidate, batch, repeat)
            if baseline_tokens is None or candidate_tokens is None:
                token_mismatches.append({"batch": batch, "repeat": repeat, "reason": "missing record"})
                continue
            differing = [
                index for index, (actual, want) in enumerate(
                    zip(candidate_tokens, baseline_tokens)
                )
                if actual != want
            ]
            if differing:
                token_mismatches.append({
                    "batch": batch, "repeat": repeat, "requests": differing,
                })

    index = {(row["backend"], int(row["batch"])): row for row in summary}
    checks = []
    for batch in MOE_GATES["batches"]:
        baseline = index.get((MOE_GATES["baseline"], batch))
        measured = index.get((candidate, batch))
        if baseline is None or measured is None:
            checks.append({"batch": batch, "pass": False, "reason": "missing benchmark point"})
            continue
        throughput_ratio = _ratio(measured["throughput_median"], baseline["throughput_median"])
        tpot_ratio = _ratio(measured["tpot_p95_median_s"], baseline["tpot_p95_median_s"])
        passed = (
            throughput_ratio is not None
            and tpot_ratio is not None
            and throughput_ratio >= MOE_GATES["throughput_ratio_min"]
            and tpot_ratio <= MOE_GATES["tpot_ratio_max"]
            and measured["throughput_cv"] <= MOE_GATES["cv_max"]
        )
        checks.append({
            "batch": batch,
            "throughput_ratio": throughput_ratio,
            "tpot_ratio": tpot_ratio,
            "candidate_cv": measured["throughput_cv"],
            "pass": passed,
        })
    token_identity = not token_mismatches
    adopt = token_identity and bool(checks) and all(check["pass"] for check in checks)
    return {
        "candidate": candidate,
        "decision": f"adopt_{candidate}" if adopt else "keep_grouped_mm",
        "adopt": adopt,
        "token_identity": token_identity,
        "token_mismatches": token_mismatches,
        "points": checks,
    }


def decide_graph(payload):
    rows = payload["summary"]
    index = {(row["graph"], int(row["batch"])): row for row in rows}
    checks = []
    for batch in (1, 8):
        eager = index.get((False, batch))
        graph = index.get((True, batch))
        if eager is None or graph is None:
            checks.append({"batch": batch, "pass": False, "reason": "missing benchmark point"})
            continue
        ratio = _ratio(graph["throughput_median"], eager["throughput_median"])
        required = (
            GRAPH_GATES["b1_gain_min"] if batch == 1 else GRAPH_GATES["b8_gain_min"]
        )
        checks.append({
            "batch": batch,
            "throughput_ratio": ratio,
            "required_gain": required,
            "pass": ratio is not None and ratio >= 1.0 + required,
        })
    regressions = []
    for batch in payload.get("regression_batches", [32, 64]):
        eager = index.get((False, batch))
        graph = index.get((True, batch))
        if eager is None or graph is None:
            continue
        ratio = _ratio(graph["throughput_median"], eager["throughput_median"])
        regressions.append({
            "batch": batch,
            "throughput_ratio": ratio,
            "pass": ratio is not None and ratio >= GRAPH_GATES["regression_ratio_min"],
        })
    token_identity = all(bool(row.get("token_identity", True)) for row in rows)
    cv_ok = all(row.get("throughput_cv", 0.0) <= GRAPH_GATES["cv_max"] for row in rows)
    memory_ok = payload.get("memory_delta_gib", 0.0) <= GRAPH_GATES["memory_delta_max_gib"]
    adopt = (
        token_identity
        and cv_ok
        and memory_ok
        and bool(checks) and all(check["pass"] for check in checks)
        and all(check["pass"] for check in regressions)
    )
    return {
        "decision": "adopt_cudagraph" if adopt else "keep_eager",
        "adopt": adopt,
        "token_identity": token_identity,
        "cv_ok": cv_ok,
        "memory_ok": memory_ok,
        "points": checks,
        "regressions": regressions,
    }


def render_markdown(mode, result, thresholds):
    lines = [f"# P5 decision — {mode}", "", "## Thresholds", "", "```json",
             json.dumps(thresholds, indent=2, sort_keys=True), "```", "", "## Decision", "",
             f"**{result['decision']}**", ""]
    if mode == "attention":
        lines += ["| Batch | Context | p95 TPOT ratio | Throughput ratio | CV | Pass |",
                  "|---:|---:|---:|---:|---:|---|"]
        for point in result["points"]:
            lines.append(
                f"| {point.get('batch')} | {point.get('context')} | "
                f"{point.get('tpot_ratio', float('nan')):.3f} | "
                f"{point.get('throughput_ratio', float('nan')):.3f} | "
                f"{point.get('candidate_cv', float('nan')):.3f} | "
                f"{'PASS' if point.get('pass') else 'FAIL'} |"
            )
    elif mode == "moe":
        lines += [f"Candidate: `{result.get('candidate')}`", "",
                  "| Batch | Throughput ratio | p95 TPOT ratio | CV | Pass |",
                  "|---:|---:|---:|---:|---|"]
        for point in result["points"]:
            lines.append(
                f"| {point.get('batch')} | {point.get('throughput_ratio', float('nan')):.3f} | "
                f"{point.get('tpot_ratio', float('nan')):.3f} | "
                f"{point.get('candidate_cv', float('nan')):.3f} | "
                f"{'PASS' if point.get('pass') else 'FAIL'} |"
            )
    else:
        lines += ["| Batch | Throughput ratio | Pass |", "|---:|---:|---|"]
        for point in result["points"] + result.get("regressions", []):
            lines.append(
                f"| {point.get('batch')} | {point.get('throughput_ratio', float('nan')):.3f} | "
                f"{'PASS' if point.get('pass') else 'FAIL'} |"
            )
    lines += ["", f"token_identity: `{result.get('token_identity')}`", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("attention", "moe", "graph"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.mode == "attention":
        thresholds = ATTENTION_GATES
        result = decide_attention(payload)
    elif args.mode == "moe":
        thresholds = MOE_GATES
        result = decide_moe(payload)
    else:
        thresholds = GRAPH_GATES
        result = decide_graph(payload)

    decision = _decision_payload(args.mode, args.input, thresholds, result)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(
            render_markdown(args.mode, result, thresholds), encoding="utf-8"
        )
    print(f"[p5-decide] {args.mode}: {result['decision']}", flush=True)
    print(f"[p5-decide] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
