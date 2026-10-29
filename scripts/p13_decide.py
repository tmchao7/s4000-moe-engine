"""P13 offline evidence checks. No engine imports or mutations of frozen gates."""
import argparse
import json
import math
import statistics
from pathlib import Path


def spread(values):
    return statistics.stdev(values) / statistics.mean(values) if len(values) > 1 and statistics.mean(values) else math.inf


def aba(a1, b, a2):
    result = {"kind": "p13_aba", "points": {}, "overall": "INVALID"}
    keys = ("model", "max_qps", "requests", "repeats", "seed", "prompt_words", "prompt_mode", "max_tokens", "request_timeout_s")
    configs = [{k: p.get("args", {}).get(k) for k in keys} for p in (a1, b, a2)]
    if configs[0] != configs[1] or configs[0] != configs[2]:
        result["reason"] = "benchmark configuration mismatch"
        return result
    for fraction in (0.9, 1.0):
        point = {}
        for name, payload in (("A1", a1), ("B", b), ("A2", a2)):
            rows = [r for r in payload.get("summary", []) if r.get("load_fraction") == fraction]
            valid = (len(rows) == 3 and {r.get("repeat") for r in rows} == {1, 2, 3}
                     and all(r.get("counts", {}).get("attempted") == 128
                             and r.get("counts", {}).get("ok") == 128
                             and isinstance(r.get("tpot_p95_s"), (float, int))
                             and math.isfinite(r["tpot_p95_s"]) and r["tpot_p95_s"] > 0 for r in rows))
            values = [r["tpot_p95_s"] for r in rows] if valid else []
            point[name] = {"values": values, "median": statistics.median(values) if values else None,
                           "cv": spread(values) if values else None, "complete": valid}
        if not all(point[n]["complete"] for n in ("A1", "B", "A2")):
            point.update(verdict="INVALID", reason="missing/incomplete/failed requests")
        else:
            x, y, z = (point[n]["median"] for n in ("A1", "B", "A2"))
            point["old_drift"] = max(x, z) / min(x, z) - 1
            point["relative_to_A1"] = y / x - 1
            point["relative_to_A2"] = y / z - 1
            if max(point[n]["cv"] for n in ("A1", "B", "A2")) > .05 or point["old_drift"] > .05:
                point.update(verdict="INVALID", reason="CV or old-arm drift >5%")
            elif y <= min(x, z):
                point["verdict"] = "NO_REGRESSION"
            elif y >= 1.05 * max(x, z):
                point["verdict"] = "REGRESSION"
            else:
                point["verdict"] = "INCONCLUSIVE"
        result["points"][str(fraction)] = point
    verdicts = [p["verdict"] for p in result["points"].values()]
    if "INVALID" in verdicts:
        result["overall"] = "INVALID"
    elif "REGRESSION" in verdicts:
        result["overall"] = "REGRESSION"
    elif all(v == "NO_REGRESSION" for v in verdicts):
        result["overall"] = "NO_REGRESSION"
    else:
        result["overall"] = "INCONCLUSIVE"
    return result


def padding(payloads):
    result = {"kind": "p13_padding", "batches": {}, "overall": "INVALID"}
    sizes = sorted({int(b) for p in payloads for b in p.get("batches", {})})
    valid = bool(sizes) and all(p.get("complete") for p in payloads)
    for size in sizes:
        row = {}
        for mode in ("eager", "segment"):
            entries = [p["batches"][str(size)] for p in payloads
                       if p["mode"] == mode and str(size) in p.get("batches", {})]
            values = [e["decode_median_ms"] for e in entries]
            ok = (len(entries) >= 5 and spread(values) <= .05
                  and all(e["cv"] <= .05 and e["actual_batch_valid"] and e["tokens_consistent"] for e in entries))
            row[mode] = {"values": values, "median_ms": statistics.median(values) if values else None,
                         "cv": spread(values) if values else None, "valid": ok,
                         "caps": sorted({e["cap"] for e in entries}),
                         "token_hashes": sorted({r["token_sha256"] for e in entries for r in e["rounds"][1:]})}
            valid = valid and ok
        row["cross_mode_tokens_equal"] = row["eager"]["token_hashes"] == row["segment"]["token_hashes"]
        valid = valid and row["cross_mode_tokens_equal"]
        result["batches"][str(size)] = row
    if valid and {16, 17}.issubset(sizes):
        ratios = {mode: result["batches"]["17"][mode]["median_ms"] / result["batches"]["16"][mode]["median_ms"]
                  for mode in ("eager", "segment")}
        result["b17_b16"] = ratios
        result["overall"] = "STEP_SUPPORTED" if ratios["segment"] >= 1.15 and ratios["eager"] <= 1.10 else "STEP_NOT_SUPPORTED"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["aba", "padding"])
    parser.add_argument("files", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payloads = [json.loads(Path(f).read_text()) for f in args.files]
    if args.kind == "aba" and len(payloads) != 3:
        parser.error("ABA requires A1 B A2 in order")
    result = aba(*payloads) if args.kind == "aba" else padding(payloads)
    result["sources"] = args.files
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
