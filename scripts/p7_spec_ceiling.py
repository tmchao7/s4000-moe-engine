"""Analytic ceiling for speculative decoding, from the frozen P7 measurements.

No model is loaded.  The point is to turn "one measured k=1 point" into a
bound on every (k, alpha) pair, which a re-run could not do any better.

Speedup model for a naive draft (k draft steps + one k+1-token verification):

    S(k, alpha, c) = (1 - alpha^(k+1)) / ((1 - alpha) * (1 + k*c))

where ``alpha`` is the per-token acceptance rate and ``c`` the draft step time
divided by the target step time.  The model is calibrated against the measured
k=1 point before any extrapolation is reported.

Run:
  python3 scripts/p7_spec_ceiling.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "benchmarks" / "data" / "p7"
OUTPUT = DATA / "p7_spec_ceiling.json"

DRAFT_FILES = {
    "Qwen3-0.6B": DATA / "p7_alpha_06_tp2.json",
    "Qwen3-1.7B": DATA / "p7_alpha_17_tp2.json",
}
KS = [1, 2, 3, 4, 5, 8]
GATE = 1.3
# Illustrative "I19 fixed" scenarios: if the per-step fixed cost disappeared,
# both steps would shrink and c would fall.  0.067 = 3 ms draft / 45 ms target.
SCENARIO_C = {"current": None, "i19_fixed_light": 0.067, "i19_fixed_worst": 0.10}


def speedup(k, alpha, c):
    return (1 - alpha ** (k + 1)) / ((1 - alpha) * (1 + k * c))


def min_alpha(k, c, target):
    """Smallest alpha in [0,1) whose speedup reaches ``target`` (bisection)."""
    if speedup(k, 1 - 1e-12, c) < target:
        return None
    low, high = 0.0, 1 - 1e-12
    for _ in range(200):
        mid = (low + high) / 2
        if speedup(k, mid, c) >= target:
            high = mid
        else:
            low = mid
    return high


def main():
    payload = {"schema_version": 1, "kind": "spec_decode_ceiling", "gate": GATE,
               "drafts": {}, "tables": {}}
    for name, path in DRAFT_FILES.items():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        entry = frozen["drafts"][name]
        alpha = entry["alpha_marginal"]
        c = entry["c_b1"]
        measured = entry["best"]
        calibrated = speedup(1, alpha, c)
        payload["drafts"][name] = {
            "source": str(path.relative_to(REPO)),
            "alpha": alpha,
            "c": c,
            "draft_step_b1_s": entry["draft_step_b1_s"],
            "target_step_b1_s": frozen["target_step_s"]["b1"],
            "measured_best": measured,
            "model_at_k1": calibrated,
            "model_vs_measured": calibrated / measured["speedup"] - 1,
            "alpha_gap_vs_breakeven_k1": min_alpha(1, c, 1.0) / alpha,
        }
        table = {}
        for k in KS:
            table[f"k={k}"] = {
                "speedup_at_measured_alpha": speedup(k, alpha, c),
                "breakeven_alpha": min_alpha(k, c, 1.0),
                "gate_alpha": min_alpha(k, c, GATE),
                "perfect_draft_ceiling": speedup(k, 1 - 1e-12, c),
            }
        payload["tables"][name] = {"c": c, "alpha": alpha, "k": table}

        scenarios = {}
        for label, scenario_c in SCENARIO_C.items():
            if scenario_c is None:
                continue
            scenarios[label] = {
                "c": scenario_c,
                "k": {
                    f"k={k}": {
                        "breakeven_alpha": min_alpha(k, scenario_c, 1.0),
                        "gate_alpha": min_alpha(k, scenario_c, GATE),
                        "perfect_draft_ceiling": speedup(k, 1 - 1e-12, scenario_c),
                    }
                    for k in KS
                },
            }
        payload["tables"][name]["scenarios"] = scenarios

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name, info in payload["drafts"].items():
        print(
            f"[ceiling] {name}: alpha={info['alpha']:.4f} c={info['c']:.3f} "
            f"model(k=1)={info['model_at_k1']:.3f} measured={info['measured_best']['speedup']:.3f} "
            f"alpha_gap={info['alpha_gap_vs_breakeven_k1']:.1f}x",
            flush=True,
        )
    print(f"[ceiling] wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
