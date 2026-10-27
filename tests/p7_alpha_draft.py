"""P7 Phase 0c — draft replay + acceptance analysis + go/no-go verdict.

Loads each draft candidate, greedily regenerates the same 32 chat prompts, and
compares each output against the frozen 30B reference from Phase 0b via
longest-common-prefix (LCP).  Greedy speculative decoding emits
``min(LCP, k) + 1`` tokens per round (k proposed + 1 bonus), so
``E[min(LCP,k)+1]`` from the measured LCP distribution is the expected
per-round emission; the speedup is that divided by the round-cost factor
``1 + (k+1)*c`` where ``c = draft_step / target_step`` from Phase 0a.

The go/no-go gate (frozen in the approved plan): best-k single-stream
speedup >= 1.3x.

Run:  PYTHONPATH=/root/moe-server-s4000 python3 tests/p7_alpha_draft.py
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/root/moe-server-s4000")

# Frozen target step times (Phase 0 baseline, P6 matrix, adopted stack b1):
# single-stream is the gate scenario.
TARGET_STEP_S = {"b1": 0.1653, "b8": 0.1742, "b32": 0.2488}


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="/root/autodl-tmp/models/Qwen3-0.6B,/root/autodl-tmp/models/Qwen3-1.7B")
    parser.add_argument("--reference", default="/root/p7_target_ref.json")
    parser.add_argument("--draft-steps", default="/root/p7_draft_steps.json")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--gate", type=float, default=1.3)
    parser.add_argument("--output", default="/root/p7_alpha.json")
    args = parser.parse_args()

    from nanovllm import LLM, SamplingParams
    from nanovllm.layers.rotary_embedding import get_rope

    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    prompts = [row["prompt"] for row in reference["rows"]]
    target_tokens = [row["token_ids"] for row in reference["rows"]]
    target_len = reference["max_tokens"]

    draft_steps = json.loads(Path(args.draft_steps).read_text(encoding="utf-8"))
    step_by_model = {
        (r["model"], r["batch"]): r["itl_p50_s"] for r in draft_steps["records"]
    }

    analysis = {"gate": args.gate, "target_step_s": TARGET_STEP_S, "drafts": {}}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        name = Path(model).name
        get_rope.cache_clear()
        llm = LLM(
            model,
            enforce_eager=True,
            tensor_parallel_size=2,
            attention_backend="sdpa",
            scheduler_policy="fcfs",
            max_model_len=1024,
            gpu_memory_utilization=0.5,
        )
        try:
            params = SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True)
            outputs = llm.generate(prompts, params, use_tqdm=False)
            draft_tokens = [list(o["token_ids"]) for o in outputs]
        finally:
            llm.exit()
            get_rope.cache_clear()

        lcps = [lcp(t, d) for t, d in zip(target_tokens, draft_tokens)]
        alpha1 = sum(lcps) / max(sum(min(len(t), len(d)) for t, d in zip(target_tokens, draft_tokens)), 1)
        entry = {
            "lcp_per_prompt": lcps,
            "lcp_mean": statistics.mean(lcps),
            "lcp_median": statistics.median(lcps),
            "alpha_marginal": alpha1,
            "exact_match_prompts": sum(1 for l in lcps if l >= target_len),
            "speedup": {},
        }
        # single-stream (b1) verdict with measured c
        d1 = step_by_model.get((name, 1))
        c = d1 / TARGET_STEP_S["b1"] if d1 else None
        entry["draft_step_b1_s"] = d1
        entry["c_b1"] = c
        best = None
        for k in (1, 2, 3, 4, 5):
            emitted = statistics.mean(min(l, k) + 1 for l in lcps)
            su = emitted / (1 + (k + 1) * c) if c else None
            entry["speedup"][f"k={k}"] = {"emitted_per_round": emitted, "speedup": su}
            if su and (best is None or su > best[1]):
                best = (k, su)
        entry["best"] = {"k": best[0], "speedup": best[1]} if best else None
        entry["go"] = bool(best and best[1] >= args.gate)
        analysis["drafts"][name] = entry
        print(
            f"[p7c] {name}: alpha1={alpha1:.3f} lcp_mean={entry['lcp_mean']:.1f} "
            f"c={c:.3f} best_k={best[0] if best else '?'} "
            f"speedup={best[1]:.2f}x go={entry['go']}" if best else f"[p7c] {name}: no timing",
            flush=True,
        )

    Path(args.output).write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[p7c] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
