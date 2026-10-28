"""One-load nightly D2: every determinism/stability decision in a single 30B load.

Combines (and supersedes for tonight) ``tests/p5_determinism_b32.py``,
``tests/p5_d2_stability.py`` and ``scripts/p5_attention_ab.py`` so the second
30B load of the night answers all P1 questions at once:

  0. 8-token correctness gate vs the frozen reference (loop + grouped_mm, hard).
  1. determinism_b32 cases A-D (grouped/loop/zero-KV/b16 control, 3 repeats).
  2. b32 x10 series, grouped_mm + sdpa  (the historical unstable point).
  3. b32 x10 series, grouped_mm + triton_paged (candidate determinism).
  4. b32 x10 series, grouped_mm prefill + bmm_dense decode + triton_paged
     (the full nightly candidate stack).
  5. attention A/B (sdpa vs triton_paged, 16x512/32x512/16x2048/32x2048,
     output 32, 3 repeats) written in the p5_attention_ab schema so
     ``scripts/p5_decide.py --mode attention`` consumes it unchanged.
  6. b64 x3 crash reproduction (grouped_mm + sdpa, prompt 128 / output 32) —
     runs LAST because the historical failure kills the engine.

Everything streams to stdout and to a JSONL file as ``[p6d2]`` lines, so a
crash keeps the finished evidence.  Prompts are drawn with the same seeds as
the historical scripts (20261026) so series are comparable across rounds.

Run:  PYTHONPATH=<repo> python3 tests/p6_d2_nightly.py
"""
import argparse
import importlib.util
import json
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, "/root/moe-server-s4000")
sys.path.insert(0, "/root/moe-server-s4000/scripts")

MODEL = "/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
SEED = 20261026
REPO_ROOT = Path("/root/moe-server-s4000")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


det32 = load_module("p5_det32", REPO_ROOT / "tests" / "p5_determinism_b32.py")
d2 = load_module("p5_d2", REPO_ROOT / "tests" / "p5_d2_stability.py")
from p5_profile import correctness_gate  # noqa: E402

from nanovllm.benchmarking import coefficient_of_variation, percentile  # noqa: E402

WORKLOADS = [(16, 512), (32, 512), (16, 2048), (32, 2048)]
OUTPUT_JSONL = "/root/p6_d2_nightly.jsonl"
JSONL_HANDLE = open(OUTPUT_JSONL, "w", encoding="utf-8")


def log(record):
    line = json.dumps(record, sort_keys=True, default=str)
    print("[p6d2]", line, flush=True)
    JSONL_HANDLE.write(line + "\n")
    JSONL_HANDLE.flush()


def first_divergence(reference, candidate):
    return d2.first_divergence(reference, candidate)


def run_series2(llm, params, prompts, repeats, moe_backend, decode_moe_backend,
                attention_backend, label):
    """``d2.run_series`` with the two-argument set_moe_backend."""
    llm.model_runner.call("set_attention_backend", attention_backend)
    llm.model_runner.call("set_moe_backend", moe_backend, decode_moe_backend)
    runs = []
    for _ in range(repeats):
        llm.scheduler.block_manager.clear_prefix_cache()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        runs.append([output["token_ids"] for output in outputs])
    divergences = [first_divergence(runs[0], run) for run in runs[1:]]
    stable = all(divergence is None for divergence in divergences)
    record = {
        "kind": "series",
        "case": label,
        "moe_backend": moe_backend,
        "decode_moe_backend": decode_moe_backend,
        "attention_backend": attention_backend,
        "repeats": repeats,
        "stable": stable,
        "divergences": divergences,
        "first_run_head": [row[:6] for row in runs[0][:4]] if runs else None,
    }
    log(record)
    return record


def attention_ab(llm, sampling_params_cls, seed, output_len, repeats, output_path):
    """Re-implementation of scripts/p5_attention_ab.py main loop (same schema)."""
    params = sampling_params_cls(temperature=0, max_tokens=output_len, ignore_eos=True)
    vocab_size = llm.model_runner.config.hf_config.vocab_size
    records = []
    determinism = {}
    rng = random.Random(seed)
    for batch, context in WORKLOADS:
        prompts = [
            [rng.randrange(vocab_size) for _ in range(context)]
            for _ in range(batch)
        ]
        reference_tokens = None
        reference_backend = None
        for attention_backend in ("sdpa", "triton_paged"):
            llm.model_runner.call("set_attention_backend", attention_backend)
            llm.model_runner.call("set_moe_backend", "grouped_mm")
            llm.scheduler.block_manager.clear_prefix_cache()
            llm.generate(prompts, params, use_tqdm=False)
            runs = []
            for repeat in range(repeats):
                llm.scheduler.block_manager.clear_prefix_cache()
                start = time.perf_counter()
                outputs = llm.generate(prompts, params, use_tqdm=False)
                elapsed = time.perf_counter() - start
                token_ids = [output["token_ids"] for output in outputs]
                runs.append(token_ids)
                mismatches = []
                if reference_tokens is not None:
                    mismatches = [
                        index for index, (actual, want) in enumerate(
                            zip(token_ids, reference_tokens)
                        )
                        if actual != want
                    ]
                metrics = [output["metrics"] for output in outputs]
                tpots = [row["tpot_s"] for row in metrics]
                itls = [value for row in metrics for value in row["itls_s"]]
                output_tokens = sum(row["output_tokens"] for row in metrics)
                records.append({
                    "attention_backend": attention_backend,
                    "batch": batch,
                    "context": context,
                    "repeat": repeat,
                    "elapsed_s": elapsed,
                    "output_tokens": output_tokens,
                    "output_tokens_per_s": output_tokens / elapsed,
                    "ttft_p95_s": percentile([row["ttft_s"] for row in metrics], 0.95),
                    "tpot_p95_s": percentile(tpots, 0.95),
                    "itl_p95_s": percentile(itls, 0.95),
                    "reference_backend": reference_backend,
                    "token_mismatches": mismatches,
                })
                log({
                    "kind": "ab",
                    "attention_backend": attention_backend,
                    "batch": batch,
                    "context": context,
                    "repeat": repeat,
                    "tokens_per_s": output_tokens / elapsed,
                    "mismatches": len(mismatches),
                })
            deterministic = all(run == runs[0] for run in runs)
            determinism[(attention_backend, batch, context)] = deterministic
            if reference_tokens is None:
                reference_tokens = runs[0]
                reference_backend = attention_backend
    summary = []
    for batch, context in WORKLOADS:
        for attention_backend in ("sdpa", "triton_paged"):
            rows = [
                row for row in records
                if row["attention_backend"] == attention_backend
                and row["batch"] == batch and row["context"] == context
            ]
            throughput = [row["output_tokens_per_s"] for row in rows]
            mismatch_requests = sum(len(row["token_mismatches"]) for row in rows)
            summary.append({
                "attention_backend": attention_backend,
                "batch": batch,
                "context": context,
                "throughput_median": statistics.median(throughput),
                "throughput_cv": coefficient_of_variation(throughput),
                "ttft_p95_median_s": statistics.median(row["ttft_p95_s"] for row in rows),
                "tpot_p95_median_s": statistics.median(row["tpot_p95_s"] for row in rows),
                "itl_p95_median_s": statistics.median(row["itl_p95_s"] for row in rows),
                "token_mismatch_requests": mismatch_requests,
                "token_identity": mismatch_requests == 0,
                "token_agreement": 1.0 - mismatch_requests / (len(rows) * batch),
                "deterministic": determinism.get((attention_backend, batch, context), True),
            })
            log({
                "kind": "ab_summary",
                **summary[-1],
            })
    payload = {
        "schema_version": 2,
        "model": MODEL,
        "reference_backend": "sdpa",
        "records": records,
        "summary": summary,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[p6d2] wrote {output_path}", flush=True)
    return summary


def b64_crash_repro(llm, sampling_params_cls, seed, repeats):
    vocab_size = llm.model_runner.config.hf_config.vocab_size
    rng = random.Random(seed)
    # d2 drew b32x512 then b64x128 from one rng; draw-and-discard the b32 block
    # so the b64 prompts match the historical d2 run.
    for _ in range(32):
        for _ in range(512):
            rng.randrange(vocab_size)
    prompts = [[rng.randrange(vocab_size) for _ in range(128)] for _ in range(64)]
    params = sampling_params_cls(temperature=0, max_tokens=32, ignore_eos=True)
    llm.model_runner.call("set_moe_backend", "grouped_mm")
    llm.model_runner.call("set_attention_backend", "sdpa")
    b64 = {"repeats_requested": repeats, "runs": [], "errors": []}
    for repeat in range(repeats):
        try:
            llm.scheduler.block_manager.clear_prefix_cache()
            llm.generate(prompts, params, use_tqdm=False)
            b64["runs"].append(repeat)
            log({"kind": "b64", "repeat": repeat, "status": "ok"})
        except Exception as exc:  # noqa: BLE001 - the crash is the evidence
            error = f"{type(exc).__name__}: {exc}"
            b64["errors"].append({"repeat": repeat, "error": error,
                                  "traceback": traceback.format_exc(limit=8)})
            log({"kind": "b64", "repeat": repeat, "status": "error", "error": error})
            break
    b64["reproduced"] = bool(b64["errors"])
    return b64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", default="/root/p6_d2_nightly.json")
    parser.add_argument("--ab-output", default="/root/p6_attention_ab.json")
    parser.add_argument("--skip-b64", action="store_true")
    parser.add_argument("--skip-ab", action="store_true")
    args = parser.parse_args()

    from nanovllm import LLM, SamplingParams

    llm = LLM(
        MODEL,
        enforce_eager=True,
        tensor_parallel_size=2,
        expert_parallel_size=2,
        moe_backend="grouped_mm",
        attention_backend="sdpa",
        scheduler_policy="fcfs",
        max_model_len=2088,
        gpu_memory_utilization=0.85,
    )
    results = {}
    try:
        results["correctness_gate"] = correctness_gate(llm, SamplingParams, str(REPO_ROOT / "benchmarks" / "p2-reference-8tok.json"))
        log({"kind": "correctness_gate", "passed": True})

        vocab_size = llm.model_runner.config.hf_config.vocab_size
        rng_det = random.Random(SEED)
        prompts32 = [[rng_det.randrange(vocab_size) for _ in range(512)] for _ in range(32)]
        prompts16 = prompts32[:16]
        params8 = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)

        det_results = {}
        det_results["grouped_b32"] = det32.run_case(llm, params8, prompts32, "grouped_mm", False, "A grouped_mm b32")
        det_results["loop_b32"] = det32.run_case(llm, params8, prompts32, "loop", False, "B loop b32")
        det_results["grouped_b32_zero_kv"] = det32.run_case(llm, params8, prompts32, "grouped_mm", True, "C grouped_mm b32 + zero KV")
        det_results["grouped_b16"] = det32.run_case(llm, params8, prompts16, "grouped_mm", False, "D grouped_mm b16 (control)")
        results["det_cases"] = det_results
        log({"kind": "det_cases", **det_results})

        params32 = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
        results["series_grouped_sdpa"] = d2.run_series(
            llm, params32, prompts32, args.repeats, "grouped_mm", "sdpa", "b32-grouped-sdpa")
        results["series_grouped_triton"] = d2.run_series(
            llm, params32, prompts32, args.repeats, "grouped_mm", "triton_paged", "b32-grouped-triton")
        results["series_bmmdense_triton"] = run_series2(
            llm, params32, prompts32, args.repeats,
            "grouped_mm", "bmm_dense", "triton_paged", "b32-bmmdense-triton")

        if not args.skip_ab:
            results["attention_ab_summary"] = attention_ab(
                llm, SamplingParams, SEED, 32, 3, args.ab_output)

        if not args.skip_b64:
            results["b64"] = b64_crash_repro(llm, SamplingParams, SEED, 3)
    except Exception as exc:  # noqa: BLE001 - keep evidence, record the failure
        results["fatal_error"] = f"{type(exc).__name__}: {exc}"
        results["fatal_traceback"] = traceback.format_exc(limit=12)
        log({"kind": "fatal_error", "error": results["fatal_error"]})
    finally:
        try:
            llm.exit()
        except Exception:  # noqa: BLE001 - engine may be wedged
            pass

    Path(args.output).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"[p6d2] wrote {args.output}", flush=True)
    print("[p6d2] summary=" + json.dumps({
        key: value for key, value in results.items() if key != "attention_ab_summary"
    }, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
