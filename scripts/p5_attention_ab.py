"""One-load SDPA vs Triton paged-attention correctness/performance A/B."""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

from nanovllm.benchmarking import coefficient_of_variation, percentile

# Import by file path: some environments ship an unrelated site-packages
# ``scripts`` package that shadows this repository's namespace package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p5_profile import correctness_gate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_workloads(value):
    return [tuple(map(int, item.lower().split("x"))) for item in value.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--workloads", default="16x512,32x512,16x2048,32x2048")
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--attention-backends", default="sdpa,triton_paged")
    parser.add_argument("--output", default="/root/p5_attention_ab.json")
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument(
        "--reference",
        default=str(REPO_ROOT / "benchmarks" / "p2-reference-8tok.json"),
    )
    args = parser.parse_args()
    workloads = parse_workloads(args.workloads)
    attention_backends = [item.strip() for item in args.attention_backends.split(",")]
    assert all(item in ("sdpa", "triton_paged") for item in attention_backends)

    from nanovllm import LLM, SamplingParams

    max_context = max(context for _, context in workloads)
    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=2,
        expert_parallel_size=2,
        moe_backend="grouped_mm",
        attention_backend="sdpa",
        scheduler_policy="fcfs",
        max_model_len=max_context + args.output_len + 8,
        gpu_memory_utilization=0.85,
    )
    rng = random.Random(args.seed)
    vocab_size = llm.model_runner.config.hf_config.vocab_size
    params = SamplingParams(temperature=0, max_tokens=args.output_len, ignore_eos=True)
    records = []
    verification = None
    try:
        verification = correctness_gate(llm, SamplingParams, args.reference)
        for batch, context in workloads:
            prompts = [
                [rng.randrange(vocab_size) for _ in range(context)]
                for _ in range(batch)
            ]
            reference_tokens = None
            reference_backend = None
            for attention_backend in attention_backends:
                llm.model_runner.call("set_attention_backend", attention_backend)
                llm.scheduler.block_manager.clear_prefix_cache()
                llm.generate(prompts, params, use_tqdm=False)
                runs = []
                for repeat in range(args.repeats):
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
                        "requests": [
                            {**row, "token_ids": output["token_ids"]}
                            for row, output in zip(metrics, outputs)
                        ],
                    })
                    print(
                        f"[p5-ab] {attention_backend} b={batch} c={context} "
                        f"repeat={repeat} tok/s={output_tokens / elapsed:.2f} "
                        f"mismatches={len(mismatches)}",
                        flush=True,
                    )
                # Within-backend non-determinism is a real bug: always fatal.
                if not all(run == runs[0] for run in runs):
                    raise RuntimeError(
                        f"{attention_backend} nondeterministic at batch={batch} context={context}"
                    )
                if reference_tokens is None:
                    reference_tokens = runs[0]
                    reference_backend = attention_backend
                else:
                    mismatches = [
                        index for index, (actual, want) in enumerate(
                            zip(runs[0], reference_tokens)
                        )
                        if actual != want
                    ]
                    if mismatches:
                        first = mismatches[0]
                        print(
                            f"[p5-ab] TOKEN_MISMATCH {attention_backend} vs {reference_backend} "
                            f"batch={batch} context={context} requests={mismatches} "
                            f"first_actual={runs[0][first]} first_expected={reference_tokens[first]}",
                            flush=True,
                        )
    finally:
        llm.exit()

    summary = []
    for batch, context in workloads:
        for attention_backend in attention_backends:
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
            })
    payload = {
        "schema_version": 2,
        "model": args.model,
        "verification": verification,
        "reference_backend": attention_backends[0],
        "records": records,
        "summary": summary,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[p5-ab] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
