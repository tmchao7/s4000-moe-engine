"""Reproducible Qwen3-MoE offline batch benchmark.

Records request latency distributions as well as aggregate output throughput.
The model is loaded once; every measured point gets one unreported warmup and
then ``--repeats`` measured runs.

``--verify-prompts`` re-runs the fixed P2 prompts before any measurement and
compares their token ids against a reference JSON, so a regression aborts the
run before the expensive sweep instead of after it.  ``--cache-modes`` sweeps
prefix-cache cold/hot behaviour inside the same model load.
"""
import argparse
import json
import platform
import random
import statistics
import time
from pathlib import Path

VERIFY_PROMPTS = [
    "简单介绍一下 MoE（Mixture of Experts）架构的核心思想。",
    "What is paged attention in LLM inference? Answer in two sentences.",
    "1 到 100 的整数中所有质数之和是多少？给出答案即可。",
    "请用 Python 写一个快速排序，附注释。",
]


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def prompts(batch, prompt_len, vocab_size, seed, shared_prefix_len=0):
    rng = random.Random(seed)
    shared = [rng.randrange(vocab_size) for _ in range(shared_prefix_len)]
    return [
        shared + [rng.randrange(vocab_size) for _ in range(prompt_len - shared_prefix_len)]
        for _ in range(batch)
    ]


def run_verification(llm, sampling_params_cls, reference_path, max_tokens, repeats, backends, require_deterministic=False):
    """Check greedy reproducibility before spending GPU time on the sweep.

    Design (measured on 30B, see docs/HANDOFF.md):

    * The **first** forward of a backend right after ``set_moe_backend`` may
      pick a different kernel/workspace and can therefore return different
      token ids; every backend gets one discarded warm-up generate first.
    * The hard gate is then *steady-state internal* reproducibility: the
      measured repeats must produce identical token ids.
    * Agreement with the frozen reference JSON, and between backends, is
      recorded but not fatal.  A bf16 near-tie can flip the greedy argmax when
      the reduction order changes (``loop`` accumulates per expert,
      ``grouped_mm`` accumulates with ``index_add_``), so demanding 30B
      cross-backend token identity is not a sound acceptance criterion; the
      tiny float32 four-way check covers numerical equivalence instead.
    """
    reference = None
    if reference_path:
        reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
        reference = [list(row["token_ids"]) for row in reference]
    verification = {
        "prompts": VERIFY_PROMPTS,
        "reference": reference_path,
        "repeats": repeats,
        "backends": {},
    }
    unstable = []
    for backend in backends:
        llm.model_runner.call("set_moe_backend", backend)
        params = sampling_params_cls(temperature=0, max_tokens=max_tokens)
        warmup_outputs = llm.generate(VERIFY_PROMPTS, params, use_tqdm=False)
        warmup = [list(output["token_ids"]) for output in warmup_outputs]
        print(f"[verify] {backend} warmup {warmup}", flush=True)
        for prompt, output in zip(VERIFY_PROMPTS, warmup_outputs):
            print(f"[verify] {backend} {prompt[:20]!r} -> {output['text'][:60]!r}", flush=True)
        runs = []
        for repeat in range(repeats):
            if repeat:
                # different block layout: reallocates KV from the free list
                llm.scheduler.block_manager.clear_prefix_cache()
            outputs = llm.generate(VERIFY_PROMPTS, params, use_tqdm=False)
            token_ids = [list(output["token_ids"]) for output in outputs]
            runs.append(token_ids)
            print(f"[verify] {backend} repeat={repeat} {token_ids}", flush=True)
        entry = {
            "warmup": warmup,
            "runs": runs,
            "deterministic": all(run == runs[0] for run in runs),
            "warmup_matches_steady": warmup == runs[0],
        }
        if reference is not None:
            entry["reference_matches"] = [run == reference for run in runs]
            for index, (actual, want) in enumerate(zip(runs[0], reference)):
                if actual != want:
                    print(
                        f"[verify] REFERENCE_DIFF {backend} prompt={index} "
                        f"actual={actual} expected={want}",
                        flush=True,
                    )
        verification["backends"][backend] = entry
        print(
            f"[verify] {backend} deterministic={entry['deterministic']} "
            f"warmup_matches_steady={entry['warmup_matches_steady']} "
            f"reference_matches={entry.get('reference_matches')}",
            flush=True,
        )
        if not entry["deterministic"]:
            unstable.append(backend)
    if len(backends) > 1:
        baseline = verification["backends"][backends[0]]["runs"][0]
        for backend in backends[1:]:
            other = verification["backends"][backend]["runs"][0]
            agreement = [actual == want for actual, want in zip(baseline, other)]
            verification["backends"][backend]["matches_" + backends[0]] = agreement
            print(f"[verify] {backend} vs {backends[0]} prompt matches={agreement}", flush=True)
    verification["deterministic_all"] = not unstable
    if unstable:
        message = f"steady-state token ids differ between repeats: {unstable}"
        if require_deterministic:
            raise SystemExit(f"[verify] {message}")
        print(f"[verify] WARNING {message} (advisory; pass --verify-require-deterministic to abort)", flush=True)
    return verification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--batches", default="1,2,4,8,16,32,64")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--moe-backends", default="loop,grouped_mm")
    parser.add_argument("--expert-placement", choices=("linear", "round_robin", "profiled"), default="linear")
    parser.add_argument("--expert-map-path")
    parser.add_argument("--scheduler-policy", choices=("fcfs", "stall_free", "stall_free_lpm"), default="fcfs")
    parser.add_argument("--output", default="/root/p2_benchmark.json")
    parser.add_argument("--jsonl")
    parser.add_argument("--shared-prefix-len", type=int, default=0)
    parser.add_argument(
        "--cache-modes", "--cache-mode", dest="cache_modes", default="none",
        help="comma list of none|cold|hot; cold clears prefix hashes before every repeat",
    )
    parser.add_argument("--verify-prompts", action="store_true")
    parser.add_argument("--reference-outputs", default="/root/p2_outputs.json")
    parser.add_argument("--verify-max-tokens", type=int, default=8)
    parser.add_argument("--verify-repeats", type=int, default=3)
    parser.add_argument("--verify-backends", default="loop,grouped_mm")
    parser.add_argument("--verify-require-deterministic", action="store_true")
    args = parser.parse_args()

    cache_modes = [mode.strip() for mode in args.cache_modes.split(",") if mode.strip()]
    assert cache_modes and all(mode in ("none", "cold", "hot") for mode in cache_modes), cache_modes
    assert 0 <= args.shared_prefix_len <= args.prompt_len
    if any(mode != "none" for mode in cache_modes):
        assert args.shared_prefix_len > 0, "cache modes require --shared-prefix-len"

    from nanovllm import LLM, SamplingParams

    batches = [int(value) for value in args.batches.split(",")]
    backends = args.moe_backends.split(",")
    assert backends and all(backend in ("loop", "grouped_mm") for backend in backends)
    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=2,
        expert_parallel_size=2,
        moe_backend=backends[0],
        expert_placement=args.expert_placement,
        expert_map_path=args.expert_map_path,
        scheduler_policy=args.scheduler_policy,
        max_model_len=max(2048, args.prompt_len + args.output_len),
        gpu_memory_utilization=0.85,
    )
    vocab_size = llm.model_runner.config.hf_config.vocab_size
    params = SamplingParams(temperature=0, max_tokens=args.output_len, ignore_eos=True)
    records = []
    verification = None
    try:
        if args.verify_prompts:
            reference_path = args.reference_outputs
            if not Path(reference_path).is_file():
                print(f"[verify] reference {reference_path} missing; recording only", flush=True)
                reference_path = None
            verify_backends = [b.strip() for b in args.verify_backends.split(",") if b.strip()]
            assert verify_backends and all(b in ("loop", "grouped_mm") for b in verify_backends)
            verification = run_verification(
                llm, SamplingParams, reference_path, args.verify_max_tokens,
                args.verify_repeats, verify_backends,
                args.verify_require_deterministic,
            )
        for cache_mode in cache_modes:
            for backend in backends:
                llm.model_runner.call("set_moe_backend", backend)
                for batch in batches:
                    batch_prompts = prompts(
                        batch, args.prompt_len, vocab_size, seed=batch,
                        shared_prefix_len=args.shared_prefix_len,
                    )
                    llm.generate(batch_prompts, params, use_tqdm=False)
                    for repeat in range(args.repeats):
                        if cache_mode == "cold":
                            llm.scheduler.block_manager.clear_prefix_cache()
                        start = time.perf_counter()
                        outputs = llm.generate(batch_prompts, params, use_tqdm=False)
                        elapsed = time.perf_counter() - start
                        ttfts = [output["metrics"]["ttft_s"] for output in outputs]
                        tpots = [
                            interval
                            for output in outputs
                            for interval in output["metrics"]["token_intervals_s"]
                        ]
                        output_tokens = sum(len(output["token_ids"]) for output in outputs)
                        record = {
                            "backend": backend,
                            "batch": batch,
                            "repeat": repeat,
                            "prompt_len": args.prompt_len,
                            "output_len": args.output_len,
                            "shared_prefix_len": args.shared_prefix_len,
                            "cache_mode": cache_mode,
                            "elapsed_s": elapsed,
                            "output_tokens": output_tokens,
                            "output_tokens_per_s": output_tokens / elapsed,
                            "ttft_p50_s": percentile(ttfts, 0.50),
                            "ttft_p95_s": percentile(ttfts, 0.95),
                            "tpot_p50_s": percentile(tpots, 0.50),
                            "tpot_p95_s": percentile(tpots, 0.95),
                        }
                        records.append(record)
                        print("[bench]", json.dumps(record, sort_keys=True), flush=True)
    finally:
        llm.exit()

    summary = []
    for cache_mode in cache_modes:
        for backend in backends:
            for batch in batches:
                rows = [
                    record for record in records
                    if record["cache_mode"] == cache_mode
                    and record["backend"] == backend
                    and record["batch"] == batch
                ]
                throughput = [row["output_tokens_per_s"] for row in rows]
                median = statistics.median(throughput)
                summary.append({
                    "backend": backend,
                    "batch": batch,
                    "cache_mode": cache_mode,
                    "throughput_median": median,
                    "throughput_cv": statistics.pstdev(throughput) / median if len(throughput) > 1 else 0.0,
                    "ttft_p95_median_s": statistics.median(row["ttft_p95_s"] for row in rows),
                    "tpot_p95_median_s": statistics.median(row["tpot_p95_s"] for row in rows),
                })
    payload = {
        "model": args.model,
        "moe_backends": backends,
        "expert_placement": args.expert_placement,
        "scheduler_policy": args.scheduler_policy,
        "shared_prefix_len": args.shared_prefix_len,
        "cache_modes": cache_modes,
        "verification": verification,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "records": records,
        "summary": summary,
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    jsonl_path = Path(args.jsonl) if args.jsonl else output_path.with_suffix(".jsonl")
    jsonl_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"[bench] wrote {output_path}")
    print(f"[bench] wrote {jsonl_path}")


if __name__ == "__main__":
    main()
