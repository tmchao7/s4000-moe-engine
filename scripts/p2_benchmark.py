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
import hashlib
import json
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

from nanovllm.benchmarking import coefficient_of_variation, percentile

REPO_ROOT = Path(__file__).resolve().parents[1]

VERIFY_PROMPTS = [
    "简单介绍一下 MoE（Mixture of Experts）架构的核心思想。",
    "What is paged attention in LLM inference? Answer in two sentences.",
    "1 到 100 的整数中所有质数之和是多少？给出答案即可。",
    "请用 Python 写一个快速排序，附注释。",
]


def prompts(batch, prompt_len, vocab_size, seed, shared_prefix_len=0):
    rng = random.Random(seed)
    shared = [rng.randrange(vocab_size) for _ in range(shared_prefix_len)]
    return [
        shared + [rng.randrange(vocab_size) for _ in range(prompt_len - shared_prefix_len)]
        for _ in range(batch)
    ]


def run_verification(llm, sampling_params_cls, reference_path, max_tokens, repeats, backends,
                     decode_backend=None, soft=False):
    """Check greedy reproducibility before spending GPU time on the sweep.

    Design (measured on 30B, see docs/HANDOFF.md):

    * The **first** forward of a backend right after ``set_moe_backend`` may
      pick a different kernel/workspace and can therefore return different
      token ids; every backend gets one discarded warm-up generate first.
    * Steady-state reproducibility, the committed reference, and cross-backend
      token identity are hard gates.  A performance run with different output
      is not comparable to the frozen correct baseline.
    * ``decode_backend`` verifies a decode-only backend switch (e.g. bmm_dense).
      Its accumulation order differs, so the frozen reference and cross-backend
      identity become **soft** (reported, not enforced) unless ``soft=False``;
      steady-state self-determinism stays a hard gate in both modes.
    """
    reference = None
    if reference_path:
        reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
        reference = [list(row["token_ids"]) for row in reference]
    verification = {
        "prompts": VERIFY_PROMPTS,
        "reference": reference_path,
        "repeats": repeats,
        "decode_backend": decode_backend,
        "soft": soft,
        "backends": {},
    }
    failures = []
    for backend in backends:
        llm.model_runner.call("set_moe_backend", backend, decode_backend)
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
            if not all(entry["reference_matches"]) and not soft:
                failures.append(f"{backend}: committed reference mismatch")
        verification["backends"][backend] = entry
        print(
            f"[verify] {backend} deterministic={entry['deterministic']} "
            f"warmup_matches_steady={entry['warmup_matches_steady']} "
            f"reference_matches={entry.get('reference_matches')}",
            flush=True,
        )
        if not entry["deterministic"]:
            failures.append(f"{backend}: repeat nondeterminism")
    if len(backends) > 1:
        baseline = verification["backends"][backends[0]]["runs"][0]
        for backend in backends[1:]:
            other = verification["backends"][backend]["runs"][0]
            agreement = [actual == want for actual, want in zip(baseline, other)]
            verification["backends"][backend]["matches_" + backends[0]] = agreement
            print(f"[verify] {backend} vs {backends[0]} prompt matches={agreement}", flush=True)
            if not all(agreement):
                failures.append(f"{backend}: differs from {backends[0]}")
    verification["deterministic_all"] = all(
        entry["deterministic"] for entry in verification["backends"].values()
    )
    verification["passed"] = not failures
    verification["failures"] = failures
    if failures:
        raise SystemExit("[verify] hard gate failed: " + "; ".join(failures))
    return verification


def file_sha256(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--batches", default="1,2,4,8,16,32,64")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--moe-backends", default="loop,grouped_mm")
    parser.add_argument(
        "--decode-moe-backends", default="",
        help="comma list of decode-only MoE backends (e.g. grouped_mm,bmm_dense); "
        "empty means decode uses the same backend as prefill",
    )
    parser.add_argument("--attention-backends", default="sdpa", help="comma list of sdpa|triton_paged")
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
    parser.add_argument("--verify-prompts", action="store_true", help="compatibility no-op; verification is on by default")
    parser.add_argument("--skip-verification", action="store_true", help="explicitly bypass the correctness gate")
    parser.add_argument("--reference-outputs", default=str(REPO_ROOT / "benchmarks" / "p2-reference-8tok.json"))
    parser.add_argument("--verify-max-tokens", type=int, default=8)
    parser.add_argument("--verify-repeats", type=int, default=3)
    parser.add_argument("--verify-backends", default="loop,grouped_mm")
    parser.add_argument(
        "--probe-batches", default="",
        help="optional extra phase in the same model load, e.g. 8 for the prefix-cache probe",
    )
    parser.add_argument("--probe-prompt-len", type=int, default=512)
    parser.add_argument("--probe-shared-prefix-len", type=int, default=384)
    parser.add_argument("--probe-cache-modes", default="cold,hot")
    args = parser.parse_args()

    cache_modes = [mode.strip() for mode in args.cache_modes.split(",") if mode.strip()]
    assert cache_modes and all(mode in ("none", "cold", "hot") for mode in cache_modes), cache_modes
    assert 0 <= args.shared_prefix_len <= args.prompt_len
    if any(mode != "none" for mode in cache_modes):
        assert args.shared_prefix_len > 0, "cache modes require --shared-prefix-len"

    probe_batches = [int(value) for value in args.probe_batches.split(",") if value.strip()]
    probe_cache_modes = [mode.strip() for mode in args.probe_cache_modes.split(",") if mode.strip()]
    if probe_batches:
        assert probe_cache_modes and all(
            mode in ("none", "cold", "hot") for mode in probe_cache_modes
        ), probe_cache_modes
        assert 0 <= args.probe_shared_prefix_len <= args.probe_prompt_len
        if any(mode != "none" for mode in probe_cache_modes):
            # block-hash prefix caching only reuses full 256-token blocks
            assert args.probe_shared_prefix_len >= 256, (
                "prefix cache probe needs a shared prefix of at least one 256-token block"
            )

    from nanovllm import LLM, SamplingParams

    batches = [int(value) for value in args.batches.split(",")]
    backends = args.moe_backends.split(",")
    assert backends and all(backend in ("loop", "grouped_mm", "fused_grouped_mm", "vllm_fused") for backend in backends)
    decode_moe_backends = [value.strip() for value in args.decode_moe_backends.split(",") if value.strip()]
    if decode_moe_backends:
        assert all(
            value in ("loop", "grouped_mm", "fused_grouped_mm", "bmm_dense", "vllm_fused")
            for value in decode_moe_backends
        ), decode_moe_backends
    else:
        decode_moe_backends = [None]
    attention_backends = [value.strip() for value in args.attention_backends.split(",") if value.strip()]
    assert attention_backends and all(value in ("sdpa", "triton_paged") for value in attention_backends)
    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=2,
        expert_parallel_size=2,
        moe_backend=backends[0],
        attention_backend=attention_backends[0],
        expert_placement=args.expert_placement,
        expert_map_path=args.expert_map_path,
        scheduler_policy=args.scheduler_policy,
        max_model_len=max(
            2048,
            args.prompt_len + args.output_len,
            args.probe_prompt_len + args.output_len if probe_batches else 0,
        ),
        gpu_memory_utilization=0.85,
    )
    vocab_size = llm.model_runner.config.hf_config.vocab_size
    params = SamplingParams(temperature=0, max_tokens=args.output_len, ignore_eos=True)
    records = []
    verification = None
    # Stream every measurement to disk as it completes: a crash mid-matrix (e.g.
    # a MUSA device error at the last batch) must not lose the finished points.
    jsonl_path = Path(args.jsonl) if args.jsonl else Path(args.output).with_suffix(".jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_handle = jsonl_path.open("w", encoding="utf-8")
    try:
        if not args.skip_verification:
            reference_path = args.reference_outputs
            if not Path(reference_path).is_file():
                raise SystemExit(f"[verify] committed reference missing: {reference_path}")
            verify_backends = [b.strip() for b in args.verify_backends.split(",") if b.strip()]
            assert verify_backends and all(
                b in ("loop", "grouped_mm", "fused_grouped_mm", "vllm_fused")
                for b in verify_backends
            )
            verification = run_verification(
                llm, SamplingParams, reference_path, args.verify_max_tokens,
                args.verify_repeats, verify_backends,
            )
            # Soft gates for the decode-only stacks under test: their bf16
            # accumulation order differs from the committed reference, so
            # reference identity is reported but only self-determinism is
            # enforced (see run_verification).
            verification["decode_stacks"] = {}
            for decode_backend in decode_moe_backends:
                if decode_backend is None:
                    continue
                verification["decode_stacks"][decode_backend] = run_verification(
                    llm, SamplingParams, reference_path, args.verify_max_tokens,
                    args.verify_repeats, ["grouped_mm"],
                    decode_backend=decode_backend, soft=True,
                )
        def measure(attention_backend, backend, batch, prompt_len, shared_prefix_len, cache_mode, phase,
                    decode_moe_backend=None):
            llm.model_runner.call("set_attention_backend", attention_backend)
            llm.model_runner.call("set_moe_backend", backend, decode_moe_backend)
            batch_prompts = prompts(
                batch, prompt_len, vocab_size, seed=batch,
                shared_prefix_len=shared_prefix_len,
            )
            llm.generate(batch_prompts, params, use_tqdm=False)
            for repeat in range(args.repeats):
                if cache_mode == "cold":
                    llm.scheduler.block_manager.clear_prefix_cache()
                start = time.perf_counter()
                outputs = llm.generate(batch_prompts, params, use_tqdm=False)
                elapsed = time.perf_counter() - start
                ttfts = [output["metrics"]["ttft_s"] for output in outputs]
                itls = [
                    interval
                    for output in outputs
                    for interval in output["metrics"]["itls_s"]
                ]
                tpots = [output["metrics"]["tpot_s"] for output in outputs]
                output_tokens = sum(len(output["token_ids"]) for output in outputs)
                input_tokens = sum(output["metrics"]["prompt_tokens"] for output in outputs)
                record = {
                    "phase": phase,
                    "attention_backend": attention_backend,
                    "backend": backend,
                    "decode_moe_backend": decode_moe_backend,
                    "batch": batch,
                    "repeat": repeat,
                    "prompt_len": prompt_len,
                    "output_len": args.output_len,
                    "shared_prefix_len": shared_prefix_len,
                    "cache_mode": cache_mode,
                    "elapsed_s": elapsed,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "input_tokens_per_s": input_tokens / elapsed,
                    "output_tokens_per_s": output_tokens / elapsed,
                    "total_tokens_per_s": (input_tokens + output_tokens) / elapsed,
                    "ttft_p50_s": percentile(ttfts, 0.50),
                    "ttft_p95_s": percentile(ttfts, 0.95),
                    "tpot_p50_s": percentile(tpots, 0.50),
                    "tpot_p95_s": percentile(tpots, 0.95),
                    "itl_p50_s": percentile(itls, 0.50),
                    "itl_p95_s": percentile(itls, 0.95),
                    "requests": [
                        {
                            **output["metrics"],
                            "token_ids": output["token_ids"],
                        }
                        for output in outputs
                    ],
                }
                records.append(record)
                jsonl_handle.write(json.dumps(record, sort_keys=True) + "\n")
                jsonl_handle.flush()
                printable = {key: value for key, value in record.items() if key != "requests"}
                print("[bench]", json.dumps(printable, sort_keys=True), flush=True)

        for attention_backend in attention_backends:
            for cache_mode in cache_modes:
                for backend in backends:
                    for decode_moe_backend in decode_moe_backends:
                        for batch in batches:
                            measure(
                                attention_backend, backend, batch,
                                args.prompt_len, args.shared_prefix_len, cache_mode, "matrix",
                                decode_moe_backend=decode_moe_backend,
                            )

        # Extra phase in the same model load: prefix-cache cold/hot probe with a
        # long shared prefix (>= 256 tokens, otherwise no full block is reused).
        if probe_batches:
            for attention_backend in attention_backends:
                for backend in backends:
                    for decode_moe_backend in decode_moe_backends:
                        for batch in probe_batches:
                            for cache_mode in probe_cache_modes:
                                measure(
                                    attention_backend, backend, batch,
                                    args.probe_prompt_len, args.probe_shared_prefix_len,
                                    cache_mode, "probe",
                                    decode_moe_backend=decode_moe_backend,
                                )
    finally:
        jsonl_handle.close()
        llm.exit()

    summary = []
    for phase in ("matrix", "probe"):
        for attention_backend in attention_backends:
            for cache_mode in cache_modes if phase == "matrix" else probe_cache_modes:
                for backend in backends:
                    for decode_moe_backend in decode_moe_backends:
                        for batch in batches if phase == "matrix" else probe_batches:
                            rows = [
                                record for record in records
                                if record["phase"] == phase
                                and record["attention_backend"] == attention_backend
                                and record["cache_mode"] == cache_mode
                                and record["backend"] == backend
                                and record["decode_moe_backend"] == decode_moe_backend
                                and record["batch"] == batch
                            ]
                            if not rows:
                                continue
                            throughput = [row["output_tokens_per_s"] for row in rows]
                            median = statistics.median(throughput)
                            summary.append({
                                "phase": phase,
                                "attention_backend": attention_backend,
                                "backend": backend,
                                "decode_moe_backend": decode_moe_backend,
                                "batch": batch,
                                "cache_mode": cache_mode,
                                "prompt_len": rows[0]["prompt_len"],
                                "shared_prefix_len": rows[0]["shared_prefix_len"],
                                "throughput_median": median,
                                "throughput_cv": coefficient_of_variation(throughput),
                                "ttft_p95_median_s": statistics.median(row["ttft_p95_s"] for row in rows),
                                "tpot_p95_median_s": statistics.median(row["tpot_p95_s"] for row in rows),
                                "itl_p95_median_s": statistics.median(row["itl_p95_s"] for row in rows),
                            })
    payload = {
        "schema_version": 2,
        "model": args.model,
        "moe_backends": backends,
        "decode_moe_backends": decode_moe_backends,
        "attention_backends": attention_backends,
        "expert_placement": args.expert_placement,
        "scheduler_policy": args.scheduler_policy,
        "shared_prefix_len": args.shared_prefix_len,
        "cache_modes": cache_modes,
        "verification": verification,
        "provenance": {
            "git_commit": git_commit(),
            "model_config_sha256": file_sha256(Path(args.model) / "config.json"),
            "tokenizer_config_sha256": file_sha256(Path(args.model) / "tokenizer_config.json"),
            "reference_sha256": file_sha256(args.reference_outputs),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "records": records,
        "summary": summary,
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[bench] wrote {output_path}")
    print(f"[bench] wrote {jsonl_path} (streamed during the run)")


if __name__ == "__main__":
    main()
