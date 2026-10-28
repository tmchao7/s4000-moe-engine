"""Host-sync removal A/B (P1.3): sampler all_greedy + prefill cu_seqlens lists.

Gates (frozen in docs/NIGHTLY-LOG.md "P1.3 host-sync"):
  T1  token identity: greedy b1 and b32 streams identical between fix on/off
  T2  b32-mixed sample segment: median fix-on <= 0.5 x median fix-off (the 2x
      bool() device reads are gone), CV <= 5% both groups
  T3  prefill-512 step wall: fix-on median < fix-off median (96 .tolist() gone;
      report the delta, informational on 0.6B -- the -50% gate is a 30B gate)
  T4  b1/b32-greedy step wall: report delta (informational)

Run:
  PYTHONPATH=/root/moe-server-s4000 python3 tests/p8_sync_ab.py \
      --model /root/autodl-tmp/models/Qwen3-0.6B --output /root/p8_sync_fix.json
"""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/root/moe-server-s4000")

from nanovllm import LLM, SamplingParams  # noqa: E402
from nanovllm.benchmarking import coefficient_of_variation  # noqa: E402
from nanovllm.engine.model_runner import set_host_sync_fix  # noqa: E402
from nanovllm.utils.context import reset_context  # noqa: E402

SEED = 20261027
TRIALS = 5


def timed_step(llm):
    """One engine step with per-segment timing (mirrors LLMEngine.step)."""
    started = time.perf_counter()
    batch = llm.scheduler.schedule()
    seqs, mode = batch
    prepared = time.perf_counter()
    runner = llm.model_runner
    input_ids, positions = (
        runner.prepare_decode(seqs) if mode == "decode" else runner.prepare_prefill(seqs)
    )
    temperatures, all_greedy = runner.prepare_sample(seqs) if runner.rank == 0 else (None, True)
    sample_prepared = time.perf_counter()
    logits = runner.run_model(input_ids, positions, mode)
    forwarded = time.perf_counter()
    token_ids = runner.sampler(logits, temperatures, all_greedy).tolist() if runner.rank == 0 else None
    sampled = time.perf_counter()
    llm.scheduler.postprocess(seqs, token_ids, mode)
    reset_context()
    wall = time.perf_counter() - started
    return mode, {
        "schedule_s": prepared - started,
        "prepare_s": sample_prepared - prepared,
        "forward_s": forwarded - sample_prepared,
        "sample_s": sampled - forwarded,
        "wall_s": wall,
    }, seqs


def drive_b1(llm, prompt, max_tokens):
    """One request; return (tokens, decode step segments)."""
    llm.add_request(prompt, SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True))
    tokens = []
    segments = []
    first = False
    while not llm.is_finished():
        mode, seg, seqs = timed_step(llm)
        if mode == "decode":
            if first:
                segments.append(seg)
            first = True
        for seq in seqs:
            if seq.is_finished:
                tokens = list(seq.completion_token_ids)
    return tokens, segments


def drive_b32(llm, prompts, max_tokens, temperatures):
    """32 concurrent requests; return (tokens_by_request, decode step segments)."""
    for index, prompt in enumerate(prompts):
        llm.add_request(
            prompt,
            SamplingParams(temperature=temperatures[index], max_tokens=max_tokens, ignore_eos=True),
        )
    tokens = {}
    segments = []
    first = False
    while not llm.is_finished():
        mode, seg, seqs = timed_step(llm)
        if mode == "decode":
            if first:
                segments.append(seg)
            first = True
        for seq in seqs:
            if seq.is_finished:
                tokens[seq.seq_id] = list(seq.completion_token_ids)
    return tokens, segments


def drive_prefill(llm, prompt):
    """Single 512-token request; return the prefill-step segment."""
    llm.add_request(prompt, SamplingParams(temperature=0, max_tokens=1, ignore_eos=True))
    while not llm.is_finished():
        mode, seg, _seqs = timed_step(llm)
        if mode == "prefill":
            return seg
    return None


def summarize(segments):
    return {
        "median_wall_ms": statistics.median(s["wall_s"] for s in segments) * 1000,
        "median_sample_ms": statistics.median(s["sample_s"] for s in segments) * 1000,
        "median_forward_ms": statistics.median(s["forward_s"] for s in segments) * 1000,
    }


def interleave(llm, label, build, trials=TRIALS):
    """build(pair_index, fix_on) -> payload; alternates off/on within a pair.

    Both runs of a pair use the SAME inputs (build receives pair_index), so
    token identity compares like with like.
    """
    results = {"off": [], "on": []}
    tokens = {}
    for trial in range(trials * 2):
        fix_on = trial % 2 == 1
        set_host_sync_fix(fix_on)
        key = "on" if fix_on else "off"
        pair_index = trial // 2
        payload = build(pair_index, fix_on)
        results[key].append(payload["segments"])
        if payload.get("tokens") is not None:
            tokens.setdefault(key, []).append(payload["tokens"])
        print(f"[sync_ab] {label} trial {trial} fix={'on' if fix_on else 'off'} "
              f"wall={payload['segments'] and statistics.median([s['wall_s'] for s in payload['segments']]) * 1000:.2f}ms",
              flush=True)
    set_host_sync_fix(True)
    summary = {}
    for mode in ("off", "on"):
        flat = [seg for group in results[mode] for seg in group]
        summary[mode] = summarize(flat)
        summary[mode]["cv_wall"] = coefficient_of_variation([
            statistics.median([s["wall_s"] for s in group]) for group in results[mode]
        ])
    identical = None
    mismatches = []
    if tokens.get("off") and tokens.get("on"):
        # b1 runs are lists; b32 runs are dicts keyed by seq_id, and seq_ids
        # keep incrementing across drives -- compare values in sorted-key order.
        off_runs = [
            [row for _key, row in sorted(run.items())] if isinstance(run, dict) else list(run)
            for run in tokens["off"]
        ]
        on_runs = [
            [row for _key, row in sorted(run.items())] if isinstance(run, dict) else list(run)
            for run in tokens["on"]
        ]
        identical = all(a == b for a, b in zip(off_runs, on_runs))
        for index, (a, b) in enumerate(zip(off_runs, on_runs)):
            if a != b:
                mismatches.append({
                    "pair": index,
                    "off_first_diff": next(
                        (i for i, (row_x, row_y) in enumerate(zip(a, b)) if row_x != row_y), None
                    ),
                    "off_len": len(a),
                    "on_len": len(b),
                })
    return {"label": label, "summary": summary, "tokens_identical": identical,
            "token_mismatches": mismatches[:4]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    parser.add_argument("--output", default="/root/p8_sync_fix.json")
    args = parser.parse_args()

    rng = random.Random(SEED)
    vocab = 151936  # refined after load; prompts are built post-load
    payload = {
        "schema_version": 1,
        "kind": "sync_ab",
        "model": args.model,
        "gates_frozen": "docs/NIGHTLY-LOG.md P1.3 (T1-T4)",
    }
    started_at = time.time()

    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        attention_backend="sdpa",
        scheduler_policy="fcfs",
        max_num_seqs=32,
        max_num_batched_tokens=2048,
        max_model_len=2048,
        gpu_memory_utilization=0.35,
    )
    vocab = llm.model_runner.config.hf_config.vocab_size

    def rand_prompt(length, salt):
        generator = random.Random((SEED, salt))
        return [generator.randrange(vocab) for _ in range(length)]

    set_host_sync_fix(True)
    drive_b1(llm, rand_prompt(128, "w1"), 8)          # warmup b1 shapes
    drive_b32(llm, [rand_prompt(16, ("w32", i)) for i in range(32)], 8, [0.0] * 32)
    drive_prefill(llm, rand_prompt(512, "wp"))

    # T1/T4a: b1 greedy decode
    payload["b1"] = interleave(
        llm, "b1_decode",
        lambda pair_index, _fix_on: (lambda out: {"segments": out[1], "tokens": out[0]})(
            drive_b1(llm, rand_prompt(128, ("b1", pair_index)), 32)
        ),
    )

    # T1/T4b: b32 greedy decode
    payload["b32_greedy"] = interleave(
        llm, "b32_decode_greedy",
        lambda pair_index, _fix_on: (lambda out: {"segments": out[1], "tokens": out[0]})(
            drive_b32(llm, [rand_prompt(16, ("g32", pair_index, i)) for i in range(32)],
                      16, [0.0] * 32)
        ),
    )

    # T2: b32 mixed temperature (2 bool() syncs on the off path)
    mixed_temps = [0.0 if index % 2 == 0 else 1.0 for index in range(32)]
    payload["b32_mixed"] = interleave(
        llm, "b32_decode_mixed",
        lambda pair_index, _fix_on: (lambda out: {"segments": out[1], "tokens": None})(
            drive_b32(llm, [rand_prompt(16, ("m32", pair_index, i)) for i in range(32)],
                      16, mixed_temps)
        ),
    )

    # T3: prefill 512 (96 .tolist() on the off path)
    prefill_segments = {"off": [], "on": []}
    for trial in range(TRIALS * 2):
        fix_on = trial % 2 == 1
        set_host_sync_fix(fix_on)
        seg = drive_prefill(llm, rand_prompt(512, ("pf", trial // 2)))
        prefill_segments["on" if fix_on else "off"].append(seg["wall_s"] * 1000)
    set_host_sync_fix(True)
    payload["prefill_512"] = {
        "off_wall_ms": prefill_segments["off"],
        "on_wall_ms": prefill_segments["on"],
        "off_median_ms": statistics.median(prefill_segments["off"]),
        "on_median_ms": statistics.median(prefill_segments["on"]),
    }

    llm.exit()

    # Verdicts
    b32m = payload["b32_mixed"]["summary"]
    t1 = (
        payload["b1"]["tokens_identical"] is True
        and payload["b32_greedy"]["tokens_identical"] is True
    )
    t2 = (
        b32m["on"]["median_sample_ms"] <= 0.5 * b32m["off"]["median_sample_ms"]
        and b32m["off"]["cv_wall"] <= 0.05 and b32m["on"]["cv_wall"] <= 0.05
    )
    t3 = payload["prefill_512"]["on_median_ms"] < payload["prefill_512"]["off_median_ms"]
    payload["verdict"] = {
        "T1_token_identity": "PASS" if t1 else "FAIL",
        "T2_b32_mixed_sample": "PASS" if t2 else "FAIL",
        "T3_prefill_wall": "PASS" if t3 else "FAIL",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[sync_ab] verdict={json.dumps(payload['verdict'])}", flush=True)
    print(f"[sync_ab] wrote {out} wall={time.time() - started_at:.0f}s", flush=True)


if __name__ == "__main__":
    main()
