"""Per-layer decode timing for a dense model (I19).

A per-step split only says "forward is 98% of the step".  This opens the
forward and asks whether the ~2 ms/layer fixed cost is launch overhead (many
tiny kernels, GPU idle between them) or genuinely slow kernels.

Four passes in one model load (Qwen3-0.6B, TP1, eager, b1):

  baseline  no hooks                      -> decode-step wall median
  async     host wall per layer           -> launch/CPU cost per layer
  sync      synchronize after each layer  -> device span per layer
  profile   one decode step               -> kernel count + device time totals

Classification (written before the run, see docs/PLAN.md B'3):

  device_busy = kernel_device_total / baseline_step_wall
    < 0.70  launch-bound   (more than 30% of the step the GPU has no work)
    >= 0.90 kernel-bound   (the kernels themselves are slow)
    else    mixed

Run:
  PYTHONPATH=/root/moe-server-s4000 python3 tests/p6_layer_timing.py \
      --model /root/autodl-tmp/models/Qwen3-0.6B \
      --output /root/p6_layer_timing_06.json
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
from nanovllm.benchmarking import percentile  # noqa: E402
from nanovllm.utils.compat import accelerator  # noqa: E402

SEED = 20261027


def install_hooks(layers, state):
    """Time every decoder layer; ``state`` gates collection and sync mode."""
    starts = [0.0] * len(layers)
    per_layer = [[] for _ in layers]

    def pre_hook(index):
        def hook(module, args):
            starts[index] = time.perf_counter()
        return hook

    def post_hook(index):
        def hook(module, args, output):
            if not state["collect"]:
                return
            if state["sync"]:
                accelerator().synchronize()
            per_layer[index].append(time.perf_counter() - starts[index])
        return hook

    handles = []
    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(pre_hook(index)))
        handles.append(layer.register_forward_hook(post_hook(index)))
    return per_layer, handles


def drive_decode(llm, prompt, max_tokens, state):
    """Run one request to completion and return the measured decode-step walls.

    Hooks collect only from the second decode step onward: the first one pays
    for JIT/allocator warmup and would pollute the per-layer medians.
    """
    llm.add_request(
        prompt,
        SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True),
    )
    first_decode_done = False
    step_walls = []
    while not llm.is_finished():
        state["collect"] = first_decode_done
        started = time.perf_counter()
        llm.step()
        wall = time.perf_counter() - started
        if llm.last_step_metrics["mode"] == "decode":
            if first_decode_done:
                step_walls.append(wall)
            first_decode_done = True
    state["collect"] = False
    return step_walls


def profile_decode_step(llm, prompt):
    """Kernel count and device-time totals for one decode step."""
    llm.add_request(
        prompt,
        SamplingParams(temperature=0, max_tokens=4, ignore_eos=True),
    )
    while not llm.is_finished():
        llm.step()
        if llm.last_step_metrics["mode"] == "decode":
            break
    accelerator().synchronize()
    activities = [torch.profiler.ProfilerActivity.CPU]
    device_activity = getattr(torch.profiler.ProfilerActivity, "CUDA", None)
    if device_activity is not None:
        activities.append(device_activity)
    started = time.perf_counter()
    try:
        with torch.profiler.profile(activities=activities) as prof:
            llm.step()
        accelerator().synchronize()
        wall = time.perf_counter() - started
        events = prof.key_averages()
        kernels = [
            event for event in events
            if getattr(event, "self_device_time_total", 0) > 0
        ]
        top = sorted(
            kernels, key=lambda event: -getattr(event, "self_device_time_total", 0)
        )[:15]
        launch_count = sum(event.count for event in kernels)
        record = {
            "step_wall_s": wall,
            "kernel_types": len(kernels),
            "launch_count": launch_count,
            "kernel_total_device_us": sum(
                getattr(event, "self_device_time_total", 0) for event in kernels
            ),
            "cpu_total_us": sum(
                getattr(event, "self_cpu_time_total", 0) for event in kernels
            ),
            "top_kernels": [
                {
                    "key": event.key,
                    "count": event.count,
                    "device_us": round(getattr(event, "self_device_time_total", 0), 1),
                    "cpu_us": round(getattr(event, "self_cpu_time_total", 0), 1),
                }
                for event in top
            ],
        }
    except Exception as exc:  # noqa: BLE001 - profiler support is optional
        record = {"profiler_error": f"{type(exc).__name__}: {exc}"}
    while not llm.is_finished():
        llm.step()
    return record


def classify(profile, step_wall_s):
    kernel_s = (profile.get("kernel_total_device_us") or 0) / 1e6
    if not step_wall_s or not kernel_s:
        return None, None
    ratio = kernel_s / step_wall_s
    if ratio < 0.70:
        verdict = "launch_bound"
    elif ratio >= 0.90:
        verdict = "kernel_bound"
    else:
        verdict = "mixed"
    return ratio, verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--output", default="/root/p6_layer_timing_06.json")
    args = parser.parse_args()

    started_at = time.time()
    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
        attention_backend="sdpa",
        scheduler_policy="fcfs",
        max_num_seqs=1,
        max_model_len=max(512, args.prompt_len + args.max_tokens + 16),
        gpu_memory_utilization=0.35,
    )
    runner = llm.model_runner
    model = runner.model
    layers = getattr(getattr(model, "model", None), "layers", None)
    assert layers, "could not find decoder layers on the loaded model"
    vocab = runner.config.hf_config.vocab_size
    rng = random.Random(SEED)
    prompt = [rng.randrange(vocab) for _ in range(args.prompt_len)]

    state = {"collect": False, "sync": False}
    per_layer, handles = install_hooks(layers, state)
    payload = {
        "schema_version": 1,
        "kind": "layer_timing",
        "model": args.model,
        "tp": args.tp,
        "attention_backend": "sdpa",
        "enforce_eager": True,
        "prompt_len": args.prompt_len,
        "max_tokens": args.max_tokens,
        "num_layers": len(layers),
    }
    try:
        # warmup: JIT, allocator, prefix-cache and scheduler paths
        state["collect"] = False
        drive_decode(llm, prompt, 8, state)

        baseline = drive_decode(llm, prompt, args.max_tokens, state)
        payload["baseline_step_wall_median_s"] = statistics.median(baseline)
        payload["baseline_step_wall_p95_s"] = percentile(baseline, 0.95)
        payload["baseline_steps"] = len(baseline)

        state["sync"] = False
        async_walls = drive_decode(llm, prompt, args.max_tokens, state)
        async_layer = [statistics.median(values) if values else None for values in per_layer]
        for values in per_layer:
            values.clear()

        state["sync"] = True
        sync_walls = drive_decode(llm, prompt, args.max_tokens, state)
        sync_layer = [statistics.median(values) if values else None for values in per_layer]
        for values in per_layer:
            values.clear()
        state["sync"] = False

        payload["async_step_wall_median_s"] = statistics.median(async_walls)
        payload["sync_step_wall_median_s"] = statistics.median(sync_walls)
        payload["per_layer"] = [
            {
                "layer": index,
                "host_median_s": async_layer[index],
                "device_span_median_s": sync_layer[index],
            }
            for index in range(len(layers))
        ]
        payload["host_sum_s"] = sum(value for value in async_layer if value)
        payload["device_span_sum_s"] = sum(value for value in sync_layer if value)
        payload["host_median_per_layer_s"] = statistics.median(
            [value for value in async_layer if value]
        )
        payload["device_span_median_per_layer_s"] = statistics.median(
            [value for value in sync_layer if value]
        )

        profile = profile_decode_step(llm, prompt)
        payload["profile"] = profile
        ratio, verdict = classify(profile, payload["baseline_step_wall_median_s"])
        payload["device_busy_ratio"] = ratio
        payload["classification"] = verdict
        if payload["baseline_step_wall_median_s"] and profile.get("launch_count"):
            payload["us_per_launch"] = (
                profile["kernel_total_device_us"] / profile["launch_count"]
            )
    finally:
        for handle in handles:
            handle.remove()
        llm.exit()

    payload["wall_seconds"] = round(time.time() - started_at, 1)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[layer_timing] {json.dumps({k: v for k, v in payload.items() if k != 'per_layer'}, sort_keys=True)}", flush=True)
    print(f"[layer_timing] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
