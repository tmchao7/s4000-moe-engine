"""Where does one decode step's time go? (P6 phase 1.0)

The offline matrix says batch 1 costs 0.323 s per decode step while the 30B
model's GPU work at batch 1 is only a few milliseconds (8 experts per layer out
of 128, ~1.8 GiB of expert weights).  Every batch from 1 to 16 sits on a
0.32-0.45 s floor, which points at per-step overhead rather than math.  Graph
capture only pays if that floor is launch/CPU-graph overhead, so measure it
before spending a 30B load:

  * build the dense Qwen3-8B (single card, ~1 min load) with graphs captured;
  * run the same workload twice in one process: eager forward, then graph replay;
  * split every step into schedule / prepare / forward / sample;
  * optionally profile one step and report kernel count and CPU-vs-device time;
  * measure the cost of the per-step ``pin_memory=True`` tensor construction.

The 8B dense model has no MoE, so it is fully capturable with ``triton_paged``
attention.  Its absolute numbers differ from the 30B, but the *overhead
structure* (Python + launch per layer) is the same engine code.

Run:
  PYTHONPATH=. python3 tests/p6_step_overhead.py \
      --batches 1,8,32 --output /root/p6_step_overhead.json
"""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nanovllm.benchmarking import percentile  # noqa: E402
from nanovllm.utils.context import reset_context  # noqa: E402

SEED = 20261026


def make_prompts(batch, length, vocab, rng):
    return [[rng.randrange(vocab) for _ in range(length)] for _ in range(batch)]


def install_timed_run(runner):
    """Replace ModelRunner.run with the same body plus per-phase timers.

    ``run`` is the only place that knows the prepare/forward/sample split; the
    engine's own metrics collapse all three into ``forward_s``.
    """
    phases = {"prepare": [], "forward": [], "sample": []}

    @torch.inference_mode()
    def timed_run(seqs, mode):
        started = time.perf_counter()
        input_ids, positions = (
            runner.prepare_decode(seqs) if mode == "decode" else runner.prepare_prefill(seqs)
        )
        temperatures, all_greedy = runner.prepare_sample(seqs) if runner.rank == 0 else (None, True)
        prepared = time.perf_counter()
        logits = runner.run_model(input_ids, positions, mode)
        forwarded = time.perf_counter()
        token_ids = (
            runner.sampler(logits, temperatures, all_greedy).tolist() if runner.rank == 0 else None
        )
        sampled = time.perf_counter()
        reset_context()
        phases["prepare"].append(prepared - started)
        phases["forward"].append(forwarded - prepared)
        phases["sample"].append(sampled - forwarded)
        return token_ids

    runner.run = timed_run
    return phases


def run_workload(llm, phases, args, batch, rng, vocab, label):
    """Add ``batch`` requests and drive steps manually, recording per-step time."""
    from nanovllm import SamplingParams

    llm.scheduler.block_manager.clear_prefix_cache()
    prompts = make_prompts(batch, args.prompt_len, vocab, rng)
    params = SamplingParams(
        temperature=0, max_tokens=args.max_tokens, ignore_eos=True
    )
    for prompt in prompts:
        llm.add_request(prompt, params)

    steps = []
    while not llm.is_finished():
        started = time.perf_counter()
        llm.step()
        elapsed = time.perf_counter() - started
        steps.append(
            {
                "wall_s": elapsed,
                "mode": llm.last_step_metrics["mode"],
                "schedule_s": llm.last_step_metrics["schedule_s"],
                "forward_s": llm.last_step_metrics["forward_s"],
            }
        )

    decode_steps = [step for step in steps if step["mode"] == "decode"]
    prefill_steps = [step for step in steps if step["mode"] != "decode"]
    decode_wall = [step["wall_s"] for step in decode_steps]
    record = {
        "label": label,
        "batch": batch,
        "steps": len(steps),
        "prefill_steps": len(prefill_steps),
        "decode_steps": len(decode_steps),
        "decode_tokens": sum(1 for step in decode_steps),
        "decode_step_wall_median_s": statistics.median(decode_wall) if decode_wall else None,
        "decode_step_wall_p95_s": percentile(decode_wall, 0.95) if decode_wall else None,
        "decode_step_forward_median_s": (
            statistics.median([step["forward_s"] for step in decode_steps])
            if decode_steps else None
        ),
        "decode_step_schedule_median_s": (
            statistics.median([step["schedule_s"] for step in decode_steps])
            if decode_steps else None
        ),
        "prefill_wall_s": sum(step["wall_s"] for step in prefill_steps),
        "last_step_metrics": llm.last_step_metrics,
    }
    if decode_wall:
        record["decode_tokens_per_s"] = batch / statistics.median(decode_wall)
    for phase in ("prepare", "forward", "sample"):
        values = phases[phase]
        record[f"{phase}_median_s"] = statistics.median(values) if values else None
    record["phase_share"] = {
        phase: (
            record[f"{phase}_median_s"] / record["decode_step_wall_median_s"]
            if record[f"{phase}_median_s"] and record["decode_step_wall_median_s"]
            else None
        )
        for phase in ("prepare", "forward", "sample")
    }
    print(f"[overhead] {json.dumps(record, sort_keys=True)}", flush=True)
    return record


def pinned_buffer_cost():
    """How much of a step is rebuilding pinned host -> device tensors?"""
    device = "musa"
    sizes = {
        "input_ids": ([1], torch.int64),
        "positions": ([1], torch.int64),
        "slot_mapping": ([32], torch.int32),
        "context_lens": ([32], torch.int32),
        "block_tables": ([32, 8], torch.int32),
    }
    rows = {}
    for name, (shape, dtype) in sizes.items():
        values = [[7] * shape[1] for _ in range(shape[0])] if len(shape) > 1 else [7] * shape[0]
        torch.musa.synchronize()
        started = time.perf_counter()
        for _ in range(100):
            torch.tensor(values, dtype=dtype, pin_memory=True).to(
                device, non_blocking=True
            )
        torch.musa.synchronize()
        rows[name] = (time.perf_counter() - started) / 100
    rows["total_per_step_s"] = sum(rows.values())
    return rows


def profile_one_step(llm):
    """Kernel count and CPU-vs-device split for one decode step, if supported."""
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        device_activity = getattr(torch.profiler.ProfilerActivity, "CUDA", None)
        if device_activity is not None:
            activities.append(device_activity)
        with torch.profiler.profile(activities=activities) as prof:
            llm.step()
        events = prof.key_averages()
        kernel_events = [
            event
            for event in events
            if getattr(event, "self_device_time_total", 0) > 0
        ]
        return {
            "kernel_count": len(kernel_events),
            "kernel_total_us": sum(
                getattr(event, "self_device_time_total", 0) for event in kernel_events
            ),
            "cpu_total_us": sum(
                getattr(event, "self_cpu_time_total", 0) for event in kernel_events
            ),
            "top_cpu_ops": [
                {
                    "key": event.key,
                    "cpu_us": round(getattr(event, "self_cpu_time_total", 0), 1),
                    "device_us": round(getattr(event, "self_device_time_total", 0), 1),
                    "count": event.count,
                }
                for event in sorted(
                    events, key=lambda e: -getattr(e, "self_cpu_time_total", 0)
                )[:15]
            ],
        }
    except Exception as exc:  # noqa: BLE001 - profiler support is optional
        return {"profiler_error": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/models/Qwen3-8B")
    parser.add_argument("--batches", default="1,8,32")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--attention-backend", default="triton_paged")
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--num-kvcache-blocks", type=int, default=0,
                        help="0 = let the utilisation heuristic size the cache")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--output", default="/root/p6_step_overhead.json")
    args = parser.parse_args()
    batches = [int(value) for value in args.batches.split(",")]

    from nanovllm import LLM

    started = time.time()
    llm = LLM(
        args.model,
        enforce_eager=False,  # capture at init; eager is selected at runtime
        tensor_parallel_size=1,
        attention_backend=args.attention_backend,
        scheduler_policy="fcfs",
        max_num_seqs=args.max_num_seqs,
        max_model_len=max(2048, args.prompt_len + args.max_tokens),
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_kvcache_blocks=args.num_kvcache_blocks,
    )
    runner = llm.model_runner
    graph_report = runner.call("graph_report")
    print(f"[overhead] graph_report={json.dumps(graph_report)}", flush=True)

    phases = install_timed_run(runner)
    rng = random.Random(SEED)
    vocab = llm.model_runner.config.hf_config.vocab_size
    payload = {
        "schema_version": 1,
        "kind": "step_overhead",
        "model": args.model,
        "attention_backend": args.attention_backend,
        "prompt_len": args.prompt_len,
        "max_tokens": args.max_tokens,
        "graph_report": graph_report,
        "pinned_buffer_cost": pinned_buffer_cost(),
        "profiles": {},
        "records": [],
    }
    print(
        f"[overhead] pinned_buffer_cost={json.dumps(payload['pinned_buffer_cost'])}",
        flush=True,
    )

    for batch in batches:
        # warmup for this batch size (JIT, allocator, bucket selection)
        for _ in range(args.warmup_runs):
            run_workload(llm, phases, args, batch, rng, vocab, f"warmup-b{batch}")
        phases["prepare"].clear()
        phases["forward"].clear()
        phases["sample"].clear()

        runner.call("set_enforce_eager", True)
        eager = run_workload(llm, phases, args, batch, rng, vocab, f"eager-b{batch}")
        phases["prepare"].clear()
        phases["forward"].clear()
        phases["sample"].clear()

        runner.call("set_enforce_eager", False)
        graph = run_workload(llm, phases, args, batch, rng, vocab, f"graph-b{batch}")

        eager["throughput_ratio_graph_over_eager"] = (
            graph["decode_tokens_per_s"] / eager["decode_tokens_per_s"]
            if eager["decode_tokens_per_s"] else None
        )
        payload["records"].append(eager)
        payload["records"].append(graph)

        # one profiled step for kernel counts, on the graph path
        runner.call("set_enforce_eager", False)
        llm.scheduler.block_manager.clear_prefix_cache()
        prompts = make_prompts(batch, args.prompt_len, vocab, rng)
        from nanovllm import SamplingParams
        params = SamplingParams(
            temperature=0, max_tokens=args.max_tokens, ignore_eos=True
        )
        for prompt in prompts:
            llm.add_request(prompt, params)
        while not llm.is_finished():
            llm.step()
            if llm.last_step_metrics["mode"] == "decode":
                break
        payload["profiles"][f"b{batch}"] = profile_one_step(llm)
        while not llm.is_finished():
            llm.step()
        print(
            f"[overhead] profile b{batch}={json.dumps(payload['profiles'][f'b{batch}'])[:600]}",
            flush=True,
        )

    payload["wall_seconds"] = round(time.time() - started, 1)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[overhead] wrote {out} (wall {payload['wall_seconds']}s)", flush=True)
    llm.exit()


if __name__ == "__main__":
    main()
