"""P3 (P11): decode step-time A/B — eager vs segment graph, b1.

Directional probe for I19's ~20-30 ms fixed wait: does removing per-op host
launches (segment replay) change the decode step wall?  One mode per process
(two-process alternation sidesteps the two-engine context); the caller
alternates eager/segment runs and gates on within-arm CV.

Frozen gate (benchmarks/P11-graph.md P3-STEP): 3 prompts x 128 tokens, b1
greedy, 80 decode tokens; per-run decode-step median; CV<=5%.
"""
import argparse
import json
import statistics
import time
import os
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eager", "segment"], required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--attention-backend", default="triton_paged")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=80)
    parser.add_argument("--rounds", type=int, default=4,
                        help="rounds of 3 single-batch generates; round 1 = warmup, dropped")
    parser.add_argument("--output", default="/root/p11_step.json")
    parser.add_argument("--batches", default="1", help="comma-separated batch sizes; P13 scans non-boundaries")
    parser.add_argument("--code-revision", default="unknown")
    args = parser.parse_args()
    batches = [int(value) for value in args.batches.split(",")]
    assert batches and min(batches) > 0 and args.rounds >= 3 and args.decode_tokens >= 3
    os.environ["NANOVLLM_SEGMENT_GRAPH"] = "1" if args.mode == "segment" else "0"

    from nanovllm import LLM, SamplingParams

    kwargs = {}
    if not args.dense:
        kwargs = {"moe_backend": "grouped_mm", "decode_moe_backend": "bmm_dense"}
    llm = LLM(
        args.model,
        enforce_eager=(args.mode == "eager"),
        tensor_parallel_size=args.tp,
        expert_parallel_size=args.tp,
        attention_backend=args.attention_backend,
        max_model_len=1024,
        max_num_seqs=max(8, ((max(batches) + 15) // 16) * 16) if batches != [1] else 8,
        gpu_memory_utilization=0.3,
        **kwargs,
    )
    if args.batches != "1":
        try:
            scan_batches(llm, batches, args, SamplingParams)
        finally:
            llm.exit()
        return
    prompts = [
        [(i * 7919 + j) % min(900, llm.model_runner.config.hf_config.vocab_size - 10) + 10 for j in range(args.prompt_tokens)]
        for i in range(3)
    ]
    payload = {"mode": args.mode, "rounds": []}
    for round_index in range(args.rounds):
        round_ms = []
        prefill_ms = []
        for prompt in prompts:
            llm.add_request(
                prompt,
                SamplingParams(temperature=0, max_tokens=args.decode_tokens,
                               ignore_eos=True),
            )
            while not llm.is_finished():
                t0 = time.perf_counter()
                llm.step()
                dt = (time.perf_counter() - t0) * 1000.0
                if llm.last_step_metrics["mode"] == "decode":
                    round_ms.append(dt)
                else:
                    prefill_ms.append(dt)
        payload["rounds"].append(
            {"round": round_index, "warmup": round_index == 0,
             "decode_steps": len(round_ms),
             "decode_median_ms": statistics.median(round_ms),
             "prefill_median_ms": statistics.median(prefill_ms) if prefill_ms else None}
        )
        print(f"[step-ab] mode={args.mode} round={round_index} "
              f"decode_median={statistics.median(round_ms):.2f}ms "
              f"steps={len(round_ms)}", flush=True)

    measured = [r["decode_median_ms"] for r in payload["rounds"] if not r["warmup"]]
    median = statistics.median(measured)
    mean = statistics.fmean(measured)
    cv_pct = (statistics.pstdev(measured) / mean * 100.0) if mean > 0 else None
    payload["decode_step_median_ms"] = median
    payload["within_arm_cv_pct"] = cv_pct
    payload["measured_rounds"] = len(measured)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    print(f"[step-ab] mode={args.mode} FINAL decode_step_median_ms={median:.2f} "
          f"cv_pct={cv_pct:.2f} rounds={len(measured)}", flush=True)
    llm.exit()


def scan_batches(llm, batches, args, sampling_params):
    """P13 shape sweep. First full batch at EACH shape is discarded (I31)."""
    payload = {"kind": "p13_padding", "mode": args.mode, "args": vars(args),
               "started": time.time(), "batches": {},
               "kv_blocks": llm.model_runner.config.num_kvcache_blocks,
               "graph_buckets": getattr(llm.model_runner, "graph_bs", []),
               "versions": {}, "code_revision": args.code_revision}
    import torch
    import torch_musa
    payload["versions"] = {"torch": torch.__version__, "torch_musa": torch_musa.__version__}
    for batch in batches:
        prompts = [[(i * 7919 + j) % min(900, llm.model_runner.config.hf_config.vocab_size - 10) + 10 for j in range(args.prompt_tokens)]
                   for i in range(batch)]
        rounds = []
        for index in range(args.rounds):
            for prompt in prompts:
                llm.add_request(prompt, sampling_params(temperature=0,
                    max_tokens=args.decode_tokens, ignore_eos=True))
            steps, actual_batches, tokens = [], [], {}
            while not llm.is_finished():
                started = time.perf_counter()
                outputs, num_tokens = llm.step()
                elapsed = (time.perf_counter() - started) * 1000
                if llm.last_step_metrics["mode"] == "decode":
                    steps.append(elapsed)
                    actual_batches.append(-num_tokens)
                for seq_id, token_ids, _ in outputs:
                    tokens[seq_id] = list(token_ids)
            ordered = [tokens[key] for key in sorted(tokens)]
            rounds.append({"warmup": index == 0, "decode_ms": steps,
                "decode_median_ms": statistics.median(steps),
                "actual_batches": sorted(set(actual_batches)),
                "token_sha256": hashlib.sha256(json.dumps(ordered).encode()).hexdigest()})
            print(f"[p13-padding] {args.mode} b{batch} round={index} "
                  f"median={rounds[-1]['decode_median_ms']:.3f}ms", flush=True)
        values = [r["decode_median_ms"] for r in rounds[1:]]
        buckets = payload["graph_buckets"]
        payload["batches"][str(batch)] = {"rounds": rounds,
            "cap": next((n for n in buckets if n >= batch), batch),
            "decode_median_ms": statistics.median(values),
            "cv": statistics.stdev(values) / statistics.mean(values),
            "actual_batch_valid": all(r["actual_batches"] == [batch] for r in rounds),
            "tokens_consistent": len({r["token_sha256"] for r in rounds[1:]}) == 1}
        payload["finished"] = time.time()
        Path(args.output).write_text(json.dumps(payload, indent=2))
    payload["complete"] = True
    Path(args.output).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
