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
    args = parser.parse_args()

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
        max_num_seqs=8,
        gpu_memory_utilization=0.3,
        **kwargs,
    )
    prompts = [
        [(i * 7919 + j) % 900 + 10 for j in range(args.prompt_tokens)]
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


if __name__ == "__main__":
    main()
