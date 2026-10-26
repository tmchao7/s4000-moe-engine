"""P5 graph A/B: does MUSA graph capture pay for itself on the decode path?

One weight load, one process, three configurations measured back to back:

  ``eager_default``   today's stack: ``enforce_eager`` + grouped_mm MoE + SDPA
  ``eager_candidate`` the capturable stack run **eagerly**: vllm_fused MoE +
                      Triton paged decode attention (same kernels the graph embeds,
                      so this isolates "graph vs no graph" from "kernel change")
  ``graph_candidate`` the same capturable stack replayed from captured graphs

The graphs are captured during ``LLM(...)`` init (``enforce_eager=False``) with
``decode_moe_backend`` as the MoE kernel inside them; eager prefill keeps
``moe_backend``.  Switching ``enforce_eager`` at runtime is what selects replay
vs eager forward, so no reload is needed.

Output: a payload consumable by the frozen gate
``python3 scripts/p5_decide.py --mode graph --payload <output>`` plus records.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from nanovllm.benchmarking import coefficient_of_variation, percentile

# Import by file path: some environments ship an unrelated site-packages
# ``scripts`` package that shadows this repository's namespace package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_benchmark import git_commit, prompts  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIGS = ("graph_candidate", "eager_candidate", "eager_default")


def summarize(records, config, batch):
    rows = [row for row in records if row["config"] == config and row["batch"] == batch]
    if not rows:
        return None
    throughputs = [row["output_tokens_per_s"] for row in rows]
    ttfts = [row["ttft_p95_s"] for row in rows]
    tpots = [row["tpot_p95_s"] for row in rows]
    return {
        "config": config,
        "graph": config == "graph_candidate",
        "batch": batch,
        "repeats": len(rows),
        "throughput_median": statistics.median(throughputs),
        "throughput_cv": coefficient_of_variation(throughputs),
        "ttft_p95_median": statistics.median(ttfts),
        "tpot_p95_median": statistics.median(tpots),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--batches", default="1,2,4,8,16,32")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--decode-moe-backend", default="vllm_fused")
    parser.add_argument("--prefill-moe-backend", default="grouped_mm")
    parser.add_argument("--attention-backend", default="triton_paged")
    parser.add_argument("--max-num-seqs", type=int, default=32)
    # Graph capture needs headroom on top of the KV cache, which is sized to
    # gpu_memory_utilization.  At 0.85 the capture OOMed (384 MiB short with
    # 44.29 GiB already allocated), so the default here leaves room for the pool.
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    # Explicit, small KV cache: capture needs contiguous headroom, and the
    # utilisation heuristic fills the device (two OOMs during capture).
    parser.add_argument("--num-kvcache-blocks", type=int, default=64)
    parser.add_argument("--verify-tokens", type=int, default=32)
    parser.add_argument("--output", default="/root/p5_graph_ab.json")
    args = parser.parse_args()
    batches = [int(value) for value in args.batches.split(",")]

    from nanovllm import LLM, SamplingParams

    started = time.time()
    payload = {
        "schema_version": 2,
        "kind": "graph_ab",
        "git_commit": git_commit(),
        "decode_moe_backend": args.decode_moe_backend,
        "prefill_moe_backend": args.prefill_moe_backend,
        "attention_backend": args.attention_backend,
        "prompt_len": args.prompt_len,
        "output_len": args.output_len,
        "repeats": args.repeats,
        "batches": batches,
        "regression_batches": [32],
        "capture_error": None,
        "configs": {},
    }

    try:
        llm = LLM(
            args.model,
            enforce_eager=False,  # capture the decode graphs at init
            tensor_parallel_size=2,
            expert_parallel_size=2,
            moe_backend=args.prefill_moe_backend,
            decode_moe_backend=args.decode_moe_backend,
            attention_backend=args.attention_backend,
            scheduler_policy="fcfs",
            max_num_seqs=args.max_num_seqs,
            max_model_len=max(2048, args.prompt_len + args.output_len),
            gpu_memory_utilization=args.gpu_memory_utilization,
            num_kvcache_blocks=args.num_kvcache_blocks,
        )
    except Exception as exc:  # noqa: BLE001 - capture failure is a result, not a crash
        payload["capture_error"] = f"{type(exc).__name__}: {exc}"
        payload["wall_seconds"] = round(time.time() - started, 1)
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[graph-ab] CAPTURE FAILED: {payload['capture_error']}", flush=True)
        print(f"[graph-ab] wrote {args.output}", flush=True)
        raise SystemExit(2)

    runner = llm.model_runner
    payload["graph_report"] = runner.call("graph_report")
    allocated_delta = payload["graph_report"].get("allocated_delta_gib") or 0.0
    reserved_delta = payload["graph_report"].get("reserved_delta_gib") or 0.0
    # The frozen gate bounds the extra memory graph capture costs; a private
    # graph pool usually appears as *reserved* rather than allocated memory.
    payload["memory_delta_gib"] = max(allocated_delta, reserved_delta)
    payload["memory_delta_detail"] = {
        "allocated_delta_gib": allocated_delta,
        "reserved_delta_gib": reserved_delta,
    }
    print(f"[graph-ab] captured: {json.dumps(payload['graph_report'])}", flush=True)

    vocab_size = runner.config.hf_config.vocab_size
    params = SamplingParams(temperature=0, max_tokens=args.output_len, ignore_eos=True)
    records = []
    tokens = {}

    def configure(config):
        print(f"[graph-ab] configure {config}", flush=True)
        if config == "eager_default":
            runner.call("set_enforce_eager", True)
            runner.call("set_moe_backend", "grouped_mm", "grouped_mm")
            runner.call("set_attention_backend", "sdpa")
        elif config == "eager_candidate":
            runner.call("set_enforce_eager", True)
            # Prefill stays on the fast grouped_mm path; only decode uses the
            # capturable backend (bmm_dense's dense padding is prohibitive for
            # a thousands-token prefill chunk).
            runner.call(
                "set_moe_backend", args.prefill_moe_backend, args.decode_moe_backend
            )
            runner.call("set_attention_backend", args.attention_backend)
        else:
            # replay path; the captured graphs already embed the candidate kernels
            runner.call("set_enforce_eager", False)

    def generate(config, batch):
        configure(config)
        print(f"[graph-ab] start {config} batch={batch}", flush=True)
        batch_prompts = prompts(batch, args.prompt_len, vocab_size, seed=batch)
        llm.scheduler.block_manager.clear_prefix_cache()
        llm.generate(batch_prompts, params, use_tqdm=False)  # unreported warmup
        runs = []
        for repeat in range(args.repeats):
            llm.scheduler.block_manager.clear_prefix_cache()
            start = time.perf_counter()
            outputs = llm.generate(batch_prompts, params, use_tqdm=False)
            elapsed = time.perf_counter() - start
            ttfts = [output["metrics"]["ttft_s"] for output in outputs]
            tpots = [output["metrics"]["tpot_s"] for output in outputs]
            itls = [
                interval for output in outputs for interval in output["metrics"]["itls_s"]
            ]
            output_tokens = sum(len(output["token_ids"]) for output in outputs)
            input_tokens = sum(output["metrics"]["prompt_tokens"] for output in outputs)
            record = {
                "config": config,
                "graph": config == "graph_candidate",
                "batch": batch,
                "repeat": repeat,
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
            }
            runs.append(record)
            records.append(record)
            print(
                "[graph-ab]", json.dumps({
                    "config": config, "batch": batch, "repeat": repeat,
                    "out_tok_s": round(record["output_tokens_per_s"], 2),
                    "ttft_p95": round(record["ttft_p95_s"], 3),
                    "tpot_p95": round(record["tpot_p95_s"], 4),
                }, sort_keys=True), flush=True,
            )
        return runs

    # Correctness + throughput ordering: every graph replay must happen before
    # the first eager decode step.  Eager decode on the MCCL communicator makes
    # a later graph replay hang (reproduced on 8B TP2; tiny and a pure-allreduce
    # probe are fine), and production never does eager decode: decode always
    # replays the graphs, only prefill is eager.  Graph is therefore measured
    # first, eager configs afterwards; the gate compares the same kernels.
    verify_prompts = prompts(4, args.prompt_len, vocab_size, seed=999)
    verify_params = SamplingParams(temperature=0, max_tokens=args.verify_tokens, ignore_eos=True)

    def verify(config):
        configure(config)
        print(f"[graph-ab] verify {config}", flush=True)
        llm.scheduler.block_manager.clear_prefix_cache()
        llm.generate(verify_prompts, verify_params, use_tqdm=False)  # warmup
        outputs = llm.generate(verify_prompts, verify_params, use_tqdm=False)
        return [list(output["token_ids"]) for output in outputs]

    tokens["graph_candidate"] = verify("graph_candidate")

    summary = []
    summary_default = []
    for config in CONFIGS:
        for batch in batches:
            generate(config, batch)

    # Eager token identity last: it is the only remaining use of eager decode.
    tokens["eager_candidate"] = verify("eager_candidate")
    tokens["eager_default"] = verify("eager_default")
    payload["verify_tokens"] = tokens
    for config in CONFIGS:
        payload.setdefault("configs", {})[config] = {
            "matches_eager_default": tokens[config] == tokens["eager_default"],
            "matches_eager_candidate": tokens[config] == tokens["eager_candidate"],
        }
    print(
        "[graph-ab] token identity: "
        + json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in payload["configs"].items()}),
        flush=True,
    )

    shared = len(tokens["graph_candidate"][0]) if tokens.get("graph_candidate") else 0
    graph_identity = tokens.get("graph_candidate") == tokens.get("eager_candidate")
    payload["graph_token_identity"] = graph_identity
    payload["graph_token_identity_detail"] = {
        "graph": tokens.get("graph_candidate"),
        "eager": tokens.get("eager_candidate"),
        "tokens_per_prompt": shared,
    }

    for batch in batches:
        for config in CONFIGS:
            row = summarize(records, config, batch)
            if row is None:
                continue
            if config == "eager_default":
                # kept apart: the frozen gate compares graph replay against the
                # *same kernels* running eagerly, not against today's stack.
                summary_default.append(row)
                continue
            row["token_identity"] = bool(graph_identity) if row["graph"] else True
            if row["graph"]:
                row["token_identity_detail"] = payload["graph_token_identity_detail"]
            summary.append(row)

    payload["cross_kernel"] = []
    for batch in batches:
        graph = next((r for r in summary if r["batch"] == batch and r["graph"]), None)
        candidate = next((r for r in summary if r["batch"] == batch and not r["graph"]), None)
        default = next((r for r in summary_default if r["batch"] == batch), None)
        payload["cross_kernel"].append({
            "batch": batch,
            "graph_vs_candidate": (graph["throughput_median"] / candidate["throughput_median"])
            if graph and candidate and candidate["throughput_median"] else None,
            "candidate_vs_default": (
                candidate["throughput_median"] / default["throughput_median"]
            ) if candidate and default and default["throughput_median"] else None,
            "graph_vs_default": (graph["throughput_median"] / default["throughput_median"])
            if graph and default and default["throughput_median"] else None,
        })
    print(f"[graph-ab] cross-kernel: {json.dumps(payload['cross_kernel'])}", flush=True)

    payload["summary"] = summary
    payload["summary_eager_default"] = summary_default
    payload["records"] = records
    payload["wall_seconds"] = round(time.time() - started, 1)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    out.with_suffix(".jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8"
    )
    print(f"[graph-ab] wrote {out} (wall {payload['wall_seconds']}s)", flush=True)
    print(
        "[graph-ab] decide with: "
        f"python3 scripts/p5_decide.py --mode graph "
        f"--input {out} --output {out.with_name(out.stem + '-decision.json')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
