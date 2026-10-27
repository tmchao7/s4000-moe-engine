"""Diagnose SDPA vs Triton paged decode logits on the frozen P5/P6 workloads.

The runner compares both backends at exactly the same decode state and commits
the SDPA KV result.  Every later step is therefore teacher-forced on the SDPA
history; an early Triton argmax cannot cascade into a misleading difference.
The script intentionally uses grouped_mm throughout because that is the frozen
attention-gate configuration consumed by ``scripts/p5_decide.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import traceback
from pathlib import Path


DEFAULT_MODEL = "/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
DEFAULT_WORKLOADS = "16x512,32x512,16x2048,32x2048"
SEED = 20261026


def parse_workloads(value: str) -> list[tuple[int, int]]:
    workloads = [tuple(map(int, item.lower().split("x"))) for item in value.split(",")]
    assert workloads and all(batch > 0 and context > 0 for batch, context in workloads)
    return workloads


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_teacher_forced_probe(llm, prompts, params, emit):
    """Run one workload and retain only compact runner diagnostics."""
    for prompt in prompts:
        llm.add_request(prompt, params)
    completions = {}
    diagnostics = []
    decode_step = 0
    while not llm.is_finished():
        batch = llm.scheduler.schedule()
        seqs, mode = batch
        if mode == "decode":
            result = llm.model_runner.call("probe_attention_backends", seqs)
            token_ids = result["token_ids"]
            diagnostic = result["diagnostics"]
            diagnostic["decode_step"] = decode_step
            diagnostics.append(diagnostic)
            emit({"kind": "decode_step", **diagnostic})
            decode_step += 1
        else:
            assert mode == "prefill", f"expected fcfs prefill/decode only, got {mode}"
            token_ids = llm.model_runner.call("run", seqs, mode)
        llm.scheduler.postprocess(seqs, token_ids, mode)
        for seq in seqs:
            if seq.is_finished:
                completions[seq.seq_id] = seq.completion_token_ids
    return {
        "decode_steps": decode_step,
        "completion_token_ids": [completions[key] for key in sorted(completions)],
        "diagnostics": diagnostics,
    }


def summarise_run(result: dict) -> dict:
    steps = result["diagnostics"]
    request_rows = [request for step in steps for request in step["requests"]]
    layer_rows = [layer for step in steps for layer in step["layers"] if not layer.get("missing")]
    finite = all(step["finite"] for step in steps)
    return {
        "decode_steps": result["decode_steps"],
        "finite": finite,
        "logit_max_abs": max((step["logits"]["max_abs"] or 0.0) for step in steps) if steps else 0.0,
        "logit_mean_abs_mean": (
            sum((step["logits"]["mean_abs"] or 0.0) for step in steps) / len(steps) if steps else 0.0
        ),
        "top1_mismatch_requests": sum(not row["top1_equal"] for row in request_rows),
        "top1_compared_requests": len(request_rows),
        "first_top1_mismatch": next(
            (
                {
                    "decode_step": step["decode_step"],
                    "request": request,
                }
                for step in steps
                for request in step["requests"]
                if not request["top1_equal"]
            ),
            None,
        ),
        "max_layer_abs": max((row["max_abs"] or 0.0) for row in layer_rows) if layer_rows else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workloads", default=DEFAULT_WORKLOADS)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", default="/root/p6_attention_logits.json")
    parser.add_argument("--jsonl", default="/root/p6_attention_logits.jsonl")
    args = parser.parse_args()
    assert args.output_len > 1 and args.repeats > 0
    workloads = parse_workloads(args.workloads)
    output_path = Path(args.output)
    jsonl_path = Path(args.jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    from nanovllm import LLM, SamplingParams

    payload = {
        "schema_version": 1,
        "model": args.model,
        "seed": args.seed,
        "workloads": [{"batch": batch, "context": context} for batch, context in workloads],
        "output_len": args.output_len,
        "repeats": args.repeats,
        "moe_backend": "grouped_mm",
        "decode_moe_backend": "grouped_mm",
        "attention_backends": ["sdpa", "triton_paged"],
        "trajectory": "SDPA teacher-forced after each measured decode step",
        "runs": [],
        "errors": [],
    }
    stream = jsonl_path.open("w", encoding="utf-8")

    def emit(record: dict):
        line = json.dumps(record, sort_keys=True, allow_nan=False)
        print("[p6-attention-logits]", line, flush=True)
        stream.write(line + "\n")
        stream.flush()

    llm = None
    try:
        llm = LLM(
            args.model,
            enforce_eager=True,
            tensor_parallel_size=2,
            expert_parallel_size=2,
            moe_backend="grouped_mm",
            decode_moe_backend="grouped_mm",
            attention_backend="sdpa",
            scheduler_policy="fcfs",
            max_model_len=max(context for _, context in workloads) + args.output_len + 8,
            gpu_memory_utilization=0.85,
        )
        vocab_size = llm.model_runner.config.hf_config.vocab_size
        params = SamplingParams(temperature=0, max_tokens=args.output_len, ignore_eos=True)
        rng = random.Random(args.seed)
        for batch, context in workloads:
            prompts = [[rng.randrange(vocab_size) for _ in range(context)] for _ in range(batch)]
            for repeat in range(args.repeats):
                llm.scheduler.block_manager.clear_prefix_cache()
                emit({"kind": "run_start", "batch": batch, "context": context, "repeat": repeat})
                try:
                    result = run_teacher_forced_probe(llm, prompts, params, emit)
                    summary = summarise_run(result)
                    run = {
                        "batch": batch,
                        "context": context,
                        "repeat": repeat,
                        "summary": summary,
                        "result": result,
                    }
                    payload["runs"].append(run)
                    emit({"kind": "run_summary", "batch": batch, "context": context, "repeat": repeat, **summary})
                except Exception as exc:  # preserve completed JSONL evidence on a device failure
                    error = {
                        "batch": batch,
                        "context": context,
                        "repeat": repeat,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=12),
                    }
                    payload["errors"].append(error)
                    emit({"kind": "run_error", **error})
                    raise
    finally:
        if llm is not None:
            try:
                llm.exit()
            except Exception as exc:  # a device error may make teardown unusable
                payload["errors"].append({"stage": "exit", "error": f"{type(exc).__name__}: {exc}"})
        stream.close()
        output_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        digest = sha256_file(output_path)
        jsonl_digest = sha256_file(jsonl_path)
        print(
            f"[p6-attention-logits] wrote {output_path} sha256={digest} "
            f"jsonl_sha256={jsonl_digest}",
            flush=True,
        )


if __name__ == "__main__":
    main()
