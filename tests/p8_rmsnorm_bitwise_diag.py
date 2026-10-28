"""Diagnose the fused-RMSNorm end-to-end divergence (P1.1 G2 vs G1 contradiction).

The kernel probe is bitwise-identical to eager, yet one free-decode prompt
flipped argmax with margin 1.375.  This script runs one prompt under both
modes with a hook on EVERY RMSNorm module, compares outputs bitwise per call,
and reports the first divergent call (module, call index, shape, dtype, number
of differing elements).  It also re-checks token streams.

Run:
  PYTHONPATH=/root/moe-server-s4000 python3 tests/p8_rmsnorm_bitwise_diag.py
"""
import argparse
import json
import random
import sys

import torch

sys.path.insert(0, "/root/moe-server-s4000")

from nanovllm import LLM, SamplingParams  # noqa: E402
from nanovllm.layers.layernorm import RMSNorm, set_fused_rmsnorm  # noqa: E402

SEED = 20261027
PROMPT_LEN = 128
STEPS = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    parser.add_argument("--prompt-index", type=int, default=6)
    parser.add_argument("--output", default="/root/p8_rmsnorm_bitwise_diag.json")
    args = parser.parse_args()

    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        attention_backend="sdpa",
        scheduler_policy="fcfs",
        max_num_seqs=1,
        max_model_len=PROMPT_LEN + STEPS + 16,
        gpu_memory_utilization=0.35,
    )
    vocab = llm.model_runner.config.hf_config.vocab_size
    rng = random.Random(SEED)
    prompts = [[rng.randrange(vocab) for _ in range(PROMPT_LEN)] for _ in range(8)]
    prompt = prompts[args.prompt_index]

    norm_modules = [
        (name, module)
        for name, module in llm.model_runner.model.named_modules()
        if isinstance(module, RMSNorm)
    ]
    print(f"[diag] {len(norm_modules)} RMSNorm modules", flush=True)

    def capture_run(mode):
        records = []
        handles = []

        def make_hook(entry):
            def hook(_module, _inputs, output):
                out = output[0] if isinstance(output, tuple) else output
                records.append((entry, out.detach().clone()))
            return hook

        for name, module in norm_modules:
            handles.append(module.register_forward_hook(make_hook(name)))
        set_fused_rmsnorm(mode == "fused")
        llm.add_request(prompt, SamplingParams(temperature=0, max_tokens=STEPS, ignore_eos=True))
        tokens = []
        while not llm.is_finished():
            _outputs, _num = llm.step()
        set_fused_rmsnorm(False)
        for handle in handles:
            handle.remove()
        # collect token ids from the finished sequence (LLM IS the engine)
        scheduler = llm.scheduler
        for seq in list(scheduler.running) + list(scheduler.waiting):
            if seq.is_finished:
                tokens = list(seq.completion_token_ids)
        return tokens, records

    eager_tokens, eager_records = capture_run("eager")
    fused_tokens, fused_records = capture_run("fused")
    eager_again_tokens, _ = capture_run("eager")

    tokens_match = eager_tokens == fused_tokens
    print(f"[diag] tokens_match={tokens_match} "
          f"eager_self_consistent={eager_tokens == eager_again_tokens} "
          f"n_records eager={len(eager_records)} fused={len(fused_records)}", flush=True)

    first_divergence = None
    mismatch_calls = 0
    if len(eager_records) == len(fused_records):
        for index, ((e_name, e_out), (f_name, f_out)) in enumerate(zip(eager_records, fused_records)):
            if e_out.shape != f_out.shape or e_out.dtype != f_out.dtype:
                first_divergence = {
                    "call_index": index, "module": e_name,
                    "kind": "shape_or_dtype",
                    "eager": list(e_out.shape), "fused": list(f_out.shape),
                }
                break
            if not torch.equal(e_out, f_out):
                mismatch_calls += 1
                diff = (e_out.float() - f_out.float()).abs()
                if first_divergence is None:
                    first_divergence = {
                        "call_index": index,
                        "module": e_name,
                        "kind": "value",
                        "shape": list(e_out.shape),
                        "dtype": str(e_out.dtype),
                        "mismatch_elements": int((e_out != f_out).sum()),
                        "max_abs": float(diff.max()),
                        "input_is_contiguous_hint": None,
                    }
    else:
        first_divergence = {"kind": "record_count_mismatch",
                            "eager": len(eager_records), "fused": len(fused_records)}

    print(f"[diag] mismatch_calls={mismatch_calls} first={json.dumps(first_divergence)}",
          flush=True)

    payload = {
        "schema_version": 1,
        "kind": "rmsnorm_bitwise_diag",
        "prompt_index": args.prompt_index,
        "tokens_match": tokens_match,
        "eager_self_consistent": eager_tokens == eager_again_tokens,
        "record_count": len(eager_records),
        "mismatch_calls": mismatch_calls,
        "first_divergence": first_divergence,
        "eager_tokens_head": eager_tokens[:8],
        "fused_tokens_head": fused_tokens[:8],
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=str)
    print(f"[diag] wrote {args.output}", flush=True)
    llm.exit()


if __name__ == "__main__":
    main()
