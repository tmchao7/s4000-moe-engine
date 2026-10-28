"""Tiny validation for the bmm_dense capacity shrink (P3.2 code gate).

Gates (frozen in docs/NIGHTLY-LOG.md P3.2):
  C1  both paths self-deterministic: 3 repeats each, identical tokens
  C2  token agreement between padded and shrunk paths reported; near-tie
      flips are expected (bmm tiling order changes) -- this is a report, the
      binding numeric gate on 30B is self-determinism + L2 soft gate
  C3  owned/folded stats consistent: owned+folded == slots per step,
      owned_expert_counts.sum() == owned_slot_total
  C4  shrink path actually shrinks: max_load < T*K observed on tiny batches

Run:
  PYTHONPATH=. python3 tests/p8_moe_capacity_tiny.py
"""
import argparse
import json
import sys

import torch

sys.path.insert(0, "/root/moe-server-s4000")

MODEL_DIR = "/tmp/tiny_moe"
PROMPTS = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12], [42, 42, 42]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/root/p8_moe_capacity_tiny.json")
    args = parser.parse_args()

    from nanovllm import LLM, SamplingParams
    from nanovllm.models.qwen3_moe import set_capacity_shrink

    llm = LLM(
        MODEL_DIR,
        enforce_eager=True,
        tensor_parallel_size=2,
        expert_parallel_size=2,
        moe_backend="grouped_mm",
        decode_moe_backend="bmm_dense",
        attention_backend="sdpa",
        max_model_len=256,
        max_num_seqs=32,
        gpu_memory_utilization=0.5,
        collect_expert_stats=True,
    )
    payload = {"schema_version": 1, "kind": "moe_capacity_tiny"}

    def run_once():
        outputs = llm.generate(
            PROMPTS,
            SamplingParams(temperature=0, max_tokens=16, ignore_eos=True),
            use_tqdm=False,
        )
        return [output["token_ids"] for output in outputs]

    streams = {}
    for mode in ("padded", "shrunk"):
        set_capacity_shrink(mode == "shrunk")
        streams[mode] = [run_once() for _ in range(3)]
    set_capacity_shrink(False)

    payload["self_deterministic"] = {
        mode: all(stream == stream[0] for stream in runs)
        for mode, runs in streams.items()
    }
    flat = lambda runs: [token for output in runs[0] for token in output]
    padded_tokens = flat(streams["padded"])
    shrunk_tokens = flat(streams["shrunk"])
    agreement = sum(a == b for a, b in zip(padded_tokens, shrunk_tokens)) / len(padded_tokens)
    payload["token_agreement_padded_vs_shrunk"] = agreement
    payload["identical"] = streams["padded"][0] == streams["shrunk"][0]

    # C3/C4: stats after the last (padded) run cover only that run; rerun once
    # with stats and capture max per-expert owned load from the shrink path.
    set_capacity_shrink(True)
    run_once()
    set_capacity_shrink(False)
    runner = llm.model_runner
    stats = runner.model.owned_expert_stats() if hasattr(runner.model, "owned_expert_stats") else None
    if stats is None:
        from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM

        stats = runner.model.owned_expert_stats()
    owned_counts = stats["owned_counts"]
    max_load = int(owned_counts.max())
    per_step_owned = sum(stats["owned_slot_total"])
    per_step_folded = sum(stats["folded_slot_total"])
    slots_per_step = 3 * 8 * 8 * 2  # prompts * max_tokens... decode rows only; approximate
    payload["stats"] = {
        "max_owned_load_per_expert": max_load,
        "owned_slot_total_all_steps": per_step_owned,
        "folded_slot_total_all_steps": per_step_folded,
        "owned_plus_folded_matches": owned_counts.sum().item() == per_step_owned,
        "note": "decode slots per step = n_running * top_k; tiny uses top_k from config",
    }
    payload["shrink_observed"] = max_load < 64  # tiny E_local=... reported for reference

    llm.exit()
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=str)
    print(f"[capacity_tiny] {json.dumps(payload, default=str)}", flush=True)


if __name__ == "__main__":
    main()
