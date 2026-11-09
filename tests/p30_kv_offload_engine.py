"""P30-R1 Round B engine gates: KV offload evict-to-host + prefix revival.

Single engine, kv_offload ON throughout (init-time config — no runtime
switch, per the pre-registered design in benchmarks/P30-kv-offload.md §2):

  Arm R   two requests sharing a 3-block prefix, run to finish (reference
          finals; the offload path is idle — no pressure).
  Arm X   a junk load forces _allocate_block to evict P's cached prefix
          blocks to the host pool (puts > 0), then the same two P requests
          run again: their prefixes must revive from the host (gets > 0)
          and the outputs must be BITWISE equal to Arm R.  The block-level
          roundtrip is bitwise (p30_kv_offload_unit t5), so any mismatch is
          a wiring bug — EXCEPT legitimate near-tie flips from the junk
          load's history (I31/I34), which the third arm disambiguates.
  Arm R2  Arm R's load again, junk-free → R2 == R proves history alone
          doesn't flip these prompts; X != R with R2 == R would then point
          at offload.  If R2 != R the gate design is INVALID for this load
          (reported as such, never faked).

  Accounting: pool in_use == len(host_map) after every arm;
  clear_prefix_cache() drains both to zero.

Run:  PYTHONPATH=. python3 tests/p30_kv_offload_engine.py   # rc 0 = pass
"""
import argparse
import json
import sys

REPO = __file__.rsplit("/", 2)[0] if "/" in __file__ else "."
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm import LLM, SamplingParams

PREFIX_LEN = 768          # 3 full blocks
TAIL_LEN = 100
MAX_TOKENS = 16


def make_prompt(rng, tail_seed):
    return ([1000 + (i * 7919 % 50000) for i in range(PREFIX_LEN)]
            + [rng.randrange(50000, 100000) for _ in range(TAIL_LEN)])


def run_prompts(llm, prompts, sp):
    return [list(o["token_ids"]) for o in
            llm.generate(prompts, sp, use_tqdm=False)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    parser.add_argument("--out", default="/root/p30_data/kv_offload_engine.json")
    parser.add_argument("--tp", type=int, default=1,
                        help="tensor parallel size; 2 exercises the rank-"
                             "symmetric offload command stream (slot streams "
                             "must stay identical across ranks)")
    args = parser.parse_args()

    import random
    rng = random.Random(20261110)
    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
        attention_backend="triton_paged",
        max_model_len=2048,
        num_kvcache_blocks=14,      # tight: the junk load must evict
        kv_offload_enabled=True,
        kv_offload_host_blocks=32,
        scheduler_policy="fcfs",
    )
    verdict = {}
    try:
        sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)

        # P30-R1 follow-up gate: raw out-of-vocab ids must raise ValueError
        # (mapped to 400 by the API), never reach the device — an OOB id
        # wedges the MUSA submission path with no traceback (found live in
        # this round while debugging the junk load below).
        vocab = llm.model_runner.config.hf_config.vocab_size
        try:
            llm.add_request([vocab + 5], SamplingParams(max_tokens=1))
            verdict["vocab_gate_raises"] = False
        except ValueError:
            verdict["vocab_gate_raises"] = True
        try:
            llm.add_request([-1, 5], SamplingParams(max_tokens=1))
            verdict["vocab_gate_negative"] = False
        except ValueError:
            verdict["vocab_gate_negative"] = True
        p_prompts = [make_prompt(rng, 1) + [7], make_prompt(rng, 2) + [7]]
        # make both P prompts share the SAME 3-block prefix
        p_prompts = [[t for t in p_prompts[0][:PREFIX_LEN]] + p_prompts[0][PREFIX_LEN:],
                     [t for t in p_prompts[0][:PREFIX_LEN]] + p_prompts[1][PREFIX_LEN:]]

        # --- Arm R: reference (offload idle) ---
        ref = run_prompts(llm, p_prompts, sp)
        bm = llm.scheduler.block_manager
        verdict["armR_puts"] = bm.host_puts
        assert bm.host_puts == 0, "arm R must not touch the host pool"

        # --- Arm X: junk pressure (evict-to-host) then P revival ---
        # ids stay well inside vocab: an out-of-vocab id wedges the MUSA
        # submission path (OOB embedding) and would fake a gate failure.
        junk = [[rng.randrange(20000, 100000) for _ in range(700)]
                for _ in range(8)]
        run_prompts(llm, junk, SamplingParams(temperature=0, max_tokens=4))
        verdict["junk_puts"] = bm.host_puts
        verdict["junk_host_entries"] = len(bm._host_map)
        assert bm.host_puts > 0, "junk load must have evicted cached blocks to host"
        x_finals = run_prompts(llm, p_prompts, sp)
        verdict["revives"] = bm.host_revives
        assert bm.host_revives > 0, "P rerun must have revived from the host"

        # --- Arm R2: junk-free rerun (history disambiguation) ---
        r2 = run_prompts(llm, p_prompts, sp)

        verdict["x_bitwise_equal_ref"] = x_finals == ref
        verdict["r2_bitwise_equal_ref"] = r2 == ref
        m = llm.metrics()
        verdict["metrics_host_entries"] = m["kv_offload_host_entries"]
        verdict["pool_consistent"] = (
            llm.model_runner.kv_offloader.metrics()["kv_offload_slots_in_use"]
            == len(bm._host_map))
        bm.clear_prefix_cache()
        verdict["clear_drains_pool"] = (
            llm.model_runner.kv_offloader.metrics()["kv_offload_slots_in_use"] == 0
            and len(bm._host_map) == 0)
        verdict["prefix_hits_grew"] = m["kv_prefix_hit_tokens_total"] > 0

        # HARD gates: revival correctness + clean accounting + vocab gate.
        verdict["stage_pass"] = (
            verdict["vocab_gate_raises"]
            and verdict["vocab_gate_negative"]
            and verdict["x_bitwise_equal_ref"]
            and verdict["r2_bitwise_equal_ref"]
            and verdict["pool_consistent"]
            and verdict["clear_drains_pool"])
    finally:
        llm.exit()
    print(json.dumps(verdict, indent=1))
    print("[p30-kv-offload-engine]", "PASS" if verdict["stage_pass"] else "FAIL")
    return 0 if verdict["stage_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
