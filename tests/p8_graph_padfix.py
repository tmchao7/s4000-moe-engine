"""Graph padding-slot fix verification (P2.2).

The replay drift hypothesis: the captured graph executes the whole bucket, and
the replay path used to point padding rows' slot_mapping at slot 0, overwriting
the first KV slot of whichever request owns block 0.  The fix reserves one
scratch block (invisible to the BlockManager) and points padding rows'
slot_mapping and block_tables at it.

Gates (frozen in docs/NIGHTLY-LOG.md "P2.2"):
  GA  slot-0 readback: KV slot 0 of every layer bitwise unchanged across
      replay-only decode steps (eager baseline: also unchanged)
  GB  5/5: five consecutive graph generates produce tokens identical to the
      eager reference (was 0/5 before the fix)
  GC  shared prefix: two requests sharing a 300-token prefix produce identical
      tokens under graph and eager (block 0 is shared and must not be clobbered)
  GD  capture failures counted separately from replay drift

Run:
  PYTHONPATH=. python3 tests/p8_graph_padfix.py                 # tiny MoE
  PYTHONPATH=. python3 tests/p8_graph_padfix.py --model /root/models/Qwen3-8B --dense
"""
import argparse
import json
import sys
import time

import torch

sys.path.insert(0, "/root/moe-server-s4000")

TINY_MODEL = "/tmp/tiny_moe"
PROMPTS = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12], [42, 42, 42]]
SHARED_PREFIX_LEN = 300
SUFFIX_LEN = 10


def build_llm(args, enforce_eager):
    from nanovllm import LLM

    kwargs = {}
    if not args.dense:
        kwargs = {"moe_backend": "grouped_mm", "decode_moe_backend": args.decode_moe_backend}
    return LLM(
        args.model,
        enforce_eager=enforce_eager,
        tensor_parallel_size=args.tp,
        expert_parallel_size=args.tp,
        attention_backend=args.attention_backend,
        max_model_len=args.max_model_len,
        max_num_seqs=8,
        gpu_memory_utilization=args.gpu_mem,
        **kwargs,
    )


def exit_llm(llm):
    """Exit the engine and release its cached segments to the driver, so the
    next engine in this process sees real free memory in mem_get_info."""
    llm.exit()
    import gc

    gc.collect()
    from nanovllm.utils.compat import accelerator

    accelerator().empty_cache()


def slot0_snapshot(runner, probe_block=None):
    """K/V of the first token's slot for every layer -> [2, L, kh, d] cpu.

    The device sync is load-bearing: a bare .cpu() on MUSA can read the slot
    before the prefill's KV writes have landed (observed as NaN snapshots)."""
    block = 0 if probe_block is None else probe_block
    from nanovllm.utils.compat import accelerator

    accelerator().synchronize()
    # torch.musa D2H of the strided [2, L, kh, d] view returned garbage (NaN)
    # while K-only/V-only and device-side reads were clean; materialise a
    # contiguous device tensor before the copy.
    return runner.kv_cache[:, :, block, 0].contiguous().cpu().clone()


def slot0_probe(llm, max_tokens=4):
    """Run prompts to decode, then replay-only steps, snapshotting the first
    token's KV slot of the FIRST scheduled sequence (its actual block -- block
    0 is not guaranteed to be in use after allocate/free cycles).

    The first token's slot is written only at prefill; decode appends write
    later slots.  Any bitwise change between snapshots is replay padding-row
    damage on a live block.
    """
    runner = llm.model_runner
    from nanovllm import SamplingParams

    for prompt in PROMPTS:
        llm.add_request(prompt, SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True))
    first_decode = False
    reference = None
    probe_block = None
    snapshots = []
    while not llm.is_finished():
        llm.step()
        running = llm.scheduler.running
        if llm.last_step_metrics["mode"] == "decode":
            if not first_decode:
                # Reference right after the first decode step; the probe block
                # is the first sequence's first block (written at prefill, and
                # only decode appends touch later slots afterwards).
                probe_block = running[0].block_table[0] if running else None
                reference = slot0_snapshot(runner, probe_block)
                first_decode = True
            else:
                snapshots.append(slot0_snapshot(runner, probe_block))
    changed = [not torch.equal(reference, snapshot) for snapshot in snapshots]
    nan_reference = bool(reference.isnan().any()) if reference is not None else None
    max_abs = [
        float((reference.float() - snapshot.float()).abs().max())
        if reference is not None and not reference.isnan().any() else None
        for snapshot in snapshots
    ]
    return {
        "probe_block": probe_block,
        "reference_has_nan": nan_reference,
        "replay_steps_after_reference": len(snapshots),
        "slot0_changed": changed,
        "slot0_max_abs": max_abs,
        "any_change": any(changed),
    }


def collect_tokens(llm, prompts, max_tokens):
    from nanovllm import SamplingParams

    outputs = llm.generate(
        prompts, SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True),
        use_tqdm=False,
    )
    return [output["token_ids"] for output in outputs]


def shared_pair(vocab, seed=20261028):
    import random

    generator = random.Random(seed)
    prefix = [generator.randrange(vocab) for _ in range(SHARED_PREFIX_LEN)]
    suffix_a = [generator.randrange(vocab) for _ in range(SUFFIX_LEN)]
    suffix_b = [generator.randrange(vocab) for _ in range(SUFFIX_LEN)]
    return [prefix + suffix_a, prefix + suffix_b]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--model", default=TINY_MODEL)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--decode-moe-backend", default="bmm_dense")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--generates", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-mem", type=float, default=0.3)
    parser.add_argument("--output", default="/root/p8_graph_padfix.json")
    parser.add_argument(
        "--phase", choices=["eager", "graph", "both"], default="both",
        help="eager: reference only, write JSON; graph: compare against "
             "--eager-from JSON (two-process split, sidesteps the two-engine "
             "illegal access); both: legacy single-process flow",
    )
    parser.add_argument("--eager-from", default=None,
                        help="eager-phase JSON to compare against (--phase graph)")
    parser.add_argument("--attempts", type=int, default=2,
                        help="graph-engine build attempts (async capture flake retry)")
    args = parser.parse_args()

    from nanovllm import SamplingParams

    payload = {
        "schema_version": 2,
        "kind": "graph_padfix",
        "model": args.model,
        "phase": args.phase,
        "gates_frozen": "benchmarks/P11-graph.md P0-S2 (GA-GD)",
        "started": time.time(),
    }
    vocab = 128 if args.model == TINY_MODEL else 151936

    eager_reference = shared_eager_tokens = None

    if args.phase in ("eager", "both"):
        # Phase 1: eager reference.
        print("[padfix] phase 1: eager reference", flush=True)
        llm = build_llm(args, enforce_eager=True)
        eager_reference = collect_tokens(llm, PROMPTS, args.max_tokens)
        payload["eager_probe"] = slot0_probe(llm, max_tokens=4)
        shared_eager_tokens = collect_tokens(llm, shared_pair(vocab), args.max_tokens)
        exit_llm(llm)
        payload["eager_reference"] = eager_reference
        payload["shared_eager_tokens"] = shared_eager_tokens
        if args.phase == "eager":
            payload["finished"] = time.time()
            with open(args.output, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, default=str)
            print(f"[padfix] eager phase written to {args.output}", flush=True)
            return

    if args.phase == "graph":
        if not args.eager_from:
            raise SystemExit("phase graph requires --eager-from <eager.json>")
        with open(args.eager_from, encoding="utf-8") as stream:
            reference_payload = json.load(stream)
        eager_reference = reference_payload["eager_reference"]
        shared_eager_tokens = reference_payload["shared_eager_tokens"]
        payload["eager_from"] = args.eager_from
        payload["eager_probe_reference"] = reference_payload.get("eager_probe")

    # Phase 2: graph engine (capture failures counted separately).
    print("[padfix] phase 2: graph engine", flush=True)
    llm = None
    for attempt in range(1, args.attempts + 1):
        try:
            llm = build_llm(args, enforce_eager=False)
            break
        except Exception as exc:  # noqa: BLE001
            payload.setdefault("capture_failures", []).append(
                f"attempt {attempt}: {type(exc).__name__}: {exc}"
            )
            print(f"[padfix] capture failure attempt {attempt}: {exc}", flush=True)
            # A failed init leaks the default process group; release it so the
            # retry (or the next engine in this process) can re-initialise.
            try:
                import torch.distributed as _dist

                if _dist.is_initialized():
                    _dist.destroy_process_group()
            except Exception:  # noqa: BLE001
                pass
    if llm is None:
        payload["verdict"] = {
            "GD_capture": f"FAIL (all {args.attempts} attempts)",
            "overall": "FAIL",
        }
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, default=str)
        return

    runner = llm.model_runner
    config = runner.config
    payload["scratch"] = {
        "scratch_block_id": getattr(runner, "scratch_block_id", None),
        "scratch_slot": getattr(runner, "scratch_slot", None),
        "kv_blocks_in_tensor": runner.kv_cache.shape[2],
        "blocks_in_manager": config.num_kvcache_blocks,
    }
    assert runner.kv_cache.shape[2] == config.num_kvcache_blocks + 1, "scratch block missing"

    graph_runs = []
    for index in range(args.generates):
        graph_runs.append(collect_tokens(llm, PROMPTS, args.max_tokens))
        print(f"[padfix] graph generate {index + 1}/{args.generates}: {graph_runs[-1]}", flush=True)
    payload["graph_runs_match_eager"] = [run == eager_reference for run in graph_runs]
    payload["graph_runs_self_consistent"] = [run == graph_runs[0] for run in graph_runs]
    payload["eager_reference"] = eager_reference
    payload["graph_first_run"] = graph_runs[0]

    payload["graph_probe"] = slot0_probe(llm, max_tokens=4)

    # GC: shared prefix (block 0 reused across two live requests), graph side.
    shared_graph_tokens = collect_tokens(llm, shared_pair(vocab), args.max_tokens)
    exit_llm(llm)

    payload["shared_prefix"] = {
        "eager": shared_eager_tokens,
        "graph": shared_graph_tokens,
        "identical": shared_eager_tokens == shared_graph_tokens,
    }

    ga = not payload["graph_probe"]["any_change"]
    gb = all(payload["graph_runs_match_eager"]) and len(graph_runs) == args.generates
    gc = payload["shared_prefix"]["identical"]
    gd = len(payload.get("capture_failures", []))
    payload["verdict"] = {
        "GA_slot0_readback": "PASS" if ga else "FAIL",
        "GB_5x_match_eager": "PASS" if gb else "FAIL",
        "GC_shared_prefix": "PASS" if gc else "FAIL",
        "GD_capture_failures": gd,
        "overall": "PASS" if (ga and gb and gc) else "FAIL",
    }
    payload["finished"] = time.time()
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=str)
    print(f"[padfix] verdict={json.dumps(payload['verdict'])}", flush=True)


if __name__ == "__main__":
    main()
