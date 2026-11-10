"""P30-R1 follow-up: offload's device-wide synchronize under the overlap
scheduler (pre-registered, read-only probe; gates in
benchmarks/P30-kv-offload.md section 6).

Question: `offload_put` calls torch.musa.synchronize() before snapshotting a
block.  In the serial step loop scheduling happens while the GPU is idle, so
the sync is nearly free.  With the P23 overlap scheduler the trailing
speculative schedule runs while the just-launched batch is still on the GPU:
every offload_put/get in that window waits for the in-flight forward.  Is the
end-to-end cost visible enough to justify async double-buffering (the
pre-registered later round), or does blocking v1 stay acceptable?

Arms (one process each; 0.6B TP1 eager triton_paged fcfs):
  overlap_off          overlap ON,  offload OFF            (wall baseline)
  overlap_offload      overlap ON,  offload ON, 64 slots   (target)
  -- the two above are the arms the default interleaved run alternates --
  overlap_offload_zero overlap ON,  offload ON, 0 slots    (manual ablation:
                       put runs its sync + gather but the pool is empty ->
                       -1, no pinned copy; separates bytes from framework
                       cost and exposed the I43 drop-path waste)
  serial_offload       overlap OFF, offload ON, 64 slots   (manual: idle
                       copy-only baseline)

Frozen load (evictions INSIDE the speculative-schedule window):
  KV pool = 36 blocks.  12 requests fill it exactly: 6 "long" prompts of 600
  tokens (3 blocks, last one partial -> first growth at len=769, i.e. decode
  token 169) with max_tokens=200 ignore_eos, and 6 "short" prompts of 768
  tokens with max_tokens=1 (finish after one decode token, leaving 18 hashed
  blocks in the LRU).  Four more 768-token requests are queued from the
  start and wait for KV.  After the shorts finish, the speculative schedule
  admits the 4 waiters (12 blocks) while the long rows' forward is in
  flight; ~169 decode steps later the long rows grow by 6 blocks in another
  speculative schedule.  Both allocation batches are all-eviction and land
  inside the in-flight window.

Pre-registered gates (do not move after running):
  G1 hard   target greedy outputs == overlap_off outputs (every repeat);
            mismatch => INVALID (I31/I34 history caveat), no claim.
  G2 hard   target puts > 0 AND at least one offload call observed with
            engine._pending != None; else INVALID (load missed the window).
  G3 hard   pool accounting closes: slots_in_use == len(host_map).
  G4 decision  median wall penalty from 5 INTERLEAVED rounds (project A/B
            protocol; per round: one fresh engine per arm, arm order
            alternates, second repeat after the in-engine warmup is the
            sample).  A first pass ran the arms in sequential blocks and was
            invalidated by drift -- the 0-slot ablation arm (strictly more
            work) beat the baseline by 7.6%, which is only explicable as GPU
            clock/power drift between blocks.  Thresholds unchanged:
            <= +5%  -> blocking v1 acceptable; async double-buffering NOT
                       justified; trigger stays unmet (record).
             5-10%  -> gray; rerun with more rounds before deciding.
             > +10%  -> justify an opt-in async prototype round.
            Arm CV > 5% => decision INVALID (noise protocol).
  G5 record in-flight vs idle offload-call split (sync vs copy), per-step
            walls, serial-arm copy baseline, eviction counts.

Note: TP2 ranks execute the same sync symmetrically; this probe uses TP1 for
load time.  p30kvo's --tp 2 stage already covers rank symmetry.

Run:  PYTHONPATH=. python3 scripts/p30_offload_overlap_probe.py \
          --model /root/autodl-tmp/models/Qwen3-0.6B       # interleaved, 5 rounds
      PYTHONPATH=. python3 scripts/p30_offload_overlap_probe.py --arm overlap_offload
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

POOL = 36                  # 12 x 3-block prompts fill it exactly
BLOCK = 256
LONG_PROMPT = 600          # 3 blocks, partial last -> growth at len=769
SHORT_PROMPT = 768         # 3 full blocks
N_LONG = 6                 # decode 200 tokens, cross the boundary in-window
N_SHORT = 6                # max_tokens=1 -> 18 hashed LRU blocks early
N_WAIT = 4                 # admitted by the speculative schedule
N_REQ = N_LONG + N_SHORT + N_WAIT
LONG_TOKENS = 200
MAX_STEPS = 2000

# torch.musa.synchronize() timings collected while an offload call is active.
SYNC_STATE = {"active": False, "buf": []}


def install_sync_probe():
    import torch
    original = torch.musa.synchronize

    def probed(*args, **kwargs):
        t0 = time.perf_counter()
        result = original(*args, **kwargs)
        if SYNC_STATE["active"]:
            SYNC_STATE["buf"].append(time.perf_counter() - t0)
        return result

    torch.musa.synchronize = probed


def make_prompts():
    prompts = []
    for i in range(N_REQ):
        start = 1000 + i * 1000
        length = LONG_PROMPT if i < N_LONG else SHORT_PROMPT
        prompts.append(list(range(start, start + length)))
    return prompts


def build_llm(args, overlap: bool, offload: bool, host_blocks: int):
    from nanovllm import LLM
    return LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        attention_backend="triton_paged",
        max_model_len=1024,
        num_kvcache_blocks=POOL,
        scheduler_policy="fcfs",
        scheduler_overlap=overlap,
        kv_offload_enabled=offload,
        kv_offload_host_blocks=host_blocks,
    )


def instrument(llm, records):
    """Time every offload_* broadcast and tag whether a batch was in flight.

    sync_dt separates the explicit device-wide synchronize inside
    offload_put from the rest (gather + hosted copy), so a big in-flight sync
    is attributable rather than hidden in one number."""
    original = llm.model_runner.call

    def timed(name, *call_args, **call_kwargs):
        if not name.startswith("offload_"):
            return original(name, *call_args, **call_kwargs)
        in_flight = llm._pending is not None
        SYNC_STATE["buf"] = []
        SYNC_STATE["active"] = True
        t0 = time.perf_counter()
        try:
            result = original(name, *call_args, **call_kwargs)
        finally:
            SYNC_STATE["active"] = False
        dt = time.perf_counter() - t0
        records.append({
            "name": name,
            "in_flight": in_flight,
            "dt": dt,
            "sync_dt": sum(SYNC_STATE["buf"]),
        })
        return result

    llm.model_runner.call = timed


def p50_ms(values):
    return round(1000 * statistics.median(values), 3) if values else None


def p90_ms(values):
    if not values:
        return None
    ordered = sorted(values)
    return round(1000 * ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], 3)


def summarize(bm, runner, records, rec_start, wall, per_step, outputs):
    calls = records[rec_start:]
    puts = [r for r in calls if r["name"] == "offload_put"]
    put_inflight = [r for r in puts if r["in_flight"]]
    put_idle = [r for r in puts if not r["in_flight"]]
    gets_inflight = [r for r in calls
                     if r["name"] == "offload_get" and r["in_flight"]]
    pool = runner.kv_offloader.metrics() if runner.kv_offloader else None
    return {
        "wall_s": round(wall, 4),
        "steps": len(per_step),
        "step_p50_ms": p50_ms([s["wall_s"] for s in per_step]),
        "step_p90_ms": p90_ms([s["wall_s"] for s in per_step]),
        "step_mean_ms": round(
            1000 * statistics.mean([s["wall_s"] for s in per_step]), 3),
        "step_with_put_p50_ms": p50_ms([s["wall_s"] for s in per_step if s["puts"]]),
        "step_no_put_p50_ms": p50_ms([s["wall_s"] for s in per_step if not s["puts"]]),
        "puts": bm.host_puts,
        "drops": bm.host_drops,
        "revives": bm.host_revives,
        "put_calls": len(puts),
        "put_inflight_calls": len(put_inflight),
        "put_p50_ms": p50_ms([r["dt"] for r in puts]),
        "put_sync_p50_ms": p50_ms([r["sync_dt"] for r in puts]),
        "put_sync_p90_ms": p90_ms([r["sync_dt"] for r in puts]),
        "put_sync_max_ms": round(1000 * max(r["sync_dt"] for r in puts), 3) if puts else None,
        "put_inflight_p50_ms": p50_ms([r["dt"] for r in put_inflight]),
        "put_inflight_sync_max_ms": (
            round(1000 * max(r["sync_dt"] for r in put_inflight), 3)
            if put_inflight else None),
        "put_idle_p50_ms": p50_ms([r["dt"] for r in put_idle]),
        "get_inflight_calls": len(gets_inflight),
        "get_inflight_sync_max_ms": (
            round(1000 * max(r["sync_dt"] for r in gets_inflight), 3)
            if gets_inflight else None),
        "offload_total_ms": round(1000 * sum(r["dt"] for r in calls), 2),
        "offload_sync_total_ms": round(1000 * sum(r["sync_dt"] for r in calls), 2),
        "pool_in_use": pool["kv_offload_slots_in_use"] if pool else 0,
        "host_entries": len(bm._host_map),
        "outputs": outputs,
    }


def run_repeat(llm, prompts, records):
    import torch
    from nanovllm import SamplingParams
    bm = llm.scheduler.block_manager
    # Cold-cache shape for every repeat: drop hashes and drain host slots.
    bm.clear_prefix_cache()
    torch.musa.synchronize()
    base_puts, base_drops, base_revives = bm.host_puts, bm.host_drops, bm.host_revives
    rec_start = len(records)
    for i, prompt in enumerate(prompts):
        llm.add_request(prompt, SamplingParams(
            temperature=0,
            max_tokens=LONG_TOKENS if i < N_LONG else 1,
            ignore_eos=True,
        ))
    outputs = {}
    per_step = []
    steps = 0
    t_all = time.perf_counter()
    while not llm.is_finished():
        step_rec_start = len(records)
        puts_before = bm.host_puts
        t0 = time.perf_counter()
        out, _ = llm.step()
        per_step.append({
            "wall_s": round(time.perf_counter() - t0, 6),
            "puts": bm.host_puts - puts_before,
            "offload_calls": len(records) - step_rec_start,
            "inflight_calls": sum(1 for r in records[step_rec_start:] if r["in_flight"]),
            "sync_ms": round(
                1000 * sum(r["sync_dt"] for r in records[step_rec_start:]), 3),
        })
        for seq_id, token_ids, _metrics in out:
            outputs[seq_id] = list(token_ids)
        steps += 1
        if steps > MAX_STEPS:
            raise RuntimeError(f"probe did not finish in {MAX_STEPS} steps")
    wall = time.perf_counter() - t_all
    assert llm._pending is None and llm._sched_next is None, (
        "pipeline not drained after is_finished()")
    return summarize(bm, llm.model_runner, records, rec_start, wall, per_step,
                     outputs), {
        "puts": bm.host_puts - base_puts,
        "drops": bm.host_drops - base_drops,
        "revives": bm.host_revives - base_revives,
    }


ARM_SPECS = {
    # arm -> (overlap, offload, host_blocks)
    "overlap_off": (True, False, 0),
    "overlap_offload": (True, True, 64),
    "overlap_offload_zero": (True, True, 0),
    "serial_offload": (False, True, 64),
}


def stage_arm(args, arm):
    overlap, offload, host_blocks = ARM_SPECS[arm]
    llm = build_llm(args, overlap, offload, host_blocks)
    records = []
    instrument(llm, records)
    prompts = make_prompts()
    repeats = []
    try:
        for rep in range(args.repeats):
            summary, deltas = run_repeat(llm, prompts, records)
            summary.update(deltas)
            summary["repeat"] = rep
            repeats.append(summary)
            print(f"[{arm}] rep{rep} wall={summary['wall_s']}s "
                  f"steps={summary['steps']} puts={summary['puts']} "
                  f"inflight_puts={summary['put_inflight_calls']} "
                  f"step_p50={summary['step_p50_ms']}ms "
                  f"sync_max={summary['put_sync_max_ms']}ms", flush=True)
    finally:
        try:
            llm.exit()
        except Exception as exc:      # keep the stage JSON on teardown noise
            print(f"[{arm}] exit warning: {exc}", flush=True)
    stage = {"arm": arm, "repeats": repeats}
    print("STAGE_JSON " + json.dumps(stage))
    return stage


def _median(values):
    return statistics.median(values)


def _cv(values):
    mean = statistics.mean(values)
    return statistics.pstdev(values) / mean if mean else float("inf")


ARM_ORDER = ["overlap_off", "overlap_offload", "overlap_offload_zero",
             "serial_offload"]


def verdict_from(walls_base, walls_target, g1, g2, g3, extra=None):
    med_base, med_target = _median(walls_base), _median(walls_target)
    penalty = med_target / med_base - 1.0
    cv_base, cv_target = _cv(walls_base), _cv(walls_target)
    decision_valid = cv_base <= 0.05 and cv_target <= 0.05
    if not decision_valid:
        decision = "INVALID (arm CV > 5%; rerun with more repeats)"
    elif penalty <= 0.05:
        decision = "NOT_JUSTIFIED (<= +5%: keep blocking v1; async trigger unmet)"
    elif penalty <= 0.10:
        decision = "GRAY (5-10%: rerun with more repeats)"
    else:
        decision = "PROTOTYPE (> +10%: justify opt-in async double-buffer)"
    verdict = {
        "g1_outputs_equal": g1,
        "g2_window_exercised": g2,
        "g3_pool_consistent": g3,
        "walls_base": walls_base,
        "walls_target": walls_target,
        "median_wall_base_s": med_base,
        "median_wall_target_s": med_target,
        "penalty_ratio": round(penalty, 4),
        "cv_base": round(cv_base, 4),
        "cv_target": round(cv_target, 4),
        "decision": decision,
        "stage_pass": bool(g1 and g2 and g3),
    }
    verdict.update(extra or {})
    return verdict


def run_interleaved(args):
    """Interleaved A/B (the project protocol): sequential arm blocks drifted
    with GPU clock/power state -- the 0-slot ablation beat the baseline by
    7.6% while doing strictly more work, which is only explicable as drift.
    Alternate the arm order every round and take the second (post-warmup)
    repeat of each fresh engine as the sample."""
    arms = ["overlap_off", "overlap_offload"]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    walls = {arm: [] for arm in arms}
    outputs = {arm: [] for arm in arms}
    stages = {arm: [] for arm in arms}
    for rnd in range(args.rounds):
        order = arms if rnd % 2 == 0 else list(reversed(arms))
        for arm in order:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--arm", arm,
                 "--model", args.model, "--repeats", "2"],
                env=env, capture_output=True, text=True, timeout=1200)
            if proc.returncode != 0:
                print(f"[probe] round {rnd} arm {arm} rc={proc.returncode} "
                      f"stderr_tail={proc.stderr[-600:] if proc.stderr else ''}",
                      flush=True)
                return 1
            stage = json.loads(next(
                line for line in proc.stdout.splitlines()
                if line.startswith("STAGE_JSON ")).split(" ", 1)[1])
            timed = stage["repeats"][-1]
            walls[arm].append(timed["wall_s"])
            outputs[arm].append(timed["outputs"])
            stages[arm].append(timed)
            print(f"[round {rnd}] {arm}: wall={timed['wall_s']}s "
                  f"steps={timed['steps']} puts={timed['puts']} "
                  f"inflight_puts={timed['put_inflight_calls']} "
                  f"sync_max={timed['put_sync_max_ms']}ms", flush=True)

    g1 = all(a == b for a, b in zip(outputs["overlap_off"],
                                    outputs["overlap_offload"]))
    target = stages["overlap_offload"]
    g2 = (sum(s["puts"] for s in target) > 0
          and sum(s["put_inflight_calls"] for s in target) > 0)
    last = target[-1]
    g3 = last["pool_in_use"] == last["host_entries"]
    last_base = stages["overlap_off"][-1]
    verdict = verdict_from(
        walls["overlap_off"], walls["overlap_offload"], g1, g2, g3,
        extra={
            "mode": "interleaved",
            "rounds": args.rounds,
            "put_calls_target": last["put_calls"],
            "put_inflight_calls_target": last["put_inflight_calls"],
            "put_p50_ms_target": last["put_p50_ms"],
            "put_sync_p50_ms_target": last["put_sync_p50_ms"],
            "put_sync_p90_ms_target": last["put_sync_p90_ms"],
            "put_sync_max_ms_target": last["put_sync_max_ms"],
            "put_inflight_sync_max_ms_target":
                last["put_inflight_sync_max_ms"],
            "put_p50_ms_serial_copy_baseline":
                stages["overlap_off"][-1].get("put_p50_ms"),
            "step_with_put_p50_ms_target": last["step_with_put_p50_ms"],
            "step_no_put_p50_ms_target": last["step_no_put_p50_ms"],
            "offload_total_ms_target": last["offload_total_ms"],
            "offload_sync_total_ms_target": last["offload_sync_total_ms"],
            "steps_target": last["steps"],
            "steps_base": last_base["steps"],
            "base_cv": round(_cv(walls["overlap_off"]), 4),
        })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump({"stages": stages, "verdict": verdict}, stream, indent=1)
    print(json.dumps(verdict, indent=1))
    badge = "PASS" if verdict["stage_pass"] else "FAIL"
    print(f"[p30-offload-overlap] {badge} | decision: {verdict['decision']}")
    return 0 if verdict["stage_pass"] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    parser.add_argument("--arm", choices=ARM_ORDER)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5,
                        help="interleaved rounds (each = one fresh engine per arm)")
    parser.add_argument("--out", default="/root/p30_data/offload_overlap_probe.json")
    args = parser.parse_args()

    if args.arm is not None:
        install_sync_probe()
        stage_arm(args, args.arm)
        return 0

    return run_interleaved(args)

if __name__ == "__main__":
    sys.exit(main())
