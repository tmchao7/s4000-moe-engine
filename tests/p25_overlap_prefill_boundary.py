"""P25/F1 follow-up: overlap scheduler stale-row regressions (pure CPU).

Covers four defects (C1–C4) found by acceptance audit + code review:

C1  A parked prefill row whose prompt completed in the in-flight collect used
    to be clamped to a 1-token prefill row (num_cached == prompt_len).  When
    prompt_len is an exact multiple of block_size, model_runner._prepare_rows
    then indexes block_table[num_cached // block_size] out of range.  The row
    must be DROPPED, not launched.

C2  The trailing speculative schedule rewrites live seq fields for the NEXT
    batch before the current batch is collected.  postprocess must therefore
    use the batch FROZEN query_lens / prefill_flags; otherwise a 200-token
    prefill row is credited +1 and num_cached drifts.

C3  The I37 cumulative pre-check must be exercised through the real Scheduler
    (fcfs + stall_free): per-seq can_append_at passes while the summed
    shortfall exceeds the free list, and schedule(speculative=True) must
    return mode=defer.

C4  A preempted sequence is re-prefilled from 0 with num_tokens >
    num_prompt_tokens.  A legitimate mid-way re-prefill chunk can cross the
    prompt length; it must NOT be dropped as "prefill complete" (regression
    introduced by the first C1 fix and caught in review).

C5  A prefill completes in the in-flight collect while the seq is stranded in
    waiting (speculative schedule deferred).  The fallback sync schedule must
    not emit it as a q=1 prefill row: batch mode must match the frozen row
    flags, and every row's block_table must cover its query range.

C6  A cancel landing while the seq sits in the PARKED speculative batch
    (_sched_next) must have its row PRUNED at launch (it was dropped, never
    launched with a stale/deallocated block_table).

C7  A cancel landing while the seq sits in the IN-FLIGHT batch (_pending)
    must be SKIPPED at collect: the cancel already deallocated its blocks;
    appending the collected token would re-grow a torn-down seq (it would
    then linger in running forever — queues would never drain).

Runs on the cheapest Python >= 3.10 (same re-exec trick as
tests/p25_review_repro.py).  No torch / GPU / model needed.

Run:  PYTHONPATH=. python3 tests/p25_overlap_prefill_boundary.py   # rc 0 = pass
"""
import importlib.util
import os
import shutil
import sys
import types
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if sys.version_info < (3, 10):
    _py = next((shutil.which(c) for c in ("python3.12", "python3.11", "python3.10")
                if shutil.which(c)), None)
    if _py is None:
        sys.exit("p25_overlap_prefill_boundary: engine modules need Python >= 3.10")
    os.execv(_py, [_py, os.path.abspath(__file__), *sys.argv[1:]])


def stub(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


# The engine modules import torch/transformers/tqdm; only the CPU scheduler
# state machine is under test, so stub the device-facing imports.
torch = stub("torch")
mp = stub("torch.multiprocessing")


class _Ctx:
    def Event(self):
        return object()

    def Process(self, *args, **kwargs):
        raise RuntimeError("this CPU test never spawns workers")


mp.get_context = lambda mode: _Ctx()
torch.multiprocessing = mp
tf = stub("transformers")


class _Tok:
    eos_token_id = -1

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


tf.AutoTokenizer = _Tok
tf.AutoConfig = object
stub("tqdm")
tq = stub("tqdm.auto")


class _P:
    def __init__(self, *args, **kwargs):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass


tq.tqdm = _P
stub("nanovllm.engine.model_runner", ModelRunner=object)


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pkg = types.ModuleType("nanovllm")
pkg.__path__ = [str(REPO / "nanovllm")]
sys.modules["nanovllm"] = pkg
engine_pkg = types.ModuleType("nanovllm.engine")
engine_pkg.__path__ = [str(REPO / "nanovllm" / "engine")]
sys.modules["nanovllm.engine"] = engine_pkg
config_mod = types.ModuleType("nanovllm.config")
config_mod.Config = object
sys.modules["nanovllm.config"] = config_mod

sp = load("nanovllm.sampling_params", "nanovllm/sampling_params.py")
sq = load("nanovllm.engine.sequence", "nanovllm/engine/sequence.py")
load("nanovllm.engine.block_manager", "nanovllm/engine/block_manager.py")
sc_mod = load("nanovllm.engine.scheduler", "nanovllm/engine/scheduler.py")
le_mod = load("nanovllm.engine.llm_engine", "nanovllm/engine/llm_engine.py")

Sequence = sq.Sequence
SequenceStatus = sq.SequenceStatus
Scheduler = sc_mod.Scheduler
SamplingParams = sp.SamplingParams
LLMEngine = le_mod.LLMEngine
BS = 256
Sequence.block_size = BS


class FakeRunner:
    def __init__(self):
        self.rows = 0
        self.launches = []

    def call(self, cmd, *args):
        if cmd in ("run_launch", "run"):
            seqs = args[0]
            mode = args[1]
            self.rows = len(seqs)
            self.launches.append((mode, [
                (s.seq_id, s.is_prefill, s.num_cached_tokens, s.num_tokens,
                 s.num_scheduled_tokens, len(s.block_table)) for s in seqs
            ]))
            return ([1] * self.rows, None) if cmd == "run" else None
        if cmd == "collect":
            return [1] * self.rows, None
        raise AssertionError(cmd)


def make_engine(blocks=64, max_num_seqs=16, chunk=256):
    engine = object.__new__(LLMEngine)
    cfg = SimpleNamespace(
        max_num_seqs=max_num_seqs, max_num_batched_tokens=16384, eos=10,
        kvcache_block_size=BS, scheduler_policy="stall_free",
        prefill_chunk_size=chunk, scheduler_starvation_steps=32,
        num_kvcache_blocks=blocks,
    )
    engine.scheduler = Scheduler(cfg)
    engine.model_runner = FakeRunner()
    engine._overlap = True
    engine._pending = None
    engine._sched_next = None
    engine._pending_launch_time = 0.0
    engine.overlap_defers = 0
    engine.overlap_pruned_rows = 0
    engine.last_token_events = []
    engine.schedule_mode_counts = Counter()
    engine.step_count = 0
    engine.prefill_tokens_total = 0
    engine.decode_tokens_total = 0
    engine.schedule_time_total = 0.0
    engine.forward_time_total = 0.0
    engine.metrics_path = None
    engine.last_step_metrics = None
    return engine


def run_to_finish(engine, max_steps=100):
    steps = 0
    while not engine.is_finished() and steps < max_steps:
        engine.step()
        steps += 1
    return steps


def scenario_c1():
    print("[C1] parked prefill row at block boundary must be dropped")
    bad = []
    for length, chunk in ((512, 256), (768, 256), (1024, 256), (512, 512), (600, 512)):
        engine = make_engine(chunk=chunk)
        seq = Sequence(list(range(2000, 2000 + length)),
                       SamplingParams(max_tokens=4, temperature=0))
        engine.scheduler.add(seq)
        steps = run_to_finish(engine)
        for mode, rows in engine.model_runner.launches:
            for row in rows:
                sid, is_prefill, cached, ntok, scheduled, table_len = row
                if is_prefill and cached >= seq.num_prompt_tokens:
                    bad.append((length, chunk, mode, row))
                if is_prefill and scheduled <= 0:
                    bad.append((length, chunk, mode + ":empty", row))
                if is_prefill and cached // BS >= table_len:
                    bad.append((length, chunk, mode + ":oob", row))
        print("  L=%d chunk=%d: steps=%d launches=%d" %
              (length, chunk, steps, len(engine.model_runner.launches)))
    if bad:
        for item in bad[:5]:
            print("  BAD:", item)
        print("  -> FAIL")
        return False
    print("  -> PASS (no completed-prefill row launched, no out-of-range block index)")
    return True


def scenario_c2():
    print("[C2] postprocess must credit the FROZEN prefill chunk, not live q")
    engine = make_engine()
    long_seq = Sequence(list(range(5000, 5600)), SamplingParams(max_tokens=20, temperature=0))
    engine.scheduler.add(long_seq)
    short_seq = None
    captured = []
    original = Scheduler.postprocess

    def spy(self, seqs, token_ids, mode, logprobs=None, query_lens=None, prefill_flags=None):
        before = [(s.seq_id, s.num_cached_tokens, s.num_tokens, s.num_scheduled_tokens, s.is_prefill)
                  for s in seqs]
        try:
            out = original(self, seqs, token_ids, mode, logprobs, query_lens, prefill_flags)
        except TypeError:
            # pre-fix signature: the spy must still run so the test fails on
            # the assertion instead of erroring out on the call contract.
            out = original(self, seqs, token_ids, mode, logprobs)
        after = [(s.seq_id, s.num_cached_tokens, s.num_tokens) for s in seqs]
        captured.append((mode, before, after))
        return out

    Scheduler.postprocess = spy
    try:
        steps = 0
        while not engine.is_finished() and steps < 100:
            if steps == 2:
                short_seq = Sequence(list(range(9000, 9200)),
                                     SamplingParams(max_tokens=5, temperature=0))
                engine.scheduler.add(short_seq)
            engine.step()
            steps += 1
    finally:
        Scheduler.postprocess = original
    first = None
    for mode, before, after in captured:
        for row_before, row_after in zip(before, after):
            if row_before[0] == short_seq.seq_id and row_before[1] == 0:
                first = (mode, row_before, row_after)
                break
        if first is not None:
            break
    print("  first short-prompt postprocess:", first)
    ok = first is not None and first[2][1] == short_seq.num_prompt_tokens
    print("  -> PASS" if ok else "  -> FAIL")
    return ok


def launch_invariant(engine):
    """Replicate the two _prepare_rows preconditions for every launch."""
    bad = []
    for mode, rows in engine.model_runner.launches:
        flags = [row[1] for row in rows]
        derived = ("mixed" if any(flags) and any(not f for f in flags)
                   else ("prefill" if any(flags) else "decode"))
        if mode != derived:
            bad.append(("mode-mismatch", mode, derived, rows))
        for row in rows:
            sid, is_prefill, cached, ntok, scheduled, table_len = row
            if scheduled > 0 and (cached + scheduled - 1) // BS >= table_len:
                bad.append(("oob", mode, row))
    return bad


def scenario_c5():
    print("[C5] fallback sync schedule must not emit a stranded prefill as q=1 prefill")
    engine = make_engine(blocks=8, chunk=256)
    for i, length in enumerate([512, 600, 768]):
        engine.scheduler.add(Sequence(
            list(range(10000 + i * 2000, 10000 + i * 2000 + length)),
            SamplingParams(max_tokens=20, temperature=0)))
    steps = run_to_finish(engine)
    bad = launch_invariant(engine)
    print("  steps=%d launches=%d bad=%d" %
          (steps, len(engine.model_runner.launches), len(bad)))
    for item in bad[:5]:
        print("  BAD:", item)
    ok = not bad and engine.is_finished() and steps < 200
    print("  -> PASS" if ok else "  -> FAIL")
    return ok


def scenario_c4():
    print("[C4] preempted seq re-prefill crossing prompt length must not be dropped")
    engine = make_engine(blocks=64, chunk=256)
    prompt, total = 200, 600
    seq = Sequence(list(range(2000, 2000 + prompt)),
                   SamplingParams(max_tokens=410, temperature=0))
    # Simulate: 400 tokens generated, then preempted (block_table freed,
    # num_cached reset, is_prefill=True) while num_tokens stays 600.
    seq.token_ids.extend([1] * (total - prompt))
    seq.num_tokens = total
    seq.last_token = 1
    seq.status = SequenceStatus.RUNNING
    engine.scheduler.preempt(seq)
    steps = run_to_finish(engine)
    bad = []
    for mode, rows in engine.model_runner.launches:
        for row in rows:
            sid, is_prefill, cached, ntok, scheduled, table_len = row
            if not is_prefill and scheduled > 1:
                bad.append(("non-prefill q>1", mode, row))
            if is_prefill and cached // BS >= table_len:
                bad.append(("oob", mode, row))
    crossed = [row for mode, rows in engine.model_runner.launches
               for row in rows
               if row[1] and row[2] < prompt <= row[2] + row[4]]
    print("  steps=%d launches=%d bad=%d crossed=%d" %
          (steps, len(engine.model_runner.launches), len(bad), len(crossed)))
    for item in bad[:5]:
        print("  BAD:", item)
    ok = (not bad and crossed and steps < 100 and seq.is_finished)
    print("  -> PASS" if ok else "  -> FAIL")
    return ok


def scenario_c3():
    print("[C3] cumulative scheduler gate must defer when per-seq checks pass")
    all_ok = True
    for policy in ("fcfs", "stall_free"):
        cfg = SimpleNamespace(
            max_num_seqs=4, max_num_batched_tokens=16384, eos=10,
            kvcache_block_size=BS, scheduler_policy=policy,
            prefill_chunk_size=256, scheduler_starvation_steps=32,
            num_kvcache_blocks=5,
        )
        scheduler = Scheduler(cfg)
        seqs = []
        for i in range(2):
            seq = Sequence(list(range(1000 + i * 1000, 1000 + i * 1000 + 512)),
                           SamplingParams(max_tokens=4, temperature=0))
            seq.is_prefill = False
            scheduler.block_manager.allocate(seq, 0)   # 2 blocks each
            scheduler.running.append(seq)
            seqs.append(seq)
        free = len(scheduler.block_manager.free_block_ids)
        per_seq = [scheduler.block_manager.can_append_at(s, len(s) + 1) for s in seqs]
        total = scheduler.block_manager.total_launch_shortfall(
            seqs, [len(s) + 1 for s in seqs])
        batch = scheduler.schedule(speculative=True, inflight=seqs)
        ok = per_seq == [True, True] and total > free and batch.mode == "defer"
        print("  policy=%s free=%d per_seq=%s total=%d batch=%s -> %s" %
              (policy, free, per_seq, total, batch.mode, "PASS" if ok else "FAIL"))
        all_ok &= ok
    print("  -> PASS" if all_ok else "  -> FAIL")
    return all_ok


def scenario_c6_c7():
    """C6/C7: cancel landing inside the overlap pipeline (both windows).

    The async API cancels via scheduler.cancel() between engine steps, so the
    cancelled seq can sit (a) in the IN-FLIGHT batch (_pending): postprocess
    must SKIP it — the cancel already deallocated its blocks and cleared its
    block_table, appending the collected token would re-grow a torn-down seq;
    or (b) in the PARKED speculative batch (_sched_next): the row must be
    PRUNED at launch (I39-style drop), never launched with a stale table.

    No engine-level test exercised either window before P30 (the postprocess
    skip and prune counters existed only by code inspection).

    Validity gates: the cancel must demonstrably hit both defenses
    (overlap_pruned_rows counts dropped parked rows; completion counters of
    cancelled seqs must stay frozen at their cancel-time value), plus the
    shared launch invariants and a clean block ledger at drain.
    """
    print("[C6/C7] cancel x overlap: in-flight skip (C7) + parked-row prune (C6)")
    engine = make_engine(blocks=64, chunk=BS)
    prompts = [list(range(3000 + i * 1000, 3000 + i * 1000 + 768))
               for i in range(4)]
    seqs = [Sequence(p, SamplingParams(max_tokens=8, temperature=0))
            for p in prompts]
    for seq in seqs:
        engine.scheduler.add(seq)

    cancelled = {}
    steps = 0
    parked_only = None
    while not engine.is_finished() and steps < 100:
        if steps == 1:
            # After step 1 the first prefill chunk of every seq is IN FLIGHT
            # (_pending) and the second chunk is PARKED (_sched_next).
            pending_sids = [s.seq_id for s in engine._pending.sequences]
            assert seqs[0].seq_id in pending_sids, (
                f"test setup: seq0 not in flight: {pending_sids}")
            before = seqs[0].num_completion_tokens
            assert engine.scheduler.cancel(seqs[0].seq_id), "cancel seq0 missed"
            cancelled[seqs[0].seq_id] = before          # C7 window
        if steps == 2:
            # A seq added between steps only ever exists in the parked batch
            # until its first launch — the pure C6 window.
            late = Sequence(list(range(9000, 9000 + 768)),
                            SamplingParams(max_tokens=8, temperature=0))
            engine.scheduler.add(late)
        if steps == 3:
            parked = [] if engine._sched_next is None else \
                [s.seq_id for s in engine._sched_next.sequences]
            assert parked_only is None
            if late.seq_id in parked:
                parked_only = late
                assert engine.scheduler.cancel(late.seq_id), "cancel late missed"
                cancelled[late.seq_id] = late.num_completion_tokens
        engine.step()
        steps += 1
    assert parked_only is not None, (
        "test setup: never caught the late seq parked (timing changed)")

    launched_sids = {row[0] for _mode, rows in engine.model_runner.launches
                     for row in rows}
    bad = launch_invariant(engine)
    frozen_ok = all(s.num_completion_tokens == cancelled[s.seq_id]
                    for s in (seqs[0], parked_only))
    ledger_clean = len(engine.scheduler.block_manager.used_block_ids) == 0
    queues_empty = (not engine.scheduler.waiting
                    and not engine.scheduler.running)
    finished_ok = (seqs[0].is_finished and seqs[0].finish_reason == "cancelled"
                   and parked_only.is_finished
                   and parked_only.finish_reason == "cancelled"
                   and all(s.is_finished and s.finish_reason == "length"
                           for s in seqs[1:]))
    ok = (not bad
          and engine.overlap_pruned_rows >= 1
          and parked_only.seq_id not in launched_sids
          and frozen_ok and ledger_clean and queues_empty and finished_ok)
    print("  steps=%d launches=%d pruned_rows=%d bad=%d ledger_clean=%s "
          "queues_empty=%s frozen_ok=%s" %
          (steps, len(engine.model_runner.launches),
           engine.overlap_pruned_rows, len(bad), ledger_clean,
           queues_empty, frozen_ok))
    for item in bad[:5]:
        print("  BAD:", item)
    print("  -> PASS" if ok else "  -> FAIL")
    return ok


def scenario_c8_lpm():
    """C8: stall_free_lpm admission ordering (the policy had ZERO test
    coverage — selectable via Config.scheduler_policy yet never exercised
    by any suite, in the most defect-dense part of the engine).

    Four-order assertion against _waiting_candidate():
      1. lpm picks the MOST-CACHEABLE seq over the FIFO-first one (removing
         the lpm branch would make this pick FIFO order -> FAIL);
      2. the starvation guard outranks cacheability (a starved 0-cache seq
         beats a cacheable one);
      3. priority outranks everything (a priority-5 seq with the biggest
         cache is picked last, behind priority-0 seqs);
      4. within a tier the non-starved remainder falls back to lpm/FIFO.
    Pure CPU, direct Scheduler drive (C3 style): donor seqs hold cached
    prefix blocks so waiting seqs' can_allocate probes hit real hashes.
    """
    print("[C8] stall_free_lpm ordering: cacheable-first / starvation / priority")
    cfg = SimpleNamespace(
        max_num_seqs=8, max_num_batched_tokens=16384, eos=10,
        kvcache_block_size=BS, scheduler_policy="stall_free_lpm",
        prefill_chunk_size=256, scheduler_starvation_steps=32,
        num_kvcache_blocks=32,
    )
    scheduler = Scheduler(cfg)
    bm = scheduler.block_manager

    def add(seq):
        scheduler.add(seq)
        return seq

    # Donors hold cached prefixes so waiting seqs' probes hit real hashes.
    donor_big = Sequence(list(range(2000, 2768)),
                         SamplingParams(max_tokens=4))
    donor_mid = Sequence(list(range(5000, 5256)),
                         SamplingParams(max_tokens=4))
    for donor in (donor_big, donor_mid):
        bm.allocate(donor, 0)
        bm.hash_blocks(donor, len(donor))
        scheduler.running.append(donor)

    tail_big = list(range(7000, 7064))
    tail_mid = list(range(7100, 7164))
    tail_none = list(range(9000, 9064))
    s_none = add(Sequence(list(range(9000, 9768)) + tail_none,
                          SamplingParams(max_tokens=4)))          # pri 0, 0 cached
    s_mid = add(Sequence(list(range(5000, 5256)) + tail_mid,
                         SamplingParams(max_tokens=4)))           # pri 0, 1 cached
    s_big = add(Sequence(list(range(2000, 2768)) + tail_big,
                         SamplingParams(max_tokens=4)))           # pri 0, 3 cached
    s_lowpri = add(Sequence(list(range(2000, 2768)) + list(range(7500, 7564)),
                            SamplingParams(max_tokens=4, priority=5)))  # pri 5, 3 cached
    s_none.wait_steps = 99                   # starved: 0-cache but guard outranks lpm

    # Expected: none first (starvation guard beats cacheability AND FIFO),
    # then big over FIFO-earlier mid (lpm branch load-bearing: without it
    # the within-tier pick would be FIFO -> mid before big -> FAIL),
    # then mid, and finally lowpri whose priority-5 tier is excluded until
    # every priority-0 seq is gone (priority dominates both lpm and cache).
    order = [scheduler._waiting_candidate().seq_id for _ in range(4)]
    expected = [s_none.seq_id, s_big.seq_id, s_mid.seq_id, s_lowpri.seq_id]
    ok = order == expected
    print("  pick order:", order, "expected:", expected)
    assert not scheduler.waiting
    print("  -> PASS" if ok else "  -> FAIL")
    return ok


def main():
    c1 = scenario_c1()
    c2 = scenario_c2()
    c3 = scenario_c3()
    c4 = scenario_c4()
    c5 = scenario_c5()
    c67 = scenario_c6_c7()
    c8 = scenario_c8_lpm()
    print()
    print("VERDICT:", "PASS" if (c1 and c2 and c3 and c4 and c5 and c67 and c8) else "FAIL",
          "(C1=%s, C2=%s, C3=%s, C4=%s, C5=%s, C6/C7=%s, C8=%s)" % (
              c1, c2, c3, c4, c5, c67, c8))
    return 0 if (c1 and c2 and c3 and c4 and c5 and c67 and c8) else 1


if __name__ == "__main__":
    sys.exit(main())
