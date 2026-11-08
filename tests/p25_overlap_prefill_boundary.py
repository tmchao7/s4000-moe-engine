"""P25/F1 follow-up: overlap scheduler stale-row regressions (pure CPU).

Covers two defects found by the P26+ acceptance audit:

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

def main():
    c1 = scenario_c1()
    c2 = scenario_c2()
    c3 = scenario_c3()
    print()
    print("VERDICT:", "PASS" if (c1 and c2 and c3) else "FAIL",
          "(C1=%s, C2=%s, C3=%s)" % (c1, c2, c3))
    return 0 if (c1 and c2 and c3) else 1


if __name__ == "__main__":
    sys.exit(main())
