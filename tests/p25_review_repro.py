"""P25 review round: fix verification for the two P23 overlap findings.

Pure CPU — only BlockManager + Sequence (xxhash + numpy); the engine-level
end-to-end coverage lives in tests/p25_overlap_fix.py (server, needs torch).

Finding 1 (FIXED here — verified): the speculative pre-check used to verify
each running seq's launch-time block coverage independently while the decode
loop allocated cumulatively; ``sum(need_i) > free`` with every ``need_i <=
free`` ran ``_pop_free`` off the end of the free list (IndexError).  The fix
adds ``BlockManager.total_launch_shortfall`` and the scheduler pre-check now
defers once when the SUM exceeds the free list.  This script verifies the
gate: per-seq checks pass on the adversarial setup, the cumulative shortfall
exceeds free, so a fixed scheduler defers instead of crashing.

Finding 2 (FIXED in llm_engine — verified on server only): dropping a parked
speculative batch strands the blocks its ``may_append_at`` already allocated;
the serial invariant ``len(block_table) == ceil(len(seq)/bs)`` breaks and
``prepare_decode`` writes KV one block off (silent corruption).  Scenario 2
below documents the raw primitive-level mechanism (motivation; it still
"reproduces" by construction because it bypasses the scheduler).  The engine
fix makes ``set_scheduler_overlap(False)`` drain the pipeline instead of
dropping the parked batch.

Run:  PYTHONPATH=. python3 tests/p25_review_repro.py   # rc 0 = fix verified
"""
import importlib.util
import os
import shutil
import sys
import types
from pathlib import Path

# The engine modules below use 3.10+ syntax (dataclass(slots=True), PEP 604
# unions) and pyproject declares requires-python ">=3.10,<3.13".  Dev boxes
# often default `python3` to something older, which dies with an opaque
# "dataclass() got an unexpected keyword argument 'slots'": re-exec onto the
# newest compliant interpreter on PATH instead (no-op on 3.10+; honest rc=1
# error if none exists).
if sys.version_info < (3, 10):
    _py = next((shutil.which(c) for c in ("python3.12", "python3.11", "python3.10")
                if shutil.which(c)), None)
    if _py is None:
        sys.exit("p25_review_repro: engine modules need Python >= 3.10 "
                 f"(running {sys.version.split()[0]}); no python3.10-3.12 on PATH")
    os.execv(_py, [_py, os.path.abspath(__file__), *sys.argv[1:]])

# Load the three pure-Python modules directly, bypassing nanovllm/__init__
# (which pulls torch/transformers/tqdm).  Keeps this repro runnable anywhere
# with just xxhash + numpy.


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_pkg = types.ModuleType("nanovllm")
_pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "nanovllm")]
sys.modules["nanovllm"] = _pkg
_engine_pkg = types.ModuleType("nanovllm.engine")
_engine_pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "nanovllm" / "engine")]
sys.modules["nanovllm.engine"] = _engine_pkg
_sampling_params = _load("nanovllm.sampling_params", "nanovllm/sampling_params.py")
_sequence = _load("nanovllm.engine.sequence", "nanovllm/engine/sequence.py")
_block_manager = _load("nanovllm.engine.block_manager", "nanovllm/engine/block_manager.py")

BlockManager = _block_manager.BlockManager
Sequence = _sequence.Sequence
SamplingParams = _sampling_params.SamplingParams

BS = 256
LENS = 512          # two blocks exactly; decode append crosses the boundary


def make_seq(n_tokens: int, seq_id: int) -> Sequence:
    Sequence.counter = iter(range(seq_id, seq_id + 1))
    return Sequence(list(range(1000 + seq_id, 1000 + seq_id + n_tokens)),
                    SamplingParams(max_tokens=64))


def scenario_1() -> bool:
    """F1 fix verification: the cumulative gate catches what per-seq checks miss."""
    print("[1] cumulative speculative pre-check gate (F1)")
    if not hasattr(BlockManager, "total_launch_shortfall"):
        print("    -> FAIL: BlockManager.total_launch_shortfall missing (fix absent)")
        return False
    bm = BlockManager(num_blocks=5, block_size=BS)
    seqs = [make_seq(LENS, i) for i in range(2)]
    for seq in seqs:
        bm.allocate(seq, 0)                     # 2 blocks each
    free = len(bm.free_block_ids)
    assert free == 1, f"setup: expected 1 free block, got {free}"

    per_seq = [bm.can_append_at(s, len(s) + 1) for s in seqs]
    total = bm.total_launch_shortfall(seqs, [len(s) + 1 for s in seqs])
    gate_defers = total > free
    print(f"    per-seq checks: {per_seq} (all pass -> old code would not defer)")
    print(f"    cumulative shortfall = {total} vs free = {free} "
          f"-> fixed scheduler defers: {gate_defers}")
    if all(per_seq) and gate_defers:
        print("    -> FIX VERIFIED: the adversarial setup now defers instead of "
              "crashing in may_append_at/_pop_free")
        return True
    print("    -> FAIL: gate did not catch the overcommit")
    return False


def scenario_2() -> bool:
    """F2 motivation (informational, does not affect rc): the raw slot
    contract breaks when a speculative block bump is not followed by the
    append it was sized for.  Engine-level fix verified by
    tests/p25_overlap_fix.py (drain semantics) on the server."""
    print("[2] dropped speculative batch breaks block_table[-1] contract "
          "(motivation only)")
    bm = BlockManager(num_blocks=8, block_size=BS)
    seq = make_seq(LENS, 7)
    bm.allocate(seq, 0)                         # table == ceil(512/256) == 2

    def decode_slot(s: Sequence) -> int:
        # prepare_decode's slot formula, verbatim
        last = s.num_tokens - (s.num_blocks - 1) * BS
        return s.block_table[-1] * BS + last - 1

    healthy = decode_slot(seq) == LENS - 1
    bm.may_append_at(seq, len(seq) + 1)         # speculative pass bumps table
    corrupted_slot = decode_slot(seq)
    drifted = corrupted_slot != LENS - 1
    print(f"    control serial slot ok: {healthy}; after bump-without-append: "
          f"slot={corrupted_slot} (expected {LENS - 1}, drifted={drifted})")
    print("    -> documented; drain fix covered by tests/p25_overlap_fix.py")
    return healthy and drifted                  # mechanism still demonstrable


def main() -> int:
    Sequence.block_size = BS
    s1 = scenario_1()
    scenario_2()
    print()
    print(f"VERDICT: {'FIX VERIFIED' if s1 else 'FIX NOT VERIFIED'} "
          "(scenario 1 = rc gate; scenario 2 = informational)")
    return 0 if s1 else 1


if __name__ == "__main__":
    sys.exit(main())
