"""P30 review gate: KV-offload host-map slot invariant (pure CPU).

Found by post-closeout self-review of the P30-R1 wiring: a *dropped* eviction
(full host pool) deletes hash_to_block_id[h] but leaves no host entry, so the
device hash chain is broken.  A later request sharing that prefix gets
num_cached_blocks == 0, recomputes the block, and hash_blocks registers it
device-side while the block's STALE host entry is still mapped.  When that new
device copy is later evicted, offload_put succeeds into a slot freed by some
other revive and the map assignment in _allocate_block overwrote the entry:
the old slot stayed in the pool's in_use dict forever (in_use > host_map,
clear_prefix_cache could not drain it) -- a silent host-capacity leak.

The fix keeps host_map and hash_to_block_id exclusive:
  - hash_blocks releases a stale host entry before registering the device
    block (the device copy supersedes the host copy);
  - _allocate_block releases a stale entry before overwriting (defensive);
  - clear_prefix_cache goes through the same helper.

This test drives the real BlockManager with a fake executor that mirrors
HostBlockPool's slot accounting (put -> slot / -1 when full, release frees the
slot; no bytes are moved).  It asserts the invariant len(in_use) ==
len(host_map) at EVERY step and that clear drains the pool.  Mutation check:
drop the hash_blocks release and the "A recompute" / "clear" assertions fail
(in_use=2 vs host_map=1, then 1 vs 0).

Runs on the cheapest Python >= 3.10 (same re-exec trick as
tests/p25_overlap_prefill_boundary.py).  No torch / GPU / model needed.

Run:  PYTHONPATH=. python3 tests/p30_kv_offload_hostmap.py   # rc 0 = pass
"""
import importlib.util
import os
import shutil
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if sys.version_info < (3, 10):
    _py = next((shutil.which(c) for c in ("python3.12", "python3.11", "python3.10")
                if shutil.which(c)), None)
    if _py is None:
        sys.exit("p30_kv_offload_hostmap: engine modules need Python >= 3.10")
    os.execv(_py, [_py, os.path.abspath(__file__), *sys.argv[1:]])


def stub(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


# block_manager only needs xxhash/numpy/Sequence, but importing the package
# would pull the device-facing tree — stub exactly what the boundary suite does.
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
bm_mod = load("nanovllm.engine.block_manager", "nanovllm/engine/block_manager.py")
Sequence = sq.Sequence
SamplingParams = sp.SamplingParams


class FakePool:
    """HostBlockPool slot accounting without the bytes."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.free = list(range(capacity - 1, -1, -1))
        self.in_use = {}
        self.puts = self.gets = self.releases = self.full = 0

    def call(self, name, *args):
        if name == "offload_put":
            if not self.free:
                self.full += 1
                return -1
            slot = self.free.pop()
            self.in_use[slot] = args[0]
            self.puts += 1
            return slot
        if name == "offload_get":
            slot, block_id = args
            assert slot in self.in_use, f"get on free slot {slot}"
            self.in_use[slot] = block_id
            self.gets += 1
            return None
        if name == "offload_release":
            slot = args[0]
            assert slot in self.in_use, f"release on free slot {slot}"
            del self.in_use[slot]
            self.free.append(slot)
            self.releases += 1
            return None
        raise AssertionError(name)


def scenario_hostmap():
    """Drop one block, recompute the prefix, re-evict the stale-hash block."""
    print("[H1] host_map / pool slots stay exclusive across drop + recompute")
    bm = bm_mod.BlockManager(num_blocks=8, block_size=256)
    pool = FakePool(capacity=2)       # 2 slots: H1 pooled, H0 dropped
    bm.set_offload_executor(pool)
    ok = True

    def check(tag, host_entries, in_use, drops=None):
        nonlocal ok
        good = len(bm._host_map) == host_entries and len(pool.in_use) == in_use
        if drops is not None:
            good &= bm.host_drops == drops
        ok &= good
        print(f"  {tag}: host_map={len(bm._host_map)} in_use={len(pool.in_use)} "
              f"drops={bm.host_drops} -> {'PASS' if good else 'FAIL'}")
        return good

    def donor(tokens):
        seq = Sequence(tokens, SamplingParams(max_tokens=1))
        bm.allocate(seq, 0)
        bm.hash_blocks(seq, len(seq))
        bm.deallocate(seq)

    # Donors: X has one cached block (left: 257 tokens -> 2 blocks), A has two
    # (513 tokens -> 3 blocks).  deallocate walks reversed(block_table), so the
    # LRU order is [X0, A1, A0]: the LATER prefix block is evicted first.
    tok_x = list(range(1000, 1257))
    tok_a = list(range(2000, 2513))
    donor(tok_x)
    donor(tok_a)
    assert len(bm.hash_to_block_id) == 3, bm.hash_to_block_id

    # One 8-block allocation drains the fresh list, then evicts in LRU order:
    # X0 -> host slot 0, A1 -> host slot 1 (pool FULL), A0 -> legacy drop.
    pressure = Sequence(list(range(3000, 4793)), SamplingParams(max_tokens=1))
    bm.allocate(pressure, 0)
    bm.deallocate(pressure)
    check("after pressure (H1 pooled, H0 dropped)", 2, 2, drops=1)

    # Rerun X: revives its host block and frees slot 0.
    rerun_x = Sequence(tok_x, SamplingParams(max_tokens=1))
    cached = bm.can_allocate(rerun_x)
    assert cached == 1, cached
    bm.allocate(rerun_x, cached)
    bm.hash_blocks(rerun_x, len(rerun_x))
    check("after X rerun (slot 0 free)", 1, 1)

    # Rerun A: block 0 was dropped, so the chain breaks, everything is
    # recomputed, and A1's stale host entry must be RELEASED here — keeping it
    # is the bug (it would be orphaned by the next eviction's put).
    rerun_a = Sequence(tok_a, SamplingParams(max_tokens=1))
    cached = bm.can_allocate(rerun_a)
    assert cached == 0, "block 0 was dropped, the chain must break"
    bm.allocate(rerun_a, cached)
    bm.hash_blocks(rerun_a, len(rerun_a))
    check("after A recompute (stale entry released)", 0, 0)
    bm.deallocate(rerun_a)

    # Evict A1's NEW device copy: one 5-block allocation consumes 4 fresh
    # blocks then the LRU head A1.  The put lands in the freed slot; a buggy
    # map overwrite would orphan the other one (in_use=2 vs host_map=1).
    pressure = Sequence(list(range(5000, 6025)), SamplingParams(max_tokens=1))
    bm.allocate(pressure, 0)
    bm.deallocate(pressure)
    check("after re-evict (no orphan)", 1, 1)

    bm.clear_prefix_cache()
    check("after clear_prefix_cache", 0, 0)
    print("  -> PASS" if ok else "  -> FAIL")
    return ok


def main():
    ok = scenario_hostmap()
    print()
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
