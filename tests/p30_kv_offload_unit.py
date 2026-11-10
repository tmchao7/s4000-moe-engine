"""P30-R1: HostBlockPool unit gates (KV offload Round A foundation).

Pre-registered gates:
  t1  put/get roundtrip is bitwise exact (bf16 CPU tensors).
  t2  release + reuse returns the pool to full capacity (no slot leak).
  t3  put on a full pool returns -1 and bumps full_count (graceful
      degradation primitive for Round B); after a release, put succeeds.
  t4  shape mismatch raises ValueError (never silently corrupts).
  t5  MUSA gate (SKIP-labeled where MUSA is absent, rc-neutral but loudly
      printed): a realistic TP2 30B block shape [2, 48, 256, 2, 128] bf16
      (12 MiB) roundtrips through the pool BITWISE from and to a MUSA
      device, with the swap timing recorded.
  t6  no-room fast path: a full or disabled pool reports has_room()==False
      so the engine can skip the device sync + gather on the drop path;
      note_full() keeps full_count truthful; releasing restores room.

Run:  PYTHONPATH=. python3 tests/p30_kv_offload_unit.py        # rc 0 = pass
"""
import sys
import time

import torch

from nanovllm.engine.kv_offload import HostBlockPool

BLOCK_SHAPE = (2, 4, 8, 2, 16)          # (kv, layers, block_size, heads, dim)
BLOCK_SHAPE_30B = (2, 48, 256, 2, 128)  # TP2 30B bf16: 12 MiB


def t1_roundtrip_bitwise():
    pool = HostBlockPool(3, BLOCK_SHAPE, dtype=torch.bfloat16, pin=False)
    src = torch.randn(*BLOCK_SHAPE, dtype=torch.bfloat16)
    slot = pool.put(src, key="h1")
    out = torch.empty(*BLOCK_SHAPE, dtype=torch.bfloat16)
    pool.get(slot, out)
    ok = torch.equal(src, out)
    print(f"[t1] roundtrip bitwise: {ok}")
    return ok


def t2_release_reuse():
    pool = HostBlockPool(2, BLOCK_SHAPE, dtype=torch.bfloat16, pin=False)
    a = pool.put(torch.randn(*BLOCK_SHAPE, dtype=torch.bfloat16), "a")
    b = pool.put(torch.randn(*BLOCK_SHAPE, dtype=torch.bfloat16), "b")
    pool.release(a)
    a2 = pool.put(torch.randn(*BLOCK_SHAPE, dtype=torch.bfloat16), "a2")
    m = pool.metrics()
    ok = (a2 in (a, b) and m["kv_offload_slots_in_use"] == 2
          and m["kv_offload_releases_total"] == 1
          and m["kv_offload_puts_total"] == 3
          and m["kv_offload_slots_capacity"] == 2)
    print(f"[t2] release/reuse accounting: {ok} slots={sorted(pool._in_use)}")
    return ok


def t3_full_pool_degrades():
    pool = HostBlockPool(2, BLOCK_SHAPE, dtype=torch.bfloat16, pin=False)
    s1 = pool.put(torch.randn(*BLOCK_SHAPE, dtype=torch.bfloat16), "1")
    s2 = pool.put(torch.randn(*BLOCK_SHAPE, dtype=torch.bfloat16), "2")
    full = pool.put(torch.randn(*BLOCK_SHAPE, dtype=torch.bfloat16), "3")
    m = pool.metrics()
    pool.release(s1)
    s3 = pool.put(torch.randn(*BLOCK_SHAPE, dtype=torch.bfloat16), "3")
    ok = (s1 != -1 and s2 != -1 and full == -1
          and m["kv_offload_full_total"] == 1 and s3 != -1)
    print(f"[t3] full-pool -1 + recovery: {ok}")
    return ok


def t4_shape_mismatch_raises():
    pool = HostBlockPool(1, BLOCK_SHAPE, dtype=torch.bfloat16, pin=False)
    try:
        pool.put(torch.randn(1, 2, 3), "x")
        print("[t4] shape mismatch: FAIL (no raise)")
        return False
    except ValueError:
        out = torch.empty(1, 2, 3)
        try:
            pool.put(torch.randn(*BLOCK_SHAPE, dtype=torch.bfloat16), "k")
            pool.get(0, out)
            print("[t4] shape mismatch: FAIL (get accepted bad out)")
            return False
        except ValueError:
            print("[t4] shape mismatch raises: True")
            return True


def t5_musa_block_roundtrip():
    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        print("[t5] MUSA gate: SKIP (no MUSA on this box — run on the server)")
        return True
    pool = HostBlockPool(2, BLOCK_SHAPE_30B, dtype=torch.bfloat16)
    src = torch.randn(*BLOCK_SHAPE_30B, dtype=torch.bfloat16, device="musa")
    slot = pool.put(src, "blk0")
    if slot == -1:
        print("[t5] MUSA gate: FAIL (put returned -1 on an empty pool)")
        return False
    out = torch.empty(*BLOCK_SHAPE_30B, dtype=torch.bfloat16, device="musa")
    torch.musa.synchronize()
    t0 = time.perf_counter()
    pool.get(slot, out)
    torch.musa.synchronize()
    dt = time.perf_counter() - t0
    ok = torch.equal(src, out)
    gbps = (12 * 1024 * 1024) / dt / 1e9
    print(f"[t5] MUSA 12MiB block roundtrip bitwise: {ok} "
          f"(H2D {gbps:.1f} GB/s, {dt * 1e3:.2f} ms)")
    return ok


def t6_no_room_fast_path():
    """Full/disabled pool must expose no_room so the engine can skip the
    device sync + gather on the drop path (measured ~2.5 ms of dead work per
    dropped put before the fix; benchmark section 6.2).  note_full keeps the
    full_count metric truthful for the engine-level poolfull gate."""
    pool = HostBlockPool(1, BLOCK_SHAPE, dtype=torch.bfloat16, pin=False)
    has_room_empty = pool.has_room()
    slot = pool.put(torch.randn(*BLOCK_SHAPE, dtype=torch.bfloat16), "a")
    full = not pool.has_room()
    rc = pool.note_full()
    counted = pool.metrics()["kv_offload_full_total"] == 1
    disabled = HostBlockPool(0, BLOCK_SHAPE, dtype=torch.bfloat16, pin=False)
    disabled_ok = (not disabled.has_room() and disabled.note_full() == -1)
    pool.release(slot)
    recovered = pool.has_room()
    ok = (has_room_empty and slot != -1 and full and rc == -1 and counted
          and disabled_ok and recovered)
    print(f"[t6] no-room fast path (full + disabled + recovery): {ok}")
    return ok


def main():
    results = {
        "t1": t1_roundtrip_bitwise(),
        "t2": t2_release_reuse(),
        "t3": t3_full_pool_degrades(),
        "t4": t4_shape_mismatch_raises(),
        "t5": t5_musa_block_roundtrip(),
        "t6": t6_no_room_fast_path(),
    }
    print("VERDICT:", "PASS" if all(results.values()) else "FAIL", results)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
