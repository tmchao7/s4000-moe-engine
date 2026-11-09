"""Opt-in KV offload staging pool (P30-R1, Round A foundation).

Pre-registered context (docs/P30-OPPORTUNITIES.md section 4.5 + the probe):
pinned host-memory bandwidth on this box is 29.7 GB/s at 6 MB / 52 GB/s at
64 MB, so a 12 MiB TP2 30B KV block swaps in ~0.4 ms — capacity extension
by swapping LRU-evicted blocks to the host is bandwidth-feasible.

Round A scope (this module ONLY — nothing imports it yet; Round B wires it
into BlockManager eviction/revive behind an opt-in flag):

  HostBlockPool is a hash-agnostic slot allocator over one pinned host
  buffer shaped [num_slots, *block_shape].  The caller (Round B's
  BlockManager integration) owns the hash->slot mapping; the pool only
  moves bytes and accounts slots:

    put(tensor, key)   copy `tensor` (any device) into a free host slot,
                       return the slot; return -1 when the pool is full —
                       the caller then degrades to today's drop-on-evict.
    get(slot, out)     copy the slot back into `out` (device view).
    release(slot)      return the slot to the free list (LIFO reuse).
    metrics()          capacity/in_use/puts/gets/releases/full_count.

Pinned memory is a hard prerequisite: the probe measured 6.5x degradation
unpinned.  Copies are blocking (v1 keeps the scheduler's single-thread
determinism; async + event double-buffering is pre-registered as a later
round, only if the blocking cost shows up in a measured step-time gate).
"""
import torch


class HostBlockPool:

    def __init__(self, num_slots: int, block_shape, dtype=torch.bfloat16,
                 pin: bool = True):
        if num_slots < 0:
            raise ValueError("num_slots must be >= 0")
        self.num_slots = int(num_slots)
        self.block_shape = tuple(block_shape)
        self.dtype = dtype
        self.pin = pin
        self._buf = None
        if self.num_slots:
            # Explicit CPU: the engine creates the pool under a MUSA default
            # device (allocate_kv_cache), and pin_memory requires a dense CPU
            # tensor regardless of the ambient default device.
            self._buf = torch.empty((self.num_slots, *self.block_shape),
                                    dtype=dtype, device="cpu", pin_memory=pin)
        # LIFO free list: reusing the most recently released slot keeps the
        # working set hot in cache and makes slot choice deterministic.
        self._free = list(range(self.num_slots - 1, -1, -1))
        self._in_use: dict[int, object] = {}   # slot -> caller key
        self.puts = 0
        self.gets = 0
        self.releases = 0
        self.full_count = 0

    def put(self, tensor: torch.Tensor, key: object = None) -> int:
        """Copy `tensor` into a free slot; -1 when full (caller degrades)."""
        if self._buf is None:
            self.full_count += 1
            return -1
        if not self._free:
            self.full_count += 1
            return -1
        if tuple(tensor.shape) != self.block_shape:
            raise ValueError(
                f"block shape mismatch: got {tuple(tensor.shape)}, "
                f"pool expects {self.block_shape}")
        slot = self._free.pop()
        self._buf[slot].copy_(tensor)
        self._in_use[slot] = key
        self.puts += 1
        return slot

    def get(self, slot: int, out: torch.Tensor):
        """Copy slot content into `out` (bitwise; source slot stays valid)."""
        if slot not in self._in_use:
            raise KeyError(f"slot {slot} is not in use")
        if tuple(out.shape) != self.block_shape:
            raise ValueError(
                f"out shape mismatch: got {tuple(out.shape)}, "
                f"pool expects {self.block_shape}")
        out.copy_(self._buf[slot])
        self.gets += 1

    def view(self, slot: int) -> torch.Tensor:
        """Borrow the pinned slot tensor (engine path: caller moves it with
        .to(device) — the PD kv_channel M2-proven H2D pattern — instead of a
        copy_ into a device view)."""
        if slot not in self._in_use:
            raise KeyError(f"slot {slot} is not in use")
        self.gets += 1
        return self._buf[slot]

    def release(self, slot: int):
        if slot not in self._in_use:
            raise KeyError(f"slot {slot} is not in use")
        del self._in_use[slot]
        self._free.append(slot)
        self.releases += 1

    def metrics(self):
        return {
            "kv_offload_slots_capacity": self.num_slots,
            "kv_offload_slots_in_use": len(self._in_use),
            "kv_offload_puts_total": self.puts,
            "kv_offload_gets_total": self.gets,
            "kv_offload_releases_total": self.releases,
            "kv_offload_full_total": self.full_count,
            "kv_offload_pinned": 1 if self.pin else 0,
        }
