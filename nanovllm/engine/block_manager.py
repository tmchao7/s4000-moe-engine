from collections import deque, OrderedDict
import os

import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence

# P16-B: hit-aware (LRU) victim choice for block reuse.  Cached-unreferenced
# blocks always live in the free list, so the pool was never "leaking"; the
# FIFO head could however sacrifice a HOT prefix (recently hit) while colder
# entries survived.  NANOVLLM_LRU_EVICT=0 restores the plain FIFO pop.
_LRU_EVICT = os.environ.get("NANOVLLM_LRU_EVICT", "1") == "1"


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        # P16-B victim ordering (both subsets are ref-0 blocks inside
        # free_block_ids): _fresh = never-cached blocks (hash == -1, zero
        # cache value, consumed first); _lru = cached-unreferenced blocks,
        # MRU at the end, touched on can_allocate hits.  Reuse therefore
        # costs the cache nothing until every fresh block is gone, and then
        # sacrifices the COLDEST prefix instead of the FIFO head (which may
        # be a recently-hit hot prefix).  NANOVLLM_LRU_EVICT=0 restores the
        # plain FIFO pop.
        self._fresh: deque[int] = deque(range(num_blocks))
        self._fresh_set: set[int] = set(range(num_blocks))
        self._lru: OrderedDict[int, None] = OrderedDict()
        self.lru_enabled = _LRU_EVICT
        self.prefix_cache_hit_tokens = 0
        self.allocations = 0
        self.reclaims = 0
        self.evictions = 0
        # P30-R1 opt-in KV offload: evicted cached blocks are swapped to a
        # pinned host pool instead of being dropped, and copied back when a
        # prefix is revived.  Metadata (hash -> (slot, token_ids)) lives here;
        # the byte moves run per-rank through the executor's call() channel
        # (BlockManager is driver-only, the KV halves are not).  None = the
        # feature is off and every path below is byte-for-byte the legacy one.
        self._offload_exec = None
        self._host_map: dict[int, tuple[int, list[int]]] = {}
        self.host_puts = 0
        self.host_revives = 0
        self.host_drops = 0

    def set_offload_executor(self, runner):
        """Driver-side wiring (LLMEngine init): the executor's call(method,
        *args) broadcasts to every rank AND runs it locally, so pool slot
        streams stay identical across ranks.  Pass None to disarm."""
        self._offload_exec = runner

    def _release_host_entry(self, hash_key: int):
        """Drop a host-swapped entry and return its slot (no-op when absent).

        P30 review: host_map and hash_to_block_id must never describe the
        same block at once.  A dropped eviction (full pool) breaks the device
        hash chain, so a later prefill can recompute a block whose host copy
        is still mapped; when that new device copy is evicted, the put would
        overwrite the map entry and orphan the old slot forever (in_use >
        host_map, clear_prefix_cache cannot drain it).  Callers below keep
        the two maps exclusive."""
        entry = self._host_map.pop(hash_key, None)
        if entry is not None:
            self._offload_exec.call("offload_release", entry[0])

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _pop_free(self) -> int:
        """Victim choice for block reuse: fresh (never-cached) blocks first —
        they cost the cache nothing — then the coldest cached block (LRU).
        With lru disabled this degrades to the legacy FIFO free-list pop."""
        if not self.lru_enabled:
            block_id = self.free_block_ids.popleft()
            if block_id in self._fresh_set:
                if block_id == self._fresh[0]:
                    self._fresh.popleft()
                else:
                    self._fresh.remove(block_id)     # legacy path only
                self._fresh_set.discard(block_id)
            else:
                self._lru.pop(block_id, None)
            return block_id
        if self._fresh:
            block_id = self._fresh.popleft()
            self._fresh_set.discard(block_id)
        elif self._lru:
            block_id = next(iter(self._lru))
            self.evictions += 1
        else:
            block_id = self.free_block_ids.popleft()
        if block_id == self.free_block_ids[0]:
            self.free_block_ids.popleft()
        else:
            self.free_block_ids.remove(block_id)     # O(n), a few per step
        return block_id

    def _allocate_block(self) -> int:
        block_id = self._pop_free()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        self._lru.pop(block_id, None)
        self._fresh_set.discard(block_id)
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            # P30-R1: evict-to-host instead of dropping.  The data copy must
            # happen HERE — after pop, before reset overwrites the block.  A
            # full pool degrades to the legacy drop (host_drops), never an
            # error: offload must not create new failure paths.
            if self._offload_exec is not None:
                slot = self._offload_exec.call("offload_put", block_id)
                if slot != -1:
                    # Defensive (hash_blocks already keeps the maps
                    # exclusive): overwriting a stale entry here would orphan
                    # its slot.  Costs one dict lookup on the eviction path.
                    self._release_host_entry(block.hash)
                    self._host_map[block.hash] = (slot, list(block.token_ids))
                    self.host_puts += 1
                else:
                    self.host_drops += 1
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        self.allocations += 1
        return block_id

    def _revive_from_host(self, hash_key: int) -> int:
        """Copy a host-swapped block back into a fresh device block and
        restore its cache identity (hash + token_ids), so later prefix
        blocks chain-hash and future allocations find it device-side."""
        slot, token_ids = self._host_map.pop(hash_key)
        block_id = self._allocate_block()
        self._offload_exec.call("offload_get", slot, block_id)
        self._offload_exec.call("offload_release", slot)
        self.blocks[block_id].update(hash_key, token_ids)
        self.hash_to_block_id[hash_key] = block_id
        self.host_revives += 1
        return block_id

    def _deallocate_block(self, block_id: int):
        block = self.blocks[block_id]
        assert block.ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
        if block.hash != -1:
            self._lru[block_id] = None    # (re)insert at MRU end
        else:
            self._fresh.append(block_id)
            self._fresh_set.add(block_id)
        self.reclaims += 1

    def can_allocate(self, seq: Sequence, touch: bool = True) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id != -1 and self.blocks[block_id].token_ids == token_ids:
                num_cached_blocks += 1
                # Touch on hit: a REAL admission's prefix is about to be hot —
                # move it to MRU so interleaved allocations don't sacrifice it.
                # The lpm ranking probe (scheduler._waiting_candidate) passes
                # touch=False (P30): ranking must not reorder the LRU on
                # phantom hits — only admitted prefixes get bumped.  (P26
                # audit item closed: touch=False now has a caller.)
                if touch and self.lru_enabled and block_id in self._lru:
                    self._lru.move_to_end(block_id)
                if block_id in self.used_block_ids:
                    num_new_blocks -= 1
                continue
            # P30-R1: device miss — the prefix block may be host-swapped.
            # It still counts as cached (no recompute) but needs a fresh
            # device block on allocate, so num_new_blocks is NOT decremented.
            if self._offload_exec is not None:
                entry = self._host_map.get(h)
                if entry is not None and entry[1] == token_ids:
                    num_cached_blocks += 1
                    continue
            break
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert num_cached_blocks >= 0, "can_allocate returned -1 (insufficient KV)"
        assert not seq.block_table
        self.prefix_cache_hit_tokens += num_cached_blocks * self.block_size
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            if h not in self.hash_to_block_id:
                # P30-R1: host-swapped prefix — can_allocate verified it.
                # _revive_from_host allocates a FRESH device block (ref 1 via
                # reset) and restores hash + token_ids on it.
                seq.block_table.append(self._revive_from_host(h))
                continue
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
                self._lru.pop(block_id, None)    # referenced again: leave LRU
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def _blocks_needed(self, seq: Sequence, launch_len: int) -> int:
        """P23: block-table shortfall for a decode row whose input token sits
        at index ``launch_len - 1`` (the stale schedule sees the sequence one
        postprocess-append shorter than it will be at launch)."""
        need = (launch_len - 1) // self.block_size + 1
        return max(0, need - len(seq.block_table))

    def can_append_at(self, seq: Sequence, launch_len: int) -> bool:
        return len(self.free_block_ids) >= self._blocks_needed(seq, launch_len)

    def total_launch_shortfall(self, seqs: list[Sequence],
                               launch_lens: list[int]) -> int:
        """P25/F1: cumulative block shortfall over a whole speculative decode
        batch.  A per-seq ``can_append_at`` check passes whenever each
        individual shortfall fits the free list, but the decode loop then
        allocates sequentially — ``sum(shortfall) > free`` runs ``_pop_free``
        off the end of the list (IndexError) even though every per-seq check
        passed.  Sequential allocation consumes exactly ``sum(shortfall)``
        blocks, so the sum is the exact no-crash guarantee."""
        return sum(self._blocks_needed(seq, ll)
                   for seq, ll in zip(seqs, launch_lens))

    def may_append_at(self, seq: Sequence, launch_len: int):
        while self._blocks_needed(seq, launch_len) > 0:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence, num_new_tokens: int | None = None):
        """Hash fully covered blocks after a step.

        ``num_new_tokens`` is the batch's frozen query length (overlap
        postprocess passes it explicitly); the default preserves the legacy
        live-field callers in tests/archive.
        """
        q = seq.num_scheduled_tokens if num_new_tokens is None else num_new_tokens
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + q) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            # P30 review: a full-pool drop breaks the device hash chain, so a
            # later prefill recomputes this block while its host copy may
            # still be mapped.  The device copy supersedes it — keeping both
            # would orphan the slot on the next eviction (put overwrite).
            if h in self._host_map:
                self._release_host_entry(h)
            self.hash_to_block_id[h] = block.block_id

    def metrics(self):
        capacity = len(self.blocks)
        used = len(self.used_block_ids)
        return {
            "kv_blocks_capacity": capacity,
            "kv_blocks_used": used,
            "kv_cache_utilization": used / capacity if capacity else 0.0,
            "kv_prefix_hit_tokens_total": self.prefix_cache_hit_tokens,
            "kv_block_allocations_total": self.allocations,
            "kv_block_reclaims_total": self.reclaims,
            "kv_lru_size": len(self._lru),
            "kv_lru_evictions_total": self.evictions,
            "kv_lru_enabled": 1 if self.lru_enabled else 0,
            # P30-R1: KV offload counters (all zero when the feature is off)
            "kv_offload_host_entries": len(self._host_map),
            "kv_offload_puts_total": self.host_puts,
            "kv_offload_revives_total": self.host_revives,
            "kv_offload_drops_total": self.host_drops,
        }

    def clear_prefix_cache(self):
        """Drop reusable hashes without touching blocks owned by live requests."""
        if self._offload_exec is not None:
            for hash_key in list(self._host_map):
                self._release_host_entry(hash_key)
        self._host_map.clear()
        self.hash_to_block_id.clear()
        for block in self.blocks:
            block.hash = -1
            block.token_ids = []
        for bid in self._lru:             # freshly uncached: zero reuse cost
            self._fresh.append(bid)
            self._fresh_set.add(bid)
        self._lru.clear()
