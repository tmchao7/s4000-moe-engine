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
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        self.allocations += 1
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
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            # Touch on hit: a queued request's prefix is about to be hot —
            # move it to MRU so interleaved allocations don't sacrifice it.
            # Ranking-only probes (stall_free_lpm candidate scoring) pass
            # touch=False so repeated probing can't pin blocks at MRU.
            if touch and self.lru_enabled and block_id in self._lru:
                self._lru.move_to_end(block_id)
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
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

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
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
        }

    def clear_prefix_cache(self):
        """Drop reusable hashes without touching blocks owned by live requests."""
        self.hash_to_block_id.clear()
        for block in self.blocks:
            block.hash = -1
            block.token_ids = []
        for bid in self._lru:             # freshly uncached: zero reuse cost
            self._fresh.append(bid)
            self._fresh_set.add(bid)
        self._lru.clear()
