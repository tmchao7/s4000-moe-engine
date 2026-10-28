"""CPU-only scheduler/KV metrics and ScheduledBatch regression."""
from types import SimpleNamespace

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import ScheduledBatch, Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def test_prefix_metrics():
    Sequence.block_size = 4
    manager = BlockManager(num_blocks=8, block_size=4)
    first = Sequence(list(range(8)), SamplingParams(max_tokens=1))
    manager.allocate(first, manager.can_allocate(first))
    first.num_scheduled_tokens = 8
    manager.hash_blocks(first)
    manager.deallocate(first)

    second = Sequence(list(range(8)), SamplingParams(max_tokens=1))
    cached = manager.can_allocate(second)
    assert cached == 1
    manager.allocate(second, cached)
    metrics = manager.metrics()
    assert metrics["kv_prefix_hit_tokens_total"] == 4
    assert metrics["kv_blocks_used"] == 2
    manager.deallocate(second)
    manager.clear_prefix_cache()
    assert not manager.hash_to_block_id


def test_scheduled_batch():
    Sequence.block_size = 4
    config = SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=4,
        eos=-1,
        kvcache_block_size=4,
        scheduler_policy="stall_free",
        prefill_chunk_size=2,
        scheduler_starvation_steps=3,
        num_kvcache_blocks=16,
    )
    scheduler = Scheduler(config)
    scheduler.add(Sequence([1, 2, 3, 4], SamplingParams(max_tokens=1, ignore_eos=True)))
    batch = scheduler.schedule()
    assert isinstance(batch, ScheduledBatch)
    seqs, mode = batch
    assert mode == "prefill"
    assert batch.query_lens == [2]
    assert batch.context_lens == [2]
    assert batch.num_scheduled_tokens == 2
    assert scheduler.metrics()["scheduler_waiting_requests"] == 1


def main():
    test_prefix_metrics()
    test_scheduled_batch()
    print("[metrics] ScheduledBatch and KV counters passed")


if __name__ == "__main__":
    main()
