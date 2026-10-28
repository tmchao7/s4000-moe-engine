"""Unit tests for the three scheduler hard gaps (P4.1, CPU-only).

  gap 1 (I26): KV admission never asserts -- an unadmittable queue yields an
               empty batch ("empty" mode) instead of crashing
  gap 2 (I23): mixed-step split is a model_runner concern (GPU integration in
               L3); here we only pin the scheduler's mixed-batch contract
  gap 3 (HOL): a long prompt that cannot be admitted is skipped so shorter
               prompts still schedule; FIFO order of skipped candidates is
               preserved; skips are counted

Run:
  PYTHONPATH=. python3 -m pytest tests/p8_scheduler_gaps.py -q
  (or: PYTHONPATH=. python3 tests/p8_scheduler_gaps.py)
"""
import sys
from types import SimpleNamespace

sys.path.insert(0, "/root/moe-server-s4000")

from nanovllm.engine.scheduler import Scheduler  # noqa: E402
from nanovllm.engine.sequence import Sequence  # noqa: E402
from nanovllm.sampling_params import SamplingParams  # noqa: E402


def make_scheduler(num_blocks=4, block_size=256, policy="stall_free",
                   chunk=256, budget=2048, max_num_seqs=8):
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=budget,
        eos=0,
        kvcache_block_size=block_size,
        scheduler_policy=policy,
        prefill_chunk_size=chunk,
        scheduler_starvation_steps=32,
        num_kvcache_blocks=num_blocks,
    )
    return Scheduler(config)


def seq(num_tokens, max_tokens=1):
    return Sequence([0] * num_tokens, SamplingParams(temperature=0, max_tokens=max_tokens))


def test_hol_skip_schedules_short_prompt():
    """A 5-block prompt with 4 free blocks must not block a 1-block prompt."""
    scheduler = make_scheduler(num_blocks=4)
    long_seq = seq(1100)          # ceil(1100/256) = 5 blocks > 4 -> unadmittable
    short_seq = seq(10)
    scheduler.waiting.extend([long_seq, short_seq])
    batch = scheduler.schedule()
    scheduled_ids = [s.seq_id for s in batch.sequences]
    assert short_seq.seq_id in scheduled_ids, f"short prompt not scheduled: {scheduled_ids}"
    assert long_seq.seq_id not in scheduled_ids
    assert long_seq in scheduler.waiting, "long prompt must stay queued"
    assert scheduler.kv_admission_skips_total == 1


def test_hol_skip_preserves_fifo_order():
    scheduler = make_scheduler(num_blocks=4)
    big_a = seq(1100)
    small = seq(10)
    big_b = seq(1300)
    scheduler.waiting.extend([big_a, small, big_b])
    scheduler.schedule()
    assert [s.seq_id for s in scheduler.waiting] == [big_a.seq_id, big_b.seq_id], (
        "skipped candidates must keep their relative FIFO order"
    )
    assert scheduler.kv_admission_skips_total == 2


def test_empty_batch_no_assert():
    """Nothing runnable (KV exhausted, no decode) returns an empty batch."""
    scheduler = make_scheduler(num_blocks=2)
    huge = seq(1000)              # needs 4 blocks, only 2 exist
    scheduler.waiting.append(huge)
    batch = scheduler.schedule()  # must not raise
    assert batch.mode == "empty"
    assert batch.sequences == []
    assert len(scheduler.waiting) == 1


def test_normal_scheduling_unaffected():
    scheduler = make_scheduler(num_blocks=4)
    first = seq(10)
    second = seq(20)
    scheduler.waiting.extend([first, second])
    batch = scheduler.schedule()
    assert batch.mode == "prefill"
    assert [s.seq_id for s in batch.sequences] == [first.seq_id, second.seq_id]
    assert scheduler.kv_admission_skips_total == 0


def test_fcfs_empty_batch_no_assert():
    scheduler = make_scheduler(num_blocks=2, policy="fcfs")
    scheduler.waiting.append(seq(1000))
    batch = scheduler.schedule()
    assert batch.mode == "empty"


def test_decode_and_prefill_mixed_contract():
    """Scheduler still emits mixed batches: running decodes + a new prefill."""
    scheduler = make_scheduler(num_blocks=8)
    decode_seq = seq(10, max_tokens=8)
    scheduler.waiting.append(decode_seq)
    first = scheduler.schedule()
    assert first.mode == "prefill"
    # The first schedule moved decode_seq to running (full prompt scheduled);
    # admit a new prefill alongside the in-flight decode.
    assert decode_seq in scheduler.running
    prefill_seq = seq(12)
    scheduler.waiting.append(prefill_seq)
    decode_seq.is_prefill = False
    decode_seq.num_cached_tokens = decode_seq.num_tokens
    batch = scheduler.schedule()
    assert batch.mode == "mixed"
    assert len(batch.sequences) == 2


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
