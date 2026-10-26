from collections import deque
from dataclasses import dataclass
from time import perf_counter

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


@dataclass(slots=True)
class ScheduledBatch:
    """A scheduler decision with enough shape information for instrumentation.

    Iteration intentionally yields ``(sequences, mode)`` so existing diagnostic
    scripts written against the original nano-vLLM tuple API remain valid.
    """

    sequences: list[Sequence]
    query_lens: list[int]
    context_lens: list[int]
    num_scheduled_tokens: int
    mode: str

    def __iter__(self):
        yield self.sequences
        yield self.mode


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.policy = config.scheduler_policy
        self.prefill_chunk_size = config.prefill_chunk_size
        self.starvation_steps = config.scheduler_starvation_steps
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.preemptions = 0

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def cancel(self, seq_id: int) -> bool:
        for queue in (self.waiting, self.running):
            for seq in queue:
                if seq.seq_id != seq_id:
                    continue
                queue.remove(seq)
                if seq.block_table:
                    self.block_manager.deallocate(seq)
                seq.status = SequenceStatus.FINISHED
                seq.finish_time = perf_counter()
                seq.finish_reason = "cancelled"
                return True
        return False

    def schedule(self) -> ScheduledBatch:
        if self.policy == "fcfs":
            seqs, mode = self._schedule_fcfs()
        else:
            seqs, mode = self._schedule_stall_free()
        return ScheduledBatch(
            sequences=seqs,
            query_lens=[seq.num_scheduled_tokens for seq in seqs],
            context_lens=[seq.num_cached_tokens + seq.num_scheduled_tokens for seq in seqs],
            num_scheduled_tokens=sum(seq.num_scheduled_tokens for seq in seqs),
            mode=mode,
        )

    def _schedule_fcfs(self) -> tuple[list[Sequence], str]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            if seq.first_scheduled_time is None:
                seq.first_scheduled_time = perf_counter()
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, "prefill"

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, "decode"

    def _waiting_candidate(self):
        if self.policy != "stall_free_lpm" or len(self.waiting) == 1:
            return self.waiting.popleft()
        for seq in self.waiting:
            seq.wait_steps += 1
        starved = [seq for seq in self.waiting if seq.wait_steps >= self.starvation_steps]
        if starved:
            candidate = max(starved, key=lambda seq: seq.wait_steps)
        else:
            def cached_blocks(seq):
                if seq.block_table:
                    return seq.num_cached_tokens // self.block_size
                value = self.block_manager.can_allocate(seq)
                return value if value >= 0 else -1
            candidate = max(self.waiting, key=cached_blocks)
        self.waiting.remove(candidate)
        return candidate

    def _schedule_stall_free(self) -> tuple[list[Sequence], str]:
        decode_seqs = []
        prefill_seqs = []
        token_budget = self.max_num_batched_tokens
        seq_budget = self.max_num_seqs

        # Keep every admitted decode moving before admitting prefill work.
        num_running = len(self.running)
        for _ in range(num_running):
            if token_budget == 0 or len(decode_seqs) >= seq_budget:
                break
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                decode_seqs.append(seq)
                token_budget -= 1

        # Each waiting sequence contributes at most one bounded chunk per step.
        num_waiting = len(self.waiting)
        for _ in range(num_waiting):
            if token_budget == 0 or len(decode_seqs) + len(prefill_seqs) >= seq_budget:
                break
            seq = self._waiting_candidate()
            if seq.first_scheduled_time is None:
                seq.first_scheduled_time = perf_counter()
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    self.waiting.appendleft(seq)
                    break
                self.block_manager.allocate(seq, num_cached_blocks)
            num_tokens = seq.num_tokens - seq.num_cached_tokens
            seq.num_scheduled_tokens = min(
                num_tokens,
                token_budget,
                self.prefill_chunk_size,
            )
            seq.wait_steps = 0
            token_budget -= seq.num_scheduled_tokens
            prefill_seqs.append(seq)
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            else:
                self.waiting.append(seq)

        self.running.extendleft(reversed(decode_seqs))
        scheduled = decode_seqs + prefill_seqs
        assert scheduled
        if decode_seqs and prefill_seqs:
            mode = "mixed"
        elif decode_seqs:
            mode = "decode"
        else:
            mode = "prefill"
        return scheduled, mode

    def preempt(self, seq: Sequence):
        self.preemptions += 1
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def metrics(self):
        return {
            **self.block_manager.metrics(),
            "scheduler_waiting_requests": len(self.waiting),
            "scheduler_running_requests": len(self.running),
            "scheduler_preemptions_total": self.preemptions,
        }

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], mode: str):
        token_events = []
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            now = perf_counter()
            seq.record_token(now)
            hit_eos = not seq.ignore_eos and token_id == self.eos
            hit_length = seq.num_completion_tokens == seq.max_tokens
            if hit_eos or hit_length:
                seq.status = SequenceStatus.FINISHED
                seq.finish_time = now
                seq.finish_reason = "stop" if hit_eos else "length"
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
            token_events.append((
                seq.seq_id,
                token_id,
                seq.is_finished,
                seq.metrics() if seq.is_finished else None,
            ))
        return token_events
