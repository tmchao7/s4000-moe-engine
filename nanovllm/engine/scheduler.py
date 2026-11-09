from collections import deque
from dataclasses import dataclass, field
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
    schedule_s: float = 0.0     # P23: wall time of the scheduling pass
    # Frozen per-row phase at schedule time.  The overlap pipeline mutates the
    # live ``seq.is_prefill`` / ``num_scheduled_tokens`` while the batch is in
    # flight (that mutation is for the *next* batch), so postprocess/prune must
    # read the row type from this snapshot, not from live sequence state.
    prefill_flags: list[bool] = field(default_factory=list)

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
        self.kv_admission_skips_total = 0

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

    def schedule(self, speculative: bool = False, inflight: list | None = None) -> ScheduledBatch:
        """P23: ``speculative=True`` runs the pass one step ahead of the GPU
        (overlap pipeline).  The pass must then never preempt — the victim
        choice would see stale state and could evict a sequence that is about
        to finish (double free at its postprocess).  A decode row that cannot
        cover its launch-time block table without preemption therefore DEFERS
        the whole decision to the next serial schedule (which runs after that
        step's postprocess, i.e. with exactly the state the serial path would
        see).  ``inflight`` lists the sequences of the launch currently on the
        GPU: each of them grows by one token before the speculative batch
        launches, so its block coverage must be checked at len+1 (I36 fix —
        the naive ``len % block_size == 1`` check missed every block boundary
        and produced out-of-range KV slots / block-table reads)."""
        schedule_start = perf_counter()
        if self.policy == "fcfs":
            seqs, mode = self._schedule_fcfs(speculative, inflight)
        else:
            seqs, mode = self._schedule_stall_free(speculative, inflight)
        if mode == "defer":
            return ScheduledBatch(sequences=[], query_lens=[], context_lens=[],
                                  num_scheduled_tokens=0, mode="defer",
                                  schedule_s=perf_counter() - schedule_start)
        return ScheduledBatch(
            sequences=seqs,
            query_lens=[seq.num_scheduled_tokens for seq in seqs],
            context_lens=[seq.num_cached_tokens + seq.num_scheduled_tokens for seq in seqs],
            num_scheduled_tokens=sum(seq.num_scheduled_tokens for seq in seqs),
            mode=mode,
            schedule_s=perf_counter() - schedule_start,
            prefill_flags=[seq.is_prefill for seq in seqs],
        )

    def _schedule_fcfs(self, speculative: bool = False,
                       inflight: list | None = None) -> tuple[list[Sequence], str]:
        scheduled_seqs = []
        num_batched_tokens = 0
        if speculative:
            # P23: read-only pre-check before any mutation.  launch_len is the
            # sequence length when the speculative batch launches: the batch
            # currently on the GPU postprocesses first and appends one token
            # to its own members (I36 fix — coverage must be checked at
            # launch time, not at stale-schedule time).  P25/F1: the check is
            # CUMULATIVE — per-seq checks pass while the sum of shortfalls
            # exceeds the free list, and the decode loop's sequential
            # may_append_at allocation would then run off it.  The decode loop
            # pops in this same order under the same budget/count breaks, so
            # `checked` is exactly the set it will allocate for.
            inflight_ids = {id(s) for s in (inflight or [])}
            launch_len = lambda seq: len(seq) + (1 if id(seq) in inflight_ids else 0)
            budget, count = self.max_num_batched_tokens, 0
            checked = []
            for seq in self.running:
                if budget == 0 or count >= self.max_num_seqs:
                    break
                checked.append(seq)
                budget -= 1
                count += 1
            if checked and self.block_manager.total_launch_shortfall(
                    checked, [launch_len(s) for s in checked]
            ) > len(self.block_manager.free_block_ids):
                return [], "defer"

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
            if speculative:
                self.block_manager.may_append_at(seq, launch_len(seq))
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                scheduled_seqs.append(seq)
                continue
            while not self.block_manager.can_append(seq):
                if self.running:
                    victim = self._preempt_victim(requester=seq)
                    if victim is None:
                        self.preempt(seq)
                        break
                    self.preempt(victim)
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if not scheduled_seqs:
            return ScheduledBatch(
                sequences=[], query_lens=[], context_lens=[],
                num_scheduled_tokens=0, mode="empty",
            )
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, "decode"

    def _waiting_candidate(self, touch: bool = True):
        """P16-C: highest priority (lowest value) first; within a tier the
        policy applies — FIFO for stall_free, most-cacheable for lpm.  The
        starvation guard outranks priority so a long-queued low tier still
        makes progress (bounded fairness instead of hard starvation).

        wait_steps is incremented once per schedule() pass (in
        _schedule_stall_free), NOT per candidate pick — a per-pick count made
        the starvation bound load-dependent (code-review finding)."""
        starved = [seq for seq in self.waiting if seq.wait_steps >= self.starvation_steps]
        if starved:
            candidate = max(starved, key=lambda seq: (seq.wait_steps, -seq.priority))
        else:
            candidate = min(self.waiting, key=lambda seq: seq.priority)  # FIFO in tier
            if self.policy == "stall_free_lpm":
                tier = [seq for seq in self.waiting if seq.priority == candidate.priority]
                if len(tier) > 1:
                    def cached_blocks(seq):
                        if seq.block_table:
                            return seq.num_cached_tokens // self.block_size
                        value = self.block_manager.can_allocate(seq, touch=touch)
                        return value if value >= 0 else -1
                    candidate = max(tier, key=cached_blocks)
        self.waiting.remove(candidate)
        return candidate

    def _preempt_victim(self, requester: Sequence | None) -> Sequence | None:
        """P16-C: preempt the newest lowest-priority running seq (legacy
        behavior = plain newest via running.pop()).  Returns None when every
        running seq outranks the requester — the requester must then give up
        its own blocks instead of evicting a more important stream."""
        victim_idx = None
        for i, seq in enumerate(self.running):
            if victim_idx is None or seq.priority >= self.running[victim_idx].priority:
                victim_idx = i          # >= keeps the LAST max = newest
        if victim_idx is None:
            return None
        victim = self.running[victim_idx]
        if requester is not None and victim.priority < requester.priority:
            return None
        del self.running[victim_idx]
        return victim

    def _schedule_stall_free(self, speculative: bool = False,
                             inflight: list | None = None) -> tuple[list[Sequence], str]:
        decode_seqs = []
        prefill_seqs = []
        token_budget = self.max_num_batched_tokens
        seq_budget = self.max_num_seqs
        if speculative:
            # P23: read-only pre-check before any mutation.  launch_len is the
            # sequence length when the speculative batch launches: the batch
            # currently on the GPU postprocesses first and appends one token
            # to its own members (I36 fix — coverage must be checked at
            # launch time, not at stale-schedule time).  P25/F1: the check is
            # CUMULATIVE (see _schedule_fcfs) — the decode loop below pops in
            # this same order under the same budget/count breaks, so `checked`
            # is exactly the set its may_append_at allocations must cover.
            inflight_ids = {id(s) for s in (inflight or [])}
            launch_len = lambda seq: len(seq) + (1 if id(seq) in inflight_ids else 0)
            budget, count = token_budget, 0
            checked = []
            for seq in self.running:
                if budget == 0 or count >= seq_budget:
                    break
                checked.append(seq)
                budget -= 1
                count += 1
            if checked and self.block_manager.total_launch_shortfall(
                    checked, [launch_len(s) for s in checked]
            ) > len(self.block_manager.free_block_ids):
                return [], "defer"

        # Keep every admitted decode moving before admitting prefill work.
        # while (not for-range): each preemption removes an extra running seq,
        # so a fixed iteration count would popleft an empty deque (latent
        # upstream bug, fires only under heavy KV pressure).
        while self.running:
            if token_budget == 0 or len(decode_seqs) >= seq_budget:
                break
            seq = self.running.popleft()
            if speculative:
                # pre-check (same scan order/budgets) already verified the
                # batch's TOTAL launch-time block coverage without preemption
                self.block_manager.may_append_at(seq, launch_len(seq))
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                decode_seqs.append(seq)
                token_budget -= 1
                continue
            while not self.block_manager.can_append(seq):
                if self.running:
                    victim = self._preempt_victim(requester=seq)
                    if victim is None:
                        self.preempt(seq)
                        break
                    self.preempt(victim)
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                decode_seqs.append(seq)
                token_budget -= 1

        # wait_steps advances once per scheduler pass (review finding: a
        # per-pick increment made the starvation bound admission-count-dependent).
        for seq in self.waiting:
            seq.wait_steps += 1
        # Each waiting sequence contributes at most one bounded chunk per step.
        # A candidate whose KV cannot be admitted is skipped (and re-queued at
        # the head, preserving order) instead of blocking every later, possibly
        # smaller prompt: head-of-line blocking made one long unadmittable
        # prompt starve the whole waiting queue (I8's TTFT collapse).  Chunked
        # seqs are likewise held out in `requeued` until after the loop — with
        # priority picking, re-queuing mid-loop let one chunked seq be picked
        # (and re-forwarded) several times in a single step (review finding).
        num_waiting = len(self.waiting)
        skipped = []
        requeued = []
        for _ in range(num_waiting):
            if token_budget == 0 or len(decode_seqs) + len(prefill_seqs) >= seq_budget:
                break
            seq = self._waiting_candidate()
            if seq.first_scheduled_time is None:
                seq.first_scheduled_time = perf_counter()
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    skipped.append(seq)
                    self.kv_admission_skips_total += 1
                    continue
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
                requeued.append(seq)

        for seq in reversed(skipped):
            self.waiting.appendleft(seq)
        self.waiting.extend(requeued)
        self.running.extendleft(reversed(decode_seqs))
        scheduled = decode_seqs + prefill_seqs
        if not scheduled:
            # Nothing runnable: KV exhausted and no in-flight decode.  Returning
            # an empty batch keeps queued requests waiting (until a cancel frees
            # blocks) instead of crashing on an assert.
            return ScheduledBatch(
                sequences=[],
                query_lens=[],
                context_lens=[],
                num_scheduled_tokens=0,
                mode="empty",
            )
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
            "scheduler_kv_admission_skips_total": self.kv_admission_skips_total,
        }

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], mode: str,
                    logprobs: list | None = None,
                    query_lens: list[int] | None = None,
                    prefill_flags: list[bool] | None = None):
        """Apply one launched batch output.

        query_lens / prefill_flags are the FROZEN row metadata of the batch
        being collected.  The overlap pipeline schedules the next batch before
        this batch collect and rewrites the live seq fields while doing so;
        reading live seq.num_scheduled_tokens / seq.is_prefill here would
        therefore credit this batch with the NEXT batch row shape (P25/F1
        follow-up: a 200-token prefill row advanced by +1, and a completed
        prefill row could be re-launched as a 1-token prefill row).  vLLM V1
        has the same rule: output processing consumes
        scheduler_output.num_scheduled_tokens, never live request state.
        The fallback keeps the serial/archive callers unchanged.
        """
        if query_lens is None:
            query_lens = [seq.num_scheduled_tokens for seq in seqs]
        if prefill_flags is None:
            prefill_flags = [seq.is_prefill for seq in seqs]
        token_events = []
        for row, (seq, token_id) in enumerate(zip(seqs, token_ids)):
            if seq.is_finished:
                # P23 overlap: a cancel (or any finish) can land between the
                # in-flight launch and its collect; the seq was already torn
                # down (blocks deallocated, removed from queues) — skip it.
                # Serial never schedules finished seqs, so this is a no-op there.
                continue
            q = query_lens[row]
            was_prefill = prefill_flags[row]
            seq.is_prefill = was_prefill
            # Pass the frozen q explicitly: the live num_scheduled_tokens may
            # already describe the NEXT batch (review simplification).
            self.block_manager.hash_blocks(seq, q)
            seq.num_cached_tokens += q
            seq.num_scheduled_tokens = 0
            if was_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if was_prefill:
                # Final prefill chunk: the next schedule is decode.  Serial
                # sets this in the decode loop; doing it here keeps the live
                # invariant true for overlap prune/defer decisions.
                seq.is_prefill = False
            now = perf_counter()
            seq.record_token(now)
            hit_eos = not seq.ignore_eos and token_id == self.eos
            # P22-A: per-request stop token ids fire regardless of ignore_eos
            # (vLLM semantics — only the model eos is gated by ignore_eos).
            hit_stop_id = token_id in seq.stop_token_ids
            # P24: advance the grammar state machine, then check completion —
            # a finished JSON/regex grammar stops the sequence even though the
            # model would keep going (eos is usually masked out of the support).
            hit_grammar = False
            if seq.grammar_matcher is not None:
                seq.grammar_matcher.accept_token(token_id)
                hit_grammar = seq.grammar_matcher.is_terminated()
            hit_length = seq.num_completion_tokens == seq.max_tokens
            if hit_eos or hit_stop_id or hit_grammar or hit_length:
                seq.status = SequenceStatus.FINISHED
                seq.finish_time = now
                seq.finish_reason = "stop" if (hit_eos or hit_stop_id or hit_grammar) else "length"
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
            token_events.append((
                seq.seq_id,
                token_id,
                seq.is_finished,
                seq.metrics() if seq.is_finished else None,
                logprobs[row] if logprobs is not None else None,
            ))
        return token_events
