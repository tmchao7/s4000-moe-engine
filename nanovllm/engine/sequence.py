from collections import Counter
from copy import copy
from enum import Enum, auto
from itertools import count
from time import perf_counter

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.priority = sampling_params.priority
        # P22-A sampling face.  The knob attributes are read on the driver
        # (rank 0) only — prepare_sample/postprocess never run on TP workers,
        # so __getstate__ stays untouched.
        self.top_p = sampling_params.top_p
        self.top_k = sampling_params.top_k
        self.min_p = sampling_params.min_p
        self.presence_penalty = sampling_params.presence_penalty
        self.frequency_penalty = sampling_params.frequency_penalty
        self.repetition_penalty = sampling_params.repetition_penalty
        self.logprobs = sampling_params.logprobs or 0
        self.stop_token_ids = sampling_params.stop_token_ids or ()
        # P24 structured output: data here, matcher attached by the engine
        # (driver-side only — never crosses the TP pickle boundary).
        self.guided_json = sampling_params.guided_json
        self.guided_regex = sampling_params.guided_regex
        self.grammar_matcher = None
        # Penalties need token statistics: gen_counts (generated only) drives
        # presence/frequency (OpenAI semantics); rep_counts (prompt+generated)
        # drives repetition_penalty (vLLM/HF semantics), built lazily because
        # PD handoffs set num_prompt_tokens after construction.
        self.gen_counts = {}
        self.rep_counts = None
        self.arrival_time = perf_counter()
        self.first_scheduled_time = None
        self.first_token_time = None
        self.finish_time = None
        self.token_times = []
        self.wait_steps = 0
        self.finish_reason = None

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def has_penalty(self):
        return (self.presence_penalty != 0.0 or self.frequency_penalty != 0.0
                or self.repetition_penalty != 1.0)

    def ensure_rep_counts(self):
        """Prompt+generated token counts for repetition_penalty (vLLM/HF
        semantics: the penalty sees prompt tokens too).  Lazy so PD handoffs
        that fix num_prompt_tokens post-construction stay correct; updated
        incrementally in append_token afterwards."""
        if self.rep_counts is None:
            self.rep_counts = dict(Counter(self.prompt_token_ids))
        return self.rep_counts

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        self.gen_counts[token_id] = self.gen_counts.get(token_id, 0) + 1
        if self.rep_counts is not None:
            self.rep_counts[token_id] = self.rep_counts.get(token_id, 0) + 1

    def record_token(self, timestamp: float):
        if self.first_token_time is None:
            self.first_token_time = timestamp
        self.token_times.append(timestamp)

    def metrics(self):
        finish = self.finish_time or (self.token_times[-1] if self.token_times else None)
        ttft = None if self.first_token_time is None else self.first_token_time - self.arrival_time
        e2e = None if finish is None else finish - self.arrival_time
        intervals = [
            right - left for left, right in zip(self.token_times, self.token_times[1:])
        ]
        tpot = None
        if e2e is not None and ttft is not None and self.num_completion_tokens > 1:
            tpot = (e2e - ttft) / (self.num_completion_tokens - 1)
        return {
            "prompt_tokens": self.num_prompt_tokens,
            "output_tokens": self.num_completion_tokens,
            "queue_s": None if self.first_scheduled_time is None else self.first_scheduled_time - self.arrival_time,
            "ttft_s": ttft,
            "e2e_s": e2e,
            # TPOT is a per-request decode average.  ITL is the distribution
            # of individual gaps between streamed tokens.  Keep the old key
            # for one release so existing result readers do not break.
            "tpot_s": tpot,
            "mean_tpot_s": tpot,
            "itls_s": intervals,
            "token_intervals_s": intervals,
            "finish_reason": self.finish_reason,
        }

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.block_table,
            self.is_prefill,
            last_state,
        )

    def __setstate__(self, state):
        (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
         self.num_scheduled_tokens, self.block_table, self.is_prefill,
         last_state) = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
