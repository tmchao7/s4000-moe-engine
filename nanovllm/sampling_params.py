from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    # P16-C: scheduling tier; lower value = higher priority.  0 (default)
    # keeps exact legacy FIFO behavior for equal priorities.
    priority: int = 0
    # P22-A sampling face (vLLM/OpenAI aligned).  Every default below
    # reproduces legacy behaviour; the sampler runs the verbatim pre-P22
    # code path while all knobs sit at their defaults.
    top_p: float = 1.0
    top_k: int = -1                  # -1 disables; 0 is rejected
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    # Extra stop token ids checked in scheduler.postprocess (vLLM semantics:
    # these fire regardless of ignore_eos; only the model eos is gated by it).
    stop_token_ids: tuple[int, ...] | list[int] | None = None
    # Top-k count of per-token logprobs to report (None/0 = off, cap 20).
    logprobs: int | None = None
    skip_special_tokens: bool = True

    def __post_init__(self):
        assert self.temperature >= 0, "temperature must be non-negative (0 = greedy)"
        assert self.priority >= 0, "priority must be non-negative (0 = highest)"
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k == 0 or self.top_k < -1:
            raise ValueError("top_k must be -1 (disabled) or a positive integer")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be in [0, 1]")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("presence_penalty must be in [-2, 2]")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("frequency_penalty must be in [-2, 2]")
        if not 0.0 < self.repetition_penalty <= 2.0:
            raise ValueError("repetition_penalty must be in (0, 2]")
        if self.logprobs is not None and not 0 <= int(self.logprobs) <= 20:
            raise ValueError("logprobs must be an integer in [0, 20]")
        if self.stop_token_ids is not None:
            if not isinstance(self.stop_token_ids, (list, tuple)) or not all(
                    isinstance(t, int) and not isinstance(t, bool)
                    for t in self.stop_token_ids):
                raise ValueError("stop_token_ids must be a list of integers")
            self.stop_token_ids = tuple(self.stop_token_ids)
