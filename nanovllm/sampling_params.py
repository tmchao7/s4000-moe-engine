from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    # P16-C: scheduling tier; lower value = higher priority.  0 (default)
    # keeps exact legacy FIFO behavior for equal priorities.
    priority: int = 0

    def __post_init__(self):
        assert self.temperature >= 0, "temperature must be non-negative (0 = greedy)"
        assert self.priority >= 0, "priority must be non-negative (0 = highest)"
