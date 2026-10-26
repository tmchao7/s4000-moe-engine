"""Shared metric definitions for offline and online benchmarks.

The names follow vLLM's client-side convention: TPOT is calculated once per
request while ITL is the collection of gaps between streamed outputs.
"""

from __future__ import annotations

import math
import statistics


def percentile(values, q: float):
    values = [value for value in values if value is not None]
    if not values:
        return None
    if not 0 <= q <= 1:
        raise ValueError("percentile must be between zero and one")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def request_tpot(e2e_s: float | None, ttft_s: float | None, output_tokens: int):
    if e2e_s is None or ttft_s is None or output_tokens < 2:
        return None
    return max(0.0, e2e_s - ttft_s) / (output_tokens - 1)


def coefficient_of_variation(values) -> float:
    """Sample standard deviation divided by the arithmetic mean."""
    values = list(values)
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0 if all(value == 0 for value in values) else math.inf
    return statistics.stdev(values) / mean


def distribution(values, prefix: str) -> dict[str, float | None]:
    values = [value for value in values if value is not None]
    return {
        f"{prefix}_p50_s": percentile(values, 0.50),
        f"{prefix}_p95_s": percentile(values, 0.95),
        f"{prefix}_p99_s": percentile(values, 0.99),
    }


def goodput(records, duration_s: float, ttft_slo_s: float, tpot_slo_s: float) -> float:
    if duration_s <= 0:
        return 0.0
    conforming = sum(
        1
        for row in records
        if row.get("ok")
        and row.get("ttft_s") is not None
        and row.get("tpot_s") is not None
        and row["ttft_s"] <= ttft_slo_s
        and row["tpot_s"] <= tpot_slo_s
    )
    return conforming / duration_s
