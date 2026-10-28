"""Shared metric definitions for offline and online benchmarks.

The names follow vLLM's client-side convention: TPOT is calculated once per
request while ITL is the collection of gaps between streamed outputs.
"""

from __future__ import annotations

import math
import statistics


def percentile(values, q: float, method: str = "nearest"):
    values = [value for value in values if value is not None]
    if not values:
        return None
    if not 0 <= q <= 1:
        raise ValueError("percentile must be between zero and one")
    ordered = sorted(values)
    if method == "linear":
        # numpy default: interpolate between adjacent order statistics.
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
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


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054):
    """Wilson score interval for a binomial proportion (95% by default)."""
    if total <= 0:
        return (None, None)
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (p + z2 / (2.0 * total)) / denominator
    spread = z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def token_goodput(records, duration_s: float, conforming_key: str = "conforming_standard") -> float:
    """SLO-conforming output tokens per second (v3 `goodput_tokens_per_s`)."""
    if duration_s <= 0:
        return 0.0
    tokens = sum(
        row.get("output_tokens") or 0
        for row in records
        if row.get(conforming_key)
    )
    return tokens / duration_s


def max_itl_s(row) -> float | None:
    itls = row.get("itls_s") or []
    return max(itls) if itls else None


def classify_error(row) -> str:
    """Bucket a failed request into v3 `error_kind`.

    Buckets: none, timeout, http, reject_429, stream, token_count.  The caller
    tags the row with `timed_out` / `http_status`; everything else that failed
    without a usage payload is a stream-level failure.
    """
    if row.get("ok"):
        return "none"
    if row.get("timed_out"):
        return "timeout"
    status = row.get("http_status")
    if status == 429:
        return "reject_429"
    if status is not None:
        return "http"
    if "token usage missing" in (row.get("error") or ""):
        return "token_count"
    return "stream"


def error_bucket_counts(rows) -> dict[str, int]:
    buckets = {kind: 0 for kind in ("none", "timeout", "http", "reject_429", "stream", "token_count")}
    for row in rows:
        kind = classify_error(row)
        buckets.setdefault(kind, 0)
        buckets[kind] += 1
    return buckets


def conforming(row, ttft_slo_s: float, tpot_slo_s: float, strict_max_itl_s: float = 1.0) -> tuple[bool, bool]:
    """Return (conforming_standard, conforming_strict) for one record.

    standard: ttft <= SLO and request-level tpot <= SLO (v2/v3/v4 comparable).
    strict: standard AND every inter-token gap <= strict_max_itl_s.  A request
    with fewer than two tokens cannot show a stall and inherits standard.
    """
    standard = bool(
        row.get("ok")
        and row.get("ttft_s") is not None
        and row.get("tpot_s") is not None
        and row["ttft_s"] <= ttft_slo_s
        and row["tpot_s"] <= tpot_slo_s
    )
    if not standard:
        return (False, False)
    itls = row.get("itls_s") or []
    strict = all(value <= strict_max_itl_s for value in itls)
    return (True, strict)
