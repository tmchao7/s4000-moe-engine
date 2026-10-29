"""Unit tests for the v3 benchmark metric helpers (docs/P8-benchmark-spec.md).

Pure functions only: Wilson intervals, token goodput, dual-SLO conformance,
error bucketing, linear percentiles, v2-JSON replay, and the fixed-concurrency
worker loop.  Run with `python3 -m pytest tests/test_benchmarking_v3.py`.
"""

import asyncio
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import pytest


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, REPO / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPO = Path(__file__).resolve().parents[1]

# Stub the nanovllm package so loading scripts/benchmark_online.py does not
# execute the real package __init__ (which imports torch/tqdm via the engine).
# The real pure-python benchmarking module is registered under its name.
sys.modules.setdefault("nanovllm", types.ModuleType("nanovllm"))
benchmarking = _load_module("nanovllm.benchmarking", "nanovllm/benchmarking.py")
sys.modules["nanovllm.benchmarking"] = benchmarking
sys.modules["nanovllm"].benchmarking = benchmarking

classify_error = benchmarking.classify_error
conforming = benchmarking.conforming
error_bucket_counts = benchmarking.error_bucket_counts
percentile = benchmarking.percentile
token_goodput = benchmarking.token_goodput
wilson_ci = benchmarking.wilson_ci


def _load_client_module():
    return _load_module("benchmark_online_v3", "scripts/benchmark_online.py")


def test_wilson_ci_full_success():
    low, high = wilson_ci(128, 128)
    assert low == pytest.approx(0.971, abs=0.002)
    assert high == pytest.approx(1.0, abs=0.002)


def test_wilson_ci_degenerate():
    assert wilson_ci(0, 0) == (None, None)
    low, high = wilson_ci(0, 128)
    assert low == 0.0
    assert high < 0.03


def test_token_goodput():
    records = [
        {"output_tokens": 64, "conforming_standard": index < 100}
        for index in range(128)
    ]
    assert token_goodput(records, 100.0) == pytest.approx(64.0)


def test_token_goodput_zero_duration():
    assert token_goodput([{"output_tokens": 64, "conforming_standard": True}], 0.0) == 0.0


def test_dual_slo_itl_stall():
    row = {
        "ok": True,
        "ttft_s": 0.5,
        "tpot_s": 0.2,
        "itls_s": [0.1] * 20 + [2.0],
    }
    standard, strict = conforming(row, ttft_slo_s=2.0, tpot_slo_s=0.5)
    assert standard is True
    assert strict is False


def test_dual_slo_clean_request():
    row = {"ok": True, "ttft_s": 0.5, "tpot_s": 0.2, "itls_s": [0.1] * 20}
    standard, strict = conforming(row, ttft_slo_s=2.0, tpot_slo_s=0.5)
    assert (standard, strict) == (True, True)


def test_dual_slo_failed_request():
    row = {"ok": False, "ttft_s": None, "tpot_s": None, "itls_s": []}
    assert conforming(row, ttft_slo_s=2.0, tpot_slo_s=0.5) == (False, False)


def test_error_buckets():
    rows = [
        {"ok": False, "timed_out": True, "http_status": None, "error": "timeout"},
        {"ok": False, "timed_out": False, "http_status": 500, "error": "http error"},
        {"ok": False, "timed_out": False, "http_status": None, "error": "stream ended before [DONE]"},
        {"ok": False, "timed_out": False, "http_status": None,
         "error": "token usage missing and no local tokenizer was configured"},
    ]
    counts = error_bucket_counts(rows)
    assert counts["timeout"] == 1
    assert counts["http"] == 1
    assert counts["stream"] == 1
    assert counts["token_count"] == 1
    assert sum(counts.values()) == 4
    assert all(classify_error(row) != "none" for row in rows)


def test_error_buckets_reject_429():
    row = {"ok": False, "timed_out": False, "http_status": 429, "error": "429"}
    assert classify_error(row) == "reject_429"


def test_linear_percentile():
    assert percentile([0, 10], 0.95, method="linear") == pytest.approx(9.5)
    # nearest-rank stays the default so v2 callers keep their numbers
    assert percentile([0, 10], 0.95) == 10
    assert percentile([0, 10], 0.95, method="linear") != percentile([0, 10], 0.95)


def test_v2_json_replay():
    """A v2 payload must run through the v3 record pipeline without errors."""
    payload_path = REPO / "benchmarks" / "data" / "p6d" / "online-v4-paged.json"
    if not payload_path.exists():
        pytest.skip("v4 baseline JSON not present")
    payload = json.loads(payload_path.read_text())
    assert payload.get("schema_version") == 2
    records = payload["records"]
    assert records
    for row in records:
        # must not raise; v2 rows simply get their buckets re-derived
        classify_error(row)
    good = [row for row in records if row.get("ok")]
    assert good
    for row in good[:10]:
        standard, strict = conforming(row, ttft_slo_s=2.0, tpot_slo_s=0.5)
        assert isinstance(standard, bool) and isinstance(strict, bool)


def test_concurrency_mode_achieved_qps(monkeypatch):
    module = _load_client_module()

    async def fake_one_request(client, url, model, prompt, max_tokens, request_id,
                               due, tracker, tokenizer=None, request_timeout_s=None,
                               include_metrics=True):
        tracker.enter()
        start = time.perf_counter()
        await asyncio.sleep(0.1)
        tracker.exit()
        return {
            "request_id": request_id, "scheduled_s": due, "started_s": start,
            "ended_s": time.perf_counter(), "e2e_s": 0.1, "ttft_s": 0.05,
            "tpot_s": 0.001, "itls_s": [0.001] * 15, "prompt_tokens": 128,
            "output_tokens": 16, "stream_events": 16, "done_received": True,
            "finish_reason": "length", "ok": True, "error": None,
            "timed_out": False, "http_status": None, "error_kind": "none",
        }

    async def fake_warmup(*args, **kwargs):
        return

    async def fake_scrape(client, base_url):
        return {}

    monkeypatch.setattr(module, "one_request", fake_one_request)
    monkeypatch.setattr(module, "warmup", fake_warmup)
    monkeypatch.setattr(module, "scrape_metrics", fake_scrape)

    args = type("Args", (), {})()
    args.requests = 40
    args.warmup_requests = 0
    args.duration = None
    args.request_timeout_s = 120.0
    args.max_tokens = 16
    args.slo_ttft_ms = 2000.0
    args.slo_tpot_ms = 500.0
    args.strict_max_itl_s = 1.0
    args.prompt_mode = "shared"
    args.prompt_word = "benchmark"
    args.seed = 1
    args.prompt_words = 8
    args.base_url = "http://127.0.0.1:8000"
    args.model = "test-model"

    rows, before, after, max_conc = asyncio.run(module.run_point_concurrency(
        client=None, args=args, concurrency=4, point_index=0, repeat=1,
        prompt_for_request=lambda rid: "prompt", tokenizer=None,
    ))
    assert len(rows) == 40
    assert max_conc == 4
    duration = max(row["ended_s"] for row in rows) - min(row["started_s"] for row in rows)
    achieved_qps = 40 / duration
    # 40 requests x 100 ms at concurrency 4 -> ~10 s of work in 4 workers = 1 s
    assert achieved_qps == pytest.approx(40.0, rel=0.3)


def test_mixed_bucket_distribution():
    module = _load_client_module()
    counts = {"short": 0, "medium": 0, "long": 0}
    for request_id in range(1000):
        counts[module.mixed_bucket(request_id)] += 1
    assert counts["short"] == 300
    assert counts["medium"] == 400
    assert counts["long"] == 300


def test_gpu_sampler_line_parsing():
    module = _load_client_module()
    line = "0    MTT S4000            |00000000:27:00.0    |0%    0MiB(49152MiB)"
    match = module.GpuSampler._LINE.match(line.strip())
    assert match is not None
    assert match.group(1) == "0"
    assert match.group(2) == "0"
    assert match.group(3) == "0"
    assert match.group(4) == "49152"


def test_aggregate_pools_repeats():
    module = _load_client_module()
    summaries = [
        {"load_fraction": 1.0, "repeat": index + 1, "slo_standard_rate": rate,
         "slo_strict_rate": rate, "achieved_qps": 0.8,
         "counts": {"attempted": 128, "conforming_standard": int(128 * rate),
                    "conforming_strict": int(128 * rate)}}
        for index, rate in enumerate((1.0, 0.992))
    ]
    aggregates = module.aggregate(summaries, ["load_fraction"])
    assert len(aggregates) == 1
    entry = aggregates[0]
    assert entry["slo_standard_rate_per_repeat"] == [1.0, 0.992]
    low, high = entry["slo_standard_rate_ci95"]
    assert low is not None and 0.97 < low < 1.0


def test_admission_denominators_and_unknown():
    module = _load_client_module()
    rows = [
        {"http_status": 200, "conforming_standard": True, "conforming_strict": True},
        {"http_status": 200, "conforming_standard": False},  # admitted, stream failed
        {"http_status": 429}, {"http_status": None}, {"http_status": 400},
    ]
    result = module.admission_summary(rows)
    assert result["admitted_slo_standard_rate"] == .5
    assert result["reject_429_rate"] == .2
    assert result["admission_counts"]["unknown"] == 1
    assert result["admission_counts"]["other_http"] == 1
    assert module.admission_summary([])["admitted_slo_standard_rate"] is None
    assert module.admission_summary([{"http_status": 429}])["reject_429_rate"] == 1
    assert module.admission_summary([{"http_status": 429}])["admitted_slo_standard_rate"] is None


def test_admission_aggregate_weights_counts():
    module = _load_client_module()
    rows = []
    for repeat, statuses in enumerate(([200, 429], [200, 200, None]), 1):
        admission = module.admission_summary([{"http_status": s, "conforming_standard": s == 200,
                                              "conforming_strict": s == 200} for s in statuses])
        rows.append({"load_fraction": 1, "repeat": repeat, **admission,
                     "counts": {"attempted": len(statuses), "conforming_standard": statuses.count(200),
                                "conforming_strict": statuses.count(200)},
                     "slo_standard_rate": statuses.count(200) / len(statuses),
                     "slo_strict_rate": statuses.count(200) / len(statuses), "achieved_qps": 1})
    result = module.aggregate(rows, ["load_fraction"])[0]
    assert result["reject_429_rate"] == .2
    assert result["admitted_slo_standard_rate"] == 1
    assert result["admission_counts"]["unknown"] == 1


@pytest.mark.parametrize("failure", ["disconnect", "timeout_before", "timeout_after"])
def test_http_status_survives_stream_failure(failure):
    module = _load_client_module()

    class Response:
        status_code = 200
        def raise_for_status(self):
            pass
        async def aiter_lines(self):
            if failure == "timeout_after":
                raise asyncio.TimeoutError()
            raise RuntimeError("disconnected")
            yield ""

    class Stream:
        async def __aenter__(self):
            if failure == "timeout_before":
                raise asyncio.TimeoutError()
            return Response()
        async def __aexit__(self, *args):
            return False

    class Client:
        def stream(self, *args, **kwargs):
            return Stream()

    row = asyncio.run(module.one_request(Client(), "http://test", "model", "prompt", 4,
                                        0, time.perf_counter(), module.ConcurrencyTracker()))
    assert not row["ok"]
    assert row["http_status"] == (None if failure == "timeout_before" else 200)


def test_aba_decisions():
    judge = _load_module("p13_decide", "scripts/p13_decide.py")
    def arm(value):
        return {"summary": [{"load_fraction": f, "repeat": r, "counts": {"attempted": 128, "ok": 128},
                              "tpot_p95_s": value} for f in (.9, 1.) for r in (1, 2, 3)]}
    assert judge.aba(arm(1), arm(.99), arm(1))["overall"] == "NO_REGRESSION"
    assert judge.aba(arm(1), arm(1.06), arm(1))["overall"] == "REGRESSION"
    assert judge.aba(arm(1), arm(1.02), arm(1))["overall"] == "INCONCLUSIVE"
    assert judge.aba(arm(1), arm(.99), arm(1.1))["overall"] == "INVALID"
    assert judge.aba(arm(1), {}, arm(1))["overall"] == "INVALID"
    noisy = arm(1)
    noisy["summary"][0]["tpot_p95_s"] = 2
    assert judge.aba(arm(1), noisy, arm(1))["overall"] == "INVALID"


def test_summary_keeps_all_attempts_slo_and_server_timing():
    module = _load_client_module()
    args = types.SimpleNamespace(slo_ttft_ms=2000, slo_tpot_ms=500, strict_max_itl_s=1)
    rows = []
    for status in (200, 429, None):
        rows.append({"http_status": status, "ok": status == 200, "started_s": 0, "ended_s": 1,
                     "ttft_s": .4, "tpot_s": .1, "itls_s": [.1], "e2e_s": 1,
                     "prompt_tokens": 128, "output_tokens": 8, "error": None if status == 200 else "failed",
                     "timed_out": status is None, "server_ttft_s": .3 if status == 200 else None,
                     "server_queue_s": .1 if status == 200 else None})
    result = module.summarize(rows, 3, {}, args, {"load_fraction": 1, "repeat": 1})
    assert result["slo_standard_rate"] == pytest.approx(1/3)
    assert result["admitted_slo_standard_rate"] == 1
    assert result["reject_429_rate"] == pytest.approx(1/3)
    assert result["scheduled_to_first_token_p50_s"] == pytest.approx(.2)
    assert result["output_tokens_p50"] == 8


def test_aba_rejects_different_workloads():
    judge = _load_module("p13_decide", "scripts/p13_decide.py")
    assert judge.aba({"args": {"seed": 1}}, {"args": {"seed": 2}},
                     {"args": {"seed": 1}})["reason"] == "benchmark configuration mismatch"


def test_padding_requires_repeats_valid_batches_and_tokens():
    judge = _load_module("p13_decide", "scripts/p13_decide.py")
    payloads = []
    for _ in range(5):
        for mode in ("eager", "segment"):
            payloads.append({"complete": True, "mode": mode, "batches": {
                str(n): {"decode_median_ms": (12 if mode == "segment" else 10.5) if n == 17 else 10,
                         "cv": .01, "actual_batch_valid": True, "tokens_consistent": True,
                         "cap": 32 if mode == "segment" and n == 17 else n,
                         "rounds": [{"token_sha256": "warmup"}, {"token_sha256": str(n)}]}
                for n in (16, 17)}})
    assert judge.padding(payloads)["overall"] == "STEP_SUPPORTED"
    assert judge.padding(payloads[:-1])["overall"] == "INVALID"
    payloads[0]["batches"]["16"]["actual_batch_valid"] = False
    assert judge.padding(payloads)["overall"] == "INVALID"
