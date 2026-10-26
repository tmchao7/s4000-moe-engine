"""CPU-only tests for P5 benchmark metric semantics."""

from nanovllm.benchmarking import (
    coefficient_of_variation,
    distribution,
    goodput,
    request_tpot,
)
from nanovllm.prometheus_metrics import PrometheusMetrics


def main():
    assert request_tpot(1.0, 0.2, 0) is None
    assert request_tpot(1.0, 0.2, 1) is None
    assert abs(request_tpot(1.0, 0.2, 2) - 0.8) < 1e-9
    assert abs(request_tpot(1.0, 0.2, 5) - 0.2) < 1e-9
    assert coefficient_of_variation([2.0, 2.0, 2.0]) == 0.0
    assert round(coefficient_of_variation([1.0, 2.0, 3.0]), 6) == 0.5
    stats = distribution([0.1, 0.2, 0.3, 0.4, 0.5], "itl")
    assert stats == {"itl_p50_s": 0.3, "itl_p95_s": 0.5, "itl_p99_s": 0.5}
    rows = [
        {"ok": True, "ttft_s": 1.0, "tpot_s": 0.4},
        {"ok": True, "ttft_s": 3.0, "tpot_s": 0.4},
        {"ok": False, "ttft_s": 0.1, "tpot_s": 0.1},
    ]
    assert goodput(rows, 2.0, 2.0, 0.5) == 0.5
    exporter = PrometheusMetrics()
    exporter.requests.inc()
    exporter.active.set(1)
    exporter.observe_request({
        "queue_s": 0.1, "ttft_s": 0.2, "tpot_s": 0.3, "e2e_s": 1.0,
    })
    exporter.observe_step({"schedule_s": 0.001, "forward_s": 0.2})
    text = exporter.render({"engine_steps_total": 2, "kv_cache_utilization": 0.5})
    assert "# TYPE nanovllm_request_tpot_seconds histogram" in text
    assert 'nanovllm_request_tpot_seconds_count 1.0' in text
    assert 'nanovllm_step_forward_seconds_count 1.0' in text
    assert "# TYPE nanovllm_engine_steps_total counter" in text
    assert "nanovllm_engine_steps_total 2" in text
    print("[p5-metrics] TPOT/ITL/CV/goodput semantics passed")


if __name__ == "__main__":
    main()
