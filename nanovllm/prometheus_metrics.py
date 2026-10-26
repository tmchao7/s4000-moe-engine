"""Low-cardinality Prometheus metrics for the serving process."""

from __future__ import annotations


class PrometheusMetrics:

    def __init__(self):
        try:
            from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
        except ImportError as exc:
            raise RuntimeError(
                "Prometheus metrics require the 'serve' optional dependencies"
            ) from exc

        self.registry = CollectorRegistry(auto_describe=True)
        namespace = "nanovllm"
        self.requests = Counter(
            "requests", "Requests admitted by the engine.", namespace=namespace,
            registry=self.registry,
        )
        self.finished = Counter(
            "requests_finished", "Requests completed normally.", namespace=namespace,
            registry=self.registry,
        )
        self.cancelled = Counter(
            "requests_cancelled", "Requests cancelled before completion.", namespace=namespace,
            registry=self.registry,
        )
        self.errors = Counter(
            "requests_errors", "Requests terminated by an engine error.", namespace=namespace,
            registry=self.registry,
        )
        self.generated_tokens = Counter(
            "generated_tokens", "Output tokens emitted by the engine.", namespace=namespace,
            registry=self.registry,
        )
        self.rejected = Counter(
            "requests_rejected", "Requests rejected by admission control.", namespace=namespace,
            registry=self.registry,
        )
        self.active = Gauge(
            "active_requests", "Requests currently owned by the engine.", namespace=namespace,
            registry=self.registry,
        )
        latency_buckets = (
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
            2.0, 5.0, 10.0, 30.0, 60.0, 120.0,
        )
        token_buckets = (
            0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1,
            0.25, 0.5, 1.0, 2.0,
        )
        self.queue = Histogram(
            "request_queue_seconds", "Time from admission to first scheduling.",
            namespace=namespace, registry=self.registry, buckets=latency_buckets,
        )
        self.ttft = Histogram(
            "request_ttft_seconds", "Time from admission to first output token.",
            namespace=namespace, registry=self.registry, buckets=latency_buckets,
        )
        self.tpot = Histogram(
            "request_tpot_seconds", "Per-request decode time per output token.",
            namespace=namespace, registry=self.registry, buckets=token_buckets,
        )
        self.e2e = Histogram(
            "request_e2e_seconds", "End-to-end request latency.", namespace=namespace,
            registry=self.registry, buckets=latency_buckets,
        )
        step_buckets = (
            0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005,
            0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
        )
        self.step_schedule = Histogram(
            "step_schedule_seconds", "CPU scheduler latency per engine step.",
            namespace=namespace, registry=self.registry, buckets=step_buckets,
        )
        self.step_forward = Histogram(
            "step_forward_seconds", "Model forward latency per engine step.",
            namespace=namespace, registry=self.registry, buckets=step_buckets,
        )

    def observe_request(self, metrics: dict | None):
        if not metrics:
            return
        for collector, key in (
            (self.queue, "queue_s"),
            (self.ttft, "ttft_s"),
            (self.tpot, "tpot_s"),
            (self.e2e, "e2e_s"),
        ):
            value = metrics.get(key)
            if value is not None:
                collector.observe(value)

    def observe_step(self, metrics: dict | None):
        if not metrics:
            return
        self.step_schedule.observe(metrics["schedule_s"])
        self.step_forward.observe(metrics["forward_s"])

    def render(self, engine_metrics: dict) -> str:
        from prometheus_client import generate_latest

        output = bytearray(generate_latest(self.registry))
        service_names = {
            "requests_total", "requests_finished", "requests_cancelled",
            "requests_errors", "requests_rejected", "generated_tokens_total",
            "active_requests",
        }
        for name, value in engine_metrics.items():
            if name in service_names:
                continue
            metric = f"nanovllm_{name}"
            metric_type = "counter" if name.endswith("_total") else "gauge"
            output.extend(f"# HELP {metric} nano-vLLM engine metric {name}.\n".encode())
            output.extend(f"# TYPE {metric} {metric_type}\n".encode())
            output.extend(f"{metric} {value}\n".encode())
        return output.decode("utf-8")
