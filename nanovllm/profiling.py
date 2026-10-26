"""Opt-in device-event profiler with one synchronization per snapshot."""

from collections import defaultdict
from contextlib import contextmanager

from nanovllm.benchmarking import percentile


class DeviceProfiler:

    def __init__(self, accelerator):
        self.acc = accelerator
        self.enabled = False
        self.samples = []
        self.observations = []

    def start(self, reset=True):
        if reset:
            self.reset()
        self.enabled = True

    def stop(self):
        self.enabled = False

    def reset(self):
        self.samples.clear()
        self.observations.clear()
        if hasattr(self.acc, "reset_peak_memory_stats"):
            self.acc.reset_peak_memory_stats()

    @contextmanager
    def range(self, name, **metadata):
        if not self.enabled:
            yield
            return
        start = self.acc.Event(enable_timing=True)
        end = self.acc.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.samples.append((name, start, end, metadata))

    def observe(self, name, value, **metadata):
        if self.enabled:
            self.observations.append({"name": name, "value": value, **metadata})

    def snapshot(self, reset=True):
        self.acc.synchronize()
        grouped = defaultdict(list)
        by_layer = defaultdict(list)
        for name, start, end, metadata in self.samples:
            elapsed_ms = start.elapsed_time(end)
            grouped[name].append(elapsed_ms)
            layer = metadata.get("layer")
            if layer is not None:
                by_layer[(name, layer)].append(elapsed_ms)

        def summarize(values):
            return {
                "count": len(values),
                "total_ms": sum(values),
                "mean_ms": sum(values) / len(values),
                "p50_ms": percentile(values, 0.50),
                "p95_ms": percentile(values, 0.95),
            }

        result = {
            "segments": {name: summarize(values) for name, values in grouped.items()},
            "layers": {
                f"{name}:layer_{layer}": summarize(values)
                for (name, layer), values in by_layer.items()
            },
            "observations": list(self.observations),
        }
        if hasattr(self.acc, "max_memory_allocated"):
            result["peak_memory_bytes"] = self.acc.max_memory_allocated()
        if reset:
            self.reset()
        return result
