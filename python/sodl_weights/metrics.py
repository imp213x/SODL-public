"""Metrics Collector — counters, gauges, histograms for SODL operations.

Provides lightweight, in-process metrics for observability without
external dependencies. Thread-safe.

Example
-------
>>> metrics = MetricsCollector()
>>> metrics.increment("blobs.put")
>>> with metrics.timer("blobs.put.latency"):
...     store.put(blob_id, data)
>>> metrics.snapshot()
"""

from __future__ import annotations

import time
import threading
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HistogramBucket:
    """A histogram with configurable bucket boundaries."""
    boundaries: list[float] = field(default_factory=lambda: [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    ])
    counts: list[int] = field(default_factory=list)
    total: float = 0.0
    count: int = 0
    _min: float = float("inf")
    _max: float = float("-inf")

    def __post_init__(self):
        if not self.counts:
            self.counts = [0] * (len(self.boundaries) + 1)  # +1 for overflow

    def observe(self, value: float) -> None:
        self.total += value
        self.count += 1
        self._min = min(self._min, value)
        self._max = max(self._max, value)
        for i, bound in enumerate(self.boundaries):
            if value <= bound:
                self.counts[i] += 1
                return
        self.counts[-1] += 1  # overflow bucket

    @property
    def mean(self) -> float:
        return self.total / max(self.count, 1)

    @property
    def min(self) -> float:
        return self._min if self._min != float("inf") else 0.0

    @property
    def max(self) -> float:
        return self._max if self._max != float("-inf") else 0.0

    def percentile_bucket(self, p: float) -> float:
        """Estimate the bucket boundary for a given percentile."""
        target = p * self.count
        cumulative = 0
        for i, c in enumerate(self.counts):
            cumulative += c
            if cumulative >= target:
                return self.boundaries[i] if i < len(self.boundaries) else self.boundaries[-1]
        return self.boundaries[-1] if self.boundaries else 0.0

    def to_dict(self) -> dict:
        return {
            "count": self.count, "total": round(self.total, 4),
            "mean": round(self.mean, 4), "min": round(self.min, 6),
            "max": round(self.max, 4), "p50": self.percentile_bucket(0.5),
            "p95": self.percentile_bucket(0.95), "p99": self.percentile_bucket(0.99),
        }


class Timer:
    """Context manager for timing operations."""

    def __init__(self, collector: "MetricsCollector", name: str) -> None:
        self._collector = collector
        self._name = name
        self._start = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        elapsed = time.perf_counter() - self._start
        self._collector.observe(self._name, elapsed)


class MetricsCollector:
    """Thread-safe metrics collector with counters, gauges, and histograms.

    Parameters
    ----------
    prefix : str
        Prefix for all metric names (e.g. "sodl").
    """

    def __init__(self, prefix: str = "sodl") -> None:
        self._prefix = prefix
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, HistogramBucket] = {}
        self._lock = threading.Lock()
        self._start_time = time.time()

    def _key(self, name: str) -> str:
        return f"{self._prefix}.{name}" if self._prefix else name

    def increment(self, name: str, value: float = 1.0) -> None:
        """Increment a counter."""
        key = self._key(name)
        with self._lock:
            self._counters[key] += value

    def gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        key = self._key(name)
        with self._lock:
            self._gauges[key] = value

    def observe(self, name: str, value: float) -> None:
        """Record an observation in a histogram."""
        key = self._key(name)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = HistogramBucket()
            self._histograms[key].observe(value)

    def timer(self, name: str) -> Timer:
        """Create a timing context manager."""
        return Timer(self, name)

    def get_counter(self, name: str) -> float:
        key = self._key(name)
        with self._lock:
            return self._counters.get(key, 0.0)

    def get_gauge(self, name: str) -> float:
        key = self._key(name)
        with self._lock:
            return self._gauges.get(key, 0.0)

    def get_histogram(self, name: str) -> Optional[dict]:
        key = self._key(name)
        with self._lock:
            h = self._histograms.get(key)
            return h.to_dict() if h else None

    def snapshot(self) -> dict:
        """Produce a snapshot of all metrics."""
        with self._lock:
            return {
                "uptime_sec": round(time.time() - self._start_time, 1),
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: v.to_dict() for k, v in self._histograms.items()},
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._start_time = time.time()


# Global default collector
_default: Optional[MetricsCollector] = None


def get_metrics(prefix: str = "sodl") -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _default
    if _default is None:
        _default = MetricsCollector(prefix)
    return _default
