import pytest
import time
import threading

from sodl_weights.metrics import MetricsCollector, HistogramBucket, Timer, get_metrics


class TestHistogramBucket:
    def test_observe(self):
        h = HistogramBucket()
        h.observe(0.05)
        h.observe(0.1)
        h.observe(1.5)
        assert h.count == 3
        assert h.mean == pytest.approx(0.55, abs=0.01)
        assert h.min == pytest.approx(0.05)
        assert h.max == pytest.approx(1.5)

    def test_percentile_bucket(self):
        h = HistogramBucket()
        for i in range(100):
            h.observe(i * 0.01)
        p50 = h.percentile_bucket(0.5)
        assert p50 > 0

    def test_to_dict(self):
        h = HistogramBucket()
        h.observe(0.5)
        d = h.to_dict()
        assert "count" in d and "p50" in d and "p95" in d


class TestMetricsCollector:
    def test_counter(self):
        m = MetricsCollector(prefix="test")
        m.increment("ops")
        m.increment("ops")
        m.increment("ops", 3)
        assert m.get_counter("ops") == 5.0

    def test_gauge(self):
        m = MetricsCollector(prefix="test")
        m.gauge("connections", 42)
        assert m.get_gauge("connections") == 42
        m.gauge("connections", 10)
        assert m.get_gauge("connections") == 10

    def test_histogram(self):
        m = MetricsCollector(prefix="test")
        m.observe("latency", 0.1)
        m.observe("latency", 0.5)
        h = m.get_histogram("latency")
        assert h is not None
        assert h["count"] == 2

    def test_timer(self):
        m = MetricsCollector(prefix="test")
        with m.timer("op_duration"):
            time.sleep(0.01)
        h = m.get_histogram("op_duration")
        assert h is not None
        assert h["count"] == 1
        assert h["mean"] > 0

    def test_snapshot(self):
        m = MetricsCollector(prefix="test")
        m.increment("a")
        m.gauge("b", 1)
        m.observe("c", 0.1)
        snap = m.snapshot()
        assert "counters" in snap
        assert "gauges" in snap
        assert "histograms" in snap
        assert "uptime_sec" in snap

    def test_reset(self):
        m = MetricsCollector(prefix="test")
        m.increment("x")
        m.reset()
        assert m.get_counter("x") == 0

    def test_thread_safety(self):
        m = MetricsCollector(prefix="ts")
        def worker():
            for _ in range(100):
                m.increment("threaded")
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert m.get_counter("threaded") == 400

    def test_get_metrics_singleton(self):
        import sodl_weights.metrics as mod
        mod._default = None
        m1 = get_metrics("singleton_test")
        m2 = get_metrics("singleton_test")
        assert m1 is m2
