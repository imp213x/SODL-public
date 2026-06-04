import pytest
import time
from pathlib import Path

from sodl_weights.tracer import OperationTracer, Span


class TestSpan:
    def test_basic(self):
        s = Span(trace_id="t1", span_id="s1", operation="test")
        s.start_time = time.time()
        s.set_tag("key", "value")
        s.add_event("checkpoint")
        s.finish()
        assert s.duration_ms >= 0
        assert s.tags["key"] == "value"
        assert len(s.events) == 1

    def test_error(self):
        s = Span(trace_id="t1", span_id="s1", operation="test")
        s.set_error("boom")
        assert s.status == "error"
        assert s.tags["error"] == "boom"

    def test_to_dict(self):
        s = Span(trace_id="t1", span_id="s1", operation="test")
        d = s.to_dict()
        assert d["trace_id"] == "t1"
        assert d["operation"] == "test"


class TestOperationTracer:
    @pytest.fixture
    def tracer(self):
        return OperationTracer(service_name="test")

    def test_span_context_manager(self, tracer):
        with tracer.span("blob.put", size=100) as s:
            time.sleep(0.005)
        assert tracer.span_count == 1
        traces = tracer.get_traces()
        assert traces[0]["operation"] == "blob.put"
        assert traces[0]["duration_ms"] >= 1

    def test_span_error(self, tracer):
        with pytest.raises(ValueError):
            with tracer.span("fail") as s:
                raise ValueError("test error")
        traces = tracer.get_traces()
        assert traces[0]["status"] == "error"
        assert "test error" in traces[0]["tags"]["error"]

    def test_nested_spans(self, tracer):
        with tracer.span("parent") as parent:
            with tracer.span("child", parent_id=parent.span_id):
                pass
        assert tracer.span_count == 2
        traces = tracer.get_traces()
        child = [t for t in traces if t["operation"] == "child"][0]
        assert child["parent_id"] == parent.span_id

    def test_get_trace(self, tracer):
        tid = tracer.new_trace_id()
        with tracer.span("a", trace_id=tid):
            pass
        with tracer.span("b", trace_id=tid):
            pass
        spans = tracer.get_trace(tid)
        assert len(spans) == 2

    def test_slow_spans(self, tracer):
        with tracer.span("fast"):
            pass
        with tracer.span("slow"):
            time.sleep(0.05)
        slow = tracer.get_slow_spans(threshold_ms=10)
        assert len(slow) >= 1
        assert slow[0]["operation"] == "slow"

    def test_summary(self, tracer):
        for _ in range(5):
            with tracer.span("op"):
                pass
        s = tracer.summary()
        assert "op" in s
        assert s["op"]["count"] == 5

    def test_export(self, tracer, tmp_path):
        with tracer.span("export_test"):
            pass
        path = tracer.export_traces(tmp_path / "traces.json")
        assert Path(path).exists()

    def test_clear(self, tracer):
        with tracer.span("x"):
            pass
        tracer.clear()
        assert tracer.span_count == 0

    def test_max_traces(self):
        tracer = OperationTracer(max_traces=5)
        for i in range(10):
            with tracer.span(f"op_{i}"):
                pass
        assert tracer.span_count == 5
