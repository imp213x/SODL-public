"""Operation Tracer — span-based tracing for SODL operations.

Lightweight distributed tracing with correlation IDs for tracking
operations across federated nodes.

Example
-------
>>> tracer = OperationTracer()
>>> with tracer.span("blob.put", blob_id=blob_id) as span:
...     store.put(blob_id, data)
...     span.set_tag("size_bytes", len(data))
>>> tracer.export_traces()
"""

from __future__ import annotations

import time
import uuid
import json
import threading
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """A single traced operation span."""
    trace_id: str
    span_id: str
    operation: str
    parent_id: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    status: str = "ok"
    tags: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    def set_tag(self, key: str, value) -> None:
        self.tags[key] = value

    def add_event(self, name: str, **kwargs) -> None:
        self.events.append({"name": name, "time": time.time(), **kwargs})

    def finish(self) -> None:
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 3)

    def set_error(self, error: str) -> None:
        self.status = "error"
        self.tags["error"] = error

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id, "span_id": self.span_id,
            "operation": self.operation, "parent_id": self.parent_id,
            "start_time": self.start_time, "duration_ms": self.duration_ms,
            "status": self.status, "tags": self.tags, "events": self.events,
        }


class OperationTracer:
    """Span-based operation tracer with correlation IDs.

    Parameters
    ----------
    service_name : str
        Name of this service (for trace context).
    max_traces : int
        Maximum traces to keep in memory.
    export_path : str or Path, optional
        Path to auto-export traces.
    """

    def __init__(
        self,
        service_name: str = "sodl",
        max_traces: int = 10000,
        export_path: Optional[str | Path] = None,
    ) -> None:
        self._service = service_name
        self._max_traces = max_traces
        self._export_path = Path(export_path) if export_path else None
        self._spans: list[Span] = []
        self._lock = threading.Lock()
        self._active_trace: Optional[str] = None

    def new_trace_id(self) -> str:
        """Generate a new trace ID."""
        return uuid.uuid4().hex[:16]

    def new_span_id(self) -> str:
        """Generate a new span ID."""
        return uuid.uuid4().hex[:8]

    @contextmanager
    def span(
        self,
        operation: str,
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        **tags,
    ):
        """Context manager to trace an operation.

        Parameters
        ----------
        operation : str
            Name of the operation being traced.
        trace_id : str, optional
            Correlation ID (auto-generated if None).
        parent_id : str, optional
            Parent span ID for nested spans.
        **tags
            Initial tags for the span.
        """
        s = Span(
            trace_id=trace_id or self._active_trace or self.new_trace_id(),
            span_id=self.new_span_id(),
            operation=operation,
            parent_id=parent_id,
            start_time=time.time(),
            tags={"service": self._service, **tags},
        )

        prev_trace = self._active_trace
        self._active_trace = s.trace_id

        try:
            yield s
        except Exception as e:
            s.set_error(str(e))
            raise
        finally:
            s.finish()
            self._active_trace = prev_trace
            with self._lock:
                self._spans.append(s)
                if len(self._spans) > self._max_traces:
                    self._spans = self._spans[-self._max_traces:]

    def record_span(self, span: Span) -> None:
        """Manually record a completed span."""
        with self._lock:
            self._spans.append(span)

    def get_traces(self, limit: int = 100) -> list[dict]:
        """Get recent trace spans."""
        with self._lock:
            return [s.to_dict() for s in self._spans[-limit:]]

    def get_trace(self, trace_id: str) -> list[dict]:
        """Get all spans for a specific trace ID."""
        with self._lock:
            return [s.to_dict() for s in self._spans if s.trace_id == trace_id]

    def get_slow_spans(self, threshold_ms: float = 100.0, limit: int = 50) -> list[dict]:
        """Get spans exceeding a duration threshold."""
        with self._lock:
            slow = [s for s in self._spans if s.duration_ms >= threshold_ms]
            return [s.to_dict() for s in sorted(slow, key=lambda x: x.duration_ms, reverse=True)[:limit]]

    def export_traces(self, path: Optional[str | Path] = None) -> str:
        """Export all traces to a JSON file."""
        export_path = Path(path) if path else self._export_path
        if export_path is None:
            export_path = Path(f"traces_{self._service}_{int(time.time())}.json")

        export_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                "service": self._service,
                "exported_at": time.time(),
                "span_count": len(self._spans),
                "spans": [s.to_dict() for s in self._spans],
            }
        export_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return str(export_path)

    def summary(self) -> dict:
        """Generate a summary of traced operations."""
        with self._lock:
            ops: dict[str, list[float]] = {}
            for s in self._spans:
                ops.setdefault(s.operation, []).append(s.duration_ms)

        result = {}
        for op, durations in ops.items():
            durations.sort()
            count = len(durations)
            result[op] = {
                "count": count,
                "mean_ms": round(sum(durations) / count, 3),
                "min_ms": round(durations[0], 3),
                "max_ms": round(durations[-1], 3),
                "p50_ms": round(durations[count // 2], 3),
                "p95_ms": round(durations[int(count * 0.95)], 3) if count > 1 else round(durations[0], 3),
                "errors": sum(1 for s in self._spans if s.operation == op and s.status == "error"),
            }
        return result

    @property
    def span_count(self) -> int:
        with self._lock:
            return len(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()
