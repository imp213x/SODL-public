"""SODL Semantic Router — Python SDK wrapper for sodl-semantic-router.

Mirrors the Rust ``SemanticRouter`` to select processing pipelines
based on document content type. Routes training documents to the
appropriate preprocessing strategy.

Routes:
    code      → function-boundary-aware chunking
    textbook  → semantic sentence-level chunking
    reasoning → proof-step-aware chunking
    qa        → turn-boundary chunking
    generic   → default stride-based windowing

Usage::

    from sodl_weights.semantic_router import SemanticRouteSelector

    router = SemanticRouteSelector()
    route = router.route(text)
    print(f"Pipeline: {route.pipeline}, stride: {route.stride}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class RouteDecision:
    """Routing decision for a training document."""
    pipeline: str        # code, textbook, reasoning, qa, generic
    stride: int          # recommended stride for windowing
    chunk_boundary: str  # what defines a chunk boundary
    confidence: float    # 0.0-1.0


# Signal patterns for content type detection
_CODE_SIGNALS = re.compile(
    r"(def |class |import |from |function |return |const |var |pub fn |async fn |=>|{|})",
    re.MULTILINE,
)
_MATH_SIGNALS = re.compile(
    r"(theorem|proof|lemma|equation|therefore|hence|Q\.E\.D)",
    re.IGNORECASE,
)
_QA_SIGNALS = re.compile(
    r"(^Q:|^A:|^Question:|^Answer:|^User:|^Assistant:|^Human:)",
    re.MULTILINE | re.IGNORECASE,
)


class SemanticRouteSelector:
    """Content-based pipeline routing matching Rust ``SemanticRouter``.

    Analyzes document content and selects the optimal preprocessing
    pipeline parameters (stride, chunk boundaries, etc.).
    """

    # Pipeline configurations
    ROUTES = {
        "code": RouteDecision(
            pipeline="code",
            stride=256,
            chunk_boundary="function_boundary",
            confidence=0.0,
        ),
        "reasoning": RouteDecision(
            pipeline="reasoning",
            stride=384,
            chunk_boundary="proof_step",
            confidence=0.0,
        ),
        "qa": RouteDecision(
            pipeline="qa",
            stride=256,
            chunk_boundary="turn_boundary",
            confidence=0.0,
        ),
        "textbook": RouteDecision(
            pipeline="textbook",
            stride=512,
            chunk_boundary="sentence_boundary",
            confidence=0.0,
        ),
        "generic": RouteDecision(
            pipeline="generic",
            stride=512,
            chunk_boundary="fixed_stride",
            confidence=0.0,
        ),
    }

    def route(self, text: str, *, default_stride: int = 512) -> RouteDecision:
        """Route a document to the best preprocessing pipeline.

        Parameters
        ----------
        text : str
            Document text (first 3000 chars are analyzed).
        default_stride : int
            Fallback stride for generic documents.
        """
        sample = text[:3000]
        n_chars = max(1, len(sample))

        # Score each pipeline
        code_score = len(_CODE_SIGNALS.findall(sample)) / n_chars * 1000
        math_score = len(_MATH_SIGNALS.findall(sample)) / n_chars * 1000
        qa_score = len(_QA_SIGNALS.findall(sample)) / n_chars * 1000

        # Pick the best
        scores = {
            "code": code_score,
            "reasoning": math_score,
            "qa": qa_score,
        }

        best = max(scores, key=scores.get)
        best_score = scores[best]

        if best_score > 0.5:
            route = RouteDecision(
                pipeline=self.ROUTES[best].pipeline,
                stride=self.ROUTES[best].stride,
                chunk_boundary=self.ROUTES[best].chunk_boundary,
                confidence=min(1.0, best_score / 5.0),
            )
        elif any(kw in sample.lower() for kw in ["textbook", "chapter", "section", "abstract", "introduction"]):
            route = RouteDecision(
                pipeline="textbook",
                stride=512,
                chunk_boundary="sentence_boundary",
                confidence=0.6,
            )
        else:
            route = RouteDecision(
                pipeline="generic",
                stride=default_stride,
                chunk_boundary="fixed_stride",
                confidence=0.3,
            )

        return route

    def route_batch(
        self, texts: list[str], *, default_stride: int = 512
    ) -> list[RouteDecision]:
        """Route a batch of documents."""
        return [self.route(t, default_stride=default_stride) for t in texts]

    def metrics(self) -> dict:
        """Return router info for dashboard."""
        return {
            "router_enabled": True,
            "available_pipelines": list(self.ROUTES.keys()),
        }


# ── Backwards-compatible shims (imported by sodl_weights/__init__.py) ──

@dataclass(slots=True)
class CapabilityQuery:
    """Legacy compatibility: mirrors Rust CapabilityQuery."""
    principal: str = ""
    query_text: str = ""
    requested_caps: list[str] | None = None
    max_results: int = 5

    def __post_init__(self):
        if self.requested_caps is None:
            self.requested_caps = []


@dataclass(slots=True)
class RouteCandidate:
    """Legacy compatibility: mirrors Rust RouteCandidate."""
    label: str = ""
    score: int = 1
    basis: str = "semantic"


def simple_route(labels: list[str], query: CapabilityQuery) -> dict:
    """Legacy compatibility: mirrors Rust SemanticRouter::simple_route."""
    candidates = [
        RouteCandidate(label=label, score=1, basis="semantic")
        for label in labels[:query.max_results]
    ]
    return {
        "allow": bool(query.requested_caps),
        "candidates": candidates,
    }

