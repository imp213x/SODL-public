"""Training data quality scoring and curriculum helpers for SODL."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from sodl_weights.artifact_store import ArtifactMetadata, ArtifactStore

_TOKEN_RE = re.compile(r"\w+")


@dataclass
class QualityRecord:
    chunk_id: str
    origin_id: str
    quality_score: float
    signal_scores: dict[str, float]
    created_at: str
    source_blob_id: str | None = None
    loss_before: float | None = None
    loss_after: float | None = None
    loss_improvement: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "QualityRecord":
        return cls(**json.loads(raw))


class DataQualityScorer:
    """Score training chunks and persist append-only JSONL quality records."""

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self._artifact_store = artifact_store

    @staticmethod
    def _utcnow() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def score_text_chunk(
        self,
        origin_id: str,
        chunk_id: str,
        text: str,
        *,
        source_blob_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        loss_before: float | None = None,
        loss_after: float | None = None,
    ) -> QualityRecord:
        stripped = text.strip()
        tokens = self._tokenize(stripped)
        token_count = len(tokens)
        unique_ratio = (len(set(tokens)) / token_count) if token_count else 0.0
        length_score = min(token_count / 128.0, 1.0) if token_count else 0.0
        repetition_penalty = 1.0 - min(max(1.0 - unique_ratio, 0.0), 0.75)
        newline_bonus = min(stripped.count("\n") / 8.0, 1.0)
        punctuation_penalty = min(sum(ch in "{}[]<>" for ch in stripped) / max(len(stripped), 1), 0.3)
        code_bonus = min(
            sum(keyword in stripped for keyword in ("def ", "class ", "import ", "return ", "SELECT ", "{", "};"))
            / 4.0,
            1.0,
        )
        loss_improvement = None
        if loss_before is not None and loss_after is not None:
            loss_improvement = float(loss_before - loss_after)
        improvement_bonus = 0.0 if loss_improvement is None else max(min(loss_improvement, 1.0), -1.0)

        signal_scores = {
            "length": round(length_score, 6),
            "lexical_diversity": round(unique_ratio, 6),
            "repetition_penalty": round(repetition_penalty, 6),
            "structure": round(min(newline_bonus + code_bonus, 1.0), 6),
            "punctuation_penalty": round(punctuation_penalty, 6),
            "loss_improvement": round(improvement_bonus, 6),
        }
        quality_score = (
            0.28 * signal_scores["length"]
            + 0.26 * signal_scores["lexical_diversity"]
            + 0.18 * signal_scores["repetition_penalty"]
            + 0.18 * signal_scores["structure"]
            + 0.16 * signal_scores["loss_improvement"]
            - 0.08 * signal_scores["punctuation_penalty"]
        )
        quality_score = max(0.0, min(1.0, quality_score))

        return QualityRecord(
            chunk_id=chunk_id,
            origin_id=origin_id,
            quality_score=round(quality_score, 6),
            signal_scores=signal_scores,
            created_at=self._utcnow(),
            source_blob_id=source_blob_id,
            loss_before=loss_before,
            loss_after=loss_after,
            loss_improvement=loss_improvement,
            metadata=dict(metadata or {}),
        )

    def score_samples(
        self,
        origin_id: str,
        samples: Sequence[dict[str, Any]],
        *,
        text_field: str = "text",
        chunk_id_field: str = "chunk_id",
    ) -> list[QualityRecord]:
        records: list[QualityRecord] = []
        for index, sample in enumerate(samples):
            text = str(sample.get(text_field, ""))
            chunk_id = str(sample.get(chunk_id_field, f"chunk:{index}"))
            records.append(
                self.score_text_chunk(
                    origin_id,
                    chunk_id,
                    text,
                    source_blob_id=sample.get("source_blob_id"),
                    metadata={
                        key: value
                        for key, value in sample.items()
                        if key not in {text_field, chunk_id_field, "source_blob_id", "loss_before", "loss_after"}
                    },
                    loss_before=sample.get("loss_before"),
                    loss_after=sample.get("loss_after"),
                )
            )
        return records

    def store_records(
        self,
        origin_id: str,
        records: Iterable[QualityRecord],
        *,
        name: str = "data-quality",
        tags: dict[str, str] | None = None,
    ) -> ArtifactMetadata:
        payload = "\n".join(record.to_json() for record in records).encode("utf-8")
        return self._artifact_store.store(
            origin_id,
            payload,
            f"{name}.jsonl",
            tags={
                "artifact_kind": "data_quality_jsonl",
                **dict(tags or {}),
            },
        )

    def load_records(self, blob_id: str) -> list[QualityRecord]:
        payload = self._artifact_store.load(blob_id).decode("utf-8")
        return [
            QualityRecord.from_json(line)
            for line in payload.splitlines()
            if line.strip()
        ]

    @staticmethod
    def rank_records(records: Sequence[QualityRecord], *, descending: bool = True) -> list[QualityRecord]:
        return sorted(records, key=lambda record: record.quality_score, reverse=descending)

    @staticmethod
    def curriculum(records: Sequence[QualityRecord], *, min_score: float = 0.0) -> list[str]:
        return [
            record.chunk_id
            for record in DataQualityScorer.rank_records(records)
            if record.quality_score >= min_score
        ]


__all__ = [
    "QualityRecord",
    "DataQualityScorer",
]
