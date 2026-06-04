"""SODL-backed vector index storage and retrieval."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sodl_weights.artifact_store import ArtifactMetadata, ArtifactStore


@dataclass
class VectorIndexShard:
    shard_id: int
    vector_blob_id: str
    record_blob_id: str
    size: int


@dataclass
class VectorIndexManifest:
    origin_id: str
    index_name: str
    corpus_version: str
    metric: str
    dim: int
    total_vectors: int
    created_at: str
    shards: list[VectorIndexShard] = field(default_factory=list)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "origin_id": self.origin_id,
                    "index_name": self.index_name,
                    "corpus_version": self.corpus_version,
                    "metric": self.metric,
                    "dim": self.dim,
                    "total_vectors": self.total_vectors,
                    "created_at": self.created_at,
                    "shards": [asdict(shard) for shard in self.shards],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndexManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            origin_id=payload["origin_id"],
            index_name=payload["index_name"],
            corpus_version=payload["corpus_version"],
            metric=payload["metric"],
            dim=int(payload["dim"]),
            total_vectors=int(payload["total_vectors"]),
            created_at=payload["created_at"],
            shards=[VectorIndexShard(**shard) for shard in payload.get("shards", [])],
        )


@dataclass
class VectorSearchResult:
    item_id: str
    score: float
    metadata: dict[str, Any]
    shard_id: int


class SODLVectorIndex:
    """Persist vector index shards as SODL artifacts with lineage metadata."""

    def __init__(
        self,
        artifact_store: ArtifactStore,
        manifest_dir: str | Path = "vector_indexes",
    ) -> None:
        self._artifact_store = artifact_store
        self._manifest_dir = Path(manifest_dir)
        self._manifest_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _utcnow() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _safe_manifest_stem(origin_id: str, index_name: str) -> str:
        return f"{origin_id.replace(':', '__')}-{index_name}"

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        return vectors / safe_norms

    def build(
        self,
        origin_id: str,
        vectors: np.ndarray,
        *,
        ids: Sequence[str] | None = None,
        metadata: Sequence[dict[str, Any]] | None = None,
        index_name: str = "vector-index",
        corpus_version: str = "v1",
        shard_size: int = 1024,
        metric: str = "cosine",
    ) -> VectorIndexManifest:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"vectors must be 2D, got shape {matrix.shape}")
        total_vectors, dim = matrix.shape
        if ids is not None and len(ids) != total_vectors:
            raise ValueError("ids length must match number of vectors")
        if metadata is not None and len(metadata) != total_vectors:
            raise ValueError("metadata length must match number of vectors")
        if metric not in {"cosine", "dot"}:
            raise ValueError("metric must be 'cosine' or 'dot'")

        working = self._normalize_vectors(matrix) if metric == "cosine" else matrix
        item_ids = list(ids or [f"vec:{i}" for i in range(total_vectors)])
        item_metadata = list(metadata or [{} for _ in range(total_vectors)])

        shards: list[VectorIndexShard] = []
        for start in range(0, total_vectors, shard_size):
            end = min(start + shard_size, total_vectors)
            shard_id = len(shards)
            vector_meta = self._artifact_store.store_numpy(
                origin_id,
                working[start:end],
                f"{index_name}-vectors-{shard_id}",
                tags={
                    "artifact_kind": "vector_index_shard",
                    "index_name": index_name,
                    "corpus_version": corpus_version,
                    "metric": metric,
                },
            )
            record_meta = self._artifact_store.store_json(
                origin_id,
                [
                    {"item_id": item_ids[i], "metadata": item_metadata[i]}
                    for i in range(start, end)
                ],
                f"{index_name}-records-{shard_id}",
                tags={
                    "artifact_kind": "vector_index_records",
                    "index_name": index_name,
                    "corpus_version": corpus_version,
                },
            )
            shards.append(
                VectorIndexShard(
                    shard_id=shard_id,
                    vector_blob_id=vector_meta.blob_id,
                    record_blob_id=record_meta.blob_id,
                    size=end - start,
                )
            )

        manifest = VectorIndexManifest(
            origin_id=origin_id,
            index_name=index_name,
            corpus_version=corpus_version,
            metric=metric,
            dim=dim,
            total_vectors=total_vectors,
            created_at=self._utcnow(),
            shards=shards,
        )
        manifest.save(
            self._manifest_dir / f"{self._safe_manifest_stem(origin_id, index_name)}.json"
        )
        return manifest

    def load_manifest(self, path: str | Path) -> VectorIndexManifest:
        return VectorIndexManifest.load(path)

    def query(
        self,
        manifest: VectorIndexManifest | str | Path,
        query_vector: np.ndarray,
        *,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        loaded_manifest = (
            manifest
            if isinstance(manifest, VectorIndexManifest)
            else self.load_manifest(manifest)
        )
        query = np.asarray(query_vector, dtype=np.float32)
        if query.shape != (loaded_manifest.dim,):
            raise ValueError(
                f"query_vector must have shape {(loaded_manifest.dim,)}, got {query.shape}"
            )
        if loaded_manifest.metric == "cosine":
            query = self._normalize_vectors(query.reshape(1, -1))[0]

        results: list[VectorSearchResult] = []
        for shard in loaded_manifest.shards:
            vectors = self._artifact_store.load_numpy(shard.vector_blob_id)
            records = self._artifact_store.load_json(shard.record_blob_id)
            scores = vectors @ query
            order = np.argsort(scores)[::-1]
            for idx in order[:top_k]:
                record = records[int(idx)]
                metadata = dict(record.get("metadata") or {})
                if metadata_filter and any(
                    metadata.get(key) != value for key, value in metadata_filter.items()
                ):
                    continue
                results.append(
                    VectorSearchResult(
                        item_id=str(record["item_id"]),
                        score=float(scores[int(idx)]),
                        metadata=metadata,
                        shard_id=shard.shard_id,
                    )
                )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]


__all__ = [
    "SODLVectorIndex",
    "VectorIndexManifest",
    "VectorIndexShard",
    "VectorSearchResult",
]
