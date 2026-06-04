"""Semantic Context Logic (SCL) primitives for clustered attention and memory."""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sodl_weights.artifact_store import ArtifactMetadata, ArtifactStore
from sodl_weights.token_hash import TokenHashIndex
from sodl_weights.types import WeightCluster


@dataclass
class SCLQueryResult:
    attention_output: np.ndarray
    active_clusters: list[int]
    retrieved_indices: list[int]
    mode: str


@dataclass
class SCLMemoryManifest:
    origin_id: str
    created_at: str
    cluster_blob_ids: list[str]
    total_records: int
    dim: int
    modes: list[str]
    memory_records: list[dict[str, Any]]


@dataclass
class SCLTokenPrediction:
    token_id: int
    probability: float
    cluster_id: int
    mode: str


class SemanticContextLogic:
    """Clustered KV attention with gated retrieval and append-only memory saveback."""

    def __init__(self, n_clusters: int = 32, top_k_clusters: int = 3) -> None:
        self._n_clusters = n_clusters
        self._top_k_clusters = top_k_clusters
        self._keys: np.ndarray | None = None
        self._values: np.ndarray | None = None
        self._centroids: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._cluster_members: dict[int, list[int]] = {}
        self._mode_pins: dict[str, set[int]] = {}
        self._knowledge_records: list[dict[str, Any]] = []

    @staticmethod
    def _utcnow() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _cluster(data: np.ndarray, n_clusters: int, max_iter: int = 25) -> tuple[np.ndarray, np.ndarray]:
        effective_clusters = min(max(1, n_clusters), len(data))
        try:
            from sklearn.cluster import MiniBatchKMeans

            kmeans = MiniBatchKMeans(
                n_clusters=effective_clusters,
                batch_size=min(len(data), 1024),
                max_iter=max_iter,
                random_state=42,
                n_init=3,
            )
            labels = kmeans.fit_predict(data).astype(np.int32, copy=False)
            return kmeans.cluster_centers_.astype(np.float32, copy=False), labels
        except ImportError:
            rng = np.random.RandomState(42)
            centroids = data[rng.choice(len(data), effective_clusters, replace=False)].copy()
            labels = np.zeros(len(data), dtype=np.int32)
            for _ in range(max_iter):
                distances = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
                labels = np.argmin(distances, axis=1).astype(np.int32, copy=False)
                new_centroids = centroids.copy()
                for cluster_id in range(effective_clusters):
                    mask = labels == cluster_id
                    if np.any(mask):
                        new_centroids[cluster_id] = data[mask].mean(axis=0)
                if np.allclose(new_centroids, centroids):
                    break
                centroids = new_centroids
            return centroids.astype(np.float32, copy=False), labels

    def build(self, keys: np.ndarray, values: np.ndarray) -> dict[str, Any]:
        key_matrix = np.asarray(keys, dtype=np.float32)
        value_matrix = np.asarray(values, dtype=np.float32)
        if key_matrix.shape != value_matrix.shape:
            raise ValueError("keys and values must share the same shape")
        if key_matrix.ndim != 2:
            raise ValueError(f"keys/values must be 2D, got {key_matrix.shape}")

        centroids, labels = self._cluster(key_matrix, self._n_clusters)
        self._keys = key_matrix
        self._values = value_matrix
        self._centroids = centroids
        self._labels = labels
        self._cluster_members = {
            cluster_id: np.where(labels == cluster_id)[0].astype(int).tolist()
            for cluster_id in range(len(centroids))
            if np.any(labels == cluster_id)
        }
        sizes = [len(members) for members in self._cluster_members.values()]
        return {
            "n_clusters": len(self._cluster_members),
            "cluster_size_min": min(sizes),
            "cluster_size_max": max(sizes),
            "cluster_size_mean": float(np.mean(sizes)),
        }

    def detect_mode(self, context_text: str) -> str:
        text = context_text.strip().lower()
        if any(token in text for token in ("def ", "class ", "import ", "return ", "{", "};", "```")):
            return "code"
        if text.endswith("?") or text.startswith(("who", "what", "when", "where", "why", "how")):
            return "qa"
        if any(token in text for token in ("therefore", "because", "premise", "conclusion")):
            return "logic"
        return "narrative"

    def pin_clusters_for_mode(self, mode: str, cluster_ids: Sequence[int]) -> None:
        self._mode_pins.setdefault(mode, set()).update(int(cluster_id) for cluster_id in cluster_ids)

    @staticmethod
    def _embed_text(text: str, dim: int) -> np.ndarray:
        """Deterministic lightweight text embedding for persistent SCL memory."""
        tokens = [token for token in text.lower().split() if token]
        if not tokens:
            return np.zeros(dim, dtype=np.float32)
        acc = np.zeros(dim, dtype=np.float32)
        for token in tokens:
            seed = int.from_bytes(
                hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(),
                "little",
            )
            rng = np.random.RandomState(seed & 0x7FFFFFFF)
            acc += rng.normal(0.0, 1.0, size=dim).astype(np.float32)
        norm = np.linalg.norm(acc)
        if norm > 0:
            acc = acc / norm
        return acc.astype(np.float32, copy=False)

    def query(
        self,
        query: np.ndarray,
        *,
        top_k_clusters: int | None = None,
        top_k_items: int = 8,
        context_text: str | None = None,
    ) -> SCLQueryResult:
        if self._centroids is None or self._keys is None or self._values is None or self._labels is None:
            raise RuntimeError("SCL index not built")

        query_vector = np.asarray(query, dtype=np.float32)
        if query_vector.shape != (self._centroids.shape[1],):
            raise ValueError(
                f"query must have shape {(self._centroids.shape[1],)}, got {query_vector.shape}"
            )

        mode = self.detect_mode(context_text or "")
        cluster_scores = self._centroids @ query_vector
        requested = top_k_clusters or self._top_k_clusters
        active_clusters = np.argsort(cluster_scores)[-requested:][::-1].astype(int).tolist()

        pinned = self._mode_pins.get(mode, set())
        added_pins = 0
        for cluster_id in pinned:
            if cluster_id not in active_clusters and cluster_id in self._cluster_members:
                active_clusters.append(cluster_id)
                added_pins += 1
        active_clusters = active_clusters[: requested + added_pins]

        candidate_indices: list[int] = []
        for cluster_id in active_clusters:
            candidate_indices.extend(self._cluster_members.get(cluster_id, []))
        if not candidate_indices:
            candidate_indices = list(range(len(self._keys)))

        candidate_keys = self._keys[candidate_indices]
        scores = candidate_keys @ query_vector
        order = np.argsort(scores)[::-1][:top_k_items]
        retrieved_indices = [candidate_indices[int(i)] for i in order]
        retrieved_scores = scores[order]
        logits = retrieved_scores - np.max(retrieved_scores)
        weights = np.exp(logits)
        weights = weights / max(np.sum(weights), 1e-8)
        attention_output = np.sum(
            self._values[retrieved_indices] * weights[:, None],
            axis=0,
        ).astype(np.float32, copy=False)

        return SCLQueryResult(
            attention_output=attention_output,
            active_clusters=active_clusters,
            retrieved_indices=retrieved_indices,
            mode=mode,
        )

    def record_memory(
        self,
        origin_id: str,
        *,
        kind: str,
        content: str,
        query: str | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "context_logic",
    ) -> dict[str, Any]:
        record = {
            "ts": self._utcnow(),
            "source": source,
            "origin_id": origin_id,
            "kind": kind,
            "query": query,
            "content": content,
            "metadata": dict(metadata or {}),
        }
        self._knowledge_records.append(record)
        return record

    def persist_memory_log(
        self,
        origin_id: str,
        *,
        artifact_store: ArtifactStore | None = None,
        path: str | Path | None = None,
        name: str = "scl-memory",
    ) -> ArtifactMetadata | Path:
        payload = "\n".join(json.dumps(record, sort_keys=True) for record in self._knowledge_records).encode("utf-8")
        if artifact_store is not None:
            return artifact_store.store(
                origin_id,
                payload,
                f"{name}.jsonl",
                tags={"artifact_kind": "scl_memory_jsonl"},
            )
        if path is None:
            raise ValueError("Provide either artifact_store or path to persist the knowledge log")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return target

    def persist_memory_to_weight_store(
        self,
        origin_id: str,
        *,
        weight_store_service: Any,
        model_name: str = "scl-memory",
        quantization: str = "float32",
        dim: int = 64,
        pin_logic_modes: bool = True,
    ) -> SCLMemoryManifest:
        """Persist recorded memory into the SODL weight store as semantic clusters."""
        if not self._knowledge_records:
            raise ValueError("No memory records available to persist")

        try:
            weight_store_service.get_model(origin_id)
        except Exception:
            weight_store_service.create_model(model_name, quantization)
            created_origin = weight_store_service.get_model_by_name(model_name)
            origin_id = created_origin.origin_id

        records = list(self._knowledge_records)
        embeddings = np.stack(
            [
                self._embed_text(
                    " ".join(
                        part
                        for part in (
                            str(record.get("query") or ""),
                            str(record.get("content") or ""),
                            json.dumps(record.get("metadata") or {}, sort_keys=True),
                        )
                        if part
                    ),
                    dim,
                )
                for record in records
            ]
        ).astype(np.float32, copy=False)
        cluster_count = min(max(1, self._n_clusters), len(records))
        centroids, labels = self._cluster(embeddings, cluster_count)

        mode_to_cluster_ids: dict[str, set[int]] = {}
        cluster_blob_ids: list[str] = []
        for cluster_id in range(len(centroids)):
            member_indices = np.where(labels == cluster_id)[0].astype(int).tolist()
            if not member_indices:
                continue
            cluster_vectors = embeddings[member_indices]
            centroid = centroids[cluster_id]
            offsets = (cluster_vectors - centroid).tolist()
            cluster = WeightCluster(
                centroid=centroid.tolist(),
                member_token_ids=member_indices,
                offsets=offsets,
                dim=dim,
                cluster_id=f"memory:{cluster_id}",
            )
            stats = weight_store_service.store_cluster(origin_id, cluster)
            cluster_blob_ids.append(stats.blob_id)
            for member_index in member_indices:
                mode = self.detect_mode(
                    " ".join(
                        part
                        for part in (
                            str(records[member_index].get("query") or ""),
                            str(records[member_index].get("content") or ""),
                        )
                        if part
                    )
                )
                mode_to_cluster_ids.setdefault(mode, set()).add(cluster_id)
                if pin_logic_modes and mode in {"logic", "code"}:
                    weight_store_service.pin_logic_cluster(origin_id, stats.blob_id)

        for mode, cluster_ids in mode_to_cluster_ids.items():
            self.pin_clusters_for_mode(mode, sorted(cluster_ids))

        return SCLMemoryManifest(
            origin_id=origin_id,
            created_at=self._utcnow(),
            cluster_blob_ids=cluster_blob_ids,
            total_records=len(records),
            dim=dim,
            modes=sorted(mode_to_cluster_ids),
            memory_records=records,
        )

    def hydrate_memory_from_weight_store(
        self,
        origin_id: str,
        cluster_blob_ids: Sequence[str],
        *,
        weight_store_service: Any,
    ) -> dict[str, Any]:
        """Load persisted SCL memory clusters back into the live context index."""
        keys: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for blob_id in cluster_blob_ids:
            cluster = weight_store_service.load_cluster(origin_id, blob_id)
            centroid = np.asarray(cluster.centroid, dtype=np.float32)
            offsets = np.asarray(cluster.offsets, dtype=np.float32)
            for offset in offsets:
                vector = centroid + offset
                keys.append(vector)
                values.append(vector)
        if not keys:
            raise ValueError("No persisted memory vectors could be hydrated")
        return self.build(np.stack(keys), np.stack(values))


class ClusteredAttentionLayer:
    """Batch-oriented clustered attention built on Semantic Context Logic."""

    def __init__(self, n_clusters: int = 32, top_k_clusters: int = 3) -> None:
        self._logic = SemanticContextLogic(
            n_clusters=n_clusters,
            top_k_clusters=top_k_clusters,
        )

    def forward(
        self,
        queries: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        *,
        context_texts: Sequence[str] | None = None,
        top_k_items: int = 8,
    ) -> tuple[np.ndarray, list[SCLQueryResult]]:
        stats = self._logic.build(keys, values)
        _ = stats  # build stats are useful during debugging but not returned here
        query_matrix = np.asarray(queries, dtype=np.float32)
        if query_matrix.ndim == 1:
            query_matrix = query_matrix.reshape(1, -1)
        if query_matrix.ndim != 2:
            raise ValueError(f"queries must be 1D or 2D, got {query_matrix.shape}")

        outputs: list[np.ndarray] = []
        results: list[SCLQueryResult] = []
        for index, query in enumerate(query_matrix):
            result = self._logic.query(
                query,
                context_text=(context_texts[index] if context_texts is not None else None),
                top_k_items=top_k_items,
            )
            outputs.append(result.attention_output)
            results.append(result)
        return np.stack(outputs).astype(np.float32, copy=False), results


class SCLClusteredDecoder:
    """Bridge SCL attention outputs into clustered token prediction."""

    def __init__(
        self,
        token_index: TokenHashIndex,
        *,
        n_clusters: int = 32,
        top_k_clusters: int = 3,
    ) -> None:
        self._token_index = token_index
        self._attention = ClusteredAttentionLayer(
            n_clusters=n_clusters,
            top_k_clusters=top_k_clusters,
        )

    def predict(
        self,
        queries: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        *,
        context_texts: Sequence[str] | None = None,
        top_k_items: int = 8,
        top_k_tokens: int = 5,
    ) -> tuple[list[list[SCLTokenPrediction]], list[SCLQueryResult]]:
        outputs, query_results = self._attention.forward(
            queries,
            keys,
            values,
            context_texts=context_texts,
            top_k_items=top_k_items,
        )
        predictions: list[list[SCLTokenPrediction]] = []
        for index, output in enumerate(outputs):
            ranked = self._token_index.hierarchical_softmax(output)[:top_k_tokens]
            predictions.append(
                [
                    SCLTokenPrediction(
                        token_id=int(item.token_id),
                        probability=float(item.probability),
                        cluster_id=int(item.cluster_id),
                        mode=query_results[index].mode,
                    )
                    for item in ranked
                ]
            )
        return predictions, query_results


__all__ = [
    "SemanticContextLogic",
    "ClusteredAttentionLayer",
    "SCLQueryResult",
    "SCLMemoryManifest",
    "SCLTokenPrediction",
    "SCLClusteredDecoder",
]
