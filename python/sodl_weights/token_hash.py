"""Token Hash Index — locality-sensitive token hashing for training acceleration.

Clusters vocabulary tokens by embedding similarity, enabling:
1. Hierarchical softmax (O(k + n/k) instead of O(n))
2. Gradient sharing via cluster centroids
3. Sparse vocabulary updates (only active clusters)

This module extends the SODL Weight Store with training-aware indexing.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from sodl_weights.artifact_store import ArtifactMetadata, ArtifactStore
from sodl_weights.types import WeightCluster


@dataclass
class TokenHashEntry:
    """A single token's position in the hash index."""
    token_id: int
    cluster_id: int
    offset: np.ndarray  # residual from cluster centroid


@dataclass
class ClusterInfo:
    """Metadata for a cluster in the hash index."""
    cluster_id: int
    centroid: np.ndarray
    member_ids: list[int]
    size: int

    @property
    def density(self) -> float:
        """Average offset magnitude — lower = tighter cluster."""
        return 0.0  # set externally after building


@dataclass
class HierarchicalSoftmaxResult:
    """Result from a hierarchical softmax lookup."""
    token_id: int
    probability: float
    cluster_id: int
    cluster_prob: float
    token_prob_within_cluster: float


@dataclass
class IncrementalClusterDelta:
    """Delta between a previous export snapshot and the current cluster state."""

    cluster_id: int
    member_token_ids: list[int]
    centroid_delta: list[float]
    offset_deltas: list[list[float]]


@dataclass
class IncrementalExportResult:
    """Incremental export summary for weight-cluster storage."""

    changed_token_ids: list[int]
    changed_cluster_ids: list[int]
    unchanged_cluster_ids: list[int]
    clusters: list[WeightCluster]
    deltas: list[IncrementalClusterDelta]
    delta_savings_ratio: float


@dataclass
class IncrementalExportArtifactManifest:
    """Persisted incremental-export manifest stored through SODL artifacts."""

    origin_id: str
    export_name: str
    created_at: str
    changed_token_ids: list[int]
    changed_cluster_ids: list[int]
    unchanged_cluster_ids: list[int]
    delta_savings_ratio: float
    cluster_artifacts: list[ArtifactMetadata] = field(default_factory=list)
    delta_artifacts: list[ArtifactMetadata] = field(default_factory=list)
    manifest_artifact: ArtifactMetadata | None = None
    base_snapshot_artifact_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "origin_id": self.origin_id,
            "export_name": self.export_name,
            "created_at": self.created_at,
            "changed_token_ids": list(self.changed_token_ids),
            "changed_cluster_ids": list(self.changed_cluster_ids),
            "unchanged_cluster_ids": list(self.unchanged_cluster_ids),
            "delta_savings_ratio": float(self.delta_savings_ratio),
            "base_snapshot_artifact_id": self.base_snapshot_artifact_id,
            "cluster_artifacts": [asdict(item) for item in self.cluster_artifacts],
            "delta_artifacts": [asdict(item) for item in self.delta_artifacts],
            "manifest_artifact": (
                asdict(self.manifest_artifact) if self.manifest_artifact is not None else None
            ),
        }


class TokenHashIndex:
    """Locality-Sensitive Token Hash Index.

    Maps each token to a semantic cluster based on embedding similarity.
    Enables hierarchical softmax, gradient sharing, and sparse updates.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (default 512). Should be sqrt(vocab_size) for
        balanced hierarchical softmax.
    top_k_clusters : int
        Number of top clusters to search during hierarchical softmax
        (default 3). Higher = more accurate, slower.
    """

    def __init__(
        self,
        n_clusters: int = 512,
        top_k_clusters: int = 3,
    ) -> None:
        self._n_clusters = n_clusters
        self._top_k = top_k_clusters

        # Populated by build()
        self._centroids: Optional[np.ndarray] = None   # (n_clusters, dim)
        self._labels: Optional[np.ndarray] = None       # (vocab_size,)
        self._offsets: Optional[np.ndarray] = None       # (vocab_size, dim)
        self._cluster_members: dict[int, list[int]] = {}
        self._vocab_size: int = 0
        self._dim: int = 0
        self._is_built = False
        self._last_export_embeddings: Optional[np.ndarray] = None

    @property
    def is_built(self) -> bool:
        return self._is_built

    @property
    def n_clusters(self) -> int:
        return self._n_clusters

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def _utcnow() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # -----------------------------------------------------------------------
    # Build the index
    # -----------------------------------------------------------------------

    def build(
        self,
        embeddings: np.ndarray,
        max_iter: int = 50,
        fit_sample_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        n_init: int = 3,
        adaptive: bool = False,
        max_cluster_ratio: float = 5.0,
        uniform_ratio_threshold: float = 1.5,
        max_rebalance_steps: int = 8,
    ) -> dict:
        """Build the token hash index from an embedding matrix.

        Parameters
        ----------
        embeddings : np.ndarray, shape (vocab_size, dim)
            The full embedding weight matrix.
        max_iter : int
            Max K-means iterations.

        Returns
        -------
        dict
            Build statistics: cluster sizes, inertia, convergence info.
        """
        vocab_size, dim = embeddings.shape
        self._vocab_size = vocab_size
        self._dim = dim
        data = embeddings.astype(np.float32)
        rng = np.random.RandomState(42)
        fit_data = data
        fit_sample_actual: Optional[int] = None
        sample_indices: Optional[np.ndarray] = None
        initial_n_clusters = self._n_clusters

        if adaptive:
            initial_n_clusters = max(2, int(round(math.sqrt(max(vocab_size, 1)))))
        initial_n_clusters = min(initial_n_clusters, vocab_size)

        if fit_sample_size is not None and 0 < fit_sample_size < vocab_size:
            fit_sample_actual = min(int(fit_sample_size), vocab_size)
            sample_indices = rng.choice(vocab_size, fit_sample_actual, replace=False)
            fit_data = data[sample_indices]

        kmeans_batch_size = batch_size or min(4096, max(len(fit_data), 1))
        effective_clusters = min(initial_n_clusters, max(len(fit_data), 1))

        # K-means clustering
        try:
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(
                n_clusters=effective_clusters,
                batch_size=kmeans_batch_size,
                max_iter=max_iter,
                random_state=42,
                n_init=n_init,
            )
            fit_labels = kmeans.fit_predict(fit_data)
            centroids = kmeans.cluster_centers_
            iteration = kmeans.n_iter_
            labels = np.empty(vocab_size, dtype=np.int32)

            if fit_sample_actual is None:
                labels = fit_labels.astype(np.int32, copy=False)
            else:
                assign_batch_size = max(1024, min(4096, vocab_size))
                for i in range(0, vocab_size, assign_batch_size):
                    batch = data[i:i + assign_batch_size]
                    dists = np.sum(
                        (batch[:, None, :] - centroids[None, :, :]) ** 2, axis=2
                    )
                    labels[i:i + assign_batch_size] = np.argmin(dists, axis=1)
        except ImportError:
            # Fallback to manual numpy implementation
            idx = rng.choice(len(fit_data), effective_clusters, replace=False)
            centroids = fit_data[idx].copy()

            fit_labels = np.zeros(len(fit_data), dtype=np.int32)
            for iteration in range(max_iter):
                # Assign (batched for memory efficiency)
                assign_batch_size = max(1024, min(4096, len(fit_data)))
                for i in range(0, len(fit_data), assign_batch_size):
                    batch = fit_data[i:i + assign_batch_size]
                    dists = np.sum(
                        (batch[:, None, :] - centroids[None, :, :]) ** 2, axis=2
                    )
                    fit_labels[i:i + assign_batch_size] = np.argmin(dists, axis=1)

                # Update centroids
                new_centroids = np.zeros_like(centroids)
                for k in range(effective_clusters):
                    mask = fit_labels == k
                    if np.any(mask):
                        new_centroids[k] = fit_data[mask].mean(axis=0)
                    else:
                        new_centroids[k] = centroids[k]

                shift = float(np.sum((new_centroids - centroids) ** 2))
                centroids = new_centroids

                if shift < 1e-6:
                    break

            labels = np.empty(vocab_size, dtype=np.int32)
            assign_batch_size = max(1024, min(4096, vocab_size))
            for i in range(0, vocab_size, assign_batch_size):
                batch = data[i:i + assign_batch_size]
                dists = np.sum(
                    (batch[:, None, :] - centroids[None, :, :]) ** 2, axis=2
                )
                labels[i:i + assign_batch_size] = np.argmin(dists, axis=1)

        centroids, labels, rebalance_steps = _rebalance_cluster_layout(
            data,
            centroids,
            labels,
            adaptive=adaptive,
            max_cluster_ratio=max_cluster_ratio,
            uniform_ratio_threshold=uniform_ratio_threshold,
            max_rebalance_steps=max_rebalance_steps,
        )

        # Compute offsets (residuals)
        offsets = data - centroids[labels]

        # Compute inertia (total squared distance to centroids)
        inertia = float(np.sum(offsets ** 2))
        silhouette = _estimate_silhouette_score(data, labels)

        # Build member index
        cluster_members: dict[int, list[int]] = {}
        for k in range(len(centroids)):
            members = np.where(labels == k)[0].tolist()
            if members:
                cluster_members[k] = members

        # Store
        self._centroids = centroids
        self._labels = labels
        self._offsets = offsets
        self._cluster_members = cluster_members
        self._n_clusters = len(cluster_members)
        self._is_built = True

        # Stats
        sizes = [len(m) for m in cluster_members.values()]
        offset_norms = np.linalg.norm(offsets, axis=1)

        return {
            "n_clusters": len(cluster_members),
            "initial_n_clusters": int(initial_n_clusters),
            "vocab_size": vocab_size,
            "dim": dim,
            "iterations": iteration + 1,
            "fit_sample_size": fit_sample_actual or vocab_size,
            "inertia": inertia,
            "silhouette_score": silhouette,
            "cluster_size_min": min(sizes),
            "cluster_size_max": max(sizes),
            "cluster_size_mean": float(np.mean(sizes)),
            "cluster_size_ratio": float(max(sizes) / max(min(sizes), 1)),
            "offset_norm_mean": float(np.mean(offset_norms)),
            "offset_norm_max": float(np.max(offset_norms)),
            "compression_potential": float(
                1.0 - (np.sum(offsets ** 2) / np.sum(data ** 2))
            ),
            "adaptive_rebalanced": bool(rebalance_steps > 0),
            "rebalance_steps": int(rebalance_steps),
        }

    # -----------------------------------------------------------------------
    # Token hash lookup
    # -----------------------------------------------------------------------

    def token_hash(self, token_id: int) -> int:
        """Get the cluster (hash) for a token."""
        assert self._is_built, "Index not built — call build() first"
        return int(self._labels[token_id])

    def cluster_members(self, cluster_id: int) -> list[int]:
        """Get all token IDs in a cluster."""
        return self._cluster_members.get(cluster_id, [])

    def get_centroid(self, cluster_id: int) -> np.ndarray:
        """Get a cluster's centroid vector."""
        assert self._is_built
        return self._centroids[cluster_id]

    def get_offset(self, token_id: int) -> np.ndarray:
        """Get a token's offset from its cluster centroid."""
        assert self._is_built
        return self._offsets[token_id]

    def reconstruct(self, token_id: int) -> np.ndarray:
        """Reconstruct a token's embedding from centroid + offset."""
        cid = self.token_hash(token_id)
        return self._centroids[cid] + self._offsets[token_id]

    # -----------------------------------------------------------------------
    # Hierarchical softmax
    # -----------------------------------------------------------------------

    def hierarchical_softmax(
        self,
        hidden_state: np.ndarray,
    ) -> list[HierarchicalSoftmaxResult]:
        """Compute hierarchical softmax: cluster selection → within-cluster ranking.

        Parameters
        ----------
        hidden_state : np.ndarray, shape (dim,)
            The hidden state to compute logits for.

        Returns
        -------
        list[HierarchicalSoftmaxResult]
            Top tokens with their hierarchical probabilities.
        """
        assert self._is_built
        h = hidden_state.astype(np.float32)

        # Step 1: Score clusters via centroid dot product
        cluster_logits = self._centroids @ h  # (n_clusters,)
        cluster_probs = _softmax(cluster_logits)

        # Top-k clusters
        top_cluster_ids = np.argsort(cluster_probs)[-self._top_k:][::-1]

        # Step 2: Score tokens within top clusters
        results = []
        for cid in top_cluster_ids:
            members = self._cluster_members.get(int(cid), [])
            if not members:
                continue

            cp = float(cluster_probs[cid])

            # Reconstruct member embeddings and compute logits
            member_embeddings = (
                self._centroids[cid] + self._offsets[members]
            )  # (n_members, dim)
            member_logits = member_embeddings @ h
            member_probs = _softmax(member_logits)

            for j, mid in enumerate(members):
                results.append(HierarchicalSoftmaxResult(
                    token_id=mid,
                    probability=cp * float(member_probs[j]),
                    cluster_id=int(cid),
                    cluster_prob=cp,
                    token_prob_within_cluster=float(member_probs[j]),
                ))

        # Sort by probability descending
        results.sort(key=lambda r: r.probability, reverse=True)
        return results

    # -----------------------------------------------------------------------
    # Gradient sharing
    # -----------------------------------------------------------------------

    def compute_shared_gradient(
        self,
        cluster_id: int,
        token_gradients: dict[int, np.ndarray],
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Decompose per-token gradients into shared centroid gradient + sparse offsets.

        Parameters
        ----------
        cluster_id : int
        token_gradients : dict[int, ndarray]
            Mapping token_id → gradient vector.

        Returns
        -------
        (centroid_gradient, offset_corrections)
        """
        if not token_gradients:
            return np.zeros(self._dim, dtype=np.float32), {}

        grads = np.stack(list(token_gradients.values()))
        centroid_grad = grads.mean(axis=0)

        offset_corrections = {}
        for tid, grad in token_gradients.items():
            residual = grad - centroid_grad
            # Only keep significant corrections (sparsify)
            mask = np.abs(residual) > 1e-6
            if np.any(mask):
                sparse = np.zeros_like(residual)
                sparse[mask] = residual[mask]
                offset_corrections[tid] = sparse

        return centroid_grad, offset_corrections

    def compute_update_savings(
        self,
        batch_token_ids: list[int],
    ) -> dict:
        """Estimate compute savings for a batch using cluster-aware updates.

        Parameters
        ----------
        batch_token_ids :
            Token IDs appearing in the current batch.

        Returns
        -------
        dict with savings metrics.
        """
        assert self._is_built
        unique_ids = set(batch_token_ids)
        active_clusters = set(self._labels[list(unique_ids)])

        standard_params = self._vocab_size * self._dim
        cluster_params = len(active_clusters) * self._dim  # centroid updates
        offset_params = len(unique_ids) * self._dim         # per-token offsets
        sparse_params = cluster_params + offset_params

        return {
            "unique_tokens": len(unique_ids),
            "active_clusters": len(active_clusters),
            "standard_params_updated": standard_params,
            "sparse_params_updated": sparse_params,
            "savings_ratio": 1.0 - (sparse_params / standard_params),
            "speedup_factor": standard_params / max(sparse_params, 1),
        }

    # -----------------------------------------------------------------------
    # Incremental export / runtime adaptation
    # -----------------------------------------------------------------------

    def export_weight_clusters(
        self,
        embeddings: np.ndarray | None = None,
        *,
        cluster_ids: list[int] | None = None,
    ) -> list[WeightCluster]:
        """Export the current cluster layout as SODL ``WeightCluster`` objects."""
        assert self._is_built, "Index not built — call build() first"

        data = None if embeddings is None else np.asarray(embeddings, dtype=np.float32)
        if data is not None and data.shape != (self._vocab_size, self._dim):
            raise ValueError(
                f"Embeddings shape mismatch: expected {(self._vocab_size, self._dim)}, got {data.shape}"
            )

        selected = sorted(cluster_ids) if cluster_ids is not None else sorted(self._cluster_members)
        clusters: list[WeightCluster] = []
        for cluster_id in selected:
            member_ids = self._cluster_members.get(int(cluster_id), [])
            if not member_ids:
                continue
            centroid = self._centroids[cluster_id]
            if data is None:
                offsets = self._offsets[member_ids]
            else:
                offsets = data[member_ids] - centroid
            clusters.append(
                WeightCluster(
                    centroid=centroid.tolist(),
                    member_token_ids=list(member_ids),
                    offsets=offsets.tolist(),
                    dim=self._dim,
                    cluster_id=str(cluster_id),
                )
            )
        return clusters

    def mark_export_snapshot(self, embeddings: np.ndarray) -> None:
        data = np.asarray(embeddings, dtype=np.float32)
        if data.shape != (self._vocab_size, self._dim):
            raise ValueError(
                f"Embeddings shape mismatch: expected {(self._vocab_size, self._dim)}, got {data.shape}"
            )
        self._last_export_embeddings = data.copy()

    def changed_token_ids(
        self,
        embeddings: np.ndarray,
        *,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> list[int]:
        data = np.asarray(embeddings, dtype=np.float32)
        if data.shape != (self._vocab_size, self._dim):
            raise ValueError(
                f"Embeddings shape mismatch: expected {(self._vocab_size, self._dim)}, got {data.shape}"
            )
        if self._last_export_embeddings is None:
            return list(range(self._vocab_size))
        changed = ~np.isclose(
            data,
            self._last_export_embeddings,
            atol=atol,
            rtol=rtol,
        ).all(axis=1)
        return np.where(changed)[0].astype(int).tolist()

    def export_incremental_clusters(
        self,
        embeddings: np.ndarray,
        *,
        atol: float = 1e-6,
        rtol: float = 1e-5,
        mark_snapshot: bool = True,
    ) -> IncrementalExportResult:
        assert self._is_built, "Index not built — call build() first"
        data = np.asarray(embeddings, dtype=np.float32)
        if data.shape != (self._vocab_size, self._dim):
            raise ValueError(
                f"Embeddings shape mismatch: expected {(self._vocab_size, self._dim)}, got {data.shape}"
            )

        changed_token_ids = self.changed_token_ids(data, atol=atol, rtol=rtol)
        changed_cluster_ids = sorted(
            {int(self._labels[token_id]) for token_id in changed_token_ids}
        )
        unchanged_cluster_ids = sorted(
            set(self._cluster_members).difference(changed_cluster_ids)
        )
        clusters = self.export_weight_clusters(data, cluster_ids=changed_cluster_ids)

        deltas: list[IncrementalClusterDelta] = []
        if self._last_export_embeddings is not None:
            previous = self._last_export_embeddings
            for cluster_id in changed_cluster_ids:
                member_ids = self._cluster_members.get(cluster_id, [])
                if not member_ids:
                    continue
                current_centroid = data[member_ids].mean(axis=0)
                previous_centroid = previous[member_ids].mean(axis=0)
                current_offsets = data[member_ids] - current_centroid
                previous_offsets = previous[member_ids] - previous_centroid
                deltas.append(
                    IncrementalClusterDelta(
                        cluster_id=cluster_id,
                        member_token_ids=list(member_ids),
                        centroid_delta=(current_centroid - previous_centroid).tolist(),
                        offset_deltas=(current_offsets - previous_offsets).tolist(),
                    )
                )

        if mark_snapshot:
            self.mark_export_snapshot(data)

        changed_cluster_count = len(changed_cluster_ids)
        total_cluster_count = max(len(self._cluster_members), 1)
        return IncrementalExportResult(
            changed_token_ids=changed_token_ids,
            changed_cluster_ids=changed_cluster_ids,
            unchanged_cluster_ids=unchanged_cluster_ids,
            clusters=clusters,
            deltas=deltas,
            delta_savings_ratio=1.0 - (changed_cluster_count / total_cluster_count),
        )

    def store_incremental_export(
        self,
        artifact_store: ArtifactStore,
        origin_id: str,
        embeddings: np.ndarray,
        *,
        export_name: str = "token-hash-incremental",
        base_snapshot_artifact_id: str | None = None,
        atol: float = 1e-6,
        rtol: float = 1e-5,
        mark_snapshot: bool = True,
        tags: dict[str, str] | None = None,
    ) -> IncrementalExportArtifactManifest:
        """Persist an incremental export as SODL artifacts.

        Stores only changed clusters plus per-cluster deltas, then writes a
        manifest artifact that ties the export together for lineage/resume use.
        """
        data = np.asarray(embeddings, dtype=np.float32)
        if data.shape != (self._vocab_size, self._dim):
            raise ValueError(
                f"Embeddings shape mismatch: expected {(self._vocab_size, self._dim)}, got {data.shape}"
            )

        result = self.export_incremental_clusters(
            data,
            atol=atol,
            rtol=rtol,
            mark_snapshot=False,
        )
        common_tags = dict(tags or {})
        common_tags.update(
            {
                "export_name": export_name,
                "export_mode": "incremental",
            }
        )

        cluster_artifacts: list[ArtifactMetadata] = []
        for cluster in result.clusters:
            cluster_artifacts.append(
                artifact_store.store_json(
                    origin_id,
                    cluster.to_dict(),
                    f"{export_name}-cluster-{cluster.cluster_id}.json",
                    tags={
                        **common_tags,
                        "artifact_kind": "token_hash_cluster",
                        "cluster_id": str(cluster.cluster_id),
                    },
                )
            )

        delta_artifacts: list[ArtifactMetadata] = []
        for delta in result.deltas:
            delta_artifacts.append(
                artifact_store.store_json(
                    origin_id,
                    asdict(delta),
                    f"{export_name}-delta-{delta.cluster_id}.json",
                    tags={
                        **common_tags,
                        "artifact_kind": "token_hash_cluster_delta",
                        "cluster_id": str(delta.cluster_id),
                    },
                )
            )

        manifest = IncrementalExportArtifactManifest(
            origin_id=origin_id,
            export_name=export_name,
            created_at=self._utcnow(),
            changed_token_ids=list(result.changed_token_ids),
            changed_cluster_ids=list(result.changed_cluster_ids),
            unchanged_cluster_ids=list(result.unchanged_cluster_ids),
            delta_savings_ratio=float(result.delta_savings_ratio),
            cluster_artifacts=cluster_artifacts,
            delta_artifacts=delta_artifacts,
            base_snapshot_artifact_id=base_snapshot_artifact_id,
        )
        manifest_payload = manifest.to_dict()
        manifest_artifact = artifact_store.store_json(
            origin_id,
            manifest_payload,
            f"{export_name}-manifest.json",
            tags={
                **common_tags,
                "artifact_kind": "token_hash_incremental_manifest",
            },
        )
        manifest.manifest_artifact = manifest_artifact

        if mark_snapshot:
            self.mark_export_snapshot(data)

        return manifest

    def adapt(
        self,
        embeddings: np.ndarray,
        *,
        max_cluster_ratio: float = 5.0,
        uniform_ratio_threshold: float = 1.5,
        max_rebalance_steps: int = 8,
    ) -> dict:
        """Rebalance an existing index against updated embeddings during training."""
        assert self._is_built, "Index not built — call build() first"
        data = np.asarray(embeddings, dtype=np.float32)
        if data.shape != (self._vocab_size, self._dim):
            raise ValueError(
                f"Embeddings shape mismatch: expected {(self._vocab_size, self._dim)}, got {data.shape}"
            )

        assign_batch_size = max(1024, min(4096, self._vocab_size))
        labels = np.empty(self._vocab_size, dtype=np.int32)
        for i in range(0, self._vocab_size, assign_batch_size):
            batch = data[i:i + assign_batch_size]
            dists = np.sum(
                (batch[:, None, :] - self._centroids[None, :, :]) ** 2, axis=2
            )
            labels[i:i + assign_batch_size] = np.argmin(dists, axis=1)

        centroids, labels, rebalance_steps = _rebalance_cluster_layout(
            data,
            self._centroids,
            labels,
            adaptive=True,
            max_cluster_ratio=max_cluster_ratio,
            uniform_ratio_threshold=uniform_ratio_threshold,
            max_rebalance_steps=max_rebalance_steps,
        )
        offsets = data - centroids[labels]
        cluster_members = {
            cluster_id: np.where(labels == cluster_id)[0].tolist()
            for cluster_id in range(len(centroids))
            if np.any(labels == cluster_id)
        }
        self._centroids = centroids
        self._labels = labels
        self._offsets = offsets
        self._cluster_members = cluster_members
        self._n_clusters = len(cluster_members)

        sizes = [len(members) for members in cluster_members.values()]
        return {
            "n_clusters": self._n_clusters,
            "cluster_size_min": min(sizes),
            "cluster_size_max": max(sizes),
            "cluster_size_mean": float(np.mean(sizes)),
            "cluster_size_ratio": float(max(sizes) / max(min(sizes), 1)),
            "adaptive_rebalanced": bool(rebalance_steps > 0),
            "rebalance_steps": int(rebalance_steps),
        }

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "centroids.npy", self._centroids)
        np.save(path / "labels.npy", self._labels)
        np.save(path / "offsets.npy", self._offsets)

        meta = {
            "n_clusters": self._n_clusters,
            "top_k": self._top_k,
            "vocab_size": self._vocab_size,
            "dim": self._dim,
            "cluster_members": {
                str(k): v for k, v in self._cluster_members.items()
            },
        }
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: str | Path) -> TokenHashIndex:
        """Load a pre-built index from disk."""
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())

        idx = cls(
            n_clusters=meta["n_clusters"],
            top_k_clusters=meta["top_k"],
        )
        idx._centroids = np.load(path / "centroids.npy")
        idx._labels = np.load(path / "labels.npy")
        idx._offsets = np.load(path / "offsets.npy")
        idx._vocab_size = meta["vocab_size"]
        idx._dim = meta["dim"]
        idx._cluster_members = {
            int(k): v for k, v in meta["cluster_members"].items()
        }
        idx._is_built = True
        return idx


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def _estimate_silhouette_score(
    data: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 2048,
) -> float | None:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return None
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        return None
    if len(data) > sample_size:
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(len(data), sample_size, replace=False)
        sample_data = data[sample_indices]
        sample_labels = labels[sample_indices]
        if len(np.unique(sample_labels)) < 2:
            return None
        return float(silhouette_score(sample_data, sample_labels))
    return float(silhouette_score(data, labels))


def _split_cluster_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(points) < 4:
        return None
    try:
        from sklearn.cluster import MiniBatchKMeans

        kmeans = MiniBatchKMeans(
            n_clusters=2,
            batch_size=min(len(points), 1024),
            max_iter=25,
            random_state=42,
            n_init=3,
        )
        local_labels = kmeans.fit_predict(points).astype(np.int32, copy=False)
        if len(np.unique(local_labels)) < 2:
            return None
        return kmeans.cluster_centers_.astype(np.float32, copy=False), local_labels
    except ImportError:
        center = points.mean(axis=0)
        dists = np.sum((points - center) ** 2, axis=1)
        first = points[int(np.argmax(dists))]
        dists_to_first = np.sum((points - first) ** 2, axis=1)
        second = points[int(np.argmax(dists_to_first))]
        centroids = np.stack([first, second]).astype(np.float32, copy=False)
        for _ in range(10):
            local_dists = np.sum(
                (points[:, None, :] - centroids[None, :, :]) ** 2, axis=2
            )
            local_labels = np.argmin(local_dists, axis=1).astype(np.int32, copy=False)
            if len(np.unique(local_labels)) < 2:
                return None
            new_centroids = np.stack(
                [points[local_labels == i].mean(axis=0) for i in range(2)]
            ).astype(np.float32, copy=False)
            if np.allclose(new_centroids, centroids):
                break
            centroids = new_centroids
        return centroids, local_labels


def _renumber_clusters(
    centroids: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unique_ids = sorted(int(cluster_id) for cluster_id in np.unique(labels))
    mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    remapped_labels = np.array([mapping[int(label)] for label in labels], dtype=np.int32)
    remapped_centroids = np.stack(
        [centroids[old_id] for old_id in unique_ids]
    ).astype(np.float32, copy=False)
    return remapped_centroids, remapped_labels


def _recompute_centroids(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    unique_ids = sorted(int(cluster_id) for cluster_id in np.unique(labels))
    return np.stack(
        [data[labels == cluster_id].mean(axis=0) for cluster_id in unique_ids]
    ).astype(np.float32, copy=False)


def _rebalance_cluster_layout(
    data: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray,
    *,
    adaptive: bool,
    max_cluster_ratio: float,
    uniform_ratio_threshold: float,
    max_rebalance_steps: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    centroids = centroids.astype(np.float32, copy=False)
    labels = labels.astype(np.int32, copy=False)
    if not adaptive:
        centroids, labels = _renumber_clusters(centroids, labels)
        return centroids, labels, 0

    steps = 0
    centroids, labels = _renumber_clusters(centroids, labels)
    while steps < max_rebalance_steps:
        counts = np.bincount(labels, minlength=len(centroids))
        active_counts = counts[counts > 0]
        if len(active_counts) < 2:
            break
        ratio = float(active_counts.max() / max(active_counts.min(), 1))
        if ratio > max_cluster_ratio:
            largest_cluster_id = int(np.argmax(counts))
            member_indices = np.where(labels == largest_cluster_id)[0]
            split = _split_cluster_points(data[member_indices])
            if split is None:
                break
            split_centroids, local_labels = split
            new_cluster_id = len(centroids)
            centroids[largest_cluster_id] = split_centroids[0]
            centroids = np.vstack([centroids, split_centroids[1]])
            labels[member_indices] = np.where(
                local_labels == 0, largest_cluster_id, new_cluster_id
            ).astype(np.int32, copy=False)
            centroids, labels = _renumber_clusters(centroids, labels)
            steps += 1
            continue

        if ratio < uniform_ratio_threshold and len(centroids) > 2:
            smallest_cluster_id = int(np.argmin(counts))
            remaining = [idx for idx in range(len(centroids)) if idx != smallest_cluster_id]
            if not remaining:
                break
            smallest_centroid = centroids[smallest_cluster_id]
            nearest_cluster_id = min(
                remaining,
                key=lambda idx: float(np.sum((centroids[idx] - smallest_centroid) ** 2)),
            )
            labels[labels == smallest_cluster_id] = nearest_cluster_id
            centroids, labels = _renumber_clusters(centroids, labels)
            steps += 1
            continue
        break

    centroids = _recompute_centroids(data, labels)
    centroids, labels = _renumber_clusters(centroids, labels)
    return centroids, labels, steps
