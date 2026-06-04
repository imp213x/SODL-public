"""SODL training lifecycle helpers for clustered weight workflows.

This module keeps training-adjacent SODL behaviors in the SDK boundary so
application repos like Carla do not own long-lived weight-store control logic.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover - optional torch path
    HAS_TORCH = False

from sodl_weights.types import WeightCluster
from sodl_weights.weight_manifest import WeightManifestStore, export_manifest_clusters

logger = logging.getLogger(__name__)


def _to_numpy_matrix(embedding_matrix: Any) -> np.ndarray:
    if HAS_TORCH and torch.is_tensor(embedding_matrix):
        return embedding_matrix.detach().cpu().to(dtype=torch.float32).numpy()
    return np.asarray(embedding_matrix, dtype=np.float32)


def cluster_ids_for_token_batch(token_index: Any, token_ids: Any) -> set[int]:
    if token_index is None:
        return set()
    token_array = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    cluster_ids: set[int] = set()
    vocab_size = int(getattr(token_index, "vocab_size", 0))
    for token_id in token_array:
        tid = int(token_id)
        if 0 <= tid < vocab_size:
            cluster_ids.add(int(token_index.token_hash(tid)))
    return cluster_ids


def build_weight_cluster(
    token_index: Any,
    cluster_id: int,
    embedding_matrix: Any | None = None,
) -> WeightCluster:
    member_ids = list(token_index.cluster_members(cluster_id))
    if not member_ids:
        raise ValueError(f"Cluster {cluster_id} has no members")

    if embedding_matrix is not None:
        matrix = _to_numpy_matrix(embedding_matrix)
        members = matrix[np.asarray(member_ids, dtype=np.int64)]
        centroid = members.mean(axis=0, dtype=np.float32)
        offsets = members - centroid
    else:
        centroid = np.asarray(token_index.get_centroid(cluster_id), dtype=np.float32)
        offsets = np.stack(
            [np.asarray(token_index.get_offset(token_id), dtype=np.float32) for token_id in member_ids],
            axis=0,
        )

    return WeightCluster(
        centroid=centroid.astype(np.float32).tolist(),
        member_token_ids=member_ids,
        offsets=offsets.astype(np.float32).tolist(),
        dim=int(centroid.shape[0]),
        cluster_id=str(cluster_id),
    )


def build_weight_clusters(
    token_index: Any,
    embedding_matrix: Any | None = None,
    cluster_ids: Iterable[int] | None = None,
) -> dict[int, WeightCluster]:
    resolved_ids = sorted(set(int(cluster_id) for cluster_id in (cluster_ids or range(token_index.n_clusters))))
    return {
        cluster_id: build_weight_cluster(token_index, cluster_id, embedding_matrix)
        for cluster_id in resolved_ids
        if token_index.cluster_members(cluster_id)
    }


def export_token_clusters(
    weight_service: Any,
    origin_id: str,
    token_index: Any,
    embedding_matrix: Any | None = None,
    cluster_ids: Iterable[int] | None = None,
) -> dict[int, str]:
    blob_ids: dict[int, str] = {}
    for cluster_id, cluster in build_weight_clusters(token_index, embedding_matrix, cluster_ids).items():
        stats = weight_service.store_cluster(origin_id, cluster)
        blob_ids[cluster_id] = str(getattr(stats, "blob_id"))
    return blob_ids


def _manifest_store(path: str | Path, weight_service: Any | None = None) -> WeightManifestStore:
    blob_root = None
    if weight_service is not None and hasattr(weight_service, "blob_root"):
        blob_root = weight_service.blob_root
    return WeightManifestStore(path, blob_root=blob_root)


def load_sodl_manifest(path: str | Path) -> dict[str, Any] | None:
    manifest = _manifest_store(path).load_manifest()
    if manifest is None:
        return None
    return {
        "origin_id": manifest.origin_id,
        "vocab_size": manifest.vocab_size,
        "embedding_dim": manifest.embedding_dim,
        "clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "blob_id": cluster.blob_id,
                "member_token_ids": list(cluster.member_token_ids),
            }
            for cluster in manifest.clusters
        ],
        "metadata": dict(manifest.metadata),
    }


def resolve_sodl_origin_id(
    manifest_path: str | Path,
    *,
    checkpoint_origin: str,
    resume_record: Mapping[str, Any] | None = None,
) -> tuple[str | None, str]:
    return _manifest_store(manifest_path).resolve_origin_id(
        checkpoint_origin,
        resume_record=resume_record,
    )


def write_sodl_manifest(
    path: str | Path,
    origin_id: str,
    token_index: Any,
    cluster_blob_ids: Mapping[int, str],
    *,
    metadata: Mapping[str, Any] | None = None,
    weight_service: Any | None = None,
) -> Path:
    manifest_path = Path(path)
    store = _manifest_store(manifest_path, weight_service=weight_service)
    clusters = export_manifest_clusters(token_index, cluster_blob_ids)
    store.write_manifest(
        origin_id,
        int(getattr(token_index, "vocab_size", 0)),
        int(getattr(token_index, "dim", 0)),
        clusters,
        metadata=metadata,
    )
    return manifest_path


class VeinPrefetcher:
    """Pre-load SODL weight clusters in a background thread."""

    def __init__(
        self,
        weight_service: Any,
        origin_id: str,
        prefetch_depth: int = 2,
        cluster_blob_ids: Mapping[Any, str] | None = None,
    ) -> None:
        self._service = weight_service
        self._origin_id = origin_id
        self._depth = prefetch_depth
        self._cluster_blob_ids: dict[Any, str] = dict(cluster_blob_ids or {})
        self._request_queue: queue.Queue[set[Any] | None] = queue.Queue(maxsize=prefetch_depth * 2)
        self._cache: dict[Any, Any] = {}
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._stats = {"prefetch_requests": 0, "cache_hits": 0, "cache_misses": 0}

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="sodl-vein-prefetcher",
        )
        self._worker_thread.start()
        logger.info("VeinPrefetcher started (depth=%d)", self._depth)

    def stop(self) -> None:
        self._running = False
        self._request_queue.put(None)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        logger.info("VeinPrefetcher stopped. Stats: %s", self._stats)

    def update_blob_ids(self, cluster_blob_ids: Mapping[Any, str]) -> None:
        self._cluster_blob_ids.update(cluster_blob_ids)

    def _resolve_blob_id(self, cluster_id: Any) -> str:
        return self._cluster_blob_ids.get(cluster_id, str(cluster_id))

    def request_prefetch(self, cluster_ids: set[Any]) -> None:
        if not self._running:
            return
        try:
            self._request_queue.put_nowait(cluster_ids)
        except queue.Full:
            pass

    def get_cluster(self, cluster_id: Any) -> Any:
        with self._lock:
            if cluster_id in self._cache:
                self._stats["cache_hits"] += 1
                return self._cache[cluster_id]

        self._stats["cache_misses"] += 1
        blob_id = self._resolve_blob_id(cluster_id)
        cluster = self._service.load_cluster(self._origin_id, blob_id)
        with self._lock:
            self._cache[cluster_id] = cluster
        return cluster

    def _prefetch_blob_ids(self, blob_ids: list[str]) -> None:
        if not blob_ids:
            return
        if hasattr(self._service, "prefetch_clusters"):
            self._service.prefetch_clusters(self._origin_id, blob_ids)
            return
        for blob_id in blob_ids:
            if hasattr(self._service, "prefetch_cluster"):
                self._service.prefetch_cluster(self._origin_id, blob_id)

    def _worker(self) -> None:
        while self._running:
            try:
                request = self._request_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if request is None:
                break

            self._stats["prefetch_requests"] += 1
            requested_ids = [cluster_id for cluster_id in request if cluster_id is not None]
            missing_ids: list[Any] = []
            with self._lock:
                for cluster_id in requested_ids:
                    if cluster_id not in self._cache:
                        missing_ids.append(cluster_id)
            try:
                blob_pairs = [(cluster_id, self._resolve_blob_id(cluster_id)) for cluster_id in missing_ids]
                self._prefetch_blob_ids([blob_id for _, blob_id in blob_pairs])
                for cluster_id, blob_id in blob_pairs:
                    if not self._running:
                        break
                    cluster = self._service.load_cluster(self._origin_id, blob_id)
                    with self._lock:
                        self._cache[cluster_id] = cluster
            except Exception as exc:
                logger.warning("Prefetch failed for clusters %s: %s", missing_ids, exc)

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)


class ArteryPulsar:
    """Incrementally export updated weight clusters to SODL."""

    def __init__(
        self,
        weight_service: Any,
        origin_id: str,
        pulse_interval: float = 30.0,
        min_dirty_clusters: int = 5,
        cluster_blob_ids: Mapping[Any, str] | None = None,
        manifest_path: str | Path | None = None,
        token_index: Any | None = None,
    ) -> None:
        self._service = weight_service
        self._origin_id = origin_id
        self._interval = pulse_interval
        self._min_dirty = min_dirty_clusters
        self._cluster_blob_ids: dict[Any, str] = dict(cluster_blob_ids or {})
        self._manifest_path = Path(manifest_path) if manifest_path is not None else None
        self._token_index = token_index
        self._dirty_clusters: dict[Any, Any] = {}
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._stats = {"pulses": 0, "clusters_exported": 0, "bytes_exported": 0}

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="sodl-artery-pulsar",
        )
        self._worker_thread.start()
        logger.info(
            "ArteryPulsar started (interval=%.1fs, min_dirty=%d)",
            self._interval,
            self._min_dirty,
        )

    def stop(self) -> None:
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10.0)
        self._pulse(force=True)
        logger.info("ArteryPulsar stopped. Stats: %s", self._stats)

    def mark_dirty(self, cluster_id: Any, cluster: Any) -> None:
        with self._lock:
            self._dirty_clusters[cluster_id] = cluster

    def _worker(self) -> None:
        while self._running:
            time.sleep(self._interval)
            if not self._running:
                break
            self._pulse()

    def _pulse(self, force: bool = False) -> None:
        with self._lock:
            if not force and len(self._dirty_clusters) < self._min_dirty:
                return
            to_export = dict(self._dirty_clusters)
            self._dirty_clusters.clear()

        if not to_export:
            return

        self._stats["pulses"] += 1
        exported_bytes = 0
        manifest_updated = False

        for cluster_id, cluster in to_export.items():
            try:
                stats = self._service.store_cluster(self._origin_id, cluster)
                self._stats["clusters_exported"] += 1
                exported_bytes += int(getattr(stats, "stored_bytes", 0))
                if hasattr(stats, "blob_id"):
                    self._cluster_blob_ids[cluster_id] = str(stats.blob_id)
                    manifest_updated = True
            except Exception as exc:
                logger.warning("Export failed for cluster %s: %s", cluster_id, exc)
                with self._lock:
                    self._dirty_clusters[cluster_id] = cluster

        self._stats["bytes_exported"] += exported_bytes
        if manifest_updated and self._manifest_path is not None and self._token_index is not None:
            write_sodl_manifest(
                self._manifest_path,
                self._origin_id,
                self._token_index,
                self._cluster_blob_ids,
                weight_service=self._service,
            )
        if hasattr(self._service, "save_pin_registry"):
            self._service.save_pin_registry()
        logger.info("Pulse: exported %d clusters (%d bytes)", len(to_export), exported_bytes)

    def flush(self) -> None:
        self._pulse(force=True)

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    @property
    def dirty_count(self) -> int:
        with self._lock:
            return len(self._dirty_clusters)
