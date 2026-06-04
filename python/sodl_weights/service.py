"""Weight Store Service - unified high-level facade."""

from __future__ import annotations

import atexit
import json
from pathlib import Path
from typing import Optional, Sequence

from . import _rust_bridge
from sodl_weights.crypto import CryptoProvider
from sodl_weights.optimizer_state import OptimizerStateStore
from sodl_weights.pin_registry import WeightPinRegistry
from sodl_weights.store import BlobStore, WeightBlobStore
from sodl_weights.types import (
    ImportSummary,
    StoreStats,
    WeightCluster,
    WeightOrigin,
    WeightPinReason,
)


def _is_placeholder_cluster(cluster: WeightCluster) -> bool:
    return (
        cluster.dim == 0
        and not cluster.centroid
        and not cluster.member_token_ids
        and not cluster.offsets
    )


class WeightStoreService:
    """High-level weight store combining persistent blob storage with hot caching."""

    def __init__(
        self,
        blob_dir: str,
        crypto: Optional[CryptoProvider] = None,
        compression_level: int = 3,
        cache_capacity: int = 256,
        serialization_codec: str = "json",
        source_blob_dirs: Sequence[str | Path] | None = None,
        peer_urls: Sequence[str] | None = None,
        edge_urls: Sequence[str] | None = None,
        pin_registry_path: str | Path | None = None,
        pin_registry: WeightPinRegistry | None = None,
    ) -> None:
        blob_store = BlobStore(
            blob_dir,
            source_blob_dirs,
            peer_urls=peer_urls,
            edge_urls=edge_urls,
        )
        self._store = WeightBlobStore(
            blob_store,
            crypto,
            compression_level,
            serialization_codec=serialization_codec,
        )
        self._pin_registry_path = (
            Path(pin_registry_path) if pin_registry_path is not None else None
        )
        if pin_registry is not None:
            self._cache = pin_registry
        elif self._pin_registry_path is not None and self._pin_registry_path.exists():
            self._cache = WeightPinRegistry.load(
                self._pin_registry_path, max_entries=cache_capacity
            )
        else:
            self._cache = WeightPinRegistry(cache_capacity)
        self._origins: dict[str, WeightOrigin] = {}
        self._name_index: dict[str, str] = {}
        self._origin_registry = _rust_bridge.create_weight_origin_registry()
        self._closed = False
        atexit.register(self.close)

    @property
    def pin_registry(self) -> WeightPinRegistry:
        return self._cache

    @property
    def blob_root(self) -> Path:
        return Path(self._store._store._root)

    def save_pin_registry(self, path: str | Path | None = None) -> Path | None:
        target = Path(path) if path is not None else self._pin_registry_path
        if target is None:
            return None
        self._cache.save(target)
        return target

    def close(self) -> None:
        if self._closed:
            return
        self.save_pin_registry()
        self._closed = True

    def _cache_origin(self, origin: WeightOrigin) -> WeightOrigin:
        self._origins[origin.origin_id] = origin
        self._name_index[origin.model_name] = origin.origin_id
        return origin

    def _origin_from_payload(self, payload: str) -> WeightOrigin:
        data = json.loads(payload)
        origin = WeightOrigin(
            origin_id=str(data["origin_id"]),
            model_name=str(data["model_name"]),
            num_clusters=int(data.get("num_clusters", 0)),
            quantization=str(data.get("quantization", "")),
        )
        return self._cache_origin(origin)

    def _increment_origin_clusters(self, origin_id: str, amount: int) -> None:
        if amount <= 0:
            return
        if self._origin_registry is not None:
            self._origin_registry.increment_clusters(origin_id, int(amount))
        origin = self._origins.get(origin_id)
        if origin is not None:
            origin.num_clusters += int(amount)

    # ----- Model lifecycle -------------------------------------------------

    def create_model(self, model_name: str, quantization: str) -> WeightOrigin:
        if self._origin_registry is not None:
            return self._origin_from_payload(
                self._origin_registry.create_model(model_name, quantization)
            )
        origin = WeightOrigin.new(model_name, quantization)
        return self._cache_origin(origin)

    def register_model(
        self,
        origin_id: str,
        model_name: str,
        quantization: str,
        *,
        num_clusters: int = 0,
    ) -> WeightOrigin:
        if self._origin_registry is not None:
            return self._origin_from_payload(
                self._origin_registry.register_model(
                    origin_id,
                    model_name,
                    quantization,
                    int(num_clusters),
                )
            )
        existing = self._origins.get(origin_id)
        if existing is not None:
            self._name_index[existing.model_name] = existing.origin_id
            self._name_index[model_name] = existing.origin_id
            return existing

        origin = WeightOrigin(
            origin_id=str(origin_id),
            model_name=str(model_name),
            num_clusters=max(0, int(num_clusters)),
            quantization=str(quantization),
        )
        self._origins[origin.origin_id] = origin
        self._name_index[origin.model_name] = origin.origin_id
        return origin

    def ensure_model(
        self,
        model_name: str,
        quantization: str,
        *,
        origin_id: str | None = None,
        num_clusters: int = 0,
    ) -> WeightOrigin:
        if self._origin_registry is not None:
            return self._origin_from_payload(
                self._origin_registry.ensure_model(
                    model_name,
                    quantization,
                    origin_id,
                    int(num_clusters),
                )
            )
        if origin_id:
            return self.register_model(
                origin_id,
                model_name,
                quantization,
                num_clusters=num_clusters,
            )

        existing_id = self._name_index.get(model_name)
        if existing_id is not None:
            return self._origins[existing_id]

        return self.create_model(model_name, quantization)

    def get_model(self, origin_id: str) -> WeightOrigin:
        if self._origin_registry is not None:
            return self._origin_from_payload(self._origin_registry.get_model(origin_id))
        origin = self._origins.get(origin_id)
        if origin is None:
            raise KeyError(f"Model origin not found: {origin_id}")
        return origin

    def get_model_by_name(self, model_name: str) -> WeightOrigin:
        if self._origin_registry is not None:
            return self._origin_from_payload(
                self._origin_registry.get_model_by_name(model_name)
            )
        oid = self._name_index.get(model_name)
        if oid is None:
            raise KeyError(f"Model not found: {model_name}")
        return self._origins[oid]

    # ----- Cluster storage -------------------------------------------------

    def store_cluster(self, origin_id: str, cluster: WeightCluster) -> StoreStats:
        stats = self._store.put(origin_id, cluster)
        if not stats.was_deduped:
            self._increment_origin_clusters(origin_id, 1)
        return stats

    def import_clusters(
        self, origin_id: str, clusters: list[WeightCluster]
    ) -> ImportSummary:
        cluster_ids: list[str] = []
        total_blobs = 0
        deduped = 0
        total_raw = 0
        total_stored = 0

        for cluster in clusters:
            stats = self._store.put(origin_id, cluster)
            cluster_ids.append(stats.blob_id)
            total_raw += stats.raw_bytes
            total_stored += stats.stored_bytes
            if stats.was_deduped:
                deduped += 1
            else:
                total_blobs += 1

        self._increment_origin_clusters(origin_id, total_blobs)

        return ImportSummary(
            origin_id=origin_id,
            total_clusters=len(clusters),
            total_blobs_stored=total_blobs,
            deduped_blobs=deduped,
            total_raw_bytes=total_raw,
            total_stored_bytes=total_stored,
            cluster_ids=cluster_ids,
        )

    # ----- Cluster loading (cache-first) ----------------------------------

    def load_cluster(self, origin_id: str, blob_id: str) -> WeightCluster:
        cached = self._cache.get(blob_id)
        if cached is not None and not _is_placeholder_cluster(cached):
            return cached

        cluster = self._store.get(origin_id, blob_id)
        if not self._cache.hydrate(blob_id, cluster):
            self._cache.pin(blob_id, cluster, WeightPinReason.FREQUENT_USE)
        return cluster

    def pin_identity_cluster(self, origin_id: str, blob_id: str) -> None:
        cluster = self._cache.get(blob_id)
        if cluster is None or _is_placeholder_cluster(cluster):
            cluster = self._store.get(origin_id, blob_id)
        self._cache.pin(blob_id, cluster, WeightPinReason.IDENTITY)

    def pin_logic_cluster(self, origin_id: str, blob_id: str) -> None:
        cluster = self._cache.get(blob_id)
        if cluster is None or _is_placeholder_cluster(cluster):
            cluster = self._store.get(origin_id, blob_id)
        self._cache.pin(blob_id, cluster, WeightPinReason.LOGIC)

    def prefetch_cluster(self, origin_id: str, blob_id: str) -> None:
        cached = self._cache.get(blob_id)
        if cached is not None and not _is_placeholder_cluster(cached):
            return
        cluster = self._store.get(origin_id, blob_id)
        if not self._cache.hydrate(blob_id, cluster):
            self._cache.pin(blob_id, cluster, WeightPinReason.PREFETCH)

    def prefetch_clusters(self, origin_id: str, blob_ids: Sequence[str]) -> int:
        loaded = 0
        for blob_id in blob_ids:
            cached = self._cache.get(blob_id)
            if cached is not None and not _is_placeholder_cluster(cached):
                continue
            cluster = self._store.get(origin_id, blob_id)
            if not self._cache.hydrate(blob_id, cluster):
                self._cache.pin(blob_id, cluster, WeightPinReason.PREFETCH)
            loaded += 1
        return loaded

    def evict_cluster(self, blob_id: str) -> bool:
        return self._cache.unpin(blob_id)

    def delete_blob(self, blob_id: str) -> None:
        self._store.delete(blob_id)

    # ----- Introspection ---------------------------------------------------

    def is_cached(self, blob_id: str) -> bool:
        return self._cache.is_pinned(blob_id)

    def cluster_refcount(self, blob_id: str) -> int:
        return self._cache.refcount(blob_id)

    def cache_size(self) -> int:
        return len(self._cache)

    def replica_nodes(self, blob_id: str) -> list[str]:
        return self._store._store.replica_nodes(blob_id)

    # ----- Optimizer offload helpers ---------------------------------------

    def create_optimizer_state_store(
        self,
        *,
        registry_dir: str | Path | None = None,
        cache_capacity: int = 32,
        writeback_threshold: int = 8,
    ) -> OptimizerStateStore:
        blob_root = self._store._store._root
        target_registry = (
            Path(registry_dir)
            if registry_dir is not None
            else blob_root / "optimizer_registry"
        )
        return OptimizerStateStore(
            blob_root,
            target_registry,
            compression_level=self._store._compression_level,
            cache_capacity=cache_capacity,
            writeback_threshold=writeback_threshold,
        )

    @staticmethod
    def configure_optimizer_hot_cache(
        optimizer_store: OptimizerStateStore, capacity: int
    ) -> None:
        optimizer_store.set_cache_capacity(capacity)

    @staticmethod
    def pin_optimizer_blocks(
        optimizer_store: OptimizerStateStore, origin_id: str, block_ids: list[str]
    ) -> None:
        optimizer_store.pin_blocks(origin_id, block_ids)

    @staticmethod
    def release_optimizer_blocks(
        optimizer_store: OptimizerStateStore, origin_id: str, block_ids: list[str]
    ) -> None:
        optimizer_store.unpin_blocks(origin_id, block_ids)

    @staticmethod
    def flush_optimizer_state(
        optimizer_store: OptimizerStateStore, origin_id: str
    ) -> None:
        optimizer_store.flush_origin(origin_id)

    @staticmethod
    def staged_flush_optimizer_state(
        optimizer_store: OptimizerStateStore, origin_id: str, block_ids: list[str]
    ) -> None:
        optimizer_store.flush_blocks(origin_id, block_ids)

    @staticmethod
    def evict_optimizer_blocks(
        optimizer_store: OptimizerStateStore, origin_id: str, block_ids: list[str]
    ) -> int:
        return optimizer_store.evict_blocks(origin_id, block_ids)
