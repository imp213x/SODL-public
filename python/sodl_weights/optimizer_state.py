"""Optimizer state offload store backed by SODL CAS and native manifests."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import zstandard as zstd

from . import _rust_bridge
from sodl_weights.store import BlobStore, compute_blob_id


@dataclass(slots=True)
class OptimizerBlockRecord:
    block_id: str
    blob_id: str
    step: int
    shard_key: str | None
    size_raw: int
    size_stored: int
    stored_at: str
    metadata: dict[str, Any] | None


@dataclass(slots=True)
class OptimizerStateManifest:
    schema: str
    origin_id: str
    blocks: dict[str, OptimizerBlockRecord]
    updated_at: str


@dataclass(slots=True)
class OptimizerStoreResult:
    origin_id: str
    block_id: str
    blob_id: str | None
    step: int
    staged: bool
    flushed: bool
    size_raw: int
    size_stored: int
    dirty_blocks: int
    metadata: dict[str, Any] | None


@dataclass(slots=True)
class OptimizerCacheStats:
    cache_entries: int
    dirty_entries: int
    pinned_entries: int
    cache_capacity: int
    writeback_threshold: int


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _record_from_mapping(payload: dict[str, Any]) -> OptimizerBlockRecord:
    return OptimizerBlockRecord(
        block_id=str(payload["block_id"]),
        blob_id=str(payload["blob_id"]),
        step=int(payload["step"]),
        shard_key=payload.get("shard_key"),
        size_raw=int(payload.get("size_raw", 0)),
        size_stored=int(payload.get("size_stored", 0)),
        stored_at=str(payload.get("stored_at") or _now_iso()),
        metadata=payload.get("metadata"),
    )


def _manifest_from_mapping(payload: dict[str, Any]) -> OptimizerStateManifest:
    raw_blocks = payload.get("blocks", {}) or {}
    blocks = {
        str(block_id): _record_from_mapping(block_payload)
        for block_id, block_payload in raw_blocks.items()
    }
    return OptimizerStateManifest(
        schema=str(payload.get("schema") or "sodl-v1"),
        origin_id=str(payload.get("origin_id") or ""),
        blocks=blocks,
        updated_at=str(payload.get("updated_at") or _now_iso()),
    )


def _result_from_mapping(payload: dict[str, Any]) -> OptimizerStoreResult:
    return OptimizerStoreResult(
        origin_id=str(payload.get("origin_id") or ""),
        block_id=str(payload.get("block_id") or ""),
        blob_id=payload.get("blob_id"),
        step=int(payload.get("step", 0)),
        staged=bool(payload.get("staged", False)),
        flushed=bool(payload.get("flushed", False)),
        size_raw=int(payload.get("size_raw", 0)),
        size_stored=int(payload.get("size_stored", 0)),
        dirty_blocks=int(payload.get("dirty_blocks", 0)),
        metadata=payload.get("metadata"),
    )


def _stats_from_mapping(payload: dict[str, Any]) -> OptimizerCacheStats:
    return OptimizerCacheStats(
        cache_entries=int(payload.get("cache_entries", 0)),
        dirty_entries=int(payload.get("dirty_entries", 0)),
        pinned_entries=int(payload.get("pinned_entries", 0)),
        cache_capacity=int(payload.get("cache_capacity", 0)),
        writeback_threshold=int(payload.get("writeback_threshold", 0)),
    )


class _PythonOptimizerStateBackend:
    def __init__(
        self,
        blob_root: str | Path,
        registry_dir: str | Path | None,
        *,
        compression_level: int,
        cache_capacity: int,
        writeback_threshold: int,
    ) -> None:
        self._blob_store = BlobStore(blob_root)
        self._registry_dir = Path(registry_dir or (Path(blob_root) / "optimizer_registry"))
        self._registry_dir.mkdir(parents=True, exist_ok=True)
        self._compressor = zstd.ZstdCompressor(level=compression_level)
        self._decompressor = zstd.ZstdDecompressor()
        self._cache_capacity = max(1, int(cache_capacity))
        self._writeback_threshold = max(1, int(writeback_threshold))
        self._cache: dict[str, dict[str, Any]] = {}
        self._pinned: set[str] = set()
        self._lock = threading.RLock()

    @staticmethod
    def _cache_key(origin_id: str, block_id: str) -> str:
        return f"{origin_id}::{block_id}"

    def _manifest_path(self, origin_id: str) -> Path:
        safe = origin_id.replace(":", "_").replace("/", "_")
        return self._registry_dir / f"{safe}.optimizer.json"

    def _load_manifest(self, origin_id: str) -> OptimizerStateManifest:
        path = self._manifest_path(origin_id)
        if not path.exists():
            return OptimizerStateManifest(schema="sodl-v1", origin_id=origin_id, blocks={}, updated_at=_now_iso())
        return _manifest_from_mapping(json.loads(path.read_text(encoding="utf-8")))

    def _save_manifest(self, manifest: OptimizerStateManifest) -> None:
        path = self._manifest_path(manifest.origin_id)
        temp = path.with_suffix(".tmp")
        payload = {
            "schema": manifest.schema,
            "origin_id": manifest.origin_id,
            "updated_at": manifest.updated_at,
            "blocks": {block_id: asdict(record) for block_id, record in manifest.blocks.items()},
        }
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp.replace(path)

    def _dirty_block_count(self, origin_id: str | None = None) -> int:
        return sum(
            1
            for key, entry in self._cache.items()
            if entry["dirty"] and (origin_id is None or key.startswith(f"{origin_id}::"))
        )

    def _enforce_capacity(self) -> None:
        if len(self._cache) <= self._cache_capacity:
            return
        candidates = sorted(
            (
                (key, entry["last_touch"])
                for key, entry in self._cache.items()
                if not entry["dirty"] and key not in self._pinned and not entry["pinned"]
            ),
            key=lambda item: item[1],
        )
        while len(self._cache) > self._cache_capacity and candidates:
            key, _ = candidates.pop(0)
            self._cache.pop(key, None)

    def store_block(
        self,
        origin_id: str,
        block_id: str,
        payload: bytes,
        *,
        step: int = 0,
        shard_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OptimizerStoreResult:
        with self._lock:
            key = self._cache_key(origin_id, block_id)
            self._cache[key] = {
                "bytes": bytes(payload),
                "step": int(step),
                "shard_key": shard_key,
                "metadata": dict(metadata or {}),
                "dirty": True,
                "pinned": key in self._pinned,
                "last_touch": time.time(),
                "record": None,
            }
            dirty_blocks = self._dirty_block_count(origin_id)
            result = OptimizerStoreResult(
                origin_id=origin_id,
                block_id=block_id,
                blob_id=None,
                step=int(step),
                staged=True,
                flushed=False,
                size_raw=len(payload),
                size_stored=0,
                dirty_blocks=dirty_blocks,
                metadata=dict(metadata or {}),
            )
            if dirty_blocks >= self._writeback_threshold:
                manifest = self.flush_origin(origin_id)
                record = manifest.blocks.get(block_id)
                if record is not None:
                    result.blob_id = record.blob_id
                    result.staged = False
                    result.flushed = True
                    result.size_stored = record.size_stored
                result.dirty_blocks = self._dirty_block_count(origin_id)
            return result

    def store_blocks(
        self,
        origin_id: str,
        blocks: list[dict[str, Any]],
    ) -> list[OptimizerStoreResult]:
        with self._lock:
            for block in blocks:
                key = self._cache_key(origin_id, str(block["block_id"]))
                self._cache[key] = {
                    "bytes": bytes(block["payload"]),
                    "step": int(block.get("step", 0)),
                    "shard_key": block.get("shard_key"),
                    "metadata": dict(block.get("metadata") or {}),
                    "dirty": True,
                    "pinned": key in self._pinned,
                    "last_touch": time.time(),
                    "record": None,
                }
            dirty_blocks = self._dirty_block_count(origin_id)
            flushed_manifest = self.flush_origin(origin_id) if dirty_blocks >= self._writeback_threshold else None
            dirty_after = self._dirty_block_count(origin_id)
            results: list[OptimizerStoreResult] = []
            for block in blocks:
                block_id = str(block["block_id"])
                record = flushed_manifest.blocks.get(block_id) if flushed_manifest is not None else None
                results.append(
                    OptimizerStoreResult(
                        origin_id=origin_id,
                        block_id=block_id,
                        blob_id=record.blob_id if record is not None else None,
                        step=int(block.get("step", 0)),
                        staged=record is None,
                        flushed=record is not None,
                        size_raw=len(block["payload"]),
                        size_stored=record.size_stored if record is not None else 0,
                        dirty_blocks=dirty_after,
                        metadata=dict(block.get("metadata") or {}),
                    )
                )
            return results

    def load_block(self, origin_id: str, block_id: str) -> bytes:
        with self._lock:
            key = self._cache_key(origin_id, block_id)
            cached = self._cache.get(key)
            if cached is not None:
                cached["last_touch"] = time.time()
                return bytes(cached["bytes"])

            manifest = self._load_manifest(origin_id)
            record = manifest.blocks.get(block_id)
            if record is None:
                raise FileNotFoundError(f"Optimizer block not found: {origin_id}/{block_id}")

            compressed = self._blob_store.get(record.blob_id)
            payload = self._decompressor.decompress(compressed)
            self._cache[key] = {
                "bytes": payload,
                "step": record.step,
                "shard_key": record.shard_key,
                "metadata": dict(record.metadata or {}),
                "dirty": False,
                "pinned": key in self._pinned,
                "last_touch": time.time(),
                "record": record,
            }
            self._enforce_capacity()
            return bytes(payload)

    def load_blocks(self, origin_id: str, block_ids: list[str]) -> dict[str, bytes]:
        payloads: dict[str, bytes] = {}
        for block_id in block_ids:
            try:
                payloads[str(block_id)] = self.load_block(origin_id, str(block_id))
            except FileNotFoundError:
                continue
        return payloads

    def prefetch_blocks(self, origin_id: str, block_ids: list[str]) -> int:
        loaded = 0
        for block_id in block_ids:
            try:
                self.load_block(origin_id, block_id)
                loaded += 1
            except FileNotFoundError:
                continue
        return loaded

    def pin_blocks(self, origin_id: str, block_ids: list[str]) -> None:
        with self._lock:
            for block_id in block_ids:
                key = self._cache_key(origin_id, block_id)
                self._pinned.add(key)
                if key in self._cache:
                    self._cache[key]["pinned"] = True

    def unpin_blocks(self, origin_id: str, block_ids: list[str]) -> None:
        with self._lock:
            for block_id in block_ids:
                key = self._cache_key(origin_id, block_id)
                self._pinned.discard(key)
                if key in self._cache:
                    self._cache[key]["pinned"] = False
            self._enforce_capacity()

    def evict_blocks(self, origin_id: str, block_ids: list[str]) -> int:
        evicted = 0
        with self._lock:
            for block_id in block_ids:
                key = self._cache_key(origin_id, block_id)
                entry = self._cache.get(key)
                if entry is None:
                    continue
                if entry["dirty"] or entry["pinned"] or key in self._pinned:
                    continue
                self._cache.pop(key, None)
                evicted += 1
        return evicted

    def flush_origin(self, origin_id: str) -> OptimizerStateManifest:
        with self._lock:
            keys = [
                key
                for key in self._cache.keys()
                if key.startswith(f"{origin_id}::")
            ]
        return self.flush_blocks(origin_id, [key.split("::", 1)[1] for key in keys])

    def flush_blocks(self, origin_id: str, block_ids: list[str]) -> OptimizerStateManifest:
        with self._lock:
            manifest = self._load_manifest(origin_id)
            for block_id in block_ids:
                key = self._cache_key(origin_id, block_id)
                entry = self._cache.get(key)
                if entry is None or not entry["dirty"]:
                    continue
                compressed = self._compressor.compress(entry["bytes"])
                blob_id = compute_blob_id(compressed)
                if not self._blob_store.has(blob_id):
                    self._blob_store.put(blob_id, compressed)
                record = OptimizerBlockRecord(
                    block_id=block_id,
                    blob_id=blob_id,
                    step=int(entry["step"]),
                    shard_key=entry["shard_key"],
                    size_raw=len(entry["bytes"]),
                    size_stored=len(compressed),
                    stored_at=_now_iso(),
                    metadata=dict(entry["metadata"]),
                )
                manifest.blocks[block_id] = record
                entry["dirty"] = False
                entry["record"] = record
                entry["last_touch"] = time.time()
            manifest.updated_at = _now_iso()
            self._save_manifest(manifest)
            self._enforce_capacity()
            return manifest

    def manifest(self, origin_id: str) -> OptimizerStateManifest:
        with self._lock:
            return self._load_manifest(origin_id)

    def latest_blob_id(self, origin_id: str, block_id: str) -> str | None:
        with self._lock:
            manifest = self._load_manifest(origin_id)
            record = manifest.blocks.get(block_id)
            return record.blob_id if record is not None else None

    def dirty_block_count(self, origin_id: str | None = None) -> int:
        with self._lock:
            return self._dirty_block_count(origin_id)

    def cache_stats(self) -> OptimizerCacheStats:
        with self._lock:
            return OptimizerCacheStats(
                cache_entries=len(self._cache),
                dirty_entries=self._dirty_block_count(None),
                pinned_entries=len(self._pinned),
                cache_capacity=self._cache_capacity,
                writeback_threshold=self._writeback_threshold,
            )

    def set_cache_capacity(self, value: int) -> None:
        with self._lock:
            self._cache_capacity = max(1, int(value))
            self._enforce_capacity()


class OptimizerStateStore:
    """Persist and retrieve optimizer-state shards via SODL."""

    def __init__(
        self,
        blob_root: str | Path,
        registry_dir: str | Path | None = None,
        *,
        compression_level: int = 3,
        cache_capacity: int = 32,
        writeback_threshold: int = 8,
    ) -> None:
        native = _rust_bridge.create_optimizer_state_store(
            str(blob_root),
            str(registry_dir) if registry_dir is not None else None,
            compression_level=compression_level,
            cache_capacity=cache_capacity,
            writeback_threshold=writeback_threshold,
        )
        self._native = native
        self._fallback = None if native is not None else _PythonOptimizerStateBackend(
            blob_root,
            registry_dir,
            compression_level=compression_level,
            cache_capacity=cache_capacity,
            writeback_threshold=writeback_threshold,
        )

    def _backend(self) -> Any:
        return self._native if self._native is not None else self._fallback

    def store_block(
        self,
        origin_id: str,
        block_id: str,
        state: bytes,
        *,
        step: int = 0,
        shard_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OptimizerStoreResult:
        backend = self._backend()
        if self._native is not None:
            payload = json.loads(
                backend.store_block(
                    origin_id,
                    block_id,
                    state,
                    step,
                    shard_key,
                    json.dumps(metadata or {}),
                )
            )
            return _result_from_mapping(payload)
        return backend.store_block(origin_id, block_id, state, step=step, shard_key=shard_key, metadata=metadata)

    def store_blocks(
        self,
        origin_id: str,
        blocks: list[dict[str, Any]],
    ) -> list[OptimizerStoreResult]:
        backend = self._backend()
        if self._native is not None:
            payload = json.loads(
                backend.store_blocks(
                    origin_id,
                    [str(block["block_id"]) for block in blocks],
                    [bytes(block["payload"]) for block in blocks],
                    [int(block.get("step", 0)) for block in blocks],
                    [block.get("shard_key") for block in blocks],
                    [json.dumps(block.get("metadata") or {}) for block in blocks],
                )
            )
            return [_result_from_mapping(item) for item in payload]
        return backend.store_blocks(origin_id, blocks)

    def load_block(self, origin_id: str, block_id: str) -> bytes:
        return bytes(self._backend().load_block(origin_id, block_id))

    def load_blocks(self, origin_id: str, block_ids: list[str]) -> dict[str, bytes]:
        backend = self._backend()
        if self._native is not None:
            pairs = backend.load_blocks(origin_id, block_ids)
            return {str(block_id): bytes(payload) for block_id, payload in pairs}
        return backend.load_blocks(origin_id, block_ids)

    def prefetch_blocks(self, origin_id: str, block_ids: list[str]) -> int:
        return int(self._backend().prefetch_blocks(origin_id, block_ids))

    def pin_blocks(self, origin_id: str, block_ids: list[str]) -> None:
        self._backend().pin_blocks(origin_id, block_ids)

    def unpin_blocks(self, origin_id: str, block_ids: list[str]) -> None:
        self._backend().unpin_blocks(origin_id, block_ids)

    def flush_origin(self, origin_id: str) -> OptimizerStateManifest:
        backend = self._backend()
        if self._native is not None:
            return _manifest_from_mapping(json.loads(backend.flush_origin(origin_id)))
        return backend.flush_origin(origin_id)

    def flush_blocks(self, origin_id: str, block_ids: list[str]) -> OptimizerStateManifest:
        backend = self._backend()
        if self._native is not None:
            return _manifest_from_mapping(json.loads(backend.flush_blocks(origin_id, block_ids)))
        return backend.flush_blocks(origin_id, block_ids)

    def manifest(self, origin_id: str) -> OptimizerStateManifest:
        backend = self._backend()
        if self._native is not None:
            return _manifest_from_mapping(json.loads(backend.manifest_json(origin_id)))
        return backend.manifest(origin_id)

    def evict_blocks(self, origin_id: str, block_ids: list[str]) -> int:
        return int(self._backend().evict_blocks(origin_id, block_ids))

    def latest_blob_id(self, origin_id: str, block_id: str) -> str | None:
        return self._backend().latest_blob_id(origin_id, block_id)

    def dirty_block_count(self, origin_id: str | None = None) -> int:
        return int(self._backend().dirty_block_count(origin_id))

    def cache_stats(self) -> OptimizerCacheStats:
        backend = self._backend()
        if self._native is not None:
            return _stats_from_mapping(json.loads(backend.cache_stats_json()))
        return backend.cache_stats()

    def set_cache_capacity(self, value: int) -> None:
        self._backend().set_cache_capacity(int(value))
