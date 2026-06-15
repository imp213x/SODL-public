"""Weight Blob Store — content-addressed, compressed, encrypted storage.

Mirrors the Rust ``WeightBlobStore`` in ``sodl-store/src/weight_store.rs``.

Pipeline:
    put:  cluster → serialise → compress (zstd) → encrypt → blake3 hash → store
    get:  blob_id → fetch → verify hash → decrypt → decompress → deserialise
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

import blake3
import numpy as np
import zstandard as zstd

from . import _rust_bridge
from sodl_weights.crypto import CryptoProvider, NullCrypto
from sodl_weights.types import StoreStats, WeightCluster

_DEFAULT_ZSTD_LEVEL = 3
_JSON_CODEC = "json"
_COMPACT_Q8_CODEC = "compact_q8"
_COMPACT_MAGIC = b"SODLCQ8\x00"


class SodlIntegrityError(Exception):
    """Raised when blob integrity verification fails."""


class SodlNotFoundError(Exception):
    """Raised when a blob is not found in the store."""


def canonical_blob_id(value: str | Path) -> str:
    """Normalize a blob id or ``.blob`` filename to ``blake3:<hex>``.

    SODL indexes store logical blob ids while filesystem stores use
    ``<hex>.blob`` files. Any pruning or consistency scan must compare the
    logical id, not the literal filename, to avoid deleting live blobs.
    """
    raw = str(value).strip()
    name = raw.replace("\\", "/").rsplit("/", 1)[-1]
    if name.endswith(".blob"):
        name = name[: -len(".blob")]
    if ":" in name:
        return name
    return f"blake3:{name}"


class BlobStore:
    """On-disk content-addressed blob store.

    Blobs are stored as files named by their blake3 hash under a root directory.
    """

    def __init__(
        self,
        root: str | Path,
        source_roots: Sequence[str | Path] | None = None,
        peer_urls: Sequence[str] | None = None,
        edge_urls: Sequence[str] | None = None,
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._source_roots = [Path(source_root) for source_root in source_roots or []]
        self._peer_urls = [str(url) for url in peer_urls or []]
        self._edge_urls = [str(url) for url in edge_urls or []]
        self._observed_replica_nodes: dict[str, set[str]] = {}
        self._ffi_store = _rust_bridge.create_blob_store(
            str(self._root),
            [str(source_root) for source_root in self._source_roots],
            self._peer_urls,
            self._edge_urls,
        )

    def _path(self, blob_id: str) -> Path:
        # blob_id is "blake3:<hex>" — use the hex part as the filename
        _, hex_hash = canonical_blob_id(blob_id).split(":", 1)
        return self._root / f"{hex_hash}.blob"

    @staticmethod
    def _path_for_root(root: Path, blob_id: str) -> Path:
        _, hex_hash = canonical_blob_id(blob_id).split(":", 1)
        return root / f"{hex_hash}.blob"

    def _source_path(self, source_root: Path, blob_id: str) -> Path:
        return self._path_for_root(source_root, blob_id)

    def _record_replica(self, blob_id: str, node_id: str) -> None:
        self._observed_replica_nodes.setdefault(blob_id, set()).add(node_id)

    def _remote_blob_url(self, base_url: str, blob_id: str) -> str:
        return f"{base_url.rstrip('/')}/v1/blobs/{quote(blob_id, safe=':')}"

    def _fetch_remote_blob(self, base_urls: Sequence[str], blob_id: str) -> tuple[bytes | None, str | None]:
        for base_url in base_urls:
            try:
                with urlopen(self._remote_blob_url(base_url, blob_id), timeout=10) as response:
                    if response.status == 200:
                        data = response.read()
                        return data, base_url
            except HTTPError as exc:
                if exc.code == 404:
                    continue
                raise SodlNotFoundError(f"Remote fetch failed for {blob_id}: {exc}") from exc
            except URLError as exc:
                raise SodlNotFoundError(f"Remote fetch failed for {blob_id}: {exc}") from exc
        return None, None

    def has(self, blob_id: str) -> bool:
        if self._ffi_store is not None:
            try:
                if bool(self._ffi_store.has(blob_id)):
                    return True
            except Exception:
                pass
            if self._path(blob_id).exists():
                self._record_replica(blob_id, "legacy-flat-cache")
                return True
            for index, source_root in enumerate(self._source_roots):
                if self._source_path(source_root, blob_id).exists():
                    self._record_replica(blob_id, f"legacy-flat-source:{index}:{source_root}")
                    return True
            data, base_url = self._fetch_remote_blob(self._peer_urls, blob_id)
            if data is not None and base_url is not None:
                self._record_replica(blob_id, base_url)
                return True
            data, base_url = self._fetch_remote_blob(self._edge_urls, blob_id)
            if data is not None and base_url is not None:
                self._record_replica(blob_id, base_url)
                return True
            return False
        if self._path(blob_id).exists():
            self._record_replica(blob_id, "cache")
            return True
        for index, source_root in enumerate(self._source_roots):
            if self._source_path(source_root, blob_id).exists():
                self._record_replica(blob_id, f"source:{index}:{source_root}")
                return True
        data, base_url = self._fetch_remote_blob(self._peer_urls, blob_id)
        if data is not None and base_url is not None:
            self._record_replica(blob_id, base_url)
            return True
        data, base_url = self._fetch_remote_blob(self._edge_urls, blob_id)
        if data is not None and base_url is not None:
            self._record_replica(blob_id, base_url)
            return True
        return False

    def put(self, blob_id: str, data: bytes) -> None:
        if self._ffi_store is not None:
            self._ffi_store.put(blob_id, data)
            return
        self._path(blob_id).write_bytes(data)

    def get(self, blob_id: str) -> bytes:
        if self._ffi_store is not None:
            try:
                return bytes(self._ffi_store.get(blob_id))
            except Exception as exc:
                p = self._path(blob_id)
                if p.exists():
                    data = p.read_bytes()
                    self._record_replica(blob_id, "legacy-flat-cache")
                    try:
                        self._ffi_store.put(blob_id, data)
                        self._record_replica(blob_id, "cache")
                    except Exception:
                        pass
                    return data
                for index, source_root in enumerate(self._source_roots):
                    source_path = self._source_path(source_root, blob_id)
                    if source_path.exists():
                        data = source_path.read_bytes()
                        self._record_replica(blob_id, f"legacy-flat-source:{index}:{source_root}")
                        try:
                            self._ffi_store.put(blob_id, data)
                            self._record_replica(blob_id, "cache")
                        except Exception:
                            pass
                        return data
                data, base_url = self._fetch_remote_blob(self._peer_urls, blob_id)
                if data is not None:
                    self._record_replica(blob_id, base_url or "peer")
                    try:
                        self._ffi_store.put(blob_id, data)
                        self._record_replica(blob_id, "cache")
                    except Exception:
                        pass
                    return data
                data, base_url = self._fetch_remote_blob(self._edge_urls, blob_id)
                if data is not None:
                    self._record_replica(blob_id, base_url or "edge")
                    try:
                        self._ffi_store.put(blob_id, data)
                        self._record_replica(blob_id, "cache")
                    except Exception:
                        pass
                    return data
                raise SodlNotFoundError(f"Blob not found: {blob_id}") from exc
        p = self._path(blob_id)
        if not p.exists():
            data, base_url = self._fetch_remote_blob(self._peer_urls, blob_id)
            if data is not None:
                p.write_bytes(data)
                self._record_replica(blob_id, "cache")
                if base_url is not None:
                    self._record_replica(blob_id, base_url)
                return data
            data, base_url = self._fetch_remote_blob(self._edge_urls, blob_id)
            if data is not None:
                p.write_bytes(data)
                self._record_replica(blob_id, "cache")
                if base_url is not None:
                    self._record_replica(blob_id, base_url)
                return data
            for source_root in self._source_roots:
                source_path = self._source_path(source_root, blob_id)
                if source_path.exists():
                    data = source_path.read_bytes()
                    p.write_bytes(data)
                    self._record_replica(blob_id, "cache")
                    self._record_replica(blob_id, f"source:{self._source_roots.index(source_root)}:{source_root}")
                    return data
            raise SodlNotFoundError(f"Blob not found: {blob_id}")
        self._record_replica(blob_id, "cache")
        return p.read_bytes()

    def delete(self, blob_id: str) -> None:
        if self._ffi_store is not None:
            self._ffi_store.delete(blob_id)
            return
        p = self._path(blob_id)
        if p.exists():
            p.unlink()

    def blob_count(self) -> int:
        if self._ffi_store is not None:
            return int(self._ffi_store.blob_count())
        return sum(1 for _ in self._root.glob("*.blob"))

    def replica_nodes(self, blob_id: str) -> list[str]:
        if self._ffi_store is not None and hasattr(self._ffi_store, "replica_nodes"):
            return list(self._ffi_store.replica_nodes(blob_id))
        nodes: list[str] = []
        if self._path(blob_id).exists():
            nodes.append("cache")
        for index, source_root in enumerate(self._source_roots):
            source_path = self._source_path(source_root, blob_id)
            if source_path.exists():
                nodes.append(f"source:{index}:{source_root}")
        nodes.extend(sorted(self._observed_replica_nodes.get(blob_id, set()) - set(nodes)))
        return nodes


def compute_blob_id(data: bytes) -> str:
    """Compute a content-addressed blob ID using blake3."""
    rust_blob_id = _rust_bridge.compute_blob_id(data)
    if rust_blob_id is not None:
        return rust_blob_id
    return f"blake3:{blake3.blake3(data).hexdigest()}"


def verify_integrity(blob_id: str, data: bytes) -> None:
    """Verify blob data matches its content-addressed ID."""
    try:
        if _rust_bridge.verify_integrity(blob_id, data):
            return
    except Exception as exc:
        raise SodlIntegrityError(f"Integrity check failed for {blob_id}") from exc
    expected = compute_blob_id(data)
    if expected != blob_id:
        raise SodlIntegrityError(
            f"Integrity check failed: expected {blob_id}, got {expected}"
        )


class WeightBlobStore:
    """Stores weight clusters as compressed, optionally encrypted, content-addressed blobs.

    Parameters
    ----------
    store : BlobStore
        The underlying blob storage backend.
    crypto : CryptoProvider, optional
        Crypto provider for encryption. Defaults to NullCrypto (no encryption).
    compression_level : int
        zstd compression level (1=fast, 22=max, default 3).
    """

    def __init__(
        self,
        store: BlobStore,
        crypto: Optional[CryptoProvider] = None,
        compression_level: int = _DEFAULT_ZSTD_LEVEL,
        serialization_codec: str = _JSON_CODEC,
    ) -> None:
        self._store = store
        self._crypto = crypto or NullCrypto()
        self._compressor = zstd.ZstdCompressor(level=compression_level)
        self._decompressor = zstd.ZstdDecompressor()
        self._compression_level = compression_level
        self._serialization_codec = serialization_codec.strip().lower() or _JSON_CODEC

    @staticmethod
    def _serialize_cluster_json(cluster: WeightCluster) -> bytes:
        return json.dumps(cluster.to_dict(), separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _scale_to_uint8(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        max_abs = np.max(np.abs(values), axis=1)
        scales = np.maximum(max_abs / 127.0, 1e-8).astype(np.float32, copy=False)
        quantized = np.round(values / scales[:, None]).clip(-127, 127).astype(np.int8, copy=False)
        return quantized, scales

    @staticmethod
    def _scale_to_uint8_cluster(values: np.ndarray) -> tuple[np.ndarray, float]:
        max_abs = float(np.max(np.abs(values))) if values.size else 0.0
        scale = max(max_abs / 127.0, 1e-8)
        quantized = np.round(values / scale).clip(-127, 127).astype(np.int8, copy=False)
        return quantized, float(scale)

    @classmethod
    def _serialize_cluster_compact_q8(cls, cluster: WeightCluster) -> bytes:
        member_count = len(cluster.member_token_ids)
        centroid = np.asarray(cluster.centroid, dtype=np.float16)
        implicit_ids = cluster.member_token_ids == list(range(member_count))
        member_ids = np.asarray(cluster.member_token_ids, dtype=np.uint32) if not implicit_ids else np.zeros((0,), dtype=np.uint32)
        if member_count:
            offsets = np.asarray(cluster.offsets, dtype=np.float32)
            if offsets.shape != (member_count, cluster.dim):
                raise ValueError(
                    f"Cluster offsets shape mismatch: expected {(member_count, cluster.dim)}, got {offsets.shape}"
                )
            if cluster.dim == 1:
                quantized_offsets, cluster_scale = cls._scale_to_uint8_cluster(offsets)
                scales: np.ndarray | float = cluster_scale
                scale_mode = "cluster"
            else:
                quantized_offsets, scales = cls._scale_to_uint8(offsets)
                scale_mode = "row"
        else:
            quantized_offsets = np.zeros((0, cluster.dim), dtype=np.int8)
            scales = np.zeros((0,), dtype=np.float32)
            scale_mode = "row"

        header = {
            "format": _COMPACT_Q8_CODEC,
            "dim": int(cluster.dim),
            "member_count": member_count,
            "ids_mode": "implicit_range" if implicit_ids else "explicit",
            "scale_mode": scale_mode,
        }
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        if scale_mode == "cluster":
            scale_bytes = np.asarray([float(scales)], dtype=np.float32).tobytes()
        else:
            scale_bytes = np.asarray(scales, dtype=np.float32).tobytes()
        parts = [
            _COMPACT_MAGIC,
            struct.pack("<I", len(header_bytes)),
            header_bytes,
            member_ids.tobytes(),
            centroid.tobytes(),
            scale_bytes,
            quantized_offsets.tobytes(),
        ]
        return b"".join(parts)

    def _serialize_cluster(self, cluster: WeightCluster) -> bytes:
        if self._serialization_codec == _COMPACT_Q8_CODEC:
            return self._serialize_cluster_compact_q8(cluster)
        return self._serialize_cluster_json(cluster)

    @staticmethod
    def _deserialize_cluster_json(raw: bytes) -> WeightCluster:
        return WeightCluster.from_dict(json.loads(raw))

    @classmethod
    def _deserialize_cluster_compact_q8(cls, raw: bytes) -> WeightCluster:
        header_len = struct.unpack("<I", raw[len(_COMPACT_MAGIC):len(_COMPACT_MAGIC) + 4])[0]
        header_start = len(_COMPACT_MAGIC) + 4
        header_end = header_start + header_len
        header = json.loads(raw[header_start:header_end].decode("utf-8"))
        dim = int(header["dim"])
        member_count = int(header["member_count"])
        ids_mode = str(header.get("ids_mode") or "explicit")
        scale_mode = str(header.get("scale_mode") or "row")
        cursor = header_end

        if ids_mode == "implicit_range":
            member_ids = list(range(member_count))
        else:
            ids_bytes = member_count * np.dtype(np.uint32).itemsize
            member_ids = np.frombuffer(raw[cursor:cursor + ids_bytes], dtype=np.uint32, count=member_count).astype(int).tolist()
            cursor += ids_bytes

        centroid_bytes = dim * np.dtype(np.float16).itemsize
        centroid = np.frombuffer(raw[cursor:cursor + centroid_bytes], dtype=np.float16, count=dim).astype(np.float32)
        cursor += centroid_bytes

        if scale_mode == "cluster":
            scale_bytes = np.dtype(np.float32).itemsize
            scale = float(np.frombuffer(raw[cursor:cursor + scale_bytes], dtype=np.float32, count=1)[0])
            cursor += scale_bytes
        else:
            scale_bytes = member_count * np.dtype(np.float32).itemsize
            scales = np.frombuffer(raw[cursor:cursor + scale_bytes], dtype=np.float32, count=member_count)
            cursor += scale_bytes

        offset_count = member_count * dim
        offsets_q = np.frombuffer(raw[cursor:cursor + offset_count], dtype=np.int8, count=offset_count)
        if member_count:
            offsets = offsets_q.astype(np.float32).reshape(member_count, dim)
            if scale_mode == "cluster":
                offsets = offsets * scale
            else:
                offsets = offsets * scales[:, None]
            offsets_list = offsets.tolist()
        else:
            offsets_list = []

        return WeightCluster(
            centroid=centroid.tolist(),
            member_token_ids=member_ids,
            offsets=offsets_list,
            dim=dim,
            cluster_id=None,
        )

    @classmethod
    def _deserialize_cluster(cls, raw: bytes) -> WeightCluster:
        if raw.startswith(_COMPACT_MAGIC):
            return cls._deserialize_cluster_compact_q8(raw)
        return cls._deserialize_cluster_json(raw)

    def put(self, origin_id: str, cluster: WeightCluster) -> StoreStats:
        """Store a weight cluster, returning stats including its blob ID."""
        # 1. Serialise
        raw = self._serialize_cluster(cluster)
        raw_bytes = len(raw)

        # 2. Compress
        compressed = _rust_bridge.compress_zstd(raw, self._compression_level)
        if compressed is None:
            compressed = self._compressor.compress(raw)
        compressed_bytes = len(compressed)

        # 3. Encrypt
        encrypted = self._crypto.encrypt(origin_id, compressed)
        stored_bytes = len(encrypted)

        # 4. CAS hash and store
        blob_id = compute_blob_id(encrypted)
        was_deduped = self._store.has(blob_id)
        if not was_deduped:
            self._store.put(blob_id, encrypted)

        return StoreStats(
            blob_id=blob_id,
            raw_bytes=raw_bytes,
            compressed_bytes=compressed_bytes,
            stored_bytes=stored_bytes,
            was_deduped=was_deduped,
        )

    def get(self, origin_id: str, blob_id: str) -> WeightCluster:
        """Fetch and reconstruct a weight cluster by its blob ID."""
        # 1. Fetch
        stored = self._store.get(blob_id)

        # 2. Verify integrity
        verify_integrity(blob_id, stored)

        # 3. Decrypt
        compressed = self._crypto.decrypt(origin_id, stored)

        # 4. Decompress
        raw = _rust_bridge.decompress_zstd(compressed)
        if raw is None:
            raw = self._decompressor.decompress(compressed)

        # 5. Deserialise
        return self._deserialize_cluster(raw)

    def has(self, blob_id: str) -> bool:
        return self._store.has(blob_id)

    def delete(self, blob_id: str) -> None:
        self._store.delete(blob_id)
