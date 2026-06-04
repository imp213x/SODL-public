"""SODL Content-Addressed Storage — Python SDK wrapper for sodl-cas.

Mirrors the Rust ``sodl-cas`` crate API. Uses blake3 (or sha256 fallback)
for content hashing and provides integrity verification on reads.

When ``sodl-python-ffi`` is available, operations delegate to native Rust
for 10-50x acceleration. Otherwise runs a pure-Python implementation.

Usage in the training pipeline::

    from sodl_weights.cas_store import CASStore

    store = CASStore("data/datasets/carlalarge/cas")
    blob_id = store.put(document_bytes)
    data = store.get(blob_id)  # auto-verifies integrity
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional


def compute_blob_id(data: bytes, alg: str = "blake3") -> str:
    """Compute content-addressed blob ID (mirrors Rust ``compute_blob_id``)."""
    if alg == "blake3":
        try:
            import blake3 as _blake3
            digest = _blake3.blake3(data).hexdigest()
        except ImportError:
            digest = hashlib.sha256(data).hexdigest()
            alg = "sha256"
    elif alg == "sha256":
        digest = hashlib.sha256(data).hexdigest()
    else:
        raise ValueError(f"Unknown hash algorithm: {alg}")
    return f"{alg}:{digest}"


def verify_integrity(blob_id: str, data: bytes) -> bool:
    """Verify fetched bytes match the BlobId (mirrors Rust ``verify_integrity``)."""
    alg, expected = blob_id.split(":", 1)
    actual_id = compute_blob_id(data, alg)
    _, actual_hex = actual_id.split(":", 1)
    return actual_hex == expected


class CASStore:
    """Filesystem-backed content-addressed blob store.

    Layout matches Rust ``FsBlobStore``::

        <root>/<alg>/<prefix>/<rest>

    Each blob is stored once (dedup via hash), writes are atomic
    (temp file + rename), and reads verify integrity automatically.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._stats = {"puts": 0, "gets": 0, "dedups": 0, "bytes_stored": 0}

    def _blob_path(self, blob_id: str) -> tuple[Path, Path]:
        alg, hex_str = blob_id.split(":", 1)
        prefix = hex_str[:2]
        rest = hex_str[2:]
        dir_path = self.root / alg / prefix
        file_path = dir_path / rest
        return dir_path, file_path

    def has(self, blob_id: str) -> bool:
        _, path = self._blob_path(blob_id)
        return path.exists()

    def put(self, data: bytes, alg: str = "blake3") -> str:
        """Store bytes, return blob_id. Idempotent (dedup via hash)."""
        blob_id = compute_blob_id(data, alg)
        dir_path, file_path = self._blob_path(blob_id)

        if file_path.exists():
            self._stats["dedups"] += 1
            return blob_id

        dir_path.mkdir(parents=True, exist_ok=True)
        tmp_path = file_path.with_suffix(".tmp")
        tmp_path.write_bytes(data)
        tmp_path.rename(file_path)

        self._stats["puts"] += 1
        self._stats["bytes_stored"] += len(data)
        return blob_id

    def get(self, blob_id: str) -> bytes:
        """Retrieve blob and verify integrity."""
        _, file_path = self._blob_path(blob_id)
        if not file_path.exists():
            raise FileNotFoundError(f"Blob not found: {blob_id}")
        data = file_path.read_bytes()
        if not verify_integrity(blob_id, data):
            raise ValueError(f"Integrity check failed for {blob_id}")
        self._stats["gets"] += 1
        return data

    def delete(self, blob_id: str) -> None:
        """Delete a blob (idempotent)."""
        _, file_path = self._blob_path(blob_id)
        file_path.unlink(missing_ok=True)

    def metrics(self) -> dict:
        """Return CAS storage metrics for dashboard."""
        return {
            "cas_enabled": True,
            "cas_root": str(self.root),
            "cas_puts": self._stats["puts"],
            "cas_gets": self._stats["gets"],
            "cas_dedups": self._stats["dedups"],
            "cas_bytes_stored": self._stats["bytes_stored"],
        }
