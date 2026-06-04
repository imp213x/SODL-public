"""Generic Artifact Store — content-addressed storage for any AI artifact.

Extends SODL beyond weight clusters to store arbitrary data: numpy arrays,
PyTorch tensors, JSON objects, raw bytes, or any pickled Python object.

Each artifact is stored with metadata and tracked by origin for lineage.

Pipeline:
    put: data → serialize → compress (zstd) → blake3 hash → store
    get: blob_id → fetch → verify → decompress → deserialize
"""

from __future__ import annotations

import io
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import zstandard as zstd

from sodl_weights.store import BlobStore, compute_blob_id


# ── Types ──────────────────────────────────────────────────────────────

@dataclass
class ArtifactMetadata:
    """Metadata attached to a stored artifact."""
    artifact_id: str
    blob_id: str
    name: str
    artifact_type: str  # "numpy", "tensor", "json", "bytes", "pickle"
    origin_id: str
    shape: Optional[list[int]] = None
    dtype: Optional[str] = None
    size_raw: int = 0
    size_stored: int = 0
    created_at: str = ""
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class ArtifactManifest:
    """Manifest tracking all artifacts for an origin."""
    origin_id: str
    artifacts: dict[str, ArtifactMetadata] = field(default_factory=dict)
    created_at: str = ""

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "origin_id": self.origin_id,
            "created_at": self.created_at,
            "artifacts": {k: asdict(v) for k, v in self.artifacts.items()},
        }
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ArtifactManifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        artifacts = {
            k: ArtifactMetadata(**v) for k, v in data.get("artifacts", {}).items()
        }
        return cls(
            origin_id=data["origin_id"],
            artifacts=artifacts,
            created_at=data.get("created_at", ""),
        )


# ── Serializers ────────────────────────────────────────────────────────

def _serialize_numpy(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _deserialize_numpy(data: bytes) -> np.ndarray:
    buf = io.BytesIO(data)
    return np.load(buf, allow_pickle=False)


def _serialize_json(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def _deserialize_json(data: bytes) -> Any:
    return json.loads(data.decode("utf-8"))


# ── Artifact Store ─────────────────────────────────────────────────────

class ArtifactStore:
    """Content-addressed storage for any AI artifact.

    Supports numpy arrays, PyTorch tensors, JSON objects, and raw bytes.
    All artifacts are compressed with zstd and tracked with metadata.

    Parameters
    ----------
    blob_store : BlobStore
        Underlying content-addressed blob store.
    manifest_dir : str | Path
        Directory for storing artifact manifests.
    zstd_level : int
        Zstandard compression level (default 3).

    Example
    -------
    >>> store = ArtifactStore(BlobStore("./blobs"), "./manifests")
    >>> meta = store.store_numpy("my-model", np.random.randn(100, 768), "embeddings")
    >>> arr = store.load_numpy(meta.blob_id)
    """

    def __init__(
        self,
        blob_store: BlobStore,
        manifest_dir: str | Path = "manifests",
        zstd_level: int = 3,
    ) -> None:
        self._blob_store = blob_store
        self._manifest_dir = Path(manifest_dir)
        self._manifest_dir.mkdir(parents=True, exist_ok=True)
        self._zstd_level = zstd_level
        self._compressor = zstd.ZstdCompressor(level=zstd_level)
        self._decompressor = zstd.ZstdDecompressor()
        self._manifests: dict[str, ArtifactManifest] = {}

    @staticmethod
    def _utcnow() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

    def _get_manifest(self, origin_id: str) -> ArtifactManifest:
        if origin_id not in self._manifests:
            manifest_path = self._manifest_path(origin_id)
            if manifest_path.exists():
                self._manifests[origin_id] = ArtifactManifest.load(manifest_path)
            else:
                self._manifests[origin_id] = ArtifactManifest(
                    origin_id=origin_id,
                    created_at=self._utcnow(),
                )
        return self._manifests[origin_id]

    def _save_manifest(self, origin_id: str) -> None:
        manifest = self._get_manifest(origin_id)
        path = self._manifest_path(origin_id)
        manifest.save(path)

    @staticmethod
    def _safe_origin_filename(origin_id: str) -> str:
        return origin_id.replace(":", "__")

    def _manifest_path(self, origin_id: str) -> Path:
        return self._manifest_dir / f"{self._safe_origin_filename(origin_id)}.json"

    def _all_manifests(self) -> list[ArtifactManifest]:
        manifests: list[ArtifactManifest] = []
        for manifest_path in sorted(self._manifest_dir.glob("*.json")):
            manifests.append(ArtifactManifest.load(manifest_path))
        return manifests

    def _referenced_blob_ids(self, *, exclude: tuple[str, str] | None = None) -> set[str]:
        blob_ids: set[str] = set()
        for manifest in self._all_manifests():
            for artifact_id, metadata in manifest.artifacts.items():
                if exclude is not None and exclude == (manifest.origin_id, artifact_id):
                    continue
                blob_ids.add(metadata.blob_id)
        return blob_ids

    def _store_raw(self, origin_id: str, data: bytes, name: str,
                   artifact_type: str, shape: list[int] | None = None,
                   dtype: str | None = None, tags: dict[str, str] | None = None) -> ArtifactMetadata:
        """Core storage: compress, hash, store, and register metadata."""
        raw_size = len(data)
        compressed = self._compressor.compress(data)
        blob_id = compute_blob_id(compressed)
        self._blob_store.put(blob_id, compressed)

        meta = ArtifactMetadata(
            artifact_id=f"art:{uuid.uuid4()}",
            blob_id=blob_id,
            name=name,
            artifact_type=artifact_type,
            origin_id=origin_id,
            shape=shape,
            dtype=dtype,
            size_raw=raw_size,
            size_stored=len(compressed),
            created_at=self._utcnow(),
            tags=dict(tags or {}),
        )

        manifest = self._get_manifest(origin_id)
        manifest.artifacts[meta.artifact_id] = meta
        self._save_manifest(origin_id)

        return meta

    def _load_raw(self, blob_id: str) -> bytes:
        """Load and decompress raw bytes from the blob store."""
        compressed = self._blob_store.get(blob_id)
        return self._decompressor.decompress(compressed)

    # ── Public API ─────────────────────────────────────────────────────

    def store(self, origin_id: str, data: bytes, name: str,
              tags: dict[str, str] | None = None) -> ArtifactMetadata:
        """Store raw bytes as a named artifact.

        Parameters
        ----------
        origin_id : str
            Origin to associate this artifact with.
        data : bytes
            Raw data to store.
        name : str
            Human-readable name for the artifact.
        tags : dict, optional
            Key-value tags for filtering/search.
        """
        return self._store_raw(origin_id, data, name, "bytes", tags=tags)

    def store_numpy(self, origin_id: str, array: np.ndarray, name: str,
                    tags: dict[str, str] | None = None) -> ArtifactMetadata:
        """Store a numpy array as a named artifact.

        Parameters
        ----------
        origin_id : str
            Origin to associate this artifact with.
        array : np.ndarray
            Array to store.
        name : str
            Human-readable name.
        """
        data = _serialize_numpy(array)
        return self._store_raw(
            origin_id, data, name, "numpy",
            shape=list(array.shape), dtype=str(array.dtype), tags=tags,
        )

    def store_tensor(self, origin_id: str, tensor: "torch.Tensor", name: str,
                     tags: dict[str, str] | None = None) -> ArtifactMetadata:
        """Store a PyTorch tensor as a named artifact.

        Parameters
        ----------
        origin_id : str
            Origin to associate this artifact with.
        tensor : torch.Tensor
            Tensor to store (converted to numpy internally).
        name : str
            Human-readable name.
        """
        import torch
        arr = tensor.detach().cpu().numpy()
        data = _serialize_numpy(arr)
        return self._store_raw(
            origin_id, data, name, "tensor",
            shape=list(arr.shape), dtype=str(arr.dtype), tags=tags,
        )

    def store_json(self, origin_id: str, obj: Any, name: str,
                   tags: dict[str, str] | None = None) -> ArtifactMetadata:
        """Store a JSON-serializable object as a named artifact.

        Parameters
        ----------
        origin_id : str
            Origin to associate this artifact with.
        obj : Any
            JSON-serializable Python object.
        name : str
            Human-readable name.
        """
        data = _serialize_json(obj)
        return self._store_raw(origin_id, data, name, "json", tags=tags)

    def load(self, blob_id: str) -> bytes:
        """Load raw bytes by blob ID."""
        return self._load_raw(blob_id)

    def load_numpy(self, blob_id: str) -> np.ndarray:
        """Load a numpy array by blob ID."""
        data = self._load_raw(blob_id)
        return _deserialize_numpy(data)

    def load_tensor(self, blob_id: str) -> "torch.Tensor":
        """Load a PyTorch tensor by blob ID."""
        import torch
        arr = self.load_numpy(blob_id)
        return torch.from_numpy(arr)

    def load_json(self, blob_id: str) -> Any:
        """Load a JSON object by blob ID."""
        data = self._load_raw(blob_id)
        return _deserialize_json(data)

    def list_artifacts(self, origin_id: str,
                       artifact_type: str | None = None) -> list[ArtifactMetadata]:
        """List all artifacts for an origin, optionally filtered by type."""
        manifest = self._get_manifest(origin_id)
        artifacts = list(manifest.artifacts.values())
        if artifact_type:
            artifacts = [a for a in artifacts if a.artifact_type == artifact_type]
        return sorted(artifacts, key=lambda a: a.created_at)

    def get_metadata(self, origin_id: str, artifact_id: str) -> ArtifactMetadata | None:
        """Get metadata for a specific artifact."""
        manifest = self._get_manifest(origin_id)
        return manifest.artifacts.get(artifact_id)

    def find_artifacts(
        self,
        origin_id: str | None = None,
        *,
        artifact_type: str | None = None,
        tags: dict[str, str] | None = None,
        name_contains: str | None = None,
    ) -> list[ArtifactMetadata]:
        """Search artifacts across one origin or all manifests."""
        manifests = [self._get_manifest(origin_id)] if origin_id is not None else self._all_manifests()
        lowered_name = (name_contains or "").strip().lower()
        matched: list[ArtifactMetadata] = []
        for manifest in manifests:
            for metadata in manifest.artifacts.values():
                if artifact_type and metadata.artifact_type != artifact_type:
                    continue
                if lowered_name and lowered_name not in metadata.name.lower():
                    continue
                if tags and any(metadata.tags.get(key) != value for key, value in tags.items()):
                    continue
                matched.append(metadata)
        return sorted(matched, key=lambda artifact: artifact.created_at)

    def artifact_stats(self, origin_id: str | None = None) -> dict[str, Any]:
        """Summarize stored artifacts and byte usage."""
        artifacts = self.find_artifacts(origin_id)
        by_type: dict[str, dict[str, int]] = {}
        for metadata in artifacts:
            bucket = by_type.setdefault(
                metadata.artifact_type,
                {"count": 0, "size_raw": 0, "size_stored": 0},
            )
            bucket["count"] += 1
            bucket["size_raw"] += metadata.size_raw
            bucket["size_stored"] += metadata.size_stored
        return {
            "count": len(artifacts),
            "size_raw": sum(metadata.size_raw for metadata in artifacts),
            "size_stored": sum(metadata.size_stored for metadata in artifacts),
            "by_type": by_type,
        }

    def delete_artifact(
        self,
        origin_id: str,
        artifact_id: str,
        *,
        delete_blob_if_unreferenced: bool = False,
    ) -> bool:
        """Remove an artifact from the manifest and optionally delete an orphaned blob."""
        manifest = self._get_manifest(origin_id)
        if artifact_id in manifest.artifacts:
            metadata = manifest.artifacts[artifact_id]
            del manifest.artifacts[artifact_id]
            self._save_manifest(origin_id)
            if delete_blob_if_unreferenced:
                referenced = self._referenced_blob_ids(exclude=(origin_id, artifact_id))
                if metadata.blob_id not in referenced:
                    self._blob_store.delete(metadata.blob_id)
            return True
        return False

    def enforce_retention(
        self,
        origin_id: str,
        *,
        keep_last: int = 0,
        max_age_seconds: float | None = None,
        protected_tags: dict[str, str] | None = None,
        delete_unreferenced_blobs: bool = False,
    ) -> list[ArtifactMetadata]:
        """Delete artifacts outside the retention policy and return the removed metadata."""
        artifacts = self.list_artifacts(origin_id)
        if not artifacts:
            return []

        keep_ids = {artifact.artifact_id for artifact in artifacts[-keep_last:]} if keep_last > 0 else set()
        now = datetime.now(timezone.utc)
        removed: list[ArtifactMetadata] = []
        for artifact in artifacts:
            if artifact.artifact_id in keep_ids:
                continue
            if protected_tags and all(artifact.tags.get(key) == value for key, value in protected_tags.items()):
                continue
            if max_age_seconds is not None:
                age = (now - self._parse_timestamp(artifact.created_at)).total_seconds()
                if age < max_age_seconds:
                    continue
            elif keep_last <= 0:
                continue
            if self.delete_artifact(
                origin_id,
                artifact.artifact_id,
                delete_blob_if_unreferenced=delete_unreferenced_blobs,
            ):
                removed.append(artifact)
        return removed

    @property
    def manifest_dir(self) -> Path:
        return self._manifest_dir
