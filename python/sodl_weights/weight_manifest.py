"""Weight-manifest lifecycle helpers backed by SODL and optional native acceleration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import _rust_bridge
from sodl_weights.store import BlobStore


@dataclass(slots=True)
class WeightManifestCluster:
    cluster_id: int
    blob_id: str
    member_token_ids: list[int]


@dataclass(slots=True)
class WeightManifest:
    origin_id: str
    vocab_size: int
    embedding_dim: int
    clusters: list[WeightManifestCluster]
    metadata: dict[str, Any]


def _cluster_from_mapping(payload: Mapping[str, Any]) -> WeightManifestCluster:
    return WeightManifestCluster(
        cluster_id=int(payload["cluster_id"]),
        blob_id=str(payload["blob_id"]),
        member_token_ids=[int(item) for item in (payload.get("member_token_ids") or [])],
    )


def _manifest_from_mapping(payload: Mapping[str, Any]) -> WeightManifest:
    return WeightManifest(
        origin_id=str(payload.get("origin_id") or ""),
        vocab_size=int(payload.get("vocab_size", 0)),
        embedding_dim=int(payload.get("embedding_dim", 0)),
        clusters=[_cluster_from_mapping(item) for item in (payload.get("clusters") or [])],
        metadata=dict(payload.get("metadata") or {}),
    )


class _PythonWeightManifestBackend:
    def __init__(self, manifest_path: str | Path, blob_root: str | Path | None = None) -> None:
        self._manifest_path = Path(manifest_path)
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._blob_store = BlobStore(blob_root) if blob_root is not None else None

    def load_manifest(self) -> WeightManifest | None:
        if not self._manifest_path.exists():
            return None
        payload = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Weight manifest must be a JSON object: {self._manifest_path}")
        return _manifest_from_mapping(payload)

    def resolve_origin_id(
        self,
        checkpoint_origin: str,
        resume_record: Mapping[str, Any] | None = None,
    ) -> tuple[str | None, str]:
        metadata = dict((resume_record or {}).get("metadata") or {})
        resume_origin_id = str(metadata.get("sodl_origin_id") or "").strip()
        if resume_origin_id:
            return resume_origin_id, "resume_record"

        manifest = self.load_manifest()
        if manifest is None:
            return None, "new"
        origin_id = str(manifest.origin_id).strip()
        if not origin_id:
            return None, "new"
        recorded_checkpoint_origin = str(manifest.metadata.get("checkpoint_origin") or "").strip()
        if recorded_checkpoint_origin and recorded_checkpoint_origin != checkpoint_origin:
            return None, "new"
        return origin_id, "manifest"

    def write_manifest(
        self,
        origin_id: str,
        vocab_size: int,
        embedding_dim: int,
        clusters: list[WeightManifestCluster],
        metadata: Mapping[str, Any] | None = None,
    ) -> WeightManifest:
        previous_manifest = self.load_manifest()
        resolved_metadata = dict((previous_manifest.metadata if previous_manifest is not None else {}) or {})
        if metadata is not None:
            resolved_metadata.update(dict(metadata))
        manifest = WeightManifest(
            origin_id=str(origin_id),
            vocab_size=int(vocab_size),
            embedding_dim=int(embedding_dim),
            clusters=sorted(clusters, key=lambda item: int(item.cluster_id)),
            metadata=resolved_metadata,
        )
        self._manifest_path.write_text(
            json.dumps(
                {
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
                    "metadata": manifest.metadata,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if previous_manifest is not None and self._blob_store is not None:
            live_blob_ids = {cluster.blob_id for cluster in manifest.clusters}
            stale_blob_ids = {
                cluster.blob_id
                for cluster in previous_manifest.clusters
                if cluster.blob_id not in live_blob_ids
            }
            for blob_id in sorted(stale_blob_ids):
                try:
                    self._blob_store.delete(blob_id)
                except FileNotFoundError:
                    continue
        return manifest


class WeightManifestStore:
    def __init__(self, manifest_path: str | Path, blob_root: str | Path | None = None) -> None:
        native = _rust_bridge.create_weight_manifest_store(
            str(manifest_path),
            str(blob_root) if blob_root is not None else None,
        )
        self._native = native
        self._fallback = None if native is not None else _PythonWeightManifestBackend(
            manifest_path,
            blob_root,
        )

    def _backend(self) -> Any:
        return self._native if self._native is not None else self._fallback

    def load_manifest(self) -> WeightManifest | None:
        backend = self._backend()
        if self._native is not None:
            payload = backend.load_manifest()
            return None if payload is None else _manifest_from_mapping(json.loads(payload))
        return backend.load_manifest()

    def resolve_origin_id(
        self,
        checkpoint_origin: str,
        resume_record: Mapping[str, Any] | None = None,
    ) -> tuple[str | None, str]:
        backend = self._backend()
        if self._native is not None:
            payload = json.loads(
                backend.resolve_origin_id(
                    checkpoint_origin,
                    json.dumps(dict(resume_record)) if resume_record is not None else None,
                )
            )
            origin_id = payload.get("origin_id")
            return (None if origin_id in (None, "") else str(origin_id), str(payload.get("source") or "new"))
        return backend.resolve_origin_id(checkpoint_origin, resume_record)

    def write_manifest(
        self,
        origin_id: str,
        vocab_size: int,
        embedding_dim: int,
        clusters: Iterable[WeightManifestCluster],
        metadata: Mapping[str, Any] | None = None,
    ) -> WeightManifest:
        cluster_list = list(clusters)
        backend = self._backend()
        if self._native is not None:
            payload = json.loads(
                backend.write_manifest(
                    origin_id,
                    int(vocab_size),
                    int(embedding_dim),
                    json.dumps(
                        [
                            {
                                "cluster_id": int(cluster.cluster_id),
                                "blob_id": str(cluster.blob_id),
                                "member_token_ids": [int(item) for item in cluster.member_token_ids],
                            }
                            for cluster in cluster_list
                        ]
                    ),
                    json.dumps(dict(metadata or {})),
                )
            )
            return _manifest_from_mapping(payload)
        return backend.write_manifest(origin_id, vocab_size, embedding_dim, cluster_list, metadata)


def export_manifest_clusters(
    token_index: Any,
    cluster_blob_ids: Mapping[int, str],
) -> list[WeightManifestCluster]:
    return [
        WeightManifestCluster(
            cluster_id=int(cluster_id),
            blob_id=str(blob_id),
            member_token_ids=[int(token_id) for token_id in token_index.cluster_members(int(cluster_id))],
        )
        for cluster_id, blob_id in sorted(cluster_blob_ids.items(), key=lambda item: int(item[0]))
    ]
