"""Shared types for the SODL Weight Store SDK."""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from typing import Optional


class WeightPinReason(enum.Enum):
    """Why a weight cluster is pinned in the hot cache."""
    IDENTITY = "identity"          # Core identity weights — never evicted
    LOGIC = "logic"                # Logic / routing weights — keep resident
    FREQUENT_USE = "frequent_use"  # High access frequency
    PREFETCH = "prefetch"          # Predicted to be needed soon


@dataclass
class WeightCluster:
    """A cluster of semantically related weight vectors.

    Stores a centroid vector plus per-token lightweight offsets.
    Multiple tokens share the same centroid — "store once, reference many."
    """
    centroid: list[float]
    member_token_ids: list[int]
    offsets: list[list[float]]
    dim: int
    cluster_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "centroid": self.centroid,
            "member_token_ids": self.member_token_ids,
            "offsets": self.offsets,
            "dim": self.dim,
            "cluster_id": self.cluster_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> WeightCluster:
        return cls(
            centroid=d["centroid"],
            member_token_ids=d["member_token_ids"],
            offsets=d["offsets"],
            dim=d["dim"],
            cluster_id=d.get("cluster_id"),
        )


@dataclass
class StoreStats:
    """Statistics from a store operation."""
    blob_id: str
    raw_bytes: int
    compressed_bytes: int
    stored_bytes: int
    was_deduped: bool


@dataclass
class ImportSummary:
    """Summary of a bulk import operation."""
    origin_id: str
    total_clusters: int
    total_blobs_stored: int
    deduped_blobs: int
    total_raw_bytes: int
    total_stored_bytes: int
    cluster_ids: list[str] = field(default_factory=list)


@dataclass
class WeightOrigin:
    """Metadata for a weight store origin."""
    origin_id: str
    model_name: str
    num_clusters: int
    quantization: str

    @staticmethod
    def new(model_name: str, quantization: str) -> WeightOrigin:
        return WeightOrigin(
            origin_id=f"origin:{uuid.uuid4()}",
            model_name=model_name,
            num_clusters=0,
            quantization=quantization,
        )
