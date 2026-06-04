"""Checkpoint Manager — SODL-backed model checkpoint storage with lineage.

Stores model checkpoints (weights, optimizer state, metadata) as
content-addressed SODL blobs with automatic deduplication and lineage tracking.

Example
-------
>>> mgr = CheckpointManager(BlobStore("./blobs"), "./checkpoints")
>>> ckpt_id = mgr.save_checkpoint(model, optimizer, step=1000, origin_id="run-1")
>>> state = mgr.load_checkpoint(ckpt_id)
>>> model.load_state_dict(state["model"])
"""

from __future__ import annotations

import io
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import zstandard as zstd

from . import _rust_bridge
from sodl_weights.store import BlobStore, compute_blob_id


@dataclass
class CheckpointRecord:
    """Metadata for a stored checkpoint."""
    checkpoint_id: str
    blob_id: str
    origin_id: str
    step: int
    epoch: Optional[int] = None
    loss: Optional[float] = None
    metrics: dict[str, float] = field(default_factory=dict)
    parent_checkpoint_id: Optional[str] = None
    stage: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    optimizer_externalized: bool = False
    optimizer_origin_id: Optional[str] = None
    optimizer_layout_fingerprint: Optional[str] = None
    optimizer_block_count: int = 0
    dataset_manifests: list[str] = field(default_factory=list)
    knowledge_logs: list[str] = field(default_factory=list)
    size_raw: int = 0
    size_stored: int = 0
    created_at: str = ""
    notes: str = ""


def _record_from_mapping(payload: dict[str, Any]) -> CheckpointRecord:
    return CheckpointRecord(
        checkpoint_id=str(payload["checkpoint_id"]),
        blob_id=str(payload["blob_id"]),
        origin_id=str(payload["origin_id"]),
        step=int(payload["step"]),
        epoch=int(payload["epoch"]) if payload.get("epoch") is not None else None,
        loss=float(payload["loss"]) if payload.get("loss") is not None else None,
        metrics={str(key): float(value) for key, value in (payload.get("metrics") or {}).items()},
        parent_checkpoint_id=payload.get("parent_checkpoint_id"),
        stage=str(payload.get("stage") or ""),
        metadata=dict(payload.get("metadata") or {}),
        optimizer_externalized=bool(payload.get("optimizer_externalized", False)),
        optimizer_origin_id=payload.get("optimizer_origin_id"),
        optimizer_layout_fingerprint=payload.get("optimizer_layout_fingerprint"),
        optimizer_block_count=int(payload.get("optimizer_block_count", 0)),
        dataset_manifests=[str(item) for item in (payload.get("dataset_manifests") or [])],
        knowledge_logs=[str(item) for item in (payload.get("knowledge_logs") or [])],
        size_raw=int(payload.get("size_raw", 0)),
        size_stored=int(payload.get("size_stored", 0)),
        created_at=str(payload.get("created_at") or ""),
        notes=str(payload.get("notes") or ""),
    )


class CheckpointManager:
    """Manage model checkpoints with SODL content-addressed storage.

    Each checkpoint is serialized, compressed, and stored as a SODL blob.
    A registry JSON file tracks all checkpoints with their metadata.

    Parameters
    ----------
    blob_store : BlobStore
        SODL blob store for storing checkpoint data.
    registry_dir : str | Path
        Directory for checkpoint registry files.
    zstd_level : int
        Compression level for checkpoints (default 3).
    max_checkpoints : int
        Maximum checkpoints to keep per origin (oldest deleted first). 0 = unlimited.
    """

    def __init__(
        self,
        blob_store: BlobStore,
        registry_dir: str | Path = "checkpoints",
        zstd_level: int = 3,
        max_checkpoints: int = 0,
    ) -> None:
        self._blob_store = blob_store
        self._registry_dir = Path(registry_dir)
        self._registry_dir.mkdir(parents=True, exist_ok=True)
        self._compressor = zstd.ZstdCompressor(level=zstd_level)
        self._decompressor = zstd.ZstdDecompressor()
        self._max_ckpts = max_checkpoints
        blob_root = getattr(blob_store, "_root", None)
        self._backend = None
        if blob_root is not None:
            self._backend = _rust_bridge.create_checkpoint_store(
                str(blob_root),
                str(self._registry_dir),
                compression_level=zstd_level,
                max_checkpoints=max(0, int(max_checkpoints)),
            )

    def _registry_path(self, origin_id: str) -> Path:
        safe_name = origin_id.replace(":", "_").replace("/", "_")
        return self._registry_dir / f"{safe_name}.json"

    def _load_registry(self, origin_id: str) -> list[CheckpointRecord]:
        if self._backend is not None:
            payload = json.loads(self._backend.list_checkpoints(origin_id))
            return [_record_from_mapping(item) for item in payload]
        path = self._registry_path(origin_id)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [CheckpointRecord(**r) for r in data.get("checkpoints", [])]

    def _save_registry(self, origin_id: str, records: list[CheckpointRecord]) -> None:
        path = self._registry_path(origin_id)
        data = {"origin_id": origin_id, "checkpoints": [asdict(r) for r in records]}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _referenced_blob_ids(self) -> set[str]:
        referenced: set[str] = set()
        for path in self._registry_dir.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            for record in data.get("checkpoints", []):
                blob_id = record.get("blob_id")
                if blob_id:
                    referenced.add(str(blob_id))
        return referenced

    def _gc_checkpoint_blobs(self, records: list[CheckpointRecord]) -> None:
        if not records:
            return
        live_blob_ids = self._referenced_blob_ids()
        for blob_id in {record.blob_id for record in records if record.blob_id}:
            if blob_id in live_blob_ids:
                continue
            try:
                self._blob_store.delete(blob_id)
            except FileNotFoundError:
                continue

    def save_checkpoint(
        self,
        model: Any,
        optimizer: Any = None,
        *,
        step: int,
        origin_id: str,
        epoch: int | None = None,
        loss: float | None = None,
        metrics: dict[str, float] | None = None,
        parent_checkpoint_id: str | None = None,
        stage: str = "",
        metadata: dict[str, Any] | None = None,
        dataset_manifests: list[str] | None = None,
        knowledge_logs: list[str] | None = None,
        notes: str = "",
    ) -> str:
        """Save a model checkpoint to SODL.

        Parameters
        ----------
        model : nn.Module
            PyTorch model (or its state_dict).
        optimizer : Optimizer, optional
            PyTorch optimizer (state_dict is saved if provided).
        step : int
            Training step number.
        origin_id : str
            Origin/run identifier for grouping checkpoints.
        epoch : int, optional
            Training epoch.
        loss : float, optional
            Loss at this checkpoint.
        metrics : dict, optional
            Additional metrics at this checkpoint.
        notes : str, optional
            Free-text notes.

        Returns
        -------
        str
            Checkpoint ID for later retrieval.
        """
        import torch

        # Build checkpoint dict
        checkpoint_data: dict[str, Any] = {"step": step}

        if hasattr(model, "state_dict"):
            checkpoint_data["model"] = model.state_dict()
        elif isinstance(model, dict):
            checkpoint_data["model"] = model
        else:
            raise TypeError(f"model must be nn.Module or dict, got {type(model)}")

        optimizer_externalized = False
        optimizer_origin_id: str | None = None
        optimizer_layout_fingerprint: str | None = None
        optimizer_block_count = 0

        if optimizer is not None:
            if hasattr(optimizer, "external_state_dict"):
                external_state = optimizer.external_state_dict()
                checkpoint_data["optimizer_manifest"] = external_state
                checkpoint_data["optimizer_externalized"] = True
                optimizer_externalized = True
                optimizer_origin_id = external_state.get("origin_id")
                optimizer_layout_fingerprint = external_state.get("layout_fingerprint")
                optimizer_block_count = len(external_state.get("manifest", {}).get("blocks", {}))
            elif hasattr(optimizer, "state_dict"):
                checkpoint_data["optimizer"] = optimizer.state_dict()
            elif isinstance(optimizer, dict):
                checkpoint_data["optimizer"] = optimizer

        if dataset_manifests:
            checkpoint_data["dataset_manifests"] = list(dataset_manifests)
        if knowledge_logs:
            checkpoint_data["knowledge_logs"] = list(knowledge_logs)

        if epoch is not None:
            checkpoint_data["epoch"] = epoch
        if loss is not None:
            checkpoint_data["loss"] = loss
        if metrics:
            checkpoint_data["metrics"] = metrics

        # Serialize with torch.save to bytes
        buf = io.BytesIO()
        torch.save(checkpoint_data, buf)
        raw = buf.getvalue()
        if self._backend is not None:
            record = _record_from_mapping(
                json.loads(
                    self._backend.save_checkpoint(
                        origin_id,
                        raw,
                        json.dumps(
                            {
                                "step": int(step),
                                "epoch": int(epoch) if epoch is not None else None,
                                "loss": float(loss) if loss is not None else None,
                                "metrics": dict(metrics or {}),
                                "parent_checkpoint_id": parent_checkpoint_id,
                                "stage": stage,
                                "metadata": dict(metadata or {}),
                                "optimizer_externalized": optimizer_externalized,
                                "optimizer_origin_id": optimizer_origin_id,
                                "optimizer_layout_fingerprint": optimizer_layout_fingerprint,
                                "optimizer_block_count": int(optimizer_block_count),
                                "dataset_manifests": list(dataset_manifests or []),
                                "knowledge_logs": list(knowledge_logs or []),
                                "notes": notes,
                            }
                        ),
                    )
                )
            )
            return record.checkpoint_id

        raw_size = len(raw)
        compressed = self._compressor.compress(raw)
        blob_id = compute_blob_id(compressed)
        self._blob_store.put(blob_id, compressed)

        checkpoint_id = f"ckpt:{uuid.uuid4()}"
        record = CheckpointRecord(
            checkpoint_id=checkpoint_id,
            blob_id=blob_id,
            origin_id=origin_id,
            step=step,
            epoch=epoch,
            loss=loss,
            metrics=dict(metrics or {}),
            parent_checkpoint_id=parent_checkpoint_id,
            stage=stage,
            metadata=dict(metadata or {}),
            optimizer_externalized=optimizer_externalized,
            optimizer_origin_id=optimizer_origin_id,
            optimizer_layout_fingerprint=optimizer_layout_fingerprint,
            optimizer_block_count=optimizer_block_count,
            dataset_manifests=list(dataset_manifests or []),
            knowledge_logs=list(knowledge_logs or []),
            size_raw=raw_size,
            size_stored=len(compressed),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            notes=notes,
        )

        records = self._load_registry(origin_id)
        records.append(record)

        evicted_records: list[CheckpointRecord] = []
        if self._max_ckpts > 0 and len(records) > self._max_ckpts:
            evicted_records = records[:-self._max_ckpts]
            records = records[-self._max_ckpts:]

        self._save_registry(origin_id, records)
        self._gc_checkpoint_blobs(evicted_records)
        return checkpoint_id

    def load_checkpoint(self, origin_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
        """Load a checkpoint from SODL.

        Parameters
        ----------
        origin_id : str
            Origin/run identifier.
        checkpoint_id : str, optional
            Specific checkpoint ID. If None, loads the latest.

        Returns
        -------
        dict
            Checkpoint dict with "model", "optimizer" (if saved), "step", etc.
        """
        import torch

        if self._backend is not None:
            try:
                raw = bytes(self._backend.load_checkpoint(origin_id, checkpoint_id))
            except FileNotFoundError:
                raise
            except Exception as exc:
                if "not found" in str(exc).lower():
                    raise FileNotFoundError(
                        f"Checkpoint {checkpoint_id or '[latest]'} not found for origin {origin_id}"
                    ) from exc
                raise
            buf = io.BytesIO(raw)
            return torch.load(buf, map_location="cpu", weights_only=False)

        records = self._load_registry(origin_id)
        if not records:
            raise FileNotFoundError(f"No checkpoints found for origin {origin_id}")

        if checkpoint_id:
            record = next((r for r in records if r.checkpoint_id == checkpoint_id), None)
            if record is None:
                raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found")
        else:
            record = records[-1]  # latest

        # Load and decompress
        compressed = self._blob_store.get(record.blob_id)
        raw = self._decompressor.decompress(compressed)
        buf = io.BytesIO(raw)
        return torch.load(buf, map_location="cpu", weights_only=False)

    def list_checkpoints(self, origin_id: str) -> list[CheckpointRecord]:
        """List all checkpoints for an origin, ordered by step."""
        records = self._load_registry(origin_id)
        return sorted(records, key=lambda r: r.step)

    def get_latest(self, origin_id: str) -> CheckpointRecord | None:
        """Get the latest checkpoint record for an origin."""
        records = self._load_registry(origin_id)
        return records[-1] if records else None

    def get_lineage(self, origin_id: str, checkpoint_id: str | None = None) -> list[CheckpointRecord]:
        """Return lineage from the root checkpoint to the target (or latest) checkpoint."""
        if self._backend is not None:
            payload = json.loads(self._backend.get_lineage(origin_id, checkpoint_id))
            return [_record_from_mapping(item) for item in payload]
        records = self._load_registry(origin_id)
        if not records:
            return []
        record_by_id = {record.checkpoint_id: record for record in records}
        current = record_by_id.get(checkpoint_id) if checkpoint_id else records[-1]
        if current is None:
            raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found")
        lineage: list[CheckpointRecord] = []
        while current is not None:
            lineage.append(current)
            current = record_by_id.get(current.parent_checkpoint_id) if current.parent_checkpoint_id else None
        lineage.reverse()
        return lineage

    def resolve_resume(self, origin_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
        """Load checkpoint state together with the selected record and lineage metadata."""
        records = self._load_registry(origin_id)
        if not records:
            raise FileNotFoundError(f"No checkpoints found for origin {origin_id}")
        record = None
        if checkpoint_id is not None:
            record = next((candidate for candidate in records if candidate.checkpoint_id == checkpoint_id), None)
            if record is None:
                raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found")
        else:
            record = records[-1]
        state = self.load_checkpoint(origin_id, record.checkpoint_id)
        lineage = self.get_lineage(origin_id, record.checkpoint_id)
        return {
            "state": state,
            "record": asdict(record),
            "lineage": [asdict(item) for item in lineage],
            "optimizer_resume": {
                "externalized": record.optimizer_externalized,
                "optimizer_origin_id": record.optimizer_origin_id,
                "layout_fingerprint": record.optimizer_layout_fingerprint,
                "block_count": record.optimizer_block_count,
            },
            "dataset_manifests": list(record.dataset_manifests),
            "knowledge_logs": list(record.knowledge_logs),
        }

    def delete_checkpoint(self, origin_id: str, checkpoint_id: str) -> bool:
        """Remove a checkpoint from the registry."""
        if self._backend is not None:
            return bool(self._backend.delete_checkpoint(origin_id, checkpoint_id))
        records = self._load_registry(origin_id)
        removed = [r for r in records if r.checkpoint_id == checkpoint_id]
        new_records = [r for r in records if r.checkpoint_id != checkpoint_id]
        if len(new_records) < len(records):
            self._save_registry(origin_id, new_records)
            self._gc_checkpoint_blobs(removed)
            return True
        return False

    def diff_checkpoints(
        self, origin_id: str, old_id: str, new_id: str
    ) -> dict[str, Any]:
        """Compare two checkpoints and return a summary of differences.

        Returns
        -------
        dict
            Keys: "step_delta", "loss_delta", "metric_deltas", "param_changes"
        """
        if self._backend is not None:
            return dict(json.loads(self._backend.diff_checkpoints(origin_id, old_id, new_id)))
        records = self._load_registry(origin_id)
        old_rec = next((r for r in records if r.checkpoint_id == old_id), None)
        new_rec = next((r for r in records if r.checkpoint_id == new_id), None)

        if old_rec is None or new_rec is None:
            raise FileNotFoundError("One or both checkpoint IDs not found")

        result: dict[str, Any] = {
            "step_delta": new_rec.step - old_rec.step,
            "loss_delta": None,
            "metric_deltas": {},
        }

        if old_rec.loss is not None and new_rec.loss is not None:
            result["loss_delta"] = new_rec.loss - old_rec.loss

        # Compare metrics
        all_keys = set(old_rec.metrics.keys()) | set(new_rec.metrics.keys())
        for key in all_keys:
            old_val = old_rec.metrics.get(key)
            new_val = new_rec.metrics.get(key)
            if old_val is not None and new_val is not None:
                result["metric_deltas"][key] = new_val - old_val

        return result
