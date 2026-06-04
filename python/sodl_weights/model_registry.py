"""Model registry for lineage-aware SODL model management."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4()}"


@dataclass
class BaseModelRecord:
    origin_id: str
    model_name: str
    quantization: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)


@dataclass
class LoRADerivationRecord:
    derivation_id: str
    adapter_name: str
    base_origin_id: str
    manifest_path: str
    blob_dir: str
    quantization: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)


@dataclass
class GGUFDerivationRecord:
    derivation_id: str
    artifact_name: str
    parent_id: str
    output_path: str
    quantization: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)


@dataclass
class TrainingLineageRecord:
    lineage_id: str
    adapter_derivation_id: str
    base_origin_id: str
    train_file: str
    val_file: str
    dataset_manifest: str
    hyperparameters: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)


class ModelRegistry:
    """Tracks base models, derivations, and training lineage."""

    def __init__(
        self,
        *,
        base_models: dict[str, BaseModelRecord] | None = None,
        lora_derivations: dict[str, LoRADerivationRecord] | None = None,
        gguf_derivations: dict[str, GGUFDerivationRecord] | None = None,
        training_lineage: list[TrainingLineageRecord] | None = None,
    ) -> None:
        self.base_models = base_models or {}
        self.lora_derivations = lora_derivations or {}
        self.gguf_derivations = gguf_derivations or {}
        self.training_lineage = training_lineage or []

    def register_base_model(
        self,
        *,
        origin_id: str,
        model_name: str,
        quantization: str,
        metadata: dict[str, Any] | None = None,
    ) -> BaseModelRecord:
        existing = self.base_models.get(origin_id)
        if existing is not None:
            return existing
        record = BaseModelRecord(
            origin_id=origin_id,
            model_name=model_name,
            quantization=quantization,
            metadata=dict(metadata or {}),
        )
        self.base_models[origin_id] = record
        return record

    def register_lora_derivation(
        self,
        *,
        adapter_name: str,
        base_origin_id: str,
        manifest_path: str,
        blob_dir: str,
        quantization: str,
        metadata: dict[str, Any] | None = None,
        derivation_id: str | None = None,
    ) -> LoRADerivationRecord:
        for record in self.lora_derivations.values():
            if (
                record.adapter_name == adapter_name
                and record.base_origin_id == base_origin_id
                and record.manifest_path == manifest_path
            ):
                return record
        record = LoRADerivationRecord(
            derivation_id=derivation_id or _new_id("lora"),
            adapter_name=adapter_name,
            base_origin_id=base_origin_id,
            manifest_path=manifest_path,
            blob_dir=blob_dir,
            quantization=quantization,
            metadata=dict(metadata or {}),
        )
        self.lora_derivations[record.derivation_id] = record
        return record

    def register_gguf_derivation(
        self,
        *,
        artifact_name: str,
        parent_id: str,
        output_path: str,
        quantization: str,
        metadata: dict[str, Any] | None = None,
        derivation_id: str | None = None,
    ) -> GGUFDerivationRecord:
        for record in self.gguf_derivations.values():
            if record.output_path == output_path and record.parent_id == parent_id:
                return record
        record = GGUFDerivationRecord(
            derivation_id=derivation_id or _new_id("gguf"),
            artifact_name=artifact_name,
            parent_id=parent_id,
            output_path=output_path,
            quantization=quantization,
            metadata=dict(metadata or {}),
        )
        self.gguf_derivations[record.derivation_id] = record
        return record

    def record_training_lineage(
        self,
        *,
        adapter_derivation_id: str,
        base_origin_id: str,
        train_file: str,
        val_file: str,
        dataset_manifest: str,
        hyperparameters: dict[str, Any],
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        lineage_id: str | None = None,
    ) -> TrainingLineageRecord:
        record = TrainingLineageRecord(
            lineage_id=lineage_id or _new_id("lineage"),
            adapter_derivation_id=adapter_derivation_id,
            base_origin_id=base_origin_id,
            train_file=train_file,
            val_file=val_file,
            dataset_manifest=dataset_manifest,
            hyperparameters=dict(hyperparameters),
            metrics=dict(metrics or {}),
            metadata=dict(metadata or {}),
        )
        self.training_lineage.append(record)
        return record

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_models": {key: asdict(value) for key, value in self.base_models.items()},
            "lora_derivations": {
                key: asdict(value) for key, value in self.lora_derivations.items()
            },
            "gguf_derivations": {
                key: asdict(value) for key, value in self.gguf_derivations.items()
            },
            "training_lineage": [asdict(value) for value in self.training_lineage],
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ModelRegistry":
        target = Path(path)
        payload = json.loads(target.read_text(encoding="utf-8"))
        return cls(
            base_models={
                key: BaseModelRecord(**value)
                for key, value in payload.get("base_models", {}).items()
            },
            lora_derivations={
                key: LoRADerivationRecord(**value)
                for key, value in payload.get("lora_derivations", {}).items()
            },
            gguf_derivations={
                key: GGUFDerivationRecord(**value)
                for key, value in payload.get("gguf_derivations", {}).items()
            },
            training_lineage=[
                TrainingLineageRecord(**value)
                for value in payload.get("training_lineage", [])
            ],
        )
