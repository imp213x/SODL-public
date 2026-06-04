from __future__ import annotations

from pathlib import Path

import sys

SODL_PY = Path(__file__).resolve().parents[1]
if str(SODL_PY) not in sys.path:
    sys.path.insert(0, str(SODL_PY))

from sodl_weights.model_registry import ModelRegistry


def test_registry_roundtrip_tracks_base_lora_and_gguf(tmp_path: Path) -> None:
    registry = ModelRegistry()
    base = registry.register_base_model(
        origin_id="origin:base-qwen3",
        model_name="qwen3-4b-base",
        quantization="f16",
        metadata={"family": "qwen3"},
    )
    lora = registry.register_lora_derivation(
        adapter_name="carla-sft-lora",
        base_origin_id=base.origin_id,
        manifest_path="C:/tmp/lora_manifest.json",
        blob_dir="C:/tmp/lora_blobs",
        quantization="lora-delta",
        metadata={"semantic_mode": "color"},
    )
    gguf = registry.register_gguf_derivation(
        artifact_name="carla-qwen3-4b.gguf",
        parent_id=lora.derivation_id,
        output_path="C:/tmp/carla-qwen3-4b.gguf",
        quantization="Q4_K_M",
    )

    registry_path = tmp_path / "registry.json"
    registry.save(registry_path)
    loaded = ModelRegistry.load(registry_path)

    assert base.origin_id in loaded.base_models
    assert lora.derivation_id in loaded.lora_derivations
    assert gguf.derivation_id in loaded.gguf_derivations
    assert loaded.lora_derivations[lora.derivation_id].base_origin_id == base.origin_id
    assert loaded.gguf_derivations[gguf.derivation_id].parent_id == lora.derivation_id


def test_training_lineage_links_dataset_and_hyperparameters() -> None:
    registry = ModelRegistry()
    registry.register_base_model(
        origin_id="origin:base-qwen3",
        model_name="qwen3-4b-base",
        quantization="f16",
    )
    lora = registry.register_lora_derivation(
        adapter_name="carla-sft-lora",
        base_origin_id="origin:base-qwen3",
        manifest_path="C:/tmp/lora_manifest.json",
        blob_dir="C:/tmp/lora_blobs",
        quantization="lora-delta",
    )

    lineage = registry.record_training_lineage(
        adapter_derivation_id=lora.derivation_id,
        base_origin_id="origin:base-qwen3",
        train_file="C:/tmp/train.jsonl",
        val_file="C:/tmp/val.jsonl",
        dataset_manifest="C:/tmp/train_manifest.json",
        hyperparameters={"learning_rate": 2e-4, "epochs": 1.0},
        metrics={"train_loss": 1.23},
        metadata={"pipeline_hash": "abc123"},
    )

    assert lineage.adapter_derivation_id == lora.derivation_id
    assert lineage.hyperparameters["learning_rate"] == 2e-4
    assert lineage.metrics["train_loss"] == 1.23
    assert registry.training_lineage[0].metadata["pipeline_hash"] == "abc123"
