from __future__ import annotations

import tempfile

import numpy as np

from sodl_weights import BlobStore
from sodl_weights.artifact_store import ArtifactStore
from sodl_weights.context_logic import (
    ClusteredAttentionLayer,
    SCLClusteredDecoder,
    SemanticContextLogic,
)
from sodl_weights.service import WeightStoreService
from sodl_weights.token_hash import TokenHashIndex


def _setup_artifact_store():
    tmpdir = tempfile.mkdtemp()
    blob_store = BlobStore(tmpdir + "/blobs")
    artifact_store = ArtifactStore(blob_store, tmpdir + "/manifests")
    return artifact_store


def test_scl_query_returns_gated_retrieval() -> None:
    logic = SemanticContextLogic(n_clusters=2, top_k_clusters=1)
    keys = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ],
        dtype=np.float32,
    )
    values = keys.copy()
    stats = logic.build(keys, values)
    result = logic.query(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k_items=2)

    assert stats["n_clusters"] == 2
    assert len(result.active_clusters) == 1
    assert result.retrieved_indices[0] in {0, 1}
    assert result.attention_output.shape == (3,)


def test_scl_detects_mode_and_pins_clusters() -> None:
    logic = SemanticContextLogic(n_clusters=2, top_k_clusters=1)
    keys = np.eye(3, dtype=np.float32)
    values = keys.copy()
    logic.build(keys, values)
    logic.pin_clusters_for_mode("code", [1])
    result = logic.query(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        context_text="def hello():\n    return 1",
    )

    assert result.mode == "code"
    assert 1 in result.active_clusters


def test_scl_memory_log_persists_as_jsonl() -> None:
    logic = SemanticContextLogic()
    artifact_store = _setup_artifact_store()
    logic.record_memory(
        "origin:scl",
        kind="retrieval_hit",
        query="What is Python?",
        content="Python is a programming language.",
        metadata={"url": "https://python.org"},
    )
    artifact = logic.persist_memory_log("origin:scl", artifact_store=artifact_store)
    payload = artifact_store.load(artifact.blob_id).decode("utf-8")

    assert '"kind": "retrieval_hit"' in payload
    assert '"origin_id": "origin:scl"' in payload


def test_clustered_attention_layer_batches_queries() -> None:
    layer = ClusteredAttentionLayer(n_clusters=2, top_k_clusters=1)
    keys = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    values = keys.copy()
    outputs, results = layer.forward(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        keys,
        values,
        context_texts=["def answer(): return 1", "What is Rust?"],
        top_k_items=2,
    )

    assert outputs.shape == (2, 3)
    assert results[0].mode == "code"
    assert results[1].mode == "qa"


def test_scl_memory_persists_to_weight_store_and_rehydrates() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        logic = SemanticContextLogic(n_clusters=2, top_k_clusters=1)
        service = WeightStoreService(tmpdir + "/blobs", pin_registry_path=tmpdir + "/pins.json")

        origin = service.create_model("scl-memory-test", "float32")
        logic.record_memory(
            origin.origin_id,
            kind="fact",
            query="What is Python?",
            content="Python is a programming language used for scripting and applications.",
        )
        logic.record_memory(
            origin.origin_id,
            kind="code",
            query="Show hello world",
            content="def hello():\n    return 'hello world'",
        )

        manifest = logic.persist_memory_to_weight_store(
            origin.origin_id,
            weight_store_service=service,
            model_name="scl-memory-test",
            dim=16,
        )

        assert manifest.cluster_blob_ids
        assert manifest.total_records == 2
        stats = logic.hydrate_memory_from_weight_store(
            origin.origin_id,
            manifest.cluster_blob_ids,
            weight_store_service=service,
        )
        assert stats["n_clusters"] >= 1
        assert service.cache_size() >= 1


def test_scl_clustered_decoder_predicts_from_attention_output() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ],
        dtype=np.float32,
    )
    token_index = TokenHashIndex(n_clusters=2, top_k_clusters=2)
    token_index.build(embeddings)
    decoder = SCLClusteredDecoder(token_index, n_clusters=2, top_k_clusters=1)

    predictions, results = decoder.predict(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        embeddings,
        embeddings,
        context_texts=["What token is closest to python-style vector?"],
        top_k_items=2,
        top_k_tokens=3,
    )

    assert len(predictions) == 1
    assert predictions[0]
    assert predictions[0][0].token_id in {0, 1}
    assert results[0].mode == "qa"
