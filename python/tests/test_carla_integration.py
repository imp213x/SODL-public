"""
Cross-validation tests for the SODL Weight Store ↔ Carla integration.

These tests run without a real model — they use synthetic weight matrices
to validate the full export → store → load → verify pipeline.

For real-model tests, run:
    python scripts/sodl_weight_export.py --model models/carla-merged
    python scripts/sodl_weight_load.py --verify
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure SDK is importable when running tests from a checkout.
SODL_PY = Path(__file__).resolve().parents[1]
if SODL_PY.exists():
    sys.path.insert(0, str(SODL_PY))

from sodl_weights.service import WeightStoreService
from sodl_weights.store import BlobStore, SodlIntegrityError, WeightBlobStore
from sodl_weights.pin_registry import WeightPinError
from sodl_weights.types import WeightCluster, WeightPinReason


def _make_synthetic_embeddings(vocab_size: int = 1000, dim: int = 64):
    """Create a synthetic embedding matrix for testing."""
    rng = np.random.RandomState(42)
    return rng.randn(vocab_size, dim).astype(np.float32)


def _cluster_embeddings(embeddings: np.ndarray, n_clusters: int = 32):
    """Simple K-means clustering for test purposes."""
    n_tokens, dim = embeddings.shape
    rng = np.random.RandomState(42)
    idx = rng.choice(n_tokens, n_clusters, replace=False)
    centroids = embeddings[idx].copy()

    for _ in range(20):
        labels = np.empty(n_tokens, dtype=np.int32)
        for i in range(0, n_tokens, 512):
            batch = embeddings[i:i + 512]
            dists = np.sum((batch[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            labels[i:i + 512] = np.argmin(dists, axis=1)

        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                centroids[k] = embeddings[mask].mean(axis=0)

    clusters = []
    for k in range(n_clusters):
        mask = labels == k
        member_ids = np.where(mask)[0].tolist()
        if not member_ids:
            continue
        centroid = centroids[k].tolist()
        offsets = (embeddings[mask] - centroids[k]).tolist()
        clusters.append(WeightCluster(
            centroid=centroid,
            member_token_ids=member_ids,
            offsets=offsets,
            dim=dim,
        ))

    return clusters, labels


class TestEndToEndPipeline:
    """Full export → store → load → verify pipeline."""

    def test_roundtrip_preserves_floats(self, tmp_path: Path) -> None:
        """The essential test: store embeddings via SODL, reconstruct, verify equality."""
        embeddings = _make_synthetic_embeddings(500, 32)
        clusters, _ = _cluster_embeddings(embeddings, n_clusters=16)

        svc = WeightStoreService(str(tmp_path / "blobs"))
        model = svc.create_model("test-model", "F32")
        summary = svc.import_clusters(model.origin_id, clusters)

        # Reconstruct
        reconstructed = np.zeros_like(embeddings)
        for cluster, blob_id in zip(clusters, summary.cluster_ids):
            loaded = svc.load_cluster(model.origin_id, blob_id)
            centroid = np.array(loaded.centroid, dtype=np.float32)
            for j, token_id in enumerate(loaded.member_token_ids):
                offset = np.array(loaded.offsets[j], dtype=np.float32)
                reconstructed[token_id] = centroid + offset

        assert np.allclose(embeddings, reconstructed, atol=1e-6), \
            f"Max diff: {np.max(np.abs(embeddings - reconstructed))}"

    def test_manifest_roundtrip(self, tmp_path: Path) -> None:
        """Manifest can be serialised and used for reconstruction."""
        embeddings = _make_synthetic_embeddings(200, 16)
        clusters, _ = _cluster_embeddings(embeddings, n_clusters=8)

        svc = WeightStoreService(str(tmp_path / "blobs"))
        model = svc.create_model("manifest-test", "Q4_K_M")
        summary = svc.import_clusters(model.origin_id, clusters)

        # Write manifest
        manifest = {
            "origin_id": model.origin_id,
            "vocab_size": 200,
            "embedding_dim": 16,
            "clusters": [
                {"blob_id": bid, "member_token_ids": c.member_token_ids}
                for c, bid in zip(clusters, summary.cluster_ids)
            ],
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        # Load from manifest
        loaded_manifest = json.loads(manifest_path.read_text())
        svc2 = WeightStoreService(str(tmp_path / "blobs"))
        reconstructed = np.zeros((200, 16), dtype=np.float32)

        for cm in loaded_manifest["clusters"]:
            cluster = svc2.load_cluster(loaded_manifest["origin_id"], cm["blob_id"])
            centroid = np.array(cluster.centroid, dtype=np.float32)
            for j, tid in enumerate(cluster.member_token_ids):
                offset = np.array(cluster.offsets[j], dtype=np.float32)
                reconstructed[tid] = centroid + offset

        assert np.allclose(embeddings, reconstructed, atol=1e-6)


class TestDeduplication:
    """Dedup behaviour across imports."""

    def test_second_import_fully_deduped(self, tmp_path: Path) -> None:
        embeddings = _make_synthetic_embeddings(100, 16)
        clusters, _ = _cluster_embeddings(embeddings, n_clusters=8)

        svc = WeightStoreService(str(tmp_path / "blobs"))
        model = svc.create_model("dedup-test", "F32")

        s1 = svc.import_clusters(model.origin_id, clusters)
        s2 = svc.import_clusters(model.origin_id, clusters)

        assert s1.total_blobs_stored > 0
        assert s2.deduped_blobs == s2.total_clusters
        assert s2.total_blobs_stored == 0  # all deduped on second pass


class TestIntegrity:
    """Integrity verification catches tampering."""

    def test_tampered_blob_rejected(self, tmp_path: Path) -> None:
        blob_store = BlobStore(tmp_path / "blobs")
        ws = WeightBlobStore(blob_store)
        cluster = WeightCluster(
            centroid=[1.0, 2.0], member_token_ids=[0], offsets=[[0.0, 0.0]], dim=2,
        )

        stats = ws.put("origin:1", cluster)
        raw = blob_store.get(stats.blob_id)
        tampered = bytes([raw[0] ^ 0xFF]) + raw[1:]
        blob_store.put(stats.blob_id, tampered)

        with pytest.raises(SodlIntegrityError):
            ws.get("origin:1", stats.blob_id)


class TestPinProtection:
    """Identity pins survive eviction pressure."""

    def test_identity_cluster_persists(self, tmp_path: Path) -> None:
        svc = WeightStoreService(str(tmp_path / "blobs"), cache_capacity=3)
        model = svc.create_model("pin-test", "F32")

        # Store and identity-pin a special cluster
        identity_cluster = WeightCluster(
            centroid=[99.0] * 8, member_token_ids=[0], offsets=[[0.0] * 8], dim=8,
        )
        stats_identity = svc.store_cluster(model.origin_id, identity_cluster)
        svc.pin_identity_cluster(model.origin_id, stats_identity.blob_id)

        # Fill cache beyond capacity with regular clusters
        for i in range(5):
            c = WeightCluster(
                centroid=[float(i)] * 8, member_token_ids=[i + 1],
                offsets=[[0.0] * 8], dim=8,
            )
            s = svc.store_cluster(model.origin_id, c)
            svc.load_cluster(model.origin_id, s.blob_id)

        # Identity cluster must still be cached
        assert svc.is_cached(stats_identity.blob_id)

        # And cannot be evicted manually
        with pytest.raises(WeightPinError):
            svc.evict_cluster(stats_identity.blob_id)
