"""Tests for the Token Hash Index — hierarchical softmax, gradient sharing, sparse updates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import sys
SODL_PY = Path(__file__).resolve().parents[1]
if str(SODL_PY) not in sys.path:
    sys.path.insert(0, str(SODL_PY))

from sodl_weights.token_hash import TokenHashIndex
from sodl_weights.artifact_store import ArtifactStore
from sodl_weights.store import BlobStore


def _synthetic_embeddings(vocab: int = 1000, dim: int = 64) -> np.ndarray:
    rng = np.random.RandomState(42)
    return rng.randn(vocab, dim).astype(np.float32)


class TestBuild:
    def test_builds_successfully(self) -> None:
        emb = _synthetic_embeddings()
        idx = TokenHashIndex(n_clusters=32)
        stats = idx.build(emb)

        assert idx.is_built
        assert idx.vocab_size == 1000
        assert idx.dim == 64
        assert stats["n_clusters"] == 32
        assert stats["cluster_size_min"] >= 1
        assert stats["compression_potential"] > 0

    def test_every_token_assigned(self) -> None:
        emb = _synthetic_embeddings(500, 32)
        idx = TokenHashIndex(n_clusters=16)
        idx.build(emb)

        for tid in range(500):
            cid = idx.token_hash(tid)
            assert 0 <= cid < 16
            assert tid in idx.cluster_members(cid)

    def test_reconstruction_exact(self) -> None:
        emb = _synthetic_embeddings(200, 16)
        idx = TokenHashIndex(n_clusters=8)
        idx.build(emb)

        for tid in range(200):
            reconstructed = idx.reconstruct(tid)
            np.testing.assert_allclose(reconstructed, emb[tid], atol=1e-5)

    def test_adaptive_build_uses_sqrt_vocab_initial_clusters(self) -> None:
        emb = _synthetic_embeddings(400, 16)
        idx = TokenHashIndex(n_clusters=8)
        stats = idx.build(emb, adaptive=True, max_iter=10)

        assert stats["initial_n_clusters"] == 20
        assert stats["silhouette_score"] is None or -1.0 <= stats["silhouette_score"] <= 1.0

    def test_adaptive_split_rebalances_imbalanced_clusters(self) -> None:
        rng = np.random.RandomState(7)
        giant = rng.normal(0.0, 0.08, size=(600, 8))
        tiny_a = rng.normal(5.0, 0.08, size=(20, 8))
        tiny_b = rng.normal(-5.0, 0.08, size=(20, 8))
        emb = np.vstack([giant, tiny_a, tiny_b]).astype(np.float32)

        idx = TokenHashIndex(n_clusters=3)
        baseline = idx.build(emb, adaptive=False, max_iter=15)
        adaptive = idx.build(
            emb,
            adaptive=True,
            max_iter=15,
            max_cluster_ratio=2.0,
            max_rebalance_steps=4,
        )

        assert adaptive["adaptive_rebalanced"] is True
        assert adaptive["rebalance_steps"] >= 1
        assert adaptive["n_clusters"] > adaptive["initial_n_clusters"]

    def test_adaptive_merge_reduces_over_uniform_clusters(self) -> None:
        clusters: list[np.ndarray] = []
        for cluster_id in range(16):
            center = np.full(8, cluster_id * 5.0, dtype=np.float32)
            cluster_rng = np.random.RandomState(cluster_id)
            points = center + cluster_rng.normal(0.0, 0.02, size=(16, 8)).astype(np.float32)
            clusters.append(points)
        emb = np.vstack(clusters)
        idx = TokenHashIndex(n_clusters=32)
        adaptive = idx.build(
            emb,
            adaptive=True,
            max_iter=10,
            uniform_ratio_threshold=1.5,
            max_rebalance_steps=3,
        )

        assert adaptive["initial_n_clusters"] == 16
        assert adaptive["n_clusters"] < adaptive["initial_n_clusters"]
        assert adaptive["adaptive_rebalanced"] is True
        assert adaptive["rebalance_steps"] >= 1


class TestHierarchicalSoftmax:
    def test_returns_probabilities(self) -> None:
        emb = _synthetic_embeddings(500, 32)
        idx = TokenHashIndex(n_clusters=16, top_k_clusters=3)
        idx.build(emb)

        hidden = np.random.RandomState(99).randn(32).astype(np.float32)
        results = idx.hierarchical_softmax(hidden)

        assert len(results) > 0
        for r in results:
            assert 0 <= r.probability <= 1
            assert 0 <= r.token_id < 500

    def test_probabilities_sum_approximately(self) -> None:
        emb = _synthetic_embeddings(200, 16)
        idx = TokenHashIndex(n_clusters=8, top_k_clusters=8)  # all clusters
        idx.build(emb)

        hidden = np.random.RandomState(99).randn(16).astype(np.float32)
        results = idx.hierarchical_softmax(hidden)

        total_p = sum(r.probability for r in results)
        # With all clusters searched, total should be close to 1.0
        assert abs(total_p - 1.0) < 0.05, f"Total prob = {total_p}"

    def test_top_token_is_plausible(self) -> None:
        emb = _synthetic_embeddings(100, 16)
        idx = TokenHashIndex(n_clusters=4, top_k_clusters=4)
        idx.build(emb)

        # Use an embedding as the hidden state — its own token should score high
        hidden = emb[42]
        results = idx.hierarchical_softmax(hidden)

        top_ids = [r.token_id for r in results[:5]]
        assert 42 in top_ids, f"Token 42 not in top 5: {top_ids}"


class TestGradientSharing:
    def test_centroid_gradient_is_mean(self) -> None:
        emb = _synthetic_embeddings(100, 16)
        idx = TokenHashIndex(n_clusters=4)
        idx.build(emb)

        # Fake gradients for tokens in cluster 0
        members = idx.cluster_members(0)[:5]
        rng = np.random.RandomState(7)
        grads = {tid: rng.randn(16).astype(np.float32) for tid in members}

        centroid_grad, offsets = idx.compute_shared_gradient(0, grads)

        expected_mean = np.stack(list(grads.values())).mean(axis=0)
        np.testing.assert_allclose(centroid_grad, expected_mean, atol=1e-5)

    def test_offsets_are_residuals(self) -> None:
        emb = _synthetic_embeddings(100, 16)
        idx = TokenHashIndex(n_clusters=4)
        idx.build(emb)

        members = idx.cluster_members(0)[:3]
        grads = {tid: np.ones(16, dtype=np.float32) * (tid + 1) for tid in members}

        centroid_grad, offsets = idx.compute_shared_gradient(0, grads)

        for tid, offset in offsets.items():
            reconstructed = centroid_grad + offset
            np.testing.assert_allclose(
                reconstructed, grads[tid], atol=1e-5,
            )


class TestSparseUpdates:
    def test_savings_ratio(self) -> None:
        emb = _synthetic_embeddings(10000, 64)
        idx = TokenHashIndex(n_clusters=100)
        idx.build(emb)

        # Batch of 128 tokens from ~30 unique IDs
        batch = list(range(30)) * 4 + list(range(30, 38))
        savings = idx.compute_update_savings(batch)

        assert savings["unique_tokens"] == 38
        assert savings["active_clusters"] <= 38
        assert savings["savings_ratio"] > 0.95  # should save >95%
        assert savings["speedup_factor"] > 10


class TestPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        emb = _synthetic_embeddings(200, 16)
        idx = TokenHashIndex(n_clusters=8)
        idx.build(emb)

        idx.save(tmp_path / "index")
        loaded = TokenHashIndex.load(tmp_path / "index")

        assert loaded.is_built
        assert loaded.vocab_size == 200
        assert loaded.dim == 16

        for tid in range(200):
            assert loaded.token_hash(tid) == idx.token_hash(tid)
            np.testing.assert_allclose(
                loaded.reconstruct(tid), idx.reconstruct(tid), atol=1e-6,
            )


class TestIncrementalExport:
    def test_export_weight_clusters_matches_membership(self) -> None:
        emb = _synthetic_embeddings(120, 12)
        idx = TokenHashIndex(n_clusters=6)
        idx.build(emb)

        clusters = idx.export_weight_clusters()
        exported_ids = sorted(int(cluster.cluster_id) for cluster in clusters if cluster.cluster_id is not None)
        assert exported_ids == sorted(range(idx.n_clusters))
        assert sum(len(cluster.member_token_ids) for cluster in clusters) == 120

    def test_incremental_export_tracks_changed_tokens_and_clusters(self) -> None:
        emb = _synthetic_embeddings(90, 10)
        idx = TokenHashIndex(n_clusters=5)
        idx.build(emb)
        idx.mark_export_snapshot(emb)

        updated = emb.copy()
        updated[3] += 0.5
        updated[17] -= 0.25

        result = idx.export_incremental_clusters(updated)

        assert result.changed_token_ids == [3, 17]
        assert len(result.changed_cluster_ids) >= 1
        exported_members = {
            token_id
            for cluster in result.clusters
            for token_id in cluster.member_token_ids
        }
        assert 3 in exported_members
        assert 17 in exported_members
        assert result.delta_savings_ratio >= 0.0
        assert len(result.deltas) == len(result.changed_cluster_ids)

    def test_incremental_export_persists_changed_clusters_and_deltas(self, tmp_path: Path) -> None:
        emb = _synthetic_embeddings(90, 10)
        idx = TokenHashIndex(n_clusters=5)
        idx.build(emb)
        idx.mark_export_snapshot(emb)

        updated = emb.copy()
        updated[3] += 0.5
        updated[17] -= 0.25

        artifact_store = ArtifactStore(
            BlobStore(str(tmp_path / "blobs")),
            tmp_path / "manifests",
        )
        manifest = idx.store_incremental_export(
            artifact_store,
            "origin:test",
            updated,
            export_name="delta-run",
        )

        assert manifest.manifest_artifact is not None
        assert manifest.changed_token_ids == [3, 17]
        assert len(manifest.cluster_artifacts) == len(manifest.changed_cluster_ids)
        assert len(manifest.delta_artifacts) == len(manifest.changed_cluster_ids)

        stored_manifest = artifact_store.load_json(manifest.manifest_artifact.blob_id)
        assert stored_manifest["export_name"] == "delta-run"
        assert stored_manifest["changed_token_ids"] == [3, 17]
        assert sorted(stored_manifest["changed_cluster_ids"]) == manifest.changed_cluster_ids
        assert len(stored_manifest["cluster_artifacts"]) == len(manifest.changed_cluster_ids)
        assert len(stored_manifest["delta_artifacts"]) == len(manifest.changed_cluster_ids)
        assert set(stored_manifest["changed_cluster_ids"]).isdisjoint(
            stored_manifest["unchanged_cluster_ids"]
        )

        first_cluster = artifact_store.load_json(manifest.cluster_artifacts[0].blob_id)
        first_delta = artifact_store.load_json(manifest.delta_artifacts[0].blob_id)
        assert first_cluster["cluster_id"] in {str(cluster_id) for cluster_id in manifest.changed_cluster_ids}
        assert first_delta["cluster_id"] in manifest.changed_cluster_ids

    def test_adapt_rebalances_existing_index(self) -> None:
        rng = np.random.RandomState(7)
        giant = rng.normal(0.0, 0.08, size=(600, 8))
        tiny_a = rng.normal(5.0, 0.08, size=(20, 8))
        tiny_b = rng.normal(-5.0, 0.08, size=(20, 8))
        emb = np.vstack([giant, tiny_a, tiny_b]).astype(np.float32)

        idx = TokenHashIndex(n_clusters=3)
        idx.build(emb, adaptive=False, max_iter=15)
        stats = idx.adapt(
            emb,
            max_cluster_ratio=2.0,
            max_rebalance_steps=4,
        )

        assert stats["adaptive_rebalanced"] is True
        assert stats["rebalance_steps"] >= 1
