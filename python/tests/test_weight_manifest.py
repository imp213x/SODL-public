from __future__ import annotations

import tempfile

from sodl_weights.store import BlobStore
from sodl_weights.weight_manifest import WeightManifestCluster, WeightManifestStore


def test_weight_manifest_roundtrip_and_blob_prune() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        blob_store = BlobStore(f"{tmpdir}/blobs")
        stale_blob = "blake3:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
        live_blob = "blake3:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        blob_store.put(stale_blob, b"stale")
        blob_store.put(live_blob, b"live")

        store = WeightManifestStore(
            f"{tmpdir}/run/sodl_manifest.json",
            blob_root=f"{tmpdir}/blobs",
        )
        store.write_manifest(
            "origin:stable",
            2,
            4,
            [WeightManifestCluster(cluster_id=0, blob_id=stale_blob, member_token_ids=[0])],
            metadata={"checkpoint_origin": "carla-large", "note": "keep"},
        )
        updated = store.write_manifest(
            "origin:stable",
            2,
            4,
            [WeightManifestCluster(cluster_id=0, blob_id=live_blob, member_token_ids=[0])],
        )

        assert updated.metadata["note"] == "keep"
        assert not blob_store.has(stale_blob)
        assert blob_store.has(live_blob)


def test_weight_manifest_resolve_origin_prefers_resume_then_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = WeightManifestStore(f"{tmpdir}/sodl_manifest.json")
        store.write_manifest(
            "origin:manifest",
            1,
            1,
            [],
            metadata={"checkpoint_origin": "carla-large"},
        )

        resolved = store.resolve_origin_id(
            "carla-large",
            resume_record={"metadata": {"sodl_origin_id": "origin:resume"}},
        )
        assert resolved == ("origin:resume", "resume_record")

        resolved = store.resolve_origin_id("carla-large")
        assert resolved == ("origin:manifest", "manifest")

        resolved = store.resolve_origin_id("carla-medium")
        assert resolved == (None, "new")
