from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

from sodl_weights.store import BlobStore
from sodl_weights.training_lifecycle import (
    ArteryPulsar,
    VeinPrefetcher,
    resolve_sodl_origin_id,
    write_sodl_manifest,
)


class MockWeightService:
    def __init__(self, blob_root: str | Path | None = None):
        self._clusters = {}
        self.load_count = 0
        self.store_count = 0
        self.prefetch_batch_calls: list[list[str]] = []
        self.pin_registry_saved = 0
        self.blob_root = Path(blob_root) if blob_root is not None else None

    def load_cluster(self, origin_id, cluster_id):
        self.load_count += 1
        return self._clusters.get(cluster_id, {"id": cluster_id, "data": [1, 2, 3]})

    def store_cluster(self, origin_id, cluster):
        self.store_count += 1
        return MagicMock(stored_bytes=100, blob_id=f"blob-{self.store_count}")

    def prefetch_clusters(self, origin_id, blob_ids):
        self.prefetch_batch_calls.append(list(blob_ids))
        return len(blob_ids)

    def save_pin_registry(self):
        self.pin_registry_saved += 1


def test_sdk_prefetcher_uses_batch_prefetch_when_available() -> None:
    service = MockWeightService()
    prefetcher = VeinPrefetcher(
        service,
        "sdk-origin",
        cluster_blob_ids={1: "blob-a", 2: "blob-b"},
    )
    prefetcher.start()
    prefetcher.request_prefetch({1, 2})
    time.sleep(1.0)
    prefetcher.stop()

    assert service.prefetch_batch_calls
    assert sorted(service.prefetch_batch_calls[0]) == ["blob-a", "blob-b"]
    assert prefetcher.stats["prefetch_requests"] >= 1


def test_sdk_pulsar_flush_updates_manifest_and_persists_pin_registry(tmp_path: Path) -> None:
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)
    blob_a = "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    blob_b = "blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    blob_store.put(blob_a, b"a")
    blob_store.put(blob_b, b"b")

    service = MockWeightService(blob_root=blob_root)
    token_index = MagicMock()
    token_index.vocab_size = 2
    token_index.dim = 4
    token_index.cluster_members.side_effect = lambda cluster_id: [int(cluster_id)]
    manifest_path = tmp_path / "sodl_manifest.json"

    write_sodl_manifest(
        manifest_path,
        "origin:stable",
        token_index,
        {0: blob_a, 1: blob_b},
        metadata={"checkpoint_origin": "carla-large"},
        weight_service=service,
    )

    pulsar = ArteryPulsar(
        service,
        "origin:stable",
        min_dirty_clusters=1,
        cluster_blob_ids={0: blob_a, 1: blob_b},
        manifest_path=manifest_path,
        token_index=token_index,
    )
    pulsar.mark_dirty(1, {"id": 1})
    pulsar.flush()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cluster_map = {str(item["cluster_id"]): item["blob_id"] for item in payload["clusters"]}
    assert cluster_map["1"] == "blob-1"
    assert payload["metadata"]["checkpoint_origin"] == "carla-large"
    assert service.pin_registry_saved == 1


def test_sdk_resolve_origin_prefers_resume_metadata(tmp_path: Path) -> None:
    manifest_path = tmp_path / "sodl_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "origin_id": "origin:manifest",
                "clusters": [],
                "metadata": {"checkpoint_origin": "carla-large"},
            }
        ),
        encoding="utf-8",
    )

    origin_id, source = resolve_sodl_origin_id(
        manifest_path,
        checkpoint_origin="carla-large",
        resume_record={"metadata": {"sodl_origin_id": "origin:resume"}},
    )

    assert origin_id == "origin:resume"
    assert source == "resume_record"
