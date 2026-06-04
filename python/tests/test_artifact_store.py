import pytest
import tempfile
import numpy as np
from pathlib import Path

from sodl_weights import BlobStore
from sodl_weights.artifact_store import ArtifactStore, ArtifactMetadata


@pytest.fixture
def store_setup():
    """Create a temporary ArtifactStore."""
    tmpdir = tempfile.mkdtemp()
    blob_store = BlobStore(tmpdir + "/blobs")
    artifact_store = ArtifactStore(blob_store, tmpdir + "/manifests")
    return artifact_store, tmpdir


class TestArtifactStoreBytes:
    def test_store_and_load_bytes(self, store_setup):
        store, _ = store_setup
        data = b"Hello, SODL artifacts!"
        meta = store.store("test-origin", data, "greeting")
        assert meta.artifact_type == "bytes"
        assert meta.size_raw == len(data)
        assert meta.size_stored > 0
        
        loaded = store.load(meta.blob_id)
        assert loaded == data

    def test_deduplication(self, store_setup):
        store, _ = store_setup
        data = b"duplicate content"
        meta1 = store.store("origin-1", data, "copy-1")
        meta2 = store.store("origin-1", data, "copy-2")
        assert meta1.blob_id == meta2.blob_id  # same content = same blob


class TestArtifactStoreNumpy:
    def test_store_and_load_numpy(self, store_setup):
        store, _ = store_setup
        arr = np.random.randn(100, 64).astype(np.float32)
        meta = store.store_numpy("model-1", arr, "embeddings")
        assert meta.artifact_type == "numpy"
        assert meta.shape == [100, 64]
        assert meta.dtype == "float32"
        
        loaded = store.load_numpy(meta.blob_id)
        np.testing.assert_array_almost_equal(arr, loaded)

    def test_store_integer_array(self, store_setup):
        store, _ = store_setup
        arr = np.arange(1000, dtype=np.int32)
        meta = store.store_numpy("model-1", arr, "token_ids")
        loaded = store.load_numpy(meta.blob_id)
        np.testing.assert_array_equal(arr, loaded)


class TestArtifactStoreJSON:
    def test_store_and_load_json(self, store_setup):
        store, _ = store_setup
        obj = {"model": "carla", "version": 2, "metrics": [0.1, 0.2, 0.3]}
        meta = store.store_json("config-origin", obj, "config")
        assert meta.artifact_type == "json"
        
        loaded = store.load_json(meta.blob_id)
        assert loaded == obj

    def test_store_nested_json(self, store_setup):
        store, _ = store_setup
        obj = {"a": {"b": {"c": [1, 2, 3]}}, "d": None}
        meta = store.store_json("test", obj, "nested")
        loaded = store.load_json(meta.blob_id)
        assert loaded == obj


class TestArtifactListing:
    def test_list_artifacts(self, store_setup):
        store, _ = store_setup
        store.store("origin-1", b"data1", "first")
        store.store("origin-1", b"data2", "second")
        store.store_json("origin-1", {"x": 1}, "config")
        
        all_arts = store.list_artifacts("origin-1")
        assert len(all_arts) == 3

    def test_list_by_type(self, store_setup):
        store, _ = store_setup
        store.store("origin-1", b"data", "raw")
        store.store_json("origin-1", {"x": 1}, "config")
        
        json_arts = store.list_artifacts("origin-1", artifact_type="json")
        assert len(json_arts) == 1
        assert json_arts[0].artifact_type == "json"

    def test_delete_artifact(self, store_setup):
        store, _ = store_setup
        meta = store.store("origin-1", b"data", "deleteme")
        assert store.delete_artifact("origin-1", meta.artifact_id)
        assert len(store.list_artifacts("origin-1")) == 0

    def test_tags(self, store_setup):
        store, _ = store_setup
        meta = store.store("origin-1", b"data", "tagged",
                          tags={"epoch": "5", "phase": "sft"})
        assert meta.tags["epoch"] == "5"
        assert meta.tags["phase"] == "sft"

    def test_find_artifacts_by_tag(self, store_setup):
        store, _ = store_setup
        store.store("origin-1", b"data", "raw-a", tags={"phase": "train"})
        store.store_json("origin-2", {"x": 1}, "cfg", tags={"phase": "eval"})
        results = store.find_artifacts(tags={"phase": "train"})
        assert len(results) == 1
        assert results[0].name == "raw-a"

    def test_artifact_stats(self, store_setup):
        store, _ = store_setup
        store.store("origin-1", b"abc", "raw")
        store.store_json("origin-1", {"x": 1}, "cfg")
        stats = store.artifact_stats("origin-1")
        assert stats["count"] == 2
        assert set(stats["by_type"]) == {"bytes", "json"}

    def test_enforce_retention_keeps_latest(self, store_setup):
        store, tmpdir = store_setup
        first = store.store("origin-1", b"one", "first")
        second = store.store("origin-1", b"two", "second")
        removed = store.enforce_retention("origin-1", keep_last=1, delete_unreferenced_blobs=True)
        assert len(removed) == 1
        assert removed[0].artifact_id == first.artifact_id
        assert len(store.list_artifacts("origin-1")) == 1
        blob_path = Path(tmpdir) / "blobs" / f"{first.blob_id.split(':', 1)[1]}.blob"
        assert not blob_path.exists()
