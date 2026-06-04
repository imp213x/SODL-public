import pytest
import tempfile
import numpy as np

from sodl_weights import BlobStore
from sodl_weights.dataset import SODLDataset


@pytest.fixture
def dataset_setup():
    tmpdir = tempfile.mkdtemp()
    blob_store = BlobStore(tmpdir + "/blobs")
    return blob_store, tmpdir


class TestSODLDataset:
    def test_from_numpy_single_shard(self, dataset_setup):
        blob_store, tmpdir = dataset_setup
        data = np.random.randn(50, 10).astype(np.float32)
        ds = SODLDataset.from_numpy([data], blob_store)
        assert len(ds) == 50
        assert ds.num_shards == 1
        sample = ds[0]
        np.testing.assert_array_almost_equal(sample, data[0])

    def test_from_numpy_multiple_shards(self, dataset_setup):
        blob_store, tmpdir = dataset_setup
        shard1 = np.random.randn(30, 10).astype(np.float32)
        shard2 = np.random.randn(20, 10).astype(np.float32)
        ds = SODLDataset.from_numpy([shard1, shard2], blob_store)
        assert len(ds) == 50
        assert ds.num_shards == 2
        # First shard samples
        np.testing.assert_array_almost_equal(ds[0], shard1[0])
        np.testing.assert_array_almost_equal(ds[29], shard1[29])
        # Second shard samples
        np.testing.assert_array_almost_equal(ds[30], shard2[0])
        np.testing.assert_array_almost_equal(ds[49], shard2[19])

    def test_negative_indexing(self, dataset_setup):
        blob_store, tmpdir = dataset_setup
        data = np.arange(100).reshape(100, 1).astype(np.float32)
        ds = SODLDataset.from_numpy([data], blob_store)
        np.testing.assert_array_almost_equal(ds[-1], data[-1])

    def test_out_of_range_raises(self, dataset_setup):
        blob_store, tmpdir = dataset_setup
        data = np.random.randn(10, 5).astype(np.float32)
        ds = SODLDataset.from_numpy([data], blob_store)
        with pytest.raises(IndexError):
            ds[10]

    def test_manifest_roundtrip(self, dataset_setup):
        blob_store, tmpdir = dataset_setup
        data = np.random.randn(25, 8).astype(np.float32)
        manifest_path = tmpdir + "/manifest.json"
        ds1 = SODLDataset.from_numpy([data], blob_store, manifest_path=manifest_path)
        
        # Reload from manifest
        ds2 = SODLDataset.from_manifest(manifest_path, tmpdir + "/blobs")
        assert len(ds2) == 25
        np.testing.assert_array_almost_equal(ds2[0], ds1[0])

    def test_cache_clear(self, dataset_setup):
        blob_store, tmpdir = dataset_setup
        data = np.random.randn(10, 4).astype(np.float32)
        ds = SODLDataset.from_numpy([data], blob_store)
        ds[0]  # trigger cache
        ds.clear_cache()
        ds[0]  # should reload

    def test_transform(self, dataset_setup):
        blob_store, tmpdir = dataset_setup
        data = np.ones((5, 3), dtype=np.float32)
        ds = SODLDataset.from_numpy([data], blob_store, transform=lambda x: x * 2)
        sample = ds[0]
        np.testing.assert_array_almost_equal(sample, np.array([2.0, 2.0, 2.0]))

    def test_prefetch_shards(self, dataset_setup):
        blob_store, _ = dataset_setup
        shard1 = np.random.randn(10, 2).astype(np.float32)
        shard2 = np.random.randn(10, 2).astype(np.float32)
        ds = SODLDataset.from_numpy([shard1, shard2], blob_store, cache_capacity=2)
        ds.prefetch_shards([0, 1])
        assert len(ds._cache) == 2

    def test_lru_cache_capacity(self, dataset_setup):
        blob_store, _ = dataset_setup
        shards = [np.random.randn(4, 2).astype(np.float32) for _ in range(3)]
        ds = SODLDataset.from_numpy(shards, blob_store, cache_capacity=1)
        ds[0]
        ds[4]
        assert len(ds._cache) == 1

    def test_worker_shard_assignment(self, dataset_setup):
        blob_store, _ = dataset_setup
        shards = [np.random.randn(4, 2).astype(np.float32) for _ in range(5)]
        ds = SODLDataset.from_numpy(shards, blob_store)
        worker_zero = ds.shard_ids_for_worker(worker_id=0, num_workers=2)
        worker_one = ds.shard_ids_for_worker(worker_id=1, num_workers=2)
        assert set(worker_zero).isdisjoint(worker_one)
        assert sorted(worker_zero + worker_one) == [0, 1, 2, 3, 4]
