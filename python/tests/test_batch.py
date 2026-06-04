import pytest
import tempfile
import numpy as np

from sodl_weights import BlobStore
from sodl_weights.batch import BatchOps, BatchResult


@pytest.fixture
def batch_setup():
    tmpdir = tempfile.mkdtemp()
    blob_store = BlobStore(tmpdir)
    ops = BatchOps(blob_store, max_workers=2)
    return ops, tmpdir


class TestBatchOps:
    def test_batch_store_bytes(self, batch_setup):
        ops, _ = batch_setup
        items = [f"item-{i}".encode() * 100 for i in range(5)]
        result = ops.batch_store(items)
        assert result.n_items == 5
        assert len(result.blob_ids) == 5
        assert result.total_raw_bytes > 0
        assert result.compression_ratio > 0
        assert result.throughput_mb_sec > 0

    def test_batch_load(self, batch_setup):
        ops, _ = batch_setup
        items = [f"batch-load-{i}".encode() * 50 for i in range(3)]
        result = ops.batch_store(items)
        
        loaded = ops.batch_load(result.blob_ids)
        assert len(loaded) == 3
        for i, data in enumerate(loaded):
            assert data == items[i]

    def test_batch_store_numpy(self, batch_setup):
        ops, _ = batch_setup
        arrays = [np.random.randn(100, 64).astype(np.float32) for _ in range(4)]
        result = ops.batch_store_numpy(arrays)
        assert result.n_items == 4
        assert result.compression_ratio > 0

    def test_batch_load_numpy(self, batch_setup):
        ops, _ = batch_setup
        arrays = [np.random.randn(50, 32).astype(np.float32) for _ in range(3)]
        result = ops.batch_store_numpy(arrays)
        
        loaded = ops.batch_load_numpy(result.blob_ids)
        assert len(loaded) == 3
        for i, arr in enumerate(loaded):
            np.testing.assert_array_almost_equal(arr, arrays[i])

    def test_empty_batch(self, batch_setup):
        ops, _ = batch_setup
        result = ops.batch_store([])
        assert result.n_items == 0
        assert len(result.blob_ids) == 0

    def test_result_properties(self, batch_setup):
        ops, _ = batch_setup
        data = [b"x" * 10000 for _ in range(5)]
        result = ops.batch_store(data)
        assert 0 < result.compression_ratio < 1
        assert result.elapsed_sec > 0
        assert len(result.errors) == 0
