import pytest
import tempfile
import time

from sodl_weights import BlobStore
from sodl_weights.async_store import AsyncBlobStore
from sodl_weights.store import compute_blob_id


@pytest.fixture
def async_setup():
    tmpdir = tempfile.mkdtemp()
    blob_store = BlobStore(tmpdir)
    async_store = AsyncBlobStore(blob_store, max_workers=2)
    return async_store, tmpdir


class TestAsyncBlobStore:
    def test_async_put_get(self, async_setup):
        store, _ = async_setup
        data = b"async hello world"
        blob_id = compute_blob_id(data)
        
        f = store.async_put(blob_id, data)
        f.result()
        
        f2 = store.async_get(blob_id)
        result = f2.result()
        assert result == data

    def test_async_has(self, async_setup):
        store, _ = async_setup
        data = b"check me"
        blob_id = compute_blob_id(data)
        store.store.put(blob_id, data)
        
        f = store.async_has(blob_id)
        assert f.result() is True

    def test_batch_put_get(self, async_setup):
        store, _ = async_setup
        items = []
        for i in range(10):
            data = f"batch-item-{i}".encode()
            blob_id = compute_blob_id(data)
            items.append((blob_id, data))
        
        put_futures = store.async_put_batch(items)
        store.wait_all(put_futures)
        
        blob_ids = [bid for bid, _ in items]
        get_futures = store.async_get_batch(blob_ids)
        results = store.collect_results(get_futures)
        
        for i, result in enumerate(results):
            assert result == items[i][1]

    def test_stats_tracking(self, async_setup):
        store, _ = async_setup
        data = b"stats test"
        blob_id = compute_blob_id(data)
        
        store.async_put(blob_id, data).result()
        store.async_get(blob_id).result()
        
        assert store.stats.puts == 1
        assert store.stats.gets == 1
        assert store.stats.total_bytes_put == len(data)

    def test_shutdown(self, async_setup):
        store, _ = async_setup
        store.shutdown()
