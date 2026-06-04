import pytest
import tempfile

from sodl_weights import BlobStore
from sodl_weights.store import compute_blob_id
from sodl_weights.registry import NodeInfo, NodeRegistry
from sodl_weights.federation import FederationManager, FederationConfig


@pytest.fixture
def fed_setup():
    tmpdir = tempfile.mkdtemp()
    store = BlobStore(tmpdir)
    registry = NodeRegistry()
    config = FederationConfig(write_quorum=1, read_quorum=1)
    fed = FederationManager(store, registry, config)
    return fed, store, registry


class TestFederationManager:
    def test_federated_put_local_only(self, fed_setup):
        fed, store, _ = fed_setup
        data = b"federation test"
        blob_id = compute_blob_id(data)
        
        # Quorum=1, so local write suffices
        assert fed.federated_put(blob_id, data)
        assert store.has(blob_id)

    def test_federated_get_local(self, fed_setup):
        fed, store, _ = fed_setup
        data = b"local read test"
        blob_id = compute_blob_id(data)
        store.put(blob_id, data)
        
        result = fed.federated_get(blob_id)
        assert result == data
        assert fed.stats.local_hits == 1

    def test_federated_get_missing(self, fed_setup):
        fed, _, _ = fed_setup
        result = fed.federated_get("blake3:nonexistent")
        assert result is None

    def test_federated_has(self, fed_setup):
        fed, store, _ = fed_setup
        data = b"existence check"
        blob_id = compute_blob_id(data)
        store.put(blob_id, data)
        assert fed.federated_has(blob_id)
        assert not fed.federated_has("blake3:nonexistent")

    def test_consistency_check(self, fed_setup):
        fed, store, _ = fed_setup
        blob_ids = []
        for i in range(3):
            data = f"consistency-{i}".encode()
            bid = compute_blob_id(data)
            store.put(bid, data)
            blob_ids.append(bid)
        
        report = fed.check_consistency(blob_ids)
        assert report.total_blobs == 3
        assert report.healthy == 3

    def test_stats_tracking(self, fed_setup):
        fed, store, _ = fed_setup
        data = b"stats test"
        blob_id = compute_blob_id(data)
        
        fed.federated_put(blob_id, data)
        fed.federated_get(blob_id)
        
        assert fed.stats.puts == 1
        assert fed.stats.gets == 1

    def test_replication_engine_access(self, fed_setup):
        fed, _, _ = fed_setup
        assert fed.replication_engine is not None
        assert fed.consistency_checker is not None
