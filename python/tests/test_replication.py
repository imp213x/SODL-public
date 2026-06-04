import pytest
import tempfile

from sodl_weights import BlobStore
from sodl_weights.store import compute_blob_id
from sodl_weights.registry import NodeInfo, NodeRegistry
from sodl_weights.replication import ReplicationEngine, ReplicationPolicy, ReplicationStatus


@pytest.fixture
def repl_setup():
    tmpdir = tempfile.mkdtemp()
    store = BlobStore(tmpdir)
    registry = NodeRegistry()
    # Register some local-only nodes (no real HTTP)
    registry.register(NodeInfo(id="local", url="http://localhost:9999"))
    policy = ReplicationPolicy(min_replicas=1, max_replicas=3)
    engine = ReplicationEngine(store, registry, policy)
    return engine, store, registry


class TestReplicationPolicy:
    def test_defaults(self):
        policy = ReplicationPolicy()
        assert policy.min_replicas == 2
        assert policy.replicate_on_read is True

    def test_custom(self):
        policy = ReplicationPolicy(min_replicas=3, prefer_regions=["us-east"])
        assert policy.min_replicas == 3
        assert "us-east" in policy.prefer_regions


class TestReplicationEngine:
    def test_record_replica(self, repl_setup):
        engine, _, _ = repl_setup
        engine.record_replica("blake3:abc123", "node-1")
        status = engine.check_replication("blake3:abc123")
        assert status.replica_count == 1
        assert "node-1" in status.replica_nodes

    def test_check_unknown_blob(self, repl_setup):
        engine, _, _ = repl_setup
        status = engine.check_replication("blake3:unknown")
        assert status.replica_count == 0
        assert not status.meets_policy

    def test_meets_policy(self, repl_setup):
        engine, _, _ = repl_setup
        engine.record_replica("blake3:abc", "n1")
        status = engine.check_replication("blake3:abc")
        assert status.meets_policy  # min_replicas=1

    def test_replicate_batch(self, repl_setup):
        engine, store, _ = repl_setup
        # Store some blobs
        blob_ids = []
        for i in range(3):
            data = f"data-{i}".encode()
            bid = compute_blob_id(data)
            store.put(bid, data)
            blob_ids.append(bid)
            engine.record_replica(bid, "local")
        
        results = engine.replicate_batch(blob_ids)
        assert len(results) == 3

    def test_stats(self, repl_setup):
        engine, _, _ = repl_setup
        assert engine.stats["replicated"] == 0
        assert engine.stats["failed"] == 0
