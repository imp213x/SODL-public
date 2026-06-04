import pytest
import time

from sodl_weights.registry import NodeInfo, NodeRegistry, NodeRole, NodeStatus


class TestNodeInfo:
    def test_defaults(self):
        node = NodeInfo(id="n1", url="http://host:8080")
        assert node.role == NodeRole.REPLICA
        assert node.status == NodeStatus.HEALTHY
        assert node.available_gb == 100.0

    def test_utilization(self):
        node = NodeInfo(id="n1", url="http://host:8080", capacity_gb=100, used_gb=75)
        assert node.utilization == 0.75
        assert node.available_gb == 25.0

    def test_zero_capacity(self):
        node = NodeInfo(id="n1", url="http://host:8080", capacity_gb=0)
        assert node.utilization == 1.0


class TestNodeRegistry:
    @pytest.fixture
    def registry(self):
        return NodeRegistry(heartbeat_timeout_sec=2.0)

    def test_register_and_get(self, registry):
        node = NodeInfo(id="n1", url="http://host:8080")
        registry.register(node)
        assert registry.get("n1") is not None
        assert registry.node_count == 1

    def test_deregister(self, registry):
        registry.register(NodeInfo(id="n1", url="http://host:8080"))
        removed = registry.deregister("n1")
        assert removed is not None
        assert registry.node_count == 0

    def test_heartbeat(self, registry):
        registry.register(NodeInfo(id="n1", url="http://host:8080"))
        assert registry.heartbeat("n1")
        assert not registry.heartbeat("nonexistent")

    def test_heartbeat_updates_usage(self, registry):
        registry.register(NodeInfo(id="n1", url="http://host:8080"))
        registry.heartbeat("n1", used_gb=50.0)
        assert registry.get("n1").used_gb == 50.0

    def test_healthy_nodes(self, registry):
        registry.register(NodeInfo(id="n1", url="http://h1:8080", role=NodeRole.PRIMARY))
        registry.register(NodeInfo(id="n2", url="http://h2:8080", role=NodeRole.REPLICA))
        healthy = registry.healthy_nodes()
        assert len(healthy) == 2

    def test_healthy_nodes_by_role(self, registry):
        registry.register(NodeInfo(id="n1", url="http://h1:8080", role=NodeRole.PRIMARY))
        registry.register(NodeInfo(id="n2", url="http://h2:8080", role=NodeRole.REPLICA))
        primaries = registry.healthy_nodes(role=NodeRole.PRIMARY)
        assert len(primaries) == 1
        assert primaries[0].id == "n1"

    def test_unreachable_after_timeout(self, registry):
        node = NodeInfo(id="n1", url="http://host:8080")
        node.last_heartbeat = time.time() - 10  # 10 seconds ago, timeout is 2
        registry._nodes["n1"] = node
        healthy = registry.healthy_nodes()
        assert len(healthy) == 0

    def test_nodes_by_region(self, registry):
        registry.register(NodeInfo(id="n1", url="http://h1:8080", region="us-east"))
        registry.register(NodeInfo(id="n2", url="http://h2:8080", region="eu-west"))
        us = registry.nodes_by_region("us-east")
        assert len(us) == 1

    def test_least_loaded(self, registry):
        registry.register(NodeInfo(id="n1", url="http://h1:8080", used_gb=80))
        registry.register(NodeInfo(id="n2", url="http://h2:8080", used_gb=20))
        least = registry.least_loaded()
        assert least.id == "n2"

    def test_persistence(self, tmp_path):
        path = tmp_path / "registry.json"
        reg1 = NodeRegistry(persist_path=str(path))
        reg1.register(NodeInfo(id="n1", url="http://h1:8080"))
        assert path.exists()

        reg2 = NodeRegistry(persist_path=str(path))
        assert reg2.node_count == 1
        assert reg2.get("n1").url == "http://h1:8080"
