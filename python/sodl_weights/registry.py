"""Node Registry — service discovery and health tracking for SODL federation.

Manages a registry of SODL storage nodes with health checking,
role assignment, and metadata.

Example
-------
>>> registry = NodeRegistry()
>>> registry.register(NodeInfo(id="node-1", url="http://host:8080", role="primary"))
>>> healthy = registry.healthy_nodes()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence
from urllib.error import URLError
from urllib.request import urlopen, Request


class NodeRole(str, Enum):
    PRIMARY = "primary"
    REPLICA = "replica"
    EDGE = "edge"
    OBSERVER = "observer"


class NodeStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNREACHABLE = "unreachable"
    DRAINING = "draining"


@dataclass
class NodeInfo:
    """Information about a SODL storage node.

    Parameters
    ----------
    id : str
        Unique node identifier.
    url : str
        Base URL for the node's blob API.
    role : str
        Node role (primary, replica, edge, observer).
    region : str
        Geographic region or zone.
    capacity_gb : float
        Total storage capacity in GB.
    used_gb : float
        Used storage in GB.
    """
    id: str
    url: str
    role: str = NodeRole.REPLICA
    region: str = "default"
    capacity_gb: float = 100.0
    used_gb: float = 0.0
    status: str = NodeStatus.HEALTHY
    last_heartbeat: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def available_gb(self) -> float:
        return max(0, self.capacity_gb - self.used_gb)

    @property
    def utilization(self) -> float:
        if self.capacity_gb == 0:
            return 1.0
        return self.used_gb / self.capacity_gb


class NodeRegistry:
    """Registry of SODL storage nodes with health tracking.

    Parameters
    ----------
    heartbeat_timeout_sec : float
        Seconds before a node is considered unreachable (default 60).
    persist_path : str or Path, optional
        Path to persist registry state.
    """

    def __init__(
        self,
        heartbeat_timeout_sec: float = 60.0,
        persist_path: Optional[str | Path] = None,
    ) -> None:
        self._nodes: dict[str, NodeInfo] = {}
        self._heartbeat_timeout = heartbeat_timeout_sec
        self._persist_path = Path(persist_path) if persist_path else None

        if self._persist_path and self._persist_path.exists():
            self._load()

    def register(self, node: NodeInfo) -> None:
        """Register or update a node."""
        node.last_heartbeat = time.time()
        self._nodes[node.id] = node
        self._save()

    def deregister(self, node_id: str) -> Optional[NodeInfo]:
        """Remove a node from the registry."""
        node = self._nodes.pop(node_id, None)
        self._save()
        return node

    def get(self, node_id: str) -> Optional[NodeInfo]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def heartbeat(self, node_id: str, used_gb: Optional[float] = None) -> bool:
        """Record a heartbeat from a node. Returns True if node is registered."""
        node = self._nodes.get(node_id)
        if node is None:
            return False
        node.last_heartbeat = time.time()
        node.status = NodeStatus.HEALTHY
        if used_gb is not None:
            node.used_gb = used_gb
        return True

    def healthy_nodes(self, role: Optional[str] = None) -> list[NodeInfo]:
        """List all healthy nodes, optionally filtered by role."""
        now = time.time()
        result = []
        for node in self._nodes.values():
            if now - node.last_heartbeat > self._heartbeat_timeout:
                node.status = NodeStatus.UNREACHABLE
            if node.status in (NodeStatus.HEALTHY, NodeStatus.DEGRADED):
                if role is None or node.role == role:
                    result.append(node)
        return result

    def all_nodes(self) -> list[NodeInfo]:
        """List all registered nodes."""
        return list(self._nodes.values())

    def nodes_by_region(self, region: str) -> list[NodeInfo]:
        """List nodes in a specific region."""
        return [n for n in self._nodes.values() if n.region == region]

    def least_loaded(self, role: Optional[str] = None) -> Optional[NodeInfo]:
        """Get the healthy node with the lowest utilization."""
        candidates = self.healthy_nodes(role)
        if not candidates:
            return None
        return min(candidates, key=lambda n: n.utilization)

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def _save(self) -> None:
        if self._persist_path is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {nid: asdict(node) for nid, node in self._nodes.items()}
        self._persist_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if self._persist_path is None or not self._persist_path.exists():
            return
        data = json.loads(self._persist_path.read_text(encoding="utf-8"))
        for nid, ndata in data.items():
            self._nodes[nid] = NodeInfo(**ndata)
