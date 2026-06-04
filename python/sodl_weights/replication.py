"""Replication Engine — blob replication across SODL nodes.

Manages replication policies, tracks replica placement, and provides
background workers for syncing blobs between nodes.

Example
-------
>>> policy = ReplicationPolicy(min_replicas=3, prefer_regions=["us-east", "eu-west"])
>>> engine = ReplicationEngine(local_store, registry, policy)
>>> engine.replicate_blob(blob_id)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen, Request

from sodl_weights.store import BlobStore
from sodl_weights.registry import NodeInfo, NodeRegistry, NodeStatus

logger = logging.getLogger(__name__)


@dataclass
class ReplicationPolicy:
    """Policy governing how blobs are replicated.

    Parameters
    ----------
    min_replicas : int
        Minimum number of replicas per blob (including local).
    max_replicas : int
        Maximum replicas (0 = unlimited).
    prefer_regions : list of str
        Preferred regions for placement.
    exclude_regions : list of str
        Regions to avoid.
    replicate_on_read : bool
        Whether to create a local replica when reading from remote.
    """
    min_replicas: int = 2
    max_replicas: int = 5
    prefer_regions: list[str] = field(default_factory=list)
    exclude_regions: list[str] = field(default_factory=list)
    replicate_on_read: bool = True


@dataclass
class ReplicationStatus:
    """Status of a blob's replication."""
    blob_id: str
    replica_count: int = 0
    replica_nodes: list[str] = field(default_factory=list)
    meets_policy: bool = False
    last_checked: float = 0.0


class ReplicationEngine:
    """Manages blob replication across federated SODL nodes.

    Parameters
    ----------
    local_store : BlobStore
        The local blob store.
    registry : NodeRegistry
        Registry of available nodes.
    policy : ReplicationPolicy
        Replication rules.
    max_workers : int
        Thread pool size for background replication.
    """

    def __init__(
        self,
        local_store: BlobStore,
        registry: NodeRegistry,
        policy: Optional[ReplicationPolicy] = None,
        max_workers: int = 4,
    ) -> None:
        self._store = local_store
        self._registry = registry
        self._policy = policy or ReplicationPolicy()
        self._max_workers = max_workers
        self._replica_map: dict[str, set[str]] = {}
        self._stats = {"replicated": 0, "failed": 0, "bytes_transferred": 0}

    def check_replication(self, blob_id: str) -> ReplicationStatus:
        """Check the replication status of a blob."""
        nodes = self._replica_map.get(blob_id, set())
        status = ReplicationStatus(
            blob_id=blob_id,
            replica_count=len(nodes),
            replica_nodes=list(nodes),
            meets_policy=len(nodes) >= self._policy.min_replicas,
            last_checked=time.time(),
        )
        return status

    def replicate_blob(self, blob_id: str) -> ReplicationStatus:
        """Ensure a blob meets the replication policy.

        Reads the blob from local store and pushes to enough healthy
        nodes to satisfy min_replicas.

        Returns
        -------
        ReplicationStatus
            Updated status after replication.
        """
        current = self._replica_map.get(blob_id, set())
        needed = self._policy.min_replicas - len(current)

        if needed <= 0:
            return self.check_replication(blob_id)

        # Get target nodes
        targets = self._select_targets(blob_id, needed, exclude=current)
        if not targets:
            logger.warning(f"No available targets for replicating {blob_id}")
            return self.check_replication(blob_id)

        # Read blob data
        try:
            data = self._store.get(blob_id)
        except Exception as e:
            logger.error(f"Failed to read {blob_id} for replication: {e}")
            return self.check_replication(blob_id)

        # Push to targets
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self._push_to_node, blob_id, data, node): node
                for node in targets
            }
            for future in as_completed(futures):
                node = futures[future]
                try:
                    future.result()
                    current.add(node.id)
                    self._stats["replicated"] += 1
                    self._stats["bytes_transferred"] += len(data)
                except Exception as e:
                    self._stats["failed"] += 1
                    logger.error(f"Replication to {node.id} failed: {e}")

        self._replica_map[blob_id] = current
        return self.check_replication(blob_id)

    def replicate_batch(self, blob_ids: Sequence[str]) -> list[ReplicationStatus]:
        """Replicate multiple blobs."""
        return [self.replicate_blob(bid) for bid in blob_ids]

    def record_replica(self, blob_id: str, node_id: str) -> None:
        """Record that a replica exists on a node (e.g., after a read)."""
        self._replica_map.setdefault(blob_id, set()).add(node_id)

    def _select_targets(
        self, blob_id: str, count: int, exclude: set[str]
    ) -> list[NodeInfo]:
        """Select target nodes for replication based on policy."""
        candidates = self._registry.healthy_nodes()

        # Filter out excluded and this node
        candidates = [n for n in candidates if n.id not in exclude]

        # Filter excluded regions
        if self._policy.exclude_regions:
            candidates = [
                n for n in candidates
                if n.region not in self._policy.exclude_regions
            ]

        # Sort: prefer specified regions, then least loaded
        def sort_key(node: NodeInfo) -> tuple[int, float]:
            region_pref = 0 if node.region in self._policy.prefer_regions else 1
            return (region_pref, node.utilization)

        candidates.sort(key=sort_key)
        return candidates[:count]

    def _push_to_node(self, blob_id: str, data: bytes, node: NodeInfo) -> None:
        """Push blob data to a remote node via HTTP PUT."""
        url = f"{node.url.rstrip('/')}/v1/blobs/{quote(blob_id, safe=':')}"
        req = Request(url, data=data, method="PUT")
        req.add_header("Content-Type", "application/octet-stream")
        try:
            with urlopen(req, timeout=30) as resp:
                if resp.status not in (200, 201, 204):
                    raise RuntimeError(f"PUT failed: HTTP {resp.status}")
        except (URLError, HTTPError) as e:
            raise RuntimeError(f"Push to {node.id} failed: {e}") from e

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    @property
    def replica_map(self) -> dict[str, set[str]]:
        return dict(self._replica_map)
