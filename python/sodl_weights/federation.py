"""Federation Manager — coordinated multi-node blob operations.

Provides a high-level API for federated put/get with quorum semantics,
locality-aware routing, and cross-region blob management.

Example
-------
>>> fed = FederationManager(local_store, registry, policy)
>>> fed.federated_put(blob_id, data)  # writes to quorum of nodes
>>> data = fed.federated_get(blob_id)  # reads from nearest healthy node
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from typing import Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen, Request

from sodl_weights.store import BlobStore, compute_blob_id
from sodl_weights.registry import NodeInfo, NodeRegistry, NodeRole, NodeStatus
from sodl_weights.replication import ReplicationEngine, ReplicationPolicy
from sodl_weights.consistency import ConsistencyChecker, ConsistencyReport

logger = logging.getLogger(__name__)


@dataclass
class FederationConfig:
    """Configuration for federated operations.

    Parameters
    ----------
    write_quorum : int
        Minimum nodes that must acknowledge a write.
    read_quorum : int
        Minimum nodes to consult for a read.
    local_region : str
        Region of the local node (for locality).
    max_workers : int
        Thread pool size.
    timeout_sec : float
        Per-node operation timeout.
    """
    write_quorum: int = 2
    read_quorum: int = 1
    local_region: str = "default"
    max_workers: int = 4
    timeout_sec: float = 30.0


@dataclass
class FederationStats:
    """Statistics for federation operations."""
    puts: int = 0
    gets: int = 0
    quorum_failures: int = 0
    local_hits: int = 0
    remote_hits: int = 0
    bytes_transferred: int = 0


class FederationManager:
    """Coordinates multi-node SODL blob operations.

    Provides quorum-based writes, locality-aware reads, and
    cross-region blob management.

    Parameters
    ----------
    local_store : BlobStore
        Local blob store.
    registry : NodeRegistry
        Node discovery registry.
    config : FederationConfig, optional
        Federation settings.
    replication_policy : ReplicationPolicy, optional
        Replication rules (used when auto-replicating).
    """

    def __init__(
        self,
        local_store: BlobStore,
        registry: NodeRegistry,
        config: Optional[FederationConfig] = None,
        replication_policy: Optional[ReplicationPolicy] = None,
    ) -> None:
        self._store = local_store
        self._registry = registry
        self._config = config or FederationConfig()
        self._repl_policy = replication_policy or ReplicationPolicy()
        self._repl_engine = ReplicationEngine(local_store, registry, self._repl_policy)
        self._checker = ConsistencyChecker(local_store, registry)
        self._stats = FederationStats()

    def federated_put(self, blob_id: str, data: bytes) -> bool:
        """Write a blob to a quorum of nodes.

        Writes locally first, then pushes to enough remote nodes
        to satisfy the write quorum.

        Parameters
        ----------
        blob_id : str
            Blob identifier.
        data : bytes
            Blob data.

        Returns
        -------
        bool
            True if write quorum was achieved.
        """
        self._stats.puts += 1
        acks = 0

        # Write locally
        try:
            self._store.put(blob_id, data)
            acks += 1
            self._repl_engine.record_replica(blob_id, "local")
        except Exception as e:
            logger.error(f"Local put failed for {blob_id}: {e}")

        # Quorum already met?
        if acks >= self._config.write_quorum:
            # Still replicate in background for durability
            self._repl_engine.replicate_blob(blob_id)
            return True

        # Push to remote nodes
        needed = self._config.write_quorum - acks
        targets = self._registry.healthy_nodes()
        targets = sorted(targets, key=lambda n: (
            0 if n.region == self._config.local_region else 1,
            n.utilization,
        ))[:needed + 2]  # extra targets for fault tolerance

        with ThreadPoolExecutor(max_workers=self._config.max_workers) as pool:
            futures: dict[Future, NodeInfo] = {}
            for node in targets:
                futures[pool.submit(self._push_blob, blob_id, data, node)] = node

            for future in as_completed(futures):
                try:
                    future.result()
                    acks += 1
                    node = futures[future]
                    self._repl_engine.record_replica(blob_id, node.id)
                    self._stats.bytes_transferred += len(data)
                    if acks >= self._config.write_quorum:
                        break
                except Exception as e:
                    logger.warning(f"Remote put failed: {e}")

        if acks < self._config.write_quorum:
            self._stats.quorum_failures += 1
            logger.error(f"Write quorum not met for {blob_id}: {acks}/{self._config.write_quorum}")
            return False

        return True

    def federated_get(self, blob_id: str) -> Optional[bytes]:
        """Read a blob, preferring local then nearest healthy node.

        Parameters
        ----------
        blob_id : str
            Blob to read.

        Returns
        -------
        bytes or None
            Blob data, or None if not found anywhere.
        """
        self._stats.gets += 1

        # Try local first
        try:
            if self._store.has(blob_id):
                data = self._store.get(blob_id)
                self._stats.local_hits += 1
                return data
        except Exception:
            pass

        # Try remote nodes, nearest first
        nodes = self._registry.healthy_nodes()
        nodes = sorted(nodes, key=lambda n: (
            0 if n.region == self._config.local_region else 1,
            n.utilization,
        ))

        for node in nodes:
            try:
                data = self._fetch_from_node(blob_id, node)
                if data is not None:
                    self._stats.remote_hits += 1
                    self._stats.bytes_transferred += len(data)
                    # Cache locally
                    self._store.put(blob_id, data)
                    self._repl_engine.record_replica(blob_id, "local")
                    self._repl_engine.record_replica(blob_id, node.id)
                    return data
            except Exception as e:
                logger.warning(f"Fetch from {node.id} failed: {e}")
                continue

        logger.error(f"Blob not found in federation: {blob_id}")
        return None

    def federated_has(self, blob_id: str) -> bool:
        """Check if a blob exists in the federation."""
        if self._store.has(blob_id):
            return True
        for node in self._registry.healthy_nodes():
            try:
                url = f"{node.url.rstrip('/')}/v1/blobs/{quote(blob_id, safe=':')}"
                req = Request(url, method="HEAD")
                with urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                continue
        return False

    def check_consistency(self, blob_ids: Sequence[str]) -> ConsistencyReport:
        """Run a consistency check on the specified blobs."""
        return self._checker.scan_local(blob_ids)

    def _push_blob(self, blob_id: str, data: bytes, node: NodeInfo) -> None:
        """Push blob to a remote node."""
        url = f"{node.url.rstrip('/')}/v1/blobs/{quote(blob_id, safe=':')}"
        req = Request(url, data=data, method="PUT")
        req.add_header("Content-Type", "application/octet-stream")
        with urlopen(req, timeout=self._config.timeout_sec) as resp:
            if resp.status not in (200, 201, 204):
                raise RuntimeError(f"PUT failed: HTTP {resp.status}")

    def _fetch_from_node(self, blob_id: str, node: NodeInfo) -> Optional[bytes]:
        """Fetch blob from a remote node."""
        url = f"{node.url.rstrip('/')}/v1/blobs/{quote(blob_id, safe=':')}"
        try:
            with urlopen(url, timeout=self._config.timeout_sec) as resp:
                if resp.status == 200:
                    return resp.read()
        except HTTPError as e:
            if e.code == 404:
                return None
            raise
        return None

    @property
    def stats(self) -> FederationStats:
        return self._stats

    @property
    def replication_engine(self) -> ReplicationEngine:
        return self._repl_engine

    @property
    def consistency_checker(self) -> ConsistencyChecker:
        return self._checker
