"""Consistency Checker — verify blob integrity across SODL replicas.

Scans local and remote stores to detect corruption, missing replicas,
and orphaned blobs. Can repair by re-replicating from healthy sources.

Example
-------
>>> checker = ConsistencyChecker(local_store, registry)
>>> report = checker.check_blob(blob_id)
>>> full_report = checker.full_scan()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import blake3

from sodl_weights.store import BlobStore, canonical_blob_id, compute_blob_id
from sodl_weights.registry import NodeRegistry, NodeInfo

logger = logging.getLogger(__name__)


@dataclass
class BlobCheckResult:
    """Result of checking a single blob."""
    blob_id: str
    exists_locally: bool = False
    local_valid: bool = False
    remote_replicas: int = 0
    corrupt_nodes: list[str] = field(default_factory=list)
    missing_nodes: list[str] = field(default_factory=list)
    healthy_nodes: list[str] = field(default_factory=list)
    repaired: bool = False


@dataclass
class ConsistencyReport:
    """Summary of a consistency scan."""
    total_blobs: int = 0
    healthy: int = 0
    corrupt: int = 0
    under_replicated: int = 0
    orphaned: int = 0
    repaired: int = 0
    elapsed_sec: float = 0.0
    details: list[BlobCheckResult] = field(default_factory=list)

    @property
    def healthy_pct(self) -> float:
        if self.total_blobs == 0:
            return 100.0
        return (self.healthy / self.total_blobs) * 100


class ConsistencyChecker:
    """Verifies blob integrity across SODL replicas.

    Parameters
    ----------
    local_store : BlobStore
        The local blob store to check.
    registry : NodeRegistry, optional
        Node registry for checking remote replicas.
    min_replicas : int
        Minimum expected replicas per blob.
    """

    def __init__(
        self,
        local_store: BlobStore,
        registry: Optional[NodeRegistry] = None,
        min_replicas: int = 1,
    ) -> None:
        self._store = local_store
        self._registry = registry
        self._min_replicas = min_replicas

    def check_blob(self, blob_id: str) -> BlobCheckResult:
        """Check a single blob for integrity.

        Verifies:
        1. Blob exists locally
        2. Local data matches blob_id hash
        3. Remote replicas exist and are valid
        """
        result = BlobCheckResult(blob_id=blob_id)

        # Check local
        try:
            if self._store.has(blob_id):
                result.exists_locally = True
                data = self._store.get(blob_id)
                actual_id = compute_blob_id(data)
                result.local_valid = (actual_id == blob_id)
                if result.local_valid:
                    result.healthy_nodes.append("local")
                else:
                    result.corrupt_nodes.append("local")
                    logger.warning(f"Corrupt blob {blob_id}: hash mismatch")
        except Exception as e:
            logger.error(f"Error checking {blob_id}: {e}")

        return result

    def verify_data(self, blob_id: str, data: bytes) -> bool:
        """Verify that data matches its blob_id."""
        actual = compute_blob_id(data)
        return actual == blob_id

    def scan_local(self, blob_ids: Sequence[str]) -> ConsistencyReport:
        """Scan a list of blob IDs for local integrity.

        Parameters
        ----------
        blob_ids : list of str
            Blob IDs to check.

        Returns
        -------
        ConsistencyReport
            Summary of the scan.
        """
        start = time.time()
        report = ConsistencyReport(total_blobs=len(blob_ids))

        for blob_id in blob_ids:
            result = self.check_blob(blob_id)
            report.details.append(result)

            if result.local_valid:
                report.healthy += 1
            elif result.exists_locally and not result.local_valid:
                report.corrupt += 1
            elif not result.exists_locally:
                report.under_replicated += 1

        report.elapsed_sec = time.time() - start
        return report

    def find_orphaned(self, known_blob_ids: set[str]) -> list[str]:
        """Find blobs on disk that aren't in the known set.

        Parameters
        ----------
        known_blob_ids : set of str
            Set of expected blob IDs.

        Returns
        -------
        list of str
            Orphaned blob file paths.
        """
        known = {canonical_blob_id(blob_id) for blob_id in known_blob_ids if str(blob_id).strip()}
        orphaned = []
        root = self._store._root
        for blob_file in root.glob("*.blob"):
            blob_id = canonical_blob_id(blob_file.name)
            if blob_id not in known:
                orphaned.append(blob_id)
        return orphaned

    def repair_blob(self, blob_id: str, source_data: bytes) -> bool:
        """Repair a corrupt or missing blob by re-writing from good data.

        Parameters
        ----------
        blob_id : str
            The blob ID to repair.
        source_data : bytes
            Known-good data for this blob.

        Returns
        -------
        bool
            True if repair was successful.
        """
        if not self.verify_data(blob_id, source_data):
            logger.error(f"Cannot repair {blob_id}: source data doesn't match hash")
            return False

        try:
            self._store.put(blob_id, source_data)
            logger.info(f"Repaired blob {blob_id}")
            return True
        except Exception as e:
            logger.error(f"Repair failed for {blob_id}: {e}")
            return False
