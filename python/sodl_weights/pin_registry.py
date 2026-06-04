"""Weight Pin Registry — hot/cold cluster RAM cache with refcount-based eviction.

Mirrors the Rust ``WeightPinRegistry`` in ``sodl-store/src/weight_store.rs``.
Now with disk persistence for cache warmth across restarts.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from sodl_weights.types import WeightCluster, WeightPinReason

logger = logging.getLogger(__name__)




class WeightPinError(Exception):
    """Raised on illegal pin operations (e.g. unpinning protected clusters)."""


@dataclass
class _PinnedEntry:
    cluster: WeightCluster
    reason: WeightPinReason
    access_count: int = 1


class WeightPinRegistry:
    """In-memory hot/cold weight cluster cache.

    - Identity and logic clusters are always pinned and never evicted.
    - Other clusters are evicted LRU-by-refcount when cache exceeds ``max_entries``.

    Thread-safe via internal lock.
    """

    def __init__(self, max_entries: int = 256) -> None:
        self._max_entries = max_entries
        self._entries: dict[str, _PinnedEntry] = {}
        # `pin()` may call eviction helpers that also take the registry lock.
        # Use a re-entrant lock so evaluator runs do not deadlock on nested access.
        self._lock = threading.RLock()

    def pin(
        self,
        cluster_id: str,
        cluster: WeightCluster,
        reason: WeightPinReason,
    ) -> None:
        """Pin a cluster in the hot cache."""
        with self._lock:
            keep_limit = max(self._max_entries - 1, 0)
            self.shed_except(set(), keep_limit)
            self._entries[cluster_id] = _PinnedEntry(
                cluster=cluster, reason=reason, access_count=1,
            )

    def get(self, cluster_id: str) -> Optional[WeightCluster]:
        """Get a cached cluster, incrementing its access count."""
        with self._lock:
            entry = self._entries.get(cluster_id)
            if entry is None:
                return None
            entry.access_count += 1
            return entry.cluster

    def is_pinned(self, cluster_id: str) -> bool:
        with self._lock:
            return cluster_id in self._entries

    def reason(self, cluster_id: str) -> Optional[WeightPinReason]:
        with self._lock:
            entry = self._entries.get(cluster_id)
            return entry.reason if entry is not None else None

    def hydrate(self, cluster_id: str, cluster: WeightCluster) -> bool:
        """Replace a placeholder cluster while preserving pin metadata."""
        with self._lock:
            entry = self._entries.get(cluster_id)
            if entry is None:
                return False
            entry.cluster = cluster
            return True

    def unpin(self, cluster_id: str) -> bool:
        """Remove a cluster from cache. Raises for protected clusters."""
        with self._lock:
            entry = self._entries.get(cluster_id)
            if entry is None:
                return False
            if entry.reason in {WeightPinReason.IDENTITY, WeightPinReason.LOGIC}:
                raise WeightPinError("Cannot unpin identity- or logic-pinned cluster")
            self._entries.pop(cluster_id, None)
            return True

    def refcount(self, cluster_id: str) -> int:
        with self._lock:
            entry = self._entries.get(cluster_id)
            return entry.access_count if entry else 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def shed_non_active(self, active_cluster_ids: set[str]) -> int:
        """Aggressively evict all clusters NOT in the active set.
        
        Excludes protected pins. Returns the number of clusters evicted.
        """
        with self._lock:
            to_remove = [
                cid for cid, entry in self._entries.items()
                if cid not in active_cluster_ids and entry.reason not in {WeightPinReason.IDENTITY, WeightPinReason.LOGIC}
            ]
            for cid in to_remove:
                self._entries.pop(cid)
            return len(to_remove)

    def shed_except(self, protected_cluster_ids: set[str], max_to_keep: int) -> int:
        """Shed weights until under limit, but PROTECT specific IDs from LRU.
        
        Useful for keeping the 'next batch' hot while shedding the 'previous batch'.
        """
        with self._lock:
            if len(self._entries) <= max_to_keep:
                return 0
                
            # Candidates: evictable and not specifically protected
            candidates = sorted(
                (
                    (cid, e.access_count)
                    for cid, e in self._entries.items()
                    if e.reason not in {WeightPinReason.IDENTITY, WeightPinReason.LOGIC} and cid not in protected_cluster_ids
                ),
                key=lambda x: x[1],
            )
            
            evicted = 0
            while len(self._entries) > max_to_keep and candidates:
                cid, _ = candidates.pop(0)
                self._entries.pop(cid)
                evicted += 1
            return evicted

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Persist pin metadata (reasons + access counts) to a JSON file.

        Cluster data itself is NOT persisted — it is reconstructed from
        SODL blobs on load. This keeps the snapshot small (<1 KB/entry).
        """
        with self._lock:
            snapshot: list[dict[str, Any]] = []
            for cid, entry in self._entries.items():
                snapshot.append(
                    {
                        "cluster_id": cid,
                        "reason": entry.reason.value,
                        "access_count": entry.access_count,
                    }
                )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"version": 1, "max_entries": self._max_entries, "pins": snapshot},
            indent=2,
        )
        path.write_text(payload, encoding="utf-8")
        logger.info("Pin registry saved: %d entries -> %s", len(snapshot), path)

    @classmethod
    def load(cls, path: Path, *, max_entries: int | None = None) -> "WeightPinRegistry":
        """Restore a registry from a JSON snapshot.

        Cluster data is NOT restored (set to empty); callers should
        re-fetch clusters from the blob store as needed.
        """
        path = Path(path)
        if not path.exists():
            logger.info("No pin registry snapshot at %s — starting fresh.", path)
            return cls(max_entries=max_entries or 256)

        raw = json.loads(path.read_text(encoding="utf-8"))
        stored_max = raw.get("max_entries", 256)
        registry = cls(max_entries=max_entries or stored_max)

        for pin_record in raw.get("pins", []):
            cid = pin_record["cluster_id"]
            reason = WeightPinReason(pin_record["reason"])
            access_count = int(pin_record.get("access_count", 1))
            # Create a placeholder cluster (empty centroid/members)
            placeholder = WeightCluster(
                centroid=[], member_token_ids=[], offsets=[], dim=0, cluster_id=cid,
            )
            registry._entries[cid] = _PinnedEntry(
                cluster=placeholder, reason=reason, access_count=access_count,
            )

        logger.info("Pin registry loaded: %d entries from %s", len(registry._entries), path)
        return registry

    def stats(self) -> dict[str, Any]:
        """Return cache occupancy stats."""
        with self._lock:
            identity_count = sum(
                1 for e in self._entries.values()
                if e.reason == WeightPinReason.IDENTITY
            )
            logic_count = sum(
                1 for e in self._entries.values()
                if e.reason == WeightPinReason.LOGIC
            )
            return {
                "total_pinned": len(self._entries),
                "max_entries": self._max_entries,
                "identity_pins": identity_count,
                "logic_pins": logic_count,
                "evictable_pins": len(self._entries) - identity_count - logic_count,
                "utilization": len(self._entries) / max(1, self._max_entries),
            }
