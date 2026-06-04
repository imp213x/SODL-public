"""Async Blob Store — non-blocking I/O for SODL blob operations.

Wraps BlobStore with a thread pool executor for async put/get/has operations,
enabling concurrent blob access without blocking the training loop.

Example
-------
>>> async_store = AsyncBlobStore(BlobStore("./blobs"), max_workers=4)
>>> futures = async_store.async_put_batch(blob_ids, data_list)
>>> results = async_store.wait_all(futures)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Sequence

from sodl_weights.store import BlobStore, compute_blob_id

logger = logging.getLogger(__name__)


@dataclass
class AsyncStats:
    """Statistics for async operations."""
    puts: int = 0
    gets: int = 0
    batch_puts: int = 0
    batch_gets: int = 0
    total_bytes_put: int = 0
    total_bytes_get: int = 0
    errors: int = 0
    total_time_sec: float = 0.0


class AsyncBlobStore:
    """Non-blocking wrapper around BlobStore using a thread pool.

    Parameters
    ----------
    blob_store : BlobStore
        The underlying synchronous blob store.
    max_workers : int
        Maximum number of concurrent I/O threads (default 4).
    """

    def __init__(self, blob_store: BlobStore, max_workers: int = 4) -> None:
        self._store = blob_store
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sodl-async")
        self._stats = AsyncStats()

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool."""
        self._pool.shutdown(wait=wait)

    # ── Single ops ─────────────────────────────────────────────────────

    def async_put(self, blob_id: str, data: bytes) -> Future[None]:
        """Asynchronously store a blob."""
        def _put():
            self._store.put(blob_id, data)
            self._stats.puts += 1
            self._stats.total_bytes_put += len(data)
        return self._pool.submit(_put)

    def async_get(self, blob_id: str) -> Future[bytes]:
        """Asynchronously retrieve a blob."""
        def _get():
            result = self._store.get(blob_id)
            self._stats.gets += 1
            self._stats.total_bytes_get += len(result)
            return result
        return self._pool.submit(_get)

    def async_has(self, blob_id: str) -> Future[bool]:
        """Asynchronously check if a blob exists."""
        return self._pool.submit(self._store.has, blob_id)

    # ── Batch ops ──────────────────────────────────────────────────────

    def async_put_batch(
        self,
        items: Sequence[tuple[str, bytes]],
    ) -> list[Future[None]]:
        """Store multiple blobs concurrently.

        Parameters
        ----------
        items : list of (blob_id, data) tuples
            Blobs to store.

        Returns
        -------
        list[Future]
            Futures for each put operation.
        """
        self._stats.batch_puts += 1
        return [self.async_put(blob_id, data) for blob_id, data in items]

    def async_get_batch(self, blob_ids: Sequence[str]) -> list[Future[bytes]]:
        """Retrieve multiple blobs concurrently.

        Parameters
        ----------
        blob_ids : list of str
            Blob IDs to retrieve.

        Returns
        -------
        list[Future]
            Futures for each get operation.
        """
        self._stats.batch_gets += 1
        return [self.async_get(blob_id) for blob_id in blob_ids]

    def wait_all(self, futures: list[Future]) -> list[Any]:
        """Wait for all futures and return results. Raises on first error."""
        start = time.perf_counter()
        results = []
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                self._stats.errors += 1
                raise
        self._stats.total_time_sec += time.perf_counter() - start
        return results

    def collect_results(self, futures: list[Future]) -> list[Any]:
        """Wait for all futures preserving order. Returns results in input order."""
        start = time.perf_counter()
        results = []
        for f in futures:
            try:
                results.append(f.result())
            except Exception as e:
                self._stats.errors += 1
                results.append(e)
        self._stats.total_time_sec += time.perf_counter() - start
        return results

    @property
    def stats(self) -> AsyncStats:
        return self._stats

    @property
    def store(self) -> BlobStore:
        return self._store
