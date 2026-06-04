"""Batch Operations — efficient bulk store/load for SODL.

Provides high-level batch operations that automatically parallelize I/O,
compress data, and track statistics.

Example
-------
>>> ops = BatchOps(BlobStore("./blobs"), max_workers=4)
>>> results = ops.batch_store_numpy("run-1", arrays, names)
>>> loaded = ops.batch_load_numpy(blob_ids)
"""

from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import zstandard as zstd

from sodl_weights.store import BlobStore, compute_blob_id


@dataclass
class BatchResult:
    """Result of a batch operation."""
    blob_ids: list[str]
    total_raw_bytes: int = 0
    total_stored_bytes: int = 0
    n_items: int = 0
    elapsed_sec: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        if self.total_raw_bytes == 0:
            return 0.0
        return 1.0 - (self.total_stored_bytes / self.total_raw_bytes)

    @property
    def throughput_mb_sec(self) -> float:
        if self.elapsed_sec == 0:
            return 0.0
        return (self.total_raw_bytes / 1024 / 1024) / self.elapsed_sec


class BatchOps:
    """High-level batch operations for SODL blob stores.

    Parallelizes compression and I/O across a thread pool for significantly
    faster bulk data handling.

    Parameters
    ----------
    blob_store : BlobStore
        Underlying content-addressed store.
    max_workers : int
        Thread pool size (default 4).
    zstd_level : int
        Compression level (default 3).
    """

    def __init__(
        self,
        blob_store: BlobStore,
        max_workers: int = 4,
        zstd_level: int = 3,
    ) -> None:
        self._store = blob_store
        self._max_workers = max_workers
        self._zstd_level = zstd_level

    def _compress_and_store(self, data: bytes) -> tuple[str, int, int]:
        """Compress, hash, and store a single blob. Returns (blob_id, raw_size, stored_size)."""
        compressor = zstd.ZstdCompressor(level=self._zstd_level)
        raw_size = len(data)
        compressed = compressor.compress(data)
        blob_id = compute_blob_id(compressed)
        self._store.put(blob_id, compressed)
        return blob_id, raw_size, len(compressed)

    def batch_store(self, data_items: Sequence[bytes]) -> BatchResult:
        """Store multiple raw byte blobs concurrently.

        Parameters
        ----------
        data_items : list of bytes
            Data to store.

        Returns
        -------
        BatchResult
            Summary of the batch operation.
        """
        start = time.perf_counter()
        result = BatchResult(blob_ids=[], n_items=len(data_items))

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self._compress_and_store, d): i
                       for i, d in enumerate(data_items)}

            # Maintain order
            ordered = [None] * len(data_items)
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    blob_id, raw_size, stored_size = future.result()
                    ordered[idx] = blob_id
                    result.total_raw_bytes += raw_size
                    result.total_stored_bytes += stored_size
                except Exception as e:
                    result.errors.append(f"Item {idx}: {e}")
                    ordered[idx] = ""

        result.blob_ids = ordered
        result.elapsed_sec = time.perf_counter() - start
        return result

    def batch_load(self, blob_ids: Sequence[str]) -> list[bytes]:
        """Load and decompress multiple blobs concurrently.

        Parameters
        ----------
        blob_ids : list of str
            Blob IDs to load.

        Returns
        -------
        list of bytes
            Decompressed data in the same order as input blob_ids.
        """
        def _load_one(blob_id: str) -> bytes:
            # Each thread creates its own decompressor (zstd is NOT thread-safe)
            decompressor = zstd.ZstdDecompressor()
            compressed = self._store.get(blob_id)
            try:
                return decompressor.decompress(compressed)
            except zstd.ZstdError:
                return compressed

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(_load_one, bid): i for i, bid in enumerate(blob_ids)}
            results = [None] * len(blob_ids)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return results

    def batch_store_numpy(
        self,
        arrays: Sequence[np.ndarray],
        names: Sequence[str] | None = None,
    ) -> BatchResult:
        """Store multiple numpy arrays as blobs concurrently.

        Parameters
        ----------
        arrays : list of np.ndarray
            Arrays to store.
        names : list of str, optional
            Names for each array (for logging).

        Returns
        -------
        BatchResult
            Summary including blob IDs, compression, and throughput.
        """
        # Serialize arrays to bytes
        serialized = []
        for arr in arrays:
            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=False)
            serialized.append(buf.getvalue())

        return self.batch_store(serialized)

    def batch_load_numpy(self, blob_ids: Sequence[str]) -> list[np.ndarray]:
        """Load multiple numpy arrays from blobs concurrently.

        Parameters
        ----------
        blob_ids : list of str
            Blob IDs containing numpy arrays.

        Returns
        -------
        list of np.ndarray
            Arrays in the same order as input blob_ids.
        """
        raw_data = self.batch_load(blob_ids)
        return [np.load(io.BytesIO(d), allow_pickle=False) for d in raw_data]
