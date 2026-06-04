"""Memory-Mapped Blob Reader — zero-copy access for large SODL blobs.

Uses `mmap` for read-only access to blob files without loading them
entirely into memory. Ideal for large weight files and embedding matrices.

Example
-------
>>> reader = MMapBlobReader("./blobs")
>>> with reader.open(blob_id) as view:
...     chunk = view[0:1024]  # read first 1KB without loading entire file
"""

from __future__ import annotations

import mmap
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


class MMapView:
    """A memory-mapped view of a blob file.

    Supports slice access, length, and iteration without loading
    the entire file into memory.
    """

    def __init__(self, mm: mmap.mmap, path: Path) -> None:
        self._mm = mm
        self._path = path

    def __len__(self) -> int:
        return len(self._mm)

    def __getitem__(self, key: int | slice) -> bytes:
        return self._mm[key]

    def read(self, size: int = -1) -> bytes:
        """Read bytes from current position."""
        return self._mm.read(size)

    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek to position."""
        return self._mm.seek(offset, whence)

    def tell(self) -> int:
        """Return current position."""
        return self._mm.tell()

    def close(self) -> None:
        """Close the memory map."""
        self._mm.close()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def size(self) -> int:
        return len(self._mm)


class MMapBlobReader:
    """Zero-copy blob reader using memory-mapped files.

    Parameters
    ----------
    blob_dir : str | Path
        Root directory containing blob files.
    """

    def __init__(self, blob_dir: str | Path) -> None:
        self._root = Path(blob_dir)

    def _blob_path(self, blob_id: str) -> Path:
        _, hex_hash = blob_id.split(":", 1) if ":" in blob_id else ("", blob_id)
        return self._root / f"{hex_hash}.blob"

    @contextmanager
    def open(self, blob_id: str) -> Iterator[MMapView]:
        """Open a blob for memory-mapped reading.

        Usage::
            with reader.open(blob_id) as view:
                header = view[0:64]
                total_size = len(view)
        """
        path = self._blob_path(blob_id)
        if not path.exists():
            raise FileNotFoundError(f"Blob not found: {blob_id} at {path}")

        fd = os.open(str(path), os.O_RDONLY)
        try:
            size = os.path.getsize(str(path))
            if size == 0:
                raise ValueError(f"Empty blob: {blob_id}")
            mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            view = MMapView(mm, path)
            try:
                yield view
            finally:
                view.close()
        finally:
            os.close(fd)

    def read_chunks(
        self, blob_id: str, chunk_size: int = 64 * 1024
    ) -> Iterator[bytes]:
        """Read a blob in fixed-size chunks.

        Useful for streaming processing of large blobs without
        loading them entirely into memory.

        Parameters
        ----------
        blob_id : str
            Blob to read.
        chunk_size : int
            Size of each chunk in bytes (default 64KB).

        Yields
        ------
        bytes
            Chunks of the blob data.
        """
        with self.open(blob_id) as view:
            offset = 0
            total = len(view)
            while offset < total:
                end = min(offset + chunk_size, total)
                yield view[offset:end]
                offset = end

    def blob_size(self, blob_id: str) -> int:
        """Get the on-disk size of a blob without reading it."""
        path = self._blob_path(blob_id)
        if not path.exists():
            raise FileNotFoundError(f"Blob not found: {blob_id}")
        return path.stat().st_size

    def exists(self, blob_id: str) -> bool:
        """Check if a blob exists on disk."""
        return self._blob_path(blob_id).exists()


class ArenaReader:
    """Sequential arena reader for bulk blob access.

    Pre-maps multiple blobs and reads them sequentially,
    optimized for scan-style workloads.

    Parameters
    ----------
    blob_dir : str | Path
        Root directory containing blob files.
    """

    def __init__(self, blob_dir: str | Path) -> None:
        self._reader = MMapBlobReader(blob_dir)
        self._stats = {"blobs_read": 0, "bytes_read": 0}

    def scan(
        self, blob_ids: list[str], chunk_size: int = 64 * 1024
    ) -> Iterator[tuple[str, bytes]]:
        """Sequentially read multiple blobs in chunks.

        Yields
        ------
        tuple[str, bytes]
            (blob_id, chunk) for each chunk of each blob.
        """
        for blob_id in blob_ids:
            self._stats["blobs_read"] += 1
            for chunk in self._reader.read_chunks(blob_id, chunk_size):
                self._stats["bytes_read"] += len(chunk)
                yield blob_id, chunk

    def read_all(self, blob_ids: list[str]) -> dict[str, bytes]:
        """Read all blobs into memory as a dict."""
        result = {}
        for blob_id in blob_ids:
            with self._reader.open(blob_id) as view:
                result[blob_id] = view[:]
                self._stats["blobs_read"] += 1
                self._stats["bytes_read"] += len(view)
        return result

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)
