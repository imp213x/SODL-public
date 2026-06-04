"""Minimal multi-tier blob fetch example."""

from __future__ import annotations

from pathlib import Path

from sodl import BlobStore, compute_blob_id


def main() -> None:
    base = Path("./demo-fetch")
    source = BlobStore(base / "source")
    cache = BlobStore(base / "cache", source_roots=[base / "source"])

    payload = b"distributed fetch demo"
    blob_id = compute_blob_id(payload)
    source.put(blob_id, payload)

    print(cache.get(blob_id).decode("utf-8"))
    print(cache.replica_nodes(blob_id))


if __name__ == "__main__":
    main()
