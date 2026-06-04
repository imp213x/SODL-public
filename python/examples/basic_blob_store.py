"""Basic blob store demo — content-addressed storage in 20 lines.

Usage:
    python examples/basic_blob_store.py
"""

import tempfile
from sodl_weights import BlobStore, compute_blob_id, verify_integrity

# Create a temporary blob store
with tempfile.TemporaryDirectory() as tmpdir:
    store = BlobStore(tmpdir)

    # Store some data
    data = b"The quick brown fox jumps over the lazy dog."
    blob_id = compute_blob_id(data)
    store.put(blob_id, data)
    print(f"Stored: {blob_id}")
    print(f"Blob count: {store.blob_count()}")

    # Retrieve and verify integrity
    retrieved = store.get(blob_id)
    verify_integrity(blob_id, retrieved)
    print(f"Retrieved: {retrieved.decode()}")
    print("Integrity verified ✓")

    # Deduplication: storing the same content again is a no-op
    already_exists = store.has(blob_id)
    print(f"Deduplication check: blob already exists = {already_exists}")
