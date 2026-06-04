# SODL V1 Decisions (initial)

## V1 goal
A skeleton that locks in *module boundaries* and *data model primitives*.

## Invariants
- Blobs are immutable and integrity-checked by hash.
- Origins group blobs and enable lineage tracking.
- Derivations are *manifests*; transforms may optionally materialize new blobs.
- Durability requires at least one pinned/durable backend (not implemented in skeleton).

## Deferred (explicitly)
- Concrete storage backend(s)
- Concrete encryption implementation (KeyManager interface only)
- Real peer transport / NAT traversal
- Media segmentation/transcoding pipeline
- Watermark embedding / perceptual hashing implementations

## Added in Step 6
- Retrieval pipeline contracts (StoreSource adapter)
- Durable store + replica tracking boundaries (`sodl-store`)
- In-memory BlobStore for examples/tests (`MemBlobStore`)
