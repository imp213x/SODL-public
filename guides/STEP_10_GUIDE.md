# Step 10 Guide — Policy-aware GC, Tombstones & Blob-level Deletion (Choice B)

This step implements the foundation of deletion safety.

## Key decision: B (blob-level GC)
- Origins are metadata objects
- Blobs are the byte objects
- Bytes are deleted only when **blob_refcount == 0**

## What was added

- `crates/sodl-index`
  - `ScanIndex` trait so GC can list known origins/blobs (MemIndex implements it)

- `crates/sodl-gc`
  - `Tombstone`, `TombstoneStore`, `MemTombstoneStore`
  - `GcPlanner` (policy-aware, conservative)
  - `GcExecutor` (tombstone -> delete bytes)

- `crates/sodl-service`
  - `delete_origin(origin_id)` decrements origin+blob refcounts for representations and deletes origin metadata
  - demo: `policy_gc_demo`

## Run

```bash
cargo test
cargo run -p sodl-service --example policy_gc_demo
```

## Notes / limitations
- TTL enforcement needs origin creation timestamps (next step)
- Real systems should tombstone origin metadata before deletion
