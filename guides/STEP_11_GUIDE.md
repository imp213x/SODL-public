# Step 11 Guide — TTL Enforcement + Tombstone-first Origin Deletion

This step hardens deletion semantics and begins retention enforcement.

## What changed

### Origin records now have timestamps
`crates/sodl-origin/src/lib.rs`:
- `created_at`
- `tombstoned_at`
- `tombstone_reason`

### Tombstone-first origin deletion
`sodl-service` now exposes:
- `tombstone_origin(origin_id, reason)`

This:
- marks the origin as tombstoned
- decrements origin/blob refcounts for its representations
- clears representations
- keeps metadata record (so audit and TTL enforcement remain possible)

### TTL enforcement in GC planning
`GcPlanner` now uses:
- `PolicyStore` to read `ttl_seconds`
- `OriginRegistry` to read `created_at`

If TTL is set and not yet elapsed, the origin is not eligible for tombstoning/GC actions.

## Run
```bash
cargo test
cargo run -p sodl-service --example policy_gc_demo
```

## Notes
- TTL is enforced only for origin tombstone eligibility in this step.
- Blob deletion remains refcount-driven (Choice B).
