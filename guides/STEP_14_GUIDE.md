# Step 14 Guide — Durability-safe GC (Replica Health Gate)

This step ensures GC deletion is *gated* by durability constraints when content is actively pinned.

## What was added

- `sodl-replica`
  - `healthy_count_with_stale(blob_id, stale_seconds, now)`

- `sodl-gc`
  - `ReplicaAuditor` now accepts `stale_seconds` + `now`
  - `DurabilityGate` which prevents deletion if `min_replicas` is not satisfied

- Example: `durability_safe_gc_demo`
  - Shows `can_delete_origin_bytes == false` when only 1 replica exists for `min_replicas=2`
  - After recording a second healthy replica, gate becomes true

## Run

```bash
cargo test
cargo run -p sodl-service --example durability_safe_gc_demo
```
