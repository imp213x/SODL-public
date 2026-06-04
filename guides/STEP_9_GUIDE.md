# Step 9 Guide — Refcounts, Lineage & GC Eligibility

This step introduces `sodl-index` and wires it into `sodl-service` so the system can answer:

- "Is this origin still reachable?"
- "Can GC consider deleting it (subject to durability/pins/retention)?"
- "What is the lineage/audit trail for this origin?"

## What was added

- `crates/sodl-index`
  - `RefCounter` (origin + blob refcounts)
  - `LineageStore` + `LineageEdge` (audit edges)
  - `MemIndex` (in-memory reference implementation)

- `crates/sodl-service`
  - now depends on `sodl-index`
  - increments counts on upload/share/derive/pin
  - new methods:
    - `release_share(share_id)`
    - `unpin(pin_id)`
    - `is_gc_candidate(origin_id)`

## Run
```bash
cargo test
cargo run -p sodl-service --example gc_demo
```

## Important note
This step only provides **reachability signals**. Actual byte deletion must still be
policy-aware (Durability/Pins/Retention) and performed by a GC executor later.
