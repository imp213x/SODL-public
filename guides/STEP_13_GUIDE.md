# Step 13 Guide — Replica Executor (Repair Plan Execution)

This step adds a replication executor that can satisfy `min_replicas` by copying blob bytes
to missing nodes/stores.

## What was added

- `crates/sodl-replica`
  - `RepairPlan` / `RepairItem`
  - `StoreMesh` abstraction for per-node `BlobStore`
  - `ReplicaExecutor` that copies bytes + updates `ReplicaStore`
  - `MemStoreMesh` demo implementation

- `crates/sodl-service/examples/replica_repair_demo.rs`
  - audits an origin -> gets a repair plan -> executes it -> re-audits

## Run

```bash
cargo test
cargo run -p sodl-service --example replica_repair_demo
```

Expected:
- repair items before execute: 1
- repair items after execute: 0
- store-b has blob? true
