# Step 12 Guide — Replica-aware Durability (Pin Satisfaction)

This step introduces replica tracking and a durability audit that checks whether pinned content
meets its `min_replicas` requirement.

## What was added

- `crates/sodl-replica`
  - `ReplicaRecord`, `ReplicaState`
  - `ReplicaStore` trait + `MemReplicaStore`

- `crates/sodl-gc`
  - `ReplicaAuditor`
  - `RepairPlan` / `RepairItem`

## Run

```bash
cargo test
cargo run -p sodl-service --example replica_audit_demo
```

Expected:
- first audit shows 1 missing replica
- second audit shows 0 missing replicas after adding another healthy replica

## Next
Implement a replication executor to satisfy the repair plan.
