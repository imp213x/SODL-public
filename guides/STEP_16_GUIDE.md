# Step 16 Guide — Anchored Share Proofs (Unsigned)

This step anchors each share to the origin's lineage state at the moment of sharing.

## What changed

- `sodl-manifest::ShareRecord` now includes:
  - `lineage_proof_digest` (Blake3 hex)
  - `lineage_proof_created_at`
  - `lineage_proof_key_id` (optional; reserved for signed proofs)
  - `lineage_proof_sig_b64` (optional; reserved for signed proofs)

- `sodl-service::SodlService::share(...)` now computes `lineage_proof(origin_id)` and persists the digest into the stored `ShareRecord`.

## Why

- Later reshares can prove they derived from an origin state (tamper-evident lineage snapshot).
- Enables future signed proofs and enforcement/auditing.

## Run

```bash
cargo test
cargo run -p sodl-service --example lineage_proof_demo
```
