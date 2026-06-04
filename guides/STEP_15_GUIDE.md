# Step 15 Guide — Unsigned Lineage Proofs (Crypto-aligned)

This step introduces deterministic, unsigned lineage proofs for an origin. A proof is a Blake3 digest over a canonical
representation of the origin's lineage edges.

## Added crate
- `crates/sodl-proof`
  - `LineageProof`
  - `generate_proof_unsigned(origin_id, edges, now)`

## Service method
- `SodlService::lineage_proof(origin_id) -> LineageProof`

## Run
```bash
cargo test
cargo run -p sodl-service --example lineage_proof_demo
```

## Notes
- Proofs are **unsigned** in V1. A later step can add Ed25519 signing and proof anchoring into share/manifests.
