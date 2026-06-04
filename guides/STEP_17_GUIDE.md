# Step 17 Guide — Signed Lineage Proofs (Feature-gated Ed25519)

This step adds optional cryptographic signing and verification for lineage proof digests.

## Why feature-gated?
Keeping Ed25519 behind a Cargo feature keeps the core lightweight and allows alternative signing schemes later.

## Added
- `sodl-proof`
  - `ProofSigner` trait
  - `ed25519::Ed25519Signer` (enabled with `--features sodl-proof/ed25519`)

- `sodl-service`
  - `SodlService.proof_signer: Option<&dyn ProofSigner>`
  - `share()` now signs the lineage proof digest if a signer is configured
  - `verify_share_proof(&ShareRecord)` helper

## Run
Unsigned (default):
```bash
cargo test
```

Signed demo:
```bash
cargo run -p sodl-service --example signed_share_demo --features sodl-proof/ed25519
```
