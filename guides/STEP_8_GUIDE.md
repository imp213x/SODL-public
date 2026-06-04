# Step 8 Guide — SODL Service Facade

This step introduces `sodl-service`, a high-level API that applications embed.

## Why this crate exists
The lower-level crates define boundaries (CAS, crypto, policy, manifests).
`sodl-service` is where you compose them into workflows:

- upload -> create origin + store bytes + set policy
- share -> store ShareRecord
- derive -> store DerivationManifest
- pin -> store PinRecord

## Run
```bash
cargo test
cargo run -p sodl-service --example service_demo
```

## Where to implement real backends later
Replace the in-memory stores with:
- Postgres (OriginRegistry/PolicyStore/PinStore/ShareStore/DerivationStore)
- Redis (hot indexes)
- Object storage (durable bytes) + local cache
