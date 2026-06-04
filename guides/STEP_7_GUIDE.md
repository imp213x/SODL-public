# Step 7 Guide — NullCrypto + Encrypted CAS Wiring

This step adds a minimal, testable wiring primitive that connects:

plaintext -> Encryptor -> ciphertext -> BlobId(hash(ciphertext)) -> BlobStore

and for reads:

BlobStore -> ciphertext -> integrity check -> Decryptor -> plaintext

## What was added

- `crates/sodl-crypto`:
  - `NullCrypto` (no encryption; dev only)
  - `DevXorCrypto` (deterministic XOR; dev only; NOT SECURE)

- `crates/sodl-store`:
  - `EncryptedCas` helper with `put_plain` / `get_plain`

- `crates/sodl-fetch/examples/encrypted_roundtrip.rs`

## Run

```bash
cargo test
cargo run -p sodl-fetch --example encrypted_roundtrip
```

## Security warning

`NullCrypto` and `DevXorCrypto` are placeholders only.
Replace with a real AEAD implementation (e.g. AES-GCM / ChaCha20-Poly1305) later.
