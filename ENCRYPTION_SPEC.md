# SODL V1 – Encryption & CAS Interface Spec (Skeleton)

This document defines the **interfaces** and **constraints** needed to support:
- Store-once within an origin
- Share/derivation across principals
- Integrity verification via CAS hashes

## Core constraints

1. **CAS integrity** is performed by hashing the stored bytes.
   - If you store ciphertext, BlobId must be hash(ciphertext).
   - Fetch pipeline verifies ciphertext integrity before any decrypt step.

2. **Deduplication** requires stable stored bytes for identical content.
   - If encryption is randomized per upload, ciphertext differs → dedupe breaks.
   - Therefore SODL prefers a **per-origin key** with **deterministic chunk encryption**
     (nonce derived deterministically) to keep ciphertext stable for the same plaintext chunk.

3. **Blast radius** must be controlled.
   - A single global key is unsafe.
   - Per-user keys break dedupe across users.
   - **Per-origin keys** are the best practical compromise.

## Interfaces (crate: `sodl-crypto`)

- `KeyManager`
  - `ensure_origin_key(origin_id) -> KeyRef`
  - `wrap_origin_key(origin_id, principal) -> KeyEnvelope`
  - `unwrap_origin_key(envelope) -> raw_key_bytes`

- `Encryptor`
  - `encrypt_for_origin(origin_id, plaintext_bytes) -> ciphertext_bytes`

- `Decryptor`
  - `decrypt_for_origin(origin_id, ciphertext_bytes) -> plaintext_bytes`

## Recommended encryption strategy (future implementation)

- Chunk content (fixed or content-defined).
- For each chunk:
  - `chunk_hash = blake3(plaintext_chunk)`
  - `nonce = blake3(origin_id || chunk_index || chunk_hash)[0..12]`
  - `ciphertext = AEAD_Encrypt(key_origin, nonce, plaintext_chunk, aad=origin_id)`
- Store ciphertext in CAS; BlobId computed over ciphertext.

## Notes

- This V1 skeleton does not implement cryptography.
- You can start with a "null crypto" implementation for development (plaintext passthrough),
  then swap to a real AEAD-based crypto provider later.
