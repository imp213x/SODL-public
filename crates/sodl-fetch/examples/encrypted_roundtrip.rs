//! Encrypted pipeline wiring example.
//!
//! Run:
//!   cargo run -p sodl-fetch --example encrypted_roundtrip

use bytes::Bytes;
use sodl_cas::{HashAlg, MemBlobStore};
use sodl_core::new_origin_id;
use sodl_crypto::{Decryptor, DevXorCrypto};
use sodl_fetch::{FetchPipeline, StoreSource};
use sodl_store::EncryptedCas;

fn main() -> sodl_core::Result<()> {
    let origin = new_origin_id();

    let durable = MemBlobStore::new();
    let cache = MemBlobStore::new();

    // Dev crypto (NOT SECURE) â€“ for wiring + dedupe behavior only.
    let crypto = DevXorCrypto::new(0xA5);
    let enc = EncryptedCas::new(&durable, &crypto, HashAlg::Blake3);

    // Store plaintext -> ciphertext in durable store
    let pt = Bytes::from_static(b"hello encrypted SODL");
    let blob_id = enc.put_plain(origin, pt.clone())?;

    // Fetch ciphertext into cache via pipeline
    let src = StoreSource(&durable);
    let pipe = FetchPipeline {
        cache: &cache,
        sources: vec![&src],
        authorizer: None,
    };
    let ct = pipe.get_for(None, None, &blob_id)?;

    // Decrypt fetched ciphertext
    let back = crypto.decrypt_for_origin(origin, ct)?;
    assert_eq!(back, pt);

    println!("OK: encrypted stored blob {}", blob_id.0);
    Ok(())
}
