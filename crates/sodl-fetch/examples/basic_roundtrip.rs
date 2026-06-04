//! Basic end-to-end example demonstrating:
//! - compute blob id
//! - store in a durable store
//! - fetch through pipeline into cache
//!
//! Run:
//!   cargo run --example basic_roundtrip

use bytes::Bytes;
use sodl_cas::{compute_blob_id, BlobStore, HashAlg, MemBlobStore};
use sodl_fetch::{FetchPipeline, StoreSource};

fn main() -> sodl_core::Result<()> {
    // durable store (for demo, using MemBlobStore)
    let durable = MemBlobStore::new();

    // local cache (also mem)
    let cache = MemBlobStore::new();

    let plaintext = Bytes::from_static(b"hello SODL");
    let blob_id = compute_blob_id(&plaintext, HashAlg::Blake3);

    // Put into durable store (in reality: encrypted bytes + CAS hash over ciphertext)
    durable.put(&blob_id, plaintext.clone())?;

    // Build pipeline: cache -> durable source
    let src = StoreSource(&durable);
    let pipe = FetchPipeline {
        cache: &cache,
        sources: vec![&src],
        authorizer: None,
    };

    let fetched = pipe.get_for(None, None, &blob_id)?;
    assert_eq!(fetched, plaintext);

    // Ensure it was cached
    let cached = cache.get(&blob_id)?;
    assert_eq!(cached, plaintext);

    println!("OK: fetched + cached blob {}", blob_id.0);
    Ok(())
}
