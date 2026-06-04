//! Durable store & replica tracking boundaries (skeleton).
//!
//! SODL distinguishes:
//! - **sources** used to fetch blobs (cache, peers, edge, durable stores)
//! - **durable stores** that count toward pin/replica satisfaction
//!
//! This crate defines:
//! - `DurableStore`: a store that can keep blobs durably and report zone identity
//! - `ReplicaTracker`: records which blobs are stored in which zones
//! - `ReplicaPlanner`: helper boundary used by pin planners (future)

pub mod weight_store;

use serde::{Deserialize, Serialize};
use sodl_cas::BlobStore;
use sodl_core::{BlobId, OriginId, Result};
use sodl_policy::{PinRecord, StorageZone};

/// A durable store is a BlobStore with a known failure domain (zone).
pub trait DurableStore: BlobStore {
    fn zone(&self) -> StorageZone;
    fn name(&self) -> &str;
}

/// Records where replicas exist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicaRecord {
    pub blob_id: BlobId,
    pub zone: StorageZone,
    pub store_name: String,
    pub observed_at: time::OffsetDateTime,
}

/// Tracks replicas for blobs and origins.
pub trait ReplicaTracker: Send + Sync {
    fn record_replica(&self, rec: ReplicaRecord) -> Result<()>;
    fn replicas_for_blob(&self, blob_id: &BlobId) -> Result<Vec<ReplicaRecord>>;
    fn replicas_for_origin(&self, origin_id: OriginId) -> Result<Vec<ReplicaRecord>>;
}

/// Computes whether a pin is satisfied (replica count + zones).
pub trait ReplicaPlanner: Send + Sync {
    fn is_pin_satisfied(&self, pin: &PinRecord) -> Result<bool>;
}

use bytes::Bytes;
use sodl_cas::{compute_blob_id, verify_integrity, HashAlg};
use sodl_crypto::Crypto;

/// A helper that stores **encrypted bytes** into an underlying BlobStore.
///
/// - Encrypts plaintext for an origin.
/// - Computes BlobId over the *ciphertext*.
/// - Stores ciphertext in the underlying store.
/// - On read: fetches ciphertext, verifies integrity, then decrypts.
///
/// This is a wiring primitive for V1; concrete crypto should be provided via `sodl-crypto` implementations.
pub struct EncryptedCas<'a> {
    pub store: &'a dyn BlobStore,
    pub crypto: &'a dyn Crypto,
    pub hash_alg: HashAlg,
}

impl<'a> EncryptedCas<'a> {
    pub fn new(store: &'a dyn BlobStore, crypto: &'a dyn Crypto, hash_alg: HashAlg) -> Self {
        Self {
            store,
            crypto,
            hash_alg,
        }
    }

    /// Encrypts and stores plaintext for an origin, returning the BlobId of the ciphertext.
    pub fn put_plain(&self, origin_id: OriginId, plaintext: Bytes) -> Result<BlobId> {
        let ciphertext = self.crypto.encrypt_for_origin(origin_id, plaintext)?;
        let blob_id = compute_blob_id(&ciphertext, self.hash_alg);
        self.store.put(&blob_id, ciphertext)?;
        Ok(blob_id)
    }

    /// Fetches ciphertext by BlobId, verifies integrity, decrypts, and returns plaintext.
    pub fn get_plain(&self, origin_id: OriginId, blob_id: &BlobId) -> Result<Bytes> {
        let ciphertext = self.store.get(blob_id)?;
        verify_integrity(blob_id, &ciphertext)?;
        self.crypto.decrypt_for_origin(origin_id, ciphertext)
    }
}

#[cfg(test)]
mod enc_tests {
    use super::*;
    use sodl_cas::MemBlobStore;
    use sodl_core::new_origin_id;
    use sodl_crypto::{DevXorCrypto, NullCrypto};

    #[test]
    fn null_crypto_roundtrip() {
        let store = MemBlobStore::new();
        let crypto = NullCrypto::default();
        let enc = EncryptedCas::new(&store, &crypto, HashAlg::Blake3);

        let origin = new_origin_id();
        let pt = Bytes::from_static(b"hello");
        let id = enc.put_plain(origin, pt.clone()).unwrap();
        let got = enc.get_plain(origin, &id).unwrap();
        assert_eq!(got, pt);
    }

    #[test]
    fn dev_xor_dedupes_within_origin() {
        let store = MemBlobStore::new();
        let crypto = DevXorCrypto::new(0xA5);
        let enc = EncryptedCas::new(&store, &crypto, HashAlg::Blake3);

        let origin = new_origin_id();
        let pt = Bytes::from_static(b"hello");

        let id1 = enc.put_plain(origin, pt.clone()).unwrap();
        let id2 = enc.put_plain(origin, pt.clone()).unwrap();
        assert_eq!(id1.0, id2.0);

        let ct = store.get(&id1).unwrap();
        assert_ne!(ct, pt);

        let back = enc.get_plain(origin, &id1).unwrap();
        assert_eq!(back, pt);
    }

    #[test]
    fn dev_xor_differs_across_origins() {
        let store = MemBlobStore::new();
        let crypto = DevXorCrypto::new(0xA5);
        let enc = EncryptedCas::new(&store, &crypto, HashAlg::Blake3);

        let o1 = new_origin_id();
        let o2 = new_origin_id();
        let pt = Bytes::from_static(b"same");

        let id1 = enc.put_plain(o1, pt.clone()).unwrap();
        let id2 = enc.put_plain(o2, pt.clone()).unwrap();
        assert_ne!(id1.0, id2.0);
    }
}

// ---------------------------------------------------------------------------
// AEAD integration tests (behind feature gate)
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "aead"))]
mod aead_integration_tests {
    use super::*;
    use sodl_cas::MemBlobStore;
    use sodl_core::new_origin_id;
    use sodl_crypto::AeadCrypto;

    #[test]
    fn aead_encrypted_cas_roundtrip() {
        let store = MemBlobStore::new();
        let crypto = AeadCrypto::generate();
        let enc = EncryptedCas::new(&store, &crypto, HashAlg::Blake3);

        let origin = new_origin_id();
        let pt = Bytes::from_static(b"real encryption roundtrip");
        let id = enc.put_plain(origin, pt.clone()).unwrap();

        // Ciphertext in the store is NOT the plaintext.
        let raw = store.get(&id).unwrap();
        assert_ne!(raw, pt, "stored bytes must be encrypted");

        // Decrypt roundtrip.
        let got = enc.get_plain(origin, &id).unwrap();
        assert_eq!(got, pt);
    }

    #[test]
    fn aead_dedup_within_origin() {
        let store = MemBlobStore::new();
        let crypto = AeadCrypto::generate();
        let enc = EncryptedCas::new(&store, &crypto, HashAlg::Blake3);

        let origin = new_origin_id();
        let pt = Bytes::from_static(b"dedup test");

        let id1 = enc.put_plain(origin, pt.clone()).unwrap();
        let id2 = enc.put_plain(origin, pt.clone()).unwrap();
        assert_eq!(
            id1.0, id2.0,
            "deterministic AEAD must produce same BlobId for same origin+plaintext"
        );
    }

    #[test]
    fn aead_cross_origin_isolation() {
        let store = MemBlobStore::new();
        let crypto = AeadCrypto::generate();
        let enc = EncryptedCas::new(&store, &crypto, HashAlg::Blake3);

        let o1 = new_origin_id();
        let o2 = new_origin_id();
        let pt = Bytes::from_static(b"same content different origin");

        let id1 = enc.put_plain(o1, pt.clone()).unwrap();
        let id2 = enc.put_plain(o2, pt.clone()).unwrap();
        assert_ne!(
            id1.0, id2.0,
            "different origins must produce different ciphertext/BlobIds"
        );

        // Decrypting with wrong origin fails.
        let err = enc.get_plain(o2, &id1);
        assert!(err.is_err(), "decrypt with wrong origin key must fail");
    }

    #[test]
    fn aead_large_payload_roundtrip() {
        let store = MemBlobStore::new();
        let crypto = AeadCrypto::generate();
        let enc = EncryptedCas::new(&store, &crypto, HashAlg::Blake3);

        let origin = new_origin_id();
        let pt = Bytes::from(vec![0xCDu8; 512 * 1024]); // 512 KiB
        let id = enc.put_plain(origin, pt.clone()).unwrap();
        let got = enc.get_plain(origin, &id).unwrap();
        assert_eq!(got, pt);
    }

    #[test]
    fn aead_tampered_blob_rejected() {
        let store = MemBlobStore::new();
        let crypto = AeadCrypto::generate();
        let enc = EncryptedCas::new(&store, &crypto, HashAlg::Blake3);

        let origin = new_origin_id();
        let pt = Bytes::from_static(b"don't tamper");
        let id = enc.put_plain(origin, pt.clone()).unwrap();

        // Tamper with raw ciphertext in the store.
        let mut raw = store.get(&id).unwrap().to_vec();
        if raw.len() > 30 {
            raw[30] ^= 0xFF;
        }
        // Put the tampered bytes back under the same BlobId.
        // (This bypasses integrity — the decrypt should fail due to AEAD tag.)
        store.put(&id, Bytes::from(raw)).unwrap();

        // get_plain first does integrity check (blake3) which should fail
        // because the tampered bytes don't match the BlobId.
        let err = enc.get_plain(origin, &id);
        assert!(err.is_err(), "tampered blob must be rejected");
    }
}
