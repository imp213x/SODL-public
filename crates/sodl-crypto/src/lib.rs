//! Cryptography boundaries for SODL.
//!
//! Design goals:
//! - Support **per-origin keys** (best balance between dedupe and blast radius)
//! - Provide envelope-based key distribution to principals
//! - Keep CAS immutable; encryption happens *before* storing bytes
//!
//! # Feature flags
//!
//! - **`aead`** — Enables production `AeadCrypto` (XChaCha20-Poly1305 + HKDF)
//!   and `InMemoryKeyManager`.  Without this feature only the development-only
//!   `NullCrypto` and `DevXorCrypto` are available.
//!
//! # Implementations
//!
//! | Type               | Feature  | Use case                          |
//! |--------------------|----------|-----------------------------------|
//! | `NullCrypto`       | default  | Pipeline wiring / dev (no crypto) |
//! | `DevXorCrypto`     | default  | Dedupe testing (toy XOR)          |
//! | `AeadCrypto`       | `aead`   | **Production** encryption         |
//! | `InMemoryKeyManager` | `aead` | Single-process key management     |

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use sodl_core::{KeyRef, OriginId, PrincipalId, Result};

/// An opaque envelope that carries a wrapped origin key for a principal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyEnvelope {
    pub origin_id: OriginId,
    pub principal: PrincipalId,
    /// Wrapped key bytes (opaque).
    pub wrapped_key: Vec<u8>,
    /// Optional metadata (alg, version, etc.)
    pub meta: Option<String>,
}

/// Provides origin keys and wrapping for principals.
/// Typically backed by a KMS/HSM.
pub trait KeyManager: Send + Sync {
    fn ensure_origin_key(&self, origin_id: OriginId) -> Result<KeyRef>;
    fn wrap_origin_key(&self, origin_id: OriginId, principal: &PrincipalId) -> Result<KeyEnvelope>;
    fn unwrap_origin_key(&self, envelope: &KeyEnvelope) -> Result<Vec<u8>>; // raw key bytes (opaque)
}

/// Encryptor boundary (bytes-in / bytes-out).
/// In practice you will encrypt **chunks** rather than whole files.
pub trait Encryptor: Send + Sync {
    fn encrypt_for_origin(&self, origin_id: OriginId, plaintext: Bytes) -> Result<Bytes>;
}

/// Decryptor boundary.
pub trait Decryptor: Send + Sync {
    fn decrypt_for_origin(&self, origin_id: OriginId, ciphertext: Bytes) -> Result<Bytes>;
}

/// Convenience trait for components that can both encrypt and decrypt.
pub trait Crypto: Encryptor + Decryptor {}
impl<T: Encryptor + Decryptor> Crypto for T {}

/// A development-only crypto provider that performs no encryption.
///
/// WARNING: This provides **no confidentiality**. Use only to validate pipeline wiring.
#[derive(Debug, Clone, Default)]
pub struct NullCrypto;

impl Encryptor for NullCrypto {
    fn encrypt_for_origin(&self, _origin_id: OriginId, plaintext: Bytes) -> Result<Bytes> {
        Ok(plaintext)
    }
}

impl Decryptor for NullCrypto {
    fn decrypt_for_origin(&self, _origin_id: OriginId, ciphertext: Bytes) -> Result<Bytes> {
        Ok(ciphertext)
    }
}

/// A deterministic XOR "crypto" for development/testing of dedupe behavior.
///
/// WARNING: This is **cryptographically broken** and must never be used in production.
/// It exists only to demonstrate that stable transformation (ciphertext) enables dedupe.
#[derive(Debug, Clone)]
pub struct DevXorCrypto {
    pub key_byte: u8,
}

impl DevXorCrypto {
    pub fn new(key_byte: u8) -> Self {
        Self { key_byte }
    }
}

impl Encryptor for DevXorCrypto {
    fn encrypt_for_origin(&self, origin_id: OriginId, plaintext: Bytes) -> Result<Bytes> {
        // Derive a deterministic single-byte mask from origin_id + key_byte (toy).
        let mut b = self.key_byte;
        for x in origin_id.0.as_bytes() {
            b ^= *x;
        }
        let mut out = plaintext.to_vec();
        for v in out.iter_mut() {
            *v ^= b;
        }
        Ok(Bytes::from(out))
    }
}

impl Decryptor for DevXorCrypto {
    fn decrypt_for_origin(&self, origin_id: OriginId, ciphertext: Bytes) -> Result<Bytes> {
        // XOR is symmetric
        self.encrypt_for_origin(origin_id, ciphertext)
    }
}

// ---------------------------------------------------------------------------
// Feature-gated production implementations
// ---------------------------------------------------------------------------

#[cfg(feature = "aead")]
mod aead;
#[cfg(feature = "aead")]
pub use aead::AeadCrypto;

#[cfg(feature = "aead")]
mod keymanager;
#[cfg(feature = "aead")]
pub use keymanager::InMemoryKeyManager;
