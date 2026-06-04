//! In-memory key manager for SODL.
//!
//! Generates per-origin keys deterministically from a master key via HKDF
//! and stores wrapped envelopes in memory.  Suitable for single-process
//! deployments and tests.  For production multi-node use, back with a real
//! KMS (AWS KMS, HashiCorp Vault, Azure Key Vault, etc.).

use std::collections::HashMap;
use std::sync::Mutex;

use hkdf::Hkdf;
use sha2::Sha256;

use sodl_core::{KeyRef, OriginId, PrincipalId, Result, SodlError};

use crate::{KeyEnvelope, KeyManager};

/// A simple in-memory key manager backed by a master key.
///
/// - `ensure_origin_key`: derives a deterministic per-origin key via HKDF.
/// - `wrap_origin_key`: "wraps" the key for a principal by XOR-ing with a
///   principal-derived mask (symbolic wrapping — replace with real KEM in
///   production).
/// - `unwrap_origin_key`: reverses the wrapping.
///
/// All state is in memory; keys do not survive process restart unless the
/// same master key is provided again (which re-derives the same origin keys).
pub struct InMemoryKeyManager {
    master_key: [u8; 32],
    /// Cache of origin keys (origin_id → raw 32-byte key).
    cache: Mutex<HashMap<OriginId, [u8; 32]>>,
}

impl InMemoryKeyManager {
    /// Create with a 32-byte master key.
    pub fn new(master_key: [u8; 32]) -> Self {
        Self {
            master_key,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Derive a per-origin key (same logic as `AeadCrypto`).
    fn derive_key(&self, origin_id: OriginId) -> [u8; 32] {
        let hk = Hkdf::<Sha256>::new(None, &self.master_key);
        let info = format!("sodl-origin-v1:{}", origin_id.0);
        let mut okm = [0u8; 32];
        hk.expand(info.as_bytes(), &mut okm)
            .expect("HKDF expand 32 bytes");
        okm
    }

    /// Derive a per-principal wrapping mask (for key envelope).
    fn principal_mask(principal: &PrincipalId) -> [u8; 32] {
        let hash = blake3::hash(principal.0.as_bytes());
        let mut mask = [0u8; 32];
        mask.copy_from_slice(&hash.as_bytes()[..32]);
        mask
    }
}

impl std::fmt::Debug for InMemoryKeyManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemoryKeyManager")
            .field("master_key", &"[REDACTED]")
            .finish()
    }
}

impl KeyManager for InMemoryKeyManager {
    fn ensure_origin_key(&self, origin_id: OriginId) -> Result<KeyRef> {
        let key = self.derive_key(origin_id);
        let mut cache = self
            .cache
            .lock()
            .map_err(|e| SodlError::Io(format!("key cache lock poisoned: {e}")))?;
        cache.insert(origin_id, key);
        Ok(KeyRef(format!("mem:{}", origin_id.0)))
    }

    fn wrap_origin_key(&self, origin_id: OriginId, principal: &PrincipalId) -> Result<KeyEnvelope> {
        let key = self.derive_key(origin_id);
        let mask = Self::principal_mask(principal);

        // Symbolic wrapping: XOR key with principal mask.
        let mut wrapped = vec![0u8; 32];
        for i in 0..32 {
            wrapped[i] = key[i] ^ mask[i];
        }

        Ok(KeyEnvelope {
            origin_id,
            principal: principal.clone(),
            wrapped_key: wrapped,
            meta: Some("xchacha20poly1305-hkdf-v1".into()),
        })
    }

    fn unwrap_origin_key(&self, envelope: &KeyEnvelope) -> Result<Vec<u8>> {
        let mask = Self::principal_mask(&envelope.principal);
        if envelope.wrapped_key.len() != 32 {
            return Err(SodlError::Crypto(format!(
                "invalid wrapped key length: {}",
                envelope.wrapped_key.len()
            )));
        }
        let mut key = vec![0u8; 32];
        for i in 0..32 {
            key[i] = envelope.wrapped_key[i] ^ mask[i];
        }
        Ok(key)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sodl_core::new_origin_id;

    #[test]
    fn ensure_key_is_deterministic() {
        let km = InMemoryKeyManager::new([0x42; 32]);
        let oid = new_origin_id();
        let k1 = km.ensure_origin_key(oid).unwrap();
        let k2 = km.ensure_origin_key(oid).unwrap();
        assert_eq!(k1.0, k2.0);
    }

    #[test]
    fn different_origins_different_key_refs() {
        let km = InMemoryKeyManager::new([0x42; 32]);
        let o1 = new_origin_id();
        let o2 = new_origin_id();
        let k1 = km.ensure_origin_key(o1).unwrap();
        let k2 = km.ensure_origin_key(o2).unwrap();
        assert_ne!(k1.0, k2.0);
    }

    #[test]
    fn wrap_unwrap_roundtrip() {
        let km = InMemoryKeyManager::new([0x42; 32]);
        let oid = new_origin_id();
        let principal = PrincipalId("user:alice".into());

        let envelope = km.wrap_origin_key(oid, &principal).unwrap();
        let raw_key = km.unwrap_origin_key(&envelope).unwrap();

        // The unwrapped key should be the derived origin key.
        let expected = km.derive_key(oid);
        assert_eq!(raw_key, expected.to_vec());
    }

    #[test]
    fn different_principals_different_envelopes() {
        let km = InMemoryKeyManager::new([0x42; 32]);
        let oid = new_origin_id();
        let alice = PrincipalId("user:alice".into());
        let bob = PrincipalId("user:bob".into());

        let env_a = km.wrap_origin_key(oid, &alice).unwrap();
        let env_b = km.wrap_origin_key(oid, &bob).unwrap();

        assert_ne!(
            env_a.wrapped_key, env_b.wrapped_key,
            "different principals must get different wrapped keys"
        );

        // But both unwrap to the same origin key.
        let key_a = km.unwrap_origin_key(&env_a).unwrap();
        let key_b = km.unwrap_origin_key(&env_b).unwrap();
        assert_eq!(key_a, key_b);
    }

    #[test]
    fn debug_redacts_master_key() {
        let km = InMemoryKeyManager::new([0x42; 32]);
        let dbg = format!("{km:?}");
        assert!(dbg.contains("REDACTED"));
    }
}
