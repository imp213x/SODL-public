//! Production-grade AEAD encryption for SODL.
//!
//! Uses **XChaCha20-Poly1305** — a widely-audited AEAD cipher with:
//! - 256-bit keys
//! - 192-bit (24-byte) extended nonces (safe for random generation)
//! - 128-bit authentication tags
//! - No hardware-specific requirements (unlike AES-GCM)
//!
//! # Key derivation
//!
//! A single 32-byte **master key** is expanded per-origin via HKDF-SHA256:
//!
//! ```text
//! origin_key = HKDF-Expand(master_key, info = "sodl-origin-v1:" || origin_id)
//! ```
//!
//! # Nonce strategy (deterministic / SIV-like)
//!
//! To preserve CAS dedup within an origin, the nonce is derived deterministically
//! from the per-origin key and the plaintext:
//!
//! ```text
//! nonce = BLAKE3(origin_key || plaintext)[0..24]
//! ```
//!
//! Same origin + same plaintext → same nonce → same ciphertext → same BlobId.
//! Different plaintext → different nonce (with overwhelming probability).
//!
//! > **Privacy note**: Deterministic encryption reveals whether two blobs are
//! > identical within the same origin. This is an intentional trade-off for
//! > storage efficiency via content-addressing.
//!
//! # Ciphertext format (versioned)
//!
//! ```text
//! [1 byte version = 0x01][24 bytes nonce][N + 16 bytes ciphertext || tag]
//! ```
//!
//! Total overhead: **41 bytes** per encrypted blob.

use bytes::Bytes;
use chacha20poly1305::aead::{Aead, KeyInit};
use chacha20poly1305::XChaCha20Poly1305;
use hkdf::Hkdf;
use sha2::Sha256;

use sodl_core::{OriginId, Result, SodlError};

use crate::{Decryptor, Encryptor};

/// Current ciphertext envelope version.
const ENVELOPE_VERSION: u8 = 0x01;

/// XChaCha20-Poly1305 nonce length (24 bytes).
const NONCE_LEN: usize = 24;

/// Minimum ciphertext length: version(1) + nonce(24) + tag(16).
const MIN_CIPHERTEXT_LEN: usize = 1 + NONCE_LEN + 16;

/// Production AEAD crypto provider backed by XChaCha20-Poly1305.
///
/// Construct with a 32-byte master key. Per-origin keys are derived
/// automatically via HKDF.
///
/// # Example
///
/// ```rust,ignore
/// use sodl_crypto::AeadCrypto;
///
/// let master_key = [0x42u8; 32]; // in practice, load from KMS / env
/// let crypto = AeadCrypto::new(master_key);
/// ```
#[derive(Clone)]
pub struct AeadCrypto {
    master_key: [u8; 32],
}

impl AeadCrypto {
    /// Create a new AEAD crypto provider from a 32-byte master key.
    pub fn new(master_key: [u8; 32]) -> Self {
        Self { master_key }
    }

    /// Create from a hex-encoded master key string (64 hex chars).
    pub fn from_hex(hex: &str) -> Result<Self> {
        let bytes = hex_to_bytes(hex)?;
        if bytes.len() != 32 {
            return Err(SodlError::Invalid(format!(
                "master key must be 32 bytes, got {}",
                bytes.len()
            )));
        }
        let mut key = [0u8; 32];
        key.copy_from_slice(&bytes);
        Ok(Self::new(key))
    }

    /// Generate a random master key (requires std randomness).
    ///
    /// Useful for tests or first-run key generation.
    pub fn generate() -> Self {
        use chacha20poly1305::aead::OsRng;
        use chacha20poly1305::KeyInit;
        let key = XChaCha20Poly1305::generate_key(&mut OsRng);
        let mut master = [0u8; 32];
        master.copy_from_slice(&key);
        Self::new(master)
    }

    /// Return the master key as a hex string (for persistence / display).
    pub fn master_key_hex(&self) -> String {
        bytes_to_hex(&self.master_key)
    }

    /// Derive a 32-byte per-origin key via HKDF-SHA256.
    fn derive_origin_key(&self, origin_id: OriginId) -> [u8; 32] {
        let hk = Hkdf::<Sha256>::new(None, &self.master_key);
        let info = format!("sodl-origin-v1:{}", origin_id.0);
        let mut okm = [0u8; 32];
        hk.expand(info.as_bytes(), &mut okm)
            .expect("HKDF expand should not fail for 32-byte output");
        okm
    }

    /// Derive a deterministic 24-byte nonce from origin key + plaintext
    /// using BLAKE3.
    fn deterministic_nonce(origin_key: &[u8; 32], plaintext: &[u8]) -> [u8; NONCE_LEN] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(origin_key);
        hasher.update(plaintext);
        let hash = hasher.finalize();
        let mut nonce = [0u8; NONCE_LEN];
        nonce.copy_from_slice(&hash.as_bytes()[..NONCE_LEN]);
        nonce
    }
}

impl std::fmt::Debug for AeadCrypto {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AeadCrypto")
            .field("master_key", &"[REDACTED]")
            .finish()
    }
}

impl Encryptor for AeadCrypto {
    fn encrypt_for_origin(&self, origin_id: OriginId, plaintext: Bytes) -> Result<Bytes> {
        let origin_key = self.derive_origin_key(origin_id);
        let nonce_bytes = Self::deterministic_nonce(&origin_key, &plaintext);

        let cipher = XChaCha20Poly1305::new((&origin_key).into());
        let nonce = chacha20poly1305::XNonce::from_slice(&nonce_bytes);

        let ciphertext = cipher
            .encrypt(nonce, plaintext.as_ref())
            .map_err(|e| SodlError::Crypto(format!("AEAD encrypt failed: {e}")))?;

        // Build envelope: [version][nonce][ciphertext||tag]
        let mut envelope = Vec::with_capacity(1 + NONCE_LEN + ciphertext.len());
        envelope.push(ENVELOPE_VERSION);
        envelope.extend_from_slice(&nonce_bytes);
        envelope.extend_from_slice(&ciphertext);

        Ok(Bytes::from(envelope))
    }
}

impl Decryptor for AeadCrypto {
    fn decrypt_for_origin(&self, origin_id: OriginId, ciphertext: Bytes) -> Result<Bytes> {
        if ciphertext.len() < MIN_CIPHERTEXT_LEN {
            return Err(SodlError::Crypto(format!(
                "ciphertext too short: {} bytes (min {})",
                ciphertext.len(),
                MIN_CIPHERTEXT_LEN
            )));
        }

        let version = ciphertext[0];
        if version != ENVELOPE_VERSION {
            return Err(SodlError::Crypto(format!(
                "unsupported ciphertext version: 0x{version:02x} (expected 0x{ENVELOPE_VERSION:02x})"
            )));
        }

        let nonce = chacha20poly1305::XNonce::from_slice(&ciphertext[1..1 + NONCE_LEN]);
        let ct = &ciphertext[1 + NONCE_LEN..];

        let origin_key = self.derive_origin_key(origin_id);
        let cipher = XChaCha20Poly1305::new((&origin_key).into());

        let plaintext = cipher.decrypt(nonce, ct).map_err(|e| {
            SodlError::Crypto(format!("AEAD decrypt failed (wrong key or tampered): {e}"))
        })?;

        Ok(Bytes::from(plaintext))
    }
}

// ---------------------------------------------------------------------------
// Hex helpers
// ---------------------------------------------------------------------------

fn hex_to_bytes(hex: &str) -> Result<Vec<u8>> {
    if hex.len() % 2 != 0 {
        return Err(SodlError::Invalid("hex string length must be even".into()));
    }
    (0..hex.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&hex[i..i + 2], 16)
                .map_err(|e| SodlError::Invalid(format!("invalid hex: {e}")))
        })
        .collect()
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sodl_core::new_origin_id;

    #[test]
    fn roundtrip_encrypt_decrypt() {
        let crypto = AeadCrypto::generate();
        let origin = new_origin_id();
        let plaintext = Bytes::from_static(b"hello, SODL encryption!");

        let ct = crypto
            .encrypt_for_origin(origin, plaintext.clone())
            .unwrap();
        assert_ne!(ct, plaintext, "ciphertext must differ from plaintext");
        assert!(ct.len() >= plaintext.len() + MIN_CIPHERTEXT_LEN - 16);

        let pt = crypto.decrypt_for_origin(origin, ct).unwrap();
        assert_eq!(pt, plaintext);
    }

    #[test]
    fn deterministic_same_origin_same_plaintext() {
        let crypto = AeadCrypto::generate();
        let origin = new_origin_id();
        let plaintext = Bytes::from_static(b"dedup me");

        let ct1 = crypto
            .encrypt_for_origin(origin, plaintext.clone())
            .unwrap();
        let ct2 = crypto
            .encrypt_for_origin(origin, plaintext.clone())
            .unwrap();
        assert_eq!(
            ct1, ct2,
            "same origin + same plaintext must produce identical ciphertext"
        );
    }

    #[test]
    fn different_plaintext_different_ciphertext() {
        let crypto = AeadCrypto::generate();
        let origin = new_origin_id();

        let ct1 = crypto
            .encrypt_for_origin(origin, Bytes::from_static(b"aaa"))
            .unwrap();
        let ct2 = crypto
            .encrypt_for_origin(origin, Bytes::from_static(b"bbb"))
            .unwrap();
        assert_ne!(ct1, ct2);
    }

    #[test]
    fn cross_origin_isolation() {
        let crypto = AeadCrypto::generate();
        let o1 = new_origin_id();
        let o2 = new_origin_id();
        let plaintext = Bytes::from_static(b"same content");

        let ct1 = crypto.encrypt_for_origin(o1, plaintext.clone()).unwrap();
        let ct2 = crypto.encrypt_for_origin(o2, plaintext.clone()).unwrap();
        assert_ne!(
            ct1, ct2,
            "different origins must produce different ciphertext"
        );

        // Decrypting with wrong origin must fail.
        let err = crypto.decrypt_for_origin(o2, ct1);
        assert!(err.is_err(), "decrypt with wrong origin key must fail");
    }

    #[test]
    fn envelope_version_byte() {
        let crypto = AeadCrypto::generate();
        let origin = new_origin_id();
        let ct = crypto
            .encrypt_for_origin(origin, Bytes::from_static(b"v"))
            .unwrap();
        assert_eq!(ct[0], ENVELOPE_VERSION);
    }

    #[test]
    fn tampered_ciphertext_rejected() {
        let crypto = AeadCrypto::generate();
        let origin = new_origin_id();
        let ct = crypto
            .encrypt_for_origin(origin, Bytes::from_static(b"integrity"))
            .unwrap();

        // Flip a byte in the ciphertext body.
        let mut tampered = ct.to_vec();
        let idx = 1 + NONCE_LEN + 2; // inside the ciphertext
        tampered[idx] ^= 0xFF;

        let err = crypto.decrypt_for_origin(origin, Bytes::from(tampered));
        assert!(err.is_err(), "tampered ciphertext must be rejected");
    }

    #[test]
    fn bad_version_rejected() {
        let crypto = AeadCrypto::generate();
        let origin = new_origin_id();
        let ct = crypto
            .encrypt_for_origin(origin, Bytes::from_static(b"ver"))
            .unwrap();

        let mut bad = ct.to_vec();
        bad[0] = 0xFF; // invalid version
        let err = crypto.decrypt_for_origin(origin, Bytes::from(bad));
        assert!(err.is_err());
    }

    #[test]
    fn too_short_ciphertext_rejected() {
        let crypto = AeadCrypto::generate();
        let origin = new_origin_id();
        let short = Bytes::from_static(&[0x01; 10]); // way too short
        assert!(crypto.decrypt_for_origin(origin, short).is_err());
    }

    #[test]
    fn from_hex_roundtrip() {
        let original = AeadCrypto::generate();
        let hex = original.master_key_hex();
        let restored = AeadCrypto::from_hex(&hex).unwrap();
        assert_eq!(original.master_key, restored.master_key);
    }

    #[test]
    fn from_hex_bad_length_rejected() {
        assert!(AeadCrypto::from_hex("aabb").is_err());
        assert!(AeadCrypto::from_hex("not-hex").is_err());
    }

    #[test]
    fn empty_plaintext_roundtrip() {
        let crypto = AeadCrypto::generate();
        let origin = new_origin_id();
        let ct = crypto.encrypt_for_origin(origin, Bytes::new()).unwrap();
        let pt = crypto.decrypt_for_origin(origin, ct).unwrap();
        assert!(pt.is_empty());
    }

    #[test]
    fn large_payload_roundtrip() {
        let crypto = AeadCrypto::generate();
        let origin = new_origin_id();
        let payload = Bytes::from(vec![0xABu8; 1024 * 1024]); // 1 MiB
        let ct = crypto.encrypt_for_origin(origin, payload.clone()).unwrap();
        let pt = crypto.decrypt_for_origin(origin, ct).unwrap();
        assert_eq!(pt, payload);
    }

    #[test]
    fn debug_redacts_key() {
        let crypto = AeadCrypto::generate();
        let dbg = format!("{crypto:?}");
        assert!(dbg.contains("REDACTED"));
        assert!(!dbg.contains(&crypto.master_key_hex()));
    }
}
