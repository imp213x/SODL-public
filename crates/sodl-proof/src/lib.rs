use blake3::Hasher;
use serde::{Deserialize, Serialize};
use sodl_core::{OriginId, Result};
use sodl_index::{LineageEdge, RefKind};

/// A deterministic, unsigned proof of an origin's lineage state at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LineageProof {
    pub origin_id: OriginId,
    /// Blake3 hex digest over canonicalized lineage edges.
    pub digest: String,
    pub created_at: time::OffsetDateTime,
}

/// Signs and verifies lineage proof digests.
///
/// This is deliberately digest-level (string) to keep it stable across proof versions.
pub trait ProofSigner: Send + Sync {
    /// Identifier for the key used (e.g. "ed25519:dev-1").
    fn key_id(&self) -> &str;

    /// Sign a hex digest (e.g. Blake3 hex) and return a base64 signature.
    fn sign_digest_b64(&self, digest_hex: &str) -> Result<String>;

    /// Verify a signature (base64) over a hex digest.
    fn verify_digest_b64(&self, digest_hex: &str, sig_b64: &str) -> Result<bool>;
}

#[cfg(feature = "ed25519")]
pub mod ed25519 {
    use super::*;
    use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
    use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};

    /// Ed25519 signer with an attached key id.
    pub struct Ed25519Signer {
        key_id: String,
        signing: SigningKey,
        verifying: VerifyingKey,
    }

    impl Ed25519Signer {
        /// Create from a 32-byte seed (deterministic; good for dev/tests).
        pub fn from_seed(key_id: impl Into<String>, seed32: [u8; 32]) -> Self {
            let signing = SigningKey::from_bytes(&seed32);
            let verifying = signing.verifying_key();
            Self {
                key_id: key_id.into(),
                signing,
                verifying,
            }
        }

        pub fn verifying_key(&self) -> VerifyingKey {
            self.verifying
        }
    }

    impl ProofSigner for Ed25519Signer {
        fn key_id(&self) -> &str {
            &self.key_id
        }

        fn sign_digest_b64(&self, digest_hex: &str) -> Result<String> {
            let sig: Signature = self.signing.sign(digest_hex.as_bytes());
            Ok(B64.encode(sig.to_bytes()))
        }

        fn verify_digest_b64(&self, digest_hex: &str, sig_b64: &str) -> Result<bool> {
            let bytes = B64
                .decode(sig_b64.as_bytes())
                .map_err(|e| sodl_core::SodlError::Invalid(e.to_string()))?;
            let sig = Signature::from_slice(&bytes)
                .map_err(|e| sodl_core::SodlError::Invalid(e.to_string()))?;
            Ok(self.verifying.verify(digest_hex.as_bytes(), &sig).is_ok())
        }
    }
}

/// Generate an unsigned lineage proof for an origin based on provided edges.
///
/// Canonicalization rules (v1):
/// - Sort edges by `edge_id` (lexicographic)
/// - For each edge, feed:
///   - edge_id
///   - origin_id (uuid hyphenated)
///   - blob_id (or "-")
///   - kind (stable string)
pub fn generate_proof_unsigned(
    origin_id: OriginId,
    mut edges: Vec<LineageEdge>,
    now: time::OffsetDateTime,
) -> Result<LineageProof> {
    edges.sort_by(|a, b| a.edge_id.cmp(&b.edge_id));

    let mut h = Hasher::new();

    // domain separation + versioning
    h.update(b"SODL_LINEAGE_PROOF_V1\n");
    h.update(origin_id.0.as_hyphenated().to_string().as_bytes());
    h.update(b"\n");

    for e in edges {
        h.update(e.edge_id.as_bytes());
        h.update(b"\n");

        h.update(e.origin_id.0.as_hyphenated().to_string().as_bytes());
        h.update(b"\n");

        if let Some(bid) = e.blob_id {
            h.update(bid.0.as_bytes());
        } else {
            h.update(b"-");
        }
        h.update(b"\n");

        let k = kind_string(&e.kind);
        h.update(k.as_bytes());
        h.update(b"\n");
    }

    let digest = h.finalize().to_hex().to_string();
    Ok(LineageProof {
        origin_id,
        digest,
        created_at: now,
    })
}

fn kind_string(k: &RefKind) -> String {
    match k {
        RefKind::OriginRepresentation { name } => format!("origin_rep:{}", name),
        RefKind::Share { share_id, from, to } => {
            format!("share:{}:{}:{}", share_id.0, from.0, to.0)
        }
        RefKind::Derivation { derivation_id } => format!("derivation:{}", derivation_id.0),
        RefKind::Pin { pin_id } => format!("pin:{}", pin_id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sodl_core::{new_origin_id, BlobId, DerivationId, PrincipalId, ShareId};
    use sodl_index::LineageEdge;

    #[test]
    fn proof_is_deterministic_for_same_edges() {
        let origin = new_origin_id();
        let now = time::OffsetDateTime::now_utc();

        let e1 = LineageEdge {
            edge_id: "e1".into(),
            origin_id: origin,
            blob_id: Some(BlobId("blake3:abc".into())),
            kind: RefKind::Share {
                share_id: ShareId("share:1".into()),
                from: PrincipalId("user:a".into()),
                to: PrincipalId("user:b".into()),
            },
            created_at: now,
        };

        let e2 = LineageEdge {
            edge_id: "e2".into(),
            origin_id: origin,
            blob_id: None,
            kind: RefKind::Derivation {
                derivation_id: DerivationId("der:1".into()),
            },
            created_at: now,
        };

        let p1 = generate_proof_unsigned(origin, vec![e2.clone(), e1.clone()], now).unwrap();
        let p2 = generate_proof_unsigned(origin, vec![e1, e2], now).unwrap();
        assert_eq!(p1.digest, p2.digest);
    }
}
