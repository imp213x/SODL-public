//! Demonstrates signed share proofs (requires `--features sodl-proof/ed25519`).
//!
//! Run:
//!   cargo run -p sodl-service --example signed_share_demo --features sodl-proof/ed25519
//!
//! Note: this is a dev/demo signer. Do not hardcode seeds in production.

use bytes::Bytes;
use sodl_cas::{HashAlg, MemBlobStore};
use sodl_core::{Capability, MediaKind, PrincipalId};
use sodl_crypto::NullCrypto;
use sodl_proof::ProofSigner;
use sodl_service::*;
use sodl_store::EncryptedCas;

#[cfg(feature = "ed25519")]
use sodl_proof::ed25519::Ed25519Signer;

fn main() -> sodl_core::Result<()> {
    let index = MemIndex::new();
    let origin_registry = MemOriginRegistry::new();
    let policy_store = MemPolicyStore::new();
    let pin_store = MemPinStore::new();
    let derivations = MemDerivationStore::new();
    let shares = MemShareStore::new();

    let durable = MemBlobStore::new();
    let crypto = NullCrypto::default();
    let enc = EncryptedCas::new(&durable, &crypto, HashAlg::Blake3);

    #[cfg(feature = "ed25519")]
    let signer = Ed25519Signer::from_seed("ed25519:dev-1", [7u8; 32]);

    #[cfg(feature = "ed25519")]
    let signer_ref: Option<&dyn ProofSigner> = Some(&signer);

    #[cfg(not(feature = "ed25519"))]
    let signer_ref: Option<&dyn ProofSigner> = None;

    let svc = SodlService {
        index: &index,
        lineage: &index,
        provenance: &index,
        origin_registry: &origin_registry,
        policy_store: &policy_store,
        pin_store: &pin_store,
        derivations: &derivations,
        shares: &shares,
        enc_cas: enc,
        crypto: &crypto,
        proof_signer: signer_ref,
        chunk_config: None,
    };

    let up = svc.upload(UploadRequest {
        owner: PrincipalId("user:a".into()),
        media_kind: MediaKind::Binary,
        mime: Some("application/octet-stream".into()),
        durability_policy: sodl_policy::OriginPolicy {
            origin_id: sodl_core::new_origin_id(),
            retention: sodl_policy::RetentionPolicy {
                durability: sodl_core::Durability::Durable,
                ttl_seconds: None,
                min_replicas: Some(1),
            },
            access: sodl_policy::AccessPolicy {
                default_caps: vec![Capability::Read],
                allow_reshare: true,
                allow_derivation: true,
            },
        },
        bytes: Bytes::from_static(b"signed share demo"),
    })?;

    let share_id = svc.share(
        PrincipalId("user:a".into()),
        PrincipalId("user:b".into()),
        up.origin_id,
        vec![Capability::Read],
    )?;

    let rec = shares.get(&share_id)?;
    println!("\n--- Share Info ---");
    println!("share_id: {}", rec.share_id.0);

    let current = svc.lineage_proof(rec.origin_id)?;
    println!("\n--- Digest Comparison ---");
    println!("current digest:   {}", current.digest);
    println!("record  digest:   {}", rec.lineage_proof_digest);
    println!(
        "digests match:    {}",
        current.digest == rec.lineage_proof_digest
    );

    println!("\n--- Signature Info ---");
    println!("key_id: {:?}", rec.lineage_proof_key_id);
    // println!("sig_b64: {:?}", rec.lineage_proof_sig_b64);

    println!("\n--- Verification ---");
    let ok = svc.verify_share_proof(&rec)?;
    println!("verify_share_proof: {}", ok);

    Ok(())
}
