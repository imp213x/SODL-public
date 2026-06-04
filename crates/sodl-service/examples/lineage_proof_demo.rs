//! Demonstrates computing a lineage proof for an origin.
//
// Run:
//!   cargo run -p sodl-service --example lineage_proof_demo

use bytes::Bytes;
use sodl_cas::{HashAlg, MemBlobStore};
use sodl_core::{Capability, MediaKind, PrincipalId};
use sodl_crypto::NullCrypto;
use sodl_service::*;
use sodl_store::EncryptedCas;

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
        proof_signer: None,
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
        bytes: Bytes::from_static(b"hello lineage proof"),
    })?;

    let _share_id = svc.share(
        PrincipalId("user:a".into()),
        PrincipalId("user:b".into()),
        up.origin_id,
        vec![Capability::Read],
    )?;

    let proof = svc.lineage_proof(up.origin_id)?;
    println!("origin: {}", proof.origin_id.0);
    println!("digest: {}", proof.digest);
    println!("created_at: {}", proof.created_at);

    Ok(())
}
