//! SODL Service demo
//!
//! Run:
//!   cargo run -p sodl-service --example service_demo

use bytes::Bytes;
use sodl_cas::{HashAlg, MemBlobStore};
use sodl_core::{Capability, Durability, MediaKind, PrincipalId};
use sodl_crypto::NullCrypto;
use sodl_policy::{AccessPolicy, OriginPolicy, RetentionPolicy};
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

    let policy = OriginPolicy {
        origin_id: sodl_core::new_origin_id(), // placeholder; service will allocate real origin id
        retention: RetentionPolicy {
            durability: Durability::BestEffort,
            ttl_seconds: None,
            min_replicas: Some(1),
        },
        access: AccessPolicy {
            default_caps: vec![Capability::Read],
            allow_reshare: true,
            allow_derivation: true,
        },
    };

    let up = svc.upload(UploadRequest {
        owner: PrincipalId("user:a".into()),
        media_kind: MediaKind::Binary,
        mime: Some("application/octet-stream".into()),
        durability_policy: policy,
        bytes: Bytes::from_static(b"hello from service"),
    })?;

    let share_id = svc.share(
        PrincipalId("user:a".into()),
        PrincipalId("user:b".into()),
        up.origin_id,
        vec![Capability::Read, Capability::Reshare],
    )?;
    let pin_id = svc.pin_origin(PrincipalId("user:a".into()), up.origin_id, 1)?;

    println!("Uploaded origin: {:?}", up.origin_id.0);
    println!("Stored blob: {}", up.blob_id.0);
    println!("Share id: {}", share_id.0);
    println!("Pin id: {}", pin_id);

    Ok(())
}
