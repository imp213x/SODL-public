//! Demonstrates refcounting + release + GC eligibility.
//!
//! Run:
//!   cargo run -p sodl-service --example gc_demo

use bytes::Bytes;
use sodl_cas::{HashAlg, MemBlobStore};
use sodl_core::{Capability, Durability, MediaKind, PrincipalId};
use sodl_crypto::NullCrypto;
use sodl_index::RefCounter;
use sodl_manifest::DerivationKind;
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
        origin_id: sodl_core::new_origin_id(),
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
        bytes: Bytes::from_static(b"gc demo"),
    })?;

    let share_id = svc.share(
        PrincipalId("user:a".into()),
        PrincipalId("user:b".into()),
        up.origin_id,
        vec![Capability::Read],
    )?;
    let pin_id = svc.pin_origin(PrincipalId("user:a".into()), up.origin_id, 1)?;
    let _drv = svc.derive(
        up.origin_id,
        DerivationKind::Transform {
            description: "noop".into(),
        },
        MediaKind::Binary,
    )?;

    println!(
        "refcount after share+pin+derive: {}",
        index.get_origin(up.origin_id)?
    );
    println!("gc candidate? {}", svc.is_gc_candidate(up.origin_id)?);

    svc.release_share(&share_id)?;
    svc.unpin(&pin_id)?;
    println!(
        "refcount after release/unpin: {}",
        index.get_origin(up.origin_id)?
    );
    println!("gc candidate? {}", svc.is_gc_candidate(up.origin_id)?);

    Ok(())
}
