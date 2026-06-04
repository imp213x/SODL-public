//! Demonstrates replica-aware durability auditing for pinned content.
//!
//! Run:
//!   cargo run -p sodl-service --example replica_audit_demo

use bytes::Bytes;
use sodl_cas::{BlobStore, HashAlg, MemBlobStore};
use sodl_core::{Capability, Durability, MediaKind, PrincipalId};
use sodl_crypto::NullCrypto;
use sodl_gc::ReplicaAuditor;
use sodl_policy::{AccessPolicy, OriginPolicy, RetentionPolicy};
use sodl_replica::ReplicaStore;
use sodl_replica::{MemReplicaStore, ReplicaRecord, ReplicaState};
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
            durability: Durability::Durable,
            ttl_seconds: None,
            min_replicas: Some(2),
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
        bytes: Bytes::from_static(b"replica audit"),
    })?;

    let _pin = svc.pin_origin(PrincipalId("user:a".into()), up.origin_id, 2)?;

    let replicas = MemReplicaStore::new();
    replicas.upsert_replica(ReplicaRecord {
        blob_id: up.blob_id.clone(),
        node_id: "store-a".into(),
        state: ReplicaState::Healthy,
        last_seen: time::OffsetDateTime::now_utc(),
    })?;

    let auditor = ReplicaAuditor {
        origin_registry: &origin_registry,
        policies: &policy_store,
        pins: &pin_store,
        replicas: &replicas,
        stale_seconds: 60,
        now: time::OffsetDateTime::now_utc(),
    };

    let plan1 = auditor.audit_origin(up.origin_id)?;
    println!("missing replicas (expected 1): {}", plan1.items.len());
    if let Some(item) = plan1.items.first() {
        println!(
            "blob {} required={} healthy={} missing={}",
            item.blob_id.0, item.required, item.healthy, item.missing
        );
    }

    replicas.upsert_replica(ReplicaRecord {
        blob_id: up.blob_id.clone(),
        node_id: "store-b".into(),
        state: ReplicaState::Healthy,
        last_seen: time::OffsetDateTime::now_utc(),
    })?;

    let plan2 = auditor.audit_origin(up.origin_id)?;
    println!(
        "missing replicas after adding second (expected 0): {}",
        plan2.items.len()
    );

    println!("blob exists in store? {}", durable.has(&up.blob_id)?);
    Ok(())
}
