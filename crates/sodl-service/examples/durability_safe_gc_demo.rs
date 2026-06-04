//! Demonstrates durability-safe GC: refuses deletion when min_replicas is not satisfied.
//!
//! Run:
//!   cargo run -p sodl-service --example durability_safe_gc_demo

use bytes::Bytes;
use sodl_cas::{BlobStore, HashAlg, MemBlobStore};
use sodl_core::{Capability, Durability, MediaKind, PrincipalId};
use sodl_crypto::NullCrypto;
use sodl_gc::DurabilityGate;
use sodl_policy::{AccessPolicy, OriginPolicy, RetentionPolicy};
use sodl_replica::{MemReplicaStore, ReplicaRecord, ReplicaState, ReplicaStore};
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
            ttl_seconds: Some(1),
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
        bytes: Bytes::from_static(b"durability gate"),
    })?;

    let _pin = svc.pin_origin(PrincipalId("user:a".into()), up.origin_id, 1)?;

    let replicas = MemReplicaStore::new();
    replicas.upsert_replica(ReplicaRecord {
        blob_id: up.blob_id.clone(),
        node_id: "store-a".into(),
        state: ReplicaState::Healthy,
        last_seen: time::OffsetDateTime::now_utc(),
    })?;

    let now = time::OffsetDateTime::now_utc() + time::Duration::seconds(10);

    let gate = DurabilityGate {
        origin_registry: &origin_registry,
        policies: &policy_store,
        pins: &pin_store,
        replicas: &replicas,
        stale_seconds: 60,
        now,
    };

    let can_delete = gate.can_delete_origin_bytes(up.origin_id)?;
    println!(
        "can_delete_origin_bytes with only 1 replica (expected false): {}",
        can_delete
    );

    if can_delete {
        // In a full integration, GC would delete the blob bytes here.
        // For now, we just demonstrate the gate decision.
        println!("GC WOULD EXECUTE here (gate allowed).");
    } else {
        println!("SKIP GC execute due to durability gate");
    }

    println!(
        "blob exists after attempted gc? {}",
        durable.has(&up.blob_id)?
    );

    replicas.upsert_replica(ReplicaRecord {
        blob_id: up.blob_id.clone(),
        node_id: "store-b".into(),
        state: ReplicaState::Healthy,
        last_seen: now,
    })?;

    let can_delete2 = gate.can_delete_origin_bytes(up.origin_id)?;
    println!(
        "can_delete_origin_bytes after 2 replicas (expected true): {}",
        can_delete2
    );

    Ok(())
}
