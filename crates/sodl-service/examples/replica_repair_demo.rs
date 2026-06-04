//! Demonstrates executing a durability repair plan by copying blob bytes to missing nodes.
//!
//! Run:
//!   cargo run -p sodl-service --example replica_repair_demo

use bytes::Bytes;
use sodl_cas::{HashAlg, MemBlobStore};
use sodl_core::{Capability, Durability, MediaKind, PrincipalId};
use sodl_crypto::NullCrypto;
use sodl_gc::ReplicaAuditor;
use sodl_policy::{AccessPolicy, OriginPolicy, RetentionPolicy};
use sodl_replica::{
    MemReplicaStore, MemStoreMesh, ReplicaExecutor, ReplicaRecord, ReplicaState, ReplicaStore,
    StoreMesh,
};
use sodl_service::*;
use sodl_store::EncryptedCas;

fn main() -> sodl_core::Result<()> {
    let index = MemIndex::new();
    let origin_registry = MemOriginRegistry::new();
    let policy_store = MemPolicyStore::new();
    let pin_store = MemPinStore::new();
    let derivations = MemDerivationStore::new();
    let shares = MemShareStore::new();

    // Two node stores
    let store_a = MemBlobStore::new();
    let store_b = MemBlobStore::new();

    // Service uses store_a as its durable anchor for this demo.
    let crypto = NullCrypto::default();
    let enc = EncryptedCas::new(&store_a, &crypto, HashAlg::Blake3);

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
        bytes: Bytes::from_static(b"repair me"),
    })?;

    let _pin = svc.pin_origin(PrincipalId("user:a".into()), up.origin_id, 2)?;

    // Replica registry says only store-a is healthy initially
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

    let plan = auditor.audit_origin(up.origin_id)?;
    println!("repair items before execute: {}", plan.items.len());
    if let Some(item) = plan.items.first() {
        println!("missing={} for blob {}", item.missing, item.blob_id.0);
    }

    // Mesh: both stores accessible
    let mesh = MemStoreMesh::new();
    mesh.add_node("store-a", store_a)?;
    mesh.add_node("store-b", store_b)?;

    let exec = ReplicaExecutor {
        mesh: &mesh,
        replicas: &replicas,
    };
    exec.execute(plan)?;

    // Re-audit should be satisfied
    let plan2 = auditor.audit_origin(up.origin_id)?;
    println!(
        "repair items after execute (expected 0): {}",
        plan2.items.len()
    );

    // Verify bytes exist on store-b via mesh
    let sb = mesh.store("store-b")?;
    println!("store-b has blob? {}", sb.has(&up.blob_id)?);

    Ok(())
}
