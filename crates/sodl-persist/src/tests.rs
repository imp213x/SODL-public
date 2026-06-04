//! Integration tests for SqliteStore — validates every trait implementation.

use crate::SqliteStore;
use sodl_core::*;

fn new_store() -> SqliteStore {
    SqliteStore::open_memory().unwrap()
}

// -----------------------------------------------------------------------
// OriginRegistry
// -----------------------------------------------------------------------
#[test]
fn origin_roundtrip() {
    use sodl_origin::{OriginRecord, OriginRegistry};

    let store = new_store();
    let oid = new_origin_id();
    let rec = OriginRecord::new(oid, MediaKind::Document, Durability::Durable);

    store.create_origin(rec.clone()).unwrap();
    let got = store.get_origin(oid).unwrap();
    assert_eq!(got.origin_id, oid);
    assert!(matches!(got.media_kind, MediaKind::Document));

    // Duplicate create → Conflict
    let err = store.create_origin(rec).unwrap_err();
    assert!(matches!(err, SodlError::Conflict));

    // Update
    let mut updated = store.get_origin(oid).unwrap();
    updated.tombstoned_at = Some(time::OffsetDateTime::now_utc());
    store.update_origin(updated).unwrap();
    let got2 = store.get_origin(oid).unwrap();
    assert!(got2.tombstoned_at.is_some());

    // Delete
    store.delete_origin(oid).unwrap();
    assert!(matches!(store.get_origin(oid), Err(SodlError::NotFound)));
}

// -----------------------------------------------------------------------
// PolicyStore
// -----------------------------------------------------------------------
#[test]
fn policy_roundtrip() {
    use sodl_policy::{AccessPolicy, OriginPolicy, PolicyStore, RetentionPolicy};

    let store = new_store();
    let oid = new_origin_id();
    let policy = OriginPolicy {
        origin_id: oid,
        retention: RetentionPolicy {
            durability: Durability::Ephemeral,
            ttl_seconds: Some(300),
            min_replicas: None,
        },
        access: AccessPolicy {
            default_caps: vec![Capability::Read],
            allow_reshare: true,
            allow_derivation: false,
        },
    };

    store.put_origin_policy(policy).unwrap();
    let got = store.get_origin_policy(oid).unwrap();
    assert_eq!(got.retention.ttl_seconds, Some(300));
    assert!(!got.access.allow_derivation);

    // Not found
    assert!(matches!(
        store.get_origin_policy(new_origin_id()),
        Err(SodlError::NotFound)
    ));
}

// -----------------------------------------------------------------------
// PinStore
// -----------------------------------------------------------------------
#[test]
fn pin_lifecycle() {
    use sodl_policy::{PinRecord, PinState, PinStore, PinTarget};

    let store = new_store();
    let oid = new_origin_id();
    let pin = PinRecord {
        pin_id: "pin_1".into(),
        target: PinTarget::Origin { origin_id: oid },
        requested_by: PrincipalId("user:a".into()),
        created_at: time::OffsetDateTime::now_utc(),
        state: PinState::Pending,
        min_replicas: Some(2),
        required_zones: vec![],
    };

    store.create_pin(pin).unwrap();
    let got = store.get_pin("pin_1").unwrap();
    assert!(matches!(got.state, PinState::Pending));

    // Release
    store.release_pin("pin_1").unwrap();
    let got2 = store.get_pin("pin_1").unwrap();
    assert!(matches!(got2.state, PinState::Released));

    // List by origin
    let list = store.list_pins_for_origin(oid).unwrap();
    assert_eq!(list.len(), 1);
}

// -----------------------------------------------------------------------
// RefCounter + ScanIndex
// -----------------------------------------------------------------------
#[test]
fn refcounts() {
    use sodl_index::{RefCounter, RefKind, ScanIndex};

    let store = new_store();
    let oid = new_origin_id();
    let bid = BlobId("blake3:aabbcc".into());

    assert_eq!(store.get_origin(oid).unwrap(), 0);
    store
        .inc_origin(
            oid,
            RefKind::Pin {
                pin_id: "p1".into(),
            },
        )
        .unwrap();
    store
        .inc_origin(
            oid,
            RefKind::Pin {
                pin_id: "p2".into(),
            },
        )
        .unwrap();
    assert_eq!(sodl_index::RefCounter::get_origin(&store, oid).unwrap(), 2);

    store
        .dec_origin(
            oid,
            RefKind::Pin {
                pin_id: "p1".into(),
            },
        )
        .unwrap();
    assert_eq!(sodl_index::RefCounter::get_origin(&store, oid).unwrap(), 1);

    // Floor at 0
    store
        .dec_origin(
            oid,
            RefKind::Pin {
                pin_id: "p2".into(),
            },
        )
        .unwrap();
    store
        .dec_origin(
            oid,
            RefKind::Pin {
                pin_id: "p3".into(),
            },
        )
        .unwrap();
    assert_eq!(sodl_index::RefCounter::get_origin(&store, oid).unwrap(), 0);

    // Blob refcounts
    store
        .inc_blob(
            &bid,
            RefKind::OriginRepresentation {
                name: "source".into(),
            },
        )
        .unwrap();
    assert_eq!(store.get_blob(&bid).unwrap(), 1);

    // ScanIndex
    store
        .inc_origin(
            oid,
            RefKind::Pin {
                pin_id: "p4".into(),
            },
        )
        .unwrap();
    let origins = store.list_origins().unwrap();
    assert!(origins.contains(&oid));
    let blobs = store.list_blobs().unwrap();
    assert!(blobs.iter().any(|b| b.0 == bid.0));
}

// -----------------------------------------------------------------------
// LineageStore
// -----------------------------------------------------------------------
#[test]
fn lineage_edges() {
    use sodl_index::{LineageEdge, LineageStore, RefKind};

    let store = new_store();
    let oid = new_origin_id();

    let e1 = LineageEdge {
        edge_id: "e1".into(),
        origin_id: oid,
        blob_id: Some(BlobId("blake3:aaa".into())),
        kind: RefKind::OriginRepresentation {
            name: "source".into(),
        },
        created_at: time::OffsetDateTime::now_utc(),
    };

    store.add_edge(e1).unwrap();

    let edges = store.list_edges_for_origin(oid).unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].edge_id, "e1");

    // Different origin → empty
    let other = new_origin_id();
    assert!(store.list_edges_for_origin(other).unwrap().is_empty());
}

// -----------------------------------------------------------------------
// DerivationStore
// -----------------------------------------------------------------------
#[test]
fn derivation_roundtrip() {
    use sodl_manifest::{DerivationKind, DerivationManifest};
    use sodl_service::DerivationStore;

    let store = new_store();
    let oid = new_origin_id();
    let did = DerivationId("drv:1".into());
    let m = DerivationManifest::new(
        oid,
        did.clone(),
        MediaKind::Binary,
        DerivationKind::Transform {
            description: "test".into(),
        },
    );

    store.put(m).unwrap();
    let got = store.get(oid, &did).unwrap();
    assert_eq!(got.derivation_id, did);

    let list = store.list_for_origin(oid).unwrap();
    assert_eq!(list.len(), 1);
}

// -----------------------------------------------------------------------
// ShareStore
// -----------------------------------------------------------------------
#[test]
fn share_roundtrip() {
    use sodl_manifest::ShareRecord;
    use sodl_service::ShareStore;

    let store = new_store();
    let oid = new_origin_id();
    let sid = ShareId("share:1".into());
    let s = ShareRecord {
        schema: SODL_SCHEMA_VERSION.to_string(),
        share_id: sid.clone(),
        origin_id: oid,
        derivation_id: None,
        from_principal: PrincipalId("user:a".into()),
        to_principal: PrincipalId("user:b".into()),
        created_at: time::OffsetDateTime::now_utc(),
        capabilities: vec![Capability::Read],
        lineage_proof_digest: "test_digest".into(),
        lineage_proof_created_at: time::OffsetDateTime::now_utc(),
        lineage_proof_key_id: None,
        lineage_proof_sig_b64: None,
    };

    store.put(s).unwrap();
    let got = store.get(&sid).unwrap();
    assert_eq!(got.share_id, sid);
    assert_eq!(got.origin_id, oid);

    let list = store.list_for_origin(oid).unwrap();
    assert_eq!(list.len(), 1);
}

// -----------------------------------------------------------------------
// Persistent file — survives reopen
// -----------------------------------------------------------------------
#[test]
fn persistent_survives_reopen() {
    use sodl_origin::{OriginRecord, OriginRegistry};

    let tmp = tempfile::tempdir().unwrap();
    let db_path = tmp.path().join("sodl.db");

    let oid = new_origin_id();

    // Open, write, close
    {
        let store = SqliteStore::open(&db_path).unwrap();
        let rec = OriginRecord::new(oid, MediaKind::Binary, Durability::Durable);
        store.create_origin(rec).unwrap();
    }

    // Reopen and read
    {
        let store = SqliteStore::open(&db_path).unwrap();
        let got = store.get_origin(oid).unwrap();
        assert_eq!(got.origin_id, oid);
    }
}
