//! Policy-aware GC executor (skeleton).
//!
//! Choice **B**:
//! - **blob deletion** is governed by blob refcount == 0
//! - origins are metadata objects; they can be tombstoned/deleted separately
//!
//! GC also considers:
//! - pins (durable intent)
//! - retention policy (TTL / durability)
//! - safety (tombstones + two-phase deletion)

use serde::{Deserialize, Serialize};
use sodl_cas::BlobStore;
use sodl_core::{BlobId, OriginId, Result};
use sodl_index::{RefCounter, ScanIndex};
use sodl_origin::OriginRegistry;
use sodl_policy::{PinStore, PolicyStore};
use sodl_replica::{RepairItem, RepairPlan, ReplicaStore};
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TombstoneKind {
    Origin,
    Blob,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tombstone {
    pub tombstone_id: String,
    pub kind: TombstoneKind,
    pub origin_id: Option<OriginId>,
    pub blob_id: Option<BlobId>,
    pub reason: String,
    pub created_at: time::OffsetDateTime,
}

pub trait TombstoneStore: Send + Sync {
    fn put(&self, t: Tombstone) -> Result<()>;
    fn get(&self, tombstone_id: &str) -> Result<Tombstone>;
    fn list_for_origin(&self, origin_id: OriginId) -> Result<Vec<Tombstone>>;
}

#[derive(Clone, Default)]
pub struct MemTombstoneStore {
    inner: Arc<RwLock<Vec<Tombstone>>>,
}

impl MemTombstoneStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl TombstoneStore for MemTombstoneStore {
    fn put(&self, t: Tombstone) -> Result<()> {
        self.inner
            .write()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?
            .push(t);
        Ok(())
    }
    fn get(&self, tombstone_id: &str) -> Result<Tombstone> {
        let r = self
            .inner
            .read()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        r.iter()
            .find(|t| t.tombstone_id == tombstone_id)
            .cloned()
            .ok_or(sodl_core::SodlError::NotFound)
    }
    fn list_for_origin(&self, origin_id: OriginId) -> Result<Vec<Tombstone>> {
        let r = self
            .inner
            .read()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        Ok(r.iter()
            .filter(|t| t.origin_id == Some(origin_id))
            .cloned()
            .collect())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GcPlan {
    pub blob_delete: Vec<BlobId>,
    pub origin_tombstone: Vec<OriginId>,
}

pub struct GcPlanner<'a> {
    pub origin_registry: &'a dyn OriginRegistry,
    /// Optional grace window after tombstoning before blob deletion.
    pub grace_seconds: Option<i64>,

    pub index: &'a dyn RefCounter,
    pub scan: &'a dyn ScanIndex,
    pub pins: &'a dyn PinStore,
    pub policies: &'a dyn PolicyStore,
    pub now: time::OffsetDateTime,
}

impl<'a> GcPlanner<'a> {
    pub fn plan(&self) -> Result<GcPlan> {
        let mut plan = GcPlan::default();

        for oid in self.scan.list_origins()? {
            let refs = self.index.get_origin(oid)?;
            if refs != 0 {
                continue;
            }
            let pins = self.pins.list_pins_for_origin(oid)?;
            let any_active = pins
                .iter()
                .any(|p| p.state != sodl_policy::PinState::Released);
            if any_active {
                continue;
            }
            let policy = self.policies.get_origin_policy(oid)?;
            // TTL enforcement (if configured): do not tombstone/purge before TTL window is satisfied.
            if let Some(ttl) = policy.retention.ttl_seconds {
                let rec = self.origin_registry.get_origin(oid)?;
                let expires_at = rec.created_at + time::Duration::seconds(ttl as i64);
                if self.now < expires_at {
                    continue;
                }
            }

            plan.origin_tombstone.push(oid);
        }

        for bid in self.scan.list_blobs()? {
            if self.index.get_blob(&bid)? == 0 {
                plan.blob_delete.push(bid);
            }
        }

        Ok(plan)
    }
}

pub struct GcExecutor<'a> {
    pub store: &'a dyn BlobStore,
    pub tombstones: &'a dyn TombstoneStore,
    pub now: time::OffsetDateTime,
}

impl<'a> GcExecutor<'a> {
    pub fn execute(&self, plan: GcPlan) -> Result<()> {
        for oid in plan.origin_tombstone {
            self.tombstones.put(Tombstone {
                tombstone_id: format!("tomb:{}", uuid::Uuid::new_v4()),
                kind: TombstoneKind::Origin,
                origin_id: Some(oid),
                blob_id: None,
                reason: "gc_origin_candidate".into(),
                created_at: self.now,
            })?;
        }

        for bid in plan.blob_delete {
            self.tombstones.put(Tombstone {
                tombstone_id: format!("tomb:{}", uuid::Uuid::new_v4()),
                kind: TombstoneKind::Blob,
                origin_id: None,
                blob_id: Some(bid.clone()),
                reason: "gc_blob_refcount_zero".into(),
                created_at: self.now,
            })?;
            let _ = self.store.delete(&bid);
        }

        Ok(())
    }
}

/// Audits active pins against min replica requirements and produces a repair plan.
///
/// This does not perform replication; it only identifies missing replicas.
pub struct ReplicaAuditor<'a> {
    pub origin_registry: &'a dyn OriginRegistry,
    pub policies: &'a dyn PolicyStore,
    pub pins: &'a dyn PinStore,
    pub replicas: &'a dyn ReplicaStore,

    /// Replicas older than this many seconds are treated as unhealthy.
    pub stale_seconds: i64,
    pub now: time::OffsetDateTime,
}

impl<'a> ReplicaAuditor<'a> {
    pub fn audit_origin(&self, origin_id: OriginId) -> Result<RepairPlan> {
        let mut plan = RepairPlan::default();

        let pins = self.pins.list_pins_for_origin(origin_id)?;
        let any_active = pins
            .iter()
            .any(|p| p.state != sodl_policy::PinState::Released);
        if !any_active {
            return Ok(plan);
        }

        let policy = self.policies.get_origin_policy(origin_id)?;
        let required = policy.retention.min_replicas.unwrap_or(1) as i64;

        let rec = self.origin_registry.get_origin(origin_id)?;
        for rep in &rec.representations {
            for b in &rep.root_blobs {
                let healthy =
                    self.replicas
                        .healthy_count_with_stale(b, self.stale_seconds, self.now)?;
                if healthy < required {
                    plan.items.push(RepairItem {
                        blob_id: b.clone(),
                        required,
                        healthy,
                        missing: required - healthy,
                    });
                }
            }
        }

        Ok(plan)
    }
}

/// Guards GC deletion by ensuring durability constraints are satisfied for pinned content.
pub struct DurabilityGate<'a> {
    pub origin_registry: &'a dyn OriginRegistry,
    pub policies: &'a dyn PolicyStore,
    pub pins: &'a dyn PinStore,
    pub replicas: &'a dyn ReplicaStore,
    pub stale_seconds: i64,
    pub now: time::OffsetDateTime,
}

impl<'a> DurabilityGate<'a> {
    /// Returns true if it is safe to delete bytes for this origin, considering pins + replica health.
    pub fn can_delete_origin_bytes(&self, origin_id: OriginId) -> Result<bool> {
        let pins = self.pins.list_pins_for_origin(origin_id)?;
        let any_active = pins
            .iter()
            .any(|p| p.state != sodl_policy::PinState::Released);
        if !any_active {
            return Ok(true);
        }

        let policy = self.policies.get_origin_policy(origin_id)?;
        let required = policy.retention.min_replicas.unwrap_or(1) as i64;

        let rec = self.origin_registry.get_origin(origin_id)?;
        for rep in &rec.representations {
            for b in &rep.root_blobs {
                let healthy =
                    self.replicas
                        .healthy_count_with_stale(b, self.stale_seconds, self.now)?;
                if healthy < required {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
}
