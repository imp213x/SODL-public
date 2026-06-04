//! Replica tracking (health + placement).
//!
//! Tracks where blob bytes exist across nodes/stores, enabling `min_replicas` durability intent.

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use sodl_cas::BlobStore;
use sodl_core::{BlobId, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplicaState {
    Healthy,
    Unknown,
    Dead,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicaRecord {
    pub blob_id: BlobId,
    pub node_id: String,
    pub state: ReplicaState,
    pub last_seen: time::OffsetDateTime,
}

/// Durability repair plan: blobs that need additional healthy replicas.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RepairPlan {
    pub items: Vec<RepairItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairItem {
    pub blob_id: BlobId,
    pub required: i64,
    pub healthy: i64,
    pub missing: i64,
}

pub trait ReplicaStore: Send + Sync {
    fn upsert_replica(&self, rec: ReplicaRecord) -> Result<()>;
    fn list_replicas(&self, blob_id: &BlobId) -> Result<Vec<ReplicaRecord>>;
    fn healthy_count(&self, blob_id: &BlobId) -> Result<i64>;
    fn healthy_count_with_stale(
        &self,
        blob_id: &BlobId,
        stale_seconds: i64,
        now: time::OffsetDateTime,
    ) -> Result<i64>;
}

#[derive(Clone, Default)]
pub struct MemReplicaStore {
    inner: Arc<RwLock<HashMap<(String, String), ReplicaRecord>>>,
}

impl MemReplicaStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ReplicaStore for MemReplicaStore {
    fn upsert_replica(&self, rec: ReplicaRecord) -> Result<()> {
        let mut w = self
            .inner
            .write()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        w.insert((rec.blob_id.0.clone(), rec.node_id.clone()), rec);
        Ok(())
    }

    fn list_replicas(&self, blob_id: &BlobId) -> Result<Vec<ReplicaRecord>> {
        let r = self
            .inner
            .read()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        Ok(r.values()
            .filter(|v| v.blob_id.0 == blob_id.0)
            .cloned()
            .collect())
    }

    fn healthy_count(&self, blob_id: &BlobId) -> Result<i64> {
        let reps = self.list_replicas(blob_id)?;
        Ok(reps
            .into_iter()
            .filter(|r| r.state == ReplicaState::Healthy)
            .count() as i64)
    }
    fn healthy_count_with_stale(
        &self,
        blob_id: &BlobId,
        stale_seconds: i64,
        now: time::OffsetDateTime,
    ) -> Result<i64> {
        let reps = self.list_replicas(blob_id)?;
        let mut n: i64 = 0;
        for r in reps {
            if r.state != ReplicaState::Healthy {
                continue;
            }
            let age = (now - r.last_seen).whole_seconds();
            if age <= stale_seconds {
                n += 1;
            }
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_replica_store_counts_health() {
        let rs = MemReplicaStore::new();
        let b = BlobId("blake3:abc".into());
        rs.upsert_replica(ReplicaRecord {
            blob_id: b.clone(),
            node_id: "n1".into(),
            state: ReplicaState::Healthy,
            last_seen: time::OffsetDateTime::now_utc(),
        })
        .unwrap();
        rs.upsert_replica(ReplicaRecord {
            blob_id: b.clone(),
            node_id: "n2".into(),
            state: ReplicaState::Dead,
            last_seen: time::OffsetDateTime::now_utc(),
        })
        .unwrap();
        assert_eq!(rs.healthy_count(&b).unwrap(), 1);
    }
}

/// Provides access to per-node blob stores for replication.
pub trait StoreMesh: Send + Sync {
    fn node_ids(&self) -> Vec<String>;
    fn store(&self, node_id: &str) -> Result<Arc<dyn BlobStore>>;
}

/// Executes a repair plan by copying blob bytes to missing nodes and updating replica health.
pub struct ReplicaExecutor<'a> {
    pub mesh: &'a dyn StoreMesh,
    pub replicas: &'a dyn ReplicaStore,
}

impl<'a> ReplicaExecutor<'a> {
    pub fn execute(&self, plan: RepairPlan) -> Result<()> {
        for item in plan.items {
            self.repair_blob(item)?;
        }
        Ok(())
    }

    fn repair_blob(&self, item: RepairItem) -> Result<()> {
        if item.missing <= 0 {
            return Ok(());
        }

        let current = self.replicas.list_replicas(&item.blob_id)?;
        let mut healthy_nodes: std::collections::HashSet<String> = current
            .iter()
            .filter(|r| r.state == ReplicaState::Healthy)
            .map(|r| r.node_id.clone())
            .collect();

        // Find a source node that has the bytes.
        let mut source_node: Option<String> = None;
        for nid in self.mesh.node_ids() {
            let st = self.mesh.store(&nid)?;
            if st.has(&item.blob_id)? {
                source_node = Some(nid);
                break;
            }
        }
        let source = source_node.ok_or(sodl_core::SodlError::NotFound)?;
        let source_store = self.mesh.store(&source)?;
        let bytes: Bytes = source_store.get(&item.blob_id)?;

        // Replicate to nodes that are not healthy yet, up to missing count.
        let mut remaining = item.missing;
        for nid in self.mesh.node_ids() {
            if remaining <= 0 {
                break;
            }
            if healthy_nodes.contains(&nid) {
                continue;
            }
            let st = self.mesh.store(&nid)?;
            if !st.has(&item.blob_id)? {
                st.put(&item.blob_id, bytes.clone())?;
            }
            self.replicas.upsert_replica(ReplicaRecord {
                blob_id: item.blob_id.clone(),
                node_id: nid.clone(),
                state: ReplicaState::Healthy,
                last_seen: time::OffsetDateTime::now_utc(),
            })?;
            healthy_nodes.insert(nid);
            remaining -= 1;
        }

        Ok(())
    }
}

/// In-memory store mesh for demos/tests.
#[derive(Clone, Default)]
pub struct MemStoreMesh {
    inner: Arc<RwLock<HashMap<String, Arc<dyn BlobStore>>>>,
}

impl MemStoreMesh {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node<S: BlobStore + 'static>(&self, node_id: &str, store: S) -> Result<()> {
        let mut w = self
            .inner
            .write()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        w.insert(node_id.to_string(), Arc::new(store));
        Ok(())
    }
}

impl StoreMesh for MemStoreMesh {
    fn node_ids(&self) -> Vec<String> {
        let r = self.inner.read().expect("poison");
        r.keys().cloned().collect()
    }

    fn store(&self, node_id: &str) -> Result<Arc<dyn BlobStore>> {
        let r = self
            .inner
            .read()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        let arc = r.get(node_id).ok_or(sodl_core::SodlError::NotFound)?;
        Ok(arc.clone())
    }
}

#[test]
fn counts_health_with_stale_threshold() {
    let rs = MemReplicaStore::new();
    let b = BlobId("blake3:stale".into());
    let now = time::OffsetDateTime::now_utc();

    rs.upsert_replica(ReplicaRecord {
        blob_id: b.clone(),
        node_id: "fresh".into(),
        state: ReplicaState::Healthy,
        last_seen: now,
    })
    .unwrap();

    rs.upsert_replica(ReplicaRecord {
        blob_id: b.clone(),
        node_id: "stale".into(),
        state: ReplicaState::Healthy,
        last_seen: now - time::Duration::seconds(10_000),
    })
    .unwrap();

    assert_eq!(rs.healthy_count_with_stale(&b, 60, now).unwrap(), 1);
    assert_eq!(rs.healthy_count_with_stale(&b, 20_000, now).unwrap(), 2);
}
