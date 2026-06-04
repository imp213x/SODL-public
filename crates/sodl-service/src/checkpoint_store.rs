use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sodl_cas::{compute_blob_id, BlobStore, FsBlobStore, HashAlg};
use sodl_core::{BlobId, Result, SodlError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointRecord {
    pub checkpoint_id: String,
    pub blob_id: BlobId,
    pub origin_id: String,
    pub step: u64,
    pub epoch: Option<u64>,
    pub loss: Option<f64>,
    pub metrics: BTreeMap<String, f64>,
    pub parent_checkpoint_id: Option<String>,
    pub stage: String,
    pub metadata: Value,
    pub optimizer_externalized: bool,
    pub optimizer_origin_id: Option<String>,
    pub optimizer_layout_fingerprint: Option<String>,
    pub optimizer_block_count: usize,
    pub dataset_manifests: Vec<String>,
    pub knowledge_logs: Vec<String>,
    pub size_raw: usize,
    pub size_stored: usize,
    pub created_at: time::OffsetDateTime,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CheckpointSaveRequest {
    pub step: u64,
    pub epoch: Option<u64>,
    pub loss: Option<f64>,
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
    pub parent_checkpoint_id: Option<String>,
    #[serde(default)]
    pub stage: String,
    #[serde(default = "default_json_object")]
    pub metadata: Value,
    #[serde(default)]
    pub optimizer_externalized: bool,
    pub optimizer_origin_id: Option<String>,
    pub optimizer_layout_fingerprint: Option<String>,
    #[serde(default)]
    pub optimizer_block_count: usize,
    #[serde(default)]
    pub dataset_manifests: Vec<String>,
    #[serde(default)]
    pub knowledge_logs: Vec<String>,
    #[serde(default)]
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct CheckpointRegistry {
    origin_id: String,
    checkpoints: Vec<CheckpointRecord>,
}

fn default_json_object() -> Value {
    Value::Object(serde_json::Map::new())
}

pub struct CheckpointStore {
    blob_store: FsBlobStore,
    registry_dir: PathBuf,
    compression_level: i32,
    max_checkpoints: usize,
}

impl CheckpointStore {
    pub fn open(
        blob_root: impl AsRef<Path>,
        registry_dir: impl AsRef<Path>,
        compression_level: i32,
        max_checkpoints: usize,
    ) -> Result<Self> {
        let blob_store = FsBlobStore::open(blob_root.as_ref())?;
        fs::create_dir_all(registry_dir.as_ref())
            .map_err(|err| SodlError::Io(format!("create checkpoint registry dir: {err}")))?;
        Ok(Self {
            blob_store,
            registry_dir: registry_dir.as_ref().to_path_buf(),
            compression_level,
            max_checkpoints,
        })
    }

    fn registry_path(&self, origin_id: &str) -> PathBuf {
        let safe_name = origin_id.replace(':', "_").replace('/', "_");
        self.registry_dir.join(format!("{safe_name}.json"))
    }

    fn load_registry(&self, origin_id: &str) -> Result<Vec<CheckpointRecord>> {
        let path = self.registry_path(origin_id);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let bytes = fs::read(&path).map_err(|err| {
            SodlError::Io(format!(
                "read checkpoint registry {}: {err}",
                path.display()
            ))
        })?;
        let registry: CheckpointRegistry = serde_json::from_slice(&bytes).map_err(|err| {
            SodlError::Serialization(format!(
                "parse checkpoint registry {}: {err}",
                path.display()
            ))
        })?;
        Ok(registry.checkpoints)
    }

    fn save_registry(&self, origin_id: &str, records: &[CheckpointRecord]) -> Result<()> {
        let path = self.registry_path(origin_id);
        let tmp = path.with_extension("json.tmp");
        let payload = CheckpointRegistry {
            origin_id: origin_id.to_string(),
            checkpoints: records.to_vec(),
        };
        let bytes = serde_json::to_vec_pretty(&payload).map_err(|err| {
            SodlError::Serialization(format!("serialize checkpoint registry: {err}"))
        })?;
        fs::write(&tmp, &bytes).map_err(|err| {
            SodlError::Io(format!(
                "write checkpoint registry temp {}: {err}",
                tmp.display()
            ))
        })?;
        fs::rename(&tmp, &path).map_err(|err| {
            SodlError::Io(format!(
                "replace checkpoint registry {}: {err}",
                path.display()
            ))
        })?;
        Ok(())
    }

    fn referenced_blob_ids(&self) -> Result<HashSet<BlobId>> {
        let mut referenced = HashSet::new();
        for entry in fs::read_dir(&self.registry_dir).map_err(|err| {
            SodlError::Io(format!(
                "read checkpoint registry dir {}: {err}",
                self.registry_dir.display()
            ))
        })? {
            let path = entry
                .map_err(|err| SodlError::Io(format!("read checkpoint registry entry: {err}")))?
                .path();
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if !name.ends_with(".json") {
                continue;
            }
            let bytes = fs::read(&path).map_err(|err| {
                SodlError::Io(format!(
                    "read checkpoint registry {}: {err}",
                    path.display()
                ))
            })?;
            let registry: CheckpointRegistry = serde_json::from_slice(&bytes).map_err(|err| {
                SodlError::Serialization(format!(
                    "parse checkpoint registry {}: {err}",
                    path.display()
                ))
            })?;
            referenced.extend(
                registry
                    .checkpoints
                    .into_iter()
                    .map(|record| record.blob_id),
            );
        }
        Ok(referenced)
    }

    fn gc_blob_ids(&self, candidates: HashSet<BlobId>) -> Result<usize> {
        if candidates.is_empty() {
            return Ok(0);
        }
        let referenced = self.referenced_blob_ids()?;
        let mut deleted = 0usize;
        for blob_id in candidates {
            if referenced.contains(&blob_id) {
                continue;
            }
            if self.blob_store.has(&blob_id)? {
                self.blob_store.delete(&blob_id)?;
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        zstd::stream::encode_all(Cursor::new(data), self.compression_level)
            .map_err(|err| SodlError::Compression(format!("compress checkpoint payload: {err}")))
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        zstd::stream::decode_all(Cursor::new(data))
            .map_err(|err| SodlError::Compression(format!("decompress checkpoint payload: {err}")))
    }

    fn find_record(
        &self,
        records: &[CheckpointRecord],
        checkpoint_id: Option<&str>,
    ) -> Result<CheckpointRecord> {
        if records.is_empty() {
            return Err(SodlError::NotFound);
        }
        if let Some(checkpoint_id) = checkpoint_id {
            records
                .iter()
                .find(|record| record.checkpoint_id == checkpoint_id)
                .cloned()
                .ok_or(SodlError::NotFound)
        } else {
            records.last().cloned().ok_or(SodlError::NotFound)
        }
    }

    pub fn save_checkpoint_bytes(
        &self,
        origin_id: &str,
        payload: &[u8],
        request: CheckpointSaveRequest,
    ) -> Result<CheckpointRecord> {
        let compressed = self.compress(payload)?;
        let blob_id = compute_blob_id(&compressed, HashAlg::Blake3);
        if !self.blob_store.has(&blob_id)? {
            self.blob_store
                .put(&blob_id, Bytes::from(compressed.clone()))?;
        }

        let record = CheckpointRecord {
            checkpoint_id: format!("ckpt:{}", uuid::Uuid::new_v4()),
            blob_id,
            origin_id: origin_id.to_string(),
            step: request.step,
            epoch: request.epoch,
            loss: request.loss,
            metrics: request.metrics,
            parent_checkpoint_id: request.parent_checkpoint_id,
            stage: request.stage,
            metadata: request.metadata,
            optimizer_externalized: request.optimizer_externalized,
            optimizer_origin_id: request.optimizer_origin_id,
            optimizer_layout_fingerprint: request.optimizer_layout_fingerprint,
            optimizer_block_count: request.optimizer_block_count,
            dataset_manifests: request.dataset_manifests,
            knowledge_logs: request.knowledge_logs,
            size_raw: payload.len(),
            size_stored: compressed.len(),
            created_at: time::OffsetDateTime::now_utc(),
            notes: request.notes,
        };

        let mut records = self.load_registry(origin_id)?;
        records.push(record.clone());

        let mut evicted_blob_ids = HashSet::new();
        if self.max_checkpoints > 0 && records.len() > self.max_checkpoints {
            let evicted = records.drain(..records.len() - self.max_checkpoints);
            evicted_blob_ids.extend(evicted.map(|item| item.blob_id));
        }

        self.save_registry(origin_id, &records)?;
        let _ = self.gc_blob_ids(evicted_blob_ids)?;
        Ok(record)
    }

    pub fn load_checkpoint_bytes(
        &self,
        origin_id: &str,
        checkpoint_id: Option<&str>,
    ) -> Result<Vec<u8>> {
        let records = self.load_registry(origin_id)?;
        let record = self.find_record(&records, checkpoint_id)?;
        let compressed = self.blob_store.get(&record.blob_id)?;
        self.decompress(&compressed)
    }

    pub fn list_checkpoints(&self, origin_id: &str) -> Result<Vec<CheckpointRecord>> {
        self.load_registry(origin_id)
    }

    pub fn get_checkpoint(
        &self,
        origin_id: &str,
        checkpoint_id: Option<&str>,
    ) -> Result<CheckpointRecord> {
        let records = self.load_registry(origin_id)?;
        self.find_record(&records, checkpoint_id)
    }

    pub fn get_lineage(
        &self,
        origin_id: &str,
        checkpoint_id: Option<&str>,
    ) -> Result<Vec<CheckpointRecord>> {
        let records = self.load_registry(origin_id)?;
        if records.is_empty() {
            return Ok(Vec::new());
        }
        let record_by_id: HashMap<String, CheckpointRecord> = records
            .iter()
            .cloned()
            .map(|record| (record.checkpoint_id.clone(), record))
            .collect();
        let mut current = self.find_record(&records, checkpoint_id)?;
        let mut lineage = Vec::new();
        loop {
            lineage.push(current.clone());
            let Some(parent_id) = current.parent_checkpoint_id.clone() else {
                break;
            };
            let Some(parent) = record_by_id.get(&parent_id).cloned() else {
                break;
            };
            current = parent;
        }
        lineage.reverse();
        Ok(lineage)
    }

    pub fn delete_checkpoint(&self, origin_id: &str, checkpoint_id: &str) -> Result<bool> {
        let records = self.load_registry(origin_id)?;
        let removed: Vec<CheckpointRecord> = records
            .iter()
            .filter(|record| record.checkpoint_id == checkpoint_id)
            .cloned()
            .collect();
        let retained: Vec<CheckpointRecord> = records
            .into_iter()
            .filter(|record| record.checkpoint_id != checkpoint_id)
            .collect();
        if removed.is_empty() {
            return Ok(false);
        }
        self.save_registry(origin_id, &retained)?;
        let _ = self.gc_blob_ids(removed.into_iter().map(|item| item.blob_id).collect())?;
        Ok(true)
    }

    pub fn diff_checkpoints(&self, origin_id: &str, old_id: &str, new_id: &str) -> Result<Value> {
        let records = self.load_registry(origin_id)?;
        let old_rec = self.find_record(&records, Some(old_id))?;
        let new_rec = self.find_record(&records, Some(new_id))?;

        let mut metric_deltas = serde_json::Map::new();
        let metric_keys: HashSet<String> = old_rec
            .metrics
            .keys()
            .cloned()
            .chain(new_rec.metrics.keys().cloned())
            .collect();
        for key in metric_keys {
            if let (Some(old_val), Some(new_val)) =
                (old_rec.metrics.get(&key), new_rec.metrics.get(&key))
            {
                metric_deltas.insert(key, Value::from(new_val - old_val));
            }
        }

        Ok(serde_json::json!({
            "step_delta": i64::try_from(new_rec.step).unwrap_or(i64::MAX)
                - i64::try_from(old_rec.step).unwrap_or(i64::MAX),
            "loss_delta": match (old_rec.loss, new_rec.loss) {
                (Some(old_val), Some(new_val)) => Some(new_val - old_val),
                _ => None,
            },
            "metric_deltas": metric_deltas,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(step: u64) -> CheckpointSaveRequest {
        CheckpointSaveRequest {
            step,
            ..CheckpointSaveRequest::default()
        }
    }

    #[test]
    fn checkpoint_store_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let store =
            CheckpointStore::open(tmp.path().join("blobs"), tmp.path().join("registry"), 3, 0)
                .unwrap();

        let record = store
            .save_checkpoint_bytes("run-1", b"checkpoint-a", request(10))
            .unwrap();
        let reloaded = store
            .load_checkpoint_bytes("run-1", Some(&record.checkpoint_id))
            .unwrap();

        assert_eq!(reloaded, b"checkpoint-a");
        assert_eq!(store.get_checkpoint("run-1", None).unwrap().step, 10);
    }

    #[test]
    fn max_checkpoints_reclaims_evicted_unique_blob() {
        let tmp = tempfile::tempdir().unwrap();
        let store =
            CheckpointStore::open(tmp.path().join("blobs"), tmp.path().join("registry"), 3, 2)
                .unwrap();

        let first = store
            .save_checkpoint_bytes("run-2", b"checkpoint-a", request(10))
            .unwrap();
        let first_blob_id = first.blob_id.clone();
        store
            .save_checkpoint_bytes("run-2", b"checkpoint-b", request(20))
            .unwrap();
        store
            .save_checkpoint_bytes("run-2", b"checkpoint-c", request(30))
            .unwrap();

        let records = store.list_checkpoints("run-2").unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].step, 20);
        assert!(!store.blob_store.has(&first_blob_id).unwrap());
    }

    #[test]
    fn delete_checkpoint_keeps_shared_blob_if_still_referenced() {
        let tmp = tempfile::tempdir().unwrap();
        let store =
            CheckpointStore::open(tmp.path().join("blobs"), tmp.path().join("registry"), 3, 0)
                .unwrap();

        let first = store
            .save_checkpoint_bytes("run-3", b"shared", request(10))
            .unwrap();
        let second = store
            .save_checkpoint_bytes("run-3", b"shared", request(10))
            .unwrap();

        assert_eq!(first.blob_id, second.blob_id);
        assert!(store
            .delete_checkpoint("run-3", &first.checkpoint_id)
            .unwrap());
        assert!(store.blob_store.has(&second.blob_id).unwrap());
    }

    #[test]
    fn lineage_follows_parent_chain() {
        let tmp = tempfile::tempdir().unwrap();
        let store =
            CheckpointStore::open(tmp.path().join("blobs"), tmp.path().join("registry"), 3, 0)
                .unwrap();

        let root = store
            .save_checkpoint_bytes("run-4", b"root", request(1))
            .unwrap();
        let child = store
            .save_checkpoint_bytes(
                "run-4",
                b"child",
                CheckpointSaveRequest {
                    step: 2,
                    parent_checkpoint_id: Some(root.checkpoint_id.clone()),
                    stage: "phase-2".to_string(),
                    ..CheckpointSaveRequest::default()
                },
            )
            .unwrap();

        let lineage = store
            .get_lineage("run-4", Some(&child.checkpoint_id))
            .unwrap();
        assert_eq!(lineage.len(), 2);
        assert_eq!(lineage[0].checkpoint_id, root.checkpoint_id);
        assert_eq!(lineage[1].stage, "phase-2");
    }
}
