use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sodl_cas::{compute_blob_id, BlobStore, FsBlobStore, HashAlg};
use sodl_core::{BlobId, Result, SodlError, SODL_SCHEMA_VERSION};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerBlockRecord {
    pub block_id: String,
    pub blob_id: BlobId,
    pub step: u64,
    pub shard_key: Option<String>,
    pub size_raw: usize,
    pub size_stored: usize,
    pub stored_at: time::OffsetDateTime,
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerStateManifest {
    pub schema: String,
    pub origin_id: String,
    pub blocks: BTreeMap<String, OptimizerBlockRecord>,
    pub updated_at: time::OffsetDateTime,
}

impl OptimizerStateManifest {
    fn new(origin_id: &str) -> Self {
        Self {
            schema: SODL_SCHEMA_VERSION.to_string(),
            origin_id: origin_id.to_string(),
            blocks: BTreeMap::new(),
            updated_at: time::OffsetDateTime::now_utc(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerStoreResult {
    pub origin_id: String,
    pub block_id: String,
    pub blob_id: Option<BlobId>,
    pub step: u64,
    pub staged: bool,
    pub flushed: bool,
    pub size_raw: usize,
    pub size_stored: usize,
    pub dirty_blocks: usize,
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerBlockInput {
    pub block_id: String,
    pub payload: Vec<u8>,
    pub step: u64,
    pub shard_key: Option<String>,
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OptimizerCacheStats {
    pub cache_entries: usize,
    pub dirty_entries: usize,
    pub pinned_entries: usize,
    pub cache_capacity: usize,
    pub writeback_threshold: usize,
}

#[derive(Debug, Clone)]
struct CachedOptimizerBlock {
    bytes: Vec<u8>,
    step: u64,
    shard_key: Option<String>,
    metadata: Value,
    dirty: bool,
    pinned: bool,
    last_touch: time::OffsetDateTime,
    latest_record: Option<OptimizerBlockRecord>,
}

impl CachedOptimizerBlock {
    fn new(bytes: &[u8], step: u64, shard_key: Option<String>, metadata: Value) -> Self {
        Self {
            bytes: bytes.to_vec(),
            step,
            shard_key,
            metadata,
            dirty: true,
            pinned: false,
            last_touch: time::OffsetDateTime::now_utc(),
            latest_record: None,
        }
    }

    fn touch(&mut self) {
        self.last_touch = time::OffsetDateTime::now_utc();
    }
}

pub struct OptimizerStateStore {
    blob_store: FsBlobStore,
    registry_dir: PathBuf,
    compression_level: i32,
    cache_capacity: usize,
    writeback_threshold: usize,
    cache: Mutex<HashMap<String, CachedOptimizerBlock>>,
    pinned: Mutex<HashSet<String>>,
}

impl OptimizerStateStore {
    pub fn open(
        blob_root: impl AsRef<Path>,
        registry_dir: impl AsRef<Path>,
        compression_level: i32,
        cache_capacity: usize,
        writeback_threshold: usize,
    ) -> Result<Self> {
        let blob_store = FsBlobStore::open(blob_root.as_ref())?;
        fs::create_dir_all(registry_dir.as_ref())
            .map_err(|err| SodlError::Io(format!("create optimizer registry dir: {err}")))?;
        Ok(Self {
            blob_store,
            registry_dir: registry_dir.as_ref().to_path_buf(),
            compression_level,
            cache_capacity: cache_capacity.max(1),
            writeback_threshold: writeback_threshold.max(1),
            cache: Mutex::new(HashMap::new()),
            pinned: Mutex::new(HashSet::new()),
        })
    }

    fn cache_key(origin_id: &str, block_id: &str) -> String {
        format!("{origin_id}::{block_id}")
    }

    fn manifest_path(&self, origin_id: &str) -> PathBuf {
        let safe_name = origin_id.replace(':', "_").replace('/', "_");
        self.registry_dir
            .join(format!("{safe_name}.optimizer.json"))
    }

    fn load_manifest(&self, origin_id: &str) -> Result<OptimizerStateManifest> {
        let path = self.manifest_path(origin_id);
        if !path.exists() {
            return Ok(OptimizerStateManifest::new(origin_id));
        }
        let data = fs::read(&path).map_err(|err| {
            SodlError::Io(format!("read optimizer manifest {}: {err}", path.display()))
        })?;
        serde_json::from_slice(&data)
            .map_err(|err| SodlError::Serialization(format!("parse optimizer manifest: {err}")))
    }

    fn save_manifest(&self, manifest: &OptimizerStateManifest) -> Result<()> {
        let path = self.manifest_path(&manifest.origin_id);
        let tmp = path.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(manifest).map_err(|err| {
            SodlError::Serialization(format!("serialize optimizer manifest: {err}"))
        })?;
        fs::write(&tmp, &bytes).map_err(|err| {
            SodlError::Io(format!(
                "write optimizer manifest temp {}: {err}",
                tmp.display()
            ))
        })?;
        fs::rename(&tmp, &path).map_err(|err| {
            SodlError::Io(format!(
                "replace optimizer manifest {}: {err}",
                path.display()
            ))
        })?;
        Ok(())
    }

    fn referenced_blob_ids(&self) -> Result<HashSet<BlobId>> {
        let mut referenced = HashSet::new();
        for entry in fs::read_dir(&self.registry_dir).map_err(|err| {
            SodlError::Io(format!(
                "read optimizer registry dir {}: {err}",
                self.registry_dir.display()
            ))
        })? {
            let path = entry
                .map_err(|err| SodlError::Io(format!("read optimizer registry entry: {err}")))?
                .path();
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if !name.ends_with(".optimizer.json") {
                continue;
            }
            let bytes = fs::read(&path).map_err(|err| {
                SodlError::Io(format!("read optimizer manifest {}: {err}", path.display()))
            })?;
            let manifest: OptimizerStateManifest =
                serde_json::from_slice(&bytes).map_err(|err| {
                    SodlError::Serialization(format!(
                        "parse optimizer manifest {}: {err}",
                        path.display()
                    ))
                })?;
            referenced.extend(manifest.blocks.into_values().map(|record| record.blob_id));
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
            .map_err(|err| SodlError::Compression(format!("compress optimizer block: {err}")))
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        zstd::stream::decode_all(Cursor::new(data))
            .map_err(|err| SodlError::Compression(format!("decompress optimizer block: {err}")))
    }

    fn enforce_cache_capacity(&self) -> Result<()> {
        let pinned_snapshot = self
            .pinned
            .lock()
            .map_err(|err| SodlError::Io(format!("optimizer pin cache poisoned: {err}")))?
            .clone();
        let mut cache = self
            .cache
            .lock()
            .map_err(|err| SodlError::Io(format!("optimizer cache poisoned: {err}")))?;
        if cache.len() <= self.cache_capacity {
            return Ok(());
        }

        let mut candidates: Vec<(String, time::OffsetDateTime)> = cache
            .iter()
            .filter(|(key, block)| !block.dirty && !block.pinned && !pinned_snapshot.contains(*key))
            .map(|(key, block)| (key.clone(), block.last_touch))
            .collect();
        candidates.sort_by_key(|(_, touch)| *touch);

        while cache.len() > self.cache_capacity {
            if let Some((key, _)) = candidates.first().cloned() {
                cache.remove(&key);
                candidates.remove(0);
            } else {
                break;
            }
        }
        Ok(())
    }

    pub fn store_block(
        &self,
        origin_id: &str,
        block_id: &str,
        payload: &[u8],
        step: u64,
        shard_key: Option<String>,
        metadata: Value,
    ) -> Result<OptimizerStoreResult> {
        let key = Self::cache_key(origin_id, block_id);
        let is_pinned = self
            .pinned
            .lock()
            .map_err(|err| SodlError::Io(format!("optimizer pin cache poisoned: {err}")))?
            .contains(&key);
        let dirty_blocks = {
            let mut cache = self
                .cache
                .lock()
                .map_err(|err| SodlError::Io(format!("optimizer cache poisoned: {err}")))?;
            let entry = cache.entry(key.clone()).or_insert_with(|| {
                CachedOptimizerBlock::new(payload, step, shard_key.clone(), metadata.clone())
            });
            entry.bytes = payload.to_vec();
            entry.step = step;
            entry.shard_key = shard_key;
            entry.metadata = metadata.clone();
            entry.dirty = true;
            entry.touch();
            entry.pinned = is_pinned;
            cache
                .iter()
                .filter(|(candidate, block)| {
                    candidate.starts_with(&format!("{origin_id}::")) && block.dirty
                })
                .count()
        };

        let mut result = OptimizerStoreResult {
            origin_id: origin_id.to_string(),
            block_id: block_id.to_string(),
            blob_id: None,
            step,
            staged: true,
            flushed: false,
            size_raw: payload.len(),
            size_stored: 0,
            dirty_blocks,
            metadata,
        };

        if dirty_blocks >= self.writeback_threshold {
            let manifest = self.flush_origin(origin_id)?;
            if let Some(record) = manifest.blocks.get(block_id) {
                result.blob_id = Some(record.blob_id.clone());
                result.staged = false;
                result.flushed = true;
                result.size_stored = record.size_stored;
            }
            result.dirty_blocks = self.dirty_block_count(Some(origin_id))?;
        }

        Ok(result)
    }

    pub fn store_blocks(
        &self,
        origin_id: &str,
        inputs: &[OptimizerBlockInput],
    ) -> Result<Vec<OptimizerStoreResult>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        {
            let pinned = self
                .pinned
                .lock()
                .map_err(|err| SodlError::Io(format!("optimizer pin cache poisoned: {err}")))?;
            let mut cache = self
                .cache
                .lock()
                .map_err(|err| SodlError::Io(format!("optimizer cache poisoned: {err}")))?;
            for input in inputs {
                let key = Self::cache_key(origin_id, &input.block_id);
                let entry = cache.entry(key.clone()).or_insert_with(|| {
                    CachedOptimizerBlock::new(
                        &input.payload,
                        input.step,
                        input.shard_key.clone(),
                        input.metadata.clone(),
                    )
                });
                entry.bytes = input.payload.clone();
                entry.step = input.step;
                entry.shard_key = input.shard_key.clone();
                entry.metadata = input.metadata.clone();
                entry.dirty = true;
                entry.touch();
                entry.pinned = pinned.contains(&key);
            }
        }

        let dirty_blocks = self.dirty_block_count(Some(origin_id))?;
        let manifest = if dirty_blocks >= self.writeback_threshold {
            Some(self.flush_origin(origin_id)?)
        } else {
            None
        };
        let dirty_after = if manifest.is_some() {
            self.dirty_block_count(Some(origin_id))?
        } else {
            dirty_blocks
        };

        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            let record = manifest
                .as_ref()
                .and_then(|resolved| resolved.blocks.get(&input.block_id));
            results.push(OptimizerStoreResult {
                origin_id: origin_id.to_string(),
                block_id: input.block_id.clone(),
                blob_id: record.map(|item| item.blob_id.clone()),
                step: input.step,
                staged: record.is_none(),
                flushed: record.is_some(),
                size_raw: input.payload.len(),
                size_stored: record.map(|item| item.size_stored).unwrap_or(0),
                dirty_blocks: dirty_after,
                metadata: input.metadata.clone(),
            });
        }
        Ok(results)
    }

    pub fn load_block(&self, origin_id: &str, block_id: &str) -> Result<Vec<u8>> {
        let key = Self::cache_key(origin_id, block_id);
        {
            let mut cache = self
                .cache
                .lock()
                .map_err(|err| SodlError::Io(format!("optimizer cache poisoned: {err}")))?;
            if let Some(block) = cache.get_mut(&key) {
                block.touch();
                return Ok(block.bytes.clone());
            }
        }

        let manifest = self.load_manifest(origin_id)?;
        let record = manifest
            .blocks
            .get(block_id)
            .ok_or(SodlError::NotFound)?
            .clone();
        let compressed = self.blob_store.get(&record.blob_id)?;
        let bytes = self.decompress(&compressed)?;
        let is_pinned = self
            .pinned
            .lock()
            .map_err(|err| SodlError::Io(format!("optimizer pin cache poisoned: {err}")))?
            .contains(&key);

        let mut cache = self
            .cache
            .lock()
            .map_err(|err| SodlError::Io(format!("optimizer cache poisoned: {err}")))?;
        cache.insert(
            key.clone(),
            CachedOptimizerBlock {
                bytes: bytes.clone(),
                step: record.step,
                shard_key: record.shard_key.clone(),
                metadata: record.metadata.clone(),
                dirty: false,
                pinned: is_pinned,
                last_touch: time::OffsetDateTime::now_utc(),
                latest_record: Some(record),
            },
        );
        drop(cache);
        self.enforce_cache_capacity()?;
        Ok(bytes)
    }

    pub fn load_blocks(
        &self,
        origin_id: &str,
        block_ids: &[String],
    ) -> Result<BTreeMap<String, Vec<u8>>> {
        let mut payloads = BTreeMap::new();
        let mut missing = Vec::new();

        {
            let mut cache = self
                .cache
                .lock()
                .map_err(|err| SodlError::Io(format!("optimizer cache poisoned: {err}")))?;
            for block_id in block_ids {
                let key = Self::cache_key(origin_id, block_id);
                if let Some(block) = cache.get_mut(&key) {
                    block.touch();
                    payloads.insert(block_id.clone(), block.bytes.clone());
                } else {
                    missing.push(block_id.clone());
                }
            }
        }

        if missing.is_empty() {
            return Ok(payloads);
        }

        let manifest = self.load_manifest(origin_id)?;
        let pinned_snapshot = self
            .pinned
            .lock()
            .map_err(|err| SodlError::Io(format!("optimizer pin cache poisoned: {err}")))?
            .clone();
        let mut hydrated = Vec::new();

        for block_id in missing {
            let Some(record) = manifest.blocks.get(&block_id).cloned() else {
                continue;
            };
            let compressed = self.blob_store.get(&record.blob_id)?;
            let bytes = self.decompress(&compressed)?;
            let key = Self::cache_key(origin_id, &block_id);
            payloads.insert(block_id.clone(), bytes.clone());
            hydrated.push((
                key.clone(),
                CachedOptimizerBlock {
                    bytes,
                    step: record.step,
                    shard_key: record.shard_key.clone(),
                    metadata: record.metadata.clone(),
                    dirty: false,
                    pinned: pinned_snapshot.contains(&key),
                    last_touch: time::OffsetDateTime::now_utc(),
                    latest_record: Some(record),
                },
            ));
        }

        if !hydrated.is_empty() {
            let mut cache = self
                .cache
                .lock()
                .map_err(|err| SodlError::Io(format!("optimizer cache poisoned: {err}")))?;
            for (key, block) in hydrated {
                cache.insert(key, block);
            }
            drop(cache);
            self.enforce_cache_capacity()?;
        }

        Ok(payloads)
    }

    pub fn prefetch_blocks(&self, origin_id: &str, block_ids: &[String]) -> Result<usize> {
        let mut loaded = 0usize;
        for block_id in block_ids {
            if self.load_block(origin_id, block_id).is_ok() {
                loaded += 1;
            }
        }
        Ok(loaded)
    }

    pub fn pin_blocks(&self, origin_id: &str, block_ids: &[String]) -> Result<()> {
        let mut pinned = self
            .pinned
            .lock()
            .map_err(|err| SodlError::Io(format!("optimizer pin cache poisoned: {err}")))?;
        for block_id in block_ids {
            let key = Self::cache_key(origin_id, block_id);
            pinned.insert(key.clone());
            if let Ok(mut cache) = self.cache.lock() {
                if let Some(block) = cache.get_mut(&key) {
                    block.pinned = true;
                }
            }
        }
        Ok(())
    }

    pub fn unpin_blocks(&self, origin_id: &str, block_ids: &[String]) -> Result<()> {
        let mut pinned = self
            .pinned
            .lock()
            .map_err(|err| SodlError::Io(format!("optimizer pin cache poisoned: {err}")))?;
        for block_id in block_ids {
            let key = Self::cache_key(origin_id, block_id);
            pinned.remove(&key);
            if let Ok(mut cache) = self.cache.lock() {
                if let Some(block) = cache.get_mut(&key) {
                    block.pinned = false;
                }
            }
        }
        drop(pinned);
        self.enforce_cache_capacity()?;
        Ok(())
    }

    pub fn evict_blocks(&self, origin_id: &str, block_ids: &[String]) -> Result<usize> {
        let pinned = self
            .pinned
            .lock()
            .map_err(|err| SodlError::Io(format!("optimizer pin cache poisoned: {err}")))?;
        let mut cache = self
            .cache
            .lock()
            .map_err(|err| SodlError::Io(format!("optimizer cache poisoned: {err}")))?;
        let mut evicted = 0usize;
        for block_id in block_ids {
            let key = Self::cache_key(origin_id, block_id);
            let should_skip = cache
                .get(&key)
                .map(|block| block.dirty || block.pinned || pinned.contains(&key))
                .unwrap_or(false);
            if should_skip {
                continue;
            }
            if cache.remove(&key).is_some() {
                evicted += 1;
            }
        }
        Ok(evicted)
    }

    fn flush_keys(&self, origin_id: &str, keys: Vec<String>) -> Result<OptimizerStateManifest> {
        let mut manifest = self.load_manifest(origin_id)?;
        let mut replaced_blob_ids = HashSet::new();

        if keys.is_empty() {
            return Ok(manifest);
        }

        let mut cache = self
            .cache
            .lock()
            .map_err(|err| SodlError::Io(format!("optimizer cache poisoned: {err}")))?;
        for key in keys {
            let Some(block) = cache.get_mut(&key) else {
                continue;
            };
            if !block.dirty {
                continue;
            }

            let compressed = self.compress(&block.bytes)?;
            let blob_id = compute_blob_id(&compressed, HashAlg::Blake3);
            if !self.blob_store.has(&blob_id)? {
                self.blob_store.put(&blob_id, compressed.clone().into())?;
            }

            let block_id = key
                .split_once("::")
                .map(|(_, block_id)| block_id.to_string())
                .ok_or_else(|| SodlError::Invalid(format!("invalid optimizer cache key: {key}")))?;
            let record = OptimizerBlockRecord {
                block_id: block_id.clone(),
                blob_id,
                step: block.step,
                shard_key: block.shard_key.clone(),
                size_raw: block.bytes.len(),
                size_stored: compressed.len(),
                stored_at: time::OffsetDateTime::now_utc(),
                metadata: block.metadata.clone(),
            };
            if let Some(previous) = manifest.blocks.insert(block_id, record.clone()) {
                if previous.blob_id != record.blob_id {
                    replaced_blob_ids.insert(previous.blob_id);
                }
            }
            manifest.updated_at = time::OffsetDateTime::now_utc();
            block.dirty = false;
            block.latest_record = Some(record);
            block.touch();
        }
        drop(cache);
        self.save_manifest(&manifest)?;
        let _ = self.gc_blob_ids(replaced_blob_ids)?;
        self.enforce_cache_capacity()?;
        Ok(manifest)
    }

    pub fn flush_origin(&self, origin_id: &str) -> Result<OptimizerStateManifest> {
        let keys: Vec<String> = {
            let cache = self
                .cache
                .lock()
                .map_err(|err| SodlError::Io(format!("optimizer cache poisoned: {err}")))?;
            cache
                .keys()
                .filter(|key| key.starts_with(&format!("{origin_id}::")))
                .cloned()
                .collect()
        };
        self.flush_keys(origin_id, keys)
    }

    pub fn flush_blocks(
        &self,
        origin_id: &str,
        block_ids: &[String],
    ) -> Result<OptimizerStateManifest> {
        let keys = block_ids
            .iter()
            .map(|block_id| Self::cache_key(origin_id, block_id))
            .collect();
        self.flush_keys(origin_id, keys)
    }

    pub fn manifest(&self, origin_id: &str) -> Result<OptimizerStateManifest> {
        self.load_manifest(origin_id)
    }

    pub fn latest_blob_id(&self, origin_id: &str, block_id: &str) -> Result<Option<BlobId>> {
        let manifest = self.load_manifest(origin_id)?;
        Ok(manifest
            .blocks
            .get(block_id)
            .map(|record| record.blob_id.clone()))
    }

    pub fn dirty_block_count(&self, origin_filter: Option<&str>) -> Result<usize> {
        let cache = self
            .cache
            .lock()
            .map_err(|err| SodlError::Io(format!("optimizer cache poisoned: {err}")))?;
        Ok(cache
            .iter()
            .filter(|(key, block)| {
                block.dirty
                    && origin_filter
                        .map(|origin_id| key.starts_with(&format!("{origin_id}::")))
                        .unwrap_or(true)
            })
            .count())
    }

    pub fn set_cache_capacity(&mut self, cache_capacity: usize) {
        self.cache_capacity = cache_capacity.max(1);
    }

    pub fn cache_stats(&self) -> Result<OptimizerCacheStats> {
        let cache = self
            .cache
            .lock()
            .map_err(|err| SodlError::Io(format!("optimizer cache poisoned: {err}")))?;
        let pinned = self
            .pinned
            .lock()
            .map_err(|err| SodlError::Io(format!("optimizer pin cache poisoned: {err}")))?;
        Ok(OptimizerCacheStats {
            cache_entries: cache.len(),
            dirty_entries: cache.values().filter(|block| block.dirty).count(),
            pinned_entries: pinned.len(),
            cache_capacity: self.cache_capacity,
            writeback_threshold: self.writeback_threshold,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimizer_block_stage_flush_and_reload_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let blobs = tmp.path().join("blobs");
        let registry = tmp.path().join("registry");
        let store = OptimizerStateStore::open(&blobs, &registry, 3, 8, 4).unwrap();

        let result = store
            .store_block(
                "run-1",
                "block-0",
                b"optimizer-payload",
                7,
                Some("group:0".to_string()),
                serde_json::json!({"kind": "adamw"}),
            )
            .unwrap();
        assert!(result.staged);
        assert!(!result.flushed);

        let manifest = store.flush_origin("run-1").unwrap();
        assert!(manifest.blocks.contains_key("block-0"));

        let reloaded = store.load_block("run-1", "block-0").unwrap();
        assert_eq!(reloaded, b"optimizer-payload");
    }

    #[test]
    fn optimizer_store_auto_flushes_after_threshold() {
        let tmp = tempfile::tempdir().unwrap();
        let blobs = tmp.path().join("blobs");
        let registry = tmp.path().join("registry");
        let store = OptimizerStateStore::open(&blobs, &registry, 3, 8, 2).unwrap();

        let first = store
            .store_block("run-2", "block-a", b"a", 1, None, Value::Null)
            .unwrap();
        assert!(first.staged);
        let second = store
            .store_block("run-2", "block-b", b"b", 1, None, Value::Null)
            .unwrap();
        assert!(second.flushed);

        let manifest = store.manifest("run-2").unwrap();
        assert_eq!(manifest.blocks.len(), 2);
    }

    #[test]
    fn prefetch_and_pin_blocks_update_cache_stats() {
        let tmp = tempfile::tempdir().unwrap();
        let blobs = tmp.path().join("blobs");
        let registry = tmp.path().join("registry");
        let store = OptimizerStateStore::open(&blobs, &registry, 3, 4, 1).unwrap();

        store
            .store_block("run-3", "block-a", b"state-a", 3, None, Value::Null)
            .unwrap();
        store
            .store_block("run-3", "block-b", b"state-b", 3, None, Value::Null)
            .unwrap();

        let prefetched = store
            .prefetch_blocks("run-3", &[String::from("block-a"), String::from("block-b")])
            .unwrap();
        assert_eq!(prefetched, 2);
        store
            .pin_blocks("run-3", &[String::from("block-a")])
            .unwrap();

        let stats = store.cache_stats().unwrap();
        assert!(stats.cache_entries >= 2);
        assert_eq!(stats.pinned_entries, 1);
    }

    #[test]
    fn flush_blocks_and_evict_respect_dirty_and_pinned_state() {
        let tmp = tempfile::tempdir().unwrap();
        let blobs = tmp.path().join("blobs");
        let registry = tmp.path().join("registry");
        let store = OptimizerStateStore::open(&blobs, &registry, 3, 4, 10).unwrap();

        store
            .store_block("run-4", "block-a", b"state-a", 1, None, Value::Null)
            .unwrap();
        store
            .store_block("run-4", "block-b", b"state-b", 1, None, Value::Null)
            .unwrap();

        let manifest = store
            .flush_blocks("run-4", &[String::from("block-a")])
            .unwrap();
        assert!(manifest.blocks.contains_key("block-a"));
        assert!(!manifest.blocks.contains_key("block-b"));

        store
            .pin_blocks("run-4", &[String::from("block-a")])
            .unwrap();
        let evicted = store
            .evict_blocks("run-4", &[String::from("block-a"), String::from("block-b")])
            .unwrap();
        assert_eq!(evicted, 0);

        store
            .flush_blocks("run-4", &[String::from("block-b")])
            .unwrap();
        store
            .unpin_blocks("run-4", &[String::from("block-a")])
            .unwrap();
        let evicted = store
            .evict_blocks("run-4", &[String::from("block-a"), String::from("block-b")])
            .unwrap();
        assert_eq!(evicted, 2);
    }

    #[test]
    fn flush_reclaims_superseded_unique_blob() {
        let tmp = tempfile::tempdir().unwrap();
        let blobs = tmp.path().join("blobs");
        let registry = tmp.path().join("registry");
        let store = OptimizerStateStore::open(&blobs, &registry, 3, 4, 1).unwrap();

        store
            .store_block("run-5", "block-a", b"state-a", 1, None, Value::Null)
            .unwrap();
        let first_manifest = store.manifest("run-5").unwrap();
        let first_blob_id = first_manifest.blocks["block-a"].blob_id.clone();
        assert!(store.blob_store.has(&first_blob_id).unwrap());

        store
            .store_block("run-5", "block-a", b"state-b", 2, None, Value::Null)
            .unwrap();
        let second_manifest = store.manifest("run-5").unwrap();
        let second_blob_id = second_manifest.blocks["block-a"].blob_id.clone();

        assert_ne!(first_blob_id, second_blob_id);
        assert!(!store.blob_store.has(&first_blob_id).unwrap());
        assert!(store.blob_store.has(&second_blob_id).unwrap());
    }

    #[test]
    fn flush_keeps_shared_blob_if_another_block_still_references_it() {
        let tmp = tempfile::tempdir().unwrap();
        let blobs = tmp.path().join("blobs");
        let registry = tmp.path().join("registry");
        let store = OptimizerStateStore::open(&blobs, &registry, 3, 4, 10).unwrap();

        store
            .store_block("run-6", "block-a", b"shared", 1, None, Value::Null)
            .unwrap();
        store
            .store_block("run-6", "block-b", b"shared", 1, None, Value::Null)
            .unwrap();
        let first_manifest = store.flush_origin("run-6").unwrap();
        let shared_blob_id = first_manifest.blocks["block-a"].blob_id.clone();
        assert_eq!(shared_blob_id, first_manifest.blocks["block-b"].blob_id);

        store
            .store_block("run-6", "block-a", b"updated", 2, None, Value::Null)
            .unwrap();
        let second_manifest = store.flush_origin("run-6").unwrap();
        let updated_blob_id = second_manifest.blocks["block-a"].blob_id.clone();

        assert_ne!(shared_blob_id, updated_blob_id);
        assert!(store.blob_store.has(&shared_blob_id).unwrap());
        assert!(store.blob_store.has(&updated_blob_id).unwrap());
    }

    #[test]
    fn store_and_load_blocks_batch_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let blobs = tmp.path().join("blobs");
        let registry = tmp.path().join("registry");
        let store = OptimizerStateStore::open(&blobs, &registry, 3, 4, 8).unwrap();

        let results = store
            .store_blocks(
                "run-7",
                &[
                    OptimizerBlockInput {
                        block_id: "block-a".to_string(),
                        payload: b"state-a".to_vec(),
                        step: 1,
                        shard_key: Some("group:0".to_string()),
                        metadata: Value::Null,
                    },
                    OptimizerBlockInput {
                        block_id: "block-b".to_string(),
                        payload: b"state-b".to_vec(),
                        step: 1,
                        shard_key: Some("group:0".to_string()),
                        metadata: serde_json::json!({"kind":"adamw"}),
                    },
                ],
            )
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|item| item.staged));

        store.flush_origin("run-7").unwrap();
        let payloads = store
            .load_blocks("run-7", &[String::from("block-a"), String::from("block-b")])
            .unwrap();
        assert_eq!(payloads["block-a"], b"state-a");
        assert_eq!(payloads["block-b"], b"state-b");
    }
}
