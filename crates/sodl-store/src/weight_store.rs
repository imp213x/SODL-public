//! Weight cluster storage: compress → encrypt → CAS → store.
//!
//! This module provides [`WeightBlobStore`], which serialises [`WeightCluster`]
//! values, compresses them with zstd, optionally encrypts through the SODL
//! crypto pipeline, and stores them as content-addressed blobs.
//!
//! The companion [`WeightPinRegistry`] manages a hot/cold RAM cache with
//! SODL-style refcounting and GC-based eviction.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{Read, Write};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use sodl_cas::{compute_blob_id, verify_integrity, BlobStore, HashAlg};
use sodl_core::{BlobId, ClusterId, OriginId, Result, SodlError, WeightCluster, WeightPinReason};
use sodl_crypto::Crypto;

// ---------------------------------------------------------------------------
// Weight blob serialisation helpers
// ---------------------------------------------------------------------------

/// Serialise a `WeightCluster` to compact JSON bytes.
fn serialize_cluster(cluster: &WeightCluster) -> Result<Vec<u8>> {
    serde_json::to_vec(cluster).map_err(|e| SodlError::Serialization(e.to_string()))
}

/// Deserialise a `WeightCluster` from JSON bytes.
fn deserialize_cluster(data: &[u8]) -> Result<WeightCluster> {
    serde_json::from_slice(data).map_err(|e| SodlError::Serialization(e.to_string()))
}

fn serialize_manifest(manifest: &MultiLayerShardManifest) -> Result<Vec<u8>> {
    serde_json::to_vec(manifest).map_err(|e| SodlError::Serialization(e.to_string()))
}

fn deserialize_manifest(data: &[u8]) -> Result<MultiLayerShardManifest> {
    serde_json::from_slice(data).map_err(|e| SodlError::Serialization(e.to_string()))
}

// ---------------------------------------------------------------------------
// Compression helpers
// ---------------------------------------------------------------------------

const DEFAULT_ZSTD_LEVEL: i32 = 3;

/// Compress bytes with zstd.
fn compress(data: &[u8], level: i32) -> Result<Vec<u8>> {
    let mut encoder =
        zstd::Encoder::new(Vec::new(), level).map_err(|e| SodlError::Compression(e.to_string()))?;
    encoder
        .write_all(data)
        .map_err(|e| SodlError::Compression(e.to_string()))?;
    encoder
        .finish()
        .map_err(|e| SodlError::Compression(e.to_string()))
}

/// Decompress zstd bytes.
fn decompress(data: &[u8]) -> Result<Vec<u8>> {
    let mut decoder =
        zstd::Decoder::new(data).map_err(|e| SodlError::Compression(e.to_string()))?;
    let mut out = Vec::new();
    decoder
        .read_to_end(&mut out)
        .map_err(|e| SodlError::Compression(e.to_string()))?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// WeightBlobStore — the core storage API
// ---------------------------------------------------------------------------

/// Stores weight clusters as compressed, optionally encrypted, content-addressed blobs.
///
/// Pipeline:
/// ```text
/// put:  cluster → serialise → compress → (encrypt) → CAS hash → blob store
/// get:  blob id → fetch → (verify hash) → (decrypt) → decompress → deserialise
/// ```
pub struct WeightBlobStore<'a> {
    store: &'a dyn BlobStore,
    crypto: Option<&'a dyn Crypto>,
    hash_alg: HashAlg,
    compression_level: i32,
}

/// Statistics returned after a store operation.
#[derive(Debug, Clone)]
pub struct StoreStats {
    pub blob_id: BlobId,
    pub raw_bytes: usize,
    pub compressed_bytes: usize,
    pub stored_bytes: usize,
    pub was_deduped: bool,
}

/// Layer-scoped cluster shard metadata for multi-layer exports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerShardRecord {
    pub layer_name: String,
    pub cluster_ids: Vec<ClusterId>,
    pub total_clusters: usize,
    pub total_raw_bytes: usize,
    pub total_stored_bytes: usize,
}

/// Manifest describing a multi-layer sharded model export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLayerShardManifest {
    pub origin_id: OriginId,
    pub layers: Vec<LayerShardRecord>,
    pub created_at: time::OffsetDateTime,
}

/// Aggregate statistics for storing a multi-layer manifest and its shards.
#[derive(Debug, Clone)]
pub struct MultiLayerStoreStats {
    pub manifest: MultiLayerShardManifest,
    pub manifest_blob_id: BlobId,
    pub total_clusters: usize,
    pub total_raw_bytes: usize,
    pub total_stored_bytes: usize,
}

impl<'a> WeightBlobStore<'a> {
    /// Create a weight store with optional encryption.
    pub fn new(
        store: &'a dyn BlobStore,
        crypto: Option<&'a dyn Crypto>,
        hash_alg: HashAlg,
    ) -> Self {
        Self {
            store,
            crypto,
            hash_alg,
            compression_level: DEFAULT_ZSTD_LEVEL,
        }
    }

    /// Adjust zstd compression level (1 = fast, 22 = max compression, default 3).
    pub fn with_compression_level(mut self, level: i32) -> Self {
        self.compression_level = level;
        self
    }

    /// Store a weight cluster, returning its content-addressed ID and stats.
    pub fn put(&self, origin_id: OriginId, cluster: &WeightCluster) -> Result<StoreStats> {
        // 1. Serialise
        let raw = serialize_cluster(cluster)?;
        let raw_bytes = raw.len();

        // 2. Compress
        let compressed = compress(&raw, self.compression_level)?;
        let compressed_bytes = compressed.len();

        // 3. Optionally encrypt
        let to_store = if let Some(crypto) = self.crypto {
            let ct = crypto.encrypt_for_origin(origin_id, Bytes::from(compressed))?;
            ct.to_vec()
        } else {
            compressed
        };
        let stored_bytes = to_store.len();

        // 4. CAS hash and store
        let blob_id = compute_blob_id(&to_store, self.hash_alg);
        let was_deduped = self.store.has(&blob_id)?;
        if !was_deduped {
            self.store.put(&blob_id, Bytes::from(to_store))?;
        }

        Ok(StoreStats {
            blob_id,
            raw_bytes,
            compressed_bytes,
            stored_bytes,
            was_deduped,
        })
    }

    /// Fetch and reconstruct a weight cluster by its blob ID.
    pub fn get(&self, origin_id: OriginId, blob_id: &BlobId) -> Result<WeightCluster> {
        // 1. Fetch raw bytes
        let stored = self.store.get(blob_id)?;

        // 2. Verify integrity
        verify_integrity(blob_id, &stored)?;

        // 3. Optionally decrypt
        let compressed = if let Some(crypto) = self.crypto {
            crypto.decrypt_for_origin(origin_id, stored)?
        } else {
            stored
        };

        // 4. Decompress
        let raw = decompress(&compressed)?;

        // 5. Deserialise
        deserialize_cluster(&raw)
    }

    /// Check if a blob already exists (dedup check).
    pub fn has(&self, blob_id: &BlobId) -> Result<bool> {
        self.store.has(blob_id)
    }

    /// Delete a weight cluster blob.
    pub fn delete(&self, blob_id: &BlobId) -> Result<()> {
        self.store.delete(blob_id)
    }

    /// Store all clusters for one named layer and return its shard record.
    pub fn put_layer_shard(
        &self,
        origin_id: OriginId,
        layer_name: &str,
        clusters: &[WeightCluster],
    ) -> Result<LayerShardRecord> {
        let mut cluster_ids = Vec::with_capacity(clusters.len());
        let mut total_raw_bytes = 0usize;
        let mut total_stored_bytes = 0usize;
        for cluster in clusters {
            let stats = self.put(origin_id, cluster)?;
            cluster_ids.push(stats.blob_id);
            total_raw_bytes += stats.raw_bytes;
            total_stored_bytes += stats.stored_bytes;
        }
        Ok(LayerShardRecord {
            layer_name: layer_name.to_string(),
            cluster_ids,
            total_clusters: clusters.len(),
            total_raw_bytes,
            total_stored_bytes,
        })
    }

    /// Load all clusters referenced by one layer shard record.
    pub fn get_layer_shard(
        &self,
        origin_id: OriginId,
        record: &LayerShardRecord,
    ) -> Result<Vec<WeightCluster>> {
        record
            .cluster_ids
            .iter()
            .map(|cluster_id| self.get(origin_id, cluster_id))
            .collect()
    }

    /// Store a full multi-layer export plus a manifest blob that can be resumed later.
    pub fn put_multi_layer_shards(
        &self,
        origin_id: OriginId,
        layers: &BTreeMap<String, Vec<WeightCluster>>,
    ) -> Result<MultiLayerStoreStats> {
        let mut layer_records = Vec::with_capacity(layers.len());
        let mut total_clusters = 0usize;
        let mut total_raw_bytes = 0usize;
        let mut total_stored_bytes = 0usize;

        for (layer_name, clusters) in layers {
            let record = self.put_layer_shard(origin_id, layer_name, clusters)?;
            total_clusters += record.total_clusters;
            total_raw_bytes += record.total_raw_bytes;
            total_stored_bytes += record.total_stored_bytes;
            layer_records.push(record);
        }

        let manifest = MultiLayerShardManifest {
            origin_id,
            layers: layer_records,
            created_at: time::OffsetDateTime::now_utc(),
        };
        let manifest_blob_id = self.store_multilayer_manifest(origin_id, &manifest)?;

        Ok(MultiLayerStoreStats {
            manifest,
            manifest_blob_id,
            total_clusters,
            total_raw_bytes,
            total_stored_bytes,
        })
    }

    /// Store the multi-layer manifest itself as a content-addressed blob.
    pub fn store_multilayer_manifest(
        &self,
        origin_id: OriginId,
        manifest: &MultiLayerShardManifest,
    ) -> Result<BlobId> {
        let raw = serialize_manifest(manifest)?;
        let compressed = compress(&raw, self.compression_level)?;
        let to_store = if let Some(crypto) = self.crypto {
            crypto
                .encrypt_for_origin(origin_id, Bytes::from(compressed))?
                .to_vec()
        } else {
            compressed
        };
        let blob_id = compute_blob_id(&to_store, self.hash_alg);
        if !self.store.has(&blob_id)? {
            self.store.put(&blob_id, Bytes::from(to_store))?;
        }
        Ok(blob_id)
    }

    /// Load a previously stored multi-layer manifest.
    pub fn load_multilayer_manifest(
        &self,
        origin_id: OriginId,
        blob_id: &BlobId,
    ) -> Result<MultiLayerShardManifest> {
        let stored = self.store.get(blob_id)?;
        verify_integrity(blob_id, &stored)?;
        let compressed = if let Some(crypto) = self.crypto {
            crypto.decrypt_for_origin(origin_id, stored)?
        } else {
            stored
        };
        let raw = decompress(&compressed)?;
        deserialize_manifest(&raw)
    }
}

/// Tracks gradient activity per cluster and surfaces prune candidates.
pub struct GradientRefCounter {
    zero_streaks: Arc<RwLock<HashMap<String, usize>>>,
    update_counts: Arc<RwLock<HashMap<String, u64>>>,
    prune_after_zero_batches: usize,
}

impl GradientRefCounter {
    pub fn new(prune_after_zero_batches: usize) -> Self {
        Self {
            zero_streaks: Arc::new(RwLock::new(HashMap::new())),
            update_counts: Arc::new(RwLock::new(HashMap::new())),
            prune_after_zero_batches: prune_after_zero_batches.max(1),
        }
    }

    /// Observe one training batch and return clusters eligible for pruning.
    pub fn observe_batch(
        &self,
        tracked_cluster_ids: &[ClusterId],
        active_cluster_ids: &[ClusterId],
    ) -> Result<Vec<ClusterId>> {
        let active: HashSet<String> = active_cluster_ids.iter().map(|id| id.0.clone()).collect();
        let mut zero_streaks = self
            .zero_streaks
            .write()
            .map_err(|e| SodlError::Io(e.to_string()))?;
        let mut update_counts = self
            .update_counts
            .write()
            .map_err(|e| SodlError::Io(e.to_string()))?;

        let mut prune_candidates = Vec::new();
        for cluster_id in tracked_cluster_ids {
            if active.contains(&cluster_id.0) {
                zero_streaks.insert(cluster_id.0.clone(), 0);
                update_counts
                    .entry(cluster_id.0.clone())
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            } else {
                let streak = zero_streaks
                    .entry(cluster_id.0.clone())
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
                if *streak >= self.prune_after_zero_batches {
                    prune_candidates.push(cluster_id.clone());
                }
            }
        }
        Ok(prune_candidates)
    }

    pub fn update_count(&self, cluster_id: &ClusterId) -> Result<u64> {
        Ok(*self
            .update_counts
            .read()
            .map_err(|e| SodlError::Io(e.to_string()))?
            .get(&cluster_id.0)
            .unwrap_or(&0))
    }

    pub fn zero_streak(&self, cluster_id: &ClusterId) -> Result<usize> {
        Ok(*self
            .zero_streaks
            .read()
            .map_err(|e| SodlError::Io(e.to_string()))?
            .get(&cluster_id.0)
            .unwrap_or(&0))
    }
}

// ---------------------------------------------------------------------------
// WeightPinRegistry — hot/cold cluster RAM cache with GC
// ---------------------------------------------------------------------------

/// A pinned cluster held in the hot RAM cache.
struct PinnedEntry {
    cluster: WeightCluster,
    reason: WeightPinReason,
    last_accessed: Instant,
}

/// In-memory hot/cold weight cluster cache with SODL-style refcounting and eviction.
///
/// - Identity and logic clusters are always pinned and never evicted.
/// - Other clusters are evicted LRU when the cache exceeds `max_entries`.
pub struct WeightPinRegistry {
    entries: Arc<RwLock<HashMap<String, PinnedEntry>>>,
    access_counts: Arc<RwLock<HashMap<String, u64>>>,
    max_entries: usize,
}

impl WeightPinRegistry {
    /// Create a new pin registry with the given cache capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            access_counts: Arc::new(RwLock::new(HashMap::new())),
            max_entries,
        }
    }

    /// Pin a cluster in the hot cache with the given reason.
    pub fn pin(
        &self,
        cluster_id: &ClusterId,
        cluster: WeightCluster,
        reason: WeightPinReason,
    ) -> Result<()> {
        self.maybe_evict()?;

        let key = cluster_id.0.clone();
        let entry = PinnedEntry {
            cluster,
            reason,
            last_accessed: Instant::now(),
        };

        self.entries
            .write()
            .map_err(|e| SodlError::Io(e.to_string()))?
            .insert(key.clone(), entry);

        self.access_counts
            .write()
            .map_err(|e| SodlError::Io(e.to_string()))?
            .entry(key)
            .and_modify(|c| *c += 1)
            .or_insert(1);

        Ok(())
    }

    /// Get a pinned cluster if it exists in the hot cache, updating access stats.
    pub fn get(&self, cluster_id: &ClusterId) -> Result<Option<WeightCluster>> {
        let key = &cluster_id.0;

        // Update access count
        self.access_counts
            .write()
            .map_err(|e| SodlError::Io(e.to_string()))?
            .entry(key.clone())
            .and_modify(|c| *c += 1)
            .or_insert(1);

        // Get cluster and refresh LRU timestamp.
        let mut entries = self
            .entries
            .write()
            .map_err(|e| SodlError::Io(e.to_string()))?;

        match entries.get_mut(key) {
            Some(entry) => {
                entry.last_accessed = Instant::now();
                Ok(Some(entry.cluster.clone()))
            }
            None => Ok(None),
        }
    }

    /// Check if a cluster is pinned in the hot cache.
    pub fn is_pinned(&self, cluster_id: &ClusterId) -> Result<bool> {
        Ok(self
            .entries
            .read()
            .map_err(|e| SodlError::Io(e.to_string()))?
            .contains_key(&cluster_id.0))
    }

    /// Current number of pinned entries.
    pub fn len(&self) -> Result<usize> {
        Ok(self
            .entries
            .read()
            .map_err(|e| SodlError::Io(e.to_string()))?
            .len())
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self
            .entries
            .read()
            .map_err(|e| SodlError::Io(e.to_string()))?
            .is_empty())
    }

    /// Get the access count (refcount) for a cluster.
    pub fn refcount(&self, cluster_id: &ClusterId) -> Result<u64> {
        let counts = self
            .access_counts
            .read()
            .map_err(|e| SodlError::Io(e.to_string()))?;
        Ok(*counts.get(&cluster_id.0).unwrap_or(&0))
    }

    /// Explicitly unpin a cluster (removes from hot cache).
    /// Identity- and logic-pinned clusters cannot be unpinned.
    pub fn unpin(&self, cluster_id: &ClusterId) -> Result<bool> {
        let key = &cluster_id.0;
        let entries = self
            .entries
            .read()
            .map_err(|e| SodlError::Io(e.to_string()))?;

        if let Some(entry) = entries.get(key) {
            if matches!(
                entry.reason,
                WeightPinReason::Identity | WeightPinReason::Logic
            ) {
                return Err(SodlError::WeightStore(
                    "cannot unpin identity- or logic-pinned cluster".into(),
                ));
            }
        }
        drop(entries);

        let removed = self
            .entries
            .write()
            .map_err(|e| SodlError::Io(e.to_string()))?
            .remove(key)
            .is_some();
        Ok(removed)
    }

    /// Evict lowest-refcount non-identity clusters until under max_entries.
    /// Ties are broken by least-recently-used order.
    fn maybe_evict(&self) -> Result<()> {
        let len = self
            .entries
            .read()
            .map_err(|e| SodlError::Io(e.to_string()))?
            .len();

        if len < self.max_entries {
            return Ok(());
        }

        let counts = self
            .access_counts
            .read()
            .map_err(|e| SodlError::Io(e.to_string()))?
            .clone();

        let entries = self
            .entries
            .read()
            .map_err(|e| SodlError::Io(e.to_string()))?;

        // Find eviction candidates: evictable entries, sorted by refcount ascending
        let mut candidates: Vec<(String, u64, Instant)> = entries
            .iter()
            .filter(|(_, e)| {
                !matches!(e.reason, WeightPinReason::Identity | WeightPinReason::Logic)
            })
            .map(|(k, e)| (k.clone(), *counts.get(k).unwrap_or(&0), e.last_accessed))
            .collect();
        candidates.sort_by_key(|(_, count, last_accessed)| (*count, *last_accessed));

        drop(entries);

        // Evict until under limit
        let to_evict = len.saturating_sub(self.max_entries) + 1; // +1 for the incoming entry
        let mut write_entries = self
            .entries
            .write()
            .map_err(|e| SodlError::Io(e.to_string()))?;

        for (key, _, _) in candidates.iter().take(to_evict) {
            write_entries.remove(key);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sodl_cas::MemBlobStore;
    use sodl_core::new_origin_id;
    use sodl_crypto::{DevXorCrypto, NullCrypto};
    use std::collections::BTreeMap;

    fn sample_cluster(dim: usize, n_members: usize) -> WeightCluster {
        WeightCluster {
            cluster_id: None,
            centroid: vec![0.5_f32; dim],
            member_token_ids: (0..n_members as u32).collect(),
            offsets: (0..n_members).map(|i| vec![0.01 * i as f32; dim]).collect(),
            dim,
        }
    }

    // -- Compression round-trip -------------------------------------------------

    #[test]
    fn compress_decompress_roundtrip() {
        let data = b"hello sodl weight store! ".repeat(100);
        let compressed = compress(&data, DEFAULT_ZSTD_LEVEL).unwrap();
        assert!(compressed.len() < data.len());

        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    // -- WeightBlobStore: no crypto ---------------------------------------------

    #[test]
    fn put_get_no_crypto() {
        let blob_store = MemBlobStore::new();
        let ws = WeightBlobStore::new(&blob_store, None, HashAlg::Blake3);
        let origin = new_origin_id();
        let cluster = sample_cluster(64, 10);

        let stats = ws.put(origin, &cluster).unwrap();
        assert!(!stats.was_deduped);
        assert!(stats.compressed_bytes < stats.raw_bytes);

        let back = ws.get(origin, &stats.blob_id).unwrap();
        assert_eq!(back.centroid, cluster.centroid);
        assert_eq!(back.member_token_ids, cluster.member_token_ids);
        assert_eq!(back.offsets.len(), cluster.offsets.len());
        assert_eq!(back.dim, 64);
    }

    #[test]
    fn put_deduplicates() {
        let blob_store = MemBlobStore::new();
        let ws = WeightBlobStore::new(&blob_store, None, HashAlg::Blake3);
        let origin = new_origin_id();
        let cluster = sample_cluster(32, 5);

        let s1 = ws.put(origin, &cluster).unwrap();
        let s2 = ws.put(origin, &cluster).unwrap();

        assert_eq!(s1.blob_id.0, s2.blob_id.0);
        assert!(!s1.was_deduped);
        assert!(s2.was_deduped); // second store detected existing blob
    }

    // -- WeightBlobStore: with NullCrypto ----------------------------------------

    #[test]
    fn put_get_null_crypto() {
        let blob_store = MemBlobStore::new();
        let crypto = NullCrypto::default();
        let ws = WeightBlobStore::new(&blob_store, Some(&crypto), HashAlg::Blake3);
        let origin = new_origin_id();
        let cluster = sample_cluster(128, 20);

        let stats = ws.put(origin, &cluster).unwrap();
        let back = ws.get(origin, &stats.blob_id).unwrap();
        assert_eq!(back.centroid, cluster.centroid);
        assert_eq!(back.dim, 128);
    }

    // -- WeightBlobStore: with DevXorCrypto --------------------------------------

    #[test]
    fn put_get_xor_crypto() {
        let blob_store = MemBlobStore::new();
        let crypto = DevXorCrypto::new(0xAB);
        let ws = WeightBlobStore::new(&blob_store, Some(&crypto), HashAlg::Blake3);
        let origin = new_origin_id();
        let cluster = sample_cluster(64, 15);

        let stats = ws.put(origin, &cluster).unwrap();
        let back = ws.get(origin, &stats.blob_id).unwrap();
        assert_eq!(back.centroid, cluster.centroid);
        assert_eq!(back.member_token_ids, cluster.member_token_ids);
    }

    #[test]
    fn xor_crypto_dedupes_within_origin() {
        let blob_store = MemBlobStore::new();
        let crypto = DevXorCrypto::new(0xAB);
        let ws = WeightBlobStore::new(&blob_store, Some(&crypto), HashAlg::Blake3);
        let origin = new_origin_id();
        let cluster = sample_cluster(32, 5);

        let s1 = ws.put(origin, &cluster).unwrap();
        let s2 = ws.put(origin, &cluster).unwrap();
        assert_eq!(s1.blob_id.0, s2.blob_id.0); // deterministic crypto preserves dedup
    }

    #[test]
    fn xor_crypto_differs_across_origins() {
        let blob_store = MemBlobStore::new();
        let crypto = DevXorCrypto::new(0xAB);
        let ws = WeightBlobStore::new(&blob_store, Some(&crypto), HashAlg::Blake3);
        let cluster = sample_cluster(32, 5);

        let s1 = ws.put(new_origin_id(), &cluster).unwrap();
        let s2 = ws.put(new_origin_id(), &cluster).unwrap();
        assert_ne!(s1.blob_id.0, s2.blob_id.0); // different origin keys
    }

    // -- WeightBlobStore: integrity verification ---------------------------------

    #[test]
    fn integrity_failure_detected() {
        let blob_store = MemBlobStore::new();
        let ws = WeightBlobStore::new(&blob_store, None, HashAlg::Blake3);
        let origin = new_origin_id();
        let cluster = sample_cluster(16, 3);

        let stats = ws.put(origin, &cluster).unwrap();

        // Tamper with stored bytes
        let mut tampered = blob_store.get(&stats.blob_id).unwrap().to_vec();
        tampered[0] ^= 0xFF;
        blob_store
            .put(&stats.blob_id, Bytes::from(tampered))
            .unwrap();

        let err = ws.get(origin, &stats.blob_id).unwrap_err();
        assert!(matches!(err, SodlError::Integrity));
    }

    #[test]
    fn multilayer_manifest_roundtrip() {
        let blob_store = MemBlobStore::new();
        let ws = WeightBlobStore::new(&blob_store, None, HashAlg::Blake3);
        let origin = new_origin_id();

        let mut layers = BTreeMap::new();
        layers.insert("embed_tokens".to_string(), vec![sample_cluster(16, 4)]);
        layers.insert(
            "layer.0.attn".to_string(),
            vec![sample_cluster(16, 3), sample_cluster(16, 2)],
        );

        let stats = ws.put_multi_layer_shards(origin, &layers).unwrap();
        assert_eq!(stats.total_clusters, 3);
        assert_eq!(stats.manifest.layers.len(), 2);

        let manifest = ws
            .load_multilayer_manifest(origin, &stats.manifest_blob_id)
            .unwrap();
        assert_eq!(manifest.layers.len(), 2);
        assert_eq!(manifest.layers[0].layer_name, "embed_tokens");

        let loaded = ws.get_layer_shard(origin, &manifest.layers[1]).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].dim, 16);
    }

    #[test]
    fn gradient_ref_counter_marks_prune_candidates() {
        let tracker = GradientRefCounter::new(2);
        let tracked = vec![
            BlobId("cluster:a".into()),
            BlobId("cluster:b".into()),
            BlobId("cluster:c".into()),
        ];

        let first = tracker
            .observe_batch(&tracked, &[tracked[0].clone(), tracked[2].clone()])
            .unwrap();
        assert!(first.is_empty());
        assert_eq!(tracker.update_count(&tracked[0]).unwrap(), 1);
        assert_eq!(tracker.zero_streak(&tracked[1]).unwrap(), 1);

        let second = tracker
            .observe_batch(&tracked, &[tracked[2].clone()])
            .unwrap();
        assert_eq!(second, vec![tracked[1].clone()]);
        assert_eq!(tracker.zero_streak(&tracked[1]).unwrap(), 2);
        assert_eq!(tracker.zero_streak(&tracked[0]).unwrap(), 1);
    }

    // -- WeightPinRegistry -------------------------------------------------------

    #[test]
    fn pin_and_get() {
        let registry = WeightPinRegistry::new(10);
        let cluster = sample_cluster(32, 5);
        let id = BlobId("test:001".into());

        registry
            .pin(&id, cluster.clone(), WeightPinReason::FrequentUse)
            .unwrap();

        assert!(registry.is_pinned(&id).unwrap());
        let got = registry.get(&id).unwrap().unwrap();
        assert_eq!(got.centroid, cluster.centroid);
    }

    #[test]
    fn unpin_works() {
        let registry = WeightPinRegistry::new(10);
        let cluster = sample_cluster(32, 5);
        let id = BlobId("test:002".into());

        registry
            .pin(&id, cluster, WeightPinReason::FrequentUse)
            .unwrap();
        assert!(registry.is_pinned(&id).unwrap());

        let removed = registry.unpin(&id).unwrap();
        assert!(removed);
        assert!(!registry.is_pinned(&id).unwrap());
    }

    #[test]
    fn identity_pin_cannot_be_unpinned() {
        let registry = WeightPinRegistry::new(10);
        let cluster = sample_cluster(32, 5);
        let id = BlobId("test:003".into());

        registry
            .pin(&id, cluster, WeightPinReason::Identity)
            .unwrap();
        let err = registry.unpin(&id).unwrap_err();
        assert!(matches!(err, SodlError::WeightStore(_)));
    }

    #[test]
    fn logic_pin_cannot_be_unpinned_or_evicted() {
        let registry = WeightPinRegistry::new(2);
        let logic_id = BlobId("test:logic".into());
        registry
            .pin(&logic_id, sample_cluster(16, 2), WeightPinReason::Logic)
            .unwrap();
        registry
            .pin(
                &BlobId("test:regular".into()),
                sample_cluster(16, 2),
                WeightPinReason::FrequentUse,
            )
            .unwrap();
        registry
            .pin(
                &BlobId("test:overflow".into()),
                sample_cluster(16, 2),
                WeightPinReason::FrequentUse,
            )
            .unwrap();

        assert!(registry.is_pinned(&logic_id).unwrap());
        let err = registry.unpin(&logic_id).unwrap_err();
        assert!(matches!(err, SodlError::WeightStore(_)));
    }

    #[test]
    fn eviction_removes_lowest_refcount() {
        let registry = WeightPinRegistry::new(3); // max 3 entries

        // Pin 3 clusters
        for i in 0..3 {
            let id = BlobId(format!("test:evict_{i}"));
            let cluster = sample_cluster(16, 2);
            registry
                .pin(&id, cluster, WeightPinReason::FrequentUse)
                .unwrap();
        }

        // Access cluster 1 and 2 more to raise their refcounts
        let id1 = BlobId("test:evict_1".into());
        let id2 = BlobId("test:evict_2".into());
        registry.get(&id1).unwrap();
        registry.get(&id1).unwrap();
        registry.get(&id2).unwrap();

        // Pin a 4th — should evict cluster 0 (lowest refcount)
        let id3 = BlobId("test:evict_3".into());
        registry
            .pin(&id3, sample_cluster(16, 2), WeightPinReason::FrequentUse)
            .unwrap();

        let id0 = BlobId("test:evict_0".into());
        assert!(!registry.is_pinned(&id0).unwrap()); // evicted
        assert!(registry.is_pinned(&id1).unwrap()); // kept (high refcount)
    }

    #[test]
    fn identity_pins_survive_eviction() {
        let registry = WeightPinRegistry::new(2); // very small cache

        // Pin an identity cluster
        let id_identity = BlobId("test:identity".into());
        registry
            .pin(
                &id_identity,
                sample_cluster(16, 2),
                WeightPinReason::Identity,
            )
            .unwrap();

        // Fill remaining slot
        let id1 = BlobId("test:regular_1".into());
        registry
            .pin(&id1, sample_cluster(16, 2), WeightPinReason::FrequentUse)
            .unwrap();

        // Pin another — eviction should NOT touch the identity cluster
        let id2 = BlobId("test:regular_2".into());
        registry
            .pin(&id2, sample_cluster(16, 2), WeightPinReason::FrequentUse)
            .unwrap();

        assert!(registry.is_pinned(&id_identity).unwrap()); // ✓ identity survived
    }

    #[test]
    fn refcount_tracks_accesses() {
        let registry = WeightPinRegistry::new(10);
        let id = BlobId("test:refcount".into());
        registry
            .pin(&id, sample_cluster(16, 2), WeightPinReason::FrequentUse)
            .unwrap();

        assert_eq!(registry.refcount(&id).unwrap(), 1); // 1 from pin
        registry.get(&id).unwrap();
        assert_eq!(registry.refcount(&id).unwrap(), 2);
        registry.get(&id).unwrap();
        assert_eq!(registry.refcount(&id).unwrap(), 3);
    }
}
