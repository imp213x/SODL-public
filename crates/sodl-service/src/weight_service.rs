//! Weight Store Service Facade.
//!
//! High-level API that unifies [`WeightBlobStore`] and [`WeightPinRegistry`]
//! into a single ergonomic service. Follows the same facade pattern as
//! [`SodlService`](crate::SodlService).

use sodl_cas::{BlobStore, HashAlg};
use sodl_core::{
    new_origin_id, BlobId, ClusterId, OriginId, Result, SodlError, WeightCluster, WeightOrigin,
    WeightPinReason,
};
use sodl_crypto::Crypto;
use sodl_store::weight_store::{
    LayerShardRecord, MultiLayerShardManifest, MultiLayerStoreStats, StoreStats, WeightBlobStore,
    WeightPinRegistry,
};

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// ---------------------------------------------------------------------------
// WeightStoreService — the unified facade
// ---------------------------------------------------------------------------

/// High-level weight store service combining persistent blob storage with
/// in-memory hot/cold caching.
///
/// # Usage
///
/// ```text
/// let svc = WeightStoreService::new(&blob_store, Some(&crypto), HashAlg::Blake3, 256);
///
/// // Store clusters for a model
/// let origin_id = svc.create_model("carla-qwen3-4b", "Q4_K_M")?;
/// let stats = svc.store_cluster(origin_id, &cluster)?;
///
/// // Fetch with auto-caching
/// let cluster = svc.load_cluster(origin_id, &blob_id)?;
///
/// // Pin identity-critical clusters
/// svc.pin_identity_cluster(origin_id, &blob_id)?;
/// ```
pub struct WeightStoreService<'a> {
    store: WeightBlobStore<'a>,
    pin_registry: WeightPinRegistry,
    origin_registry: WeightOriginRegistry,
}

/// Result of creating a new model origin.
#[derive(Debug, Clone)]
pub struct ModelOrigin {
    pub origin_id: OriginId,
    pub model_name: String,
}

/// Summary of a bulk import operation.
#[derive(Debug, Clone)]
pub struct ImportSummary {
    pub origin_id: OriginId,
    pub total_clusters: usize,
    pub total_blobs_stored: usize,
    pub deduped_blobs: usize,
    pub total_raw_bytes: usize,
    pub total_stored_bytes: usize,
    pub cluster_ids: Vec<ClusterId>,
}

#[derive(Clone, Default)]
pub struct WeightOriginRegistry {
    origins: Arc<RwLock<HashMap<OriginId, WeightOrigin>>>,
    name_index: Arc<RwLock<HashMap<String, OriginId>>>,
}

impl WeightOriginRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_model(&self, model_name: &str, quantization: &str) -> Result<ModelOrigin> {
        let origin_id = new_origin_id();
        let origin = WeightOrigin {
            origin_id,
            model_name: model_name.to_string(),
            num_clusters: 0,
            quantization: quantization.to_string(),
            created_at: time::OffsetDateTime::now_utc(),
        };

        self.origins
            .write()
            .map_err(|e| SodlError::Io(e.to_string()))?
            .insert(origin_id, origin);

        self.name_index
            .write()
            .map_err(|e| SodlError::Io(e.to_string()))?
            .insert(model_name.to_string(), origin_id);

        Ok(ModelOrigin {
            origin_id,
            model_name: model_name.to_string(),
        })
    }

    pub fn register_model(
        &self,
        origin_id: OriginId,
        model_name: &str,
        quantization: &str,
        num_clusters: usize,
    ) -> Result<ModelOrigin> {
        if let Some(existing) = self
            .origins
            .read()
            .map_err(|e| SodlError::Io(e.to_string()))?
            .get(&origin_id)
            .cloned()
        {
            let mut index = self
                .name_index
                .write()
                .map_err(|e| SodlError::Io(e.to_string()))?;
            index.insert(existing.model_name.clone(), existing.origin_id);
            index.insert(model_name.to_string(), existing.origin_id);
            return Ok(ModelOrigin {
                origin_id: existing.origin_id,
                model_name: existing.model_name,
            });
        }

        let origin = WeightOrigin {
            origin_id,
            model_name: model_name.to_string(),
            num_clusters,
            quantization: quantization.to_string(),
            created_at: time::OffsetDateTime::now_utc(),
        };
        self.origins
            .write()
            .map_err(|e| SodlError::Io(e.to_string()))?
            .insert(origin_id, origin);
        self.name_index
            .write()
            .map_err(|e| SodlError::Io(e.to_string()))?
            .insert(model_name.to_string(), origin_id);

        Ok(ModelOrigin {
            origin_id,
            model_name: model_name.to_string(),
        })
    }

    pub fn ensure_model(
        &self,
        model_name: &str,
        quantization: &str,
        origin_id: Option<OriginId>,
        num_clusters: usize,
    ) -> Result<ModelOrigin> {
        if let Some(origin_id) = origin_id {
            return self.register_model(origin_id, model_name, quantization, num_clusters);
        }
        if let Ok(existing) = self.get_model_by_name(model_name) {
            return Ok(ModelOrigin {
                origin_id: existing.origin_id,
                model_name: existing.model_name,
            });
        }
        self.create_model(model_name, quantization)
    }

    pub fn get_model_by_name(&self, model_name: &str) -> Result<WeightOrigin> {
        let idx = self
            .name_index
            .read()
            .map_err(|e| SodlError::Io(e.to_string()))?;
        let origin_id = idx.get(model_name).ok_or_else(|| SodlError::NotFound)?;
        let origins = self
            .origins
            .read()
            .map_err(|e| SodlError::Io(e.to_string()))?;
        origins.get(origin_id).cloned().ok_or(SodlError::NotFound)
    }

    pub fn get_model(&self, origin_id: OriginId) -> Result<WeightOrigin> {
        self.origins
            .read()
            .map_err(|e| SodlError::Io(e.to_string()))?
            .get(&origin_id)
            .cloned()
            .ok_or(SodlError::NotFound)
    }

    pub fn increment_clusters(&self, origin_id: OriginId, amount: usize) -> Result<()> {
        let mut origins = self
            .origins
            .write()
            .map_err(|e| SodlError::Io(e.to_string()))?;
        let origin = origins.get_mut(&origin_id).ok_or(SodlError::NotFound)?;
        origin.num_clusters += amount;
        Ok(())
    }
}

impl<'a> WeightStoreService<'a> {
    /// Create a new weight store service.
    ///
    /// # Arguments
    /// - `store` — underlying CAS blob store
    /// - `crypto` — optional crypto provider for encryption
    /// - `hash_alg` — hash algorithm for CAS addressing
    /// - `cache_capacity` — max clusters to keep in the hot RAM cache
    pub fn new(
        store: &'a dyn BlobStore,
        crypto: Option<&'a dyn Crypto>,
        hash_alg: HashAlg,
        cache_capacity: usize,
    ) -> Self {
        Self {
            store: WeightBlobStore::new(store, crypto, hash_alg),
            pin_registry: WeightPinRegistry::new(cache_capacity),
            origin_registry: WeightOriginRegistry::new(),
        }
    }

    /// Set the zstd compression level (1 = fast, 22 = max, default 3).
    pub fn with_compression_level(mut self, level: i32) -> Self {
        self.store = self.store.with_compression_level(level);
        self
    }

    // -----------------------------------------------------------------------
    // Model lifecycle
    // -----------------------------------------------------------------------

    /// Register a new model origin. Returns the assigned `OriginId`.
    pub fn create_model(&self, model_name: &str, quantization: &str) -> Result<ModelOrigin> {
        self.origin_registry.create_model(model_name, quantization)
    }

    /// Register a known model origin so later work can continue the same lineage.
    pub fn register_model(
        &self,
        origin_id: OriginId,
        model_name: &str,
        quantization: &str,
        num_clusters: usize,
    ) -> Result<ModelOrigin> {
        self.origin_registry
            .register_model(origin_id, model_name, quantization, num_clusters)
    }

    /// Reuse a supplied origin ID or existing model-name mapping when possible.
    pub fn ensure_model(
        &self,
        model_name: &str,
        quantization: &str,
        origin_id: Option<OriginId>,
        num_clusters: usize,
    ) -> Result<ModelOrigin> {
        self.origin_registry
            .ensure_model(model_name, quantization, origin_id, num_clusters)
    }

    /// Look up a model origin by name.
    pub fn get_model_by_name(&self, model_name: &str) -> Result<WeightOrigin> {
        self.origin_registry.get_model_by_name(model_name)
    }

    /// Get a model origin by ID.
    pub fn get_model(&self, origin_id: OriginId) -> Result<WeightOrigin> {
        self.origin_registry.get_model(origin_id)
    }

    // -----------------------------------------------------------------------
    // Cluster storage
    // -----------------------------------------------------------------------

    /// Store a single weight cluster, returning storage stats including its blob ID.
    pub fn store_cluster(
        &self,
        origin_id: OriginId,
        cluster: &WeightCluster,
    ) -> Result<StoreStats> {
        let stats = self.store.put(origin_id, cluster)?;

        // Update cluster count on origin
        if !stats.was_deduped {
            let _ = self.origin_registry.increment_clusters(origin_id, 1);
        }

        Ok(stats)
    }

    /// Bulk-import multiple clusters for a model.
    pub fn import_clusters(
        &self,
        origin_id: OriginId,
        clusters: &[WeightCluster],
    ) -> Result<ImportSummary> {
        let mut cluster_ids = Vec::with_capacity(clusters.len());
        let mut total_blobs_stored = 0usize;
        let mut deduped_blobs = 0usize;
        let mut total_raw_bytes = 0usize;
        let mut total_stored_bytes = 0usize;

        for cluster in clusters {
            let stats = self.store.put(origin_id, cluster)?;
            cluster_ids.push(stats.blob_id.clone());
            total_raw_bytes += stats.raw_bytes;
            total_stored_bytes += stats.stored_bytes;

            if stats.was_deduped {
                deduped_blobs += 1;
            } else {
                total_blobs_stored += 1;
            }
        }

        // Update cluster count
        if total_blobs_stored > 0 {
            let _ = self
                .origin_registry
                .increment_clusters(origin_id, total_blobs_stored);
        }

        Ok(ImportSummary {
            origin_id,
            total_clusters: clusters.len(),
            total_blobs_stored,
            deduped_blobs,
            total_raw_bytes,
            total_stored_bytes,
            cluster_ids,
        })
    }

    /// Store all clusters for one named layer.
    pub fn store_layer_clusters(
        &self,
        origin_id: OriginId,
        layer_name: &str,
        clusters: &[WeightCluster],
    ) -> Result<LayerShardRecord> {
        let record = self
            .store
            .put_layer_shard(origin_id, layer_name, clusters)?;
        let _ = self
            .origin_registry
            .increment_clusters(origin_id, record.total_clusters);
        Ok(record)
    }

    /// Store a full model export as layer-organized cluster shards plus manifest.
    pub fn store_model_layers(
        &self,
        origin_id: OriginId,
        layers: &BTreeMap<String, Vec<WeightCluster>>,
    ) -> Result<MultiLayerStoreStats> {
        let stats = self.store.put_multi_layer_shards(origin_id, layers)?;
        let _ = self
            .origin_registry
            .increment_clusters(origin_id, stats.total_clusters);
        Ok(stats)
    }

    /// Load all clusters referenced by a layer shard record.
    pub fn load_layer_clusters(
        &self,
        origin_id: OriginId,
        record: &LayerShardRecord,
    ) -> Result<Vec<WeightCluster>> {
        self.store.get_layer_shard(origin_id, record)
    }

    /// Load a previously stored multi-layer manifest by its blob ID.
    pub fn load_model_manifest(
        &self,
        origin_id: OriginId,
        blob_id: &BlobId,
    ) -> Result<MultiLayerShardManifest> {
        self.store.load_multilayer_manifest(origin_id, blob_id)
    }

    // -----------------------------------------------------------------------
    // Cluster loading (with auto-caching)
    // -----------------------------------------------------------------------

    /// Load a cluster, serving from the pin cache if available, otherwise
    /// fetching from blob storage and auto-pinning as `FrequentUse`.
    pub fn load_cluster(&self, origin_id: OriginId, blob_id: &BlobId) -> Result<WeightCluster> {
        // 1. Check hot cache first
        if let Some(cluster) = self.pin_registry.get(blob_id)? {
            return Ok(cluster);
        }

        // 2. Cold path: fetch from blob store
        let cluster = self.store.get(origin_id, blob_id)?;

        // 3. Auto-pin into hot cache
        let _ = self
            .pin_registry
            .pin(blob_id, cluster.clone(), WeightPinReason::FrequentUse);

        Ok(cluster)
    }

    /// Pin a cluster as identity-critical (never evicted).
    pub fn pin_identity_cluster(&self, origin_id: OriginId, blob_id: &BlobId) -> Result<()> {
        let cluster = if let Some(c) = self.pin_registry.get(blob_id)? {
            c
        } else {
            self.store.get(origin_id, blob_id)?
        };

        self.pin_registry
            .pin(blob_id, cluster, WeightPinReason::Identity)
    }

    /// Pin a cluster as logic-critical (never evicted).
    pub fn pin_logic_cluster(&self, origin_id: OriginId, blob_id: &BlobId) -> Result<()> {
        let cluster = if let Some(c) = self.pin_registry.get(blob_id)? {
            c
        } else {
            self.store.get(origin_id, blob_id)?
        };

        self.pin_registry
            .pin(blob_id, cluster, WeightPinReason::Logic)
    }

    /// Prefetch a cluster into the cache.
    pub fn prefetch_cluster(&self, origin_id: OriginId, blob_id: &BlobId) -> Result<()> {
        if self.pin_registry.is_pinned(blob_id)? {
            return Ok(()); // already cached
        }

        let cluster = self.store.get(origin_id, blob_id)?;
        self.pin_registry
            .pin(blob_id, cluster, WeightPinReason::Prefetch)
    }

    /// Explicitly evict a cluster from the hot cache.
    pub fn evict_cluster(&self, blob_id: &BlobId) -> Result<bool> {
        self.pin_registry.unpin(blob_id)
    }

    // -----------------------------------------------------------------------
    // Cache introspection
    // -----------------------------------------------------------------------

    /// Check if a cluster is in the hot cache.
    pub fn is_cached(&self, blob_id: &BlobId) -> Result<bool> {
        self.pin_registry.is_pinned(blob_id)
    }

    /// Get the access count for a cluster.
    pub fn cluster_refcount(&self, blob_id: &BlobId) -> Result<u64> {
        self.pin_registry.refcount(blob_id)
    }

    /// Current number of clusters in the hot cache.
    pub fn cache_size(&self) -> Result<usize> {
        self.pin_registry.len()
    }

    // -----------------------------------------------------------------------
    // Blob store pass-through
    // -----------------------------------------------------------------------

    /// Check if a blob exists in the persistent store.
    pub fn blob_exists(&self, blob_id: &BlobId) -> Result<bool> {
        self.store.has(blob_id)
    }

    /// Delete a cluster blob from persistent storage.
    /// Will fail if the cluster is identity-pinned in the cache.
    pub fn delete_cluster(&self, blob_id: &BlobId) -> Result<()> {
        // Ensure it's not identity-pinned
        if self.pin_registry.is_pinned(blob_id)? {
            // Try to unpin — will error if identity-pinned
            self.pin_registry.unpin(blob_id)?;
        }
        self.store.delete(blob_id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sodl_cas::MemBlobStore;
    use sodl_crypto::NullCrypto;
    use std::collections::BTreeMap;

    fn sample_cluster(dim: usize, n: usize) -> WeightCluster {
        WeightCluster {
            cluster_id: None,
            centroid: vec![0.42_f32; dim],
            member_token_ids: (0..n as u32).collect(),
            offsets: (0..n).map(|i| vec![0.001 * i as f32; dim]).collect(),
            dim,
        }
    }

    fn make_svc<'a>(store: &'a MemBlobStore, crypto: &'a NullCrypto) -> WeightStoreService<'a> {
        WeightStoreService::new(store, Some(crypto), HashAlg::Blake3, 64)
    }

    #[test]
    fn full_lifecycle() {
        let store = MemBlobStore::new();
        let crypto = NullCrypto::default();
        let svc = make_svc(&store, &crypto);

        // 1. Create model
        let model = svc.create_model("carla-qwen3-4b", "Q4_K_M").unwrap();
        assert_eq!(model.model_name, "carla-qwen3-4b");

        // 2. Store a cluster
        let cluster = sample_cluster(64, 10);
        let stats = svc.store_cluster(model.origin_id, &cluster).unwrap();
        assert!(!stats.was_deduped);
        assert!(stats.compressed_bytes < stats.raw_bytes);

        // 3. Load — should come from blob store, then be cached
        let loaded = svc.load_cluster(model.origin_id, &stats.blob_id).unwrap();
        assert_eq!(loaded.centroid, cluster.centroid);
        assert!(svc.is_cached(&stats.blob_id).unwrap());

        // 4. Second load — should come from cache
        let cached = svc.load_cluster(model.origin_id, &stats.blob_id).unwrap();
        assert_eq!(cached.centroid, cluster.centroid);
        assert_eq!(svc.cluster_refcount(&stats.blob_id).unwrap(), 3); // pin + 2 loads
    }

    #[test]
    fn model_lookup_by_name() {
        let store = MemBlobStore::new();
        let crypto = NullCrypto::default();
        let svc = make_svc(&store, &crypto);

        let model = svc.create_model("test-model", "F16").unwrap();
        let found = svc.get_model_by_name("test-model").unwrap();
        assert_eq!(found.origin_id, model.origin_id);
        assert_eq!(found.quantization, "F16");

        let err = svc.get_model_by_name("nonexistent").unwrap_err();
        assert!(matches!(err, SodlError::NotFound));
    }

    #[test]
    fn register_model_reuses_supplied_origin_id() {
        let store = MemBlobStore::new();
        let crypto = NullCrypto::default();
        let svc = make_svc(&store, &crypto);

        let origin_id = new_origin_id();
        let model = svc
            .register_model(origin_id, "stable-model", "F16", 3)
            .unwrap();
        let found = svc.get_model(origin_id).unwrap();

        assert_eq!(model.origin_id, origin_id);
        assert_eq!(found.model_name, "stable-model");
        assert_eq!(found.num_clusters, 3);
    }

    #[test]
    fn ensure_model_prefers_existing_origin_id_or_name() {
        let store = MemBlobStore::new();
        let crypto = NullCrypto::default();
        let svc = make_svc(&store, &crypto);

        let origin_id = new_origin_id();
        let first = svc
            .ensure_model("stable-model", "F16", Some(origin_id), 0)
            .unwrap();
        let second = svc.ensure_model("stable-model", "F16", None, 0).unwrap();

        assert_eq!(first.origin_id, origin_id);
        assert_eq!(second.origin_id, origin_id);
    }

    #[test]
    fn bulk_import() {
        let store = MemBlobStore::new();
        let crypto = NullCrypto::default();
        let svc = make_svc(&store, &crypto);

        let model = svc.create_model("bulk-test", "Q4_K_M").unwrap();

        let clusters: Vec<_> = (0..20)
            .map(|i| WeightCluster {
                cluster_id: None,
                centroid: vec![i as f32 * 0.1; 32],
                member_token_ids: vec![i as u32],
                offsets: vec![vec![0.0; 32]],
                dim: 32,
            })
            .collect();

        let summary = svc.import_clusters(model.origin_id, &clusters).unwrap();

        assert_eq!(summary.total_clusters, 20);
        assert_eq!(summary.total_blobs_stored, 20);
        assert_eq!(summary.deduped_blobs, 0);
        assert!(summary.total_stored_bytes < summary.total_raw_bytes); // compression works
        assert_eq!(summary.cluster_ids.len(), 20);

        // Verify model updated
        let m = svc.get_model(model.origin_id).unwrap();
        assert_eq!(m.num_clusters, 20);
    }

    #[test]
    fn bulk_import_with_duplicates() {
        let store = MemBlobStore::new();
        let crypto = NullCrypto::default();
        let svc = make_svc(&store, &crypto);

        let model = svc.create_model("dedup-test", "Q4_K_M").unwrap();

        // Same cluster repeated 5 times
        let c = sample_cluster(16, 3);
        let clusters = vec![c.clone(), c.clone(), c.clone(), c.clone(), c.clone()];

        let summary = svc.import_clusters(model.origin_id, &clusters).unwrap();

        assert_eq!(summary.total_clusters, 5);
        assert_eq!(summary.total_blobs_stored, 1); // only 1 unique blob
        assert_eq!(summary.deduped_blobs, 4);
    }

    #[test]
    fn identity_pin_protection() {
        let store = MemBlobStore::new();
        let crypto = NullCrypto::default();
        let svc = make_svc(&store, &crypto);

        let model = svc.create_model("pin-test", "Q4_K_M").unwrap();
        let cluster = sample_cluster(32, 5);
        let stats = svc.store_cluster(model.origin_id, &cluster).unwrap();

        // Pin as identity
        svc.pin_identity_cluster(model.origin_id, &stats.blob_id)
            .unwrap();

        // Cannot evict
        let err = svc.evict_cluster(&stats.blob_id).unwrap_err();
        assert!(matches!(err, SodlError::WeightStore(_)));

        // Cannot delete
        let err = svc.delete_cluster(&stats.blob_id).unwrap_err();
        assert!(matches!(err, SodlError::WeightStore(_)));
    }

    #[test]
    fn logic_pin_protection() {
        let store = MemBlobStore::new();
        let crypto = NullCrypto::default();
        let svc = make_svc(&store, &crypto);

        let model = svc.create_model("logic-pin-test", "Q4_K_M").unwrap();
        let cluster = sample_cluster(32, 5);
        let stats = svc.store_cluster(model.origin_id, &cluster).unwrap();

        svc.pin_logic_cluster(model.origin_id, &stats.blob_id)
            .unwrap();

        let err = svc.evict_cluster(&stats.blob_id).unwrap_err();
        assert!(matches!(err, SodlError::WeightStore(_)));
    }

    #[test]
    fn prefetch() {
        let store = MemBlobStore::new();
        let crypto = NullCrypto::default();
        let svc = make_svc(&store, &crypto);

        let model = svc.create_model("prefetch-test", "Q4_K_M").unwrap();
        let cluster = sample_cluster(32, 5);
        let stats = svc.store_cluster(model.origin_id, &cluster).unwrap();

        assert!(!svc.is_cached(&stats.blob_id).unwrap());

        svc.prefetch_cluster(model.origin_id, &stats.blob_id)
            .unwrap();

        assert!(svc.is_cached(&stats.blob_id).unwrap());
    }

    #[test]
    fn cache_size_tracks_correctly() {
        let store = MemBlobStore::new();
        let crypto = NullCrypto::default();
        let svc = make_svc(&store, &crypto);

        let model = svc.create_model("cache-test", "Q4_K_M").unwrap();

        assert_eq!(svc.cache_size().unwrap(), 0);

        for i in 0..5 {
            let c = WeightCluster {
                cluster_id: None,
                centroid: vec![i as f32; 16],
                member_token_ids: vec![i],
                offsets: vec![vec![0.0; 16]],
                dim: 16,
            };
            let stats = svc.store_cluster(model.origin_id, &c).unwrap();
            svc.load_cluster(model.origin_id, &stats.blob_id).unwrap();
        }

        assert_eq!(svc.cache_size().unwrap(), 5);
    }

    #[test]
    fn store_model_layers_roundtrip() {
        let store = MemBlobStore::new();
        let crypto = NullCrypto::default();
        let svc = make_svc(&store, &crypto);
        let model = svc.create_model("alpha-heart", "F32").unwrap();

        let mut layers = BTreeMap::new();
        layers.insert("embed_tokens".to_string(), vec![sample_cluster(16, 4)]);
        layers.insert("layer.0.mlp".to_string(), vec![sample_cluster(16, 3)]);

        let stats = svc.store_model_layers(model.origin_id, &layers).unwrap();
        let manifest = svc
            .load_model_manifest(model.origin_id, &stats.manifest_blob_id)
            .unwrap();
        let loaded = svc
            .load_layer_clusters(model.origin_id, &manifest.layers[0])
            .unwrap();

        assert_eq!(stats.total_clusters, 2);
        assert_eq!(manifest.layers.len(), 2);
        assert_eq!(loaded.len(), 1);
        assert_eq!(svc.get_model(model.origin_id).unwrap().num_clusters, 2);
    }
}
