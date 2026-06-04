use bytes::Bytes;
use pyo3::exceptions::{PyFileNotFoundError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};
use sodl_cas::{compute_blob_id, verify_integrity, BlobStore, FsBlobStore, HashAlg};
use sodl_crypto::{AeadCrypto, Decryptor, Encryptor};
use sodl_dist::{
    DiscoveredPeerSource, EdgeFetchSource, EdgeProvider, HttpEdgeProvider, HttpPeerClient,
    PeerAddr, StaticPeerDiscovery,
};
use sodl_fetch::{FetchPipeline, FetchSource, StoreSource};
use sodl_replica::{MemReplicaStore, ReplicaRecord, ReplicaState, ReplicaStore};
use sodl_service::checkpoint_store::{CheckpointSaveRequest, CheckpointStore};
use sodl_service::optimizer_state::{OptimizerBlockInput, OptimizerStateStore};
use sodl_service::weight_manifest::{WeightManifestCluster, WeightManifestStore};
use sodl_service::weight_service::WeightOriginRegistry;
use uuid::Uuid;

fn record_replica(
    replicas: &MemReplicaStore,
    blob_id: &str,
    node_id: &str,
    state: ReplicaState,
) -> PyResult<()> {
    replicas
        .upsert_replica(ReplicaRecord {
            blob_id: sodl_core::BlobId(blob_id.to_string()),
            node_id: node_id.to_string(),
            state,
            last_seen: time::OffsetDateTime::now_utc(),
        })
        .map_err(map_sodl_error)
}

fn map_sodl_error(err: sodl_core::SodlError) -> PyErr {
    match err {
        sodl_core::SodlError::NotFound => PyFileNotFoundError::new_err("blob not found"),
        sodl_core::SodlError::Integrity => PyValueError::new_err("integrity verification failed"),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn parse_origin_id(origin_id: &str) -> PyResult<sodl_core::OriginId> {
    let normalized = origin_id
        .strip_prefix("origin:")
        .unwrap_or(origin_id)
        .trim();
    let uuid = Uuid::parse_str(normalized)
        .map_err(|err| PyValueError::new_err(format!("invalid origin id: {err}")))?;
    Ok(sodl_core::OriginId(uuid))
}

fn weight_origin_json(origin: &sodl_core::WeightOrigin) -> PyResult<String> {
    serde_json::to_string(&serde_json::json!({
        "origin_id": format!("origin:{}", origin.origin_id.0),
        "model_name": origin.model_name,
        "num_clusters": origin.num_clusters,
        "quantization": origin.quantization,
        "created_at": origin.created_at,
    }))
    .map_err(|err| PyValueError::new_err(err.to_string()))
}

#[pyclass]
struct PyFsBlobStore {
    inner: FsBlobStore,
    root: std::path::PathBuf,
    sources: Vec<(PeerAddr, FsBlobStore)>,
    peer_urls: Vec<PeerAddr>,
    edge_urls: Vec<String>,
    replicas: MemReplicaStore,
}

#[pyclass]
struct PyOptimizerStateStore {
    inner: OptimizerStateStore,
}

#[pyclass]
struct PyCheckpointStore {
    inner: CheckpointStore,
}

#[pyclass]
struct PyWeightManifestStore {
    inner: WeightManifestStore,
}

#[pyclass]
struct PyWeightOriginRegistry {
    inner: WeightOriginRegistry,
}

#[pyclass]
struct PyAeadCrypto {
    inner: AeadCrypto,
}

#[pymethods]
impl PyOptimizerStateStore {
    #[new]
    #[pyo3(signature = (blob_root, registry_dir=None, compression_level=3, cache_capacity=32, writeback_threshold=8))]
    fn new(
        blob_root: &str,
        registry_dir: Option<&str>,
        compression_level: i32,
        cache_capacity: usize,
        writeback_threshold: usize,
    ) -> PyResult<Self> {
        let registry = registry_dir
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(blob_root).join("optimizer_registry"));
        let inner = OptimizerStateStore::open(
            std::path::PathBuf::from(blob_root),
            registry,
            compression_level,
            cache_capacity,
            writeback_threshold,
        )
        .map_err(map_sodl_error)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (origin_id, block_id, payload, step=0, shard_key=None, metadata_json=None))]
    fn store_block(
        &self,
        origin_id: &str,
        block_id: &str,
        payload: &[u8],
        step: u64,
        shard_key: Option<String>,
        metadata_json: Option<&str>,
    ) -> PyResult<String> {
        let metadata = metadata_json
            .map(|raw| {
                serde_json::from_str(raw).map_err(|err| PyValueError::new_err(err.to_string()))
            })
            .transpose()?
            .unwrap_or(serde_json::Value::Null);
        let result = self
            .inner
            .store_block(origin_id, block_id, payload, step, shard_key, metadata)
            .map_err(map_sodl_error)?;
        serde_json::to_string(&result).map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[pyo3(signature = (origin_id, block_ids, payloads, steps=None, shard_keys=None, metadata_jsons=None))]
    fn store_blocks(
        &self,
        origin_id: &str,
        block_ids: Vec<String>,
        payloads: Vec<Vec<u8>>,
        steps: Option<Vec<u64>>,
        shard_keys: Option<Vec<Option<String>>>,
        metadata_jsons: Option<Vec<Option<String>>>,
    ) -> PyResult<String> {
        if block_ids.len() != payloads.len() {
            return Err(PyValueError::new_err(
                "block_ids and payloads must have the same length",
            ));
        }
        let steps = steps.unwrap_or_else(|| vec![0; block_ids.len()]);
        if steps.len() != block_ids.len() {
            return Err(PyValueError::new_err("steps must match block_ids length"));
        }
        let shard_keys = shard_keys.unwrap_or_else(|| vec![None; block_ids.len()]);
        if shard_keys.len() != block_ids.len() {
            return Err(PyValueError::new_err(
                "shard_keys must match block_ids length",
            ));
        }
        let metadata_jsons = metadata_jsons.unwrap_or_else(|| vec![None; block_ids.len()]);
        if metadata_jsons.len() != block_ids.len() {
            return Err(PyValueError::new_err(
                "metadata_jsons must match block_ids length",
            ));
        }

        let mut inputs = Vec::with_capacity(block_ids.len());
        for index in 0..block_ids.len() {
            let metadata = metadata_jsons[index]
                .as_deref()
                .map(|raw| {
                    serde_json::from_str(raw).map_err(|err| PyValueError::new_err(err.to_string()))
                })
                .transpose()?
                .unwrap_or(serde_json::Value::Null);
            inputs.push(OptimizerBlockInput {
                block_id: block_ids[index].clone(),
                payload: payloads[index].clone(),
                step: steps[index],
                shard_key: shard_keys[index].clone(),
                metadata,
            });
        }
        let results = self
            .inner
            .store_blocks(origin_id, &inputs)
            .map_err(map_sodl_error)?;
        serde_json::to_string(&results).map_err(|err| PyValueError::new_err(err.to_string()))
    }

    fn load_block<'py>(
        &self,
        py: Python<'py>,
        origin_id: &str,
        block_id: &str,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let payload = self
            .inner
            .load_block(origin_id, block_id)
            .map_err(map_sodl_error)?;
        Ok(PyBytes::new(py, &payload))
    }

    fn load_blocks(
        &self,
        origin_id: &str,
        block_ids: Vec<String>,
    ) -> PyResult<Vec<(String, Vec<u8>)>> {
        let payloads = self
            .inner
            .load_blocks(origin_id, &block_ids)
            .map_err(map_sodl_error)?;
        Ok(payloads.into_iter().collect())
    }

    fn prefetch_blocks(&self, origin_id: &str, block_ids: Vec<String>) -> PyResult<usize> {
        self.inner
            .prefetch_blocks(origin_id, &block_ids)
            .map_err(map_sodl_error)
    }

    fn pin_blocks(&self, origin_id: &str, block_ids: Vec<String>) -> PyResult<()> {
        self.inner
            .pin_blocks(origin_id, &block_ids)
            .map_err(map_sodl_error)
    }

    fn unpin_blocks(&self, origin_id: &str, block_ids: Vec<String>) -> PyResult<()> {
        self.inner
            .unpin_blocks(origin_id, &block_ids)
            .map_err(map_sodl_error)
    }

    fn evict_blocks(&self, origin_id: &str, block_ids: Vec<String>) -> PyResult<usize> {
        self.inner
            .evict_blocks(origin_id, &block_ids)
            .map_err(map_sodl_error)
    }

    fn flush_origin(&self, origin_id: &str) -> PyResult<String> {
        let manifest = self.inner.flush_origin(origin_id).map_err(map_sodl_error)?;
        serde_json::to_string(&manifest).map_err(|err| PyValueError::new_err(err.to_string()))
    }

    fn flush_blocks(&self, origin_id: &str, block_ids: Vec<String>) -> PyResult<String> {
        let manifest = self
            .inner
            .flush_blocks(origin_id, &block_ids)
            .map_err(map_sodl_error)?;
        serde_json::to_string(&manifest).map_err(|err| PyValueError::new_err(err.to_string()))
    }

    fn manifest_json(&self, origin_id: &str) -> PyResult<String> {
        let manifest = self.inner.manifest(origin_id).map_err(map_sodl_error)?;
        serde_json::to_string(&manifest).map_err(|err| PyValueError::new_err(err.to_string()))
    }

    fn latest_blob_id(&self, origin_id: &str, block_id: &str) -> PyResult<Option<String>> {
        self.inner
            .latest_blob_id(origin_id, block_id)
            .map(|opt| opt.map(|blob_id| blob_id.0))
            .map_err(map_sodl_error)
    }

    #[pyo3(signature = (origin_id=None))]
    fn dirty_block_count(&self, origin_id: Option<&str>) -> PyResult<usize> {
        self.inner
            .dirty_block_count(origin_id)
            .map_err(map_sodl_error)
    }

    fn cache_stats_json(&self) -> PyResult<String> {
        let stats = self.inner.cache_stats().map_err(map_sodl_error)?;
        serde_json::to_string(&stats).map_err(|err| PyValueError::new_err(err.to_string()))
    }

    fn set_cache_capacity(&mut self, value: usize) {
        self.inner.set_cache_capacity(value);
    }
}

#[pymethods]
impl PyCheckpointStore {
    #[new]
    #[pyo3(signature = (blob_root, registry_dir=None, compression_level=3, max_checkpoints=0))]
    fn new(
        blob_root: &str,
        registry_dir: Option<&str>,
        compression_level: i32,
        max_checkpoints: usize,
    ) -> PyResult<Self> {
        let registry = registry_dir
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(blob_root).join("checkpoints"));
        let inner = CheckpointStore::open(
            std::path::PathBuf::from(blob_root),
            registry,
            compression_level,
            max_checkpoints,
        )
        .map_err(map_sodl_error)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (origin_id, payload, record_json=None))]
    fn save_checkpoint(
        &self,
        origin_id: &str,
        payload: &[u8],
        record_json: Option<&str>,
    ) -> PyResult<String> {
        let request: CheckpointSaveRequest = record_json
            .map(|raw| {
                serde_json::from_str(raw).map_err(|err| PyValueError::new_err(err.to_string()))
            })
            .transpose()?
            .unwrap_or_default();
        let record = self
            .inner
            .save_checkpoint_bytes(origin_id, payload, request)
            .map_err(map_sodl_error)?;
        serde_json::to_string(&record).map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[pyo3(signature = (origin_id, checkpoint_id=None))]
    fn load_checkpoint<'py>(
        &self,
        py: Python<'py>,
        origin_id: &str,
        checkpoint_id: Option<&str>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let payload = self
            .inner
            .load_checkpoint_bytes(origin_id, checkpoint_id)
            .map_err(map_sodl_error)?;
        Ok(PyBytes::new(py, &payload))
    }

    fn list_checkpoints(&self, origin_id: &str) -> PyResult<String> {
        let records = self
            .inner
            .list_checkpoints(origin_id)
            .map_err(map_sodl_error)?;
        serde_json::to_string(&records).map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[pyo3(signature = (origin_id, checkpoint_id=None))]
    fn get_checkpoint(&self, origin_id: &str, checkpoint_id: Option<&str>) -> PyResult<String> {
        let record = self
            .inner
            .get_checkpoint(origin_id, checkpoint_id)
            .map_err(map_sodl_error)?;
        serde_json::to_string(&record).map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[pyo3(signature = (origin_id, checkpoint_id=None))]
    fn get_lineage(&self, origin_id: &str, checkpoint_id: Option<&str>) -> PyResult<String> {
        let lineage = self
            .inner
            .get_lineage(origin_id, checkpoint_id)
            .map_err(map_sodl_error)?;
        serde_json::to_string(&lineage).map_err(|err| PyValueError::new_err(err.to_string()))
    }

    fn delete_checkpoint(&self, origin_id: &str, checkpoint_id: &str) -> PyResult<bool> {
        self.inner
            .delete_checkpoint(origin_id, checkpoint_id)
            .map_err(map_sodl_error)
    }

    fn diff_checkpoints(&self, origin_id: &str, old_id: &str, new_id: &str) -> PyResult<String> {
        let diff = self
            .inner
            .diff_checkpoints(origin_id, old_id, new_id)
            .map_err(map_sodl_error)?;
        serde_json::to_string(&diff).map_err(|err| PyValueError::new_err(err.to_string()))
    }
}

#[pymethods]
impl PyWeightManifestStore {
    #[new]
    #[pyo3(signature = (manifest_path, blob_root=None))]
    fn new(manifest_path: &str, blob_root: Option<&str>) -> PyResult<Self> {
        let inner = WeightManifestStore::open(
            std::path::PathBuf::from(manifest_path),
            blob_root.map(std::path::PathBuf::from),
        )
        .map_err(map_sodl_error)?;
        Ok(Self { inner })
    }

    fn load_manifest(&self) -> PyResult<Option<String>> {
        self.inner
            .load_manifest()
            .map_err(map_sodl_error)?
            .map(|manifest| {
                serde_json::to_string(&manifest)
                    .map_err(|err| PyValueError::new_err(err.to_string()))
            })
            .transpose()
    }

    #[pyo3(signature = (checkpoint_origin, resume_record_json=None))]
    fn resolve_origin_id(
        &self,
        checkpoint_origin: &str,
        resume_record_json: Option<&str>,
    ) -> PyResult<String> {
        let resume_record = resume_record_json
            .map(|raw| {
                serde_json::from_str(raw).map_err(|err| PyValueError::new_err(err.to_string()))
            })
            .transpose()?;
        let (origin_id, source) = self
            .inner
            .resolve_origin_id(checkpoint_origin, resume_record.as_ref())
            .map_err(map_sodl_error)?;
        serde_json::to_string(&serde_json::json!({
            "origin_id": origin_id,
            "source": source,
        }))
        .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[pyo3(signature = (origin_id, vocab_size, embedding_dim, clusters_json, metadata_json=None))]
    fn write_manifest(
        &self,
        origin_id: &str,
        vocab_size: usize,
        embedding_dim: usize,
        clusters_json: &str,
        metadata_json: Option<&str>,
    ) -> PyResult<String> {
        let raw_clusters: Vec<serde_json::Value> = serde_json::from_str(clusters_json)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let mut clusters = Vec::with_capacity(raw_clusters.len());
        for payload in raw_clusters {
            let cluster_id = payload
                .get("cluster_id")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| PyValueError::new_err("cluster_id is required"))?;
            let blob_id = payload
                .get("blob_id")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| PyValueError::new_err("blob_id is required"))?;
            let member_token_ids = payload
                .get("member_token_ids")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| PyValueError::new_err("member_token_ids is required"))?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|item| u32::try_from(item).ok())
                        .ok_or_else(|| PyValueError::new_err("member_token_ids must be integers"))
                })
                .collect::<PyResult<Vec<_>>>()?;
            clusters.push(WeightManifestCluster {
                cluster_id,
                blob_id: sodl_core::BlobId(blob_id.to_string()),
                member_token_ids,
            });
        }
        let metadata = metadata_json
            .map(|raw| {
                serde_json::from_str(raw).map_err(|err| PyValueError::new_err(err.to_string()))
            })
            .transpose()?;
        let manifest = self
            .inner
            .write_manifest(origin_id, vocab_size, embedding_dim, clusters, metadata)
            .map_err(map_sodl_error)?;
        serde_json::to_string(&manifest).map_err(|err| PyValueError::new_err(err.to_string()))
    }
}

#[pymethods]
impl PyWeightOriginRegistry {
    #[new]
    fn new() -> Self {
        Self {
            inner: WeightOriginRegistry::new(),
        }
    }

    fn create_model(&self, model_name: &str, quantization: &str) -> PyResult<String> {
        let model = self
            .inner
            .create_model(model_name, quantization)
            .map_err(map_sodl_error)?;
        let origin = self
            .inner
            .get_model(model.origin_id)
            .map_err(map_sodl_error)?;
        weight_origin_json(&origin)
    }

    #[pyo3(signature = (origin_id, model_name, quantization, num_clusters=0))]
    fn register_model(
        &self,
        origin_id: &str,
        model_name: &str,
        quantization: &str,
        num_clusters: usize,
    ) -> PyResult<String> {
        let model = self
            .inner
            .register_model(
                parse_origin_id(origin_id)?,
                model_name,
                quantization,
                num_clusters,
            )
            .map_err(map_sodl_error)?;
        let origin = self
            .inner
            .get_model(model.origin_id)
            .map_err(map_sodl_error)?;
        weight_origin_json(&origin)
    }

    #[pyo3(signature = (model_name, quantization, origin_id=None, num_clusters=0))]
    fn ensure_model(
        &self,
        model_name: &str,
        quantization: &str,
        origin_id: Option<&str>,
        num_clusters: usize,
    ) -> PyResult<String> {
        let model = self
            .inner
            .ensure_model(
                model_name,
                quantization,
                origin_id.map(parse_origin_id).transpose()?,
                num_clusters,
            )
            .map_err(map_sodl_error)?;
        let origin = self
            .inner
            .get_model(model.origin_id)
            .map_err(map_sodl_error)?;
        weight_origin_json(&origin)
    }

    fn get_model(&self, origin_id: &str) -> PyResult<String> {
        let origin = self
            .inner
            .get_model(parse_origin_id(origin_id)?)
            .map_err(map_sodl_error)?;
        weight_origin_json(&origin)
    }

    fn get_model_by_name(&self, model_name: &str) -> PyResult<String> {
        let origin = self
            .inner
            .get_model_by_name(model_name)
            .map_err(map_sodl_error)?;
        weight_origin_json(&origin)
    }

    #[pyo3(signature = (origin_id, amount=1))]
    fn increment_clusters(&self, origin_id: &str, amount: usize) -> PyResult<()> {
        self.inner
            .increment_clusters(parse_origin_id(origin_id)?, amount)
            .map_err(map_sodl_error)
    }
}

#[pymethods]
impl PyAeadCrypto {
    #[new]
    #[pyo3(signature = (master_key_hex=None))]
    fn new(master_key_hex: Option<&str>) -> PyResult<Self> {
        let inner = match master_key_hex {
            Some(hex) => AeadCrypto::from_hex(hex).map_err(map_sodl_error)?,
            None => AeadCrypto::generate(),
        };
        Ok(Self { inner })
    }

    fn master_key_hex(&self) -> String {
        self.inner.master_key_hex()
    }

    fn encrypt<'py>(
        &self,
        py: Python<'py>,
        origin_id: &str,
        plaintext: &[u8],
    ) -> PyResult<Bound<'py, PyBytes>> {
        let ciphertext = self
            .inner
            .encrypt_for_origin(
                parse_origin_id(origin_id)?,
                Bytes::copy_from_slice(plaintext),
            )
            .map_err(map_sodl_error)?;
        Ok(PyBytes::new(py, &ciphertext))
    }

    fn decrypt<'py>(
        &self,
        py: Python<'py>,
        origin_id: &str,
        ciphertext: &[u8],
    ) -> PyResult<Bound<'py, PyBytes>> {
        let plaintext = self
            .inner
            .decrypt_for_origin(
                parse_origin_id(origin_id)?,
                Bytes::copy_from_slice(ciphertext),
            )
            .map_err(map_sodl_error)?;
        Ok(PyBytes::new(py, &plaintext))
    }
}

#[pymethods]
impl PyFsBlobStore {
    #[new]
    #[pyo3(signature = (root, source_roots=None, peer_urls=None, edge_urls=None))]
    fn new(
        root: &str,
        source_roots: Option<Vec<String>>,
        peer_urls: Option<Vec<String>>,
        edge_urls: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let root_path = std::path::PathBuf::from(root);
        let store = FsBlobStore::open(&root_path).map_err(map_sodl_error)?;
        let sources = source_roots
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(index, source_root)| {
                let source_store = FsBlobStore::open(&source_root).map_err(map_sodl_error)?;
                Ok((
                    PeerAddr(format!("source:{index}:{source_root}")),
                    source_store,
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: store,
            root: root_path,
            sources,
            peer_urls: peer_urls
                .unwrap_or_default()
                .into_iter()
                .map(PeerAddr)
                .collect(),
            edge_urls: edge_urls.unwrap_or_default(),
            replicas: MemReplicaStore::new(),
        })
    }

    fn has(&self, blob_id: &str) -> PyResult<bool> {
        let id = sodl_core::BlobId(blob_id.to_string());
        if self.inner.has(&id).map_err(map_sodl_error)? {
            record_replica(&self.replicas, blob_id, "cache", ReplicaState::Healthy)?;
            return Ok(true);
        }
        if !self.peer_urls.is_empty() {
            let discovery = StaticPeerDiscovery::new(self.peer_urls.clone());
            let client = HttpPeerClient::new(10).map_err(map_sodl_error)?;
            let peer_source = DiscoveredPeerSource::new(&discovery, &client);
            if peer_source.fetch(&id).map_err(map_sodl_error)?.is_some() {
                if let Some(peer_addr) = peer_source.last_provider() {
                    record_replica(&self.replicas, blob_id, &peer_addr.0, ReplicaState::Healthy)?;
                }
                return Ok(true);
            }
        }
        if !self.edge_urls.is_empty() {
            let edge_provider =
                HttpEdgeProvider::new(self.edge_urls.clone(), 10).map_err(map_sodl_error)?;
            if edge_provider
                .get_blob(&id)
                .map_err(map_sodl_error)?
                .is_some()
            {
                if let Some(edge_url) = edge_provider.last_provider() {
                    record_replica(&self.replicas, blob_id, &edge_url, ReplicaState::Healthy)?;
                }
                return Ok(true);
            }
        }
        for (peer_addr, source_store) in &self.sources {
            if source_store.has(&id).map_err(map_sodl_error)? {
                record_replica(&self.replicas, blob_id, &peer_addr.0, ReplicaState::Healthy)?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn put(&self, blob_id: &str, data: &[u8]) -> PyResult<()> {
        let id = sodl_core::BlobId(blob_id.to_string());
        self.inner
            .put(&id, Bytes::copy_from_slice(data))
            .map_err(map_sodl_error)?;
        record_replica(&self.replicas, blob_id, "cache", ReplicaState::Healthy)
    }

    fn get<'py>(&self, py: Python<'py>, blob_id: &str) -> PyResult<Bound<'py, PyBytes>> {
        let id = sodl_core::BlobId(blob_id.to_string());
        if self.inner.has(&id).map_err(map_sodl_error)? {
            let data = self.inner.get(&id).map_err(map_sodl_error)?;
            record_replica(&self.replicas, blob_id, "cache", ReplicaState::Healthy)?;
            return Ok(PyBytes::new(py, &data));
        }

        let discovery = StaticPeerDiscovery::new(self.peer_urls.clone());
        let peer_client = HttpPeerClient::new(10).map_err(map_sodl_error)?;
        let peer_source = DiscoveredPeerSource::new(&discovery, &peer_client);
        let edge_provider =
            HttpEdgeProvider::new(self.edge_urls.clone(), 10).map_err(map_sodl_error)?;
        let edge_source = EdgeFetchSource::new(&edge_provider);
        let mut source_index: Option<usize> = None;
        let mut adapters: Vec<StoreSource<'_>> = Vec::with_capacity(self.sources.len());
        for (index, (peer_addr, source_store)) in self.sources.iter().enumerate() {
            if source_index.is_none() && source_store.has(&id).map_err(map_sodl_error)? {
                source_index = Some(index);
                record_replica(&self.replicas, blob_id, &peer_addr.0, ReplicaState::Healthy)?;
            }
            adapters.push(StoreSource(source_store));
        }
        let mut source_refs: Vec<&dyn FetchSource> = Vec::new();
        if !self.peer_urls.is_empty() {
            source_refs.push(&peer_source);
        }
        if !self.edge_urls.is_empty() {
            source_refs.push(&edge_source);
        }
        source_refs.extend(adapters.iter().map(|adapter| adapter as &dyn FetchSource));
        let pipeline = FetchPipeline {
            cache: &self.inner,
            sources: source_refs,
            authorizer: None,
        };
        let data = pipeline.get_for(None, None, &id).map_err(map_sodl_error)?;
        record_replica(&self.replicas, blob_id, "cache", ReplicaState::Healthy)?;
        if let Some(peer_addr) = peer_source.last_provider() {
            record_replica(&self.replicas, blob_id, &peer_addr.0, ReplicaState::Healthy)?;
        } else if let Some(edge_url) = edge_provider.last_provider() {
            record_replica(&self.replicas, blob_id, &edge_url, ReplicaState::Healthy)?;
        } else if let Some(index) = source_index {
            let peer_addr = &self.sources[index].0;
            record_replica(&self.replicas, blob_id, &peer_addr.0, ReplicaState::Healthy)?;
        }
        Ok(PyBytes::new(py, &data))
    }

    fn delete(&self, blob_id: &str) -> PyResult<()> {
        let id = sodl_core::BlobId(blob_id.to_string());
        self.inner.delete(&id).map_err(map_sodl_error)
    }

    fn blob_count(&self) -> PyResult<usize> {
        // The underlying store already shards by algorithm/prefix; a small filesystem walk is sufficient here.
        fn count_blob_files(path: &std::path::Path) -> usize {
            std::fs::read_dir(path)
                .ok()
                .into_iter()
                .flat_map(|iter| iter.filter_map(Result::ok))
                .map(|entry| entry.path())
                .map(|path| {
                    if path.is_dir() {
                        count_blob_files(&path)
                    } else {
                        1usize
                    }
                })
                .sum()
        }
        Ok(count_blob_files(&self.root))
    }

    fn replica_nodes(&self, blob_id: &str) -> PyResult<Vec<String>> {
        let id = sodl_core::BlobId(blob_id.to_string());
        let replicas = self.replicas.list_replicas(&id).map_err(map_sodl_error)?;
        Ok(replicas
            .into_iter()
            .filter(|replica| replica.state == ReplicaState::Healthy)
            .map(|replica| replica.node_id)
            .collect())
    }
}

#[pyfunction]
fn compute_blob_id_py(data: &[u8]) -> String {
    compute_blob_id(data, HashAlg::Blake3).0
}

#[pyfunction]
fn verify_integrity_py(blob_id: &str, data: &[u8]) -> PyResult<()> {
    verify_integrity(&sodl_core::BlobId(blob_id.to_string()), data).map_err(map_sodl_error)
}

#[pyfunction]
#[pyo3(signature = (data, level=None))]
fn compress_zstd<'py>(
    py: Python<'py>,
    data: &[u8],
    level: Option<i32>,
) -> PyResult<Bound<'py, PyBytes>> {
    let compressed = py
        .allow_threads(|| zstd::stream::encode_all(data, level.unwrap_or(3)))
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(PyBytes::new(py, &compressed))
}

#[pyfunction]
fn decompress_zstd<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let decompressed = py
        .allow_threads(|| zstd::stream::decode_all(data))
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(PyBytes::new(py, &decompressed))
}

#[pymodule]
fn sodl_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFsBlobStore>()?;
    m.add_class::<PyOptimizerStateStore>()?;
    m.add_class::<PyCheckpointStore>()?;
    m.add_class::<PyWeightManifestStore>()?;
    m.add_class::<PyWeightOriginRegistry>()?;
    m.add_class::<PyAeadCrypto>()?;
    m.add_function(wrap_pyfunction!(compute_blob_id_py, m)?)?;
    m.add_function(wrap_pyfunction!(verify_integrity_py, m)?)?;
    m.add_function(wrap_pyfunction!(compress_zstd, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_zstd, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_blob_id_matches_sodl_cas() {
        let blob_id = compute_blob_id_py(b"hello");
        assert_eq!(
            blob_id,
            sodl_cas::compute_blob_id(b"hello", HashAlg::Blake3).0
        );
    }

    #[test]
    fn zstd_roundtrip_works() {
        Python::with_gil(|py| {
            let compressed = compress_zstd(py, b"hello ffi", Some(3)).unwrap();
            let decompressed = decompress_zstd(py, compressed.as_bytes()).unwrap();
            assert_eq!(decompressed.as_bytes(), b"hello ffi");
        });
    }

    #[test]
    fn fs_blob_store_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let store = PyFsBlobStore::new(tmp.path().to_str().unwrap(), None, None, None).unwrap();
        let blob_id = compute_blob_id_py(b"ffi-store");
        store.put(&blob_id, b"ffi-store").unwrap();
        assert!(store.has(&blob_id).unwrap());
    }

    #[test]
    fn fs_blob_store_fetches_from_source_store() {
        let cache = tempfile::tempdir().unwrap();
        let source = tempfile::tempdir().unwrap();

        let source_store =
            PyFsBlobStore::new(source.path().to_str().unwrap(), None, None, None).unwrap();
        let blob_id = compute_blob_id_py(b"fetch-me");
        source_store.put(&blob_id, b"fetch-me").unwrap();

        let cache_store = PyFsBlobStore::new(
            cache.path().to_str().unwrap(),
            Some(vec![source.path().to_str().unwrap().to_string()]),
            None,
            None,
        )
        .unwrap();

        assert!(cache_store.has(&blob_id).unwrap());
        Python::with_gil(|py| {
            let fetched = cache_store.get(py, &blob_id).unwrap();
            assert_eq!(fetched.as_bytes(), b"fetch-me");
        });
        assert!(cache_store
            .inner
            .has(&sodl_core::BlobId(blob_id.clone()))
            .unwrap());
        let replicas = cache_store.replica_nodes(&blob_id).unwrap();
        assert!(replicas.iter().any(|node| node == "cache"));
        assert!(replicas.iter().any(|node| node.starts_with("source:0:")));
    }

    #[test]
    fn optimizer_state_store_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let store = PyOptimizerStateStore::new(
            tmp.path().join("blobs").to_str().unwrap(),
            Some(tmp.path().join("registry").to_str().unwrap()),
            3,
            8,
            1,
        )
        .unwrap();

        let stored = store
            .store_block(
                "run-1",
                "block-a",
                b"state-a",
                4,
                None,
                Some("{\"kind\":\"adamw\"}"),
            )
            .unwrap();
        assert!(stored.contains("block-a"));

        Python::with_gil(|py| {
            let payload = store.load_block(py, "run-1", "block-a").unwrap();
            assert_eq!(payload.as_bytes(), b"state-a");
        });

        let manifest_json = store.manifest_json("run-1").unwrap();
        assert!(manifest_json.contains("block-a"));
    }

    #[test]
    fn checkpoint_store_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let store = PyCheckpointStore::new(
            tmp.path().join("blobs").to_str().unwrap(),
            Some(tmp.path().join("registry").to_str().unwrap()),
            3,
            2,
        )
        .unwrap();

        let first = store
            .save_checkpoint("run-1", b"payload-a", Some("{\"step\":10,\"loss\":0.5}"))
            .unwrap();
        assert!(first.contains("\"step\":10"));

        let first_record: serde_json::Value = serde_json::from_str(&first).unwrap();
        let first_id = first_record["checkpoint_id"].as_str().unwrap().to_string();
        store
            .save_checkpoint(
                "run-1",
                b"payload-b",
                Some(&format!(
                    "{{\"step\":20,\"parent_checkpoint_id\":\"{}\",\"stage\":\"phase-2\"}}",
                    first_id
                )),
            )
            .unwrap();

        let listed = store.list_checkpoints("run-1").unwrap();
        assert!(!listed.contains("payload"));
        assert!(listed.contains("\"step\":10"));
        assert!(listed.contains("\"step\":20"));

        Python::with_gil(|py| {
            let payload = store.load_checkpoint(py, "run-1", Some(&first_id)).unwrap();
            assert_eq!(payload.as_bytes(), b"payload-a");
        });

        let lineage = store.get_lineage("run-1", None).unwrap();
        assert!(lineage.contains("phase-2"));
    }

    #[test]
    fn weight_manifest_store_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let blob_root = tmp.path().join("blobs");
        let blob_store = FsBlobStore::open(&blob_root).unwrap();
        let stale_blob = "blake3:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
        let live_blob = "blake3:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
        blob_store
            .put(
                &sodl_core::BlobId(stale_blob.to_string()),
                Bytes::from_static(b"stale"),
            )
            .unwrap();
        blob_store
            .put(
                &sodl_core::BlobId(live_blob.to_string()),
                Bytes::from_static(b"live"),
            )
            .unwrap();

        let store = PyWeightManifestStore::new(
            tmp.path().join("sodl_manifest.json").to_str().unwrap(),
            Some(blob_root.to_str().unwrap()),
        )
        .unwrap();
        store
            .write_manifest(
                "origin:stable",
                2,
                4,
                &format!(
                    "[{{\"cluster_id\":0,\"blob_id\":\"{}\",\"member_token_ids\":[0]}}]",
                    stale_blob
                ),
                Some("{\"checkpoint_origin\":\"carla-large\"}"),
            )
            .unwrap();
        let updated = store
            .write_manifest(
                "origin:stable",
                2,
                4,
                &format!(
                    "[{{\"cluster_id\":0,\"blob_id\":\"{}\",\"member_token_ids\":[0]}}]",
                    live_blob
                ),
                None,
            )
            .unwrap();
        assert!(updated.contains(live_blob));
        assert!(!blob_store
            .has(&sodl_core::BlobId(stale_blob.to_string()))
            .unwrap());
    }

    #[test]
    fn optimizer_state_store_batch_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let store = PyOptimizerStateStore::new(
            tmp.path().join("blobs").to_str().unwrap(),
            Some(tmp.path().join("registry").to_str().unwrap()),
            3,
            8,
            8,
        )
        .unwrap();
        let stored = store
            .store_blocks(
                "run-batch",
                vec![String::from("block-a"), String::from("block-b")],
                vec![b"state-a".to_vec(), b"state-b".to_vec()],
                Some(vec![1, 1]),
                Some(vec![
                    Some(String::from("group:0")),
                    Some(String::from("group:0")),
                ]),
                Some(vec![
                    Some(String::from("{}")),
                    Some(String::from("{\"kind\":\"adamw\"}")),
                ]),
            )
            .unwrap();
        assert!(stored.contains("block-a"));
        store.flush_origin("run-batch").unwrap();
        let payloads = store
            .load_blocks(
                "run-batch",
                vec![String::from("block-a"), String::from("block-b")],
            )
            .unwrap();
        assert_eq!(payloads.len(), 2);
    }

    #[test]
    fn weight_origin_registry_roundtrip() {
        let registry = PyWeightOriginRegistry::new();
        let created = registry.create_model("carla-large", "F32").unwrap();
        assert!(created.contains("carla-large"));

        let model: serde_json::Value = serde_json::from_str(&created).unwrap();
        let origin_id = model["origin_id"].as_str().unwrap();
        registry.increment_clusters(origin_id, 3).unwrap();

        let fetched = registry.get_model(origin_id).unwrap();
        assert!(fetched.contains("\"num_clusters\":3"));

        let by_name = registry.get_model_by_name("carla-large").unwrap();
        assert!(by_name.contains("carla-large"));
    }

    #[test]
    fn optimizer_state_store_flush_subset_and_evict() {
        let tmp = tempfile::tempdir().unwrap();
        let store = PyOptimizerStateStore::new(
            tmp.path().join("blobs").to_str().unwrap(),
            Some(tmp.path().join("registry").to_str().unwrap()),
            3,
            8,
            10,
        )
        .unwrap();

        store
            .store_block("run-2", "block-a", b"state-a", 1, None, None)
            .unwrap();
        store
            .store_block("run-2", "block-b", b"state-b", 1, None, None)
            .unwrap();

        let manifest = store
            .flush_blocks("run-2", vec![String::from("block-a")])
            .unwrap();
        assert!(manifest.contains("block-a"));
        assert!(!manifest.contains("block-b"));

        store
            .pin_blocks("run-2", vec![String::from("block-a")])
            .unwrap();
        let evicted = store
            .evict_blocks(
                "run-2",
                vec![String::from("block-a"), String::from("block-b")],
            )
            .unwrap();
        assert_eq!(evicted, 0);

        store
            .flush_blocks("run-2", vec![String::from("block-b")])
            .unwrap();
        store
            .unpin_blocks("run-2", vec![String::from("block-a")])
            .unwrap();
        let evicted = store
            .evict_blocks(
                "run-2",
                vec![String::from("block-a"), String::from("block-b")],
            )
            .unwrap();
        assert_eq!(evicted, 2);
    }

    #[test]
    fn aead_crypto_roundtrip() {
        let crypto = PyAeadCrypto::new(None).unwrap();
        let origin_id = "550e8400-e29b-41d4-a716-446655440000";
        let plaintext = b"top secret payload";

        Python::with_gil(|py| {
            let ciphertext = crypto.encrypt(py, origin_id, plaintext).unwrap();
            assert_ne!(ciphertext.as_bytes(), plaintext);
            let decrypted = crypto
                .decrypt(py, origin_id, ciphertext.as_bytes())
                .unwrap();
            assert_eq!(decrypted.as_bytes(), plaintext);
        });
    }

    #[test]
    fn aead_crypto_roundtrip_accepts_python_origin_prefix() {
        let crypto = PyAeadCrypto::new(None).unwrap();
        let origin_id = "origin:550e8400-e29b-41d4-a716-446655440000";
        let plaintext = b"prefixed origin payload";

        Python::with_gil(|py| {
            let ciphertext = crypto.encrypt(py, origin_id, plaintext).unwrap();
            let decrypted = crypto
                .decrypt(py, origin_id, ciphertext.as_bytes())
                .unwrap();
            assert_eq!(decrypted.as_bytes(), plaintext);
        });
    }
}
