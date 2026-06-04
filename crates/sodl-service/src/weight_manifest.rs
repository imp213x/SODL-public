use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sodl_cas::{BlobStore, FsBlobStore};
use sodl_core::{BlobId, Result, SodlError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightManifestCluster {
    pub cluster_id: u64,
    pub blob_id: BlobId,
    pub member_token_ids: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightManifest {
    pub origin_id: String,
    #[serde(default)]
    pub vocab_size: usize,
    #[serde(default)]
    pub embedding_dim: usize,
    #[serde(default)]
    pub clusters: Vec<WeightManifestCluster>,
    #[serde(default = "default_json_object")]
    pub metadata: Value,
}

pub struct WeightManifestStore {
    manifest_path: PathBuf,
    blob_store: Option<FsBlobStore>,
}

impl WeightManifestStore {
    pub fn open(
        manifest_path: impl AsRef<Path>,
        blob_root: Option<impl AsRef<Path>>,
    ) -> Result<Self> {
        let manifest_path = manifest_path.as_ref().to_path_buf();
        if let Some(parent) = manifest_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|err| SodlError::Io(format!("create weight manifest dir: {err}")))?;
        }
        let blob_store = blob_root
            .map(|root| FsBlobStore::open(root.as_ref()))
            .transpose()?;
        Ok(Self {
            manifest_path,
            blob_store,
        })
    }

    pub fn load_manifest(&self) -> Result<Option<WeightManifest>> {
        if !self.manifest_path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&self.manifest_path).map_err(|err| {
            SodlError::Io(format!(
                "read weight manifest {}: {err}",
                self.manifest_path.display()
            ))
        })?;
        let manifest: WeightManifest = serde_json::from_slice(&bytes).map_err(|err| {
            SodlError::Serialization(format!(
                "parse weight manifest {}: {err}",
                self.manifest_path.display()
            ))
        })?;
        Ok(Some(manifest))
    }

    pub fn resolve_origin_id(
        &self,
        checkpoint_origin: &str,
        resume_record: Option<&Value>,
    ) -> Result<(Option<String>, String)> {
        let metadata = resume_record
            .and_then(|payload| payload.get("metadata"))
            .and_then(Value::as_object);
        if let Some(origin_id) = metadata
            .and_then(|meta| meta.get("sodl_origin_id"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return Ok((Some(origin_id.to_string()), "resume_record".to_string()));
        }

        let Some(manifest) = self.load_manifest()? else {
            return Ok((None, "new".to_string()));
        };
        let origin_id = manifest.origin_id.trim();
        if origin_id.is_empty() {
            return Ok((None, "new".to_string()));
        }
        let recorded_checkpoint_origin = manifest
            .metadata
            .get("checkpoint_origin")
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or("");
        if !recorded_checkpoint_origin.is_empty() && recorded_checkpoint_origin != checkpoint_origin
        {
            return Ok((None, "new".to_string()));
        }
        Ok((Some(origin_id.to_string()), "manifest".to_string()))
    }

    pub fn write_manifest(
        &self,
        origin_id: &str,
        vocab_size: usize,
        embedding_dim: usize,
        clusters: Vec<WeightManifestCluster>,
        metadata: Option<Value>,
    ) -> Result<WeightManifest> {
        let previous_manifest = self.load_manifest()?;
        let mut resolved_metadata = previous_manifest
            .as_ref()
            .map(|manifest| manifest.metadata.clone())
            .unwrap_or_else(default_json_object);
        if let Some(next_metadata) = metadata {
            merge_metadata(&mut resolved_metadata, next_metadata);
        }

        let mut clusters = clusters;
        clusters.sort_by_key(|item| item.cluster_id);
        let manifest = WeightManifest {
            origin_id: origin_id.to_string(),
            vocab_size,
            embedding_dim,
            clusters,
            metadata: resolved_metadata,
        };

        let tmp_path = self.manifest_path.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(&manifest)
            .map_err(|err| SodlError::Serialization(format!("serialize weight manifest: {err}")))?;
        fs::write(&tmp_path, &bytes).map_err(|err| {
            SodlError::Io(format!(
                "write weight manifest temp {}: {err}",
                tmp_path.display()
            ))
        })?;
        fs::rename(&tmp_path, &self.manifest_path).map_err(|err| {
            SodlError::Io(format!(
                "replace weight manifest {}: {err}",
                self.manifest_path.display()
            ))
        })?;

        if let (Some(previous), Some(blob_store)) = (previous_manifest, &self.blob_store) {
            let live_blob_ids: HashSet<BlobId> = manifest
                .clusters
                .iter()
                .map(|cluster| cluster.blob_id.clone())
                .collect();
            let stale_blob_ids: HashSet<BlobId> = previous
                .clusters
                .into_iter()
                .map(|cluster| cluster.blob_id)
                .filter(|blob_id| !live_blob_ids.contains(blob_id))
                .collect();
            for blob_id in stale_blob_ids {
                if blob_store.has(&blob_id)? {
                    blob_store.delete(&blob_id)?;
                }
            }
        }

        Ok(manifest)
    }
}

fn default_json_object() -> Value {
    Value::Object(serde_json::Map::new())
}

fn merge_metadata(target: &mut Value, incoming: Value) {
    let Value::Object(target_map) = target else {
        *target = default_json_object();
        merge_metadata(target, incoming);
        return;
    };
    if let Value::Object(incoming_map) = incoming {
        for (key, value) in incoming_map {
            target_map.insert(key, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_roundtrip_and_gc() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest_path = tmp.path().join("run").join("sodl_manifest.json");
        let blob_root = tmp.path().join("blobs");
        let store = WeightManifestStore::open(&manifest_path, Some(&blob_root)).unwrap();
        let blob_store = FsBlobStore::open(&blob_root).unwrap();

        let blob_a = BlobId(
            "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
        );
        let blob_b = BlobId(
            "blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
        );
        let blob_c = BlobId(
            "blake3:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_string(),
        );
        blob_store
            .put(&blob_a, bytes::Bytes::from_static(b"a"))
            .unwrap();
        blob_store
            .put(&blob_b, bytes::Bytes::from_static(b"b"))
            .unwrap();
        blob_store
            .put(&blob_c, bytes::Bytes::from_static(b"c"))
            .unwrap();

        store
            .write_manifest(
                "origin:1",
                2,
                4,
                vec![
                    WeightManifestCluster {
                        cluster_id: 0,
                        blob_id: blob_a.clone(),
                        member_token_ids: vec![0],
                    },
                    WeightManifestCluster {
                        cluster_id: 1,
                        blob_id: blob_b.clone(),
                        member_token_ids: vec![1],
                    },
                ],
                Some(serde_json::json!({"checkpoint_origin":"carla-large","note":"keep"})),
            )
            .unwrap();
        let updated = store
            .write_manifest(
                "origin:1",
                2,
                4,
                vec![
                    WeightManifestCluster {
                        cluster_id: 0,
                        blob_id: blob_a.clone(),
                        member_token_ids: vec![0],
                    },
                    WeightManifestCluster {
                        cluster_id: 1,
                        blob_id: blob_c.clone(),
                        member_token_ids: vec![1],
                    },
                ],
                None,
            )
            .unwrap();

        assert_eq!(updated.metadata["note"], "keep");
        assert!(blob_store.has(&blob_a).unwrap());
        assert!(!blob_store.has(&blob_b).unwrap());
        assert!(blob_store.has(&blob_c).unwrap());
    }

    #[test]
    fn resolve_origin_prefers_resume_record_then_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest_path = tmp.path().join("sodl_manifest.json");
        let store = WeightManifestStore::open(&manifest_path, None::<&Path>).unwrap();
        store
            .write_manifest(
                "origin:manifest",
                1,
                1,
                vec![],
                Some(serde_json::json!({"checkpoint_origin":"carla-large"})),
            )
            .unwrap();

        let resolved = store
            .resolve_origin_id(
                "carla-large",
                Some(&serde_json::json!({"metadata":{"sodl_origin_id":"origin:resume"}})),
            )
            .unwrap();
        assert_eq!(resolved.0.as_deref(), Some("origin:resume"));
        assert_eq!(resolved.1, "resume_record");

        let resolved = store.resolve_origin_id("carla-large", None).unwrap();
        assert_eq!(resolved.0.as_deref(), Some("origin:manifest"));
        assert_eq!(resolved.1, "manifest");

        let resolved = store.resolve_origin_id("carla-medium", None).unwrap();
        assert_eq!(resolved.0, None);
        assert_eq!(resolved.1, "new");
    }
}
