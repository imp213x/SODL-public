use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PipelineHash(pub String);

impl PipelineHash {
    pub fn compute(origin_id: &str, pipeline: &str, config: &str) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(origin_id.as_bytes());
        hasher.update(pipeline.as_bytes());
        hasher.update(config.as_bytes());
        Self(hasher.finalize().to_hex().to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DatasetChunk {
    pub ordinal: u32,
    pub blob_id: String,
    pub byte_size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DatasetChunkManifest {
    pub origin_id: String,
    pub manifest_id: String,
    pub pipeline_hash: PipelineHash,
    pub total_bytes: u64,
    pub chunks: Vec<DatasetChunk>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_hash_is_stable_for_same_inputs() {
        let a = PipelineHash::compute("origin:a", "semantic_chunk", "{\"k\":1}");
        let b = PipelineHash::compute("origin:a", "semantic_chunk", "{\"k\":1}");
        assert_eq!(a, b);
    }

    #[test]
    fn pipeline_hash_changes_when_config_changes() {
        let a = PipelineHash::compute("origin:a", "semantic_chunk", "{\"k\":1}");
        let b = PipelineHash::compute("origin:a", "semantic_chunk", "{\"k\":2}");
        assert_ne!(a, b);
    }
}
