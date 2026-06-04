//! Chunk manifest — the root blob for chunked payloads.
//!
//! A `ChunkManifest` is a JSON document stored as its own blob in the CAS.
//! It describes how to reassemble the original payload from its parts.

use serde::{Deserialize, Serialize};
use sodl_core::BlobId;

/// Describes a single chunk within a manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkEntry {
    /// Content-addressed ID of this chunk's blob.
    pub blob_id: BlobId,
    /// Byte offset within the original payload.
    pub offset: u64,
    /// Length of this chunk in bytes.
    pub length: u32,
}

/// Root manifest for a chunked payload.
///
/// Stored as a JSON blob in the CAS.  The blob ID of the serialized manifest
/// is what appears in `Representation.root_blobs`.
///
/// # Format stability
///
/// The `version` field allows future evolution.  Readers must check `version`
/// and reject manifests with an unsupported version number.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkManifest {
    /// Manifest format version (currently `1`).
    pub version: u32,
    /// Total size of the reassembled payload in bytes.
    pub total_size: u64,
    /// Number of chunks (redundant with `chunks.len()`, but useful for
    /// quick validation without parsing the full array).
    pub chunk_count: u32,
    /// Hash algorithm used for chunk blob IDs (e.g., "blake3", "sha256").
    pub hash_alg: String,
    /// Ordered list of chunks. Concatenating their payloads in order
    /// reconstructs the original bytes.
    pub chunks: Vec<ChunkEntry>,
}

impl ChunkManifest {
    /// Validate internal consistency.
    pub fn validate(&self) -> sodl_core::Result<()> {
        if self.version != 1 {
            return Err(sodl_core::SodlError::Invalid(format!(
                "unsupported chunk manifest version: {}",
                self.version
            )));
        }
        if self.chunk_count != self.chunks.len() as u32 {
            return Err(sodl_core::SodlError::Invalid(
                "chunk_count doesn't match chunks length".into(),
            ));
        }
        let total: u64 = self.chunks.iter().map(|c| c.length as u64).sum();
        if total != self.total_size {
            return Err(sodl_core::SodlError::Invalid(format!(
                "total_size mismatch: declared {}, actual {}",
                self.total_size, total
            )));
        }
        // Verify offsets are contiguous.
        let mut expected_offset = 0u64;
        for c in &self.chunks {
            if c.offset != expected_offset {
                return Err(sodl_core::SodlError::Invalid(format!(
                    "chunk offset gap: expected {expected_offset}, got {}",
                    c.offset
                )));
            }
            expected_offset += c.length as u64;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_manifest() {
        let m = ChunkManifest {
            version: 1,
            total_size: 300,
            chunk_count: 2,
            hash_alg: "blake3".into(),
            chunks: vec![
                ChunkEntry {
                    blob_id: BlobId("blake3:aaa".into()),
                    offset: 0,
                    length: 200,
                },
                ChunkEntry {
                    blob_id: BlobId("blake3:bbb".into()),
                    offset: 200,
                    length: 100,
                },
            ],
        };
        m.validate().unwrap();
    }

    #[test]
    fn bad_version_rejected() {
        let m = ChunkManifest {
            version: 99,
            total_size: 0,
            chunk_count: 0,
            hash_alg: "blake3".into(),
            chunks: vec![],
        };
        assert!(m.validate().is_err());
    }

    #[test]
    fn size_mismatch_rejected() {
        let m = ChunkManifest {
            version: 1,
            total_size: 999,
            chunk_count: 1,
            hash_alg: "blake3".into(),
            chunks: vec![ChunkEntry {
                blob_id: BlobId("blake3:aaa".into()),
                offset: 0,
                length: 100,
            }],
        };
        assert!(m.validate().is_err());
    }
}
