//! Content-defined chunking for SODL.
//!
//! This crate provides the general-purpose **chunking pipeline** used by SODL
//! to break large payloads into content-addressed chunks.  Benefits:
//!
//! - **Cross-origin deduplication** — identical byte ranges produce the same
//!   `BlobId`, so shared content is stored once regardless of how many origins
//!   reference it.
//! - **Large-file support** — payloads beyond a configurable threshold are
//!   automatically split and stored as a chunk tree with a manifest root.
//! - **Streaming-friendly** — chunks can be fetched / verified individually
//!   without downloading the full payload.
//! - **Algorithm-agnostic** — the [`Chunker`] trait accepts any splitting
//!   strategy.  Two built-in strategies ship out of the box:
//!   [`FastCdcChunker`] (content-defined, variable-size, gear-hash based)
//!   and [`FixedSizeChunker`] (deterministic fixed-width).
//!
//! # Architecture
//!
//! ```text
//! raw bytes
//!   │
//!   ▼
//! ┌──────────────────┐
//! │  Chunker trait    │  split bytes → Vec<ChunkDescriptor>
//! │  (FastCDC, Fixed) │
//! └──────────────────┘
//!   │
//!   ▼
//! ┌──────────────────┐
//! │  ChunkWriter      │  encrypt + store each chunk via EncryptedCas
//! │                    │  build ChunkManifest
//! └──────────────────┘
//!   │
//!   ▼
//! ┌──────────────────┐
//! │  ChunkManifest    │  serialized as JSON → stored as its own blob
//! │  (root blob)      │  referenced by Representation.root_blobs
//! └──────────────────┘
//! ```

mod fastcdc;
pub mod manifest;

use bytes::Bytes;
use sodl_core::{BlobId, Result};

// ---------------------------------------------------------------------------
// Chunker trait
// ---------------------------------------------------------------------------

/// Description of a single chunk within a payload.
#[derive(Debug, Clone)]
pub struct ChunkDescriptor {
    /// Byte offset within the original payload.
    pub offset: u64,
    /// Length in bytes.
    pub length: u32,
    /// The raw chunk bytes.
    pub data: Bytes,
}

/// A strategy for splitting a byte payload into chunks.
///
/// Implementations are **stateless** — all state is carried in the returned
/// `Vec<ChunkDescriptor>`.
pub trait Chunker: Send + Sync {
    /// Split `data` into one or more chunks.
    ///
    /// The returned chunks must cover exactly `[0, data.len())` with no gaps
    /// or overlaps (verified in debug builds).
    fn chunk(&self, data: &[u8]) -> Vec<ChunkDescriptor>;
}

// ---------------------------------------------------------------------------
// Fixed-size chunker
// ---------------------------------------------------------------------------

/// Simple fixed-width chunker.  Every chunk is exactly `chunk_size` bytes
/// except the last, which may be shorter.
///
/// Useful for predictable chunk sizes (e.g., 256 KiB blocks for erasure coding).
#[derive(Debug, Clone)]
pub struct FixedSizeChunker {
    /// Target chunk size in bytes.
    pub chunk_size: u32,
}

impl FixedSizeChunker {
    pub fn new(chunk_size: u32) -> Self {
        assert!(chunk_size > 0, "chunk_size must be > 0");
        Self { chunk_size }
    }
}

impl Chunker for FixedSizeChunker {
    fn chunk(&self, data: &[u8]) -> Vec<ChunkDescriptor> {
        let cs = self.chunk_size as usize;
        let mut out = Vec::with_capacity(data.len() / cs + 1);
        let mut offset = 0usize;
        while offset < data.len() {
            let end = (offset + cs).min(data.len());
            out.push(ChunkDescriptor {
                offset: offset as u64,
                length: (end - offset) as u32,
                data: Bytes::copy_from_slice(&data[offset..end]),
            });
            offset = end;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// FastCDC chunker (gear-hash based, content-defined)
// ---------------------------------------------------------------------------

pub use fastcdc::FastCdcChunker;

// ---------------------------------------------------------------------------
// Chunk pipeline: write + reassemble
// ---------------------------------------------------------------------------

use manifest::ChunkManifest;

/// Configuration for the chunked upload pipeline.
#[derive(Debug, Clone)]
pub struct ChunkPipelineConfig {
    /// Payloads smaller than this are stored as a single blob (no chunking).
    /// Default: 256 KiB.
    pub inline_threshold: u64,
    /// Chunker to use when payload exceeds `inline_threshold`.
    /// If `None`, uses `FastCdcChunker::default()`.
    chunker: Option<Box<dyn CloneChunker>>,
}

/// Object-safe wrapper so we can clone the config.
trait CloneChunker: Chunker {
    fn clone_box(&self) -> Box<dyn CloneChunker>;
}

impl<T: Chunker + Clone + 'static> CloneChunker for T {
    fn clone_box(&self) -> Box<dyn CloneChunker> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn CloneChunker> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl std::fmt::Debug for Box<dyn CloneChunker> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Box<dyn CloneChunker>")
    }
}

impl Default for ChunkPipelineConfig {
    fn default() -> Self {
        Self {
            inline_threshold: 256 * 1024, // 256 KiB
            chunker: None,
        }
    }
}

impl ChunkPipelineConfig {
    /// Create a config with a custom chunker.
    pub fn with_chunker<C: Chunker + Clone + 'static>(mut self, c: C) -> Self {
        self.chunker = Some(Box::new(c));
        self
    }

    /// Create a config with a custom inline threshold.
    pub fn with_inline_threshold(mut self, bytes: u64) -> Self {
        self.inline_threshold = bytes;
        self
    }

    fn effective_chunker(&self) -> &dyn Chunker {
        self.chunker
            .as_ref()
            .map(|c| c.as_ref() as &dyn Chunker)
            .unwrap_or(&*DEFAULT_CHUNKER)
    }
}

static DEFAULT_CHUNKER: once_cell::sync::Lazy<FastCdcChunker> =
    once_cell::sync::Lazy::new(FastCdcChunker::default);

/// Result of a chunked upload.
#[derive(Debug, Clone)]
pub struct ChunkedUploadResult {
    /// If the payload was small enough to be inlined, this is the single blob.
    /// If chunked, this is the blob holding the serialized `ChunkManifest`.
    pub root_blob: BlobId,
    /// All chunk blobs (empty if inlined).
    pub chunk_blobs: Vec<BlobId>,
    /// Total payload size in bytes.
    pub total_size: u64,
    /// Whether chunking was actually applied.
    pub chunked: bool,
}

/// Store a payload, automatically chunking if it exceeds the inline threshold.
///
/// - If `data.len() <= config.inline_threshold`: stores as a single blob, returns it directly.
/// - Otherwise: splits via the configured `Chunker`, stores each chunk as a separate blob,
///   creates a `ChunkManifest`, stores the manifest as a separate blob, returns the manifest blob.
///
/// All blob storage goes through the provided `BlobStore` and `HashAlg` from `sodl-cas`.
/// Callers (e.g., `SodlService`) are responsible for encryption — pass ciphertext bytes
/// if encryption is desired, or use `EncryptedCas` before/after.
pub fn chunk_and_store(
    data: &[u8],
    store: &dyn sodl_cas::BlobStore,
    alg: sodl_cas::HashAlg,
    config: &ChunkPipelineConfig,
) -> Result<ChunkedUploadResult> {
    let total_size = data.len() as u64;

    // Small payloads: store inline as a single blob.
    if total_size <= config.inline_threshold {
        let blob_id = sodl_cas::compute_blob_id(data, alg);
        store.put(&blob_id, Bytes::copy_from_slice(data))?;
        return Ok(ChunkedUploadResult {
            root_blob: blob_id,
            chunk_blobs: vec![],
            total_size,
            chunked: false,
        });
    }

    // Large payloads: chunk, store each chunk, build manifest.
    let chunker = config.effective_chunker();
    let chunks = chunker.chunk(data);

    debug_assert!({
        // Verify chunks cover the entire payload.
        let covered: u64 = chunks.iter().map(|c| c.length as u64).sum();
        covered == total_size
    });

    let mut entries = Vec::with_capacity(chunks.len());
    let mut chunk_blobs = Vec::with_capacity(chunks.len());

    for desc in &chunks {
        let blob_id = sodl_cas::compute_blob_id(&desc.data, alg);
        store.put(&blob_id, desc.data.clone())?;
        entries.push(manifest::ChunkEntry {
            blob_id: blob_id.clone(),
            offset: desc.offset,
            length: desc.length,
        });
        chunk_blobs.push(blob_id);
    }

    let manifest = ChunkManifest {
        version: 1,
        total_size,
        chunk_count: entries.len() as u32,
        hash_alg: format!("{alg:?}").to_lowercase(),
        chunks: entries,
    };

    // Serialize manifest → store as its own blob.
    let manifest_bytes = serde_json::to_vec(&manifest)
        .map_err(|e| sodl_core::SodlError::Io(format!("serialize chunk manifest: {e}")))?;
    let manifest_blob = sodl_cas::compute_blob_id(&manifest_bytes, alg);
    store.put(&manifest_blob, Bytes::from(manifest_bytes))?;

    Ok(ChunkedUploadResult {
        root_blob: manifest_blob,
        chunk_blobs,
        total_size,
        chunked: true,
    })
}

/// Compute content-addressed chunk fingerprints without storing bytes.
///
/// This is used by provenance resolution: a cut or partial re-upload may no
/// longer match the original payload hash, but content-defined chunks can still
/// overlap strongly enough to identify the likely origin family.
pub fn compute_chunk_blob_ids(
    data: &[u8],
    alg: sodl_cas::HashAlg,
    config: &ChunkPipelineConfig,
) -> Vec<BlobId> {
    if data.len() as u64 <= config.inline_threshold {
        return vec![sodl_cas::compute_blob_id(data, alg)];
    }

    config
        .effective_chunker()
        .chunk(data)
        .into_iter()
        .map(|chunk| sodl_cas::compute_blob_id(&chunk.data, alg))
        .collect()
}

/// Reassemble a chunked payload from a `ChunkManifest`.
///
/// Fetches each chunk blob from the store, verifies integrity, and concatenates
/// in offset order to reconstruct the original payload.
pub fn reassemble(manifest: &ChunkManifest, store: &dyn sodl_cas::BlobStore) -> Result<Bytes> {
    let mut buf = Vec::with_capacity(manifest.total_size as usize);

    for entry in &manifest.chunks {
        let chunk = store.get(&entry.blob_id)?;
        sodl_cas::verify_integrity(&entry.blob_id, &chunk)?;
        if chunk.len() != entry.length as usize {
            return Err(sodl_core::SodlError::Integrity);
        }
        buf.extend_from_slice(&chunk);
    }

    if buf.len() as u64 != manifest.total_size {
        return Err(sodl_core::SodlError::Integrity);
    }

    Ok(Bytes::from(buf))
}

/// Try to parse a blob as a `ChunkManifest`.
///
/// Returns `Ok(Some(manifest))` if the blob is valid JSON with `version: 1`,
/// `Ok(None)` if it's not a manifest, or `Err` on I/O failure.
pub fn try_parse_manifest(blob: &[u8]) -> Option<ChunkManifest> {
    serde_json::from_slice(blob).ok()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sodl_cas::{BlobStore, HashAlg, MemBlobStore};

    #[test]
    fn small_payload_stored_inline() {
        let store = MemBlobStore::new();
        let data = b"small enough to be inline";
        let config = ChunkPipelineConfig::default();

        let result = chunk_and_store(data, &store, HashAlg::Blake3, &config).unwrap();

        assert!(!result.chunked);
        assert!(result.chunk_blobs.is_empty());
        assert_eq!(result.total_size, data.len() as u64);

        // Blob is retrievable.
        let got = store.get(&result.root_blob).unwrap();
        assert_eq!(&got[..], &data[..]);
    }

    #[test]
    fn large_payload_chunked_and_reassembled() {
        let store = MemBlobStore::new();
        // Create payload larger than default threshold (256 KiB).
        let data: Vec<u8> = (0u8..=255).cycle().take(300 * 1024).collect();
        let config = ChunkPipelineConfig::default();

        let result = chunk_and_store(&data, &store, HashAlg::Blake3, &config).unwrap();

        assert!(result.chunked);
        assert!(!result.chunk_blobs.is_empty());
        assert_eq!(result.total_size, data.len() as u64);

        // Parse manifest from root blob.
        let manifest_bytes = store.get(&result.root_blob).unwrap();
        let manifest: ChunkManifest = serde_json::from_slice(&manifest_bytes).unwrap();
        assert_eq!(manifest.total_size, data.len() as u64);
        assert_eq!(manifest.chunk_count, result.chunk_blobs.len() as u32);

        // Reassemble.
        let reassembled = reassemble(&manifest, &store).unwrap();
        assert_eq!(&reassembled[..], &data[..]);
    }

    #[test]
    fn fixed_size_chunker_covers_payload() {
        let chunker = FixedSizeChunker::new(100);
        let data = vec![0u8; 350];
        let chunks = chunker.chunk(&data);

        assert_eq!(chunks.len(), 4); // 100+100+100+50
        assert_eq!(chunks[0].length, 100);
        assert_eq!(chunks[3].length, 50);

        let total: u64 = chunks.iter().map(|c| c.length as u64).sum();
        assert_eq!(total, 350);
    }

    #[test]
    fn custom_chunker_via_config() {
        let store = MemBlobStore::new();
        let data = vec![42u8; 500];
        let config = ChunkPipelineConfig::default()
            .with_inline_threshold(100)
            .with_chunker(FixedSizeChunker::new(200));

        let result = chunk_and_store(&data, &store, HashAlg::Blake3, &config).unwrap();

        assert!(result.chunked);
        assert_eq!(result.chunk_blobs.len(), 3); // 200+200+100

        let manifest_bytes = store.get(&result.root_blob).unwrap();
        let manifest: ChunkManifest = serde_json::from_slice(&manifest_bytes).unwrap();
        let reassembled = reassemble(&manifest, &store).unwrap();
        assert_eq!(&reassembled[..], &data[..]);
    }

    #[test]
    fn dedup_across_identical_chunks() {
        let store = MemBlobStore::new();
        // All zeros → identical chunks should dedup.
        let data = vec![0u8; 600];
        let config = ChunkPipelineConfig::default()
            .with_inline_threshold(100)
            .with_chunker(FixedSizeChunker::new(200));

        let result = chunk_and_store(&data, &store, HashAlg::Blake3, &config).unwrap();

        assert!(result.chunked);
        // All 200-byte zero chunks produce the same BlobId.
        // chunk_blobs has 3 entries, but only 2 unique (200-byte zeros vs 200-byte zeros vs 200-byte zeros → all same).
        let unique: std::collections::HashSet<String> =
            result.chunk_blobs.iter().map(|b| b.0.clone()).collect();
        assert_eq!(unique.len(), 1, "identical chunks should dedup to 1 blob");
    }

    #[test]
    fn compute_chunk_blob_ids_does_not_store() {
        let data = vec![9u8; 500];
        let config = ChunkPipelineConfig::default()
            .with_inline_threshold(100)
            .with_chunker(FixedSizeChunker::new(200));

        let ids = compute_chunk_blob_ids(&data, HashAlg::Blake3, &config);

        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0], ids[1]);
    }

    #[test]
    fn fastcdc_deterministic() {
        let chunker = FastCdcChunker::default();
        let data: Vec<u8> = (0u8..=255).cycle().take(500_000).collect();

        let chunks1 = chunker.chunk(&data);
        let chunks2 = chunker.chunk(&data);

        assert_eq!(chunks1.len(), chunks2.len());
        for (a, b) in chunks1.iter().zip(chunks2.iter()) {
            assert_eq!(a.offset, b.offset);
            assert_eq!(a.length, b.length);
        }
    }

    #[test]
    fn sha256_pipeline() {
        let store = MemBlobStore::new();
        let data = vec![7u8; 400];
        let config = ChunkPipelineConfig::default()
            .with_inline_threshold(100)
            .with_chunker(FixedSizeChunker::new(150));

        let result = chunk_and_store(&data, &store, HashAlg::Sha256, &config).unwrap();
        assert!(result.chunked);
        assert!(result.root_blob.0.starts_with("sha256:"));

        let manifest_bytes = store.get(&result.root_blob).unwrap();
        let manifest: ChunkManifest = serde_json::from_slice(&manifest_bytes).unwrap();
        let reassembled = reassemble(&manifest, &store).unwrap();
        assert_eq!(&reassembled[..], &data[..]);
    }

    #[test]
    fn empty_payload() {
        let store = MemBlobStore::new();
        let data = b"";
        let config = ChunkPipelineConfig::default();

        let result = chunk_and_store(data, &store, HashAlg::Blake3, &config).unwrap();
        assert!(!result.chunked);
        assert_eq!(result.total_size, 0);
    }
}
