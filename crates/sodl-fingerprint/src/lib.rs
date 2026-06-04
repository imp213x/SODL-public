//! Fingerprinting & watermark hooks (skeleton).
//!
//! V1: define interfaces only. Implementations can be swapped later.
//!
//! Goal:
//! - Identify origin across **transformations** (re-encode, resize, etc.)
//! - Support compliance workflows (takedown inside controlled ecosystem)
//! - Support dedupe hints when bytes are not identical

use serde::{Deserialize, Serialize};
use sodl_core::{FingerprintId, OriginId, Result};

/// A perceptual fingerprint (opaque bytes) computed from media.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fingerprint(pub Vec<u8>);

/// Fingerprint engine boundary.
pub trait Fingerprinter: Send + Sync {
    fn fingerprint_bytes(&self, media: &[u8]) -> Result<Fingerprint>;
}

/// Watermark detection boundary (forensic watermarking).
pub trait WatermarkDetector: Send + Sync {
    fn detect_origin(&self, media: &[u8]) -> Result<Option<OriginId>>;
}

/// Index for matching fingerprints to origins.
pub trait FingerprintIndex: Send + Sync {
    fn put(&self, origin_id: OriginId, fp_id: FingerprintId, fp: Fingerprint) -> Result<()>;
    fn query(&self, fp: &Fingerprint) -> Result<Vec<(OriginId, f32)>>; // (origin, score)
}
