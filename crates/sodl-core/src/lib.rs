//! Shared primitives for SODL.
//!
//! This crate intentionally stays small and dependency-light.
//!
//! In SODL, **bytes** are immutable (blobs), while **identity** and **meaning**
//! are expressed through origins, derivations, and shares.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Schema tag for manifests/policies.
pub const SODL_SCHEMA_VERSION: &str = "sodl-v1";

/// A stable identifier for an origin (logical asset family).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OriginId(pub Uuid);

/// A stable identifier for a derivation (edit/view/transform) of an origin.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DerivationId(pub String);

/// A stable identifier for a share edge in the lineage graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShareId(pub String);

/// A stable identifier for an immutable blob/chunk.
///
/// Recommended format: `{alg}:{hex}` e.g. `blake3:...` or `sha256:...`
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlobId(pub String);

/// A fingerprint id (opaque reference) used to find origin across transformations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FingerprintId(pub String);

/// An opaque reference to cryptographic material in a key manager/KMS.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KeyRef(pub String);

/// Principal represents an entity capable of accessing content (user/device/service).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PrincipalId(pub String);

/// Durability class for an origin or blob reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Durability {
    Ephemeral,
    BestEffort,
    Durable,
}

/// Access capability (coarse; can be expanded later).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Read,
    Reshare,
    Derive,
    Pin,
    Admin,
}

/// Media type hint for content behavior (streaming, fingerprinting, etc.).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaKind {
    Binary,
    Image,
    Audio,
    Video,
    Document,
    AiModel,
}

// ---------------------------------------------------------------------------
// Weight-store domain types
// ---------------------------------------------------------------------------

/// Content-addressed identifier for a weight cluster blob.
///
/// Reuses `BlobId` semantics: the cluster ID is the CAS hash of the
/// serialised, compressed (and optionally encrypted) cluster bytes.
pub type ClusterId = BlobId;

/// A cluster of semantically related weight vectors.
///
/// Stores a centroid vector plus per-token lightweight offsets.
/// Multiple tokens share the same centroid — "store once, reference many."
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightCluster {
    /// Content-addressed ID (populated after storage).
    pub cluster_id: Option<ClusterId>,
    /// The shared centroid vector for this cluster.
    pub centroid: Vec<f32>,
    /// Token IDs that belong to this cluster.
    pub member_token_ids: Vec<u32>,
    /// Per-member offset from the centroid (same order as `member_token_ids`).
    pub offsets: Vec<Vec<f32>>,
    /// Dimensionality of each vector.
    pub dim: usize,
}

/// Metadata origin for a weight store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightOrigin {
    pub origin_id: OriginId,
    pub model_name: String,
    pub num_clusters: usize,
    pub quantization: String,
    pub created_at: time::OffsetDateTime,
}

/// Reason a weight cluster is pinned in the hot cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightPinReason {
    /// Core identity weights — always pinned, never evicted.
    Identity,
    /// Logic / routing weights that should stay resident in the hot cache.
    Logic,
    /// High access frequency during inference.
    FrequentUse,
    /// Prefetched based on predicted need.
    Prefetch,
}

#[derive(thiserror::Error, Debug)]
pub enum SodlError {
    #[error("not found")]
    NotFound,
    #[error("integrity check failed")]
    Integrity,
    #[error("unauthorized")]
    Unauthorized,
    #[error("conflict")]
    Conflict,
    #[error("invalid input: {0}")]
    Invalid(String),
    #[error("io: {0}")]
    Io(String),
    #[error("crypto: {0}")]
    Crypto(String),
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    #[error("compression: {0}")]
    Compression(String),
    #[error("serialization: {0}")]
    Serialization(String),
    #[error("weight store: {0}")]
    WeightStore(String),
}

pub type Result<T> = std::result::Result<T, SodlError>;

/// Convenience helper to create a new OriginId.
pub fn new_origin_id() -> OriginId {
    OriginId(Uuid::new_v4())
}
