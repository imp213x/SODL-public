//! Origin registry + per-origin key management (skeleton).
//!
//! `OriginId` is the stable identity. An origin can have one or more representations
//! (e.g., source video + streaming segments + thumbnails) and multiple derived views.

use serde::{Deserialize, Serialize};
use sodl_core::{BlobId, Durability, FingerprintId, KeyRef, MediaKind, OriginId, Result};

/// A named representation of an origin.
///
/// Examples:
/// - `source` (original upload bytes)
/// - `hls_720p` (segmented streaming representation)
/// - `thumb_640w`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Representation {
    pub name: String,
    pub media_kind: MediaKind,
    pub mime: Option<String>,
    pub size_bytes: Option<u64>,

    /// Root blobs for this representation (single blob or chunk tree root(s)).
    pub root_blobs: Vec<BlobId>,
}

/// A registered origin (logical asset family).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OriginRecord {
    pub origin_id: OriginId,
    pub media_kind: MediaKind,
    pub durability: Durability,
    pub created_at: time::OffsetDateTime,
    pub tombstoned_at: Option<time::OffsetDateTime>,
    pub tombstone_reason: Option<String>,

    /// Multiple representations of the same logical origin.
    pub representations: Vec<Representation>,

    /// Optional fingerprints / watermark references.
    pub fingerprint_ids: Vec<FingerprintId>,

    /// Per-origin key reference (opaque).
    pub key_ref: Option<KeyRef>,

    /// Optional content owner / creator principal.
    pub owner: Option<sodl_core::PrincipalId>,
}

impl OriginRecord {
    pub fn new(origin_id: OriginId, media_kind: MediaKind, durability: Durability) -> Self {
        Self {
            origin_id,
            media_kind,
            durability,
            created_at: time::OffsetDateTime::now_utc(),
            tombstoned_at: None,
            tombstone_reason: None,
            representations: vec![],
            fingerprint_ids: vec![],
            key_ref: None,
            owner: None,
        }
    }
}

/// Origin registry interface (metadata store).
pub trait OriginRegistry: Send + Sync {
    fn create_origin(&self, record: OriginRecord) -> Result<()>;
    fn get_origin(&self, origin_id: OriginId) -> Result<OriginRecord>;
    fn update_origin(&self, record: OriginRecord) -> Result<()>;
    fn delete_origin(&self, origin_id: OriginId) -> Result<()>;
}

/// Key manager interface (skeleton).
///
/// In V1, this is purely a trait boundary: implementations may integrate KMS/HSM later.
/// We recommend **per-origin keys** to preserve dedupe across shares/derivations of the same origin.
pub trait KeyManager: Send + Sync {
    /// Ensures an origin has an associated key and returns a key reference.
    fn ensure_origin_key(&self, origin_id: OriginId) -> Result<KeyRef>;

    /// Wrap a per-origin key for a principal (user/device/service).
    fn wrap_for_principal(
        &self,
        origin_id: OriginId,
        principal: &sodl_core::PrincipalId,
    ) -> Result<Vec<u8>>;

    /// Unwrap for a principal (authorized path). Returns raw key bytes (opaque in skeleton).
    fn unwrap_for_principal(
        &self,
        origin_id: OriginId,
        principal: &sodl_core::PrincipalId,
        wrapped: &[u8],
    ) -> Result<Vec<u8>>;
}
