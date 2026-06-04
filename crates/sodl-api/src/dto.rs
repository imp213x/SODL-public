//! JSON request and response schemas for the SODL REST API.
//!
//! These are thin wrappers that translate between HTTP-friendly JSON and the
//! internal SODL domain types.  They intentionally avoid exposing internal
//! implementation details (e.g. `time::OffsetDateTime` is serialised as an
//! ISO-8601 string).

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Upload
// ---------------------------------------------------------------------------

/// Request body for `POST /v1/upload` (JSON metadata — bytes sent as multipart).
#[derive(Debug, Deserialize)]
pub struct UploadMeta {
    pub owner: String,
    pub media_kind: String,
    #[serde(default)]
    pub mime: Option<String>,
    #[serde(default = "default_durability")]
    pub durability: String,
}

fn default_durability() -> String {
    "best_effort".into()
}

#[derive(Debug, Serialize)]
pub struct UploadResponse {
    pub origin_id: String,
    pub blob_id: String,
    pub chunked: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub chunk_blobs: Vec<String>,
}

// ---------------------------------------------------------------------------
// Provenance resolution
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ResolveProvenanceMeta {
    #[serde(default)]
    pub media_kind: Option<String>,
    #[serde(default)]
    pub mime: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ProvenanceCandidateResponse {
    pub origin_id: String,
    pub match_kind: String,
    pub confidence: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_chunks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_chunks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ratio: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct ResolveProvenanceResponse {
    pub payload_fingerprint: String,
    pub chunk_count: usize,
    pub candidates: Vec<ProvenanceCandidateResponse>,
}

// ---------------------------------------------------------------------------
// Origin
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct CreateRepresentationRequest {
    pub name: String,
    #[serde(default = "default_media_kind")]
    pub media_kind: String,
    #[serde(default)]
    pub mime: Option<String>,
    #[serde(default)]
    pub size_bytes: Option<u64>,
    #[serde(default)]
    pub root_blobs: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct CreateOriginRequest {
    #[serde(default)]
    pub owner: Option<String>,
    #[serde(default = "default_media_kind")]
    pub media_kind: String,
    #[serde(default = "default_durability")]
    pub durability: String,
    #[serde(default)]
    pub representations: Vec<CreateRepresentationRequest>,
}

#[derive(Debug, Serialize)]
pub struct OriginResponse {
    pub origin_id: String,
    pub media_kind: String,
    pub durability: String,
    pub created_at: String,
    pub tombstoned_at: Option<String>,
    pub representations: Vec<RepresentationResponse>,
    pub owner: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ListOriginsResponse {
    pub origins: Vec<OriginResponse>,
}

#[derive(Debug, Serialize)]
pub struct RepresentationResponse {
    pub name: String,
    pub media_kind: String,
    pub mime: Option<String>,
    pub size_bytes: Option<u64>,
    pub root_blobs: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct RepresentationsResponse {
    pub origin_id: String,
    pub representations: Vec<RepresentationResponse>,
}

// ---------------------------------------------------------------------------
// Blob
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct BlobCreateResponse {
    pub blob_id: String,
    pub existed: bool,
    pub size_bytes: usize,
}

#[derive(Debug, Serialize)]
pub struct BlobListResponse {
    pub blobs: Vec<String>,
}

// ---------------------------------------------------------------------------
// Share
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ShareRequest {
    pub from: String,
    pub to: String,
    pub origin_id: String,
    #[serde(default = "default_caps")]
    pub capabilities: Vec<String>,
}

fn default_caps() -> Vec<String> {
    vec!["read".into()]
}

#[derive(Debug, Serialize)]
pub struct ShareResponse {
    pub share_id: String,
    pub origin_id: String,
}

#[derive(Debug, Serialize)]
pub struct ShareDetailResponse {
    pub share_id: String,
    pub origin_id: String,
    pub from: String,
    pub to: String,
    pub capabilities: Vec<String>,
    pub created_at: String,
    pub lineage_proof_digest: String,
}

// ---------------------------------------------------------------------------
// Derivation
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct DeriveRequest {
    pub origin_id: String,
    /// One of: "trim", "crop", "transform"
    pub kind: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default = "default_media_kind")]
    pub media_kind: String,
}

fn default_media_kind() -> String {
    "binary".into()
}

#[derive(Debug, Serialize)]
pub struct DeriveResponse {
    pub derivation_id: String,
    pub origin_id: String,
}

// ---------------------------------------------------------------------------
// Pin
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct PinRequest {
    pub origin_id: String,
    pub requested_by: String,
    #[serde(default = "default_replicas")]
    pub min_replicas: u8,
}

fn default_replicas() -> u8 {
    1
}

#[derive(Debug, Serialize)]
pub struct PinResponse {
    pub pin_id: String,
    pub origin_id: String,
}

// ---------------------------------------------------------------------------
// Tombstone
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct TombstoneRequest {
    #[serde(default = "default_reason")]
    pub reason: String,
}

fn default_reason() -> String {
    "deleted by owner".into()
}

// ---------------------------------------------------------------------------
// Lineage proof
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct LineageProofResponse {
    pub origin_id: String,
    pub digest: String,
    pub created_at: String,
}

// ---------------------------------------------------------------------------
// Verify
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct VerifyResponse {
    pub share_id: String,
    pub valid: bool,
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}
