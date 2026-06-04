//! Manifests define derivations (edits) and shares as metadata over immutable origins.
//!
//! SODL principle: **metadata over mutation**.
//! - View-like edits (trim/crop) should stay as manifests.
//! - Transform edits may materialize new blobs, but still belong to the same origin lineage.

use serde::{Deserialize, Serialize};
use sodl_core::{
    BlobId, Capability, DerivationId, FingerprintId, MediaKind, OriginId, PrincipalId, ShareId,
    SODL_SCHEMA_VERSION,
};

/// Optional description of a media timeline segment. Useful for video/audio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineRange {
    pub start_ms: u64,
    pub end_ms: u64,
}

/// Derivation kind.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DerivationKind {
    /// Non-destructive trim as a view over time range.
    Trim { range: TimelineRange },

    /// A logical crop (render-time), not baked into pixels.
    Crop { x: u32, y: u32, w: u32, h: u32 },

    /// A transform that may require new bytes (placeholder).
    Transform { description: String },
}

/// A derivation manifest describes how to obtain a derived view/object from its origin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivationManifest {
    pub schema: String,
    pub origin_id: OriginId,
    pub derivation_id: DerivationId,
    pub parent: Option<DerivationId>,
    pub created_at: time::OffsetDateTime,
    pub media_kind: MediaKind,
    pub kind: DerivationKind,

    /// Optional materialized output blob(s) if transform produced new bytes.
    pub output_blobs: Vec<BlobId>,

    /// Optional fingerprints for derived content (useful if transform creates new bytes).
    pub fingerprint_ids: Vec<FingerprintId>,
}

impl DerivationManifest {
    pub fn new(
        origin_id: OriginId,
        derivation_id: DerivationId,
        media_kind: MediaKind,
        kind: DerivationKind,
    ) -> Self {
        Self {
            schema: SODL_SCHEMA_VERSION.to_string(),
            origin_id,
            derivation_id,
            parent: None,
            created_at: time::OffsetDateTime::now_utc(),
            media_kind,
            kind,
            output_blobs: vec![],
            fingerprint_ids: vec![],
        }
    }

    pub fn validate(&self) -> sodl_core::Result<()> {
        if self.schema != SODL_SCHEMA_VERSION {
            return Err(sodl_core::SodlError::Invalid(format!(
                "schema mismatch: expected {}, got {}",
                SODL_SCHEMA_VERSION, self.schema
            )));
        }
        Ok(())
    }
}

/// A share edge connects two principals in the lineage graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareRecord {
    pub schema: String,
    pub share_id: ShareId,
    pub origin_id: OriginId,
    pub derivation_id: Option<DerivationId>,
    pub from_principal: PrincipalId,
    pub to_principal: PrincipalId,
    pub created_at: time::OffsetDateTime,
    pub capabilities: Vec<Capability>, // e.g., [Read, Reshare]

    /// Deterministic, unsigned lineage proof digest (Blake3 hex) computed at share time.
    pub lineage_proof_digest: String,
    /// UTC timestamp of proof computation.
    pub lineage_proof_created_at: time::OffsetDateTime,
    /// Optional signer key id (for signed proofs in later steps).
    pub lineage_proof_key_id: Option<String>,
    /// Optional signature over the proof digest (base64), if signed.
    pub lineage_proof_sig_b64: Option<String>,
}

impl ShareRecord {
    pub fn validate(&self) -> sodl_core::Result<()> {
        if self.schema != SODL_SCHEMA_VERSION {
            return Err(sodl_core::SodlError::Invalid(format!(
                "schema mismatch: expected {}, got {}",
                SODL_SCHEMA_VERSION, self.schema
            )));
        }
        if self.lineage_proof_digest.trim().is_empty() {
            return Err(sodl_core::SodlError::Invalid(
                "missing lineage_proof_digest".into(),
            ));
        }
        Ok(())
    }
}

/// Lineage graph node types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LineageNode {
    Origin {
        origin_id: OriginId,
    },
    Derivation {
        origin_id: OriginId,
        derivation_id: DerivationId,
    },
    Principal {
        principal: PrincipalId,
    },
}

/// Lineage graph edge types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LineageEdge {
    /// Principal -> Origin/Derivation share.
    Share {
        share_id: ShareId,
        from: PrincipalId,
        to: PrincipalId,
    },

    /// Derivation depends on a parent derivation.
    Derives {
        parent: DerivationId,
        child: DerivationId,
    },
}
