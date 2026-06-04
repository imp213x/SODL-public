//! Indexes: reference counting + lineage edges (skeleton).
//!
//! Why this crate exists:
//! - "Store once" requires knowing whether content is still reachable.
//! - Deletion/GC must be safe and policy-aware.
//!
//! We track:
//! - OriginRefCount: strong references to an origin (shares, derivations, pins if you treat pins as refs)
//! - BlobRefCount: strong references to blobs (origin representations, derived representations)
//! - Lineage edges for audit and provenance
//!
//! V1 keeps this as an interface + in-memory reference impl.

use serde::{Deserialize, Serialize};
use sodl_core::{BlobId, DerivationId, OriginId, PrincipalId, Result, ShareId};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RefKind {
    OriginRepresentation {
        name: String,
    },
    Share {
        share_id: ShareId,
        from: PrincipalId,
        to: PrincipalId,
    },
    Derivation {
        derivation_id: DerivationId,
    },
    Pin {
        pin_id: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageEdge {
    pub edge_id: String,
    pub origin_id: OriginId,
    pub blob_id: Option<BlobId>,
    pub kind: RefKind,
    pub created_at: time::OffsetDateTime,
}

pub trait RefCounter: Send + Sync {
    fn inc_origin(&self, origin_id: OriginId, reason: RefKind) -> Result<()>;
    fn dec_origin(&self, origin_id: OriginId, reason: RefKind) -> Result<()>;
    fn get_origin(&self, origin_id: OriginId) -> Result<i64>;

    fn inc_blob(&self, blob_id: &BlobId, reason: RefKind) -> Result<()>;
    fn dec_blob(&self, blob_id: &BlobId, reason: RefKind) -> Result<()>;
    fn get_blob(&self, blob_id: &BlobId) -> Result<i64>;
}

pub trait ScanIndex: Send + Sync {
    fn list_origins(&self) -> Result<Vec<OriginId>>;
    fn list_blobs(&self) -> Result<Vec<BlobId>>;
}

pub trait LineageStore: Send + Sync {
    fn add_edge(&self, e: LineageEdge) -> Result<()>;
    fn list_edges_for_origin(&self, origin_id: OriginId) -> Result<Vec<LineageEdge>>;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ProvenanceMatchKind {
    ExactPayload,
    ChunkOverlap {
        matched_chunks: usize,
        total_chunks: usize,
        ratio: f32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProvenanceCandidate {
    pub origin_id: OriginId,
    pub kind: ProvenanceMatchKind,
    pub confidence: f32,
}

/// Internal fingerprint index used to resolve provenance across repeated uploads
/// and byte-range overlap. Fingerprints are intentionally metadata, not storage
/// addresses: encrypted blob IDs can remain origin-specific while provenance can
/// still be discovered.
pub trait ProvenanceIndex: Send + Sync {
    fn put_payload_fingerprint(&self, origin_id: OriginId, fingerprint: &str) -> Result<()>;
    fn put_chunk_fingerprints(&self, origin_id: OriginId, fingerprints: &[String]) -> Result<()>;
    fn find_by_payload_fingerprint(&self, fingerprint: &str) -> Result<Vec<OriginId>>;
    fn find_by_chunk_fingerprints(&self, fingerprints: &[String])
        -> Result<Vec<(OriginId, usize)>>;
}

#[derive(Clone, Default)]
pub struct MemIndex {
    origin_counts: Arc<RwLock<HashMap<OriginId, i64>>>,
    blob_counts: Arc<RwLock<HashMap<String, i64>>>,
    edges: Arc<RwLock<Vec<LineageEdge>>>,
    payload_fingerprints: Arc<RwLock<HashMap<String, Vec<OriginId>>>>,
    chunk_fingerprints: Arc<RwLock<HashMap<String, Vec<OriginId>>>>,
}

impl MemIndex {
    pub fn new() -> Self {
        Self::default()
    }
}

impl RefCounter for MemIndex {
    fn inc_origin(&self, origin_id: OriginId, _reason: RefKind) -> Result<()> {
        let mut w = self
            .origin_counts
            .write()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        *w.entry(origin_id).or_insert(0) += 1;
        Ok(())
    }
    fn dec_origin(&self, origin_id: OriginId, _reason: RefKind) -> Result<()> {
        let mut w = self
            .origin_counts
            .write()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        let e = w.entry(origin_id).or_insert(0);
        *e -= 1;
        if *e < 0 {
            *e = 0;
        }
        Ok(())
    }
    fn get_origin(&self, origin_id: OriginId) -> Result<i64> {
        let r = self
            .origin_counts
            .read()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        Ok(*r.get(&origin_id).unwrap_or(&0))
    }

    fn inc_blob(&self, blob_id: &BlobId, _reason: RefKind) -> Result<()> {
        let mut w = self
            .blob_counts
            .write()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        *w.entry(blob_id.0.clone()).or_insert(0) += 1;
        Ok(())
    }
    fn dec_blob(&self, blob_id: &BlobId, _reason: RefKind) -> Result<()> {
        let mut w = self
            .blob_counts
            .write()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        let e = w.entry(blob_id.0.clone()).or_insert(0);
        *e -= 1;
        if *e < 0 {
            *e = 0;
        }
        Ok(())
    }
    fn get_blob(&self, blob_id: &BlobId) -> Result<i64> {
        let r = self
            .blob_counts
            .read()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        Ok(*r.get(&blob_id.0).unwrap_or(&0))
    }
}

impl ScanIndex for MemIndex {
    fn list_origins(&self) -> Result<Vec<OriginId>> {
        let r = self
            .origin_counts
            .read()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        Ok(r.keys().cloned().collect())
    }
    fn list_blobs(&self) -> Result<Vec<BlobId>> {
        let r = self
            .blob_counts
            .read()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        Ok(r.keys().cloned().map(BlobId).collect())
    }
}

impl LineageStore for MemIndex {
    fn add_edge(&self, e: LineageEdge) -> Result<()> {
        self.edges
            .write()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?
            .push(e);
        Ok(())
    }
    fn list_edges_for_origin(&self, origin_id: OriginId) -> Result<Vec<LineageEdge>> {
        let r = self
            .edges
            .read()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        Ok(r.iter()
            .filter(|e| e.origin_id == origin_id)
            .cloned()
            .collect())
    }
}

impl ProvenanceIndex for MemIndex {
    fn put_payload_fingerprint(&self, origin_id: OriginId, fingerprint: &str) -> Result<()> {
        let mut w = self
            .payload_fingerprints
            .write()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        let origins = w.entry(fingerprint.to_string()).or_default();
        if !origins.contains(&origin_id) {
            origins.push(origin_id);
        }
        Ok(())
    }

    fn put_chunk_fingerprints(&self, origin_id: OriginId, fingerprints: &[String]) -> Result<()> {
        let mut w = self
            .chunk_fingerprints
            .write()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        for fingerprint in fingerprints {
            let origins = w.entry(fingerprint.clone()).or_default();
            if !origins.contains(&origin_id) {
                origins.push(origin_id);
            }
        }
        Ok(())
    }

    fn find_by_payload_fingerprint(&self, fingerprint: &str) -> Result<Vec<OriginId>> {
        let r = self
            .payload_fingerprints
            .read()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        Ok(r.get(fingerprint).cloned().unwrap_or_default())
    }

    fn find_by_chunk_fingerprints(
        &self,
        fingerprints: &[String],
    ) -> Result<Vec<(OriginId, usize)>> {
        let r = self
            .chunk_fingerprints
            .read()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        let mut counts: HashMap<OriginId, usize> = HashMap::new();
        for fingerprint in fingerprints {
            if let Some(origins) = r.get(fingerprint) {
                for origin_id in origins {
                    *counts.entry(*origin_id).or_default() += 1;
                }
            }
        }
        let mut out = counts.into_iter().collect::<Vec<_>>();
        out.sort_by(|a, b| b.1.cmp(&a.1));
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Access tracking: records how often each blob is accessed and when.
// Used by the tiering policy to promote hot blobs into RAM and demote cold ones.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessRecord {
    pub blob_id: BlobId,
    pub access_count: u64,
    pub first_accessed_at: time::OffsetDateTime,
    pub last_accessed_at: time::OffsetDateTime,
}

pub trait AccessTracker: Send + Sync {
    fn record_access(&self, blob_id: &BlobId) -> Result<()>;
    fn get_record(&self, blob_id: &BlobId) -> Result<Option<AccessRecord>>;
    fn list_records(&self) -> Result<Vec<AccessRecord>>;
}

#[derive(Clone, Default)]
pub struct MemAccessTracker {
    records: Arc<RwLock<HashMap<String, AccessRecord>>>,
}

impl MemAccessTracker {
    pub fn new() -> Self {
        Self::default()
    }
}

impl AccessTracker for MemAccessTracker {
    fn record_access(&self, blob_id: &BlobId) -> Result<()> {
        let now = time::OffsetDateTime::now_utc();
        let mut w = self
            .records
            .write()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        let rec = w.entry(blob_id.0.clone()).or_insert_with(|| AccessRecord {
            blob_id: blob_id.clone(),
            access_count: 0,
            first_accessed_at: now,
            last_accessed_at: now,
        });
        rec.access_count += 1;
        rec.last_accessed_at = now;
        Ok(())
    }

    fn get_record(&self, blob_id: &BlobId) -> Result<Option<AccessRecord>> {
        let r = self
            .records
            .read()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        Ok(r.get(&blob_id.0).cloned())
    }

    fn list_records(&self) -> Result<Vec<AccessRecord>> {
        let r = self
            .records
            .read()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        Ok(r.values().cloned().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sodl_core::{new_origin_id, BlobId};

    #[test]
    fn counts_go_up_and_down() {
        let idx = MemIndex::new();
        let o = new_origin_id();
        let b = BlobId("blake3:abc".into());

        idx.inc_origin(
            o,
            RefKind::Pin {
                pin_id: "p1".into(),
            },
        )
        .unwrap();
        idx.inc_origin(
            o,
            RefKind::Pin {
                pin_id: "p2".into(),
            },
        )
        .unwrap();
        assert_eq!(idx.get_origin(o).unwrap(), 2);

        idx.dec_origin(
            o,
            RefKind::Pin {
                pin_id: "p1".into(),
            },
        )
        .unwrap();
        assert_eq!(idx.get_origin(o).unwrap(), 1);

        idx.inc_blob(
            &b,
            RefKind::OriginRepresentation {
                name: "source".into(),
            },
        )
        .unwrap();
        assert_eq!(idx.get_blob(&b).unwrap(), 1);
        idx.dec_blob(
            &b,
            RefKind::OriginRepresentation {
                name: "source".into(),
            },
        )
        .unwrap();
        assert_eq!(idx.get_blob(&b).unwrap(), 0);
    }

    #[test]
    fn provenance_index_finds_exact_and_chunk_overlap() {
        let idx = MemIndex::new();
        let a = new_origin_id();
        let b = new_origin_id();

        idx.put_payload_fingerprint(a, "payload:1").unwrap();
        idx.put_chunk_fingerprints(a, &["chunk:a".into(), "chunk:b".into()])
            .unwrap();
        idx.put_chunk_fingerprints(b, &["chunk:b".into()]).unwrap();

        assert_eq!(
            idx.find_by_payload_fingerprint("payload:1").unwrap(),
            vec![a]
        );

        let matches = idx
            .find_by_chunk_fingerprints(&["chunk:a".into(), "chunk:b".into()])
            .unwrap();
        assert_eq!(matches[0], (a, 2));
        assert_eq!(matches[1], (b, 1));
    }
}
