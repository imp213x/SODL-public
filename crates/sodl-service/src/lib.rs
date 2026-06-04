//! SODL Service Facade (V1 skeleton)
//!
//! This crate provides a *high-level* API over the lower-level crates.
//! Think of it as the "application layer" that an app/server would embed.
//!
//! Goals:
//! - Keep core primitives in their own crates
//! - Provide ergonomic operations: upload/share/derive/pin
//! - Provide in-memory reference implementations to make the system runnable end-to-end

pub mod checkpoint_store;
pub mod optimizer_state;
pub mod weight_manifest;
pub mod weight_service;

use bytes::Bytes;
use sodl_chunk::ChunkPipelineConfig;
use sodl_core::{
    new_origin_id, Capability, DerivationId, MediaKind, OriginId, PrincipalId, Result, ShareId,
};
use sodl_crypto::Crypto;
use sodl_index::{LineageEdge, ProvenanceCandidate, ProvenanceIndex, ProvenanceMatchKind, RefKind};
use sodl_manifest::{DerivationKind, DerivationManifest, ShareRecord};
use sodl_origin::{OriginRecord, OriginRegistry, Representation};
use sodl_policy::{OriginPolicy, PinRecord, PinState, PinStore, PinTarget, PolicyStore};
use sodl_proof::generate_proof_unsigned;
use sodl_proof::LineageProof;
use sodl_store::EncryptedCas;

/// Metadata store for derivations.
pub trait DerivationStore: Send + Sync {
    fn put(&self, m: DerivationManifest) -> Result<()>;
    fn get(&self, origin_id: OriginId, derivation_id: &DerivationId) -> Result<DerivationManifest>;
    fn list_for_origin(&self, origin_id: OriginId) -> Result<Vec<DerivationManifest>>;
}

/// Metadata store for shares.
pub trait ShareStore: Send + Sync {
    fn put(&self, s: ShareRecord) -> Result<()>;
    fn get(&self, share_id: &ShareId) -> Result<ShareRecord>;
    fn list_for_origin(&self, origin_id: OriginId) -> Result<Vec<ShareRecord>>;

    fn get_share(&self, share_id: &ShareId) -> Result<ShareRecord> {
        self.get(share_id)
    }
}

/// Input for upload.
pub struct UploadRequest {
    pub owner: PrincipalId,
    pub media_kind: MediaKind,
    pub mime: Option<String>,
    pub durability_policy: OriginPolicy,
    pub bytes: Bytes,
}

/// Result of upload.
pub struct UploadResult {
    pub origin_id: OriginId,
    pub blob_id: sodl_core::BlobId,
    /// If the payload was chunked, this contains the chunk blob IDs.
    pub chunk_blobs: Vec<sodl_core::BlobId>,
    /// Whether the payload was chunked.
    pub chunked: bool,
}

pub struct ProvenanceResolution {
    pub payload_fingerprint: String,
    pub chunk_fingerprints: Vec<String>,
    pub candidates: Vec<ProvenanceCandidate>,
}

/// Main facade.
pub struct SodlService<'a> {
    pub index: &'a dyn sodl_index::RefCounter,
    pub lineage: &'a dyn sodl_index::LineageStore,
    pub provenance: &'a dyn ProvenanceIndex,

    pub origin_registry: &'a dyn OriginRegistry,
    pub policy_store: &'a dyn PolicyStore,
    pub pin_store: &'a dyn PinStore,
    pub derivations: &'a dyn DerivationStore,
    pub shares: &'a dyn ShareStore,

    /// Encrypted CAS helper against a durable store.
    pub enc_cas: EncryptedCas<'a>,

    /// Crypto provider (per-origin).
    pub crypto: &'a dyn Crypto,

    /// Optional signer for lineage proof digests (feature-gated algorithms live in sodl-proof).
    pub proof_signer: Option<&'a dyn sodl_proof::ProofSigner>,

    /// Chunking configuration. If `None`, chunking is disabled (legacy single-blob mode).
    pub chunk_config: Option<ChunkPipelineConfig>,
}

impl<'a> SodlService<'a> {
    /// Upload bytes as a new origin.
    ///
    /// If `chunk_config` is set and the payload exceeds the inline threshold,
    /// the bytes are split into content-defined chunks.  Each chunk is encrypted
    /// and stored individually, and a `ChunkManifest` is stored as the root blob.
    pub fn upload(&self, mut req: UploadRequest) -> Result<UploadResult> {
        // 1) allocate origin id
        let origin_id = new_origin_id();
        let plaintext_payload_fingerprint =
            sodl_cas::compute_blob_id(&req.bytes, sodl_cas::HashAlg::Blake3).0;
        let plaintext_chunk_fingerprints = self
            .chunk_config
            .as_ref()
            .map(|cfg| {
                sodl_chunk::compute_chunk_blob_ids(&req.bytes, sodl_cas::HashAlg::Blake3, cfg)
                    .into_iter()
                    .map(|blob_id| blob_id.0)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec![plaintext_payload_fingerprint.clone()]);

        // 2) store policy (ensure policy is keyed to real origin id)
        req.durability_policy.origin_id = origin_id;
        self.policy_store
            .put_origin_policy(req.durability_policy.clone())?;

        // 3) store bytes — chunked or single-blob
        let (root_blob, chunk_blobs, chunked) = if let Some(ref cfg) = self.chunk_config {
            // Chunked path: encrypt each chunk individually via EncryptedCas.
            // For NullCrypto this is identity; for real crypto each chunk gets
            // its own ciphertext but same origin key.
            let ciphertext = self.crypto.encrypt_for_origin(origin_id, req.bytes)?;
            let result = sodl_chunk::chunk_and_store(
                &ciphertext,
                self.enc_cas.store,
                self.enc_cas.hash_alg,
                cfg,
            )?;
            (result.root_blob, result.chunk_blobs, result.chunked)
        } else {
            // Legacy single-blob path.
            let blob_id = self.enc_cas.put_plain(origin_id, req.bytes)?;
            (blob_id, vec![], false)
        };

        let blob_id = root_blob.clone();

        // 4) create origin record — root_blobs includes manifest or single blob
        let mut root_blobs = vec![root_blob];
        if chunked {
            // Also record chunk blobs for refcounting.
            root_blobs.extend(chunk_blobs.iter().cloned());
        }

        let mut record = OriginRecord::new(
            origin_id,
            req.media_kind.clone(),
            req.durability_policy.retention.durability,
        );
        record.owner = Some(req.owner);
        record.representations.push(Representation {
            name: "source".into(),
            media_kind: req.media_kind,
            mime: req.mime,
            size_bytes: None,
            root_blobs: vec![blob_id.clone()],
        });

        self.origin_registry.create_origin(record)?;

        // Refcount + lineage
        self.index.inc_origin(
            origin_id,
            RefKind::OriginRepresentation {
                name: "source".into(),
            },
        )?;
        self.index.inc_blob(
            &blob_id,
            RefKind::OriginRepresentation {
                name: "source".into(),
            },
        )?;
        // Also refcount each chunk blob.
        for cb in &chunk_blobs {
            self.index.inc_blob(
                cb,
                RefKind::OriginRepresentation {
                    name: "source".into(),
                },
            )?;
        }
        self.lineage.add_edge(LineageEdge {
            edge_id: format!("edge:{}", uuid::Uuid::new_v4()),
            origin_id,
            blob_id: Some(blob_id.clone()),
            kind: RefKind::OriginRepresentation {
                name: "source".into(),
            },
            created_at: time::OffsetDateTime::now_utc(),
        })?;
        self.provenance
            .put_payload_fingerprint(origin_id, &plaintext_payload_fingerprint)?;
        self.provenance
            .put_chunk_fingerprints(origin_id, &plaintext_chunk_fingerprints)?;

        Ok(UploadResult {
            origin_id,
            blob_id,
            chunk_blobs,
            chunked,
        })
    }

    pub fn resolve_provenance(&self, bytes: &[u8]) -> Result<ProvenanceResolution> {
        let payload_fingerprint = sodl_cas::compute_blob_id(bytes, sodl_cas::HashAlg::Blake3).0;
        let chunk_fingerprints = self
            .chunk_config
            .as_ref()
            .map(|cfg| {
                sodl_chunk::compute_chunk_blob_ids(bytes, sodl_cas::HashAlg::Blake3, cfg)
                    .into_iter()
                    .map(|blob_id| blob_id.0)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec![payload_fingerprint.clone()]);

        let mut candidates = Vec::new();
        for origin_id in self
            .provenance
            .find_by_payload_fingerprint(&payload_fingerprint)?
        {
            candidates.push(ProvenanceCandidate {
                origin_id,
                kind: ProvenanceMatchKind::ExactPayload,
                confidence: 1.0,
            });
        }

        let exact_origins = candidates
            .iter()
            .map(|candidate| candidate.origin_id)
            .collect::<std::collections::HashSet<_>>();
        let total_chunks = chunk_fingerprints.len().max(1);
        for (origin_id, matched_chunks) in self
            .provenance
            .find_by_chunk_fingerprints(&chunk_fingerprints)?
        {
            if exact_origins.contains(&origin_id) {
                continue;
            }
            let ratio = matched_chunks as f32 / total_chunks as f32;
            if ratio <= 0.0 {
                continue;
            }
            candidates.push(ProvenanceCandidate {
                origin_id,
                kind: ProvenanceMatchKind::ChunkOverlap {
                    matched_chunks,
                    total_chunks,
                    ratio,
                },
                confidence: ratio.min(0.95),
            });
        }

        candidates.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(ProvenanceResolution {
            payload_fingerprint,
            chunk_fingerprints,
            candidates,
        })
    }

    /// Retrieve the full payload for an origin, automatically handling
    /// both single-blob and chunked storage.
    ///
    /// 1. Fetches the root blob from `Representation.root_blobs[0]`.
    /// 2. Attempts to parse it as a `ChunkManifest`.
    /// 3. If it's a manifest → reassembles all chunks → decrypts.
    /// 4. If it's a plain blob → decrypts directly.
    pub fn get_payload(&self, origin_id: OriginId) -> Result<Bytes> {
        let rec = self.origin_registry.get_origin(origin_id)?;
        let rep = rec
            .representations
            .first()
            .ok_or(sodl_core::SodlError::NotFound)?;
        let root_id = rep
            .root_blobs
            .first()
            .ok_or(sodl_core::SodlError::NotFound)?;

        // Fetch root blob bytes.
        let root_bytes = self.enc_cas.store.get(root_id)?;

        // Try to interpret as a chunk manifest.
        if let Some(manifest) = sodl_chunk::try_parse_manifest(&root_bytes) {
            // Reassemble chunks from the store.
            let ciphertext = sodl_chunk::reassemble(&manifest, self.enc_cas.store)?;
            // Decrypt the full reassembled ciphertext.
            self.crypto.decrypt_for_origin(origin_id, ciphertext)
        } else {
            // Single blob — verify integrity and decrypt.
            sodl_cas::verify_integrity(root_id, &root_bytes)?;
            self.crypto.decrypt_for_origin(origin_id, root_bytes)
        }
    }

    /// Create a share record from one principal to another.
    pub fn share(
        &self,
        from: PrincipalId,
        to: PrincipalId,
        origin_id: OriginId,
        caps: Vec<Capability>,
    ) -> Result<ShareId> {
        let share_id = ShareId(format!("share:{}", uuid::Uuid::new_v4()));
        let anchor_time = time::OffsetDateTime::now_utc();
        let proof = self.lineage_proof_at(origin_id, anchor_time)?;

        let (proof_key_id, proof_sig_b64) = if let Some(signer) = self.proof_signer {
            let sig = signer.sign_digest_b64(&proof.digest)?;
            (Some(signer.key_id().to_string()), Some(sig))
        } else {
            (None, None)
        };
        let s = ShareRecord {
            schema: sodl_core::SODL_SCHEMA_VERSION.to_string(),
            share_id: share_id.clone(),
            origin_id,
            derivation_id: None,
            from_principal: from,
            to_principal: to,
            created_at: time::OffsetDateTime::now_utc(),
            capabilities: caps,
            lineage_proof_digest: proof.digest.clone(),
            lineage_proof_created_at: anchor_time,
            lineage_proof_key_id: proof_key_id,
            lineage_proof_sig_b64: proof_sig_b64,
        };
        s.validate()?;
        self.shares.put(s.clone())?;

        self.index.inc_origin(
            origin_id,
            RefKind::Share {
                share_id: share_id.clone(),
                from: s.from_principal.clone(),
                to: s.to_principal.clone(),
            },
        )?;
        self.lineage.add_edge(LineageEdge {
            edge_id: format!("edge:{}", uuid::Uuid::new_v4()),
            origin_id,
            blob_id: None,
            kind: RefKind::Share {
                share_id: share_id.clone(),
                from: s.from_principal,
                to: s.to_principal,
            },
            created_at: time::OffsetDateTime::now_utc(),
        })?;

        Ok(share_id)
    }

    /// Record a derivation manifest (view/edit) for an origin.
    pub fn derive(
        &self,
        origin_id: OriginId,
        kind: DerivationKind,
        media_kind: MediaKind,
    ) -> Result<DerivationId> {
        let derivation_id = DerivationId(format!("drv:{}", uuid::Uuid::new_v4()));
        let m = DerivationManifest::new(origin_id, derivation_id.clone(), media_kind, kind);
        m.validate()?;
        self.derivations.put(m.clone())?;

        self.index.inc_origin(
            origin_id,
            RefKind::Derivation {
                derivation_id: derivation_id.clone(),
            },
        )?;
        self.lineage.add_edge(LineageEdge {
            edge_id: format!("edge:{}", uuid::Uuid::new_v4()),
            origin_id,
            blob_id: None,
            kind: RefKind::Derivation {
                derivation_id: derivation_id.clone(),
            },
            created_at: time::OffsetDateTime::now_utc(),
        })?;

        Ok(derivation_id)
    }

    /// Release a share (decrements origin refcount).
    pub fn release_share(&self, share_id: &ShareId) -> Result<()> {
        let s = self.shares.get(share_id)?;
        // Decrement refcount. (We keep the record for audit; a real impl may tombstone it.)
        self.index.dec_origin(
            s.origin_id,
            RefKind::Share {
                share_id: s.share_id,
                from: s.from_principal,
                to: s.to_principal,
            },
        )?;
        Ok(())
    }

    /// Release a pin (decrements origin refcount and marks pin released).
    pub fn unpin(&self, pin_id: &str) -> Result<()> {
        let p = self.pin_store.get_pin(pin_id)?;
        // mark released in store
        self.pin_store.release_pin(pin_id)?;
        if let PinTarget::Origin { origin_id } = p.target {
            self.index.dec_origin(
                origin_id,
                RefKind::Pin {
                    pin_id: pin_id.to_string(),
                },
            )?;
        }
        Ok(())
    }

    /// Simple GC eligibility check (policy-aware deletion is later).
    /// Returns true if origin has no strong refs.
    pub fn is_gc_candidate(&self, origin_id: OriginId) -> Result<bool> {
        Ok(self.index.get_origin(origin_id)? == 0)
    }

    /// Create a pin for an origin (durable intent).
    pub fn pin_origin(
        &self,
        requested_by: PrincipalId,
        origin_id: OriginId,
        min_replicas: u8,
    ) -> Result<String> {
        let pin_id = format!("pin:{}", uuid::Uuid::new_v4());
        let pin = PinRecord {
            pin_id: pin_id.clone(),
            target: PinTarget::Origin { origin_id },
            requested_by,
            created_at: time::OffsetDateTime::now_utc(),
            state: PinState::Pending,
            min_replicas: Some(min_replicas),
            required_zones: vec![],
        };
        self.pin_store.create_pin(pin.clone())?;

        // Treat pins as strong refs to origin for GC safety
        self.index.inc_origin(
            origin_id,
            RefKind::Pin {
                pin_id: pin_id.clone(),
            },
        )?;
        self.lineage.add_edge(LineageEdge {
            edge_id: format!("edge:{}", uuid::Uuid::new_v4()),
            origin_id,
            blob_id: None,
            kind: RefKind::Pin {
                pin_id: pin_id.clone(),
            },
            created_at: time::OffsetDateTime::now_utc(),
        })?;

        Ok(pin_id)
    }

    /// Tombstone an origin (tombstone-first hardening) and decrement blob refcounts for its representations.
    ///
    /// This does **not** delete bytes. Bytes are deleted by policy-aware GC when blob refcount reaches 0.
    ///
    /// V1 behavior:
    /// - mark origin record as tombstoned_at = now
    /// - decrement origin/blob refcounts for representations
    /// - clear representations (so future fetch requires derivation/share records)
    pub fn tombstone_origin(&self, origin_id: OriginId, reason: &str) -> Result<()> {
        let mut rec = self.origin_registry.get_origin(origin_id)?;

        // Mark tombstone
        rec.tombstoned_at = Some(time::OffsetDateTime::now_utc());
        rec.tombstone_reason = Some(reason.to_string());

        // Decrement origin representation ref + blob refs.
        for rep in &rec.representations {
            self.index.dec_origin(
                origin_id,
                RefKind::OriginRepresentation {
                    name: rep.name.clone(),
                },
            )?;
            for b in &rep.root_blobs {
                self.index.dec_blob(
                    b,
                    RefKind::OriginRepresentation {
                        name: rep.name.clone(),
                    },
                )?;
            }
        }

        // Clear reps to prevent serving as "active"
        rec.representations.clear();
        self.origin_registry.update_origin(rec)?;
        Ok(())
    }

    /// Compute a deterministic, unsigned lineage proof for the given origin based on current lineage edges.
    pub fn lineage_proof(&self, origin_id: OriginId) -> Result<LineageProof> {
        let edges = self.lineage.list_edges_for_origin(origin_id)?;
        generate_proof_unsigned(origin_id, edges, time::OffsetDateTime::now_utc())
    }

    /// Proof cutoff time for shares
    fn lineage_proof_at(
        &self,
        origin_id: OriginId,
        cutoff: time::OffsetDateTime,
    ) -> Result<LineageProof> {
        let mut edges = self.lineage.list_edges_for_origin(origin_id)?;

        // Only include edges strictly BEFORE the cutoff
        edges.retain(|e| e.created_at < cutoff);

        generate_proof_unsigned(origin_id, edges, cutoff)
    }

    /// Verify an anchored share proof. If the share has a signature, this requires a configured `proof_signer`.
    ///
    /// Returns:
    /// - Ok(true)  => valid (digest matches current lineage; and signature verifies if present)
    /// - Ok(false) => invalid (mismatch or bad signature)
    pub fn verify_share_proof(&self, share: &ShareRecord) -> Result<bool> {
        let proof = self.lineage_proof_at(share.origin_id, share.lineage_proof_created_at)?;

        if proof.digest != share.lineage_proof_digest {
            return Ok(false);
        }

        // If signed, verify signature too
        if let (Some(_kid), Some(sig)) = (&share.lineage_proof_key_id, &share.lineage_proof_sig_b64)
        {
            if let Some(signer) = self.proof_signer {
                return signer.verify_digest_b64(&share.lineage_proof_digest, sig);
            }
            // Signature present but no verifier configured
            return Ok(false);
        }

        Ok(true)
    }
}

// -------------------------------
// In-memory reference impls
// -------------------------------

pub use sodl_index::MemIndex;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Clone, Default)]
pub struct MemOriginRegistry {
    inner: Arc<RwLock<HashMap<OriginId, OriginRecord>>>,
}

impl MemOriginRegistry {
    pub fn new() -> Self {
        Self::default()
    }
}

impl OriginRegistry for MemOriginRegistry {
    fn create_origin(&self, record: OriginRecord) -> Result<()> {
        let mut w = self
            .inner
            .write()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        if w.contains_key(&record.origin_id) {
            return Err(sodl_core::SodlError::Conflict);
        }
        w.insert(record.origin_id, record);
        Ok(())
    }

    fn get_origin(&self, origin_id: OriginId) -> Result<OriginRecord> {
        self.inner
            .read()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?
            .get(&origin_id)
            .cloned()
            .ok_or(sodl_core::SodlError::NotFound)
    }

    fn update_origin(&self, record: OriginRecord) -> Result<()> {
        let mut w = self
            .inner
            .write()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        w.insert(record.origin_id, record);
        Ok(())
    }

    fn delete_origin(&self, origin_id: OriginId) -> Result<()> {
        self.inner
            .write()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?
            .remove(&origin_id);
        Ok(())
    }
}

#[derive(Clone, Default)]
pub struct MemPolicyStore {
    inner: Arc<RwLock<HashMap<OriginId, OriginPolicy>>>,
}

impl MemPolicyStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PolicyStore for MemPolicyStore {
    fn get_origin_policy(&self, origin_id: OriginId) -> Result<OriginPolicy> {
        self.inner
            .read()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?
            .get(&origin_id)
            .cloned()
            .ok_or(sodl_core::SodlError::NotFound)
    }

    fn put_origin_policy(&self, policy: OriginPolicy) -> Result<()> {
        self.inner
            .write()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?
            .insert(policy.origin_id, policy);
        Ok(())
    }
}

#[derive(Clone, Default)]
pub struct MemPinStore {
    inner: Arc<RwLock<HashMap<String, PinRecord>>>,
}

impl MemPinStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PinStore for MemPinStore {
    fn create_pin(&self, pin: PinRecord) -> Result<()> {
        self.inner
            .write()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?
            .insert(pin.pin_id.clone(), pin);
        Ok(())
    }

    fn get_pin(&self, pin_id: &str) -> Result<PinRecord> {
        self.inner
            .read()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?
            .get(pin_id)
            .cloned()
            .ok_or(sodl_core::SodlError::NotFound)
    }

    fn list_pins_for_origin(&self, origin_id: OriginId) -> Result<Vec<PinRecord>> {
        let r = self
            .inner
            .read()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        Ok(r.values()
            .filter(|p| matches!(p.target, PinTarget::Origin{ origin_id: oid } if oid == origin_id))
            .cloned()
            .collect())
    }

    fn update_pin(&self, pin: PinRecord) -> Result<()> {
        self.inner
            .write()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?
            .insert(pin.pin_id.clone(), pin);
        Ok(())
    }

    fn release_pin(&self, pin_id: &str) -> Result<()> {
        let mut w = self
            .inner
            .write()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        if let Some(mut p) = w.get(pin_id).cloned() {
            p.state = PinState::Released;
            w.insert(pin_id.to_string(), p);
            Ok(())
        } else {
            Err(sodl_core::SodlError::NotFound)
        }
    }
}

#[derive(Clone, Default)]
pub struct MemDerivationStore {
    inner: Arc<RwLock<HashMap<(OriginId, String), DerivationManifest>>>,
}

impl MemDerivationStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl DerivationStore for MemDerivationStore {
    fn put(&self, m: DerivationManifest) -> Result<()> {
        self.inner
            .write()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?
            .insert((m.origin_id, m.derivation_id.0.clone()), m);
        Ok(())
    }

    fn get(&self, origin_id: OriginId, derivation_id: &DerivationId) -> Result<DerivationManifest> {
        self.inner
            .read()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?
            .get(&(origin_id, derivation_id.0.clone()))
            .cloned()
            .ok_or(sodl_core::SodlError::NotFound)
    }

    fn list_for_origin(&self, origin_id: OriginId) -> Result<Vec<DerivationManifest>> {
        let r = self
            .inner
            .read()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        Ok(r.values()
            .filter(|m| m.origin_id == origin_id)
            .cloned()
            .collect())
    }
}

#[derive(Clone, Default)]
pub struct MemShareStore {
    inner: Arc<RwLock<HashMap<String, ShareRecord>>>,
}

impl MemShareStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ShareStore for MemShareStore {
    fn put(&self, s: ShareRecord) -> Result<()> {
        self.inner
            .write()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?
            .insert(s.share_id.0.clone(), s);
        Ok(())
    }

    fn get(&self, share_id: &ShareId) -> Result<ShareRecord> {
        self.inner
            .read()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?
            .get(&share_id.0)
            .cloned()
            .ok_or(sodl_core::SodlError::NotFound)
    }

    fn list_for_origin(&self, origin_id: OriginId) -> Result<Vec<ShareRecord>> {
        let r = self
            .inner
            .read()
            .map_err(|e| sodl_core::SodlError::Io(e.to_string()))?;
        Ok(r.values()
            .filter(|s| s.origin_id == origin_id)
            .cloned()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sodl_cas::MemBlobStore;
    use sodl_core::Durability;
    use sodl_crypto::NullCrypto;
    use sodl_policy::{AccessPolicy, RetentionPolicy};

    #[test]
    fn upload_share_derive_pin_flow() {
        let origin_registry = MemOriginRegistry::new();
        let policy_store = MemPolicyStore::new();
        let pin_store = MemPinStore::new();
        let deriv = MemDerivationStore::new();
        let shares = MemShareStore::new();

        let durable = MemBlobStore::new();
        let crypto = NullCrypto::default();
        let enc = EncryptedCas::new(&durable, &crypto, sodl_cas::HashAlg::Blake3);

        let index = MemIndex::new();
        let svc = SodlService {
            index: &index,
            lineage: &index,
            provenance: &index,
            origin_registry: &origin_registry,
            policy_store: &policy_store,
            pin_store: &pin_store,
            derivations: &deriv,
            shares: &shares,
            enc_cas: enc,
            crypto: &crypto,
            proof_signer: None,
            chunk_config: None,
        };

        let policy = OriginPolicy {
            origin_id: new_origin_id(), // overwritten by upload
            retention: RetentionPolicy {
                durability: Durability::BestEffort,
                ttl_seconds: None,
                min_replicas: Some(1),
            },
            access: AccessPolicy {
                default_caps: vec![Capability::Read],
                allow_reshare: true,
                allow_derivation: true,
            },
        };

        let up = svc
            .upload(UploadRequest {
                owner: PrincipalId("user:a".into()),
                media_kind: MediaKind::Binary,
                mime: Some("application/octet-stream".into()),
                durability_policy: policy,
                bytes: Bytes::from_static(b"hello"),
            })
            .unwrap();

        let _share_id = svc
            .share(
                PrincipalId("user:a".into()),
                PrincipalId("user:b".into()),
                up.origin_id,
                vec![Capability::Read, Capability::Reshare],
            )
            .unwrap();
        let _drv_id = svc
            .derive(
                up.origin_id,
                DerivationKind::Transform {
                    description: "noop".into(),
                },
                MediaKind::Binary,
            )
            .unwrap();
        let _pin_id = svc
            .pin_origin(PrincipalId("user:a".into()), up.origin_id, 1)
            .unwrap();

        // origin exists
        let rec = origin_registry.get_origin(up.origin_id).unwrap();
        assert_eq!(rec.representations[0].root_blobs[0].0, up.blob_id.0);
    }

    #[test]
    fn resolve_provenance_reports_exact_payload_and_chunk_overlap() {
        let origin_registry = MemOriginRegistry::new();
        let policy_store = MemPolicyStore::new();
        let pin_store = MemPinStore::new();
        let deriv = MemDerivationStore::new();
        let shares = MemShareStore::new();
        let durable = MemBlobStore::new();
        let crypto = NullCrypto::default();
        let enc = EncryptedCas::new(&durable, &crypto, sodl_cas::HashAlg::Blake3);
        let index = MemIndex::new();
        let chunk_config = ChunkPipelineConfig::default()
            .with_inline_threshold(100)
            .with_chunker(sodl_chunk::FixedSizeChunker::new(100));
        let svc = SodlService {
            index: &index,
            lineage: &index,
            provenance: &index,
            origin_registry: &origin_registry,
            policy_store: &policy_store,
            pin_store: &pin_store,
            derivations: &deriv,
            shares: &shares,
            enc_cas: enc,
            crypto: &crypto,
            proof_signer: None,
            chunk_config: Some(chunk_config),
        };
        let policy = OriginPolicy {
            origin_id: new_origin_id(),
            retention: RetentionPolicy {
                durability: Durability::BestEffort,
                ttl_seconds: None,
                min_replicas: Some(1),
            },
            access: AccessPolicy {
                default_caps: vec![Capability::Read],
                allow_reshare: true,
                allow_derivation: true,
            },
        };
        let original = Bytes::from(vec![1u8; 300]);
        let up = svc
            .upload(UploadRequest {
                owner: PrincipalId("user:a".into()),
                media_kind: MediaKind::Binary,
                mime: None,
                durability_policy: policy,
                bytes: original.clone(),
            })
            .unwrap();

        let exact = svc.resolve_provenance(&original).unwrap();
        assert_eq!(exact.candidates[0].origin_id, up.origin_id);
        assert_eq!(exact.candidates[0].confidence, 1.0);

        let partial = svc.resolve_provenance(&original[..200]).unwrap();
        assert_eq!(partial.candidates[0].origin_id, up.origin_id);
        assert!(matches!(
            partial.candidates[0].kind,
            ProvenanceMatchKind::ChunkOverlap { .. }
        ));
    }
}
