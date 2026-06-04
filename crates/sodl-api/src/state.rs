//! Shared application state for the SODL API server.
//!
//! `AppState` is **fully trait-object based** — every backend is a `dyn Trait`
//! behind `Arc`.  This makes the server pluggable: any application can bring
//! its own storage, crypto, or metadata backends without touching handler code.
//!
//! Use [`SodlServerBuilder`] for ergonomic construction.

use std::sync::Arc;

use sodl_cas::{BlobStore, HashAlg};
use sodl_chunk::ChunkPipelineConfig;
use sodl_crypto::Crypto;
use sodl_index::{LineageStore, ProvenanceIndex, RefCounter, ScanIndex};
use sodl_origin::OriginRegistry;
use sodl_policy::{PinStore, PolicyStore};
use sodl_proof::ProofSigner;
use sodl_service::{DerivationStore, ShareStore};
use sodl_store::EncryptedCas;

/// Application state shared across all request handlers via `Arc`.
///
/// Every backend dependency is a trait object — swap implementations by
/// providing your own `Arc<dyn Trait>` at construction time.
pub struct AppState {
    // -- Content --
    /// Content-addressed blob store (filesystem, S3, memory, …).
    pub blobs: Arc<dyn BlobStore>,
    /// Crypto provider (NullCrypto for dev, AEAD for production, …).
    pub crypto: Arc<dyn Crypto>,
    /// Hash algorithm used for blob IDs.
    pub hash_alg: HashAlg,

    // -- Metadata --
    /// Origin lifecycle (create, get, update, tombstone).
    pub origin_registry: Arc<dyn OriginRegistry>,
    /// Per-origin access & retention policies.
    pub policy_store: Arc<dyn PolicyStore>,
    /// Durability pins.
    pub pin_store: Arc<dyn PinStore>,
    /// Reference counters for origins and blobs.
    pub index: Arc<dyn RefCounter>,
    /// Scan index for listing tracked origins and blobs.
    pub scan: Arc<dyn ScanIndex>,
    /// Directed lineage edges (shares, derivations).
    pub lineage: Arc<dyn LineageStore>,
    /// Plaintext-side provenance fingerprints for exact and overlap matching.
    pub provenance: Arc<dyn ProvenanceIndex>,
    /// Derivation manifests.
    pub derivations: Arc<dyn DerivationStore>,
    /// Share records.
    pub shares: Arc<dyn ShareStore>,

    // -- Signing --
    /// Optional proof signer for lineage digests.
    pub proof_signer: Option<Arc<dyn ProofSigner>>,

    // -- Chunking --
    /// Chunking pipeline configuration. `Some` enables automatic content-defined
    /// chunking for large payloads. `None` disables chunking (single-blob mode).
    pub chunk_config: Option<ChunkPipelineConfig>,
}

impl AppState {
    /// Build a request-scoped [`sodl_service::SodlService`] from the owned
    /// trait-object stores.
    ///
    /// The returned service borrows from `self`, so it lives as long as the
    /// handler's `&AppState` reference — which is fine because axum clones
    /// the `Arc<AppState>` per-request.
    pub fn service(&self) -> sodl_service::SodlService<'_> {
        let enc_cas = EncryptedCas::new(self.blobs.as_ref(), self.crypto.as_ref(), self.hash_alg);

        sodl_service::SodlService {
            index: self.index.as_ref(),
            lineage: self.lineage.as_ref(),
            provenance: self.provenance.as_ref(),
            origin_registry: self.origin_registry.as_ref(),
            policy_store: self.policy_store.as_ref(),
            pin_store: self.pin_store.as_ref(),
            derivations: self.derivations.as_ref(),
            shares: self.shares.as_ref(),
            enc_cas,
            crypto: self.crypto.as_ref(),
            proof_signer: self.proof_signer.as_deref(),
            chunk_config: self.chunk_config.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Ergonomic builder for constructing an [`AppState`] with pluggable backends.
///
/// # Custom backends
///
/// ```rust,ignore
/// use sodl_api::SodlServerBuilder;
/// let state = SodlServerBuilder::new()
///     .blobs(my_s3_store)
///     .crypto(my_aead_provider)
///     .origin_registry(my_postgres_registry)
///     // ... other backends ...
///     .build()
///     .unwrap();
/// ```
///
/// # Default backends (SQLite + filesystem)
///
/// ```no_run
/// # use sodl_api::{SodlServerBuilder, config::Config};
/// let config = Config::from_env();
/// let state = SodlServerBuilder::defaults(&config)
///     .unwrap()
///     .build()
///     .unwrap();
/// ```
pub struct SodlServerBuilder {
    blobs: Option<Arc<dyn BlobStore>>,
    crypto: Option<Arc<dyn Crypto>>,
    origin_registry: Option<Arc<dyn OriginRegistry>>,
    policy_store: Option<Arc<dyn PolicyStore>>,
    pin_store: Option<Arc<dyn PinStore>>,
    index: Option<Arc<dyn RefCounter>>,
    scan: Option<Arc<dyn ScanIndex>>,
    lineage: Option<Arc<dyn LineageStore>>,
    provenance: Option<Arc<dyn ProvenanceIndex>>,
    derivations: Option<Arc<dyn DerivationStore>>,
    shares: Option<Arc<dyn ShareStore>>,
    proof_signer: Option<Arc<dyn ProofSigner>>,
    chunk_config: Option<ChunkPipelineConfig>,
    hash_alg: HashAlg,
}

impl SodlServerBuilder {
    /// Create an empty builder.
    ///
    /// You must set at least `blobs`, `crypto`, and all metadata stores
    /// before calling [`build`](Self::build).
    pub fn new() -> Self {
        Self {
            blobs: None,
            crypto: None,
            origin_registry: None,
            policy_store: None,
            pin_store: None,
            index: None,
            scan: None,
            lineage: None,
            provenance: None,
            derivations: None,
            shares: None,
            proof_signer: None,
            chunk_config: None,
            hash_alg: HashAlg::Blake3,
        }
    }

    /// Pre-fill the builder with the **default** backends:
    /// filesystem blobs, SQLite metadata. Crypto depends on config:
    /// - `SODL_MASTER_KEY` set → production `AeadCrypto`
    /// - unset → `NullCrypto` (development only, no encryption)
    ///
    /// You can still override individual backends after calling this.
    pub fn defaults(config: &crate::config::Config) -> sodl_core::Result<Self> {
        use sodl_cas::FsBlobStore;
        use sodl_persist::SqliteStore;

        if let Some(parent) = config.db_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| sodl_core::SodlError::Io(format!("mkdir db parent: {e}")))?;
        }

        let db = Arc::new(SqliteStore::open(&config.db_path)?);
        let blobs: Arc<dyn BlobStore> = Arc::new(FsBlobStore::open(&config.blob_dir)?);

        let crypto: Arc<dyn Crypto> = match &config.encryption {
            crate::config::EncryptionMode::Aead { master_key_hex } => {
                let aead = sodl_crypto::AeadCrypto::from_hex(master_key_hex)?;
                tracing::info!("encryption: XChaCha20-Poly1305 (AEAD) enabled");
                Arc::new(aead)
            }
            crate::config::EncryptionMode::None => {
                tracing::warn!(
                    "encryption: DISABLED (NullCrypto) — set SODL_MASTER_KEY for production"
                );
                Arc::new(sodl_crypto::NullCrypto::default())
            }
        };

        Ok(Self::new()
            .blobs_arc(blobs)
            .crypto_arc(crypto)
            .enable_chunking(ChunkPipelineConfig::default())
            .metadata(db))
    }

    // -- Content backends --

    /// Set the blob store from a concrete type.
    pub fn blobs(mut self, b: impl BlobStore + 'static) -> Self {
        self.blobs = Some(Arc::new(b));
        self
    }

    /// Set the blob store from a pre-wrapped `Arc`.
    pub fn blobs_arc(mut self, b: Arc<dyn BlobStore>) -> Self {
        self.blobs = Some(b);
        self
    }

    /// Set the crypto provider from a concrete type.
    pub fn crypto(mut self, c: impl Crypto + 'static) -> Self {
        self.crypto = Some(Arc::new(c));
        self
    }

    /// Set the crypto provider from a pre-wrapped `Arc`.
    pub fn crypto_arc(mut self, c: Arc<dyn Crypto>) -> Self {
        self.crypto = Some(c);
        self
    }

    /// Set the hash algorithm (default: Blake3).
    pub fn hash_alg(mut self, alg: HashAlg) -> Self {
        self.hash_alg = alg;
        self
    }

    // -- Metadata backends (individual) --

    pub fn origin_registry(mut self, r: impl OriginRegistry + 'static) -> Self {
        self.origin_registry = Some(Arc::new(r));
        self
    }

    pub fn policy_store(mut self, p: impl PolicyStore + 'static) -> Self {
        self.policy_store = Some(Arc::new(p));
        self
    }

    pub fn pin_store(mut self, p: impl PinStore + 'static) -> Self {
        self.pin_store = Some(Arc::new(p));
        self
    }

    pub fn index(mut self, i: impl RefCounter + 'static) -> Self {
        self.index = Some(Arc::new(i));
        self
    }

    pub fn scan(mut self, s: impl ScanIndex + 'static) -> Self {
        self.scan = Some(Arc::new(s));
        self
    }

    pub fn lineage(mut self, l: impl LineageStore + 'static) -> Self {
        self.lineage = Some(Arc::new(l));
        self
    }

    pub fn provenance(mut self, p: impl ProvenanceIndex + 'static) -> Self {
        self.provenance = Some(Arc::new(p));
        self
    }

    pub fn derivations(mut self, d: impl DerivationStore + 'static) -> Self {
        self.derivations = Some(Arc::new(d));
        self
    }

    pub fn shares(mut self, s: impl ShareStore + 'static) -> Self {
        self.shares = Some(Arc::new(s));
        self
    }

    // -- Signing --

    pub fn proof_signer(mut self, s: impl ProofSigner + 'static) -> Self {
        self.proof_signer = Some(Arc::new(s));
        self
    }

    // -- Chunking --

    /// Enable automatic content-defined chunking for large payloads.
    pub fn enable_chunking(mut self, config: ChunkPipelineConfig) -> Self {
        self.chunk_config = Some(config);
        self
    }

    /// Disable chunking (single-blob mode).
    pub fn disable_chunking(mut self) -> Self {
        self.chunk_config = None;
        self
    }

    // -- Bulk setters --

    /// Set **all seven** metadata stores from a single type that implements
    /// every metadata trait.  This is the common case when you have a unified
    /// store like `SqliteStore`.
    pub fn metadata<T>(mut self, store: Arc<T>) -> Self
    where
        T: OriginRegistry
            + PolicyStore
            + PinStore
            + RefCounter
            + ScanIndex
            + LineageStore
            + ProvenanceIndex
            + DerivationStore
            + ShareStore
            + 'static,
    {
        self.origin_registry = Some(store.clone() as Arc<dyn OriginRegistry>);
        self.policy_store = Some(store.clone() as Arc<dyn PolicyStore>);
        self.pin_store = Some(store.clone() as Arc<dyn PinStore>);
        self.index = Some(store.clone() as Arc<dyn RefCounter>);
        self.scan = Some(store.clone() as Arc<dyn ScanIndex>);
        self.lineage = Some(store.clone() as Arc<dyn LineageStore>);
        self.provenance = Some(store.clone() as Arc<dyn ProvenanceIndex>);
        self.derivations = Some(store.clone() as Arc<dyn DerivationStore>);
        self.shares = Some(store as Arc<dyn ShareStore>);
        self
    }

    /// Consume the builder and produce a validated [`AppState`].
    ///
    /// Returns `Err` if any required backend has not been set.
    pub fn build(self) -> sodl_core::Result<AppState> {
        let missing =
            |name: &str| sodl_core::SodlError::Invalid(format!("missing backend: {name}"));

        Ok(AppState {
            blobs: self.blobs.ok_or_else(|| missing("blobs"))?,
            crypto: self.crypto.ok_or_else(|| missing("crypto"))?,
            hash_alg: self.hash_alg,
            origin_registry: self
                .origin_registry
                .ok_or_else(|| missing("origin_registry"))?,
            policy_store: self.policy_store.ok_or_else(|| missing("policy_store"))?,
            pin_store: self.pin_store.ok_or_else(|| missing("pin_store"))?,
            index: self.index.ok_or_else(|| missing("index"))?,
            scan: self.scan.ok_or_else(|| missing("scan"))?,
            lineage: self.lineage.ok_or_else(|| missing("lineage"))?,
            provenance: self.provenance.ok_or_else(|| missing("provenance"))?,
            derivations: self.derivations.ok_or_else(|| missing("derivations"))?,
            shares: self.shares.ok_or_else(|| missing("shares"))?,
            proof_signer: self.proof_signer,
            chunk_config: self.chunk_config,
        })
    }
}

impl Default for SodlServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}
