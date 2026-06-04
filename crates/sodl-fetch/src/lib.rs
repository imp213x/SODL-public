//! Retrieval pipeline orchestration.
//!
//! Order (policy-driven):
//! local cache -> peers (optional) -> edge (optional) -> durable origin store
//!
//! This crate is orchestration-only; it does not implement concrete transports.

use bytes::Bytes;
use sodl_cas::{verify_integrity, BlobStore};
use sodl_core::{BlobId, Capability, OriginId, PrincipalId, Result};
use sodl_policy::Authorizer;
use tracing::instrument;

/// A source that can fetch blobs (peer, edge, origin).
pub trait FetchSource: Send + Sync {
    fn fetch(&self, id: &BlobId) -> Result<Option<Bytes>>;
}

/// FetchSource adapter for any BlobStore (useful for durable stores).
pub struct StoreSource<'a>(pub &'a dyn BlobStore);
impl<'a> FetchSource for StoreSource<'a> {
    fn fetch(&self, id: &BlobId) -> Result<Option<Bytes>> {
        if self.0.has(id)? {
            Ok(Some(self.0.get(id)?))
        } else {
            Ok(None)
        }
    }
}

pub struct FetchPipeline<'a> {
    /// Local cache (fast, non-durable)
    pub cache: &'a dyn BlobStore,
    /// Ordered fallback sources
    pub sources: Vec<&'a dyn FetchSource>,
    /// Optional authorization hook
    pub authorizer: Option<&'a dyn Authorizer>,
}

impl<'a> FetchPipeline<'a> {
    /// Fetches a blob by id and validates integrity.
    /// If an authorizer is provided, it checks `Read` for the principal.
    #[instrument(skip_all, fields(blob_id = %id.0))]
    pub fn get_for(
        &self,
        principal: Option<&PrincipalId>,
        origin_id: Option<OriginId>,
        id: &BlobId,
    ) -> Result<Bytes> {
        if let (Some(authz), Some(p), Some(oid)) = (self.authorizer, principal, origin_id) {
            authz.check(oid, p, Capability::Read)?;
        }

        if self.cache.has(id)? {
            let b = self.cache.get(id)?;
            verify_integrity(id, &b)?;
            return Ok(b);
        }

        for src in &self.sources {
            if let Some(b) = src.fetch(id)? {
                verify_integrity(id, &b)?;
                self.cache.put(id, b.clone())?;
                return Ok(b);
            }
        }

        Err(sodl_core::SodlError::NotFound)
    }
}
