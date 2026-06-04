//! Distribution/peer layer.
//!
//! Provides concrete peer discovery and HTTP blob-fetch transport on top of the
//! trait boundaries used by higher SODL layers.

use bytes::Bytes;
use sodl_core::{BlobId, Result, SodlError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

/// Peer address (transport-specific).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PeerAddr(pub String);

/// Finds which peers can provide a blob.
pub trait PeerDiscovery: Send + Sync {
    fn find_providers(&self, id: &BlobId) -> Result<Vec<PeerAddr>>;
}

/// Peer client boundary (how to fetch from a peer).
pub trait PeerClient: Send + Sync {
    fn get_blob(&self, peer: &PeerAddr, id: &BlobId) -> Result<Option<Bytes>>;
    fn announce_blob(&self, id: &BlobId) -> Result<()>;
}

/// Edge provider boundary (CDN-like cache).
pub trait EdgeProvider: Send + Sync {
    fn get_blob(&self, id: &BlobId) -> Result<Option<Bytes>>;
}

/// Static peer discovery with optional blob-specific provider overrides.
#[derive(Clone, Default)]
pub struct StaticPeerDiscovery {
    fallback_peers: Arc<RwLock<Vec<PeerAddr>>>,
    providers: Arc<RwLock<HashMap<String, Vec<PeerAddr>>>>,
}

impl StaticPeerDiscovery {
    pub fn new(peers: Vec<PeerAddr>) -> Self {
        Self {
            fallback_peers: Arc::new(RwLock::new(peers)),
            providers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn register_provider(&self, blob_id: &BlobId, peer: PeerAddr) -> Result<()> {
        let mut providers = self
            .providers
            .write()
            .map_err(|err| SodlError::Io(err.to_string()))?;
        providers.entry(blob_id.0.clone()).or_default().push(peer);
        Ok(())
    }
}

impl PeerDiscovery for StaticPeerDiscovery {
    fn find_providers(&self, id: &BlobId) -> Result<Vec<PeerAddr>> {
        let providers = self
            .providers
            .read()
            .map_err(|err| SodlError::Io(err.to_string()))?;
        if let Some(values) = providers.get(&id.0) {
            return Ok(values.clone());
        }
        drop(providers);
        let fallback = self
            .fallback_peers
            .read()
            .map_err(|err| SodlError::Io(err.to_string()))?;
        Ok(fallback.clone())
    }
}

/// Blocking HTTP peer client against SODL API `/v1/blobs/:id`.
#[derive(Clone)]
pub struct HttpPeerClient {
    client: reqwest::blocking::Client,
}

impl HttpPeerClient {
    pub fn new(timeout_seconds: u64) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_seconds))
            .build()
            .map_err(|err| SodlError::Io(err.to_string()))?;
        Ok(Self { client })
    }

    fn blob_url(&self, peer: &PeerAddr, id: &BlobId) -> String {
        format!("{}/v1/blobs/{}", peer.0.trim_end_matches('/'), id.0)
    }
}

impl PeerClient for HttpPeerClient {
    fn get_blob(&self, peer: &PeerAddr, id: &BlobId) -> Result<Option<Bytes>> {
        let response = self
            .client
            .get(self.blob_url(peer, id))
            .send()
            .map_err(|err| SodlError::Io(err.to_string()))?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        let response = response
            .error_for_status()
            .map_err(|err| SodlError::Io(err.to_string()))?;
        let body = response
            .bytes()
            .map_err(|err| SodlError::Io(err.to_string()))?;
        Ok(Some(body))
    }

    fn announce_blob(&self, _id: &BlobId) -> Result<()> {
        // V1 transport bridge: announcement is best-effort/no-op until the API grows
        // a peer advertisement endpoint.
        Ok(())
    }
}

/// Fetch source that resolves providers via discovery and downloads from the
/// first peer that serves the blob.
pub struct DiscoveredPeerSource<'a> {
    discovery: &'a dyn PeerDiscovery,
    client: &'a dyn PeerClient,
    last_provider: Arc<Mutex<Option<PeerAddr>>>,
}

impl<'a> DiscoveredPeerSource<'a> {
    pub fn new(discovery: &'a dyn PeerDiscovery, client: &'a dyn PeerClient) -> Self {
        Self {
            discovery,
            client,
            last_provider: Arc::new(Mutex::new(None)),
        }
    }

    pub fn clear_last_provider(&self) {
        if let Ok(mut guard) = self.last_provider.lock() {
            *guard = None;
        }
    }

    pub fn last_provider(&self) -> Option<PeerAddr> {
        self.last_provider
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
    }
}

impl<'a> sodl_fetch::FetchSource for DiscoveredPeerSource<'a> {
    fn fetch(&self, id: &BlobId) -> Result<Option<Bytes>> {
        self.clear_last_provider();
        for peer in self.discovery.find_providers(id)? {
            if let Some(bytes) = self.client.get_blob(&peer, id)? {
                if let Ok(mut guard) = self.last_provider.lock() {
                    *guard = Some(peer);
                }
                return Ok(Some(bytes));
            }
        }
        Ok(None)
    }
}

/// Multi-endpoint HTTP edge provider against SODL API blob endpoints.
#[derive(Clone)]
pub struct HttpEdgeProvider {
    client: reqwest::blocking::Client,
    base_urls: Vec<String>,
    last_provider: Arc<Mutex<Option<String>>>,
}

impl HttpEdgeProvider {
    pub fn new(base_urls: Vec<String>, timeout_seconds: u64) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_seconds))
            .build()
            .map_err(|err| SodlError::Io(err.to_string()))?;
        Ok(Self {
            client,
            base_urls,
            last_provider: Arc::new(Mutex::new(None)),
        })
    }

    pub fn clear_last_provider(&self) {
        if let Ok(mut guard) = self.last_provider.lock() {
            *guard = None;
        }
    }

    pub fn last_provider(&self) -> Option<String> {
        self.last_provider
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
    }

    fn blob_url(&self, base_url: &str, id: &BlobId) -> String {
        format!("{}/v1/blobs/{}", base_url.trim_end_matches('/'), id.0)
    }
}

impl EdgeProvider for HttpEdgeProvider {
    fn get_blob(&self, id: &BlobId) -> Result<Option<Bytes>> {
        self.clear_last_provider();
        for base_url in &self.base_urls {
            let response = self
                .client
                .get(self.blob_url(base_url, id))
                .send()
                .map_err(|err| SodlError::Io(err.to_string()))?;
            if response.status() == reqwest::StatusCode::NOT_FOUND {
                continue;
            }
            let response = response
                .error_for_status()
                .map_err(|err| SodlError::Io(err.to_string()))?;
            let body = response
                .bytes()
                .map_err(|err| SodlError::Io(err.to_string()))?;
            if let Ok(mut guard) = self.last_provider.lock() {
                *guard = Some(base_url.clone());
            }
            return Ok(Some(body));
        }
        Ok(None)
    }
}

/// Adapter so an edge provider can participate in `sodl-fetch` pipelines.
pub struct EdgeFetchSource<'a> {
    provider: &'a dyn EdgeProvider,
}

impl<'a> EdgeFetchSource<'a> {
    pub fn new(provider: &'a dyn EdgeProvider) -> Self {
        Self { provider }
    }
}

impl<'a> sodl_fetch::FetchSource for EdgeFetchSource<'a> {
    fn fetch(&self, id: &BlobId) -> Result<Option<Bytes>> {
        self.provider.get_blob(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sodl_fetch::FetchSource;

    #[derive(Default)]
    struct FakePeerClient {
        blobs: HashMap<(String, String), Bytes>,
        announced: Arc<Mutex<Vec<String>>>,
    }

    impl FakePeerClient {
        fn with_blob(mut self, peer: &str, blob_id: &str, data: &[u8]) -> Self {
            self.blobs.insert(
                (peer.to_string(), blob_id.to_string()),
                Bytes::copy_from_slice(data),
            );
            self
        }
    }

    impl PeerClient for FakePeerClient {
        fn get_blob(&self, peer: &PeerAddr, id: &BlobId) -> Result<Option<Bytes>> {
            Ok(self.blobs.get(&(peer.0.clone(), id.0.clone())).cloned())
        }

        fn announce_blob(&self, id: &BlobId) -> Result<()> {
            self.announced.lock().expect("poison").push(id.0.clone());
            Ok(())
        }
    }

    #[derive(Default)]
    struct FakeEdgeProvider {
        blobs: HashMap<String, Bytes>,
    }

    impl EdgeProvider for FakeEdgeProvider {
        fn get_blob(&self, id: &BlobId) -> Result<Option<Bytes>> {
            Ok(self.blobs.get(&id.0).cloned())
        }
    }

    #[test]
    fn static_peer_discovery_returns_fallback_and_specific_providers() {
        let discovery = StaticPeerDiscovery::new(vec![PeerAddr("http://peer-a".into())]);
        let blob_id = BlobId("blake3:abc".into());
        assert_eq!(
            discovery.find_providers(&blob_id).unwrap(),
            vec![PeerAddr("http://peer-a".into())]
        );

        discovery
            .register_provider(&blob_id, PeerAddr("http://peer-b".into()))
            .unwrap();
        assert_eq!(
            discovery.find_providers(&blob_id).unwrap(),
            vec![PeerAddr("http://peer-b".into())]
        );
    }

    #[test]
    fn discovered_peer_source_fetches_from_first_available_peer() {
        let blob_id = BlobId("blake3:peer".into());
        let discovery = StaticPeerDiscovery::new(vec![
            PeerAddr("http://peer-a".into()),
            PeerAddr("http://peer-b".into()),
        ]);
        let client = FakePeerClient::default().with_blob("http://peer-b", &blob_id.0, b"peer-data");
        let source = DiscoveredPeerSource::new(&discovery, &client);

        let fetched = source.fetch(&blob_id).unwrap().expect("peer data");
        assert_eq!(fetched, Bytes::from_static(b"peer-data"));
        assert_eq!(
            source.last_provider(),
            Some(PeerAddr("http://peer-b".into()))
        );
    }

    #[test]
    fn edge_fetch_source_uses_edge_provider() {
        let blob_id = BlobId("blake3:edge".into());
        let mut provider = FakeEdgeProvider::default();
        provider
            .blobs
            .insert(blob_id.0.clone(), Bytes::from_static(b"edge-data"));
        let source = EdgeFetchSource::new(&provider);

        let fetched = source.fetch(&blob_id).unwrap().expect("edge data");
        assert_eq!(fetched, Bytes::from_static(b"edge-data"));
    }
}
