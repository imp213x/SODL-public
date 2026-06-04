//! Content Addressed Storage (CAS) primitives for SODL.
//!
//! V1 skeleton provides hashing + core traits. Storage backends come later.
//!
//! Key concept:
//! - The **BlobId** is computed over the bytes you are storing.
//! - If you store ciphertext, you must compute the hash over ciphertext (so integrity checks work).
//! - To preserve "store once" across shares of the same origin, encryption must be *stable* within origin.
//!   (Handled by sodl-crypto interfaces; not implemented here.)

use bytes::Bytes;
use sodl_core::{BlobId, Result, SodlError};

/// Hash algorithm identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlg {
    Blake3,
    Sha256,
}

/// Compute a blob id (content hash) for bytes.
pub fn compute_blob_id(bytes: &[u8], alg: HashAlg) -> BlobId {
    match alg {
        HashAlg::Blake3 => BlobId(format!("blake3:{}", blake3::hash(bytes).to_hex())),
        HashAlg::Sha256 => {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(bytes);
            BlobId(format!("sha256:{:x}", hasher.finalize()))
        }
    }
}

/// Core interface for a blob store.
///
/// Implementations may be local disk, object store, embedded DB, etc.
pub trait BlobStore: Send + Sync {
    fn has(&self, id: &BlobId) -> Result<bool>;
    fn put(&self, id: &BlobId, data: Bytes) -> Result<()>;
    fn get(&self, id: &BlobId) -> Result<Bytes>;
    fn delete(&self, id: &BlobId) -> Result<()>;
}

/// Verify that fetched bytes match the provided BlobId.
pub fn verify_integrity(id: &BlobId, data: &[u8]) -> Result<()> {
    let (alg, expected) =
        id.0.split_once(':')
            .ok_or_else(|| SodlError::Invalid("BlobId missing prefix".into()))?;

    let actual = match alg {
        "blake3" => blake3::hash(data).to_hex().to_string(),
        "sha256" => {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(data);
            format!("{:x}", hasher.finalize())
        }
        _ => return Err(SodlError::Invalid(format!("unknown hash alg: {alg}"))),
    };

    if actual == expected {
        Ok(())
    } else {
        Err(SodlError::Integrity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blake3_roundtrip_integrity_ok() {
        let data = b"hello sodl";
        let id = compute_blob_id(data, HashAlg::Blake3);
        verify_integrity(&id, data).unwrap();
    }

    #[test]
    fn sha256_roundtrip_integrity_ok() {
        let data = b"hello sodl";
        let id = compute_blob_id(data, HashAlg::Sha256);
        verify_integrity(&id, data).unwrap();
    }

    #[test]
    fn integrity_fails_on_modified_data() {
        let data = b"hello sodl";
        let id = compute_blob_id(data, HashAlg::Blake3);
        let modified = b"hello sodL";
        let err = verify_integrity(&id, modified).unwrap_err();
        match err {
            SodlError::Integrity => {}
            other => panic!("expected Integrity error, got {other:?}"),
        }
    }
}

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Simple in-memory blob store (useful for tests and examples).
#[derive(Clone, Default)]
pub struct MemBlobStore {
    inner: Arc<RwLock<HashMap<String, Bytes>>>,
}

impl MemBlobStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl BlobStore for MemBlobStore {
    fn has(&self, id: &BlobId) -> Result<bool> {
        Ok(self
            .inner
            .read()
            .map_err(|e| SodlError::Io(e.to_string()))?
            .contains_key(&id.0))
    }

    fn put(&self, id: &BlobId, data: Bytes) -> Result<()> {
        self.inner
            .write()
            .map_err(|e| SodlError::Io(e.to_string()))?
            .insert(id.0.clone(), data);
        Ok(())
    }

    fn get(&self, id: &BlobId) -> Result<Bytes> {
        self.inner
            .read()
            .map_err(|e| SodlError::Io(e.to_string()))?
            .get(&id.0)
            .cloned()
            .ok_or(SodlError::NotFound)
    }

    fn delete(&self, id: &BlobId) -> Result<()> {
        self.inner
            .write()
            .map_err(|e| SodlError::Io(e.to_string()))?
            .remove(&id.0);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Filesystem blob store — content-addressed directory layout
// ---------------------------------------------------------------------------

use std::path::{Path, PathBuf};

/// Filesystem-backed blob store.
///
/// Storage layout:
/// ```text
/// <root>/
///   <alg>/          e.g. "blake3" or "sha256"
///     <prefix>/     first 2 hex chars (fan-out for FS friendliness)
///       <hex>       remaining hex chars → file containing raw bytes
/// ```
///
/// This is the same sharding strategy used by Git's object store.  The two-char
/// fan-out prevents any single directory from accumulating millions of entries.
///
/// # Features
///
/// Available unconditionally (no feature gate) because filesystem storage is a
/// fundamental capability.  Can be compiled on any target with `std::fs`.
#[derive(Debug, Clone)]
pub struct FsBlobStore {
    root: PathBuf,
}

impl FsBlobStore {
    /// Create a new store rooted at `root`.  The directory is created if it does
    /// not exist.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root)
            .map_err(|e| SodlError::Io(format!("create blob root {}: {e}", root.display())))?;
        Ok(Self { root })
    }

    /// Resolve a BlobId to its file path.
    ///
    /// Returns `(dir, file_path)` where `dir` is the parent directory that may
    /// need to be created on put.
    fn blob_path(&self, id: &BlobId) -> Result<(PathBuf, PathBuf)> {
        let (alg, hex) =
            id.0.split_once(':')
                .ok_or_else(|| SodlError::Invalid("BlobId missing prefix".into()))?;
        if hex.len() < 3 {
            return Err(SodlError::Invalid("BlobId hex too short".into()));
        }
        let prefix = &hex[..2];
        let rest = &hex[2..];
        let dir = self.root.join(alg).join(prefix);
        let path = dir.join(rest);
        Ok((dir, path))
    }
}

impl BlobStore for FsBlobStore {
    fn has(&self, id: &BlobId) -> Result<bool> {
        let (_, path) = self.blob_path(id)?;
        Ok(path.exists())
    }

    fn put(&self, id: &BlobId, data: Bytes) -> Result<()> {
        let (dir, path) = self.blob_path(id)?;
        // Idempotent: if the blob already exists with the right content, skip.
        if path.exists() {
            return Ok(());
        }
        std::fs::create_dir_all(&dir)
            .map_err(|e| SodlError::Io(format!("mkdir {}: {e}", dir.display())))?;

        // Write to a temp file then rename for atomic put (crash-safe).
        let tmp_path = path.with_extension("tmp");
        std::fs::write(&tmp_path, &data)
            .map_err(|e| SodlError::Io(format!("write {}: {e}", tmp_path.display())))?;
        std::fs::rename(&tmp_path, &path)
            .map_err(|e| SodlError::Io(format!("rename {}: {e}", path.display())))?;
        Ok(())
    }

    fn get(&self, id: &BlobId) -> Result<Bytes> {
        let (_, path) = self.blob_path(id)?;
        let data = std::fs::read(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                SodlError::NotFound
            } else {
                SodlError::Io(format!("read {}: {e}", path.display()))
            }
        })?;
        Ok(Bytes::from(data))
    }

    fn delete(&self, id: &BlobId) -> Result<()> {
        let (_, path) = self.blob_path(id)?;
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(SodlError::Io(format!("delete {}: {e}", path.display()))),
        }
    }
}

#[cfg(test)]
mod mem_tests {
    use super::*;

    #[test]
    fn mem_blob_store_roundtrip() {
        let store = MemBlobStore::new();
        let data = Bytes::from_static(b"abc");
        let id = compute_blob_id(&data, HashAlg::Blake3);

        assert!(!store.has(&id).unwrap());
        store.put(&id, data.clone()).unwrap();
        assert!(store.has(&id).unwrap());

        let got = store.get(&id).unwrap();
        assert_eq!(got, data);

        store.delete(&id).unwrap();
        assert!(!store.has(&id).unwrap());
    }
}

#[cfg(test)]
mod fs_tests {
    use super::*;

    #[test]
    fn fs_blob_store_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FsBlobStore::open(tmp.path().join("blobs")).unwrap();
        let data = Bytes::from_static(b"hello sodl persistent");
        let id = compute_blob_id(&data, HashAlg::Blake3);

        assert!(!store.has(&id).unwrap());
        store.put(&id, data.clone()).unwrap();
        assert!(store.has(&id).unwrap());

        let got = store.get(&id).unwrap();
        assert_eq!(got, data);
        verify_integrity(&id, &got).unwrap();

        store.delete(&id).unwrap();
        assert!(!store.has(&id).unwrap());
    }

    #[test]
    fn fs_blob_store_idempotent_put() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FsBlobStore::open(tmp.path().join("blobs")).unwrap();
        let data = Bytes::from_static(b"dedupe me");
        let id = compute_blob_id(&data, HashAlg::Blake3);

        store.put(&id, data.clone()).unwrap();
        store.put(&id, data.clone()).unwrap(); // second put is a no-op
        let got = store.get(&id).unwrap();
        assert_eq!(got, data);
    }

    #[test]
    fn fs_blob_store_sha256() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FsBlobStore::open(tmp.path().join("blobs")).unwrap();
        let data = Bytes::from_static(b"sha256 test");
        let id = compute_blob_id(&data, HashAlg::Sha256);

        store.put(&id, data.clone()).unwrap();
        let got = store.get(&id).unwrap();
        assert_eq!(got, data);
        verify_integrity(&id, &got).unwrap();
    }

    #[test]
    fn fs_blob_store_delete_nonexistent_ok() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FsBlobStore::open(tmp.path().join("blobs")).unwrap();
        let id = BlobId("blake3:aaabbbcccddd".into());
        // Deleting something that doesn't exist should not error
        store.delete(&id).unwrap();
    }

    #[test]
    fn fs_blob_store_get_nonexistent_returns_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FsBlobStore::open(tmp.path().join("blobs")).unwrap();
        let id = BlobId("blake3:aaabbbcccddd".into());
        match store.get(&id) {
            Err(SodlError::NotFound) => {}
            other => panic!("expected NotFound, got {other:?}"),
        }
    }
}
