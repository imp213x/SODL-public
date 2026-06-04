//! Configuration for the SODL API server.
//!
//! All settings are read from environment variables with sensible defaults.
//! This keeps the server stateless and 12-factor compliant.

use std::net::SocketAddr;
use std::path::PathBuf;

/// Server configuration — all fields populated from environment.
#[derive(Debug, Clone)]
pub struct Config {
    /// Listen address (default `127.0.0.1:7700`).
    pub listen: SocketAddr,
    /// Filesystem root for blob storage (default `./sodl_data/blobs`).
    pub blob_dir: PathBuf,
    /// SQLite database path (default `./sodl_data/sodl.db`).
    pub db_path: PathBuf,
    /// Encryption mode.
    pub encryption: EncryptionMode,
}

/// How the server encrypts content at rest.
#[derive(Debug, Clone)]
pub enum EncryptionMode {
    /// No encryption (development only).
    None,
    /// Production AES/ChaCha20 encryption with a hex-encoded 32-byte master key.
    Aead {
        /// 64-character hex string representing a 32-byte master key.
        master_key_hex: String,
    },
}

impl Config {
    /// Load configuration from environment variables.
    ///
    /// | Variable              | Default                  | Description                           |
    /// |-----------------------|--------------------------|---------------------------------------|
    /// | `SODL_LISTEN`         | `127.0.0.1:7700`         | Listen address                        |
    /// | `SODL_BLOB_DIR`       | `./sodl_data/blobs`      | Blob storage root                     |
    /// | `SODL_DB_PATH`        | `./sodl_data/sodl.db`    | SQLite path                           |
    /// | `SODL_MASTER_KEY`     | *(unset = NullCrypto)*   | 64-hex-char master key for AEAD       |
    pub fn from_env() -> Self {
        let listen: SocketAddr = std::env::var("SODL_LISTEN")
            .unwrap_or_else(|_| "127.0.0.1:7700".into())
            .parse()
            .expect("SODL_LISTEN must be a valid socket address");

        let blob_dir = PathBuf::from(
            std::env::var("SODL_BLOB_DIR").unwrap_or_else(|_| "./sodl_data/blobs".into()),
        );

        let db_path = PathBuf::from(
            std::env::var("SODL_DB_PATH").unwrap_or_else(|_| "./sodl_data/sodl.db".into()),
        );

        let encryption = match std::env::var("SODL_MASTER_KEY") {
            Ok(hex) if !hex.is_empty() => EncryptionMode::Aead {
                master_key_hex: hex,
            },
            _ => EncryptionMode::None,
        };

        Self {
            listen,
            blob_dir,
            db_path,
            encryption,
        }
    }
}
