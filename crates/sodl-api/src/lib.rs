//! SODL HTTP API — an embeddable, cross-app REST server.
//!
//! This crate can be used in two ways:
//!
//! 1. **Standalone binary** (`sodl-server`) — reads config from environment
//!    variables and runs with default backends (SQLite + filesystem +
//!    NullCrypto).
//!
//! 2. **Embeddable library** — import the crate, construct [`AppState`] with
//!    your own backends via [`SodlServerBuilder`], and call [`build_router`]
//!    to get an axum [`Router`] you can compose into your own server.
//!
//! # Example — embed in another axum application
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use sodl_api::{SodlServerBuilder, build_router, config::Config};
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = Config::from_env();
//!     let state = SodlServerBuilder::defaults(&config)
//!         .unwrap()
//!         .build()
//!         .unwrap();
//!
//!     let sodl_routes = build_router(state);
//!
//!     // Nest SODL under /sodl in your own app:
//!     let app = axum::Router::new()
//!         .nest("/sodl", sodl_routes);
//!
//!     let listener = tokio::net::TcpListener::bind("0.0.0.0:8080")
//!         .await
//!         .unwrap();
//!     axum::serve(listener, app).await.unwrap();
//! }
//! ```
//!
//! # Pluggable backends
//!
//! SODL is designed as a **general-purpose, cross-app, cross-platform**
//! content-addressed storage system.  Every backend dependency is a trait
//! object, so you can bring your own:
//!
//! - **Blob store** — implement [`sodl_cas::BlobStore`] (filesystem, S3,
//!   Azure Blob, in-memory, …)
//! - **Crypto** — implement [`sodl_crypto::Crypto`] (NullCrypto for dev,
//!   AES-256-GCM, ChaCha20-Poly1305, …)
//! - **Metadata** — implement the seven metadata traits individually, or
//!   use a single type that implements all of them (like `SqliteStore`
//!   for SQLite or a future `PgStore` for PostgreSQL).
//! - **Proof signing** — implement [`sodl_proof::ProofSigner`] (Ed25519,
//!   HMAC, HSM, …)

pub mod config;
pub mod dto;
pub mod handlers;
pub mod router;
pub mod state;

use config::Config;

pub use state::{AppState, SodlServerBuilder};

/// Build an axum router with all SODL endpoints.
///
/// This is the main entry point for embedding SODL in another application.
/// Pass the result to `axum::serve` or nest it under a prefix in your own
/// router.
pub fn build_router(state: AppState) -> axum::Router {
    router::build_with_config(std::sync::Arc::new(state), &Config::from_env())
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use serde_json::Value;
    use tower::util::ServiceExt;

    use super::{build_router, AppState, SodlServerBuilder};

    fn test_state() -> AppState {
        let blobs = sodl_cas::MemBlobStore::new();
        let crypto = sodl_crypto::NullCrypto::default();
        let index = sodl_service::MemIndex::new();

        SodlServerBuilder::new()
            .blobs(blobs)
            .crypto(crypto)
            .origin_registry(sodl_service::MemOriginRegistry::new())
            .policy_store(sodl_service::MemPolicyStore::new())
            .pin_store(sodl_service::MemPinStore::new())
            .index(index.clone())
            .scan(index.clone())
            .lineage(index.clone())
            .provenance(index)
            .derivations(sodl_service::MemDerivationStore::new())
            .shares(sodl_service::MemShareStore::new())
            .build()
            .expect("test state")
    }

    #[tokio::test]
    async fn phase_d_routes_cover_blob_and_origin_contract() {
        let app = build_router(test_state());

        let create_blob = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/blobs")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from("blob-bytes"))
                    .expect("blob request"),
            )
            .await
            .expect("blob response");
        assert_eq!(create_blob.status(), StatusCode::CREATED);
        let blob_body = create_blob
            .into_body()
            .collect()
            .await
            .expect("blob body")
            .to_bytes();
        let blob_json: Value = serde_json::from_slice(&blob_body).expect("blob json");
        let blob_id = blob_json["blob_id"].as_str().expect("blob id").to_string();

        let create_origin = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/origins")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "owner": "user:test",
                            "media_kind": "binary",
                            "representations": [
                                {
                                    "name": "source",
                                    "media_kind": "binary",
                                    "root_blobs": [blob_id],
                                }
                            ],
                        })
                        .to_string(),
                    ))
                    .expect("origin request"),
            )
            .await
            .expect("origin response");
        assert_eq!(create_origin.status(), StatusCode::CREATED);
        let origin_body = create_origin
            .into_body()
            .collect()
            .await
            .expect("origin body")
            .to_bytes();
        let origin_json: Value = serde_json::from_slice(&origin_body).expect("origin json");
        let origin_id = origin_json["origin_id"]
            .as_str()
            .expect("origin id")
            .to_string();

        let list_origins = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/origins")
                    .body(Body::empty())
                    .expect("list origins request"),
            )
            .await
            .expect("list origins response");
        assert_eq!(list_origins.status(), StatusCode::OK);

        let list_representations = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(format!("/v1/origins/{origin_id}/representations"))
                    .body(Body::empty())
                    .expect("representations request"),
            )
            .await
            .expect("representations response");
        assert_eq!(list_representations.status(), StatusCode::OK);

        let v1_health = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/health")
                    .body(Body::empty())
                    .expect("health request"),
            )
            .await
            .expect("health response");
        assert_eq!(v1_health.status(), StatusCode::OK);
    }
}
