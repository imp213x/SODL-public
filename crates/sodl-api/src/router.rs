//! Axum router definition — maps HTTP routes to handlers.

use std::sync::Arc;

use axum::routing::{delete, get, post};
use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::handlers;
use crate::state::AppState;

/// Build the full application router.
pub fn build(state: Arc<AppState>) -> Router {
    Router::new()
        // Health
        .route("/health", get(handlers::health))
        .route("/v1/health", get(handlers::health))
        // Upload (multipart)
        .route("/v1/upload", post(handlers::upload))
        // Provenance
        .route("/v1/provenance/resolve", post(handlers::resolve_provenance))
        // Origins
        .route("/v1/origins", get(handlers::list_origins))
        .route("/v1/origins", post(handlers::create_origin))
        .route("/v1/origins/{id}", get(handlers::get_origin))
        .route("/v1/origins/{id}", delete(handlers::tombstone_origin))
        .route(
            "/v1/origins/{id}/representations",
            get(handlers::list_origin_representations),
        )
        .route("/v1/origins/{id}/payload", get(handlers::get_payload))
        .route(
            "/v1/origins/{id}/lineage-proof",
            get(handlers::lineage_proof),
        )
        // Blobs
        .route("/v1/blobs", post(handlers::create_blob))
        .route("/v1/blobs/{id}", get(handlers::get_blob))
        .route("/v1/blobs/{id}", delete(handlers::delete_blob))
        // Shares
        .route("/v1/shares", post(handlers::create_share))
        .route("/v1/shares/{id}", get(handlers::get_share))
        .route("/v1/shares/{id}", delete(handlers::release_share))
        .route("/v1/shares/{id}/verify", post(handlers::verify_share))
        // Derivations
        .route("/v1/derivations", post(handlers::create_derivation))
        // Pins
        .route("/v1/pins", post(handlers::create_pin))
        .route("/v1/pins/{id}", delete(handlers::release_pin))
        // Middleware
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        // State
        .with_state(state)
}
