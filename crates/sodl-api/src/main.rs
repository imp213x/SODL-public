//! SODL HTTP API server — `sodl-server` binary.
//!
//! A cross-app, cross-platform REST sidecar that exposes the full SODL
//! service over HTTP.  Any application can talk to SODL via this API.
//!
//! # Quick start
//!
//! ```bash
//! # Defaults: listens on 127.0.0.1:7700, stores data in ./sodl_data/
//! cargo run -p sodl-api
//!
//! # Custom config via environment
//! SODL_LISTEN=0.0.0.0:8080 SODL_BLOB_DIR=/data/blobs SODL_DB_PATH=/data/sodl.db cargo run -p sodl-api
//! ```

use std::sync::Arc;

use sodl_api::config::Config;
use sodl_api::SodlServerBuilder;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    // Initialise tracing (respects RUST_LOG env, defaults to info).
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config = Config::from_env();
    tracing::info!(
        listen = %config.listen,
        blob_dir = %config.blob_dir.display(),
        db_path = %config.db_path.display(),
        "starting SODL server"
    );

    let state = Arc::new(
        SodlServerBuilder::defaults(&config)
            .expect("failed to initialise default backends")
            .build()
            .expect("failed to build SODL state"),
    );

    let app = sodl_api::router::build(state);

    let listener = tokio::net::TcpListener::bind(&config.listen)
        .await
        .expect("failed to bind");

    tracing::info!("SODL server listening on {}", config.listen);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");
}

/// Wait for Ctrl-C for graceful shutdown.
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to listen for ctrl-c");
    tracing::info!("shutdown signal received");
}
