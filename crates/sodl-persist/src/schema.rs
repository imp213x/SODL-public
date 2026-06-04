//! Schema definition and migration for the SODL SQLite metadata store.
//!
//! Uses SQLite's `user_version` pragma to track schema version.
//! Each migration is additive — we never drop tables.

use rusqlite::Connection;
use sodl_core::{Result, SodlError};

/// Current schema version.
const CURRENT_VERSION: i32 = 2;

/// Apply all pending migrations.  Idempotent — safe to call on every open.
pub fn apply_migrations(conn: &Connection) -> Result<()> {
    // Enable WAL for concurrent read + write.
    conn.execute_batch("PRAGMA journal_mode = WAL;")
        .map_err(|e| SodlError::Io(format!("pragma wal: {e}")))?;

    // Foreign keys on.
    conn.execute_batch("PRAGMA foreign_keys = ON;")
        .map_err(|e| SodlError::Io(format!("pragma fk: {e}")))?;

    let version: i32 = conn
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .map_err(|e| SodlError::Io(format!("pragma user_version: {e}")))?;

    if version < 1 {
        migrate_v1(conn)?;
    }
    if version < 2 {
        migrate_v2(conn)?;
    }

    // Bump version.
    conn.pragma_update(None, "user_version", CURRENT_VERSION)
        .map_err(|e| SodlError::Io(format!("pragma set user_version: {e}")))?;

    Ok(())
}

/// V2: provenance fingerprints for exact payload and chunk-overlap matching.
fn migrate_v2(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS payload_fingerprints (
            fingerprint TEXT NOT NULL,
            origin_id   TEXT NOT NULL,
            PRIMARY KEY (fingerprint, origin_id)
        );
        CREATE INDEX IF NOT EXISTS idx_payload_fingerprints_origin
            ON payload_fingerprints(origin_id);

        CREATE TABLE IF NOT EXISTS chunk_fingerprints (
            fingerprint TEXT NOT NULL,
            origin_id   TEXT NOT NULL,
            PRIMARY KEY (fingerprint, origin_id)
        );
        CREATE INDEX IF NOT EXISTS idx_chunk_fingerprints_origin
            ON chunk_fingerprints(origin_id);
        ",
    )
    .map_err(|e| SodlError::Io(format!("migrate v2: {e}")))?;

    Ok(())
}

/// V1: initial tables for all metadata stores.
fn migrate_v1(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        -- Origins (OriginRegistry)
        CREATE TABLE IF NOT EXISTS origins (
            origin_id TEXT PRIMARY KEY NOT NULL,
            data      TEXT NOT NULL
        );

        -- Policies (PolicyStore)
        CREATE TABLE IF NOT EXISTS policies (
            origin_id TEXT PRIMARY KEY NOT NULL,
            data      TEXT NOT NULL
        );

        -- Pins (PinStore)
        CREATE TABLE IF NOT EXISTS pins (
            pin_id    TEXT PRIMARY KEY NOT NULL,
            origin_id TEXT NOT NULL,
            data      TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_pins_origin ON pins(origin_id);

        -- Origin reference counts (RefCounter)
        CREATE TABLE IF NOT EXISTS origin_refcounts (
            origin_id TEXT PRIMARY KEY NOT NULL,
            count     INTEGER NOT NULL DEFAULT 0
        );

        -- Blob reference counts (RefCounter)
        CREATE TABLE IF NOT EXISTS blob_refcounts (
            blob_id TEXT PRIMARY KEY NOT NULL,
            count   INTEGER NOT NULL DEFAULT 0
        );

        -- Lineage edges (LineageStore)
        CREATE TABLE IF NOT EXISTS lineage_edges (
            edge_id   TEXT PRIMARY KEY NOT NULL,
            origin_id TEXT NOT NULL,
            data      TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_lineage_origin ON lineage_edges(origin_id);

        -- Derivations (DerivationStore)
        CREATE TABLE IF NOT EXISTS derivations (
            origin_id     TEXT NOT NULL,
            derivation_id TEXT NOT NULL,
            data          TEXT NOT NULL,
            PRIMARY KEY (origin_id, derivation_id)
        );

        -- Shares (ShareStore)
        CREATE TABLE IF NOT EXISTS shares (
            share_id  TEXT PRIMARY KEY NOT NULL,
            origin_id TEXT NOT NULL,
            data      TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_shares_origin ON shares(origin_id);
        ",
    )
    .map_err(|e| SodlError::Io(format!("migrate v1: {e}")))?;

    Ok(())
}

#[cfg(test)]
mod schema_tests {
    use super::*;

    #[test]
    fn migration_is_idempotent() {
        let conn = Connection::open_in_memory().unwrap();
        apply_migrations(&conn).unwrap();
        apply_migrations(&conn).unwrap(); // second call is a no-op
    }
}
