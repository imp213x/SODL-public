//! SQLite-backed persistent metadata stores for SODL.
//!
//! This crate provides durable implementations of every metadata trait defined
//! across the SODL workspace.  It is designed as a **pluggable** persistence
//! layer — any application (Carla, The Scholar, or a third-party consumer) can
//! depend on `sodl-persist` to get production-grade storage without writing
//! their own database code.
//!
//! # Traits implemented
//!
//! | Trait               | Source crate    |
//! |---------------------|-----------------|
//! | `OriginRegistry`    | `sodl-origin`   |
//! | `PolicyStore`       | `sodl-policy`   |
//! | `PinStore`          | `sodl-policy`   |
//! | `RefCounter`        | `sodl-index`    |
//! | `ScanIndex`         | `sodl-index`    |
//! | `LineageStore`      | `sodl-index`    |
//! | `DerivationStore`   | `sodl-service`  |
//! | `ShareStore`        | `sodl-service`  |
//!
//! # Design principles
//!
//! - **Single-file SQLite** – one `.db` file per SODL instance.
//! - **WAL mode** – concurrent reads while writing.
//! - **Serialised metadata** – complex nested structs stored as JSON columns;
//!   indexed columns (IDs, timestamps, counts) are native SQL types.
//! - **Schema migration** via a `schema_version` user_version pragma.
//! - **Thread-safe** – the store is `Send + Sync` via `Mutex<Connection>`.

mod schema;

use std::path::Path;
use std::sync::Mutex;

use rusqlite::Connection;
use sodl_core::{BlobId, DerivationId, OriginId, Result, ShareId, SodlError};

/// A single SQLite-backed metadata store for SODL.
///
/// Wraps one `Connection` behind a `Mutex` for thread safety.  For higher
/// concurrency, callers can open the same file from multiple `SqliteStore`
/// instances (WAL mode supports this), but a single instance is sufficient
/// for most embedded use-cases.
pub struct SqliteStore {
    conn: Mutex<Connection>,
}

impl SqliteStore {
    /// Open (or create) a persistent store at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path.as_ref())
            .map_err(|e| SodlError::Io(format!("sqlite open: {e}")))?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.init()?;
        Ok(store)
    }

    /// Open an in-memory store (useful for tests).
    pub fn open_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| SodlError::Io(format!("sqlite open :memory: {e}")))?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.init()?;
        Ok(store)
    }

    fn init(&self) -> Result<()> {
        let conn = self.conn.lock().map_err(|e| SodlError::Io(e.to_string()))?;
        schema::apply_migrations(&conn)?;
        Ok(())
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Connection>> {
        self.conn.lock().map_err(|e| SodlError::Io(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Helper: convert rusqlite errors to SodlError
// ---------------------------------------------------------------------------
fn map_sql(e: rusqlite::Error) -> SodlError {
    match e {
        rusqlite::Error::QueryReturnedNoRows => SodlError::NotFound,
        other => SodlError::Io(format!("sqlite: {other}")),
    }
}

// ---------------------------------------------------------------------------
// OriginRegistry
// ---------------------------------------------------------------------------
impl sodl_origin::OriginRegistry for SqliteStore {
    fn create_origin(&self, record: sodl_origin::OriginRecord) -> Result<()> {
        let conn = self.lock()?;
        let oid = record.origin_id.0.to_string();
        let json =
            serde_json::to_string(&record).map_err(|e| SodlError::Io(format!("json: {e}")))?;
        conn.execute(
            "INSERT INTO origins (origin_id, data) VALUES (?1, ?2)",
            rusqlite::params![oid, json],
        )
        .map_err(|e| match e {
            rusqlite::Error::SqliteFailure(ref err, _)
                if err.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                SodlError::Conflict
            }
            other => map_sql(other),
        })?;
        Ok(())
    }

    fn get_origin(&self, origin_id: OriginId) -> Result<sodl_origin::OriginRecord> {
        let conn = self.lock()?;
        let oid = origin_id.0.to_string();
        let json: String = conn
            .query_row(
                "SELECT data FROM origins WHERE origin_id = ?1",
                rusqlite::params![oid],
                |row| row.get(0),
            )
            .map_err(map_sql)?;
        serde_json::from_str(&json).map_err(|e| SodlError::Io(format!("json: {e}")))
    }

    fn update_origin(&self, record: sodl_origin::OriginRecord) -> Result<()> {
        let conn = self.lock()?;
        let oid = record.origin_id.0.to_string();
        let json =
            serde_json::to_string(&record).map_err(|e| SodlError::Io(format!("json: {e}")))?;
        conn.execute(
            "INSERT OR REPLACE INTO origins (origin_id, data) VALUES (?1, ?2)",
            rusqlite::params![oid, json],
        )
        .map_err(map_sql)?;
        Ok(())
    }

    fn delete_origin(&self, origin_id: OriginId) -> Result<()> {
        let conn = self.lock()?;
        let oid = origin_id.0.to_string();
        conn.execute(
            "DELETE FROM origins WHERE origin_id = ?1",
            rusqlite::params![oid],
        )
        .map_err(map_sql)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PolicyStore
// ---------------------------------------------------------------------------
impl sodl_policy::PolicyStore for SqliteStore {
    fn get_origin_policy(&self, origin_id: OriginId) -> Result<sodl_policy::OriginPolicy> {
        let conn = self.lock()?;
        let oid = origin_id.0.to_string();
        let json: String = conn
            .query_row(
                "SELECT data FROM policies WHERE origin_id = ?1",
                rusqlite::params![oid],
                |row| row.get(0),
            )
            .map_err(map_sql)?;
        serde_json::from_str(&json).map_err(|e| SodlError::Io(format!("json: {e}")))
    }

    fn put_origin_policy(&self, policy: sodl_policy::OriginPolicy) -> Result<()> {
        let conn = self.lock()?;
        let oid = policy.origin_id.0.to_string();
        let json =
            serde_json::to_string(&policy).map_err(|e| SodlError::Io(format!("json: {e}")))?;
        conn.execute(
            "INSERT OR REPLACE INTO policies (origin_id, data) VALUES (?1, ?2)",
            rusqlite::params![oid, json],
        )
        .map_err(map_sql)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PinStore
// ---------------------------------------------------------------------------
impl sodl_policy::PinStore for SqliteStore {
    fn create_pin(&self, pin: sodl_policy::PinRecord) -> Result<()> {
        let conn = self.lock()?;
        let oid = match &pin.target {
            sodl_policy::PinTarget::Origin { origin_id } => origin_id.0.to_string(),
            sodl_policy::PinTarget::Representation { origin_id, .. } => origin_id.0.to_string(),
        };
        let json = serde_json::to_string(&pin).map_err(|e| SodlError::Io(format!("json: {e}")))?;
        conn.execute(
            "INSERT INTO pins (pin_id, origin_id, data) VALUES (?1, ?2, ?3)",
            rusqlite::params![pin.pin_id, oid, json],
        )
        .map_err(map_sql)?;
        Ok(())
    }

    fn get_pin(&self, pin_id: &str) -> Result<sodl_policy::PinRecord> {
        let conn = self.lock()?;
        let json: String = conn
            .query_row(
                "SELECT data FROM pins WHERE pin_id = ?1",
                rusqlite::params![pin_id],
                |row| row.get(0),
            )
            .map_err(map_sql)?;
        serde_json::from_str(&json).map_err(|e| SodlError::Io(format!("json: {e}")))
    }

    fn list_pins_for_origin(&self, origin_id: OriginId) -> Result<Vec<sodl_policy::PinRecord>> {
        let conn = self.lock()?;
        let oid = origin_id.0.to_string();
        let mut stmt = conn
            .prepare("SELECT data FROM pins WHERE origin_id = ?1")
            .map_err(map_sql)?;
        let rows = stmt
            .query_map(rusqlite::params![oid], |row| row.get::<_, String>(0))
            .map_err(map_sql)?;
        let mut out = Vec::new();
        for r in rows {
            let json = r.map_err(map_sql)?;
            out.push(serde_json::from_str(&json).map_err(|e| SodlError::Io(format!("json: {e}")))?);
        }
        Ok(out)
    }

    fn update_pin(&self, pin: sodl_policy::PinRecord) -> Result<()> {
        let conn = self.lock()?;
        let json = serde_json::to_string(&pin).map_err(|e| SodlError::Io(format!("json: {e}")))?;
        conn.execute(
            "UPDATE pins SET data = ?1 WHERE pin_id = ?2",
            rusqlite::params![json, pin.pin_id],
        )
        .map_err(map_sql)?;
        Ok(())
    }

    fn release_pin(&self, pin_id: &str) -> Result<()> {
        let mut pin = sodl_policy::PinStore::get_pin(self, pin_id)?;
        pin.state = sodl_policy::PinState::Released;
        sodl_policy::PinStore::update_pin(self, pin)
    }
}

// ---------------------------------------------------------------------------
// RefCounter + ScanIndex
// ---------------------------------------------------------------------------
impl sodl_index::RefCounter for SqliteStore {
    fn inc_origin(&self, origin_id: OriginId, _reason: sodl_index::RefKind) -> Result<()> {
        let conn = self.lock()?;
        let oid = origin_id.0.to_string();
        conn.execute(
            "INSERT INTO origin_refcounts (origin_id, count) VALUES (?1, 1)
             ON CONFLICT(origin_id) DO UPDATE SET count = count + 1",
            rusqlite::params![oid],
        )
        .map_err(map_sql)?;
        Ok(())
    }

    fn dec_origin(&self, origin_id: OriginId, _reason: sodl_index::RefKind) -> Result<()> {
        let conn = self.lock()?;
        let oid = origin_id.0.to_string();
        conn.execute(
            "UPDATE origin_refcounts SET count = MAX(count - 1, 0) WHERE origin_id = ?1",
            rusqlite::params![oid],
        )
        .map_err(map_sql)?;
        Ok(())
    }

    fn get_origin(&self, origin_id: OriginId) -> Result<i64> {
        let conn = self.lock()?;
        let oid = origin_id.0.to_string();
        let count: i64 = conn
            .query_row(
                "SELECT count FROM origin_refcounts WHERE origin_id = ?1",
                rusqlite::params![oid],
                |row| row.get(0),
            )
            .unwrap_or(0);
        Ok(count)
    }

    fn inc_blob(&self, blob_id: &BlobId, _reason: sodl_index::RefKind) -> Result<()> {
        let conn = self.lock()?;
        conn.execute(
            "INSERT INTO blob_refcounts (blob_id, count) VALUES (?1, 1)
             ON CONFLICT(blob_id) DO UPDATE SET count = count + 1",
            rusqlite::params![blob_id.0],
        )
        .map_err(map_sql)?;
        Ok(())
    }

    fn dec_blob(&self, blob_id: &BlobId, _reason: sodl_index::RefKind) -> Result<()> {
        let conn = self.lock()?;
        conn.execute(
            "UPDATE blob_refcounts SET count = MAX(count - 1, 0) WHERE blob_id = ?1",
            rusqlite::params![blob_id.0],
        )
        .map_err(map_sql)?;
        Ok(())
    }

    fn get_blob(&self, blob_id: &BlobId) -> Result<i64> {
        let conn = self.lock()?;
        let count: i64 = conn
            .query_row(
                "SELECT count FROM blob_refcounts WHERE blob_id = ?1",
                rusqlite::params![blob_id.0],
                |row| row.get(0),
            )
            .unwrap_or(0);
        Ok(count)
    }
}

impl sodl_index::ScanIndex for SqliteStore {
    fn list_origins(&self) -> Result<Vec<OriginId>> {
        let conn = self.lock()?;
        let mut stmt = conn
            .prepare("SELECT origin_id FROM origin_refcounts WHERE count > 0")
            .map_err(map_sql)?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(map_sql)?;
        let mut out = Vec::new();
        for r in rows {
            let s = r.map_err(map_sql)?;
            let uuid =
                uuid::Uuid::parse_str(&s).map_err(|e| SodlError::Io(format!("uuid: {e}")))?;
            out.push(OriginId(uuid));
        }
        Ok(out)
    }

    fn list_blobs(&self) -> Result<Vec<BlobId>> {
        let conn = self.lock()?;
        let mut stmt = conn
            .prepare("SELECT blob_id FROM blob_refcounts WHERE count > 0")
            .map_err(map_sql)?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(map_sql)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(BlobId(r.map_err(map_sql)?));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// LineageStore
// ---------------------------------------------------------------------------
impl sodl_index::LineageStore for SqliteStore {
    fn add_edge(&self, e: sodl_index::LineageEdge) -> Result<()> {
        let conn = self.lock()?;
        let oid = e.origin_id.0.to_string();
        let json = serde_json::to_string(&e).map_err(|e| SodlError::Io(format!("json: {e}")))?;
        conn.execute(
            "INSERT INTO lineage_edges (edge_id, origin_id, data) VALUES (?1, ?2, ?3)",
            rusqlite::params![e.edge_id, oid, json],
        )
        .map_err(map_sql)?;
        Ok(())
    }

    fn list_edges_for_origin(&self, origin_id: OriginId) -> Result<Vec<sodl_index::LineageEdge>> {
        let conn = self.lock()?;
        let oid = origin_id.0.to_string();
        let mut stmt = conn
            .prepare("SELECT data FROM lineage_edges WHERE origin_id = ?1 ORDER BY rowid")
            .map_err(map_sql)?;
        let rows = stmt
            .query_map(rusqlite::params![oid], |row| row.get::<_, String>(0))
            .map_err(map_sql)?;
        let mut out = Vec::new();
        for r in rows {
            let json = r.map_err(map_sql)?;
            out.push(serde_json::from_str(&json).map_err(|e| SodlError::Io(format!("json: {e}")))?);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// ProvenanceIndex
// ---------------------------------------------------------------------------
impl sodl_index::ProvenanceIndex for SqliteStore {
    fn put_payload_fingerprint(&self, origin_id: OriginId, fingerprint: &str) -> Result<()> {
        let conn = self.lock()?;
        conn.execute(
            "INSERT OR IGNORE INTO payload_fingerprints (fingerprint, origin_id) VALUES (?1, ?2)",
            rusqlite::params![fingerprint, origin_id.0.to_string()],
        )
        .map_err(map_sql)?;
        Ok(())
    }

    fn put_chunk_fingerprints(&self, origin_id: OriginId, fingerprints: &[String]) -> Result<()> {
        let mut unique = std::collections::HashSet::new();
        let conn = self.lock()?;
        for fingerprint in fingerprints {
            if !unique.insert(fingerprint) {
                continue;
            }
            conn.execute(
                "INSERT OR IGNORE INTO chunk_fingerprints (fingerprint, origin_id) VALUES (?1, ?2)",
                rusqlite::params![fingerprint, origin_id.0.to_string()],
            )
            .map_err(map_sql)?;
        }
        Ok(())
    }

    fn find_by_payload_fingerprint(&self, fingerprint: &str) -> Result<Vec<OriginId>> {
        let conn = self.lock()?;
        let mut stmt = conn
            .prepare("SELECT origin_id FROM payload_fingerprints WHERE fingerprint = ?1")
            .map_err(map_sql)?;
        let rows = stmt
            .query_map(rusqlite::params![fingerprint], |row| {
                row.get::<_, String>(0)
            })
            .map_err(map_sql)?;
        let mut out = Vec::new();
        for row in rows {
            let origin_id = row.map_err(map_sql)?;
            let uuid = uuid::Uuid::parse_str(&origin_id)
                .map_err(|e| SodlError::Io(format!("uuid: {e}")))?;
            out.push(OriginId(uuid));
        }
        Ok(out)
    }

    fn find_by_chunk_fingerprints(
        &self,
        fingerprints: &[String],
    ) -> Result<Vec<(OriginId, usize)>> {
        let mut counts = std::collections::HashMap::<OriginId, usize>::new();
        let mut unique = std::collections::HashSet::new();
        let conn = self.lock()?;
        for fingerprint in fingerprints {
            if !unique.insert(fingerprint) {
                continue;
            }
            let mut stmt = conn
                .prepare("SELECT origin_id FROM chunk_fingerprints WHERE fingerprint = ?1")
                .map_err(map_sql)?;
            let rows = stmt
                .query_map(rusqlite::params![fingerprint], |row| {
                    row.get::<_, String>(0)
                })
                .map_err(map_sql)?;
            for row in rows {
                let origin_id = row.map_err(map_sql)?;
                let uuid = uuid::Uuid::parse_str(&origin_id)
                    .map_err(|e| SodlError::Io(format!("uuid: {e}")))?;
                *counts.entry(OriginId(uuid)).or_default() += 1;
            }
        }
        let mut out = counts.into_iter().collect::<Vec<_>>();
        out.sort_by(|a, b| b.1.cmp(&a.1));
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// DerivationStore
// ---------------------------------------------------------------------------
impl sodl_service::DerivationStore for SqliteStore {
    fn put(&self, m: sodl_manifest::DerivationManifest) -> Result<()> {
        let conn = self.lock()?;
        let oid = m.origin_id.0.to_string();
        let did = &m.derivation_id.0;
        let json = serde_json::to_string(&m).map_err(|e| SodlError::Io(format!("json: {e}")))?;
        conn.execute(
            "INSERT OR REPLACE INTO derivations (origin_id, derivation_id, data) VALUES (?1, ?2, ?3)",
            rusqlite::params![oid, did, json],
        )
        .map_err(map_sql)?;
        Ok(())
    }

    fn get(
        &self,
        origin_id: OriginId,
        derivation_id: &DerivationId,
    ) -> Result<sodl_manifest::DerivationManifest> {
        let conn = self.lock()?;
        let oid = origin_id.0.to_string();
        let json: String = conn
            .query_row(
                "SELECT data FROM derivations WHERE origin_id = ?1 AND derivation_id = ?2",
                rusqlite::params![oid, derivation_id.0],
                |row| row.get(0),
            )
            .map_err(map_sql)?;
        serde_json::from_str(&json).map_err(|e| SodlError::Io(format!("json: {e}")))
    }

    fn list_for_origin(
        &self,
        origin_id: OriginId,
    ) -> Result<Vec<sodl_manifest::DerivationManifest>> {
        let conn = self.lock()?;
        let oid = origin_id.0.to_string();
        let mut stmt = conn
            .prepare("SELECT data FROM derivations WHERE origin_id = ?1")
            .map_err(map_sql)?;
        let rows = stmt
            .query_map(rusqlite::params![oid], |row| row.get::<_, String>(0))
            .map_err(map_sql)?;
        let mut out = Vec::new();
        for r in rows {
            let json = r.map_err(map_sql)?;
            out.push(serde_json::from_str(&json).map_err(|e| SodlError::Io(format!("json: {e}")))?);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// ShareStore
// ---------------------------------------------------------------------------
impl sodl_service::ShareStore for SqliteStore {
    fn put(&self, s: sodl_manifest::ShareRecord) -> Result<()> {
        let conn = self.lock()?;
        let oid = s.origin_id.0.to_string();
        let sid = &s.share_id.0;
        let json = serde_json::to_string(&s).map_err(|e| SodlError::Io(format!("json: {e}")))?;
        conn.execute(
            "INSERT OR REPLACE INTO shares (share_id, origin_id, data) VALUES (?1, ?2, ?3)",
            rusqlite::params![sid, oid, json],
        )
        .map_err(map_sql)?;
        Ok(())
    }

    fn get(&self, share_id: &ShareId) -> Result<sodl_manifest::ShareRecord> {
        let conn = self.lock()?;
        let json: String = conn
            .query_row(
                "SELECT data FROM shares WHERE share_id = ?1",
                rusqlite::params![share_id.0],
                |row| row.get(0),
            )
            .map_err(map_sql)?;
        serde_json::from_str(&json).map_err(|e| SodlError::Io(format!("json: {e}")))
    }

    fn list_for_origin(&self, origin_id: OriginId) -> Result<Vec<sodl_manifest::ShareRecord>> {
        let conn = self.lock()?;
        let oid = origin_id.0.to_string();
        let mut stmt = conn
            .prepare("SELECT data FROM shares WHERE origin_id = ?1")
            .map_err(map_sql)?;
        let rows = stmt
            .query_map(rusqlite::params![oid], |row| row.get::<_, String>(0))
            .map_err(map_sql)?;
        let mut out = Vec::new();
        for r in rows {
            let json = r.map_err(map_sql)?;
            out.push(serde_json::from_str(&json).map_err(|e| SodlError::Io(format!("json: {e}")))?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
