# Priority #2 — SODL HTTP API (`sodl-api`)

> **Status:** COMPLETE — compiled, 28/28 workspace tests green, live smoke-tested  
> **Crate:** `crates/sodl-api`  
> **Binary:** `sodl-server`

---

## What was built

A standalone **axum 0.8** REST server that wraps `SodlService` with the
persistent backends from Priority #1 (`FsBlobStore` + `SqliteStore`).
It exposes every SODL operation as a JSON/multipart HTTP endpoint,
ready for any language or platform to consume.

## Architecture

```
┌────────────────────────────────────────────────────┐
│  sodl-api  (binary crate)                          │
│                                                    │
│  main.rs   → tokio runtime, tracing, graceful      │
│               shutdown (Ctrl-C / SIGTERM)           │
│  config.rs → env-based: SODL_LISTEN, SODL_BLOB_DIR,│
│               SODL_DB_PATH                          │
│  state.rs  → AppState owns Arc'd stores, builds    │
│               request-scoped SodlService<'_>        │
│  dto.rs    → serde request / response schemas       │
│  handlers.rs → 13 handler functions                 │
│  router.rs → route table + CORS + tracing layers    │
└────────────────────────────────────────────────────┘
         ▼ delegates to ▼
┌──────────────────────────────────┐
│  sodl-service::SodlService<'a>   │
│  sodl-persist::SqliteStore       │
│  sodl-cas::FsBlobStore           │
│  sodl-store::NullCrypto          │
└──────────────────────────────────┘
```

### Key design decisions

| Decision | Rationale |
|---|---|
| **Request-scoped `SodlService`** | `SodlService<'a>` borrows its stores. Axum requires `'static` state. `AppState` owns stores in `Arc`; `state.service()` builds a short-lived `SodlService<'_>` per request. |
| **Multipart upload** | File bytes arrive as `multipart/form-data`. Owner and media_type are form fields, not JSON. Simpler for CLI tools & browser uploads. |
| **JSON everywhere else** | All non-binary responses are `application/json`. |
| **`/v1/` prefix** | Allows future versioned breaking changes via `/v2/`. |
| **CorsLayer::permissive** | For development. Tighten before production. |

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness check (`{"status":"ok","version":"sodl-v1"}`) |
| `POST` | `/v1/upload` | Upload bytes → create Origin (multipart) |
| `GET` | `/v1/origins/{id}` | Get origin metadata |
| `DELETE` | `/v1/origins/{id}` | Tombstone an origin |
| `GET` | `/v1/origins/{id}/lineage-proof` | Get lineage digest |
| `GET` | `/v1/blobs/{id}` | Download raw blob bytes |
| `POST` | `/v1/shares` | Create a share (from → to) |
| `GET` | `/v1/shares/{id}` | Get share record |
| `DELETE` | `/v1/shares/{id}` | Release (revoke) share |
| `POST` | `/v1/shares/{id}/verify` | Verify share proof integrity |
| `POST` | `/v1/derivations` | Declare a derivation |
| `POST` | `/v1/pins` | Pin an origin (durability) |
| `DELETE` | `/v1/pins/{id}` | Release pin |

## Configuration (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `SODL_LISTEN` | `127.0.0.1:7700` | Bind address |
| `SODL_BLOB_DIR` | `./sodl_data/blobs` | Filesystem blob root |
| `SODL_DB_PATH` | `./sodl_data/sodl.db` | SQLite database path |

## Quick start

```bash
cargo build -p sodl-api --release
./target/release/sodl-server          # or sodl-server.exe on Windows

# Health check
curl http://127.0.0.1:7700/health

# Upload a file
curl -X POST http://127.0.0.1:7700/v1/upload \
  -F "file=@myfile.pdf" \
  -F "owner=user:alice" \
  -F "media_type=application/pdf"

# Retrieve blob bytes
curl http://127.0.0.1:7700/v1/blobs/blake3:<hex>

# Create a share
curl -X POST http://127.0.0.1:7700/v1/shares \
  -H "Content-Type: application/json" \
  -d '{"origin_id":"<uuid>","from":"user:alice","to":"user:bob"}'
```

## Smoke test results (live-verified)

| Test | Result |
|------|--------|
| `GET /health` | `{"status":"ok","version":"sodl-v1"}` ✅ |
| `POST /v1/upload` (multipart) | Returns `origin_id` + `blob_id` ✅ |
| `GET /v1/origins/{id}` | Full metadata with reps, owner, durability ✅ |
| `POST /v1/shares` | Returns `share_id` ✅ |
| `POST /v1/shares/{id}/verify` | `{"valid":true}` ✅ |
| `GET /v1/origins/{id}/lineage-proof` | Returns digest ✅ |
| `POST /v1/pins` | Returns `pin_id` ✅ |
| `GET /v1/blobs/{id}` | Returns raw bytes ✅ |
| Disk persistence | Blob file + SQLite DB created ✅ |

## Files created / modified

| File | Action |
|------|--------|
| `crates/sodl-api/Cargo.toml` | **Created** — binary crate manifest |
| `crates/sodl-api/src/main.rs` | **Created** — entry point |
| `crates/sodl-api/src/config.rs` | **Created** — env configuration |
| `crates/sodl-api/src/state.rs` | **Created** — AppState + service builder |
| `crates/sodl-api/src/dto.rs` | **Created** — request/response types |
| `crates/sodl-api/src/handlers.rs` | **Created** — 13 endpoint handlers |
| `crates/sodl-api/src/router.rs` | **Created** — route table |
| `Cargo.toml` (workspace) | **Modified** — added `sodl-api` to members |

## What's next (Priority #3)

Carla ↔ SODL integration at the ingest layer: a Python client that
calls these endpoints to store ingested documents as Origins with full
lineage tracking.
