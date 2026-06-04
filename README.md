# SODL

SODL, the Single Origin Distributed Lineage system, is an open source storage and provenance layer for applications and AI systems.

It gives an app one durable way to answer a deceptively hard question:

> Where did this file, model artifact, dataset fragment, generated output, or transformed object come from, and can I reuse it safely?

It also asks the practical infrastructure question behind that:

> Why should 100 users sharing, downloading, and re-uploading the same 50 MB file create 5 GB of storage debt inside the same application?

SODL combines content-addressed storage, origin records, lineage, policy-aware retention, deduplication, chunking, encryption boundaries, and integration APIs. The goal is to make file and artifact management portable across web apps, AI training systems, local tools, and distributed services.

SODL was created by SoftQraft Labs Ltd and is released as open source under Apache-2.0.

SoftQraft Labs Ltd is the initial project steward. The project welcomes community use, feedback, integrations, and contributions under the contribution terms described in this repository.

## Why SODL Exists

Most applications start with simple uploads:

- store a file in object storage,
- keep a URL in a database,
- render the URL later.

That works until files become part of real workflows:

- users re-upload the same file,
- files are shared across users or teams,
- one file becomes a cropped, trimmed, encoded, or transformed derivative,
- deleted records still have downstream references,
- a private file needs different read rules from a public CDN file,
- AI artifacts need repeatable provenance and reuse,
- storage cost grows because the system stores bytes rather than origins.

SODL replaces URL-first thinking with origin-first thinking.

Applications store and reason about object IDs and origin records. SODL owns byte identity, lineage, chunk manifests, policy state, and reuse signals.

That distinction matters economically. In a conventional upload system, each user action often becomes a new physical object: user A uploads a file, user B downloads and re-uploads it, user C adds it to another workspace, and the application quietly pays for the same bytes again and again. The product may need separate ownership records, permissions, references, captions, scan states, and audit trails for each user, but it should not automatically need another full physical copy of the same payload.

SODL separates those concerns:

- many users can have their own object/reference records,
- each record can keep its own owner, policy, visibility, and lifecycle state,
- the byte payload can still resolve to the same content identity or provenance family,
- storage growth follows unique content and meaningful mutations, not repeated user behavior.

The result is less storage debt, cleaner provenance, and safer reuse.

## What SODL Provides

SODL is a Rust workspace with an HTTP service, a Python SDK, and foundations for future client SDKs.

Core capabilities:

- Content-addressed storage with Blake3 identities.
- Exact byte deduplication.
- Chunked storage for large objects.
- Chunk-overlap provenance for cuts and partial re-uploads.
- Origin records with owner, media kind, durability, and representation metadata.
- Lineage edges for shares, derivations, and future transformation graphs.
- Policy stores for durability and access decisions.
- Retention-aware garbage collection.
- Optional encryption boundary through AEAD-compatible providers.
- Payload readback through origin IDs.
- OpenAPI contract for app integration.
- Python modules for AI storage, model weights, training artifacts, and reuse-first workflows.

SODL is not trying to be a CDN, CMS, vector database, or full object-store replacement. It is the provenance and lifecycle layer that can sit beside those systems.

## Integration Model

There are three intended ways to use SODL.

### 1. HTTP Engine

Run `sodl-server` as a private sidecar or internal service. Applications talk to it over HTTP.

This is the recommended integration path for web apps and services in any language.

```text
Your App
  -> POST /v1/upload
  -> GET  /v1/origins/{id}/payload
  -> POST /v1/provenance/resolve
  -> POST /v1/shares
  -> POST /v1/derivations
```

Example production layout:

```text
SODL_LISTEN=127.0.0.1:7700
SODL_BLOB_DIR=/var/lib/sodl/blobs
SODL_DB_PATH=/var/lib/sodl/sodl.sqlite
SODL_MASTER_KEY=<managed-secret>
```

An app then configures:

```text
SODL_ENGINE_URL=http://127.0.0.1:7700
```

This is how The Scholar is intended to consume SODL: The Scholar remains a normal application, while SODL acts as the storage/provenance engine.

### 2. Rust Crates

Rust applications can embed SODL directly through the workspace crates.

Important crates:

| Crate | Role |
| --- | --- |
| `sodl-core` | Shared IDs, errors, media types, durability, capabilities. |
| `sodl-cas` | Content-addressed blob identities and filesystem blob store. |
| `sodl-chunk` | Fixed and content-defined chunking plus manifests. |
| `sodl-crypto` | Encryption provider boundaries and AEAD implementation. |
| `sodl-origin` | Origin registry and representation metadata. |
| `sodl-index` | Reference counts, lineage edges, and provenance indexes. |
| `sodl-persist` | SQLite-backed metadata stores. |
| `sodl-policy` | Access and retention policy records. |
| `sodl-service` | High-level facade for upload, payload read, share, derivation, GC inputs, and provenance. |
| `sodl-api` | HTTP API and `sodl-server` binary. |
| `sodl-gc` | Policy-aware garbage collection primitives. |
| `sodl-proof` | Lineage proof support. |

### 3. Python AI Toolkit

The Python package is aimed at AI workflows that need artifact reuse, model-weight storage, training lifecycle tracking, and content-addressed checkpointing.

Use cases:

- Store model weights and optimizer states in content-addressed form.
- Track model origins, adapters, exports, and training lineage.
- Reuse expensive artifacts when inputs and configuration match.
- Compress and cache embedding clusters.
- Build semantic or token-hash indexes for training acceleration experiments.
- Keep dataset/artifact integrity checks close to training code.

Install locally:

```bash
cd python
python -m pip install -e ".[all]"
```

Optional Rust acceleration:

```bash
python -m pip install maturin
cd ../crates/sodl-python-ffi
python -m maturin develop --release
```

## Quick Start

Build and test the Rust workspace:

```bash
cargo build
cargo test
```

Run the HTTP server:

```bash
cargo run -p sodl-api
```

By default it listens on:

```text
http://127.0.0.1:7700
```

Health check:

```bash
curl http://127.0.0.1:7700/health
```

Run with Docker:

```bash
docker build -t sodl-server:local .
docker run --rm -p 7700:7700 -v sodl-data:/data sodl-server:local
```

The container listens on `0.0.0.0:7700`, stores blobs in `/data/blobs`, stores SQLite metadata at `/data/sodl.db`, and exposes `/health` for orchestration health checks. Production deployments should mount `/data` on durable storage and set `SODL_MASTER_KEY` through a secret manager.

Upload bytes:

```bash
curl -X POST http://127.0.0.1:7700/v1/upload \
  -F 'meta={"owner":"user:demo","media_kind":"document","mime":"application/pdf","durability":"durable"}' \
  -F "file=@./example.pdf"
```

Resolve provenance before storing a new upload:

```bash
curl -X POST http://127.0.0.1:7700/v1/provenance/resolve \
  -F 'meta={"media_kind":"document","mime":"application/pdf"}' \
  -F "file=@./example.pdf"
```

Read an origin payload:

```bash
curl http://127.0.0.1:7700/v1/origins/<origin-id>/payload --output restored.bin
```

## App Integration Pattern

A production app should treat SODL as the system of record for file provenance, not necessarily as the only byte-serving layer.

Recommended flow:

1. App receives an upload.
2. App validates file type, size, actor, and quota.
3. App asks SODL to resolve provenance for the bytes.
4. App stores or mirrors bytes through SODL.
5. App stores its own business record with the SODL origin/object ID.
6. App serves files through its normal authorization layer.
7. App can use SODL payload readback or legacy object storage fallback during migration.

This lets an app keep its product-specific rules while SODL owns origin identity, dedupe, lineage, and storage lifecycle.

## AI Integration Pattern

AI systems often waste storage and compute by regenerating artifacts whose inputs already exist. SODL is designed to make reuse a first-class operation.

Typical AI flow:

1. Canonicalize the input artifact or configuration.
2. Compute a stable origin or artifact fingerprint.
3. Ask SODL whether an equivalent or overlapping artifact already exists.
4. If found, reuse the existing origin.
5. If not found, create a new origin and store the generated artifact.
6. Attach lineage from source datasets, model checkpoints, prompts, adapters, transforms, and evaluation outputs.

This supports:

- reproducible generation,
- checkpoint provenance,
- dataset lineage,
- model artifact reuse,
- storage-aware training pipelines,
- future audit trails for AI-produced content.

## Status

SODL is early but working. The repository contains stable primitives and experimental modules.

Stable enough for integration experiments:

- HTTP upload and payload readback.
- Content addressing.
- Chunking.
- SQLite persistence.
- Origin records.
- Share and derivation records.
- Reference counting.
- Exact and chunk-overlap provenance.

Still evolving:

- Semantic fingerprinting for re-encoded documents, images, audio, and video.
- Production deployment images.
- Public npm and Python client packages.
- External partner API hardening.
- Operational GC commands and dashboards.
- Multi-node replication and repair.

## Security Notes

SODL is designed to run as a private service behind an application boundary. Do not expose the development server directly to the public internet without an application gateway, authentication, authorization, request limits, and upload policy enforcement.

For production:

- Set `SODL_MASTER_KEY`.
- Use a managed secret store.
- Keep `SODL_BLOB_DIR` and `SODL_DB_PATH` outside the source checkout.
- Back up both runtime paths.
- Enforce upload limits at the calling application and reverse proxy.
- Run SODL on localhost, a private network, or behind an authenticated API gateway.

See `SECURITY.md` for vulnerability reporting and deployment guidance.

## Repository Layout

```text
crates/          Rust crates and sodl-server
docs/            OpenAPI and architecture notes
guides/          Development guides and research notes
python/          Python SDK and AI tooling
Cargo.toml       Rust workspace
LICENSE          Apache-2.0 license
SECURITY.md      Security policy
```

## Development

Run Rust tests:

```bash
cargo test
```

Run focused API/service tests:

```bash
cargo test -p sodl-api -p sodl-service -p sodl-persist -p sodl-index -p sodl-chunk
```

Run Python tests:

```bash
cd python
python -m pip install -e ".[test]"
pytest
```

## Roadmap

Near term:

- Publish a public Docker image for `sodl-server`.
- Add generated TypeScript and Python HTTP clients.
- Finish app-facing file lifecycle examples.
- Add operational GC command/API.
- Add production deployment examples.

Medium term:

- Add semantic fingerprint resolvers for transformed files.
- Add stronger external API ergonomics.
- Add multi-node replication examples.
- Publish Rust crates and Python packages.
- Add reference integrations for web apps and AI training systems.

## License

Apache License, Version 2.0. See `LICENSE`.

Copyright 2026 SoftQraft Labs Ltd and SODL contributors.
