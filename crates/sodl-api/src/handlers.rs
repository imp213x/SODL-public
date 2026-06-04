//! HTTP handlers for the SODL REST API.
//!
//! Each handler receives shared `AppState`, constructs a request-scoped
//! `SodlService`, performs the operation, and returns a JSON response.
//!
//! # Endpoint summary
//!
//! | Method | Path                            | Description                |
//! |--------|---------------------------------|----------------------------|
//! | GET    | `/health`                       | Health / version check     |
//! | POST   | `/v1/upload`                    | Upload bytes (multipart)   |
//! | GET    | `/v1/origins/:id`               | Get origin metadata        |
//! | DELETE | `/v1/origins/:id`               | Tombstone an origin        |
//! | GET    | `/v1/blobs/:id`                 | Fetch raw blob bytes       |
//! | POST   | `/v1/shares`                    | Create a share             |
//! | GET    | `/v1/shares/:id`                | Get share record           |
//! | DELETE | `/v1/shares/:id`                | Release a share            |
//! | POST   | `/v1/shares/:id/verify`         | Verify share proof         |
//! | POST   | `/v1/derivations`               | Create a derivation        |
//! | POST   | `/v1/pins`                      | Pin an origin              |
//! | DELETE | `/v1/pins/:id`                  | Release a pin              |
//! | GET    | `/v1/origins/:id/payload`       | Fetch reassembled payload  |
//! | GET    | `/v1/origins/:id/lineage-proof` | Compute lineage proof      |

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Multipart, Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use bytes::Bytes;

use sodl_cas::compute_blob_id;
use sodl_core::{
    new_origin_id, BlobId, Capability, Durability, MediaKind, OriginId, PrincipalId,
    SODL_SCHEMA_VERSION,
};
use sodl_index::{LineageEdge, ProvenanceMatchKind, RefKind};
use sodl_manifest::DerivationKind;
use sodl_origin::{OriginRecord, Representation};
use sodl_policy::{AccessPolicy, OriginPolicy, RetentionPolicy};
use sodl_service::UploadRequest;

use crate::dto::*;
use crate::state::AppState;

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

/// Map `SodlError` to an HTTP response with appropriate status code.
fn err_response(e: sodl_core::SodlError) -> Response {
    let (status, msg) = match &e {
        sodl_core::SodlError::NotFound => (StatusCode::NOT_FOUND, e.to_string()),
        sodl_core::SodlError::Conflict => (StatusCode::CONFLICT, e.to_string()),
        sodl_core::SodlError::Unauthorized => (StatusCode::FORBIDDEN, e.to_string()),
        sodl_core::SodlError::Invalid(_) => (StatusCode::BAD_REQUEST, e.to_string()),
        sodl_core::SodlError::Integrity => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "integrity check failed".into(),
        ),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    (status, Json(ErrorResponse { error: msg })).into_response()
}

// ---------------------------------------------------------------------------
// Parse helpers
// ---------------------------------------------------------------------------

fn parse_origin_id(s: &str) -> Result<OriginId, Response> {
    uuid::Uuid::parse_str(s).map(OriginId).map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("invalid origin_id: {s}"),
            }),
        )
            .into_response()
    })
}

fn parse_media_kind(s: &str) -> MediaKind {
    match s.to_lowercase().as_str() {
        "image" => MediaKind::Image,
        "audio" => MediaKind::Audio,
        "video" => MediaKind::Video,
        "document" => MediaKind::Document,
        "ai_model" | "aimodel" | "model" => MediaKind::AiModel,
        _ => MediaKind::Binary,
    }
}

fn parse_durability(s: &str) -> Durability {
    match s.to_lowercase().as_str() {
        "ephemeral" => Durability::Ephemeral,
        "durable" => Durability::Durable,
        _ => Durability::BestEffort,
    }
}

fn parse_capability(s: &str) -> Capability {
    match s.to_lowercase().as_str() {
        "reshare" => Capability::Reshare,
        "derive" => Capability::Derive,
        "pin" => Capability::Pin,
        "admin" => Capability::Admin,
        _ => Capability::Read,
    }
}

fn format_media_kind(mk: &MediaKind) -> &'static str {
    match mk {
        MediaKind::Binary => "binary",
        MediaKind::Image => "image",
        MediaKind::Audio => "audio",
        MediaKind::Video => "video",
        MediaKind::Document => "document",
        MediaKind::AiModel => "ai_model",
    }
}

fn format_durability(d: &Durability) -> &'static str {
    match d {
        Durability::Ephemeral => "ephemeral",
        Durability::BestEffort => "best_effort",
        Durability::Durable => "durable",
    }
}

fn format_cap(c: &Capability) -> &'static str {
    match c {
        Capability::Read => "read",
        Capability::Reshare => "reshare",
        Capability::Derive => "derive",
        Capability::Pin => "pin",
        Capability::Admin => "admin",
    }
}

fn representation_response(rep: &Representation) -> RepresentationResponse {
    RepresentationResponse {
        name: rep.name.clone(),
        media_kind: format_media_kind(&rep.media_kind).into(),
        mime: rep.mime.clone(),
        size_bytes: rep.size_bytes,
        root_blobs: rep.root_blobs.iter().map(|b| b.0.clone()).collect(),
    }
}

fn origin_response(rec: OriginRecord) -> OriginResponse {
    OriginResponse {
        origin_id: rec.origin_id.0.to_string(),
        media_kind: format_media_kind(&rec.media_kind).into(),
        durability: format_durability(&rec.durability).into(),
        created_at: rec.created_at.to_string(),
        tombstoned_at: rec.tombstoned_at.map(|t| t.to_string()),
        representations: rec
            .representations
            .iter()
            .map(representation_response)
            .collect(),
        owner: rec.owner.map(|o| o.0),
    }
}

// ---------------------------------------------------------------------------
// GET /health
// ---------------------------------------------------------------------------

pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        version: SODL_SCHEMA_VERSION.into(),
    })
}

// ---------------------------------------------------------------------------
// GET /v1/origins
// ---------------------------------------------------------------------------

pub async fn list_origins(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ListOriginsResponse>, Response> {
    let origin_ids = state.scan.list_origins().map_err(err_response)?;
    let mut origins = Vec::with_capacity(origin_ids.len());
    for origin_id in origin_ids {
        origins.push(origin_response(
            state
                .origin_registry
                .get_origin(origin_id)
                .map_err(err_response)?,
        ));
    }
    Ok(Json(ListOriginsResponse { origins }))
}

// ---------------------------------------------------------------------------
// POST /v1/origins
// ---------------------------------------------------------------------------

pub async fn create_origin(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateOriginRequest>,
) -> Result<(StatusCode, Json<OriginResponse>), Response> {
    let origin_id = new_origin_id();
    let media_kind = parse_media_kind(&req.media_kind);
    let durability = parse_durability(&req.durability);

    state
        .policy_store
        .put_origin_policy(OriginPolicy {
            origin_id,
            retention: RetentionPolicy {
                durability,
                ttl_seconds: None,
                min_replicas: match durability {
                    Durability::Durable => Some(1),
                    _ => None,
                },
            },
            access: AccessPolicy {
                default_caps: vec![Capability::Read],
                allow_reshare: true,
                allow_derivation: true,
            },
        })
        .map_err(err_response)?;

    let mut record = OriginRecord::new(origin_id, media_kind, durability);
    record.owner = req.owner.map(PrincipalId);

    for rep in req.representations {
        let root_blobs = rep.root_blobs.into_iter().map(BlobId).collect::<Vec<_>>();
        for blob_id in &root_blobs {
            if !state.blobs.has(blob_id).map_err(err_response)? {
                return Err((
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: format!("blob not found: {}", blob_id.0),
                    }),
                )
                    .into_response());
            }
        }

        let representation = Representation {
            name: rep.name,
            media_kind: parse_media_kind(&rep.media_kind),
            mime: rep.mime,
            size_bytes: rep.size_bytes,
            root_blobs,
        };
        state
            .index
            .inc_origin(
                origin_id,
                RefKind::OriginRepresentation {
                    name: representation.name.clone(),
                },
            )
            .map_err(err_response)?;
        for blob_id in &representation.root_blobs {
            state
                .index
                .inc_blob(
                    blob_id,
                    RefKind::OriginRepresentation {
                        name: representation.name.clone(),
                    },
                )
                .map_err(err_response)?;
            state
                .lineage
                .add_edge(LineageEdge {
                    edge_id: format!("edge:{}", uuid::Uuid::new_v4()),
                    origin_id,
                    blob_id: Some(blob_id.clone()),
                    kind: RefKind::OriginRepresentation {
                        name: representation.name.clone(),
                    },
                    created_at: time::OffsetDateTime::now_utc(),
                })
                .map_err(err_response)?;
        }
        record.representations.push(representation);
    }

    state
        .origin_registry
        .create_origin(record.clone())
        .map_err(err_response)?;

    Ok((StatusCode::CREATED, Json(origin_response(record))))
}

// ---------------------------------------------------------------------------
// POST /v1/upload  (multipart: field "meta" JSON + field "file" bytes)
// ---------------------------------------------------------------------------

pub async fn upload(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<(StatusCode, Json<UploadResponse>), Response> {
    let mut meta: Option<UploadMeta> = None;
    let mut file_bytes: Option<Bytes> = None;

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("multipart error: {e}"),
            }),
        )
            .into_response()
    })? {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "meta" => {
                let text = field.text().await.map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: format!("read meta: {e}"),
                        }),
                    )
                        .into_response()
                })?;
                meta = Some(serde_json::from_str(&text).map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: format!("parse meta: {e}"),
                        }),
                    )
                        .into_response()
                })?);
            }
            "file" => {
                file_bytes = Some(field.bytes().await.map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: format!("read file: {e}"),
                        }),
                    )
                        .into_response()
                })?);
            }
            _ => { /* ignore unknown fields */ }
        }
    }

    let meta = meta.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "missing 'meta' field".into(),
            }),
        )
            .into_response()
    })?;
    let file_bytes = file_bytes.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "missing 'file' field".into(),
            }),
        )
            .into_response()
    })?;

    let media_kind = parse_media_kind(&meta.media_kind);
    let durability = parse_durability(&meta.durability);

    // Build a policy with sensible defaults.
    let policy = OriginPolicy {
        origin_id: sodl_core::new_origin_id(), // overwritten by service
        retention: RetentionPolicy {
            durability,
            ttl_seconds: None,
            min_replicas: match durability {
                Durability::Durable => Some(1),
                _ => None,
            },
        },
        access: AccessPolicy {
            default_caps: vec![Capability::Read],
            allow_reshare: true,
            allow_derivation: true,
        },
    };

    let svc = state.service();
    let result = svc
        .upload(UploadRequest {
            owner: PrincipalId(meta.owner),
            media_kind,
            mime: meta.mime,
            durability_policy: policy,
            bytes: file_bytes,
        })
        .map_err(err_response)?;

    Ok((
        StatusCode::CREATED,
        Json(UploadResponse {
            origin_id: result.origin_id.0.to_string(),
            blob_id: result.blob_id.0,
            chunked: result.chunked,
            chunk_blobs: result.chunk_blobs.iter().map(|b| b.0.clone()).collect(),
        }),
    ))
}

fn provenance_candidate_response(
    candidate: sodl_index::ProvenanceCandidate,
) -> ProvenanceCandidateResponse {
    match candidate.kind {
        ProvenanceMatchKind::ExactPayload => ProvenanceCandidateResponse {
            origin_id: candidate.origin_id.0.to_string(),
            match_kind: "exact_payload".into(),
            confidence: candidate.confidence,
            matched_chunks: None,
            total_chunks: None,
            ratio: None,
        },
        ProvenanceMatchKind::ChunkOverlap {
            matched_chunks,
            total_chunks,
            ratio,
        } => ProvenanceCandidateResponse {
            origin_id: candidate.origin_id.0.to_string(),
            match_kind: "chunk_overlap".into(),
            confidence: candidate.confidence,
            matched_chunks: Some(matched_chunks),
            total_chunks: Some(total_chunks),
            ratio: Some(ratio),
        },
    }
}

// ---------------------------------------------------------------------------
// POST /v1/provenance/resolve  (multipart: optional "meta" JSON + "file" bytes)
// ---------------------------------------------------------------------------

pub async fn resolve_provenance(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<ResolveProvenanceResponse>, Response> {
    let mut file_bytes: Option<Bytes> = None;

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("multipart error: {e}"),
            }),
        )
            .into_response()
    })? {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "meta" => {
                // Reserved for future media-aware resolvers. Parse to reject malformed JSON.
                let text = field.text().await.map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: format!("read meta: {e}"),
                        }),
                    )
                        .into_response()
                })?;
                let _: ResolveProvenanceMeta = serde_json::from_str(&text).map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: format!("parse meta: {e}"),
                        }),
                    )
                        .into_response()
                })?;
            }
            "file" => {
                file_bytes = Some(field.bytes().await.map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: format!("read file: {e}"),
                        }),
                    )
                        .into_response()
                })?);
            }
            _ => {}
        }
    }

    let file_bytes = file_bytes.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "missing 'file' field".into(),
            }),
        )
            .into_response()
    })?;

    let svc = state.service();
    let resolved = svc.resolve_provenance(&file_bytes).map_err(err_response)?;

    Ok(Json(ResolveProvenanceResponse {
        payload_fingerprint: resolved.payload_fingerprint,
        chunk_count: resolved.chunk_fingerprints.len(),
        candidates: resolved
            .candidates
            .into_iter()
            .map(provenance_candidate_response)
            .collect(),
    }))
}

// ---------------------------------------------------------------------------
// GET /v1/origins/:id
// ---------------------------------------------------------------------------

pub async fn get_origin(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<OriginResponse>, Response> {
    let oid = parse_origin_id(&id)?;
    Ok(Json(origin_response(
        state
            .origin_registry
            .get_origin(oid)
            .map_err(err_response)?,
    )))
}

// ---------------------------------------------------------------------------
// DELETE /v1/origins/:id
// ---------------------------------------------------------------------------

pub async fn tombstone_origin(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    body: Option<Json<TombstoneRequest>>,
) -> Result<StatusCode, Response> {
    let oid = parse_origin_id(&id)?;
    let reason = body
        .map(|b| b.0.reason)
        .unwrap_or_else(|| "deleted by owner".into());

    let svc = state.service();
    svc.tombstone_origin(oid, &reason).map_err(err_response)?;

    Ok(StatusCode::NO_CONTENT)
}

// ---------------------------------------------------------------------------
// GET /v1/origins/:id/representations
// ---------------------------------------------------------------------------

pub async fn list_origin_representations(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<RepresentationsResponse>, Response> {
    let oid = parse_origin_id(&id)?;
    let rec = state
        .origin_registry
        .get_origin(oid)
        .map_err(err_response)?;
    Ok(Json(RepresentationsResponse {
        origin_id: rec.origin_id.0.to_string(),
        representations: rec
            .representations
            .iter()
            .map(representation_response)
            .collect(),
    }))
}

// ---------------------------------------------------------------------------
// POST /v1/blobs
// ---------------------------------------------------------------------------

pub async fn create_blob(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<(StatusCode, Json<BlobCreateResponse>), Response> {
    let blob_id = compute_blob_id(&body, state.hash_alg);
    if let Some(expected) = headers
        .get("x-blob-id")
        .and_then(|value| value.to_str().ok())
    {
        if expected != blob_id.0 {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!(
                        "x-blob-id mismatch: expected {expected}, computed {}",
                        blob_id.0
                    ),
                }),
            )
                .into_response());
        }
    }

    let existed = state.blobs.has(&blob_id).map_err(err_response)?;
    if !existed {
        state
            .blobs
            .put(&blob_id, body.clone())
            .map_err(err_response)?;
    }

    Ok((
        if existed {
            StatusCode::OK
        } else {
            StatusCode::CREATED
        },
        Json(BlobCreateResponse {
            blob_id: blob_id.0,
            existed,
            size_bytes: body.len(),
        }),
    ))
}

// ---------------------------------------------------------------------------
// GET /v1/blobs/:id  (raw bytes)
// ---------------------------------------------------------------------------

pub async fn get_blob(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, Response> {
    let blob_id = BlobId(id);
    let data = state.blobs.get(&blob_id).map_err(err_response)?;

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/octet-stream")
        .body(Body::from(data))
        .unwrap())
}

// ---------------------------------------------------------------------------
// DELETE /v1/blobs/:id
// ---------------------------------------------------------------------------

pub async fn delete_blob(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<StatusCode, Response> {
    let blob_id = BlobId(id);
    if !state.blobs.has(&blob_id).map_err(err_response)? {
        return Err(err_response(sodl_core::SodlError::NotFound));
    }
    if state.index.get_blob(&blob_id).map_err(err_response)? > 0 {
        return Err((
            StatusCode::CONFLICT,
            Json(ErrorResponse {
                error: format!(
                    "blob is still referenced and cannot be deleted: {}",
                    blob_id.0
                ),
            }),
        )
            .into_response());
    }
    state.blobs.delete(&blob_id).map_err(err_response)?;
    Ok(StatusCode::NO_CONTENT)
}

// ---------------------------------------------------------------------------
// GET /v1/origins/:id/payload  (transparent chunk reassembly + decryption)
// ---------------------------------------------------------------------------

pub async fn get_payload(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, Response> {
    let oid = parse_origin_id(&id)?;
    let svc = state.service();
    let data = svc.get_payload(oid).map_err(err_response)?;

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/octet-stream")
        .body(Body::from(data))
        .unwrap())
}

// ---------------------------------------------------------------------------
// POST /v1/shares
// ---------------------------------------------------------------------------

pub async fn create_share(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ShareRequest>,
) -> Result<(StatusCode, Json<ShareResponse>), Response> {
    let oid = parse_origin_id(&req.origin_id)?;
    let caps: Vec<Capability> = req
        .capabilities
        .iter()
        .map(|c| parse_capability(c))
        .collect();

    let svc = state.service();
    let share_id = svc
        .share(PrincipalId(req.from), PrincipalId(req.to), oid, caps)
        .map_err(err_response)?;

    Ok((
        StatusCode::CREATED,
        Json(ShareResponse {
            share_id: share_id.0,
            origin_id: oid.0.to_string(),
        }),
    ))
}

// ---------------------------------------------------------------------------
// GET /v1/shares/:id
// ---------------------------------------------------------------------------

pub async fn get_share(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<ShareDetailResponse>, Response> {
    let sid = sodl_core::ShareId(id);
    let s = state.shares.get(&sid).map_err(err_response)?;

    Ok(Json(ShareDetailResponse {
        share_id: s.share_id.0,
        origin_id: s.origin_id.0.to_string(),
        from: s.from_principal.0,
        to: s.to_principal.0,
        capabilities: s
            .capabilities
            .iter()
            .map(|c| format_cap(c).to_string())
            .collect(),
        created_at: s.created_at.to_string(),
        lineage_proof_digest: s.lineage_proof_digest,
    }))
}

// ---------------------------------------------------------------------------
// DELETE /v1/shares/:id
// ---------------------------------------------------------------------------

pub async fn release_share(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<StatusCode, Response> {
    let sid = sodl_core::ShareId(id);
    let svc = state.service();
    svc.release_share(&sid).map_err(err_response)?;
    Ok(StatusCode::NO_CONTENT)
}

// ---------------------------------------------------------------------------
// POST /v1/shares/:id/verify
// ---------------------------------------------------------------------------

pub async fn verify_share(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<VerifyResponse>, Response> {
    let sid = sodl_core::ShareId(id.clone());
    let s = state.shares.get(&sid).map_err(err_response)?;

    let svc = state.service();
    let valid = svc.verify_share_proof(&s).map_err(err_response)?;

    Ok(Json(VerifyResponse {
        share_id: id,
        valid,
    }))
}

// ---------------------------------------------------------------------------
// POST /v1/derivations
// ---------------------------------------------------------------------------

pub async fn create_derivation(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DeriveRequest>,
) -> Result<(StatusCode, Json<DeriveResponse>), Response> {
    let oid = parse_origin_id(&req.origin_id)?;
    let media_kind = parse_media_kind(&req.media_kind);
    let kind = match req.kind.to_lowercase().as_str() {
        "trim" => DerivationKind::Trim {
            range: sodl_manifest::TimelineRange {
                start_ms: 0,
                end_ms: 0,
            },
        },
        "crop" => DerivationKind::Crop {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        },
        _ => DerivationKind::Transform {
            description: req.description.unwrap_or_else(|| "transform".into()),
        },
    };

    let svc = state.service();
    let did = svc.derive(oid, kind, media_kind).map_err(err_response)?;

    Ok((
        StatusCode::CREATED,
        Json(DeriveResponse {
            derivation_id: did.0,
            origin_id: oid.0.to_string(),
        }),
    ))
}

// ---------------------------------------------------------------------------
// POST /v1/pins
// ---------------------------------------------------------------------------

pub async fn create_pin(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PinRequest>,
) -> Result<(StatusCode, Json<PinResponse>), Response> {
    let oid = parse_origin_id(&req.origin_id)?;

    let svc = state.service();
    let pin_id = svc
        .pin_origin(PrincipalId(req.requested_by), oid, req.min_replicas)
        .map_err(err_response)?;

    Ok((
        StatusCode::CREATED,
        Json(PinResponse {
            pin_id,
            origin_id: oid.0.to_string(),
        }),
    ))
}

// ---------------------------------------------------------------------------
// DELETE /v1/pins/:id
// ---------------------------------------------------------------------------

pub async fn release_pin(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<StatusCode, Response> {
    let svc = state.service();
    svc.unpin(&id).map_err(err_response)?;
    Ok(StatusCode::NO_CONTENT)
}

// ---------------------------------------------------------------------------
// GET /v1/origins/:id/lineage-proof
// ---------------------------------------------------------------------------

pub async fn lineage_proof(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<LineageProofResponse>, Response> {
    let oid = parse_origin_id(&id)?;
    let svc = state.service();
    let proof = svc.lineage_proof(oid).map_err(err_response)?;

    Ok(Json(LineageProofResponse {
        origin_id: proof.origin_id.0.to_string(),
        digest: proof.digest,
        created_at: proof.created_at.to_string(),
    }))
}
