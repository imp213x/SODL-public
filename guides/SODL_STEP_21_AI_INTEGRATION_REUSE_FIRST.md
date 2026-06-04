# SODL Step 21 — AI Integration (Reuse‑First Artifacts)

**Goal (Step 21):** Reduce AI/RAG latency + CPU/RAM churn by **reusing** dataset artifacts (chunk maps, embeddings, ANN indexes) via SODL’s **store-once/share-many** primitives, **without** introducing compression/chunked-CAS research in this step.

This step is **model-agnostic** (works for Carla today, another model tomorrow). Carla-specific wiring stays in *adapters/examples*, not in SODL core.

---

## What Step 21 Adds

### 1) AI Artifact Registry (metadata)
Introduce a small “AI artifacts” layer that stores *references* to artifacts already stored as SODL blobs.

Artifacts we care about:

1. **Dataset** (Origin)
   - Canonical bytes (e.g., `.jsonl`, `.parquet`, `.tar.zst`) uploaded once.
2. **Chunk Map** (Derivation Manifest)
   - How the dataset was chunked (boundaries, chunk ids, normalization rules).
3. **Embeddings** (Derivation Manifest + output blob)
   - Vectors blob + metadata (embedder id, dims, normalization, pooling).
4. **ANN Index** (Derivation Manifest + output blob(s))
   - Index type + parameters + mapping to chunk ids.

SODL already supports “metadata over mutation”; Step 21 formalizes artifact typing + lookup keys.

### 2) Reuse‑First Resolver API
Add a resolver that **returns an existing artifact if it exists**, otherwise computes and stores it.

Required behavior:
- Lookup key is deterministic: **(dataset_origin_id + pipeline_hash + artifact_kind)**.
- If found and valid → return immediately.
- If not found → build artifact → store blobs → write manifests → pin → return.

### 3) Cache Strategy for Runtime
Step 21 focuses on *latency*:
- Keep small manifests and lookup results in memory (LRU).
- Allow an optional on-disk cache for fetched blobs (embeddings/index) to speed restarts.
- Pin datasets/indexes; apply TTL to transient traces.

---

## Design: Pipeline Hash (the key to reuse)

Define a stable deterministic hash over:

- `dataset_origin_id`
- `chunker_config` (chunk_size, overlap, tokenizer_id, normalization rules)
- `embedder_config` (model_id, dims, pooling, normalization)
- `index_config` (index_type, params)

**Rule:** changing any of the above must change the pipeline hash.

**Implementation recommendation:**
- Serialize config to canonical JSON (stable key ordering) or an explicit Rust struct → bytes.
- Hash with blake3 → hex string.

---

## Storage Model in SODL

### Dataset
- Stored as **Origin**
- Owner: service principal (e.g., `system:carla`) or dataset owner
- Must be **pinned** (or policy says durable retention)

### Chunk Map / Embeddings / Index
- Stored as **DerivationManifest** under the same origin lineage
- If artifact produces bytes, store bytes as blob(s) and list in `output_blobs`

---

## Minimal Public API Surface for Step 21

Implement as either:
- a new crate `sodl-ai` (preferred), OR
- a module in `sodl-service` (acceptable for internal-only)

### Types

```text
ArtifactKind = Dataset | ChunkMap | Embeddings | AnnIndex

ArtifactRef {
  kind: ArtifactKind,
  dataset_origin_id: OriginId,
  pipeline_hash: String,
  derivation_id: Option<DerivationId>,
  output_blobs: Vec<BlobId>,
  created_at: OffsetDateTime,
}
```

### Resolver functions (example)

```text
ensure_dataset(bytes) -> OriginId
ensure_chunk_map(dataset_origin_id, chunker_cfg) -> ArtifactRef
ensure_embeddings(dataset_origin_id, chunk_map_ref, embedder_cfg) -> ArtifactRef
ensure_ann_index(dataset_origin_id, embeddings_ref, index_cfg) -> ArtifactRef
```

The resolver depends on:
- `sodl-service` for upload/derive/share/pin
- a small `AiArtifactStore` to persist `ArtifactRef` mappings

---

## Acceptance Criteria (Step 21 PASS)

### A) Reuse test
1. Ingest a dataset once.
2. Build embeddings + index once.
3. Restart service (or rebuild resolver).
4. Call `ensure_*` again with same configs.
5. **Expected:** no rebuild; resolver returns same artifact refs (same pipeline hash; same output blob ids).

### B) Integrity test
1. Fetch the stored artifact blob(s) from SODL.
2. Verify blob integrity using existing CAS hash verification.
3. **Expected:** integrity ok.

### C) Policy/pinning test
1. Attempt GC.
2. **Expected:** pinned dataset/index are not deleted.

---

## Implementation Notes

- Do **not** introduce compression/chunked-CAS in Step 21.
- Use existing blob store APIs.
- Prefer streaming ingestion for multi‑GB datasets later (Step 22+). Step 21 can validate reuse using small sample data.

---

## Suggested Example (local)

Add `crates/sodl-service/examples/ai_reuse_first_demo.rs` that:

1. Uploads a small dataset (bytes)
2. Computes a pipeline hash
3. Calls `ensure_embeddings` + `ensure_ann_index`
4. Calls them again (should reuse)
5. Prints whether artifact IDs/blobs match

Expected:

```text
STEP21 PASS ✅: reuse-first returned existing embeddings + index
```
