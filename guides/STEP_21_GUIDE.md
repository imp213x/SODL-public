# Step 21 Guide — AI Integration (Reuse‑First)

This guide is the practical checklist to implement and validate **Step 21** on your machine.

## Objective
Make AI/RAG pipelines **reuse** dataset artifacts (chunk map, embeddings, ANN index) so Carla doesn’t keep rebuilding them—reducing cold-start time, CPU spikes, and retrieval latency.

## Where to put files

- Specs live in: `guides/`
- Code lives in: `crates/`
- Demos live in: `crates/sodl-service/examples/`

## Minimal changes expected

1) Add an artifact registry store (in-memory for now is OK)
2) Add a resolver that implements “reuse-first”
3) Add an example demo + a small test

## PASS Test (what you run)

### 1) Build + run the Step 21 demo

```powershell
cd C:\SODL

# Demo should: upload dataset -> ensure embeddings+index -> call ensure again -> reuse
cargo run -p sodl-service --example ai_reuse_first_demo
```

### Expected output

You must see:

- the first run creates artifacts
- the second run reuses artifacts

Example expected markers:

```text
created embeddings artifact: <id>
created index artifact: <id>
reused embeddings artifact: <same id>
reused index artifact: <same id>
STEP21 PASS ✅: reuse-first returned existing embeddings + index
```

### 2) Optional: prove restart reuse

If your registry is persisted (file/db), restart the process and rerun the demo.
If it’s in-memory only, skip this until persistence is added.

## Notes

- Step 21 does **not** attempt to shrink blobs (compression/chunked-CAS is future).
- Step 21 is about: **artifact reuse** + **pinning** + **integrity**.

---

## Troubleshooting

### Artifact always rebuilds
Check:
- pipeline hash is deterministic
- lookup uses `(origin_id + pipeline_hash + kind)`
- registry writes happen before returning

### GC deletes artifacts
Check:
- datasets/indexes are pinned
- GC respects pin/policy

---

## Done criteria

Step 21 is complete when:

- demo passes and consistently reuses artifacts
- outputs are integrity-verified (CAS)
- GC does not delete pinned artifacts

---

## Next

Step 22+ can add:
- streaming ingestion for multi‑GB datasets
- chunked CAS / delta / compression research
