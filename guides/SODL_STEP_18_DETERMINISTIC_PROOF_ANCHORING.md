# SODL — Step 18 Technical Specification: Deterministic Lineage Proof Anchoring (Sequence-Based)

## 1. Problem Being Solved

Current model (Step 17):

- Proof digest is computed over lineage edges.
- Proof is signed.
- Proof is stored in `ShareRecord`.

**But:**

- Lineage graph evolves over time.
- Recomputing “current digest” does not match historical snapshot.
- Timestamp anchoring is vulnerable to clock drift, edge reordering, and non-deterministic filtering.

**We need:**

- A deterministic, replayable, unambiguous lineage snapshot model.

## 2. Core Concept: Monotonic Edge Sequence

Every lineage edge must have a `seq: u64`.

**Properties:**

- Strictly increasing.
- Assigned at insertion time.
- Unique per origin (not global).
- Never reused.
- Never rewritten.

This sequence number becomes the canonical anchor.

## 3. Updated LineageEdge Model

```rust
pub struct LineageEdge {
    pub edge_id: String,
    pub origin_id: OriginId,
    pub seq: u64,                // NEW in Step 18
    pub created_at: OffsetDateTime,
    pub kind: RefKind,
}
```

## 4. Sequence Allocation Rule

When inserting an edge:

1. Retrieve current max `seq` for that origin.
2. Increment by 1.
3. Assign to new edge.
4. Persist.

This must be atomic per origin. In-memory implementations should maintain a `HashMap<OriginId, u64>` for the latest sequence.

## 5. Proof Model Change

### 5.1 ShareRecord Update

Add `pub lineage_seq_cutoff: u64`. This proof covers all lineage edges where `seq <= lineage_seq_cutoff`.

### 5.2 New Proof Generation Logic

Replace timestamp anchoring with sequence anchoring.

```rust
fn lineage_proof_at_seq(&self, origin_id: OriginId, cutoff_seq: u64) -> Result<LineageProof> {
    let mut edges = self.lineage.list_edges_for_origin(origin_id)?;
    edges.retain(|e| e.seq <= cutoff_seq);
    generate_proof_unsigned(origin_id, edges, cutoff_seq)
}
```

## 6. Updated Share Flow

Inside `share()`:

- **Step A — Append share edge first**: Insert share edge with incremented `seq`.
- **Step B — Anchor proof**: `let cutoff_seq = edge.seq;`
- **Step C — Sign (if enabled)**: Sign the digest.
- **Step D — Store ShareRecord**: Store `lineage_seq_cutoff` and digest.

## 7. Verification Logic

```rust
pub fn verify_share_proof(&self, share: &ShareRecord) -> Result<bool> {
    let proof = self.lineage_proof_at_seq(
        share.origin_id,
        share.lineage_seq_cutoff,
    )?;

    if proof.digest != share.lineage_proof_digest {
        return Ok(false);
    }

    if let (Some(_kid), Some(sig)) = (&share.lineage_proof_key_id, &share.lineage_proof_sig_b64) {
        if let Some(signer) = self.proof_signer {
            return signer.verify_digest_b64(&share.lineage_proof_digest, sig);
        }
        return Ok(false);
    }

    Ok(true)
}
```

## 8. Determinism Requirements

When hashing edges:

- Sort edges by `seq`.
- Serialize deterministically.
- Hash input must include: `origin_id`, `seq`, `edge_type`, `principal_ids`, `derivation_ids` in canonical order.

## 9. Why Sequence > Timestamp

| Feature | Timestamp | Sequence |
| :--- | :--- | :--- |
| Clock Dependence | Dependent | Deterministic |
| Ordering | Reorderable | Strictly ordered |
| Distributed Safety | Ambiguous | Deterministic per origin |
| Stability | Vulnerable to skew | Stable |

## 10. Backwards Compatibility

If `lineage_seq_cutoff` exists → use sequence model.
If absent → fallback to legacy timestamp model (temporary support).

## 11. Testing Requirements

1. **Test 1 — Deterministic Replay**: Upload -> Share -> Derive -> Share again. Verify both shares succeed.
2. **Test 2 — Mutation Attempt**: Tamper with an edge; verify failure.
3. **Test 3 — Order Integrity**: Confirm sequence monotonicity.

## 12. Architectural Impact

This step transforms SODL from a signed snapshot system into a **deterministic, replayable provenance ledger** without blockchain complexity or consensus overhead.
