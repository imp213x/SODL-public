# SODL V1 – Durability, Pinning, Retention & GC Spec

SODL’s core claim is **store-once with lineage**, but reliability requires an explicit durability model.
This document defines the **behavioral contract** for durability classes and the mechanisms that enforce them.

## Key terms

- **Blob**: immutable bytes stored in CAS.
- **Origin**: stable identity (OriginId) that groups blobs and derivations.
- **Reference**: metadata edge that points to a blob/origin (shares, derivations, representations).
- **Pin**: a policy-controlled statement that *some durable store* must retain the bytes.
- **Replica**: an independent durable copy of a blob (or origin representation) in a distinct store/zone.

## Durability classes

### 1) Ephemeral
**Intent:** short-lived sharing (e.g., “send once”, temporary links).

**Contract:**
- Bytes MAY be dropped at any time after TTL.
- No minimum replica guarantee.
- Retrieval is best-effort: cache/peers/edge may help, but absence is acceptable.

**Required policy fields:**
- `ttl_seconds` MUST be set.
- `min_replicas` SHOULD be 0 or None.

### 2) BestEffort
**Intent:** inexpensive storage where loss is acceptable, but the system tries to keep content around.

**Contract:**
- System SHOULD keep at least 1 durable copy if feasible.
- System MAY delete if space pressure / quotas / policy triggers.
- TTL optional; quotas can override.

**Required policy fields:**
- `ttl_seconds` optional
- `min_replicas` typically 1 (soft target)

### 3) Durable (Pinned)
**Intent:** reliability and integrity guarantee.

**Contract:**
- System MUST ensure at least `min_replicas` durable copies exist.
- Content MUST NOT be deleted while pinned and referenced.
- Deletion only occurs after:
  - pin removed / retention allows deletion, AND
  - no remaining references exist (or explicit forced delete with admin capability).

**Required policy fields:**
- `min_replicas` MUST be >= 1

## Pinning model

Pins are first-class objects.

A pin applies to:
- an entire **origin** (recommended), or
- a specific **representation** of an origin (e.g., `hls_720p`).

### Pin states
- `pending`: requested but not fully satisfied (replicas not yet met)
- `active`: requirements satisfied
- `released`: removed; content may be collected if unreferenced

## Retention vs References vs Pins

SODL deletes bytes ONLY when *all* are true:
1. No active pins require retention, AND
2. Retention policy permits deletion (TTL expired or BestEffort reclaimed), AND
3. No strong references remain (refcount == 0), unless forced by admin.

## Replica placement & zones (future)

Replicas should be placed across failure domains described by `StorageZone` identifiers.

## Garbage collection (GC)

An origin is eligible if:
- `pin_count == 0`, AND
- durability/TTL/reclaim rules permit deletion, AND
- `refcount == 0` unless forced.

Two-phase delete is recommended:
1. Mark -> `deleting`
2. Delete bytes
3. Tombstone/finalize

## Out of scope for V1 skeleton
- Concrete storage backends
- Replica placement logic
- Pin satisfaction workers/scheduler
- Quota enforcement
