# SODL V1 -- Architecture Specification

## 1. System Overview

SODL (Single Origin Distributed Lineage) is a content-addressed,
lineage-aware, policy-governed storage system.

Core principle:

> Store bytes once. Reference them many times. Delete safely and
> deterministically.

SODL separates:

-   Logical objects (Origins)
-   Physical bytes (Blobs)
-   References (Shares, Derivations, Pins)
-   Storage lifecycle (Garbage Collection)

------------------------------------------------------------------------

## 2. Core Primitives

### Origin

A logical object created by a user (e.g., video, document, binary).
Origins are metadata objects.

### Blob

A content-addressed binary object identified by: hash(ciphertext) Blobs
are immutable and represent physical storage units.

### Representation

A mapping between an Origin and one or more root blobs.

### Share

A reference edge granting another principal access to an Origin. Does
NOT duplicate blob storage.

### Derivation

A transformation of an Origin (e.g., trim, transcode). May introduce new
blobs.

### Pin

A durability intent preventing GC from reclaiming content prematurely.

------------------------------------------------------------------------

## 3. Reference Counting Model

SODL maintains: - Origin refcount - Blob refcount

Deletion eligibility is determined by these counters.

------------------------------------------------------------------------

## 4. Garbage Collection (GC)

GC in SODL is policy-aware storage reclamation.

It: - Detects unreachable Origins - Detects unreachable Blobs - Respects
Pins - Respects Retention Policies - Writes Tombstones - Deletes bytes
when eligible

Planner/Executor separation enables safety and auditability.

------------------------------------------------------------------------

## 5. Deletion Model (Choice B)

Blob deletion requires: blob_refcount == 0

Origin tombstoning requires: origin_refcount == 0 AND no active pins AND
retention policy satisfied

Origins are metadata. Blobs are physical storage units.

------------------------------------------------------------------------

## 6. Tombstones

Tombstones are durable deletion records ensuring: - Audit trail -
Compliance safety - Resurrection prevention

Deletion is tombstone-first.

------------------------------------------------------------------------

## 7. Durable Storage Requirement

SODL requires at least one durable storage anchor. The internet may
assist distribution but cannot replace durability.

------------------------------------------------------------------------

## 8. System Guarantees

-   No duplicate storage on share
-   Deterministic blob identity
-   Policy-bound deletion
-   Audit-capable lineage
-   Blob-level GC correctness

------------------------------------------------------------------------

## 9. Future Extensions

-   TTL enforcement
-   Replica health validation
-   Chunked storage
-   Cross-origin deduplication
-   Distributed GC workers
