# SODL V1 – Distribution, Retrieval & Replica Tracking Spec (Skeleton)

This document defines SODL’s retrieval contract and how distribution layers interact with durability.

## Retrieval pipeline

The default retrieval order is:

1. **Local cache** (fastest, non-durable)
2. **Peers** (optional acceleration)
3. **Edge provider** (optional CDN-like cache)
4. **Durable stores** (authoritative for pinned/durable content)

This ordering is policy-driven. Implementations may reorder based on:
- content type (video/audio may prefer edge)
- locality (same region peers first)
- security/compliance (only verified sources)

## Fetch sources

A FetchSource returns either:
- `Some(bytes)` if found, or
- `None` if it cannot provide the blob.

The fetch pipeline:
- verifies integrity via the blob hash
- writes-through into local cache on success

## Peers vs Durable

Peers and edges are **not** assumed durable.

Durable content is guaranteed only when:
- pins exist, and
- replica targets are satisfied across durable stores/zones.

## Replica tracking

Replica tracking records where blobs exist:

- `ReplicaRecord(blob_id, zone, store_name, observed_at)`

Replica tracker is used by pin planners to determine:
- if a pin is satisfied
- which stores need replication work

## Out of scope for V1 skeleton

- Real transport protocols (QUIC/WebRTC/TCP)
- NAT traversal / relay infrastructure
- Active replication scheduler
- Health checks and anti-entropy
