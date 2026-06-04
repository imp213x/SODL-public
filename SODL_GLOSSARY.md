# SODL Glossary

## Blob

Immutable content-addressed binary unit.

## Origin

Logical user-created object referencing blobs.

## Representation

Mapping from Origin to blobs.

## Share

Reference granting access without duplicating storage.

## Derivation

Transformation producing new representation/blobs.

## Pin

Durability lock preventing deletion.

## Refcount

Reachability counter determining deletion eligibility.

## Tombstone

Durable deletion marker for audit and safety.

## Garbage Collection (GC)

Policy-aware reclamation of unreachable storage.

## GC Planner

Determines eligible deletions.

## GC Executor

Performs deletion and tombstoning.

## Durable Anchor

Controlled storage layer ensuring availability.

## Weight Cluster

A group of semantically related weight vectors sharing a centroid.
Stored as a compressed, content-addressed blob.

## Weight Pin Registry

Hot/cold RAM cache for weight clusters with refcount-based eviction.
Identity clusters are always pinned; others are evicted by lowest access count.

## Cluster ID

Content-addressed identifier for a weight cluster blob (alias for BlobId).
