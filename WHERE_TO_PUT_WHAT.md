# Where to Put What (SODL V1)

This file is your quick navigation guide.

## Shared types / IDs / errors

**Path:** `crates/sodl-core/src/lib.rs`

Put anything that should be used across crates here:

- `OriginId`, `BlobId`, `DerivationId`, `ShareId`
- `PrincipalId`, `Capability`, `MediaKind`
- shared error type `SodlError`

## Content addressed storage

**Path:** `crates/sodl-cas/src/lib.rs`

Put:

- hash computation, chunking strategy, Merkle/outboard structures (later)
- `BlobStore` trait + backend implementations (later)

## Origin registry + keys

**Path:** `crates/sodl-origin/src/lib.rs`

Put:

- `OriginRecord`, `Representation`
- registry interface implementations (later)
- encryption/key manager adapters (later)

## Manifests + lineage graph schema

**Path:** `crates/sodl-manifest/src/lib.rs`

Put:

- derivation kinds and manifest schema
- share schema
- lineage node/edge schema
- schema validation helpers

## Policy engine

**Path:** `crates/sodl-policy/src/lib.rs`

Put:

- durability & retention logic
- authorization policy logic
- garbage collection planning rules (later)

## Retrieval pipeline

**Path:** `crates/sodl-fetch/src/lib.rs`

Put:

- source ordering rules
- caching behavior
- integrity + policy enforcement hooks
- future: fingerprint fallback lookup

## Fingerprinting / watermarking interfaces

**Path:** `crates/sodl-fingerprint/src/lib.rs`

Put:

- perceptual fingerprinting traits
- watermark detector traits
- fingerprint index/search traits

## Distribution / peers

**Path:** `crates/sodl-dist/src/lib.rs`

Put:

- peer discovery interfaces
- blob exchange client/server (later)
- relay fallback hooks (later)

## Cryptography boundaries (encryption + key envelopes)

**Path:** `crates/sodl-crypto/src/lib.rs`

Put:

- key envelope schema (`KeyEnvelope`)
- crypto trait boundaries (`Encryptor`, `Decryptor`, `KeyManager`)
- later: concrete implementations / adapters

## Durability / Pinning / GC

**Path:** `crates/sodl-policy/src/lib.rs`

Put:

- `PinRecord`, `PinStore`, `PinPlanner`
- durability enforcement rules
- GC eligibility logic

## Durable stores + replica tracking

**Path:** `crates/sodl-store/src/lib.rs`

Put:

- durable store boundaries (`DurableStore`)
- replica tracking models (`ReplicaRecord`, `ReplicaTracker`)
- pin satisfaction helpers (`ReplicaPlanner`)

## Encrypted CAS helper

**Path:** `crates/sodl-store/src/lib.rs`

Put:

- `EncryptedCas` wrapper (encrypt -> hash(ciphertext) -> store)

## High-level facade (application layer)

**Path:** `crates/sodl-service/src/lib.rs`

Put:

- `SodlService` facade methods: upload/share/derive/pin
- in-memory reference stores for metadata

## Refcount + lineage

**Path:** `crates/sodl-index/src/lib.rs`

Put:

- refcount interfaces (`RefCounter`)
- lineage edges (`LineageEdge`)
- future: persistent implementations (Postgres/Redis)

## Policy-aware GC + tombstones

**Path:** `crates/sodl-gc/src/lib.rs`

Put:

- tombstone models + store
- GC planner/executor
- safety rules

## Origin lifecycle timestamps

**Path:** `crates/sodl-origin/src/lib.rs`

Added:

- `created_at`
- `tombstoned_at`
- `tombstone_reason`

## Tombstone-first origin deletion

**Path:** `crates/sodl-service/src/lib.rs`

Added:

- `tombstone_origin(origin_id, reason)`

## Replica tracking

**Path:** `crates/sodl-replica/src/lib.rs`

Put:

- replica records
- health tracking
- placement information

## Durability auditing

**Path:** `crates/sodl-gc/src/lib.rs`

Added:

- `ReplicaAuditor`
- `RepairPlan`

## Replica repair execution

**Path:** `crates/sodl-replica/src/lib.rs`

Added:

- `StoreMesh`
- `ReplicaExecutor`
- `MemStoreMesh`

## Durability-safe GC

**Path:** `crates/sodl-gc/src/lib.rs`

Added:

- `DurabilityGate`
- `ReplicaAuditor` stale replica handling

## AI Weight Store (SODL-Weight)

**Path:** `crates/sodl-store/src/weight_store.rs`

Put:

- `WeightBlobStore` — compress → encrypt → CAS pipeline for weight clusters
- `WeightPinRegistry` — hot/cold RAM cache with refcount-based GC eviction
- Compression helpers (zstd), serialisation helpers (JSON)

Core types live in `crates/sodl-core/src/lib.rs`:

- `WeightCluster`, `ClusterId`, `WeightOrigin`, `WeightPinReason`

## Step 22 semantic modules
**Paths:** `crates/sodl-semantic-color`, `crates/sodl-semantic-cube`

Added:
- color-spectrum semantic code artifact
- ruby-cube lattice semantic artifact
- Step 22 guides under `guides/`
