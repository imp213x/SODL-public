"""Top-level SODL Python package with lazy re-exports."""

from __future__ import annotations

from importlib import import_module, metadata

_EXPORTS = [
    "__version__",
    "BlobStore",
    "WeightBlobStore",
    "compute_blob_id",
    "verify_integrity",
    "WeightCluster",
    "StoreStats",
    "ImportSummary",
    "WeightPinReason",
    "WeightOrigin",
    "WeightStoreService",
    "WeightPinRegistry",
    "SODLClient",
    "RemoteBlobStore",
    "SodlClientError",
    "NullCrypto",
    "XorCrypto",
    "AEADCryptoProvider",
    "CryptoProvider",
    "LineageProof",
    "ProofSigner",
    "Ed25519ProofSigner",
    "generate_lineage_proof",
    "sign_lineage_proof",
    "verify_lineage_digest",
    "verify_lineage_signature",
    "TokenHashIndex",
    "ClusteredSoftmaxLoss",
    "build_index_from_model",
    "create_clustered_loss",
    "compute_pipeline_hash",
    "ModelRegistry",
    "ArtifactStore",
    "ArtifactMetadata",
    "SODLDataset",
    "CheckpointManager",
    "CheckpointRecord",
    "OptimizerBlockRecord",
    "OptimizerCacheStats",
    "OptimizerStateManifest",
    "OptimizerStateStore",
    "OptimizerStoreResult",
    "SODLAdamW",
    "SODLVectorIndex",
    "VectorIndexManifest",
    "VectorIndexShard",
    "VectorSearchResult",
    "DataQualityScorer",
    "QualityRecord",
    "ClusteredAttentionLayer",
    "SCLClusteredDecoder",
    "SCLMemoryManifest",
    "SemanticContextLogic",
    "SCLQueryResult",
    "SCLTokenPrediction",
    "AsyncBlobStore",
    "BatchOps",
    "BatchResult",
    "MMapBlobReader",
    "ArenaReader",
    "StreamingCompressor",
    "StreamingDecompressor",
    "RustBridgeStatus",
    "rust_bridge_status",
    "rust_bridge_summary",
    "NodeInfo",
    "NodeRegistry",
    "ReplicationEngine",
    "ReplicationPolicy",
    "ConsistencyChecker",
    "FederationManager",
    "FederationConfig",
    "MetricsCollector",
    "get_metrics",
    "HealthMonitor",
    "HealthStatus",
    "OperationTracer",
    "SemanticColor32",
    "LatticePoint3",
    "CapabilityQuery",
    "RouteCandidate",
    "simple_route",
]

try:
    __version__ = metadata.version("sodl")
except metadata.PackageNotFoundError:
    __version__ = "0.2.0"

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name == "__version__":
        return __version__
    if name not in _EXPORTS:
        raise AttributeError(f"module 'sodl' has no attribute {name!r}")

    module = import_module("sodl_weights")

    return getattr(module, name)
