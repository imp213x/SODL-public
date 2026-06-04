"""SODL — Content-addressed, lineage-aware data framework for AI systems.

Provides:
    - Content-addressed blob storage with blake3 hashing and zstd compression
    - Weight cluster storage with centroid + residual compression
    - Hierarchical softmax for O(K + V/K) training acceleration
    - Token hash indexing with K-means clustering and gradient sharing
    - Model lifecycle management with lineage tracking
    - Pin-based cache management with configurable eviction
    - Pluggable encryption (null, XOR, or custom providers)

Quick Start::

    from sodl_weights import BlobStore, WeightBlobStore, WeightStoreService

    # Create a blob store
    store = BlobStore("/path/to/blobs")
    weight_store = WeightBlobStore(store)

    # Or use the high-level service
    service = WeightStoreService("/path/to/blobs")
    origin = service.create_model("my-model", "float32")

Install::

    pip install sodl              # core (numpy, zstandard, blake3)
    pip install sodl[torch]       # + PyTorch (ClusteredSoftmaxLoss)
    pip install sodl[clustering]  # + scikit-learn (adaptive clustering)
    pip install sodl[all]         # everything
"""

__version__ = "0.2.0"

# --- Core storage ---
from sodl_weights.store import BlobStore, WeightBlobStore, compute_blob_id, verify_integrity
from sodl_weights.types import WeightCluster, StoreStats, ImportSummary, WeightPinReason, WeightOrigin

# --- Service layer ---
from sodl_weights.service import WeightStoreService
from sodl_weights.pin_registry import WeightPinRegistry
from sodl_weights.client import RemoteBlobStore, SODLClient, SodlClientError

# --- Crypto ---
from sodl_weights.crypto import AEADCryptoProvider, NullCrypto, XorCrypto, CryptoProvider
from sodl_weights.proof import (
    Ed25519ProofSigner,
    LineageProof,
    ProofSigner,
    generate_lineage_proof,
    sign_lineage_proof,
    verify_lineage_digest,
    verify_lineage_signature,
)

# --- Training acceleration ---
from sodl_weights.token_hash import TokenHashIndex

# --- Pipeline ---
from sodl_weights.pipeline import compute_pipeline_hash

# --- Model lifecycle ---
from sodl_weights.model_registry import ModelRegistry

# --- Phase B: Generic data storage ---
from sodl_weights.artifact_store import ArtifactStore, ArtifactMetadata
from sodl_weights.dataset import SODLDataset
from sodl_weights.checkpoint import CheckpointManager, CheckpointRecord
from sodl_weights.optimizer_state import (
    OptimizerBlockRecord,
    OptimizerCacheStats,
    OptimizerStateManifest,
    OptimizerStateStore,
    OptimizerStoreResult,
)
from sodl_weights.weight_manifest import (
    WeightManifest,
    WeightManifestCluster,
    WeightManifestStore,
    export_manifest_clusters,
)
from sodl_weights.vector_index import (
    SODLVectorIndex,
    VectorIndexManifest,
    VectorIndexShard,
    VectorSearchResult,
)
from sodl_weights.data_quality import DataQualityScorer, QualityRecord
from sodl_weights.context_logic import (
    ClusteredAttentionLayer,
    SCLClusteredDecoder,
    SCLMemoryManifest,
    SCLQueryResult,
    SCLTokenPrediction,
    SemanticContextLogic,
)
from sodl_weights.training_lifecycle import (
    ArteryPulsar,
    VeinPrefetcher,
    build_weight_cluster,
    build_weight_clusters,
    cluster_ids_for_token_batch,
    export_token_clusters,
    load_sodl_manifest,
    resolve_sodl_origin_id,
    write_sodl_manifest,
)

# --- Phase C: Performance ---
from sodl_weights.async_store import AsyncBlobStore
from sodl_weights.batch import BatchOps, BatchResult
from sodl_weights.mmap_store import MMapBlobReader, ArenaReader
from sodl_weights.streaming import StreamingCompressor, StreamingDecompressor
from sodl_weights._rust_bridge import RustBridgeStatus, status as rust_bridge_status, status_summary as rust_bridge_summary

# --- Phase D: Distributed ---
from sodl_weights.registry import NodeInfo, NodeRegistry
from sodl_weights.replication import ReplicationEngine, ReplicationPolicy
from sodl_weights.consistency import ConsistencyChecker
from sodl_weights.federation import FederationManager, FederationConfig

# --- Phase E: Observability ---
from sodl_weights.metrics import MetricsCollector, get_metrics
from sodl_weights.health import HealthMonitor, HealthStatus
from sodl_weights.tracer import OperationTracer

# --- Semantic (experimental) ---
from sodl_weights.semantic_bridge import SemanticColor32, LatticePoint3
from sodl_weights.semantic_router import CapabilityQuery, RouteCandidate, simple_route

# --- Lazy imports for optional torch dependency ---
def _get_clustered_softmax():
    """Import ClusteredSoftmaxLoss (requires PyTorch)."""
    from sodl_weights.clustered_softmax import (
        ClusteredSoftmaxLoss,
        build_index_from_model,
        create_clustered_loss,
    )
    return ClusteredSoftmaxLoss, build_index_from_model, create_clustered_loss


def _get_offload_optimizer():
    """Import SODLAdamW (requires PyTorch)."""
    from sodl_weights.offload_optimizer import SODLAdamW

    return SODLAdamW


def __getattr__(name: str):
    if name in {"ClusteredSoftmaxLoss", "build_index_from_model", "create_clustered_loss"}:
        ClusteredSoftmaxLoss, build_index_from_model, create_clustered_loss = _get_clustered_softmax()
        mapping = {
            "ClusteredSoftmaxLoss": ClusteredSoftmaxLoss,
            "build_index_from_model": build_index_from_model,
            "create_clustered_loss": create_clustered_loss,
        }
        return mapping[name]
    if name == "SODLAdamW":
        return _get_offload_optimizer()
    raise AttributeError(f"module 'sodl_weights' has no attribute {name!r}")


__all__ = [
    # Version
    "__version__",
    # Core storage
    "BlobStore",
    "WeightBlobStore",
    "compute_blob_id",
    "verify_integrity",
    # Types
    "WeightCluster",
    "StoreStats",
    "ImportSummary",
    "WeightPinReason",
    "WeightOrigin",
    # Service
    "WeightStoreService",
    "WeightPinRegistry",
    "SODLClient",
    "RemoteBlobStore",
    "SodlClientError",
    # Crypto
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
    # Training
    "TokenHashIndex",
    "ClusteredSoftmaxLoss",
    "build_index_from_model",
    "create_clustered_loss",
    # Pipeline
    "compute_pipeline_hash",
    # Model lifecycle
    "ModelRegistry",
    # Generic data storage (Phase B)
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
    "WeightManifest",
    "WeightManifestCluster",
    "WeightManifestStore",
    "export_manifest_clusters",
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
    "ArteryPulsar",
    "VeinPrefetcher",
    "build_weight_cluster",
    "build_weight_clusters",
    "cluster_ids_for_token_batch",
    "export_token_clusters",
    "load_sodl_manifest",
    "resolve_sodl_origin_id",
    "write_sodl_manifest",
    # Performance (Phase C)
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
    # Distributed (Phase D)
    "NodeInfo",
    "NodeRegistry",
    "ReplicationEngine",
    "ReplicationPolicy",
    "ConsistencyChecker",
    "FederationManager",
    "FederationConfig",
    # Observability (Phase E)
    "MetricsCollector",
    "get_metrics",
    "HealthMonitor",
    "HealthStatus",
    "OperationTracer",
    # Semantic (experimental)
    "SemanticColor32",
    "LatticePoint3",
    "CapabilityQuery",
    "RouteCandidate",
    "simple_route",
]
