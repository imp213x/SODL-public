"""Weight compression demo — store and reconstruct embedding clusters.

Usage:
    python examples/weight_compression.py
"""

import tempfile
import numpy as np
from sodl_weights import WeightStoreService, WeightCluster

# Create a temporary weight store
with tempfile.TemporaryDirectory() as tmpdir:
    service = WeightStoreService(tmpdir)

    # Register a model
    origin = service.create_model("demo-model-v1", "float32")
    print(f"Model origin: {origin.origin_id}")

    # Simulate an embedding matrix (vocab=100, dim=32)
    vocab_size, dim = 100, 32
    np.random.seed(42)
    embeddings = np.random.randn(vocab_size, dim).astype(np.float32)

    # Cluster tokens into groups (simulate K-means result)
    n_clusters = 10
    cluster_size = vocab_size // n_clusters

    total_raw = 0
    total_stored = 0
    blob_ids = []

    for k in range(n_clusters):
        start = k * cluster_size
        end = start + cluster_size
        member_ids = list(range(start, end))
        member_embeddings = embeddings[start:end]

        # Centroid = mean of cluster members
        centroid = member_embeddings.mean(axis=0)

        # Offsets = residuals from centroid
        offsets = (member_embeddings - centroid).tolist()

        cluster = WeightCluster(
            centroid=centroid.tolist(),
            member_token_ids=member_ids,
            offsets=offsets,
            dim=dim,
        )

        stats = service.store_cluster(origin.origin_id, cluster)
        blob_ids.append(stats.blob_id)
        total_raw += stats.raw_bytes
        total_stored += stats.stored_bytes

    compression_ratio = (1 - total_stored / total_raw) * 100
    print(f"Stored {n_clusters} clusters")
    print(f"Raw: {total_raw:,} bytes → Stored: {total_stored:,} bytes")
    print(f"Compression: {compression_ratio:.1f}%")

    # Reconstruct embeddings from clusters
    reconstructed = np.zeros_like(embeddings)
    for blob_id in blob_ids:
        cluster = service.load_cluster(origin.origin_id, blob_id)
        centroid = np.array(cluster.centroid, dtype=np.float32)
        for i, token_id in enumerate(cluster.member_token_ids):
            offset = np.array(cluster.offsets[i], dtype=np.float32)
            reconstructed[token_id] = centroid + offset

    # Verify reconstruction accuracy
    mse = np.mean((embeddings - reconstructed) ** 2)
    print(f"Reconstruction MSE: {mse:.6e}")
    print(f"Cache size: {service.cache_size()} clusters cached")
