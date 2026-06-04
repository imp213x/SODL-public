"""Token hash index demo — clustering and hierarchical softmax.

Usage:
    pip install sodl[clustering]
    python examples/token_hash_index.py
"""

import numpy as np
from sodl_weights import TokenHashIndex

# Simulate a vocabulary embedding matrix
vocab_size = 2048
dim = 64
np.random.seed(42)

# Create embeddings with some natural clustering structure
# (4 language domains with distinct characteristics)
embeddings = np.zeros((vocab_size, dim), dtype=np.float32)
tokens_per_group = vocab_size // 4
for group in range(4):
    start = group * tokens_per_group
    end = start + tokens_per_group
    center = np.random.randn(dim).astype(np.float32) * 3  # cluster center
    noise = np.random.randn(tokens_per_group, dim).astype(np.float32) * 0.5
    embeddings[start:end] = center + noise

# Build the token hash index with adaptive clustering
index = TokenHashIndex(n_clusters=45)  # sqrt(2048) ≈ 45
stats = index.build(embeddings, adaptive=True)

print("=== Token Hash Index Stats ===")
print(f"Clusters: {stats['n_clusters']}")
print(f"Vocab size: {stats['vocab_size']}")
print(f"Dimension: {stats['dim']}")
print(f"Compression potential: {stats['compression_potential']:.1%}")
print(f"Silhouette score: {stats['silhouette_score']:.3f}")
print(f"Cluster size range: {stats['cluster_size_min']} – {stats['cluster_size_max']}")
print(f"Adaptive rebalanced: {stats['adaptive_rebalanced']}")
print()

# Hierarchical softmax
hidden = np.random.randn(dim).astype(np.float32)
results = index.hierarchical_softmax(hidden)

print("=== Hierarchical Softmax (top 5) ===")
for r in results[:5]:
    print(f"  Token {r.token_id:4d} | p={r.probability:.4f} | "
          f"cluster={r.cluster_id} (cp={r.cluster_prob:.4f})")
print()

# Compute savings for a training batch
batch_tokens = list(range(50))  # first 50 tokens in batch
savings = index.compute_update_savings(batch_tokens)
print("=== Training Savings ===")
print(f"Unique tokens: {savings['unique_tokens']}")
print(f"Active clusters: {savings['active_clusters']}")
print(f"Savings ratio: {savings['savings_ratio']:.1%}")
print(f"Speedup factor: {savings['speedup_factor']:.1f}x")
print()

# Gradient sharing
cluster_id = index.token_hash(0)
token_grads = {
    tok: np.random.randn(dim).astype(np.float32)
    for tok in index.cluster_members(cluster_id)[:5]
}
centroid_grad, offset_corrections = index.compute_shared_gradient(cluster_id, token_grads)
print(f"Gradient sharing: {len(token_grads)} token grads → "
      f"1 centroid grad + {len(offset_corrections)} sparse corrections")

# Persistence
import tempfile, os
with tempfile.TemporaryDirectory() as tmpdir:
    index.save(tmpdir)
    loaded = TokenHashIndex.load(tmpdir)
    # Verify roundtrip
    for token_id in [0, 100, 500, 1000]:
        assert index.token_hash(token_id) == loaded.token_hash(token_id)
    print("Index save/load roundtrip verified ✓")
