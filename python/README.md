# SODL Python SDK

**Content-addressed, lineage-aware data framework for AI systems.**

SODL provides versioned storage, weight compression, and training acceleration for AI/ML workflows.

## Install

```bash
# Core (numpy, zstandard, blake3)
pip install -e .

# With PyTorch support (ClusteredSoftmaxLoss)
pip install -e ".[torch]"

# With adaptive clustering (scikit-learn)
pip install -e ".[clustering]"

# Everything
pip install -e ".[all]"
```

## Native Acceleration Setup

SODL will run without the Rust extension, but the Python bridge will fall back
to slower pure-Python paths until `sodl-native` is installed in the same Python
environment that runs your app or tests.

```bash
# 1. Install the Python SDK into your active environment
pip install -e ".[all]"

# 2. Install maturin into that same environment
python -m pip install maturin

# 3. Build/install the Rust bridge into that same environment
cd ../crates/sodl-python-ffi
python -m maturin develop --release
```

On Windows, if `maturin develop` cannot see your virtualenv, run it from an
activated shell or set `VIRTUAL_ENV` first so the wheel installs into the
interpreter you actually use for tests/runtime.

You can verify the bridge is active with:

```python
from sodl import rust_bridge_summary

print(rust_bridge_summary())
```

Expected output will look like:

```text
native bridge active via sodl_native; accelerating blob_store, hashing, compression, integrity, optimizer_state, aead_crypto
```

## Quick Start

### 1. Content-Addressed Blob Storage

```python
from sodl import BlobStore, compute_blob_id

# Create a blob store
store = BlobStore("./my-blobs")

# Store data — automatically content-addressed via blake3
data = b"Hello, SODL!"
blob_id = compute_blob_id(data)
store.put(blob_id, data)

# Retrieve and verify
retrieved = store.get(blob_id)
assert retrieved == data
```

### 2. Weight Cluster Compression

```python
import numpy as np
from sodl import WeightStoreService

# High-level service with caching and lifecycle management
service = WeightStoreService("./weight-blobs")

# Create a model origin
origin = service.create_model("my-model-v1", "float32")

# Store embedding clusters (centroid + residual compression)
from sodl import WeightCluster
cluster = WeightCluster(
    centroid=[0.1, 0.2, 0.3],
    member_token_ids=[0, 5, 12],
    offsets=[[0.01, -0.02, 0.0], [0.0, 0.01, -0.01], [-0.01, 0.0, 0.02]],
    dim=3,
)
stats = service.store_cluster(origin.origin_id, cluster)
print(f"Stored as {stats.blob_id}, compressed {stats.raw_bytes} → {stats.stored_bytes} bytes")
```

### 3. Training Acceleration (ClusteredSoftmax)

```python
# Requires: pip install sodl[torch]
from sodl import create_clustered_loss

# Build clustered loss from any PyTorch model with embed_tokens/lm_head
loss_fn, token_index, build_stats = create_clustered_loss(
    model,              # any nn.Module with embed_tokens or lm_head
    n_clusters=64,      # sqrt(vocab_size) is a good default
    adaptive=True,      # auto-rebalance clusters
)

# Use in training loop — drop-in replacement for cross_entropy
hidden_states = model(input_ids).last_hidden_state  # (B, S, D)
loss = loss_fn(hidden_states, labels)
loss.backward()
```

### 4. Token Hash Index (Embedding Clustering)

```python
import numpy as np
from sodl import TokenHashIndex

# Cluster any embedding matrix
embeddings = np.random.randn(32000, 768).astype(np.float32)  # (vocab, dim)
index = TokenHashIndex(n_clusters=180)
stats = index.build(embeddings, adaptive=True)

print(f"Clusters: {stats['n_clusters']}, Compression: {stats['compression_potential']:.1%}")

# Hierarchical softmax lookup
hidden = np.random.randn(768).astype(np.float32)
results = index.hierarchical_softmax(hidden)
print(f"Top token: {results[0].token_id} (p={results[0].probability:.4f})")

# Save/load index
index.save("./my-index")
loaded = TokenHashIndex.load("./my-index")
```

## Features

| Feature | Description |
|---------|-------------|
| **Content-Addressed Storage** | Blake3 hashing, deduplication, integrity verification |
| **Compression** | Zstandard (zstd) with configurable levels |
| **Encryption** | Pluggable providers (null, XOR, or custom AEAD) |
| **Weight Compression** | Centroid + residual encoding with Q8 compact codec |
| **ClusteredSoftmax** | O(K + V/K) hierarchical loss replacing O(V) cross-entropy |
| **Gradient Sharing** | Decompose gradients into shared centroid + sparse offsets |
| **Adaptive Clustering** | Auto-rebalance with silhouette scoring |
| **Pin Registry** | Hot/cold cache with pin reasons and disk persistence |
| **Model Registry** | Track base models, LoRA adapters, GGUF exports, training lineage |
| **Multi-Tier Fetch** | Cache → source dirs → peers → edge URLs |
| **Rust FFI** | Optional native acceleration (10-50x for hashing/compression) |

## Architecture

```
Your Python Code
    │
    ▼
┌──────────────────────────────────────────┐
│  sodl_weights (Python SDK)               │
│  ├── WeightStoreService  (high-level)    │
│  ├── WeightBlobStore     (compress+CAS)  │
│  ├── BlobStore           (raw CAS)       │
│  ├── TokenHashIndex      (clustering)    │
│  ├── ClusteredSoftmaxLoss (training)     │
│  ├── ModelRegistry       (lineage)       │
│  └── WeightPinRegistry   (caching)       │
└──────────────────┬───────────────────────┘
                   │ (optional Rust FFI)
┌──────────────────▼───────────────────────┐
│  sodl_native (Rust, via PyO3)            │
│  └── 10-50x faster blake3 + zstd        │
└──────────────────────────────────────────┘
```

## Tests

```bash
pip install -e ".[test]"
pytest tests/ -v
```

## Native Acceleration

The Python bridge tries `sodl_native` first and falls back to pure Python if the
wheel is not installed in the active environment. If you use multiple virtual
environments, install `sodl-native` into each interpreter that runs SODL.

## Benchmarks

```bash
python benchmarks/bench_blob_store.py
python benchmarks/bench_clustered_softmax.py
```

## Docs

```bash
pip install -e ".[docs]"
mkdocs serve
```
