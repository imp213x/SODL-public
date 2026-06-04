# Training Acceleration

SODL accelerates training with:

- `TokenHashIndex` for embedding clustering
- `ClusteredSoftmaxLoss` for hierarchical softmax

```python
from sodl import create_clustered_loss

loss_fn, token_index, stats = create_clustered_loss(
    model,
    n_clusters=180,
    adaptive=True,
)
```

The same cluster metadata can be exported as SODL blobs for cache-aware
training and inference paths.
