# Model Lifecycle

The Python lifecycle stack is:

1. `WeightStoreService` for cluster storage and cache management
2. `CheckpointManager` for checkpoint save/load/diff
3. `ModelRegistry` for lineage
4. `ArtifactStore` for generic AI artifacts

This keeps model bytes, metadata, and checkpoints in the same content-addressed
system.
