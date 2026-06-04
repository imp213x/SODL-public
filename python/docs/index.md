# SODL Python SDK

SODL is a content-addressed, lineage-aware data framework for AI systems.

The package exposes:

- Content-addressed blob storage
- Weight cluster storage and cache management
- Clustered softmax training helpers
- Dataset, artifact, and checkpoint helpers
- Optional Rust FFI acceleration

The clean package import surface is:

```python
from sodl import BlobStore, WeightStoreService, TokenHashIndex
```

For best performance, also install the optional Rust bridge:

```bash
python -m pip install maturin
cd ../crates/sodl-python-ffi
python -m maturin develop --release
```

Then verify native mode:

```python
from sodl import rust_bridge_summary

print(rust_bridge_summary())
```
