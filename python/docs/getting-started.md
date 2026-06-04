# Getting Started

## Install

```bash
pip install -e .
pip install -e ".[torch]"
```

## Enable The Native Bridge

For production-speed hashing, compression, optimizer offload, and AEAD crypto,
install the Rust bridge into the same environment that runs SODL:

```bash
python -m pip install maturin
cd ../crates/sodl-python-ffi
python -m maturin develop --release
```

Verify the bridge is active:

```python
from sodl import rust_bridge_summary

print(rust_bridge_summary())
```

If this still reports a fallback path, the usual cause is that `sodl-native`
was installed into a different virtualenv than the one running your tests or app.

## First Blob Store

```python
from sodl import BlobStore, compute_blob_id

store = BlobStore("./blobs")
payload = b"hello"
blob_id = compute_blob_id(payload)
store.put(blob_id, payload)
assert store.get(blob_id) == payload
```

## First Weight Export

```python
from sodl import WeightCluster, WeightStoreService

service = WeightStoreService("./weight-blobs")
origin = service.create_model("demo-model", "float32")
service.store_cluster(
    origin.origin_id,
    WeightCluster(
        centroid=[0.0, 0.0],
        member_token_ids=[0, 1],
        offsets=[[0.1, 0.0], [0.0, 0.1]],
        dim=2,
    ),
)
service.close()
```
