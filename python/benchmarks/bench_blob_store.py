from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

import blake3
import zstandard as zstd

from sodl_weights import BlobStore, WeightBlobStore, WeightCluster
from sodl_weights import _rust_bridge


def _benchmark(label: str, fn, *, iterations: int) -> dict[str, float | str]:
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    avg_ms = statistics.mean(samples)
    ops_per_sec = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    return {"label": label, "avg_ms": avg_ms, "ops_per_sec": ops_per_sec}


def _sample_cluster() -> WeightCluster:
    return WeightCluster(
        centroid=[0.1] * 64,
        member_token_ids=list(range(16)),
        offsets=[[0.01 * idx] * 64 for idx in range(16)],
        dim=64,
    )


def run(iterations: int, payload_size: int) -> list[dict[str, float | str]]:
    payload = (b"sodl-native-benchmark-" * ((payload_size // 22) + 1))[:payload_size]
    results = [
        _benchmark("python_blake3", lambda: blake3.blake3(payload).hexdigest(), iterations=iterations),
        _benchmark("python_zstd", lambda: zstd.ZstdCompressor(level=3).compress(payload), iterations=iterations),
    ]

    if _rust_bridge.available():
        results.append(_benchmark("ffi_blake3", lambda: _rust_bridge.compute_blob_id(payload), iterations=iterations))
        results.append(_benchmark("ffi_zstd", lambda: _rust_bridge.compress_zstd(payload, 3), iterations=iterations))

    with tempfile.TemporaryDirectory() as tmpdir:
        store = BlobStore(Path(tmpdir) / "blobs")
        weight_store = WeightBlobStore(store)
        cluster = _sample_cluster()
        results.append(_benchmark("blob_store_put_get", lambda: weight_store.get("bench", weight_store.put("bench", cluster).blob_id), iterations=max(1, iterations // 4)))

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SODL blob-store primitives")
    parser.add_argument("--iterations", type=int, default=200, help="Benchmark iterations")
    parser.add_argument("--payload-size", type=int, default=16384, help="Payload size in bytes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Rust bridge: {_rust_bridge.status_summary()}")
    for result in run(args.iterations, args.payload_size):
        print(f"{result['label']:>20}  avg_ms={result['avg_ms']:.3f}  ops_per_sec={result['ops_per_sec']:.1f}")


if __name__ == "__main__":
    main()
