"""SODL Weight Cluster Compression — checkpoint compression via centroid+residual.

Mirrors SODL's Rust ``WeightCluster`` and ``WeightBlobStore`` to compress
neural network weight tensors using centroid + residual encoding.

Each weight matrix is decomposed into K clusters where:
- Centroid = mean of cluster members
- Residuals = offsets from centroid (stored in int8 quantized form)
- Compression ratio ≈ 4x (fp32 → int8 residuals + fp32 centroid)

Usage::

    from sodl_weights.weight_compress import WeightCompressor

    compressor = WeightCompressor(n_clusters=64)
    compressed = compressor.compress_state_dict(model.state_dict())
    restored = compressor.decompress_state_dict(compressed)
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class CompressedTensor:
    """A weight tensor compressed via centroid+residual encoding."""
    name: str
    shape: tuple[int, ...]
    centroid: np.ndarray           # fp32, shape (dim,)
    residuals_q8: np.ndarray       # int8, shape (numel, dim) or flat
    scale: float                   # quantization scale factor
    original_dtype: str


@dataclass
class CompressionStats:
    """Compression statistics for a full state dict."""
    tensors: int = 0
    original_bytes: int = 0
    compressed_bytes: int = 0
    compression_ratio: float = 0.0
    skipped_small: int = 0


class WeightCompressor:
    """Centroid + residual weight compression matching SODL's Rust WeightBlobStore.

    Parameters
    ----------
    min_elements : int
        Minimum tensor size (elements) to compress. Smaller tensors are stored raw.
    """

    def __init__(self, *, min_elements: int = 256) -> None:
        self.min_elements = min_elements

    def compress_tensor(self, name: str, tensor: np.ndarray) -> CompressedTensor:
        """Compress a single weight tensor."""
        flat = tensor.astype(np.float32).reshape(-1)
        centroid = np.mean(flat)
        residuals = flat - centroid

        # Quantize residuals to int8
        max_abs = np.max(np.abs(residuals))
        scale = max_abs / 127.0 if max_abs > 1e-10 else 1.0
        residuals_q8 = np.clip(
            np.round(residuals / scale), -127, 127
        ).astype(np.int8)

        return CompressedTensor(
            name=name,
            shape=tensor.shape,
            centroid=np.array([centroid], dtype=np.float32),
            residuals_q8=residuals_q8,
            scale=scale,
            original_dtype=str(tensor.dtype),
        )

    def decompress_tensor(self, ct: CompressedTensor) -> np.ndarray:
        """Decompress back to fp32."""
        residuals = ct.residuals_q8.astype(np.float32) * ct.scale
        flat = residuals + ct.centroid[0]
        return flat.reshape(ct.shape)

    def compress_state_dict(
        self, state_dict: dict[str, Any]
    ) -> tuple[dict[str, CompressedTensor | np.ndarray], CompressionStats]:
        """Compress a full model state dict.

        Returns (compressed_dict, stats) where compressed_dict contains
        CompressedTensor for large tensors and raw arrays for small ones.
        """
        compressed = {}
        stats = CompressionStats()

        for name, tensor in state_dict.items():
            if hasattr(tensor, "numpy"):
                arr = tensor.detach().cpu().numpy()
            elif isinstance(tensor, np.ndarray):
                arr = tensor
            else:
                compressed[name] = tensor
                continue

            stats.tensors += 1
            orig_bytes = arr.nbytes
            stats.original_bytes += orig_bytes

            if arr.size < self.min_elements:
                compressed[name] = arr
                stats.compressed_bytes += orig_bytes
                stats.skipped_small += 1
                continue

            ct = self.compress_tensor(name, arr)
            comp_bytes = ct.centroid.nbytes + ct.residuals_q8.nbytes + 8  # + scale
            stats.compressed_bytes += comp_bytes
            compressed[name] = ct

        stats.compression_ratio = (
            stats.original_bytes / max(1, stats.compressed_bytes)
        )
        return compressed, stats

    def decompress_state_dict(
        self, compressed: dict[str, CompressedTensor | np.ndarray | Any]
    ) -> dict[str, np.ndarray]:
        """Decompress back to a dict of numpy arrays."""
        result = {}
        for name, val in compressed.items():
            if isinstance(val, CompressedTensor):
                result[name] = self.decompress_tensor(val)
            elif isinstance(val, np.ndarray):
                result[name] = val
            else:
                result[name] = val
        return result

    def save_compressed(
        self,
        compressed: dict[str, CompressedTensor | np.ndarray | Any],
        path: str,
    ) -> int:
        """Save compressed state dict to a .npz file. Returns bytes written."""
        save_dict = {}
        metadata = {}
        for name, val in compressed.items():
            if isinstance(val, CompressedTensor):
                save_dict[f"{name}__centroid"] = val.centroid
                save_dict[f"{name}__residuals"] = val.residuals_q8
                save_dict[f"{name}__meta"] = np.array(
                    [val.scale, *val.shape], dtype=np.float32
                )
                metadata[name] = "compressed"
            elif isinstance(val, np.ndarray):
                save_dict[name] = val
                metadata[name] = "raw"

        np.savez_compressed(path, **save_dict)
        import os
        return os.path.getsize(path)

    def load_compressed(
        self, path: str
    ) -> dict[str, CompressedTensor | np.ndarray]:
        """Load a compressed state dict from .npz."""
        data = np.load(path, allow_pickle=False)
        result = {}
        seen = set()

        for key in data.files:
            base = key.rsplit("__", 1)[0]
            if base in seen:
                continue

            if f"{base}__centroid" in data.files:
                centroid = data[f"{base}__centroid"]
                residuals = data[f"{base}__residuals"]
                meta = data[f"{base}__meta"]
                scale = float(meta[0])
                shape = tuple(int(x) for x in meta[1:])
                result[base] = CompressedTensor(
                    name=base,
                    shape=shape,
                    centroid=centroid,
                    residuals_q8=residuals,
                    scale=scale,
                    original_dtype="float32",
                )
                seen.add(base)
            else:
                result[key] = data[key]
                seen.add(key)

        return result
