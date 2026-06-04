"""SODL PolarQuant — Python SDK wrapper for polar-coordinate weight quantization.

Provides a pure-Python implementation of the PolarQuant algorithm. When the
Rust ``sodl-polar-quant`` crate is exposed through ``sodl-python-ffi``, this
module will automatically delegate to the native implementation for 10-50x
speedup.

Usage::

    from sodl_weights.polar_quant import PolarQuantizer

    quantizer = PolarQuantizer()

    # Quantize a weight matrix (numpy)
    compressed, stats = quantizer.quantize_matrix(weight_matrix)
    print(f"Compression: {stats['compression_ratio']:.1f}x, Error: {stats['avg_relative_error']:.4f}")

    # Dequantize back
    reconstructed = quantizer.dequantize_batch(compressed)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

TWO_PI = 2.0 * math.pi


@dataclass(slots=True)
class PolarQuantizedVector:
    """A polar-quantized weight vector."""
    norm_q8: int           # 8-bit log-quantized L2 norm
    angles: bytes          # Quantized angles (one per dim-1)
    original_dim: int


def quantize_norm(norm: float) -> int:
    """Encode f32 norm to 8-bit log-scale."""
    if norm <= 0.0 or not math.isfinite(norm):
        return 0
    q = round(math.log2(norm) * 16.0 + 128.0)
    return max(0, min(255, q))


def dequantize_norm(q: int) -> float:
    """Decode 8-bit log-quantized norm."""
    return 2.0 ** ((q - 128.0) / 16.0)


def quantize_angle(theta: float) -> int:
    """Encode angle in [0, 2π) to 8-bit."""
    normalized = theta % TWO_PI
    q = round(normalized / TWO_PI * 256.0)
    return q % 256


def dequantize_angle(q: int) -> float:
    """Decode 8-bit angle to [0, 2π)."""
    return q * (TWO_PI / 256.0)


def quantize_vector(weights: np.ndarray) -> PolarQuantizedVector:
    """Quantize an fp32 weight vector to polar form."""
    d = len(weights)
    if d == 0:
        raise ValueError("Cannot quantize empty vector")

    norm = float(np.linalg.norm(weights))
    norm_q8 = quantize_norm(norm)

    if d == 1:
        return PolarQuantizedVector(
            norm_q8=norm_q8,
            angles=bytes([0 if weights[0] >= 0 else 128]),
            original_dim=1,
        )

    safe_norm = norm if norm > 1e-30 else 1.0
    unit = weights / safe_norm

    angles_list = []
    remaining_sq = 1.0

    for i in range(d - 1):
        r = math.sqrt(max(0.0, remaining_sq))
        cos_t = float(unit[i]) / r if r > 1e-15 else 0.0
        cos_t = max(-1.0, min(1.0, cos_t))
        theta = math.acos(cos_t)
        if i == d - 2 and float(unit[i + 1]) < 0:
            theta = TWO_PI - theta
        angles_list.append(quantize_angle(theta))
        remaining_sq -= float(unit[i]) ** 2
        remaining_sq = max(0.0, remaining_sq)

    return PolarQuantizedVector(
        norm_q8=norm_q8,
        angles=bytes(angles_list),
        original_dim=d,
    )


def dequantize_vector(pq: PolarQuantizedVector) -> np.ndarray:
    """Dequantize polar-quantized vector back to fp32."""
    d = pq.original_dim
    norm = dequantize_norm(pq.norm_q8)

    if d == 1:
        sign = -1.0 if pq.angles[0] >= 128 else 1.0
        return np.array([sign * norm], dtype=np.float32)

    result = np.zeros(d, dtype=np.float32)
    remaining = 1.0

    for i, aq in enumerate(pq.angles):
        theta = dequantize_angle(aq)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        result[i] = norm * remaining * cos_t
        remaining *= sin_t
        if i == len(pq.angles) - 1:
            result[i + 1] = norm * remaining

    return result[:d]


class PolarQuantizer:
    """High-level quantizer for weight matrices.

    Quantizes each row of a weight matrix independently using polar
    coordinate transformation.
    """

    def quantize_matrix(
        self,
        weights: np.ndarray,
    ) -> tuple[list[PolarQuantizedVector], dict]:
        """Quantize a 2D weight matrix (rows = vectors).

        Returns a list of PolarQuantizedVector and statistics dict.
        """
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)
        assert weights.ndim == 2

        quantized = []
        total_error = 0.0
        n = len(weights)
        orig_bytes = weights.size * 4

        for row in weights:
            pq = quantize_vector(row.astype(np.float32))
            recon = dequantize_vector(pq)
            row_norm = float(np.linalg.norm(row))
            err_norm = float(np.linalg.norm(row.astype(np.float32) - recon))
            total_error += (err_norm / row_norm) if row_norm > 1e-30 else 0.0
            quantized.append(pq)

        quant_bytes = sum(1 + len(pq.angles) for pq in quantized)

        stats = {
            "count": n,
            "original_bytes": orig_bytes,
            "quantized_bytes": quant_bytes,
            "compression_ratio": orig_bytes / max(1, quant_bytes),
            "avg_relative_error": total_error / max(1, n),
        }
        return quantized, stats

    def dequantize_batch(
        self,
        quantized: Sequence[PolarQuantizedVector],
    ) -> np.ndarray:
        """Dequantize a batch of vectors back to a 2D np.float32 matrix."""
        rows = [dequantize_vector(pq) for pq in quantized]
        return np.stack(rows, axis=0)

    def to_bytes(self, pq: PolarQuantizedVector) -> bytes:
        """Serialize a single PolarQuantizedVector."""
        dim_bytes = pq.original_dim.to_bytes(4, "little")
        return dim_bytes + bytes([pq.norm_q8]) + pq.angles

    def from_bytes(self, data: bytes) -> PolarQuantizedVector:
        """Deserialize a PolarQuantizedVector."""
        if len(data) < 5:
            raise ValueError("Payload too short")
        dim = int.from_bytes(data[:4], "little")
        norm_q8 = data[4]
        angles = data[5:]
        return PolarQuantizedVector(norm_q8=norm_q8, angles=angles, original_dim=dim)
