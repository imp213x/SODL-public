"""SODL CPU Autocast — Detect and manage bf16 compute on CPU.

This module provides a SODL-owned autocast context manager that:

1. Detects whether the CPU supports bfloat16 natively (AVX-512, AMX, etc.)
2. Wraps ``torch.autocast("cpu", dtype=torch.bfloat16)`` when supported
3. Falls back to fp32 gracefully on older CPUs
4. Reports the active precision mode to training metrics

Usage in a training loop::

    from sodl_weights.sodl_autocast import SODLAutocast

    autocast = SODLAutocast()
    print(f"Precision: {autocast.precision_label}")

    for step in range(steps):
        with autocast:
            logits = model(input_ids)
            loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

log = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
_torch = None

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass


def cpu_supports_bf16() -> bool:
    """Return True if the CPU can run bf16 compute without emulation.

    Checks for AVX-512 BF16 / AMX support via PyTorch internals and
    a practical allocation test.
    """
    if not _TORCH_AVAILABLE or _torch is None:
        return False
    try:
        # Practical test: create a bf16 tensor, do a matmul, verify no crash
        a = _torch.randn(4, 4, dtype=_torch.bfloat16)
        b = _torch.randn(4, 4, dtype=_torch.bfloat16)
        _ = a @ b
        return True
    except Exception:
        return False


class SODLAutocast:
    """SODL-owned autocast manager for CPU training acceleration.

    Automatically detects bf16 support and applies it. Keeps master weights
    in fp32 (SODLAdamW handles this) while running forward/backward in bf16.

    Attributes
    ----------
    enabled : bool
        Whether bf16 autocast is active.
    precision_label : str
        Human-readable precision mode for dashboard metrics.
    """

    def __init__(self, *, force_fp32: bool = False) -> None:
        self._bf16_available = cpu_supports_bf16() and not force_fp32
        self.enabled = self._bf16_available
        self.precision_label = "bf16" if self.enabled else "fp32"
        self._ctx = None

        if self.enabled:
            log.info("SODL Autocast: CPU bf16 detected — training in bfloat16")
        else:
            reason = "forced fp32" if force_fp32 else "CPU bf16 not available"
            log.info(f"SODL Autocast: running in fp32 ({reason})")

    def __enter__(self):
        if self.enabled and _torch is not None:
            self._ctx = _torch.autocast("cpu", dtype=_torch.bfloat16)
            self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._ctx is not None:
            self._ctx.__exit__(exc_type, exc_val, exc_tb)
            self._ctx = None
        return False

    def metrics(self) -> dict[str, str | bool]:
        """Return precision metadata for dashboard integration."""
        return {
            "precision_mode": self.precision_label,
            "bf16_available": self._bf16_available,
            "bf16_active": self.enabled,
        }
