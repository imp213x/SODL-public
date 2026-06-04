"""SODL Gradient Monitor — Dynamic accumulation with early-stop.

Instead of blindly running all ``accumulation_steps`` micro-batches,
this monitor tracks gradient convergence via cosine similarity.  When
consecutive gradient snapshots are sufficiently aligned, it signals
that additional micro-batches won't meaningfully change the update
direction, and the step can complete early.

Usage::

    from sodl_weights.gradient_monitor import SODLGradientMonitor

    monitor = SODLGradientMonitor(
        model,
        min_steps=8,
        max_steps=128,
        similarity_threshold=0.92,
        patience=3,
    )

    for step in range(total_steps):
        optimizer.zero_grad()
        micro = 0
        while not monitor.should_stop(micro):
            batch = next(data_iter)
            loss = compute_loss(model, batch) / monitor.effective_divisor
            loss.backward()
            micro += 1
        optimizer.step()
        monitor.reset()
"""

from __future__ import annotations

import logging
import math
from typing import Iterator

log = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
_torch = None

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass


def _flatten_grads(model) -> "list | None":
    """Collect all parameter gradients into a single flat vector."""
    if _torch is None:
        return None
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().reshape(-1))
    if not grads:
        return None
    return _torch.cat(grads)


def _cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two flat tensors."""
    if _torch is None or a is None or b is None:
        return 0.0
    dot = _torch.dot(a, b).item()
    norm_a = a.norm().item()
    norm_b = b.norm().item()
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


class SODLGradientMonitor:
    """Monitors gradient convergence to enable dynamic accumulation early-stop.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose gradients are monitored.
    min_steps : int
        Minimum number of micro-batches before early-stop is allowed.
    max_steps : int
        Maximum micro-batches (the hard cap).
    similarity_threshold : float
        Cosine similarity above which gradients are considered converged.
    patience : int
        Number of consecutive above-threshold checks before stopping.
    check_every : int
        Check convergence every N micro-batches (avoids overhead on every step).
    """

    def __init__(
        self,
        model,
        *,
        min_steps: int = 8,
        max_steps: int = 128,
        similarity_threshold: float = 0.92,
        patience: int = 3,
        check_every: int = 4,
    ) -> None:
        self.model = model
        self.min_steps = max(1, min_steps)
        self.max_steps = max(self.min_steps, max_steps)
        self.similarity_threshold = similarity_threshold
        self.patience = max(1, patience)
        self.check_every = max(1, check_every)

        # Internal state — reset per optimizer step
        self._prev_grad_snapshot = None
        self._consecutive_converged = 0
        self._stopped_at: int | None = None
        self._last_similarity = 0.0

        # Running stats for metrics
        self._total_steps_done = 0
        self._total_steps_max = 0
        self._step_count = 0

    def should_stop(self, micro_batch_index: int) -> bool:
        """Return True if accumulation should stop at this micro-batch.

        This is the main API called inside the accumulation loop. It checks:
        1. Hard cap (``max_steps``) — always stop
        2. Below minimum — never stop
        3. Convergence check every ``check_every`` micro-batches
        """
        # Hard cap
        if micro_batch_index >= self.max_steps:
            self._stopped_at = micro_batch_index
            return True

        # Minimum micro-batches not reached
        if micro_batch_index < self.min_steps:
            return False

        # Only check convergence every N steps to reduce overhead
        if micro_batch_index % self.check_every != 0:
            return False

        # Snapshot current gradients
        current_grad = _flatten_grads(self.model)
        if current_grad is None:
            return False

        if self._prev_grad_snapshot is not None:
            sim = _cosine_similarity(self._prev_grad_snapshot, current_grad)
            self._last_similarity = sim

            if sim >= self.similarity_threshold:
                self._consecutive_converged += 1
                if self._consecutive_converged >= self.patience:
                    self._stopped_at = micro_batch_index
                    log.debug(
                        f"SODL GradientMonitor: early-stop at micro-batch {micro_batch_index} "
                        f"(similarity={sim:.4f}, patience={self._consecutive_converged})"
                    )
                    return True
            else:
                self._consecutive_converged = 0

        # Store snapshot for next comparison (detach + clone to avoid graph issues)
        self._prev_grad_snapshot = current_grad.clone()
        return False

    @property
    def effective_divisor(self) -> int:
        """The divisor to use for loss scaling.

        When using dynamic accumulation, divide loss by ``max_steps``
        (not actual steps) to keep gradient magnitudes consistent
        regardless of when early-stop triggers.
        """
        return self.max_steps

    @property
    def actual_steps(self) -> int:
        """Number of micro-batches actually executed in the last step."""
        return self._stopped_at if self._stopped_at is not None else self.max_steps

    @property
    def savings_pct(self) -> float:
        """Percentage of micro-batches saved by early-stop."""
        if self._stopped_at is None:
            return 0.0
        return max(0.0, (1.0 - self._stopped_at / self.max_steps) * 100.0)

    def reset(self) -> None:
        """Reset per-step state. Call after each optimizer.step()."""
        if self._stopped_at is not None:
            self._total_steps_done += self._stopped_at
        else:
            self._total_steps_done += self.max_steps
        self._total_steps_max += self.max_steps
        self._step_count += 1
        self._prev_grad_snapshot = None
        self._consecutive_converged = 0
        self._stopped_at = None

    def metrics(self) -> dict[str, float | int]:
        """Return monitoring metrics for dashboard integration."""
        avg_steps = (
            self._total_steps_done / max(1, self._step_count)
        )
        avg_savings = (
            (1.0 - self._total_steps_done / max(1, self._total_steps_max)) * 100.0
            if self._total_steps_max > 0
            else 0.0
        )
        return {
            "gradient_monitor_enabled": True,
            "actual_accumulation_steps": self.actual_steps,
            "max_accumulation_steps": self.max_steps,
            "last_cosine_similarity": round(self._last_similarity, 4),
            "savings_pct": round(self.savings_pct, 1),
            "avg_micro_batches_per_step": round(avg_steps, 1),
            "avg_savings_pct": round(avg_savings, 1),
            "total_optimizer_steps": self._step_count,
        }
