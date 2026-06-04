"""SODL Layer Scheduler — Progressive layer freezing for training acceleration.

Implements a brain-inspired approach where stable transformer layers are
"frozen" (gradients disabled) once their weights stabilize, mimicking
how the brain consolidates learned patterns and focuses compute on
active learning areas.

Usage::

    from sodl_weights.layer_scheduler import SODLLayerScheduler

    scheduler = SODLLayerScheduler(model, freeze_after_steps=500, check_every=50)

    for step in range(total_steps):
        with scheduler.step_context(step):
            loss = compute_loss(model, batch)
        loss.backward()
        scheduler.maybe_freeze(step)
        optimizer.step()
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
_torch = None

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass


class SODLLayerScheduler:
    """Progressive layer freezing based on weight stability.

    After ``freeze_after_steps`` training steps, layers whose weight
    change rate (cosine similarity between current and snapshot) exceeds
    ``stability_threshold`` are frozen— their gradients are disabled and
    they are skipped during backward, saving compute.

    Parameters
    ----------
    model : torch.nn.Module
        The model (should have named modules with "layers." or "blocks.").
    freeze_after_steps : int
        Minimum steps before any layer can be frozen.
    check_every : int
        How often to check layer stability.
    stability_threshold : float
        Cosine similarity above which a layer is considered stable.
    max_frozen_pct : float
        Maximum percentage of layers that can be frozen simultaneously.
    """

    def __init__(
        self,
        model,
        *,
        freeze_after_steps: int = 500,
        check_every: int = 50,
        stability_threshold: float = 0.995,
        max_frozen_pct: float = 0.5,
    ) -> None:
        self.model = model
        self.freeze_after_steps = max(1, freeze_after_steps)
        self.check_every = max(1, check_every)
        self.stability_threshold = stability_threshold
        self.max_frozen_pct = min(1.0, max(0.0, max_frozen_pct))

        # Discover layers
        self._layers: dict[str, Any] = {}
        self._frozen: set[str] = set()
        self._snapshots: dict[str, Any] = {}

        for name, module in model.named_modules():
            # Match transformer layer patterns
            if any(pattern in name for pattern in [".layers.", ".blocks.", ".transformer."]):
                # Only track leaf-level layer groups (e.g., "layers.0", not "layers.0.attn.q_proj")
                parts = name.split(".")
                for i, part in enumerate(parts):
                    if part in ("layers", "blocks", "transformer") and i + 1 < len(parts):
                        layer_key = ".".join(parts[: i + 2])
                        if layer_key not in self._layers:
                            self._layers[layer_key] = module
                        break

        self._total_layers = len(self._layers)
        self._max_frozen = max(0, int(self._total_layers * self.max_frozen_pct))

        if self._total_layers > 0:
            log.info(
                f"SODL LayerScheduler: tracking {self._total_layers} layers, "
                f"max frozen: {self._max_frozen} ({self.max_frozen_pct:.0%})"
            )

    def _snapshot_layer(self, name: str) -> None:
        """Save a copy of the layer's weights for later comparison."""
        if not _TORCH_AVAILABLE or _torch is None:
            return
        module = self._layers[name]
        param_data = []
        for p in module.parameters():
            param_data.append(p.data.detach().clone().reshape(-1))
        if param_data:
            self._snapshots[name] = _torch.cat(param_data)

    def _layer_similarity(self, name: str) -> float:
        """Compute cosine similarity between current weights and snapshot."""
        if not _TORCH_AVAILABLE or _torch is None:
            return 0.0
        if name not in self._snapshots:
            return 0.0

        module = self._layers[name]
        param_data = []
        for p in module.parameters():
            param_data.append(p.data.detach().reshape(-1))
        if not param_data:
            return 0.0

        current = _torch.cat(param_data)
        snapshot = self._snapshots[name]

        if current.shape != snapshot.shape:
            return 0.0

        dot = _torch.dot(current, snapshot).item()
        norm_c = current.norm().item()
        norm_s = snapshot.norm().item()
        if norm_c < 1e-12 or norm_s < 1e-12:
            return 0.0
        return dot / (norm_c * norm_s)

    def _freeze_layer(self, name: str) -> None:
        """Disable gradients for a layer."""
        module = self._layers[name]
        for p in module.parameters():
            p.requires_grad_(False)
        self._frozen.add(name)
        log.info(f"SODL LayerScheduler: froze {name}")

    def _unfreeze_layer(self, name: str) -> None:
        """Re-enable gradients for a layer."""
        module = self._layers[name]
        for p in module.parameters():
            p.requires_grad_(True)
        self._frozen.discard(name)
        log.info(f"SODL LayerScheduler: unfroze {name}")

    def maybe_freeze(self, step: int) -> None:
        """Check layer stability and freeze stable layers.

        Call this after each optimizer step.
        """
        if step < self.freeze_after_steps:
            # Take initial snapshots
            if step == 0:
                for name in self._layers:
                    self._snapshot_layer(name)
            return

        if step % self.check_every != 0:
            return

        # Compute stability for each unfrozen layer
        stabilities: list[tuple[str, float]] = []
        for name in self._layers:
            if name in self._frozen:
                continue
            sim = self._layer_similarity(name)
            stabilities.append((name, sim))

        # Sort by stability (most stable first)
        stabilities.sort(key=lambda x: -x[1])

        # Freeze layers that exceed threshold, up to max_frozen
        for name, sim in stabilities:
            if len(self._frozen) >= self._max_frozen:
                break
            if sim >= self.stability_threshold:
                self._freeze_layer(name)

        # Update snapshots for remaining unfrozen layers
        for name in self._layers:
            if name not in self._frozen:
                self._snapshot_layer(name)

    def step_context(self, step: int):
        """Context manager (no-op for now; reserved for future use)."""
        import contextlib
        return contextlib.nullcontext()

    def metrics(self) -> dict[str, Any]:
        """Return layer freezing metrics for dashboard."""
        return {
            "layer_scheduler_enabled": True,
            "total_layers": self._total_layers,
            "frozen_layers": len(self._frozen),
            "frozen_pct": round(
                len(self._frozen) / max(1, self._total_layers) * 100, 1
            ),
            "frozen_names": sorted(self._frozen),
            "trainable_params": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ) if _TORCH_AVAILABLE else 0,
        }
