"""AdamW-compatible optimizer with SODL-backed offloaded state."""

from __future__ import annotations

import hashlib
import io
import json
import math
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
from torch.optim import Optimizer

from sodl_weights.optimizer_state import OptimizerStateManifest, OptimizerStateStore


@dataclass(slots=True)
class ParameterRef:
    name: str
    group_index: int
    param_index: int
    parameter: torch.nn.Parameter


@dataclass(slots=True)
class OptimizerBlockLayout:
    block_id: str
    group_index: int
    param_names: list[str]
    param_shapes: dict[str, list[int]]


def _serialize_state(payload: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _deserialize_state(payload: bytes) -> dict[str, Any]:
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)


def _layout_fingerprint(blocks: list[OptimizerBlockLayout], block_size: int) -> str:
    payload = {
        "block_size": int(block_size),
        "blocks": [asdict(block) for block in blocks],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class SODLAdamW(Optimizer):
    """AdamW with optimizer moments externalized to an :class:`OptimizerStateStore`."""

    def __init__(
        self,
        params: Iterable[Any],
        *,
        state_store: OptimizerStateStore,
        origin_id: str,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]] | None = None,
        block_size: int = 8,
        flush_every: int = 1,
        prefetch_lookahead: int = 1,
        async_writeback: bool = False,
        use_batch_state_ops: bool = True,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        raw_params = list(params)
        inferred_named: list[tuple[str, torch.nn.Parameter]] | None = None
        optimizer_params: list[Any]

        if raw_params and isinstance(raw_params[0], tuple):
            inferred_named = [(str(name), parameter) for name, parameter in raw_params]
            optimizer_params = [parameter for _, parameter in inferred_named]
        else:
            optimizer_params = raw_params

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(optimizer_params, defaults)

        self.state_store = state_store
        self.origin_id = origin_id
        self.block_size = max(1, int(block_size))
        self.flush_every = max(1, int(flush_every))
        self.prefetch_lookahead = max(0, int(prefetch_lookahead))
        self.async_writeback = bool(async_writeback)
        self.use_batch_state_ops = bool(use_batch_state_ops)
        self._step_index = 0

        provided_named = list(named_parameters) if named_parameters is not None else inferred_named
        provided_name_map = {id(parameter): name for name, parameter in (provided_named or [])}

        self._ordered_params: list[ParameterRef] = []
        self._blocks: list[OptimizerBlockLayout] = []
        self._block_refs: dict[str, list[ParameterRef]] = {}
        self._build_layout(provided_name_map)
        self._layout_fingerprint = _layout_fingerprint(self._blocks, self.block_size)
        self._pinned_window: set[str] = set()

        self._flush_executor: ThreadPoolExecutor | None = None
        self._pending_flush: Future[OptimizerStateManifest] | None = None
        if self.async_writeback:
            self._flush_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="sodl-optimizer-flush",
            )

        # ── Phase 5: GC integration ──
        # Track block IDs from previous step so superseded blocks can be unpinned
        self._previous_step_blocks: set[str] = set()
        self._gc_blocks_cleaned: int = 0
        self._gc_enabled: bool = True

    def _build_layout(self, provided_name_map: dict[int, str]) -> None:
        for group_index, group in enumerate(self.param_groups):
            params = list(group["params"])
            for param_index, parameter in enumerate(params):
                name = provided_name_map.get(id(parameter), f"group{group_index}_param{param_index}")
                self._ordered_params.append(
                    ParameterRef(
                        name=name,
                        group_index=group_index,
                        param_index=param_index,
                        parameter=parameter,
                    )
                )

        grouped: dict[int, list[ParameterRef]] = {}
        for ref in self._ordered_params:
            grouped.setdefault(ref.group_index, []).append(ref)

        for group_index, refs in grouped.items():
            for block_index in range(0, len(refs), self.block_size):
                block_refs = refs[block_index : block_index + self.block_size]
                self._blocks.append(
                    OptimizerBlockLayout(
                        block_id=f"group{group_index}-block{block_index // self.block_size}",
                        group_index=group_index,
                        param_names=[ref.name for ref in block_refs],
                        param_shapes={
                            ref.name: list(ref.parameter.shape)
                            for ref in block_refs
                        },
                    )
                )
                self._block_refs[self._blocks[-1].block_id] = list(block_refs)

    def _refs_for_block(self, layout: OptimizerBlockLayout) -> list[ParameterRef]:
        return self._block_refs.get(layout.block_id, [])

    def _load_block_state(self, block_id: str) -> dict[str, Any]:
        try:
            payload = self.state_store.load_block(self.origin_id, block_id)
        except FileNotFoundError:
            return {"step": self._step_index, "params": {}}
        return _deserialize_state(payload)

    def _load_block_states(self, block_ids: list[str]) -> dict[str, dict[str, Any]]:
        payloads = self.state_store.load_blocks(self.origin_id, block_ids)
        states: dict[str, dict[str, Any]] = {}
        for block_id in block_ids:
            payload = payloads.get(block_id)
            if payload is None:
                states[block_id] = {"step": self._step_index, "params": {}}
            else:
                states[block_id] = _deserialize_state(payload)
        return states

    def _store_block_state(self, layout: OptimizerBlockLayout, payload: dict[str, Any]) -> None:
        metadata = {
            "group_index": layout.group_index,
            "param_names": list(layout.param_names),
            "param_shapes": dict(layout.param_shapes),
            "step": int(payload.get("step", self._step_index)),
            "layout_fingerprint": self._layout_fingerprint,
        }
        self.state_store.store_block(
            self.origin_id,
            layout.block_id,
            _serialize_state(payload),
            step=int(payload.get("step", self._step_index)),
            shard_key=f"group:{layout.group_index}",
            metadata=metadata,
        )

    def _store_block_states(self, payloads: list[dict[str, Any]]) -> None:
        if not payloads:
            return
        self.state_store.store_blocks(self.origin_id, payloads)

    def _active_layouts(self) -> list[OptimizerBlockLayout]:
        active: list[OptimizerBlockLayout] = []
        for layout in self._blocks:
            refs = self._refs_for_block(layout)
            if any(ref.parameter.grad is not None for ref in refs):
                active.append(layout)
        return active

    def _prefetch_after(self, active_layouts: list[OptimizerBlockLayout], index: int) -> None:
        if self.prefetch_lookahead <= 0:
            return
        lookahead = [
            layout.block_id
            for layout in active_layouts[index + 1 : index + 1 + self.prefetch_lookahead]
        ]
        if lookahead:
            self.state_store.prefetch_blocks(self.origin_id, lookahead)

    def wait_for_pending_flush(self) -> OptimizerStateManifest | None:
        if self._pending_flush is None:
            return None
        future = self._pending_flush
        self._pending_flush = None
        return future.result()

    def _schedule_flush(self, block_ids: list[str]) -> OptimizerStateManifest | None:
        if not block_ids:
            return None
        previous = self.wait_for_pending_flush()
        if not self.async_writeback or self._flush_executor is None:
            return self.state_store.flush_blocks(self.origin_id, block_ids)
        self._pending_flush = self._flush_executor.submit(
            self.state_store.flush_blocks,
            self.origin_id,
            block_ids,
        )
        return previous

    def _update_pin_window(self, block_ids: list[str]) -> None:
        target = set(block_ids[: max(1, self.prefetch_lookahead)])
        to_unpin = sorted(self._pinned_window - target)
        to_pin = sorted(target - self._pinned_window)
        if to_unpin:
            self.state_store.unpin_blocks(self.origin_id, to_unpin)
        if to_pin:
            self.state_store.pin_blocks(self.origin_id, to_pin)
        self._pinned_window = target

    @torch.no_grad()
    def step(self, closure=None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_index += 1
        active_layouts = self._active_layouts()
        touched_blocks: list[str] = []
        loaded_states = (
            self._load_block_states([layout.block_id for layout in active_layouts])
            if self.use_batch_state_ops
            else {}
        )
        pending_store_payloads: list[dict[str, Any]] = []

        for layout in active_layouts:
            refs = self._refs_for_block(layout)
            touched_blocks.append(layout.block_id)
            if self.use_batch_state_ops:
                block_state = loaded_states.get(layout.block_id, {"step": self._step_index, "params": {}})
            else:
                block_state = self._load_block_state(layout.block_id)
            params_state = dict(block_state.get("params", {}))

            group = self.param_groups[layout.group_index]
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])

            for ref in refs:
                grad = ref.parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("SODLAdamW does not support sparse gradients")

                param_state = params_state.get(ref.name)
                if param_state is None:
                    exp_avg = torch.zeros_like(ref.parameter, memory_format=torch.preserve_format)
                    exp_avg_sq = torch.zeros_like(ref.parameter, memory_format=torch.preserve_format)
                else:
                    exp_avg = param_state["exp_avg"].to(ref.parameter.device, dtype=ref.parameter.dtype)
                    exp_avg_sq = param_state["exp_avg_sq"].to(ref.parameter.device, dtype=ref.parameter.dtype)

                grad_data = grad.detach()
                exp_avg.mul_(beta1).add_(grad_data, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_data, grad_data, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** self._step_index
                bias_correction2 = 1.0 - beta2 ** self._step_index

                if weight_decay != 0.0:
                    ref.parameter.data.mul_(1.0 - lr * weight_decay)

                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                ref.parameter.data.addcdiv_(exp_avg, denom, value=-step_size)

                params_state[ref.name] = {
                    "exp_avg": exp_avg.detach().cpu(),
                    "exp_avg_sq": exp_avg_sq.detach().cpu(),
                }

            payload = {
                "step": self._step_index,
                "params": params_state,
            }
            if self.use_batch_state_ops:
                pending_store_payloads.append(
                    {
                        "block_id": layout.block_id,
                        "payload": _serialize_state(payload),
                        "step": int(payload.get("step", self._step_index)),
                        "shard_key": f"group:{layout.group_index}",
                        "metadata": {
                            "group_index": layout.group_index,
                            "param_names": list(layout.param_names),
                            "param_shapes": dict(layout.param_shapes),
                            "step": int(payload.get("step", self._step_index)),
                            "layout_fingerprint": self._layout_fingerprint,
                        },
                    }
                )
            else:
                self._store_block_state(layout, payload)
        if self.use_batch_state_ops:
            self._store_block_states(pending_store_payloads)

        self._update_pin_window(touched_blocks)
        if touched_blocks and self._step_index % self.flush_every == 0:
            self._schedule_flush(touched_blocks)

        # ── Phase 5: GC — unpin superseded blocks from previous step ──
        if self._gc_enabled and self._previous_step_blocks:
            current_blocks = set(touched_blocks)
            stale = self._previous_step_blocks - current_blocks
            if stale:
                try:
                    self.state_store.unpin_blocks(self.origin_id, sorted(stale))
                    self._gc_blocks_cleaned += len(stale)
                except Exception:
                    pass  # GC is best-effort; don't crash training
        self._previous_step_blocks = set(touched_blocks)

        return float(loss) if loss is not None else None

    def gc_stats(self) -> dict[str, int | bool]:
        """Return GC-related metrics for dashboard integration."""
        return {
            "gc_enabled": self._gc_enabled,
            "gc_blocks_cleaned_total": self._gc_blocks_cleaned,
            "gc_current_pinned_blocks": len(self._previous_step_blocks),
        }

    def flush(self, block_ids: list[str] | None = None) -> OptimizerStateManifest:
        pending_manifest = self.wait_for_pending_flush()
        if block_ids is None:
            manifest = self.state_store.flush_origin(self.origin_id)
        elif block_ids:
            manifest = self.state_store.flush_blocks(self.origin_id, block_ids)
        else:
            manifest = self.state_store.manifest(self.origin_id)
        return manifest if manifest.blocks or pending_manifest is None else pending_manifest

    def external_state_dict(self) -> dict[str, Any]:
        manifest = self.flush()
        return {
            "origin_id": self.origin_id,
            "step": self._step_index,
            "block_size": self.block_size,
            "flush_every": self.flush_every,
            "prefetch_lookahead": self.prefetch_lookahead,
            "async_writeback": self.async_writeback,
            "use_batch_state_ops": self.use_batch_state_ops,
            "layout_fingerprint": self._layout_fingerprint,
            "blocks": [asdict(block) for block in self._blocks],
            "manifest": {
                "schema": manifest.schema,
                "origin_id": manifest.origin_id,
                "updated_at": manifest.updated_at,
                "blocks": {block_id: asdict(record) for block_id, record in manifest.blocks.items()},
            },
        }

    def state_dict(self) -> dict[str, Any]:
        return self.external_state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        saved_origin_id = str(state_dict.get("origin_id", self.origin_id))
        if saved_origin_id != self.origin_id:
            raise ValueError(
                f"SODLAdamW origin mismatch: checkpoint={saved_origin_id} current={self.origin_id}"
            )
        saved_fingerprint = state_dict.get("layout_fingerprint")
        if saved_fingerprint and saved_fingerprint != self._layout_fingerprint:
            raise ValueError(
                "SODLAdamW layout mismatch: checkpoint layout does not match current parameter sharding"
            )
        self.wait_for_pending_flush()
        self._step_index = int(state_dict.get("step", self._step_index))
        self.prefetch_lookahead = max(
            0,
            int(state_dict.get("prefetch_lookahead", self.prefetch_lookahead)),
        )
        self.use_batch_state_ops = bool(
            state_dict.get("use_batch_state_ops", self.use_batch_state_ops)
        )
        self._update_pin_window([])

    def __del__(self) -> None:
        try:
            self.wait_for_pending_flush()
        except Exception:
            pass
        if self._flush_executor is not None:
            self._flush_executor.shutdown(wait=False, cancel_futures=False)
