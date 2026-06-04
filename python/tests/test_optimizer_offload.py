from __future__ import annotations

import tempfile

import torch
import torch.nn as nn

from sodl_weights import BlobStore
from sodl_weights.checkpoint import CheckpointManager
from sodl_weights.offload_optimizer import SODLAdamW
from sodl_weights.optimizer_state import OptimizerStateStore
from sodl_weights.service import WeightStoreService


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.GELU(),
            nn.Linear(8, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_optimizer_state_store_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = OptimizerStateStore(
            f"{tmpdir}/blobs",
            f"{tmpdir}/registry",
            cache_capacity=4,
            writeback_threshold=1,
        )
        result = store.store_block(
            "run-1",
            "block-a",
            b"optimizer-state",
            step=3,
            shard_key="group:0",
            metadata={"kind": "adamw"},
        )
        assert result.flushed
        assert result.blob_id is not None

        payload = store.load_block("run-1", "block-a")
        assert payload == b"optimizer-state"

        manifest = store.manifest("run-1")
        assert "block-a" in manifest.blocks


def test_optimizer_state_store_staged_flush_and_evict() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = OptimizerStateStore(
            f"{tmpdir}/blobs",
            f"{tmpdir}/registry",
            cache_capacity=4,
            writeback_threshold=8,
        )
        store.store_block("run-evict", "block-a", b"state-a", step=1)
        store.store_block("run-evict", "block-b", b"state-b", step=1)

        manifest = store.flush_blocks("run-evict", ["block-a"])
        assert "block-a" in manifest.blocks
        assert "block-b" not in manifest.blocks

        store.pin_blocks("run-evict", ["block-a"])
        assert store.evict_blocks("run-evict", ["block-a", "block-b"]) == 0

        store.flush_blocks("run-evict", ["block-b"])
        store.unpin_blocks("run-evict", ["block-a"])
        assert store.evict_blocks("run-evict", ["block-a", "block-b"]) == 2


def test_optimizer_state_store_batch_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = OptimizerStateStore(
            f"{tmpdir}/blobs",
            f"{tmpdir}/registry",
            cache_capacity=4,
            writeback_threshold=8,
        )
        results = store.store_blocks(
            "run-batch",
            [
                {
                    "block_id": "block-a",
                    "payload": b"state-a",
                    "step": 1,
                    "shard_key": "group:0",
                    "metadata": {"kind": "adamw"},
                },
                {
                    "block_id": "block-b",
                    "payload": b"state-b",
                    "step": 1,
                    "shard_key": "group:0",
                    "metadata": {"kind": "adamw"},
                },
            ],
        )
        assert len(results) == 2
        assert all(result.staged for result in results)

        store.flush_origin("run-batch")
        payloads = store.load_blocks("run-batch", ["block-a", "block-b"])
        assert payloads["block-a"] == b"state-a"
        assert payloads["block-b"] == b"state-b"


def test_sodl_adamw_externalizes_state_and_checkpoint_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyNet()
        state_store = OptimizerStateStore(
            f"{tmpdir}/optimizer_blobs",
            f"{tmpdir}/optimizer_registry",
            cache_capacity=8,
            writeback_threshold=1,
        )
        optimizer = SODLAdamW(
            list(model.named_parameters()),
            state_store=state_store,
            origin_id="train-run",
            block_size=2,
            flush_every=1,
            lr=5e-4,
        )

        x = torch.randn(6, 4)
        y = torch.randn(6, 2)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()

        manifest = state_store.manifest("train-run")
        assert manifest.blocks

        external = optimizer.external_state_dict()
        assert external["origin_id"] == "train-run"
        assert external["manifest"]["blocks"]

        blob_store = BlobStore(f"{tmpdir}/checkpoint_blobs")
        checkpoint_mgr = CheckpointManager(blob_store, f"{tmpdir}/checkpoints")
        ckpt_id = checkpoint_mgr.save_checkpoint(
            model,
            optimizer,
            step=1,
            origin_id="train-run",
        )
        state = checkpoint_mgr.load_checkpoint("train-run", ckpt_id)
        assert "optimizer_manifest" in state
        assert "optimizer" not in state


def test_sodl_adamw_resume_validates_layout_and_drains_async_writeback() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyNet()
        state_store = OptimizerStateStore(
            f"{tmpdir}/optimizer_blobs",
            f"{tmpdir}/optimizer_registry",
            cache_capacity=8,
            writeback_threshold=8,
        )
        optimizer = SODLAdamW(
            list(model.named_parameters()),
            state_store=state_store,
            origin_id="resume-run",
            block_size=2,
            flush_every=1,
            prefetch_lookahead=2,
            async_writeback=True,
            lr=5e-4,
        )

        x = torch.randn(6, 4)
        y = torch.randn(6, 2)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.wait_for_pending_flush()

        external = optimizer.external_state_dict()
        resumed = SODLAdamW(
            list(model.named_parameters()),
            state_store=state_store,
            origin_id="resume-run",
            block_size=2,
            flush_every=1,
            prefetch_lookahead=1,
            async_writeback=False,
            lr=5e-4,
        )
        resumed.load_state_dict(external)
        assert resumed.state_store.manifest("resume-run").blocks

        mismatched = SODLAdamW(
            list(model.named_parameters()),
            state_store=state_store,
            origin_id="resume-run",
            block_size=1,
            flush_every=1,
            lr=5e-4,
        )
        try:
            mismatched.load_state_dict(external)
        except ValueError as exc:
            assert "layout mismatch" in str(exc).lower()
        else:
            raise AssertionError("Expected layout mismatch validation to fail")


def test_sodl_adamw_keeps_pin_window_bounded() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyNet()
        state_store = OptimizerStateStore(
            f"{tmpdir}/optimizer_blobs",
            f"{tmpdir}/optimizer_registry",
            cache_capacity=8,
            writeback_threshold=8,
        )
        optimizer = SODLAdamW(
            list(model.named_parameters()),
            state_store=state_store,
            origin_id="pin-window-run",
            block_size=1,
            flush_every=1,
            prefetch_lookahead=1,
            async_writeback=False,
            lr=5e-4,
        )

        for _ in range(2):
            optimizer.zero_grad()
            x = torch.randn(6, 4)
            y = torch.randn(6, 2)
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        stats = state_store.cache_stats()
        assert stats.pinned_entries <= 1


def test_sodl_adamw_uses_batch_state_store_methods() -> None:
    class SpyStateStore:
        def __init__(self) -> None:
            self.load_block_calls = 0
            self.store_block_calls = 0
            self.load_blocks_calls = 0
            self.store_blocks_calls = 0
            self.flush_blocks_calls = 0
            self._payloads: dict[str, bytes] = {}

        def load_block(self, origin_id: str, block_id: str) -> bytes:
            self.load_block_calls += 1
            raise FileNotFoundError(block_id)

        def store_block(self, origin_id: str, block_id: str, state: bytes, *, step: int = 0, shard_key=None, metadata=None):
            self.store_block_calls += 1
            self._payloads[block_id] = bytes(state)

        def load_blocks(self, origin_id: str, block_ids: list[str]) -> dict[str, bytes]:
            self.load_blocks_calls += 1
            return {block_id: self._payloads[block_id] for block_id in block_ids if block_id in self._payloads}

        def store_blocks(self, origin_id: str, blocks: list[dict[str, object]]):
            self.store_blocks_calls += 1
            for block in blocks:
                self._payloads[str(block["block_id"])] = bytes(block["payload"])
            return []

        def flush_blocks(self, origin_id: str, block_ids: list[str]):
            self.flush_blocks_calls += 1
            return type("Manifest", (), {"blocks": {}})()

        def flush_origin(self, origin_id: str):
            return type("Manifest", (), {"blocks": {}})()

        def manifest(self, origin_id: str):
            return type("Manifest", (), {"schema": "sodl-v1", "origin_id": origin_id, "updated_at": "", "blocks": {}})()

        def pin_blocks(self, origin_id: str, block_ids: list[str]) -> None:
            return None

        def unpin_blocks(self, origin_id: str, block_ids: list[str]) -> None:
            return None

        def prefetch_blocks(self, origin_id: str, block_ids: list[str]) -> int:
            return 0

    model = TinyNet()
    store = SpyStateStore()
    optimizer = SODLAdamW(
        list(model.named_parameters()),
        state_store=store,  # type: ignore[arg-type]
        origin_id="spy-run",
        block_size=1,
        flush_every=1,
        lr=5e-4,
    )

    optimizer.zero_grad()
    x = torch.randn(6, 4)
    y = torch.randn(6, 2)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()

    assert store.load_blocks_calls == 1
    assert store.store_blocks_calls == 1
    assert store.load_block_calls == 0
    assert store.store_block_calls == 0


def test_sodl_adamw_can_force_legacy_per_block_state_store_methods() -> None:
    class SpyStateStore:
        def __init__(self) -> None:
            self.load_block_calls = 0
            self.store_block_calls = 0
            self.load_blocks_calls = 0
            self.store_blocks_calls = 0
            self.flush_blocks_calls = 0
            self._payloads: dict[str, bytes] = {}

        def load_block(self, origin_id: str, block_id: str) -> bytes:
            self.load_block_calls += 1
            if block_id not in self._payloads:
                raise FileNotFoundError(block_id)
            return self._payloads[block_id]

        def store_block(self, origin_id: str, block_id: str, state: bytes, *, step: int = 0, shard_key=None, metadata=None):
            self.store_block_calls += 1
            self._payloads[block_id] = bytes(state)
            return None

        def load_blocks(self, origin_id: str, block_ids: list[str]) -> dict[str, bytes]:
            self.load_blocks_calls += 1
            return {block_id: self._payloads[block_id] for block_id in block_ids if block_id in self._payloads}

        def store_blocks(self, origin_id: str, blocks: list[dict[str, object]]):
            self.store_blocks_calls += 1
            for block in blocks:
                self._payloads[str(block["block_id"])] = bytes(block["payload"])
            return []

        def flush_blocks(self, origin_id: str, block_ids: list[str]):
            self.flush_blocks_calls += 1
            return type("Manifest", (), {"blocks": {}})()

        def flush_origin(self, origin_id: str):
            return type("Manifest", (), {"blocks": {}})()

        def manifest(self, origin_id: str):
            return type("Manifest", (), {"schema": "sodl-v1", "origin_id": origin_id, "updated_at": "", "blocks": {}})()

        def pin_blocks(self, origin_id: str, block_ids: list[str]) -> None:
            return None

        def unpin_blocks(self, origin_id: str, block_ids: list[str]) -> None:
            return None

        def prefetch_blocks(self, origin_id: str, block_ids: list[str]) -> int:
            return 0

    model = TinyNet()
    store = SpyStateStore()
    optimizer = SODLAdamW(
        list(model.named_parameters()),
        state_store=store,  # type: ignore[arg-type]
        origin_id="spy-run-legacy",
        block_size=1,
        flush_every=1,
        use_batch_state_ops=False,
        lr=5e-4,
    )

    optimizer.zero_grad()
    x = torch.randn(6, 4)
    y = torch.randn(6, 2)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()

    assert store.load_block_calls > 0
    assert store.store_block_calls > 0
    assert store.load_blocks_calls == 0
    assert store.store_blocks_calls == 0


def test_weight_store_service_optimizer_helpers() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        service = WeightStoreService(f"{tmpdir}/blobs")
        optimizer_store = service.create_optimizer_state_store(
            registry_dir=f"{tmpdir}/optimizer_registry",
            cache_capacity=6,
            writeback_threshold=2,
        )
        service.configure_optimizer_hot_cache(optimizer_store, 3)
        service.pin_optimizer_blocks(optimizer_store, "run-2", ["block-a"])
        service.staged_flush_optimizer_state(optimizer_store, "run-2", ["block-a"])
        service.release_optimizer_blocks(optimizer_store, "run-2", ["block-a"])
        assert service.evict_optimizer_blocks(optimizer_store, "run-2", ["block-a"]) >= 0
        service.flush_optimizer_state(optimizer_store, "run-2")
        stats = optimizer_store.cache_stats()
        assert stats.cache_capacity == 3
