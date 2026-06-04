import pytest
import tempfile
import torch
import torch.nn as nn

from sodl_weights import BlobStore
from sodl_weights.checkpoint import CheckpointManager, CheckpointRecord
from sodl_weights.offload_optimizer import SODLAdamW
from sodl_weights.optimizer_state import OptimizerStateStore


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def ckpt_setup():
    tmpdir = tempfile.mkdtemp()
    blob_store = BlobStore(tmpdir + "/blobs")
    mgr = CheckpointManager(blob_store, tmpdir + "/registry")
    return mgr, tmpdir


class TestCheckpointManager:
    def test_save_and_load(self, ckpt_setup):
        mgr, _ = ckpt_setup
        model = SimpleModel()
        ckpt_id = mgr.save_checkpoint(model, step=100, origin_id="run-1", loss=0.5)
        assert ckpt_id.startswith("ckpt:")
        
        state = mgr.load_checkpoint("run-1", ckpt_id)
        assert "model" in state
        assert state["step"] == 100

    def test_load_latest(self, ckpt_setup):
        mgr, _ = ckpt_setup
        model = SimpleModel()
        mgr.save_checkpoint(model, step=100, origin_id="run-1", loss=0.5)
        mgr.save_checkpoint(model, step=200, origin_id="run-1", loss=0.3)
        
        state = mgr.load_checkpoint("run-1")  # no ID = latest
        assert state["step"] == 200

    def test_save_with_optimizer(self, ckpt_setup):
        mgr, _ = ckpt_setup
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        ckpt_id = mgr.save_checkpoint(model, optimizer, step=50, origin_id="run-1")
        state = mgr.load_checkpoint("run-1", ckpt_id)
        assert "optimizer" in state

    def test_list_checkpoints(self, ckpt_setup):
        mgr, _ = ckpt_setup
        model = SimpleModel()
        mgr.save_checkpoint(model, step=100, origin_id="run-1")
        mgr.save_checkpoint(model, step=200, origin_id="run-1")
        mgr.save_checkpoint(model, step=300, origin_id="run-1")
        
        records = mgr.list_checkpoints("run-1")
        assert len(records) == 3
        assert records[0].step == 100
        assert records[2].step == 300

    def test_get_latest(self, ckpt_setup):
        mgr, _ = ckpt_setup
        model = SimpleModel()
        mgr.save_checkpoint(model, step=100, origin_id="run-1")
        mgr.save_checkpoint(model, step=500, origin_id="run-1")
        
        latest = mgr.get_latest("run-1")
        assert latest.step == 500

    def test_delete_checkpoint(self, ckpt_setup):
        mgr, _ = ckpt_setup
        model = SimpleModel()
        ckpt_id = mgr.save_checkpoint(model, step=100, origin_id="run-1")
        assert mgr.delete_checkpoint("run-1", ckpt_id)
        assert len(mgr.list_checkpoints("run-1")) == 0

    def test_delete_checkpoint_reclaims_unique_blob(self, ckpt_setup):
        mgr, tmpdir = ckpt_setup
        model = SimpleModel()
        ckpt_id = mgr.save_checkpoint(model, step=100, origin_id="run-1")
        blob_id = mgr.get_latest("run-1").blob_id
        blob_store = BlobStore(tmpdir + "/blobs")

        assert blob_store.has(blob_id)
        assert mgr.delete_checkpoint("run-1", ckpt_id)
        assert not blob_store.has(blob_id)

    def test_delete_checkpoint_keeps_shared_blob(self, ckpt_setup):
        mgr, tmpdir = ckpt_setup
        model = SimpleModel()
        first_id = mgr.save_checkpoint(model, step=100, origin_id="run-1")
        second_id = mgr.save_checkpoint(model, step=100, origin_id="run-1")
        records = mgr.list_checkpoints("run-1")
        assert len(records) == 2
        assert records[0].blob_id == records[1].blob_id
        shared_blob_id = records[0].blob_id
        blob_store = BlobStore(tmpdir + "/blobs")

        assert mgr.delete_checkpoint("run-1", first_id)
        assert blob_store.has(shared_blob_id)
        assert mgr.get_latest("run-1").checkpoint_id == second_id

    def test_max_checkpoints(self):
        tmpdir = tempfile.mkdtemp()
        blob_store = BlobStore(tmpdir + "/blobs")
        mgr = CheckpointManager(blob_store, tmpdir + "/registry", max_checkpoints=2)
        
        model = SimpleModel()
        with torch.no_grad():
            model.linear.weight.fill_(0.1)
        first_id = mgr.save_checkpoint(model, step=100, origin_id="run-1")
        first_blob_id = mgr.get_latest("run-1").blob_id
        mgr.save_checkpoint(model, step=200, origin_id="run-1")
        with torch.no_grad():
            model.linear.weight.fill_(0.3)
        mgr.save_checkpoint(model, step=300, origin_id="run-1")
        
        records = mgr.list_checkpoints("run-1")
        assert len(records) == 2
        assert records[0].step == 200  # oldest (100) was evicted
        assert all(record.checkpoint_id != first_id for record in records)
        assert not blob_store.has(first_blob_id)

    def test_diff_checkpoints(self, ckpt_setup):
        mgr, _ = ckpt_setup
        model = SimpleModel()
        id1 = mgr.save_checkpoint(model, step=100, origin_id="run-1",
                                  loss=0.5, metrics={"acc": 0.8})
        id2 = mgr.save_checkpoint(model, step=200, origin_id="run-1",
                                  loss=0.3, metrics={"acc": 0.9})
        diff = mgr.diff_checkpoints("run-1", id1, id2)
        assert diff["step_delta"] == 100
        assert diff["loss_delta"] == pytest.approx(-0.2)
        assert diff["metric_deltas"]["acc"] == pytest.approx(0.1)

    def test_model_state_restored(self, ckpt_setup):
        mgr, _ = ckpt_setup
        model1 = SimpleModel()
        # Set known weights
        with torch.no_grad():
            model1.linear.weight.fill_(0.42)
        
        mgr.save_checkpoint(model1, step=1, origin_id="run-1")
        
        model2 = SimpleModel()
        state = mgr.load_checkpoint("run-1")
        model2.load_state_dict(state["model"])
        
        assert torch.allclose(model1.linear.weight, model2.linear.weight)

    def test_no_checkpoints_raises(self, ckpt_setup):
        mgr, _ = ckpt_setup
        with pytest.raises(FileNotFoundError):
            mgr.load_checkpoint("nonexistent")

    def test_metrics_storage(self, ckpt_setup):
        mgr, _ = ckpt_setup
        model = SimpleModel()
        mgr.save_checkpoint(model, step=1, origin_id="run-1",
                           metrics={"perplexity": 12.5, "speed": 500.0})
        record = mgr.get_latest("run-1")
        assert record.metrics["perplexity"] == 12.5
        assert record.metrics["speed"] == 500.0

    def test_checkpoint_lineage(self, ckpt_setup):
        mgr, _ = ckpt_setup
        model = SimpleModel()
        root_id = mgr.save_checkpoint(model, step=1, origin_id="run-1", stage="stage1")
        child_id = mgr.save_checkpoint(
            model,
            step=2,
            origin_id="run-1",
            parent_checkpoint_id=root_id,
            stage="stage2",
        )
        lineage = mgr.get_lineage("run-1", child_id)
        assert [record.checkpoint_id for record in lineage] == [root_id, child_id]
        assert lineage[-1].stage == "stage2"

    def test_resolve_resume(self, ckpt_setup):
        mgr, _ = ckpt_setup
        model = SimpleModel()
        root_id = mgr.save_checkpoint(model, step=10, origin_id="run-1", metadata={"phase": "warmup"})
        bundle = mgr.resolve_resume("run-1", root_id)
        assert bundle["record"]["checkpoint_id"] == root_id
        assert bundle["record"]["metadata"]["phase"] == "warmup"
        assert bundle["lineage"][0]["checkpoint_id"] == root_id
        assert "model" in bundle["state"]

    def test_externalized_optimizer_record_and_dataset_lineage(self, ckpt_setup):
        mgr, tmpdir = ckpt_setup
        model = SimpleModel()
        state_store = OptimizerStateStore(
            tmpdir + "/optimizer_blobs",
            tmpdir + "/optimizer_registry",
            cache_capacity=4,
            writeback_threshold=1,
        )
        optimizer = SODLAdamW(
            list(model.named_parameters()),
            state_store=state_store,
            origin_id="run-1-opt",
            block_size=1,
            flush_every=1,
            prefetch_lookahead=1,
            lr=1e-3,
        )

        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()

        ckpt_id = mgr.save_checkpoint(
            model,
            optimizer,
            step=20,
            origin_id="run-1",
            dataset_manifests=["dataset://train-manifest.json", "dataset://eval-manifest.json"],
            knowledge_logs=["knowledge://run-1.jsonl"],
        )
        record = mgr.get_latest("run-1")
        assert record is not None
        assert record.checkpoint_id == ckpt_id
        assert record.optimizer_externalized is True
        assert record.optimizer_origin_id == "run-1-opt"
        assert record.optimizer_layout_fingerprint
        assert record.optimizer_block_count > 0
        assert record.dataset_manifests == [
            "dataset://train-manifest.json",
            "dataset://eval-manifest.json",
        ]

        bundle = mgr.resolve_resume("run-1", ckpt_id)
        assert bundle["optimizer_resume"]["externalized"] is True
        assert bundle["optimizer_resume"]["optimizer_origin_id"] == "run-1-opt"
        assert bundle["dataset_manifests"] == [
            "dataset://train-manifest.json",
            "dataset://eval-manifest.json",
        ]
        assert bundle["knowledge_logs"] == ["knowledge://run-1.jsonl"]
