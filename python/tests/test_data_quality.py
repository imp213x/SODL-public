from __future__ import annotations

import tempfile

from sodl_weights import BlobStore
from sodl_weights.artifact_store import ArtifactStore
from sodl_weights.data_quality import DataQualityScorer


def _setup_scorer():
    tmpdir = tempfile.mkdtemp()
    blob_store = BlobStore(tmpdir + "/blobs")
    artifact_store = ArtifactStore(blob_store, tmpdir + "/manifests")
    scorer = DataQualityScorer(artifact_store)
    return scorer


def test_score_text_chunk_and_curriculum() -> None:
    scorer = _setup_scorer()
    records = scorer.score_samples(
        "origin:quality",
        [
            {"chunk_id": "a", "text": "print('hello world')\nreturn 1", "loss_before": 2.0, "loss_after": 1.0},
            {"chunk_id": "b", "text": "spam spam spam spam"},
        ],
    )

    assert len(records) == 2
    assert records[0].quality_score > records[1].quality_score
    curriculum = scorer.curriculum(records, min_score=0.1)
    assert curriculum[0] == "a"


def test_store_and_load_quality_jsonl() -> None:
    scorer = _setup_scorer()
    records = scorer.score_samples(
        "origin:quality-store",
        [
            {"chunk_id": "x", "text": "Useful training sample with good diversity."},
            {"chunk_id": "y", "text": "repeat repeat repeat"},
        ],
    )
    artifact = scorer.store_records("origin:quality-store", records, name="quality-pass")
    loaded = scorer.load_records(artifact.blob_id)

    assert [record.chunk_id for record in loaded] == ["x", "y"]
    assert loaded[0].signal_scores
