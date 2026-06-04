from __future__ import annotations

import tempfile

import numpy as np

from sodl_weights import BlobStore
from sodl_weights.artifact_store import ArtifactStore
from sodl_weights.vector_index import SODLVectorIndex, VectorIndexManifest


def _setup_store():
    tmpdir = tempfile.mkdtemp()
    blob_store = BlobStore(tmpdir + "/blobs")
    artifact_store = ArtifactStore(blob_store, tmpdir + "/manifests")
    vector_index = SODLVectorIndex(artifact_store, tmpdir + "/vector-indexes")
    return vector_index, tmpdir


def test_vector_index_build_and_query() -> None:
    index, tmpdir = _setup_store()
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    manifest = index.build(
        "origin:vec-test",
        vectors,
        ids=["python", "rust", "python-docs"],
        metadata=[{"lang": "py"}, {"lang": "rs"}, {"lang": "py"}],
        index_name="docs",
        corpus_version="corpus:v1",
        shard_size=2,
    )
    results = index.query(manifest, np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=2)

    assert manifest.total_vectors == 3
    assert results[0].item_id == "python"
    assert results[1].item_id == "python-docs"

    reloaded = index.load_manifest(f"{tmpdir}/vector-indexes/origin__vec-test-docs.json")
    assert isinstance(reloaded, VectorIndexManifest)
    assert reloaded.corpus_version == "corpus:v1"


def test_vector_index_tracks_corpus_version_and_dedupes_shards() -> None:
    index, _ = _setup_store()
    vectors = np.eye(4, dtype=np.float32)
    first = index.build(
        "origin:vec-dedup",
        vectors,
        index_name="retrieval",
        corpus_version="corpus:v1",
        shard_size=2,
    )
    second = index.build(
        "origin:vec-dedup",
        vectors,
        index_name="retrieval",
        corpus_version="corpus:v2",
        shard_size=2,
    )

    assert [shard.vector_blob_id for shard in first.shards] == [
        shard.vector_blob_id for shard in second.shards
    ]
    assert second.corpus_version == "corpus:v2"
