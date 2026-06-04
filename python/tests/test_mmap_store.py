import pytest
import tempfile
from pathlib import Path

from sodl_weights import BlobStore
from sodl_weights.store import compute_blob_id
from sodl_weights.mmap_store import MMapBlobReader, ArenaReader


@pytest.fixture
def mmap_setup():
    tmpdir = tempfile.mkdtemp()
    blob_store = BlobStore(tmpdir)
    
    # Store some test blobs
    blobs = {}
    for i in range(3):
        data = f"blob-content-{i}-{'x' * 1000}".encode()
        blob_id = compute_blob_id(data)
        blob_store.put(blob_id, data)
        blobs[blob_id] = data
    
    return MMapBlobReader(tmpdir), blobs, tmpdir


class TestMMapBlobReader:
    def test_open_and_read(self, mmap_setup):
        reader, blobs, _ = mmap_setup
        for blob_id, expected in blobs.items():
            with reader.open(blob_id) as view:
                assert view[:] == expected
                assert len(view) == len(expected)

    def test_slice_access(self, mmap_setup):
        reader, blobs, _ = mmap_setup
        blob_id = list(blobs.keys())[0]
        expected = blobs[blob_id]
        
        with reader.open(blob_id) as view:
            assert view[0:10] == expected[0:10]
            assert view[-10:] == expected[-10:]

    def test_read_chunks(self, mmap_setup):
        reader, blobs, _ = mmap_setup
        blob_id = list(blobs.keys())[0]
        expected = blobs[blob_id]
        
        reassembled = b""
        for chunk in reader.read_chunks(blob_id, chunk_size=100):
            reassembled += chunk
        assert reassembled == expected

    def test_blob_size(self, mmap_setup):
        reader, blobs, _ = mmap_setup
        for blob_id, data in blobs.items():
            assert reader.blob_size(blob_id) == len(data)

    def test_exists(self, mmap_setup):
        reader, blobs, _ = mmap_setup
        for blob_id in blobs:
            assert reader.exists(blob_id)
        assert not reader.exists("blake3:nonexistent")

    def test_missing_blob_raises(self, mmap_setup):
        reader, _, _ = mmap_setup
        with pytest.raises(FileNotFoundError):
            with reader.open("blake3:nonexistent"):
                pass


class TestArenaReader:
    def test_scan(self, mmap_setup):
        _, blobs, tmpdir = mmap_setup
        arena = ArenaReader(tmpdir)
        blob_ids = list(blobs.keys())
        
        chunks = list(arena.scan(blob_ids, chunk_size=500))
        assert len(chunks) > 0
        assert arena.stats["blobs_read"] == 3

    def test_read_all(self, mmap_setup):
        _, blobs, tmpdir = mmap_setup
        arena = ArenaReader(tmpdir)
        blob_ids = list(blobs.keys())
        
        result = arena.read_all(blob_ids)
        for blob_id in blob_ids:
            assert result[blob_id] == blobs[blob_id]
