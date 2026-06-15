import pytest
import tempfile

from sodl_weights import BlobStore
from sodl_weights.store import compute_blob_id
from sodl_weights.consistency import ConsistencyChecker, BlobCheckResult, ConsistencyReport


@pytest.fixture
def checker_setup():
    tmpdir = tempfile.mkdtemp()
    store = BlobStore(tmpdir)
    checker = ConsistencyChecker(store)
    return checker, store, tmpdir


class TestConsistencyChecker:
    def test_check_healthy_blob(self, checker_setup):
        checker, store, _ = checker_setup
        data = b"healthy test data"
        blob_id = compute_blob_id(data)
        store.put(blob_id, data)
        
        result = checker.check_blob(blob_id)
        assert result.exists_locally
        assert result.local_valid
        assert "local" in result.healthy_nodes

    def test_check_missing_blob(self, checker_setup):
        checker, _, _ = checker_setup
        result = checker.check_blob("blake3:nonexistent123")
        assert not result.exists_locally
        assert not result.local_valid

    def test_check_corrupt_blob(self, checker_setup):
        checker, store, _ = checker_setup
        data = b"original data"
        blob_id = compute_blob_id(data)
        # Write wrong data under this blob_id
        store.put(blob_id, b"corrupted data!!")
        
        result = checker.check_blob(blob_id)
        assert result.exists_locally
        assert not result.local_valid
        assert "local" in result.corrupt_nodes

    def test_verify_data(self, checker_setup):
        checker, _, _ = checker_setup
        data = b"verify me"
        blob_id = compute_blob_id(data)
        assert checker.verify_data(blob_id, data)
        assert not checker.verify_data(blob_id, b"wrong")

    def test_scan_local(self, checker_setup):
        checker, store, _ = checker_setup
        blob_ids = []
        for i in range(5):
            data = f"scan-{i}".encode()
            bid = compute_blob_id(data)
            store.put(bid, data)
            blob_ids.append(bid)
        
        report = checker.scan_local(blob_ids)
        assert report.total_blobs == 5
        assert report.healthy == 5
        assert report.corrupt == 0
        assert report.healthy_pct == 100.0

    def test_find_orphaned(self, checker_setup):
        checker, store, _ = checker_setup
        data = b"orphan data"
        bid = compute_blob_id(data)
        store.put(bid, data)
        
        # Not in known set
        orphaned = checker.find_orphaned(set())
        assert len(orphaned) >= 1

        # In known set
        orphaned = checker.find_orphaned({bid})
        assert len(orphaned) == 0

    def test_find_orphaned_normalizes_blob_suffix_and_raw_hash(self, checker_setup):
        checker, store, _ = checker_setup
        data = b"live blob protected by index"
        bid = compute_blob_id(data)
        store.put(bid, data)

        raw_hash = bid.split(":", 1)[1]

        assert checker.find_orphaned({raw_hash}) == []
        assert checker.find_orphaned({f"{raw_hash}.blob"}) == []

    def test_repair_blob(self, checker_setup):
        checker, store, _ = checker_setup
        data = b"repair test"
        blob_id = compute_blob_id(data)
        
        # Write corrupt
        store.put(blob_id, b"bad data")
        assert not checker.check_blob(blob_id).local_valid
        
        # Repair
        assert checker.repair_blob(blob_id, data)
        assert checker.check_blob(blob_id).local_valid

    def test_repair_bad_source(self, checker_setup):
        checker, _, _ = checker_setup
        assert not checker.repair_blob("blake3:abc", b"mismatched data")
