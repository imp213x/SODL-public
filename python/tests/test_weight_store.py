"""Comprehensive test suite for the SODL Weight Store Python SDK.

Validates parity with the Rust implementation by testing:
- Serialisation round-trips
- Compression efficiency
- CAS deduplication
- Crypto (null, XOR) behaviour
- Integrity verification
- Pin registry with identity protection and eviction
- Full service lifecycle
"""

from __future__ import annotations

import threading
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from sodl_weights.crypto import AEADCryptoProvider, NullCrypto, XorCrypto
from sodl_weights.pin_registry import WeightPinError, WeightPinRegistry
from sodl_weights.service import WeightStoreService
from sodl_weights.store import (
    BlobStore,
    SodlIntegrityError,
    SodlNotFoundError,
    WeightBlobStore,
    compute_blob_id,
    verify_integrity,
)
from sodl_weights.types import WeightCluster, WeightPinReason


def _sample_cluster(dim: int = 32, n: int = 5) -> WeightCluster:
    return WeightCluster(
        centroid=[0.5] * dim,
        member_token_ids=list(range(n)),
        offsets=[[0.01 * i] * dim for i in range(n)],
        dim=dim,
    )


# ---------------------------------------------------------------------------
# BlobStore
# ---------------------------------------------------------------------------

class TestBlobStore:
    def test_roundtrip(self, tmp_path: Path) -> None:
        store = BlobStore(tmp_path / "blobs")
        bid = "blake3:abc123"
        data = b"hello"

        assert not store.has(bid)
        store.put(bid, data)
        assert store.has(bid)
        assert store.get(bid) == data

    def test_delete(self, tmp_path: Path) -> None:
        store = BlobStore(tmp_path / "blobs")
        bid = "blake3:del"
        store.put(bid, b"x")
        store.delete(bid)
        assert not store.has(bid)

    def test_not_found(self, tmp_path: Path) -> None:
        store = BlobStore(tmp_path / "blobs")
        with pytest.raises(SodlNotFoundError):
            store.get("blake3:nonexistent")

    def test_delegates_to_rust_bridge_when_available(self, tmp_path: Path, monkeypatch) -> None:
        events: list[tuple[str, str]] = []

        class _FakeRustStore:
            def __init__(self) -> None:
                self._data: dict[str, bytes] = {}

            def has(self, blob_id: str) -> bool:
                events.append(("has", blob_id))
                return blob_id in self._data

            def put(self, blob_id: str, data: bytes) -> None:
                events.append(("put", blob_id))
                self._data[blob_id] = bytes(data)

            def get(self, blob_id: str) -> bytes:
                events.append(("get", blob_id))
                return self._data[blob_id]

            def delete(self, blob_id: str) -> None:
                events.append(("delete", blob_id))
                self._data.pop(blob_id, None)

            def blob_count(self) -> int:
                events.append(("count", ""))
                return len(self._data)

        monkeypatch.setattr(
            "sodl_weights.store._rust_bridge.create_blob_store",
            lambda root, source_roots=None, peer_urls=None, edge_urls=None: _FakeRustStore(),
        )
        store = BlobStore(tmp_path / "blobs")
        bid = "blake3:ffi"
        store.put(bid, b"hello")
        assert store.has(bid)
        assert store.get(bid) == b"hello"
        assert store.blob_count() == 1
        store.delete(bid)

        assert ("put", bid) in events
        assert ("get", bid) in events
        assert ("count", "") in events

    def test_fetches_missing_blob_from_source_store(self, tmp_path: Path) -> None:
        source_store = BlobStore(tmp_path / "source")
        local_store = BlobStore(tmp_path / "cache", [tmp_path / "source"])
        bid = compute_blob_id(b"remote-bytes")
        source_store.put(bid, b"remote-bytes")

        assert local_store.has(bid)
        assert local_store.get(bid) == b"remote-bytes"
        assert BlobStore(tmp_path / "cache").has(bid)
        assert "cache" in local_store.replica_nodes(bid)

    def test_fetches_missing_blob_from_peer_url(self, tmp_path: Path) -> None:
        payload = b"peer-bytes"
        bid = compute_blob_id(payload)

        class _BlobHandler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path == f"/v1/blobs/{bid}":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_response(404)
                self.end_headers()

            def log_message(self, format, *args):  # noqa: A003
                return

        server = HTTPServer(("127.0.0.1", 0), _BlobHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            peer_url = f"http://127.0.0.1:{server.server_port}"
            local_store = BlobStore(tmp_path / "cache", peer_urls=[peer_url])
            assert local_store.get(bid) == payload
            replicas = local_store.replica_nodes(bid)
            assert "cache" in replicas
            assert peer_url in replicas
        finally:
            server.shutdown()
            thread.join(timeout=2.0)
            server.server_close()

    def test_ffi_blob_store_reads_legacy_flat_blob_files(self, tmp_path: Path, monkeypatch) -> None:
        migrated: dict[str, bytes] = {}

        class _MissingRustStore:
            def has(self, blob_id: str) -> bool:
                return blob_id in migrated

            def put(self, blob_id: str, data: bytes) -> None:
                migrated[blob_id] = bytes(data)

            def get(self, blob_id: str) -> bytes:
                if blob_id not in migrated:
                    raise FileNotFoundError(blob_id)
                return migrated[blob_id]

            def delete(self, blob_id: str) -> None:
                migrated.pop(blob_id, None)

            def blob_count(self) -> int:
                return len(migrated)

        monkeypatch.setattr(
            "sodl_weights.store._rust_bridge.create_blob_store",
            lambda root, source_roots=None, peer_urls=None, edge_urls=None: _MissingRustStore(),
        )

        store = BlobStore(tmp_path / "blobs")
        bid = compute_blob_id(b"legacy-flat")
        legacy_path = tmp_path / "blobs" / f"{bid.split(':', 1)[1]}.blob"
        legacy_path.write_bytes(b"legacy-flat")

        assert store.has(bid)
        assert store.get(bid) == b"legacy-flat"
        assert migrated[bid] == b"legacy-flat"


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------

class TestIntegrity:
    def test_valid(self) -> None:
        data = b"some data"
        bid = compute_blob_id(data)
        verify_integrity(bid, data)  # should not raise

    def test_uses_rust_bridge_when_available(self, monkeypatch) -> None:
        monkeypatch.setattr("sodl_weights.store._rust_bridge.compute_blob_id", lambda data: "blake3:ffi")
        monkeypatch.setattr("sodl_weights.store._rust_bridge.verify_integrity", lambda blob_id, data: True)
        assert compute_blob_id(b"x") == "blake3:ffi"
        verify_integrity("blake3:ffi", b"x")

    def test_tampered(self) -> None:
        data = b"some data"
        bid = compute_blob_id(data)
        with pytest.raises(SodlIntegrityError):
            verify_integrity(bid, b"tampered data")


# ---------------------------------------------------------------------------
# Crypto
# ---------------------------------------------------------------------------

class TestCrypto:
    def test_null_passthrough(self) -> None:
        c = NullCrypto()
        data = b"hello"
        assert c.encrypt("org1", data) == data
        assert c.decrypt("org1", data) == data

    def test_xor_roundtrip(self) -> None:
        c = XorCrypto(0xAB)
        data = b"hello SODL"
        ct = c.encrypt("org1", data)
        assert ct != data  # actually encrypted
        assert c.decrypt("org1", ct) == data  # round-trip

    def test_xor_dedup_within_origin(self) -> None:
        c = XorCrypto(0xAB)
        data = b"same"
        assert c.encrypt("org1", data) == c.encrypt("org1", data)

    def test_xor_differs_across_origins(self) -> None:
        c = XorCrypto(0xAB)
        data = b"same"
        assert c.encrypt("org1", data) != c.encrypt("org2", data)


# ---------------------------------------------------------------------------
# WeightBlobStore
# ---------------------------------------------------------------------------

class TestWeightBlobStore:
    def test_put_get_no_crypto(self, tmp_path: Path) -> None:
        store = BlobStore(tmp_path / "blobs")
        ws = WeightBlobStore(store)
        cluster = _sample_cluster(64, 10)

        stats = ws.put("origin:1", cluster)
        assert not stats.was_deduped
        assert stats.compressed_bytes < stats.raw_bytes

        back = ws.get("origin:1", stats.blob_id)
        assert back.centroid == cluster.centroid
        assert back.member_token_ids == cluster.member_token_ids
        assert back.dim == 64

    def test_dedup(self, tmp_path: Path) -> None:
        store = BlobStore(tmp_path / "blobs")
        ws = WeightBlobStore(store)
        cluster = _sample_cluster()

        s1 = ws.put("origin:1", cluster)
        s2 = ws.put("origin:1", cluster)

        assert s1.blob_id == s2.blob_id
        assert not s1.was_deduped
        assert s2.was_deduped

    def test_xor_crypto_roundtrip(self, tmp_path: Path) -> None:
        store = BlobStore(tmp_path / "blobs")
        crypto = XorCrypto(0xAB)
        ws = WeightBlobStore(store, crypto)
        cluster = _sample_cluster(64, 15)

        stats = ws.put("origin:1", cluster)
        back = ws.get("origin:1", stats.blob_id)
        assert back.centroid == cluster.centroid

    def test_xor_dedup_within_origin(self, tmp_path: Path) -> None:
        store = BlobStore(tmp_path / "blobs")
        crypto = XorCrypto(0xAB)
        ws = WeightBlobStore(store, crypto)
        cluster = _sample_cluster()

        s1 = ws.put("origin:1", cluster)
        s2 = ws.put("origin:1", cluster)
        assert s1.blob_id == s2.blob_id

    def test_xor_differs_across_origins(self, tmp_path: Path) -> None:
        store = BlobStore(tmp_path / "blobs")
        crypto = XorCrypto(0xAB)
        ws = WeightBlobStore(store, crypto)
        cluster = _sample_cluster()

        s1 = ws.put("origin:1", cluster)
        s2 = ws.put("origin:2", cluster)
        assert s1.blob_id != s2.blob_id

    def test_aead_crypto_roundtrip(self, tmp_path: Path, monkeypatch) -> None:
        class _FakeNativeAead:
            def __init__(self, master_key_hex: str | None = None) -> None:
                self._master_key_hex = master_key_hex or ("aa" * 32)

            def master_key_hex(self) -> str:
                return self._master_key_hex

            def encrypt(self, origin_id: str, plaintext: bytes) -> bytes:
                header = f"{self._master_key_hex}|{origin_id}|".encode("utf-8")
                return b"\x01" + b"x" * 24 + header + plaintext[::-1] + b"tagtagtagtagtag!"

            def decrypt(self, origin_id: str, ciphertext: bytes) -> bytes:
                body = ciphertext[25:-16]
                prefix = f"{self._master_key_hex}|{origin_id}|".encode("utf-8")
                if not body.startswith(prefix):
                    raise ValueError("wrong key or origin")
                return body[len(prefix):][::-1]

        monkeypatch.setattr(
            "sodl_weights.crypto._rust_bridge.create_aead_crypto",
            lambda master_key_hex=None: _FakeNativeAead(master_key_hex),
        )

        store = BlobStore(tmp_path / "blobs")
        crypto = AEADCryptoProvider("12" * 32)
        ws = WeightBlobStore(store, crypto)
        cluster = _sample_cluster(32, 6)

        stats = ws.put("origin:1", cluster)
        stored_bytes = store.get(stats.blob_id)
        assert stored_bytes[:1] == b"\x01"

        back = ws.get("origin:1", stats.blob_id)
        assert back.centroid == cluster.centroid
        assert back.member_token_ids == cluster.member_token_ids

    def test_aead_differs_across_origins(self, tmp_path: Path, monkeypatch) -> None:
        class _FakeNativeAead:
            def __init__(self, master_key_hex: str | None = None) -> None:
                self._master_key_hex = master_key_hex or ("aa" * 32)

            def master_key_hex(self) -> str:
                return self._master_key_hex

            def encrypt(self, origin_id: str, plaintext: bytes) -> bytes:
                header = f"{self._master_key_hex}|{origin_id}|".encode("utf-8")
                return b"\x01" + b"x" * 24 + header + plaintext[::-1] + b"tagtagtagtagtag!"

            def decrypt(self, origin_id: str, ciphertext: bytes) -> bytes:
                body = ciphertext[25:-16]
                prefix = f"{self._master_key_hex}|{origin_id}|".encode("utf-8")
                if not body.startswith(prefix):
                    raise ValueError("wrong key or origin")
                return body[len(prefix):][::-1]

        monkeypatch.setattr(
            "sodl_weights.crypto._rust_bridge.create_aead_crypto",
            lambda master_key_hex=None: _FakeNativeAead(master_key_hex),
        )

        store = BlobStore(tmp_path / "blobs")
        crypto = AEADCryptoProvider("34" * 32)
        ws = WeightBlobStore(store, crypto)
        cluster = _sample_cluster()

        s1 = ws.put("origin:1", cluster)
        s2 = ws.put("origin:2", cluster)
        assert s1.blob_id != s2.blob_id

    def test_aead_provider_reads_legacy_plain_blobs(self, tmp_path: Path, monkeypatch) -> None:
        class _FakeNativeAead:
            def __init__(self, master_key_hex: str | None = None) -> None:
                self._master_key_hex = master_key_hex or ("aa" * 32)

            def master_key_hex(self) -> str:
                return self._master_key_hex

            def encrypt(self, origin_id: str, plaintext: bytes) -> bytes:
                header = f"{self._master_key_hex}|{origin_id}|".encode("utf-8")
                return b"\x01" + b"x" * 24 + header + plaintext[::-1] + b"tagtagtagtagtag!"

            def decrypt(self, origin_id: str, ciphertext: bytes) -> bytes:
                body = ciphertext[25:-16]
                prefix = f"{self._master_key_hex}|{origin_id}|".encode("utf-8")
                if not body.startswith(prefix):
                    raise ValueError("wrong key or origin")
                return body[len(prefix):][::-1]

        monkeypatch.setattr(
            "sodl_weights.crypto._rust_bridge.create_aead_crypto",
            lambda master_key_hex=None: _FakeNativeAead(master_key_hex),
        )

        store = BlobStore(tmp_path / "blobs")
        legacy = WeightBlobStore(store)
        secure = WeightBlobStore(store, AEADCryptoProvider("56" * 32))
        cluster = _sample_cluster(24, 4)

        stats = legacy.put("origin:1", cluster)
        back = secure.get("origin:1", stats.blob_id)
        assert back.centroid == cluster.centroid

    def test_compact_q8_roundtrip_is_close(self, tmp_path: Path) -> None:
        store = BlobStore(tmp_path / "blobs")
        ws = WeightBlobStore(store, serialization_codec="compact_q8")
        cluster = _sample_cluster(64, 12)

        stats = ws.put("origin:1", cluster)
        back = ws.get("origin:1", stats.blob_id)

        assert back.member_token_ids == cluster.member_token_ids
        assert back.dim == cluster.dim
        assert pytest.approx(back.centroid, rel=0, abs=5e-3) == cluster.centroid
        for observed, expected in zip(back.offsets, cluster.offsets):
            assert pytest.approx(observed, rel=0, abs=5e-3) == expected

    def test_integrity_failure(self, tmp_path: Path) -> None:
        store = BlobStore(tmp_path / "blobs")
        ws = WeightBlobStore(store)
        cluster = _sample_cluster()

        stats = ws.put("origin:1", cluster)

        # Tamper
        raw = store.get(stats.blob_id)
        tampered = bytes([raw[0] ^ 0xFF]) + raw[1:]
        store.put(stats.blob_id, tampered)

        with pytest.raises(SodlIntegrityError):
            ws.get("origin:1", stats.blob_id)


# ---------------------------------------------------------------------------
# Pin Registry
# ---------------------------------------------------------------------------

class TestPinRegistry:
    def test_pin_and_get(self) -> None:
        reg = WeightPinRegistry(10)
        c = _sample_cluster()
        reg.pin("c1", c, WeightPinReason.FREQUENT_USE)
        assert reg.is_pinned("c1")
        got = reg.get("c1")
        assert got is not None
        assert got.centroid == c.centroid

    def test_unpin(self) -> None:
        reg = WeightPinRegistry(10)
        reg.pin("c1", _sample_cluster(), WeightPinReason.FREQUENT_USE)
        assert reg.unpin("c1")
        assert not reg.is_pinned("c1")

    def test_identity_cannot_unpin(self) -> None:
        reg = WeightPinRegistry(10)
        reg.pin("c1", _sample_cluster(), WeightPinReason.IDENTITY)
        with pytest.raises(WeightPinError):
            reg.unpin("c1")

    def test_logic_cannot_unpin(self) -> None:
        reg = WeightPinRegistry(10)
        reg.pin("c1", _sample_cluster(), WeightPinReason.LOGIC)
        with pytest.raises(WeightPinError):
            reg.unpin("c1")

    def test_eviction(self) -> None:
        reg = WeightPinRegistry(3)
        for i in range(3):
            reg.pin(f"c{i}", _sample_cluster(), WeightPinReason.FREQUENT_USE)

        # Access c1 and c2 more
        reg.get("c1")
        reg.get("c1")
        reg.get("c2")

        # Pin a 4th → should evict c0 (lowest refcount)
        reg.pin("c3", _sample_cluster(), WeightPinReason.FREQUENT_USE)
        assert not reg.is_pinned("c0")
        assert reg.is_pinned("c1")

    def test_identity_survives_eviction(self) -> None:
        reg = WeightPinRegistry(2)
        reg.pin("identity", _sample_cluster(), WeightPinReason.IDENTITY)
        reg.pin("regular", _sample_cluster(), WeightPinReason.FREQUENT_USE)
        reg.pin("overflow", _sample_cluster(), WeightPinReason.FREQUENT_USE)
        assert reg.is_pinned("identity")  # survived

    def test_logic_survives_eviction(self) -> None:
        reg = WeightPinRegistry(2)
        reg.pin("logic", _sample_cluster(), WeightPinReason.LOGIC)
        reg.pin("regular", _sample_cluster(), WeightPinReason.FREQUENT_USE)
        reg.pin("overflow", _sample_cluster(), WeightPinReason.FREQUENT_USE)
        assert reg.is_pinned("logic")

    def test_refcount(self) -> None:
        reg = WeightPinRegistry(10)
        reg.pin("c1", _sample_cluster(), WeightPinReason.FREQUENT_USE)
        assert reg.refcount("c1") == 1
        reg.get("c1")
        assert reg.refcount("c1") == 2


# ---------------------------------------------------------------------------
# WeightStoreService
# ---------------------------------------------------------------------------

class TestWeightStoreService:
    def test_full_lifecycle(self, tmp_path: Path) -> None:
        svc = WeightStoreService(str(tmp_path / "blobs"), cache_capacity=64)

        model = svc.create_model("carla-qwen3-4b", "Q4_K_M")
        assert model.model_name == "carla-qwen3-4b"

        cluster = _sample_cluster(64, 10)
        stats = svc.store_cluster(model.origin_id, cluster)
        assert not stats.was_deduped

        loaded = svc.load_cluster(model.origin_id, stats.blob_id)
        assert loaded.centroid == cluster.centroid
        assert svc.is_cached(stats.blob_id)

        # Second load from cache
        cached = svc.load_cluster(model.origin_id, stats.blob_id)
        assert cached.centroid == cluster.centroid
        assert svc.cluster_refcount(stats.blob_id) == 2  # pin=1 + cache_hit=2

    def test_bulk_import(self, tmp_path: Path) -> None:
        svc = WeightStoreService(str(tmp_path / "blobs"))
        model = svc.create_model("bulk-test", "Q4_K_M")

        clusters = [
            WeightCluster(
                centroid=[i * 0.1] * 32,
                member_token_ids=[i],
                offsets=[[0.0] * 32],
                dim=32,
            )
            for i in range(20)
        ]

        summary = svc.import_clusters(model.origin_id, clusters)
        assert summary.total_clusters == 20
        assert summary.total_blobs_stored == 20
        assert summary.deduped_blobs == 0
        assert summary.total_stored_bytes < summary.total_raw_bytes

    def test_bulk_import_with_duplicates(self, tmp_path: Path) -> None:
        svc = WeightStoreService(str(tmp_path / "blobs"))
        model = svc.create_model("dedup-test", "Q4_K_M")

        c = _sample_cluster(16, 3)
        clusters = [c] * 5

        summary = svc.import_clusters(model.origin_id, clusters)
        assert summary.total_clusters == 5
        assert summary.total_blobs_stored == 1
        assert summary.deduped_blobs == 4

    def test_identity_pin_protection(self, tmp_path: Path) -> None:
        svc = WeightStoreService(str(tmp_path / "blobs"))
        model = svc.create_model("pin-test", "Q4_K_M")

        stats = svc.store_cluster(model.origin_id, _sample_cluster())
        svc.pin_identity_cluster(model.origin_id, stats.blob_id)

        with pytest.raises(WeightPinError):
            svc.evict_cluster(stats.blob_id)

    def test_logic_pin_protection(self, tmp_path: Path) -> None:
        svc = WeightStoreService(str(tmp_path / "blobs"))
        model = svc.create_model("logic-pin-test", "Q4_K_M")

        stats = svc.store_cluster(model.origin_id, _sample_cluster())
        svc.pin_logic_cluster(model.origin_id, stats.blob_id)

        with pytest.raises(WeightPinError):
            svc.evict_cluster(stats.blob_id)

    def test_model_lookup(self, tmp_path: Path) -> None:
        svc = WeightStoreService(str(tmp_path / "blobs"))
        model = svc.create_model("test-model", "F16")

        found = svc.get_model_by_name("test-model")
        assert found.origin_id == model.origin_id

        with pytest.raises(KeyError):
            svc.get_model_by_name("nonexistent")

    def test_register_existing_origin(self, tmp_path: Path) -> None:
        svc = WeightStoreService(str(tmp_path / "blobs"))

        model = svc.register_model("origin:stable", "stable-model", "F16")
        found = svc.get_model("origin:stable")

        assert model.origin_id == "origin:stable"
        assert found.model_name == "stable-model"

    def test_ensure_model_reuses_supplied_origin_id(self, tmp_path: Path) -> None:
        svc = WeightStoreService(str(tmp_path / "blobs"))

        first = svc.ensure_model("stable-model", "F16", origin_id="origin:stable")
        second = svc.ensure_model("stable-model", "F16", origin_id="origin:stable")

        assert first.origin_id == "origin:stable"
        assert second.origin_id == "origin:stable"
        assert second is first

    def test_service_persists_pin_registry_on_close(self, tmp_path: Path) -> None:
        pin_registry_path = tmp_path / "pins.json"
        svc = WeightStoreService(
            str(tmp_path / "blobs"),
            cache_capacity=8,
            pin_registry_path=pin_registry_path,
        )
        model = svc.create_model("persist-model", "Q4_K_M")
        stats = svc.store_cluster(model.origin_id, _sample_cluster())
        svc.load_cluster(model.origin_id, stats.blob_id)
        svc.close()

        assert pin_registry_path.exists()
        reloaded = WeightStoreService(
            str(tmp_path / "blobs"),
            cache_capacity=8,
            pin_registry_path=pin_registry_path,
        )
        assert reloaded.is_cached(stats.blob_id)
        reloaded.close()

    def test_placeholder_registry_entry_hydrates_on_load(self, tmp_path: Path) -> None:
        pin_registry_path = tmp_path / "pins.json"
        svc = WeightStoreService(
            str(tmp_path / "blobs"),
            cache_capacity=8,
            pin_registry_path=pin_registry_path,
        )
        model = svc.create_model("hydrate-model", "Q4_K_M")
        cluster = _sample_cluster(16, 3)
        stats = svc.store_cluster(model.origin_id, cluster)
        svc.load_cluster(model.origin_id, stats.blob_id)
        svc.close()

        reloaded = WeightStoreService(
            str(tmp_path / "blobs"),
            cache_capacity=8,
            pin_registry_path=pin_registry_path,
        )
        hydrated = reloaded.load_cluster(model.origin_id, stats.blob_id)
        assert hydrated.centroid == cluster.centroid
        assert hydrated.dim == cluster.dim
        reloaded.close()

    def test_service_loads_cluster_from_source_blob_dir(self, tmp_path: Path) -> None:
        source_service = WeightStoreService(str(tmp_path / "source"))
        origin = source_service.create_model("dist-model", "Q4_K_M")
        cluster = _sample_cluster(16, 4)
        stats = source_service.store_cluster(origin.origin_id, cluster)

        local_service = WeightStoreService(
            str(tmp_path / "cache"),
            source_blob_dirs=[tmp_path / "source"],
        )
        loaded = local_service.load_cluster(origin.origin_id, stats.blob_id)

        assert loaded.centroid == cluster.centroid
        assert local_service._store._store.has(stats.blob_id)
        assert any(node.startswith("source:0:") for node in local_service.replica_nodes(stats.blob_id))
        source_service.close()
        local_service.close()
