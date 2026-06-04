from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace


def _reload_bridge(monkeypatch, fake_native=None):
    monkeypatch.delitem(sys.modules, "sodl_weights._rust_bridge", raising=False)
    monkeypatch.delitem(sys.modules, "sodl_native", raising=False)
    monkeypatch.delitem(sys.modules, "sodl_python_ffi", raising=False)
    if fake_native is not None:
        monkeypatch.setitem(sys.modules, "sodl_native", fake_native)
    import sodl_weights._rust_bridge as rust_bridge

    return importlib.reload(rust_bridge)


def test_reports_fallback_status_when_native_module_missing(monkeypatch) -> None:
    rust_bridge = _reload_bridge(monkeypatch)

    status = rust_bridge.status()
    assert not status.active
    assert "sodl-native" in rust_bridge.status_summary()


def test_reports_active_status_when_native_module_present(monkeypatch) -> None:
    fake_native = SimpleNamespace(
        PyFsBlobStore=object,
        compute_blob_id_py=lambda data: "blake3:ffi",
        verify_integrity_py=lambda blob_id, data: None,
        compress_zstd=lambda data, level=3: data,
        decompress_zstd=lambda data: data,
    )
    rust_bridge = _reload_bridge(monkeypatch, fake_native=fake_native)

    status = rust_bridge.status()
    assert status.active
    assert status.module_name == "sodl_native"
