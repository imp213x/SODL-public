"""Optional Rust acceleration bridge for the SODL Python SDK."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
import warnings
from typing import Any, Sequence

logger = logging.getLogger(__name__)

_MODULE_CANDIDATES = ("sodl_native", "sodl_python_ffi")
_ACCELERATED_OPS = (
    "blob_store",
    "hashing",
    "compression",
    "integrity",
    "optimizer_state",
    "checkpoint_store",
    "weight_manifest_store",
    "weight_origin_registry",
    "aead_crypto",
)
_ffi: Any | None = None
_status: "RustBridgeStatus | None" = None
_warning_emitted = False


@dataclass(frozen=True, slots=True)
class RustBridgeStatus:
    active: bool
    module_name: str | None
    import_error: str | None
    warning: str | None
    accelerated_ops: tuple[str, ...]

    def summary(self) -> str:
        if self.active:
            return (
                f"native bridge active via {self.module_name}; "
                f"accelerating {', '.join(self.accelerated_ops)}"
            )
        return (
            "native bridge unavailable; using pure Python fallback. "
            + (self.warning or "Install `sodl-native` for 10-50x faster blob operations.")
        )


def _emit_warning_once(message: str) -> None:
    global _warning_emitted
    if _warning_emitted:
        return
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    logger.warning(message)
    _warning_emitted = True


def _load_bridge(*, warn_on_fallback: bool) -> Any | None:
    global _ffi, _status
    if _status is not None:
        if warn_on_fallback and not _status.active and _status.warning:
            _emit_warning_once(_status.warning)
        return _ffi

    errors: list[str] = []
    for module_name in _MODULE_CANDIDATES:
        try:  # pragma: no cover - depends on optional native build
            _ffi = importlib.import_module(module_name)
            _status = RustBridgeStatus(
                active=True,
                module_name=module_name,
                import_error=None,
                warning=None,
                accelerated_ops=_ACCELERATED_OPS,
            )
            return _ffi
        except Exception as exc:  # pragma: no cover - exercised via tests with monkeypatch
            errors.append(f"{module_name}: {exc}")

    warning = "Install `sodl-native` for 10-50x faster blob operations."
    _ffi = None
    _status = RustBridgeStatus(
        active=False,
        module_name=None,
        import_error=" | ".join(errors) if errors else None,
        warning=warning,
        accelerated_ops=(),
    )
    if warn_on_fallback:
        _emit_warning_once(warning)
    return None


def status() -> RustBridgeStatus:
    _load_bridge(warn_on_fallback=False)
    assert _status is not None
    return _status


def status_summary() -> str:
    return status().summary()


def available() -> bool:
    return status().active


def create_blob_store(
    root: str,
    source_roots: Sequence[str] | None = None,
    peer_urls: Sequence[str] | None = None,
    edge_urls: Sequence[str] | None = None,
) -> Any | None:
    ffi = _load_bridge(warn_on_fallback=True)
    if ffi is None:
        return None
    kwargs: list[Any] = [root]
    if source_roots or peer_urls or edge_urls:
        kwargs.append(list(source_roots or []))
        kwargs.append(list(peer_urls or []))
        kwargs.append(list(edge_urls or []))
    return ffi.PyFsBlobStore(*kwargs)


def create_optimizer_state_store(
    blob_root: str,
    registry_dir: str | None = None,
    *,
    compression_level: int = 3,
    cache_capacity: int = 32,
    writeback_threshold: int = 8,
) -> Any | None:
    ffi = _load_bridge(warn_on_fallback=True)
    if ffi is None or not hasattr(ffi, "PyOptimizerStateStore"):
        return None
    return ffi.PyOptimizerStateStore(
        blob_root,
        registry_dir,
        compression_level,
        cache_capacity,
        writeback_threshold,
    )


def create_checkpoint_store(
    blob_root: str,
    registry_dir: str | None = None,
    *,
    compression_level: int = 3,
    max_checkpoints: int = 0,
) -> Any | None:
    ffi = _load_bridge(warn_on_fallback=True)
    if ffi is None or not hasattr(ffi, "PyCheckpointStore"):
        return None
    return ffi.PyCheckpointStore(
        blob_root,
        registry_dir,
        compression_level,
        max_checkpoints,
    )


def create_weight_manifest_store(
    manifest_path: str,
    blob_root: str | None = None,
) -> Any | None:
    ffi = _load_bridge(warn_on_fallback=True)
    if ffi is None or not hasattr(ffi, "PyWeightManifestStore"):
        return None
    return ffi.PyWeightManifestStore(
        manifest_path,
        blob_root,
    )


def create_weight_origin_registry() -> Any | None:
    ffi = _load_bridge(warn_on_fallback=True)
    if ffi is None or not hasattr(ffi, "PyWeightOriginRegistry"):
        return None
    return ffi.PyWeightOriginRegistry()


def create_aead_crypto(master_key_hex: str | None = None) -> Any | None:
    ffi = _load_bridge(warn_on_fallback=True)
    if ffi is None or not hasattr(ffi, "PyAeadCrypto"):
        return None
    return ffi.PyAeadCrypto(master_key_hex)


def compute_blob_id(data: bytes) -> str | None:
    ffi = _load_bridge(warn_on_fallback=True)
    if ffi is None:
        return None
    return str(ffi.compute_blob_id_py(data))


def verify_integrity(blob_id: str, data: bytes) -> bool:
    ffi = _load_bridge(warn_on_fallback=True)
    if ffi is None:
        return False
    ffi.verify_integrity_py(blob_id, data)
    return True


def compress_zstd(data: bytes, level: int = 3) -> bytes | None:
    ffi = _load_bridge(warn_on_fallback=True)
    if ffi is None:
        return None
    return bytes(ffi.compress_zstd(data, level))


def decompress_zstd(data: bytes) -> bytes | None:
    ffi = _load_bridge(warn_on_fallback=True)
    if ffi is None:
        return None
    return bytes(ffi.decompress_zstd(data))
