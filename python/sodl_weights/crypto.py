"""Crypto providers for the SODL Weight Store.

Mirrors the Rust ``sodl-crypto`` crate interface.
"""

from __future__ import annotations

import abc
import json
import os
from pathlib import Path
from typing import Any, Sequence

from . import _rust_bridge


class CryptoProvider(abc.ABC):
    """Encrypt/decrypt bytes for a given origin."""

    @abc.abstractmethod
    def encrypt(self, origin_id: str, plaintext: bytes) -> bytes:
        ...

    @abc.abstractmethod
    def decrypt(self, origin_id: str, ciphertext: bytes) -> bytes:
        ...


class NullCrypto(CryptoProvider):
    """No-op crypto — passthrough. For development only."""

    def encrypt(self, origin_id: str, plaintext: bytes) -> bytes:
        return plaintext

    def decrypt(self, origin_id: str, ciphertext: bytes) -> bytes:
        return ciphertext


class XorCrypto(CryptoProvider):
    """Deterministic XOR "crypto" for testing dedup behaviour.

    WARNING: cryptographically broken — use only for development.
    Matches the Rust ``DevXorCrypto`` implementation.
    """

    def __init__(self, key_byte: int = 0xA5) -> None:
        self._key_byte = key_byte & 0xFF

    def _derive_mask(self, origin_id: str) -> int:
        mask = self._key_byte
        for b in origin_id.encode("utf-8"):
            mask ^= b
        return mask & 0xFF

    def encrypt(self, origin_id: str, plaintext: bytes) -> bytes:
        mask = self._derive_mask(origin_id)
        return bytes(b ^ mask for b in plaintext)

    def decrypt(self, origin_id: str, ciphertext: bytes) -> bytes:
        return self.encrypt(origin_id, ciphertext)  # XOR is symmetric


class AEADCryptoProvider(CryptoProvider):
    """Rust-backed AEAD crypto provider using the native SODL bridge.

    The native implementation is authoritative. If the native bridge is not
    available, construction fails clearly instead of silently downgrading to
    insecure behavior.

    The Rust engine derives per-origin keys from the configured master key via
    HKDF, so rotation is modeled here as a managed keyring of master keys:
    one active key for new writes plus optional legacy keys for read-back.
    """

    ENVELOPE_VERSION = 0x01
    MIN_ENVELOPE_LEN = 1 + 24 + 16

    def __init__(
        self,
        master_key_hex: str | None = None,
        *,
        legacy_key_hexes: Sequence[str] | None = None,
    ) -> None:
        native = _rust_bridge.create_aead_crypto(master_key_hex)
        if native is None:
            raise RuntimeError(
                "AEADCryptoProvider requires the native SODL bridge. "
                "Install/build `sodl-native` to enable Rust-backed AEAD encryption."
            )
        self._native = native
        self._master_key_hex = str(native.master_key_hex())
        self._legacy_key_hexes = [
            str(key_hex)
            for key_hex in (legacy_key_hexes or [])
            if str(key_hex) and str(key_hex) != self._master_key_hex
        ]
        self._legacy_natives = []
        for key_hex in self._legacy_key_hexes:
            legacy_native = _rust_bridge.create_aead_crypto(key_hex)
            if legacy_native is None:
                raise RuntimeError(
                    "AEADCryptoProvider requires the native SODL bridge to load legacy keys."
                )
            self._legacy_natives.append(legacy_native)

    @classmethod
    def generate(cls) -> "AEADCryptoProvider":
        return cls(None)

    @classmethod
    def generate_key_hex(cls) -> str:
        return cls.generate().master_key_hex

    @classmethod
    def from_env(cls, env_var: str = "SODL_MASTER_KEY_HEX") -> "AEADCryptoProvider":
        value = os.getenv(env_var)
        if not value:
            raise RuntimeError(f"Environment variable {env_var} is not set")
        return cls(value)

    @property
    def master_key_hex(self) -> str:
        return self._master_key_hex

    @property
    def legacy_key_hexes(self) -> tuple[str, ...]:
        return tuple(self._legacy_key_hexes)

    def to_keyring_dict(self) -> dict[str, Any]:
        return {
            "format": "sodl-aead-keyring-v1",
            "active_master_key_hex": self.master_key_hex,
            "legacy_master_key_hexes": list(self._legacy_key_hexes),
        }

    @classmethod
    def from_keyring_dict(cls, payload: dict[str, Any]) -> "AEADCryptoProvider":
        active = payload.get("active_master_key_hex") or payload.get("master_key_hex")
        if not active:
            raise RuntimeError("Missing active_master_key_hex in AEAD keyring payload")
        legacy = payload.get("legacy_master_key_hexes") or payload.get("legacy_keys") or []
        return cls(str(active), legacy_key_hexes=[str(item) for item in legacy])

    def save_keyring(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_keyring_dict(), indent=2, sort_keys=True))
        return target

    @classmethod
    def load_keyring(cls, path: str | Path) -> "AEADCryptoProvider":
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, dict):
            raise RuntimeError("AEAD keyring file must contain a JSON object")
        return cls.from_keyring_dict(payload)

    def rotate(self, new_master_key_hex: str | None = None) -> "AEADCryptoProvider":
        new_active = new_master_key_hex or self.generate_key_hex()
        legacy = [self.master_key_hex, *self._legacy_key_hexes]
        deduped_legacy: list[str] = []
        for key_hex in legacy:
            if key_hex != new_active and key_hex not in deduped_legacy:
                deduped_legacy.append(key_hex)
        return AEADCryptoProvider(new_active, legacy_key_hexes=deduped_legacy)

    @classmethod
    def is_encrypted_payload(cls, payload: bytes) -> bool:
        return len(payload) >= cls.MIN_ENVELOPE_LEN and payload[:1] == bytes([cls.ENVELOPE_VERSION])

    def encrypt(self, origin_id: str, plaintext: bytes) -> bytes:
        return bytes(self._native.encrypt(origin_id, plaintext))

    def decrypt(self, origin_id: str, ciphertext: bytes) -> bytes:
        # Backward compatibility for pre-AEAD blobs: if the payload does not
        # look like an AEAD envelope, leave it untouched.
        if not self.is_encrypted_payload(ciphertext):
            return ciphertext

        last_error: Exception | None = None
        for native in (self._native, *self._legacy_natives):
            try:
                return bytes(native.decrypt(origin_id, ciphertext))
            except Exception as exc:  # pragma: no cover - exercised via key rotation tests
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("AEAD decrypt failed without an active or legacy key")


def _supports_native_aead() -> bool:
    status = _rust_bridge.status()
    return bool(status.active and "aead_crypto" in status.accelerated_ops)


__all__ = [
    "CryptoProvider",
    "NullCrypto",
    "XorCrypto",
    "AEADCryptoProvider",
]
