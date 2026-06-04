from __future__ import annotations

import json

import pytest

from sodl_weights import crypto as crypto_module


class _FakeNativeAead:
    def __init__(self, master_key_hex: str | None = None) -> None:
        self._master_key_hex = master_key_hex or ("11" * 32)

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


def test_aead_crypto_provider_uses_native_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        crypto_module._rust_bridge,
        "create_aead_crypto",
        lambda master_key_hex=None: _FakeNativeAead(master_key_hex),
    )
    provider = crypto_module.AEADCryptoProvider("22" * 32)
    plaintext = b"hello aead"
    ciphertext = provider.encrypt("origin:test", plaintext)
    assert ciphertext != plaintext
    assert provider.decrypt("origin:test", ciphertext) == plaintext
    assert provider.master_key_hex == "22" * 32


def test_aead_crypto_provider_leaves_legacy_payloads_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        crypto_module._rust_bridge,
        "create_aead_crypto",
        lambda master_key_hex=None: _FakeNativeAead(master_key_hex),
    )
    provider = crypto_module.AEADCryptoProvider("33" * 32)
    legacy = b"plain-zstd-payload"
    assert provider.decrypt("origin:test", legacy) == legacy


def test_aead_crypto_provider_requires_native_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(crypto_module._rust_bridge, "create_aead_crypto", lambda master_key_hex=None: None)
    with pytest.raises(RuntimeError):
        crypto_module.AEADCryptoProvider("44" * 32)


def test_aead_crypto_provider_keyring_roundtrip(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        crypto_module._rust_bridge,
        "create_aead_crypto",
        lambda master_key_hex=None: _FakeNativeAead(master_key_hex),
    )
    provider = crypto_module.AEADCryptoProvider(
        "55" * 32,
        legacy_key_hexes=["44" * 32],
    )
    path = provider.save_keyring(tmp_path / "aead-keyring.json")

    payload = json.loads(path.read_text())
    assert payload["active_master_key_hex"] == "55" * 32
    assert payload["legacy_master_key_hexes"] == ["44" * 32]

    loaded = crypto_module.AEADCryptoProvider.load_keyring(path)
    assert loaded.master_key_hex == "55" * 32
    assert loaded.legacy_key_hexes == ("44" * 32,)


def test_aead_crypto_provider_rotation_keeps_legacy_decrypt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        crypto_module._rust_bridge,
        "create_aead_crypto",
        lambda master_key_hex=None: _FakeNativeAead(master_key_hex),
    )
    provider = crypto_module.AEADCryptoProvider("66" * 32)
    plaintext = b"rotating payload"
    ciphertext = provider.encrypt("origin:test", plaintext)

    rotated = provider.rotate("77" * 32)
    assert rotated.master_key_hex == "77" * 32
    assert rotated.legacy_key_hexes == ("66" * 32,)
    assert rotated.decrypt("origin:test", ciphertext) == plaintext
