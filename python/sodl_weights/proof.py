"""Lineage proof generation and signature helpers for the SODL Python SDK."""

from __future__ import annotations

import abc
import base64
from dataclasses import dataclass
from datetime import UTC, datetime
import os
from typing import Any, Mapping
import uuid

import blake3


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_origin_id(origin_id: str) -> str:
    candidate = str(origin_id).strip()
    if candidate.startswith("origin:"):
        candidate = candidate.split(":", 1)[1]
    return str(uuid.UUID(candidate))


def _edge_value(edge: Any, key: str, default: Any = None) -> Any:
    if isinstance(edge, Mapping):
        return edge.get(key, default)
    return getattr(edge, key, default)


def _kind_mapping_value(mapping: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key in mapping:
        return mapping[key]
    for candidate in ("payload", "value", "data"):
        nested = mapping.get(candidate)
        if isinstance(nested, Mapping) and key in nested:
            return nested[key]
    return default


def _canonical_kind(kind: Any) -> str:
    if isinstance(kind, str):
        return kind

    if isinstance(kind, Mapping):
        if "OriginRepresentation" in kind:
            payload = kind["OriginRepresentation"]
            return f"origin_rep:{_kind_mapping_value(payload, 'name', '')}"
        if "Share" in kind:
            payload = kind["Share"]
            return (
                "share:"
                f"{_kind_mapping_value(payload, 'share_id', '')}:"
                f"{_kind_mapping_value(payload, 'from', '')}:"
                f"{_kind_mapping_value(payload, 'to', '')}"
            )
        if "Derivation" in kind:
            payload = kind["Derivation"]
            return f"derivation:{_kind_mapping_value(payload, 'derivation_id', '')}"
        if "Pin" in kind:
            payload = kind["Pin"]
            return f"pin:{_kind_mapping_value(payload, 'pin_id', '')}"

        tag = (
            kind.get("type")
            or kind.get("kind")
            or kind.get("variant")
            or kind.get("name")
        )
        if isinstance(tag, str):
            normalized = tag.lower()
            if normalized in {"originrepresentation", "origin_rep", "origin-representation"}:
                return f"origin_rep:{_kind_mapping_value(kind, 'name', '')}"
            if normalized == "share":
                return (
                    "share:"
                    f"{_kind_mapping_value(kind, 'share_id', '')}:"
                    f"{_kind_mapping_value(kind, 'from', '')}:"
                    f"{_kind_mapping_value(kind, 'to', '')}"
                )
            if normalized == "derivation":
                return f"derivation:{_kind_mapping_value(kind, 'derivation_id', '')}"
            if normalized == "pin":
                return f"pin:{_kind_mapping_value(kind, 'pin_id', '')}"

    if hasattr(kind, "name") and not hasattr(kind, "share_id"):
        return f"origin_rep:{getattr(kind, 'name')}"
    if hasattr(kind, "share_id") and hasattr(kind, "from_") and hasattr(kind, "to"):
        return f"share:{kind.share_id}:{kind.from_}:{kind.to}"
    if hasattr(kind, "derivation_id"):
        return f"derivation:{kind.derivation_id}"
    if hasattr(kind, "pin_id"):
        return f"pin:{kind.pin_id}"

    raise ValueError(f"Unsupported lineage edge kind: {kind!r}")


def _canonical_edge_records(origin_id: str, edges: list[Any]) -> list[tuple[str, str, str, str]]:
    normalized_origin = _normalize_origin_id(origin_id)
    records: list[tuple[str, str, str, str]] = []
    for edge in edges:
        edge_id = str(_edge_value(edge, "edge_id"))
        edge_origin = _normalize_origin_id(str(_edge_value(edge, "origin_id", normalized_origin)))
        blob_id = _edge_value(edge, "blob_id")
        blob_id_text = "-" if blob_id in (None, "", False) else str(blob_id)
        kind = _canonical_kind(_edge_value(edge, "kind"))
        records.append((edge_id, edge_origin, blob_id_text, kind))
    return sorted(records, key=lambda item: item[0])


@dataclass(frozen=True, slots=True)
class LineageProof:
    origin_id: str
    digest: str
    created_at: str
    key_id: str | None = None
    signature_b64: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_id": self.origin_id,
            "digest": self.digest,
            "created_at": self.created_at,
            "key_id": self.key_id,
            "signature_b64": self.signature_b64,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LineageProof":
        return cls(
            origin_id=str(payload["origin_id"]),
            digest=str(payload["digest"]),
            created_at=str(payload["created_at"]),
            key_id=str(payload["key_id"]) if payload.get("key_id") else None,
            signature_b64=str(payload["signature_b64"])
            if payload.get("signature_b64")
            else None,
        )

    @property
    def is_signed(self) -> bool:
        return bool(self.key_id and self.signature_b64)


class ProofSigner(abc.ABC):
    @property
    @abc.abstractmethod
    def key_id(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def sign_digest_b64(self, digest_hex: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def verify_digest_b64(self, digest_hex: str, signature_b64: str) -> bool:
        raise NotImplementedError


def _load_ed25519():
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "Ed25519 proof signing requires `cryptography`. "
            "Install `sodl[security]` or add `cryptography` to the environment."
        ) from exc
    return Ed25519PrivateKey, Ed25519PublicKey, serialization


class Ed25519ProofSigner(ProofSigner):
    def __init__(self, key_id: str, private_key: Any) -> None:
        self._key_id = key_id
        self._private_key = private_key
        self._public_key = private_key.public_key()

    @property
    def key_id(self) -> str:
        return self._key_id

    @classmethod
    def generate(cls, key_id: str = "ed25519:auto") -> "Ed25519ProofSigner":
        Ed25519PrivateKey, _, _ = _load_ed25519()
        return cls(key_id, Ed25519PrivateKey.generate())

    @classmethod
    def from_private_key_bytes(
        cls,
        key_id: str,
        private_key_bytes: bytes,
    ) -> "Ed25519ProofSigner":
        Ed25519PrivateKey, _, _ = _load_ed25519()
        if len(private_key_bytes) != 32:
            raise ValueError("Ed25519 private key seed must be 32 bytes")
        return cls(key_id, Ed25519PrivateKey.from_private_bytes(private_key_bytes))

    @classmethod
    def from_private_key_hex(
        cls,
        key_id: str,
        private_key_hex: str,
    ) -> "Ed25519ProofSigner":
        return cls.from_private_key_bytes(key_id, bytes.fromhex(private_key_hex))

    @classmethod
    def from_env(
        cls,
        env_var: str = "SODL_PROOF_PRIVATE_KEY_HEX",
        key_id: str = "ed25519:env",
    ) -> "Ed25519ProofSigner":
        raw = os.environ.get(env_var)
        if not raw:
            raise RuntimeError(f"Environment variable {env_var} is not set")
        return cls.from_private_key_hex(key_id, raw)

    @property
    def public_key_hex(self) -> str:
        _, _, serialization = _load_ed25519()
        raw = self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return raw.hex()

    def sign_digest_b64(self, digest_hex: str) -> str:
        signature = self._private_key.sign(digest_hex.encode("utf-8"))
        return base64.b64encode(signature).decode("ascii")

    def verify_digest_b64(self, digest_hex: str, signature_b64: str) -> bool:
        try:
            signature = base64.b64decode(signature_b64.encode("ascii"))
            self._public_key.verify(signature, digest_hex.encode("utf-8"))
            return True
        except Exception:
            return False

    @staticmethod
    def verify_with_public_key_hex(
        digest_hex: str,
        signature_b64: str,
        public_key_hex: str,
    ) -> bool:
        _, Ed25519PublicKey, _ = _load_ed25519()
        try:
            public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
            signature = base64.b64decode(signature_b64.encode("ascii"))
            public_key.verify(signature, digest_hex.encode("utf-8"))
            return True
        except Exception:
            return False


def generate_lineage_proof(
    origin_id: str,
    edges: list[Any],
    created_at: str | None = None,
) -> LineageProof:
    normalized_origin = _normalize_origin_id(origin_id)
    hasher = blake3.blake3()
    hasher.update(b"SODL_LINEAGE_PROOF_V1\n")
    hasher.update(normalized_origin.encode("utf-8"))
    hasher.update(b"\n")
    for edge_id, edge_origin, blob_id, kind in _canonical_edge_records(normalized_origin, edges):
        hasher.update(edge_id.encode("utf-8"))
        hasher.update(b"\n")
        hasher.update(edge_origin.encode("utf-8"))
        hasher.update(b"\n")
        hasher.update(blob_id.encode("utf-8"))
        hasher.update(b"\n")
        hasher.update(kind.encode("utf-8"))
        hasher.update(b"\n")
    return LineageProof(
        origin_id=normalized_origin,
        digest=hasher.hexdigest(),
        created_at=created_at or _utc_now_iso(),
    )


def sign_lineage_proof(proof: LineageProof, signer: ProofSigner) -> LineageProof:
    return LineageProof(
        origin_id=proof.origin_id,
        digest=proof.digest,
        created_at=proof.created_at,
        key_id=signer.key_id,
        signature_b64=signer.sign_digest_b64(proof.digest),
    )


def verify_lineage_digest(
    proof: LineageProof,
    edges: list[Any],
) -> bool:
    regenerated = generate_lineage_proof(proof.origin_id, edges, created_at=proof.created_at)
    return regenerated.digest == proof.digest


def verify_lineage_signature(
    proof: LineageProof,
    *,
    signer: ProofSigner | None = None,
    public_key_hex: str | None = None,
) -> bool:
    if not proof.signature_b64:
        return False
    if signer is not None:
        return signer.verify_digest_b64(proof.digest, proof.signature_b64)
    if public_key_hex is not None:
        return Ed25519ProofSigner.verify_with_public_key_hex(
            proof.digest,
            proof.signature_b64,
            public_key_hex,
        )
    raise ValueError("Provide either signer or public_key_hex for signature verification")


__all__ = [
    "LineageProof",
    "ProofSigner",
    "Ed25519ProofSigner",
    "generate_lineage_proof",
    "sign_lineage_proof",
    "verify_lineage_digest",
    "verify_lineage_signature",
]
