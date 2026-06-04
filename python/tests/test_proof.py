from __future__ import annotations

from sodl_weights.proof import (
    Ed25519ProofSigner,
    LineageProof,
    generate_lineage_proof,
    sign_lineage_proof,
    verify_lineage_digest,
    verify_lineage_signature,
)


def _sample_edges() -> list[dict]:
    origin_id = "origin:550e8400-e29b-41d4-a716-446655440000"
    return [
        {
            "edge_id": "e2",
            "origin_id": origin_id,
            "blob_id": None,
            "kind": {
                "type": "Derivation",
                "derivation_id": "der:1",
            },
        },
        {
            "edge_id": "e1",
            "origin_id": "550e8400-e29b-41d4-a716-446655440000",
            "blob_id": "blake3:abc",
            "kind": {
                "Share": {
                    "share_id": "share:1",
                    "from": "user:a",
                    "to": "user:b",
                }
            },
        },
    ]


def test_generate_lineage_proof_is_deterministic() -> None:
    edges = _sample_edges()
    proof_a = generate_lineage_proof(
        "origin:550e8400-e29b-41d4-a716-446655440000",
        edges,
        created_at="2026-03-23T12:00:00Z",
    )
    proof_b = generate_lineage_proof(
        "550e8400-e29b-41d4-a716-446655440000",
        list(reversed(edges)),
        created_at="2026-03-23T12:00:00Z",
    )

    assert proof_a.digest == proof_b.digest
    assert proof_a.origin_id == "550e8400-e29b-41d4-a716-446655440000"


def test_verify_lineage_digest_rejects_tampered_edges() -> None:
    edges = _sample_edges()
    proof = generate_lineage_proof(
        "origin:550e8400-e29b-41d4-a716-446655440000",
        edges,
        created_at="2026-03-23T12:00:00Z",
    )
    tampered = [dict(edge) for edge in edges]
    tampered[0] = {
        **tampered[0],
        "kind": {
            "type": "Derivation",
            "derivation_id": "der:2",
        },
    }

    assert verify_lineage_digest(proof, edges) is True
    assert verify_lineage_digest(proof, tampered) is False


def test_ed25519_signature_roundtrip() -> None:
    proof = generate_lineage_proof(
        "origin:550e8400-e29b-41d4-a716-446655440000",
        _sample_edges(),
        created_at="2026-03-23T12:00:00Z",
    )
    signer = Ed25519ProofSigner.generate("ed25519:test")
    signed = sign_lineage_proof(proof, signer)

    assert isinstance(signed, LineageProof)
    assert signed.is_signed
    assert verify_lineage_signature(signed, signer=signer) is True
    assert verify_lineage_signature(
        signed,
        public_key_hex=signer.public_key_hex,
    ) is True


def test_ed25519_signature_rejects_tampered_digest() -> None:
    proof = generate_lineage_proof(
        "origin:550e8400-e29b-41d4-a716-446655440000",
        _sample_edges(),
        created_at="2026-03-23T12:00:00Z",
    )
    signer = Ed25519ProofSigner.generate("ed25519:test")
    signed = sign_lineage_proof(proof, signer)
    tampered = LineageProof(
        origin_id=signed.origin_id,
        digest="0" * len(signed.digest),
        created_at=signed.created_at,
        key_id=signed.key_id,
        signature_b64=signed.signature_b64,
    )

    assert verify_lineage_signature(tampered, signer=signer) is False
