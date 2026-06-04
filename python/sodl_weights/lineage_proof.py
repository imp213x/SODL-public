"""SODL Lineage Proofs — Python SDK wrapper for sodl-proof.

Mirrors the Rust ``LineageProof`` and ``generate_proof_unsigned`` to provide
deterministic, verifiable provenance records for training checkpoints.

Each checkpoint gets a Blake3 digest computed over its canonical lineage:
dataset version, optimizer state, model config, and training step.

Usage::

    from sodl_weights.lineage_proof import LineageProver

    prover = LineageProver(origin_id="carlalarge-training-v1")
    proof = prover.generate_checkpoint_proof(
        step=1000,
        dataset_hash="blake3:abc...",
        model_config={"d_model": 1792, "n_layers": 24},
        checkpoint_path="models/native/carlalarge_base/ckpt_step1000.pt",
    )
    print(f"Lineage proof: {proof.digest}")
    prover.save_proof(proof, "models/native/carlalarge_base/proofs/")
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LineageEdge:
    """A single lineage edge (mirrors Rust ``LineageEdge``)."""
    edge_id: str
    origin_id: str
    blob_id: str | None
    kind: str
    created_at: str


@dataclass(slots=True)
class LineageProof:
    """Deterministic, unsigned lineage proof (mirrors Rust ``LineageProof``)."""
    origin_id: str
    digest: str
    created_at: str
    edges: list[LineageEdge]
    metadata: dict[str, Any]


def _blake3_hex(data: bytes) -> str:
    """Compute blake3 hex digest (falls back to SHA-256 if blake3 not available)."""
    try:
        import blake3
        return blake3.blake3(data).hexdigest()
    except ImportError:
        return hashlib.sha256(data).hexdigest()


class LineageProver:
    """Generate deterministic lineage proofs for training checkpoints.

    Matches the Rust ``generate_proof_unsigned`` canonicalization:
    - Sort edges by edge_id
    - Feed each edge's fields into a Blake3 hasher
    - Produce a hex digest

    Parameters
    ----------
    origin_id : str
        Unique identifier for this training origin (e.g., "carlalarge-v1").
    """

    def __init__(self, origin_id: str) -> None:
        self.origin_id = origin_id
        self._proofs: list[LineageProof] = []

    def _canonicalize(self, edges: list[LineageEdge]) -> str:
        """Compute deterministic Blake3 digest over sorted edges."""
        sorted_edges = sorted(edges, key=lambda e: e.edge_id)

        # Build canonical payload matching Rust's domain separation
        parts = [
            "SODL_LINEAGE_PROOF_V1\n",
            f"{self.origin_id}\n",
        ]
        for e in sorted_edges:
            parts.append(f"{e.edge_id}\n")
            parts.append(f"{e.origin_id}\n")
            parts.append(f"{e.blob_id or '-'}\n")
            parts.append(f"{e.kind}\n")

        canonical = "".join(parts).encode("utf-8")
        return _blake3_hex(canonical)

    def generate_checkpoint_proof(
        self,
        *,
        step: int,
        dataset_hash: str,
        model_config: dict[str, Any],
        checkpoint_path: str | None = None,
        extra_edges: list[LineageEdge] | None = None,
    ) -> LineageProof:
        """Generate a lineage proof for a training checkpoint.

        Automatically creates lineage edges for:
        1. Dataset origin (dataset_hash)
        2. Model configuration
        3. Training step
        4. Checkpoint file (if path provided)
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        edges = []

        # Edge 1: Dataset provenance
        edges.append(LineageEdge(
            edge_id=f"dataset:{step}",
            origin_id=self.origin_id,
            blob_id=dataset_hash,
            kind=f"origin_rep:dataset",
            created_at=now,
        ))

        # Edge 2: Model config
        config_json = json.dumps(model_config, sort_keys=True, separators=(",", ":"))
        config_hash = _blake3_hex(config_json.encode("utf-8"))[:16]
        edges.append(LineageEdge(
            edge_id=f"config:{config_hash}",
            origin_id=self.origin_id,
            blob_id=f"blake3:{config_hash}",
            kind="origin_rep:model_config",
            created_at=now,
        ))

        # Edge 3: Training step
        edges.append(LineageEdge(
            edge_id=f"step:{step}",
            origin_id=self.origin_id,
            blob_id=None,
            kind=f"derivation:train_step_{step}",
            created_at=now,
        ))

        # Edge 4: Checkpoint file
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt_data = Path(checkpoint_path).read_bytes()
            ckpt_hash = _blake3_hex(ckpt_data)[:32]
            edges.append(LineageEdge(
                edge_id=f"checkpoint:{step}",
                origin_id=self.origin_id,
                blob_id=f"blake3:{ckpt_hash}",
                kind="origin_rep:checkpoint",
                created_at=now,
            ))

        # Extra custom edges
        if extra_edges:
            edges.extend(extra_edges)

        digest = self._canonicalize(edges)

        proof = LineageProof(
            origin_id=self.origin_id,
            digest=digest,
            created_at=now,
            edges=edges,
            metadata={
                "step": step,
                "model_config_hash": config_hash,
                "dataset_hash": dataset_hash,
            },
        )
        self._proofs.append(proof)
        return proof

    def save_proof(self, proof: LineageProof, proof_dir: str | Path) -> Path:
        """Save a proof to disk as JSON."""
        proof_dir = Path(proof_dir)
        proof_dir.mkdir(parents=True, exist_ok=True)
        path = proof_dir / f"proof_step{proof.metadata.get('step', 0)}.json"

        data = {
            "schema": "sodl:lineage_proof:v1",
            "origin_id": proof.origin_id,
            "digest": proof.digest,
            "created_at": proof.created_at,
            "edges": [asdict(e) for e in proof.edges],
            "metadata": proof.metadata,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    def verify_proof(self, proof: LineageProof) -> bool:
        """Verify a proof's digest matches its edges (recompute and compare)."""
        recomputed = self._canonicalize(proof.edges)
        return recomputed == proof.digest

    def metrics(self) -> dict:
        """Return proof metrics for dashboard."""
        return {
            "lineage_enabled": True,
            "lineage_origin_id": self.origin_id,
            "lineage_proofs_generated": len(self._proofs),
            "lineage_latest_digest": self._proofs[-1].digest[:16] if self._proofs else None,
        }
