from __future__ import annotations

import subprocess
import sys


def test_top_level_sodl_import_surface() -> None:
    import sodl

    assert sodl.__version__ == "0.2.0"
    assert hasattr(sodl, "BlobStore")
    assert hasattr(sodl, "WeightStoreService")
    assert hasattr(sodl, "SODLClient")
    assert hasattr(sodl, "TokenHashIndex")
    assert hasattr(sodl, "rust_bridge_status")
    assert hasattr(sodl, "OptimizerStateStore")
    assert hasattr(sodl, "SODLAdamW")
    assert hasattr(sodl, "AEADCryptoProvider")
    assert hasattr(sodl, "LineageProof")
    assert hasattr(sodl, "Ed25519ProofSigner")
    assert hasattr(sodl, "SODLVectorIndex")
    assert hasattr(sodl, "DataQualityScorer")
    assert hasattr(sodl, "SemanticContextLogic")
    assert hasattr(sodl, "ClusteredAttentionLayer")
    assert hasattr(sodl, "SCLClusteredDecoder")
    assert hasattr(sodl, "SCLMemoryManifest")


def test_cli_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "sodl", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "0.2.0"


def test_cli_doctor_reports_bridge_status() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "sodl", "doctor"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "sodl 0.2.0" in result.stdout
    assert "native bridge" in result.stdout.lower()
