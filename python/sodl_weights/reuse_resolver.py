"""SODL Reuse-First Training Artifact Resolver.

Implements the ``ensure_*`` pattern from SODL Step 21
(AI Integration — Reuse-First Resolver) for training artifacts.

Each training preprocessing step (tokenization, sharding, color tagging)
produces artifacts with a deterministic pipeline hash. Before re-running
any step, the resolver checks:

1. Does the output artifact already exist?
2. Does its pipeline hash match the current configuration?
3. If yes → skip (instant restart). If no → rerun.

This gives us:
- **Instant training restarts** after crashes or interruptions
- **No redundant preprocessing** (saves 30-60 minutes per restart)
- **Deterministic reproducibility** via pipeline hash

Usage::

    from sodl_weights.reuse_resolver import ReusResolver

    resolver = ReusResolver(artifact_dir="data/datasets/carlalarge")

    # Returns True if tokenization can be skipped
    if resolver.ensure("tokenize", config={"vocab_size": 20843, "sources": sources}):
        print("Tokenization already done — skipping")
    else:
        run_tokenization()
        resolver.mark_done("tokenize", config={"vocab_size": 20843, "sources": sources})
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ArtifactRecord:
    """Metadata for a single resolved artifact."""
    stage: str
    pipeline_hash: str
    created_at: str
    duration_sec: float
    config: dict[str, Any]


def _pipeline_hash(stage: str, config: dict[str, Any]) -> str:
    """Compute a deterministic hash for a pipeline stage + configuration."""
    payload = json.dumps({"stage": stage, **config}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ReuseResolver:
    """SODL Reuse-First resolver for training artifacts.

    Manages a ``.sodl_artifacts.json`` manifest in the artifact directory
    that records which pipeline stages have completed and their config hashes.

    Parameters
    ----------
    artifact_dir : str | Path
        Directory where artifacts and the manifest live.
    """

    MANIFEST_NAME = ".sodl_artifacts.json"

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.artifact_dir / self.MANIFEST_NAME
        self._records: dict[str, ArtifactRecord] = {}
        self._load()

    def _load(self) -> None:
        """Load existing manifest from disk."""
        if self._manifest_path.exists():
            try:
                data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
                for key, rec in data.get("artifacts", {}).items():
                    self._records[key] = ArtifactRecord(**rec)
            except (json.JSONDecodeError, TypeError, KeyError):
                self._records = {}

    def _save(self) -> None:
        """Persist manifest to disk."""
        data = {
            "schema": "sodl-reuse-v1",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "artifacts": {k: asdict(v) for k, v in self._records.items()},
        }
        self._manifest_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    def ensure(self, stage: str, *, config: dict[str, Any]) -> bool:
        """Check if a stage can be skipped (artifact exists and config matches).

        Returns
        -------
        bool
            ``True`` if the stage has already been completed with the same
            config hash — safe to skip. ``False`` if it needs to be rerun.
        """
        expected_hash = _pipeline_hash(stage, config)
        record = self._records.get(stage)
        if record is None:
            return False
        return record.pipeline_hash == expected_hash

    def mark_done(
        self,
        stage: str,
        *,
        config: dict[str, Any],
        duration_sec: float = 0.0,
    ) -> ArtifactRecord:
        """Record a completed pipeline stage.

        Parameters
        ----------
        stage : str
            Stage name (e.g., ``"tokenize"``, ``"shard"``, ``"color_tag"``).
        config : dict
            Configuration used for this stage.
        duration_sec : float
            How long the stage took.
        """
        record = ArtifactRecord(
            stage=stage,
            pipeline_hash=_pipeline_hash(stage, config),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            duration_sec=round(duration_sec, 2),
            config=config,
        )
        self._records[stage] = record
        self._save()
        return record

    def invalidate(self, stage: str) -> None:
        """Invalidate a stage, forcing it to be rerun next time."""
        self._records.pop(stage, None)
        self._save()

    def invalidate_all(self) -> None:
        """Clear all artifact records (force full reprocessing)."""
        self._records.clear()
        self._save()

    def summary(self) -> dict[str, Any]:
        """Return a summary of all resolved artifacts."""
        return {
            "artifact_dir": str(self.artifact_dir),
            "stages_completed": len(self._records),
            "stages": {
                k: {
                    "hash": v.pipeline_hash,
                    "created_at": v.created_at,
                    "duration_sec": v.duration_sec,
                }
                for k, v in self._records.items()
            },
        }
