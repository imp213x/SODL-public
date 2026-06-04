from __future__ import annotations

import hashlib
import json


def compute_pipeline_hash(origin_id: str, pipeline_kind: str, config: dict | str) -> str:
    payload = json.dumps(
        {
            "origin": origin_id,
            "pipeline": pipeline_kind,
            "config": config,
        },
        sort_keys=True,
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=32).hexdigest()

