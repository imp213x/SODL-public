Step 23 adds:
- `sodl-ai-artifacts` for pipeline hashing and dataset chunk manifests
- `sodl-semantic-router` for semantic route selection
- Python binder modules for pipeline hashing and semantic routing

Intent:
- Reuse semantic artifacts instead of recomputing them blindly
- Give Carla and training jobs a deterministic way to identify the exact preprocessing pipeline
- Prepare dataset chunking before the later semantic color / cube compression step

Current boundary:
- Build chunk manifests now
- Defer SODL semantic color / cube compression payload generation until the upgraded SODL path is ready
