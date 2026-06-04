# Contributing To SODL

Thank you for helping improve SODL.

SODL was created by SoftQraft Labs Ltd and is early-stage infrastructure. Contributions are welcome, but changes should preserve the core design goal: one origin-aware layer for storage, provenance, lineage, and artifact reuse.

## Development Setup

Rust:

```bash
cargo build
cargo test
```

Python:

```bash
cd python
python -m pip install -e ".[test]"
pytest
```

Optional native Python bridge:

```bash
python -m pip install maturin
cd crates/sodl-python-ffi
python -m maturin develop --release
```

## Pull Request Expectations

Before opening a pull request:

- run relevant Rust tests,
- run relevant Python tests when touching `python/`,
- update docs when changing API behavior,
- keep generated caches and local runtime data out of commits,
- do not commit secrets, credentials, private keys, or local database files.

## Design Principles

- Origin identity should outlive storage location.
- Provenance evidence is not authorization.
- Apps should be able to integrate over HTTP without embedding Rust.
- Rust crates should remain usable by native consumers.
- AI artifact reuse must remain tenant-aware and policy-aware.
- Deletion and GC must respect references and retention policies.

## Security Reports

Please see `SECURITY.md`. Do not disclose suspected vulnerabilities in public issues.
