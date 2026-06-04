# Security Policy

SODL is a storage, provenance, lineage, and artifact-management system created by SoftQraft Labs Ltd. Security reports are welcome and should be handled privately before public disclosure.

## Supported Versions

SODL is currently pre-1.0. Until the first stable release, security fixes will target the `main` branch and the latest published release artifacts.

When versioned releases begin, this file will list the supported release lines.

## Reporting A Vulnerability

Please do not open a public GitHub issue for suspected vulnerabilities.

Report privately through GitHub Security Advisories for this repository. If advisories are not available yet, contact the repository owner privately and include:

- affected component or crate,
- reproduction steps,
- expected impact,
- whether the issue allows unauthorized read, write, deletion, corruption, privilege escalation, denial of service, secret exposure, or integrity bypass,
- any proof-of-concept code or request payloads,
- suggested mitigation, if known.

We aim to acknowledge valid reports quickly, triage the affected surface, and publish a fix or mitigation note once the issue is understood.

## Security Boundary

SODL should be deployed as an internal service unless an application gateway adds authentication, authorization, request limits, audit logging, and upload policy enforcement.

Do not expose a development `sodl-server` directly to the public internet.

Recommended production shape:

```text
application/API gateway
  -> private SODL endpoint
  -> private runtime storage paths
```

## Production Requirements

For production deployments:

- Set `SODL_MASTER_KEY`.
- Store secrets in a managed secret store or equivalent host-level secret mechanism.
- Keep `SODL_BLOB_DIR` and `SODL_DB_PATH` outside the Git checkout.
- Back up both the blob directory and SQLite database together.
- Restrict file permissions on runtime storage.
- Bind SODL to localhost or a private network.
- Enforce upload size, type, and quota limits at the application or gateway layer.
- Monitor disk usage and configure garbage-collection operations before high-volume use.
- Use TLS for any network path that leaves localhost or a trusted private network.
- Log operational events without logging raw secrets or plaintext payloads.

## Known Pre-1.0 Limits

The following areas are still evolving and should be treated carefully in production:

- public authentication and authorization endpoints,
- semantic fingerprinting of transformed media,
- multi-node replication,
- operational garbage-collection commands,
- generated SDKs,
- public container release hardening,
- external app API ergonomics.

## Data Handling

SODL may process sensitive files, model artifacts, datasets, generated outputs, and lineage metadata. Application integrators remain responsible for:

- deciding which actors can upload and read files,
- validating user-provided content,
- enforcing tenant isolation,
- complying with retention and deletion requirements,
- protecting encryption keys,
- deciding whether SODL metadata is sensitive in their environment.

## Dependency And Supply Chain

Before production use:

- build from a tagged release or pinned commit,
- prefer pinned container image digests for deployments,
- run dependency audits appropriate for Rust and Python,
- review generated release artifacts before publishing,
- avoid running unreviewed build scripts in privileged environments.
