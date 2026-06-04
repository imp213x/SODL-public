# SODL Security Model

This document describes the intended security posture for SODL integrations. It is not a formal audit.

## Trust Boundaries

SODL has four main boundaries:

1. The calling application.
2. The SODL HTTP API or embedded service layer.
3. Runtime metadata and blob storage.
4. External clients, users, workers, or AI pipelines that provide files and artifacts.

The calling application is expected to authenticate users and decide whether an actor may upload, read, share, derive, or delete a file. SODL provides storage, provenance, lineage, and policy primitives; it does not replace product authorization by itself.

This boundary also applies to deduplication and reuse. SODL can identify that bytes, chunks, or derived artifacts match existing origins. The calling application decides whether that evidence is allowed to affect product storage, quota, billing, access, or UI state.

## Assets

Important assets include:

- uploaded file bytes,
- model artifacts and optimizer state,
- dataset fragments,
- generated AI outputs,
- origin records,
- lineage and derivation records,
- share records,
- policy and retention metadata,
- blob manifests,
- encryption keys,
- runtime SQLite database,
- runtime blob directory.

## Recommended Deployment Boundary

For a web application, the safest initial deployment is a private sidecar:

```text
user/browser
  -> application API
  -> SODL on localhost or private network
  -> runtime blob directory and metadata database
```

SODL should not be public until the deployment adds:

- authentication,
- authorization,
- request limits,
- upload policy enforcement,
- abuse controls,
- observability,
- secret management,
- backup and restore processes.

## Upload Security

SODL can identify and store bytes, but the application should still enforce:

- maximum file size,
- allowed MIME types and extensions,
- file signature validation,
- malware scanning where required,
- user quota,
- workspace or tenant quota,
- upload rate limiting,
- abuse logging.

SODL should receive files only after the application has decided the upload is allowed.

## Provenance Security

SODL provenance is intended to answer:

- is this payload exactly known,
- does this upload overlap with known chunks,
- which origin candidates should a caller inspect,
- what derivation or share records exist.

Provenance candidates are evidence, not authorization. An application must not grant read access to a candidate origin merely because bytes overlap. It must still check actor permissions.

Similarly, a provenance match is not an automatic instruction to reuse the application's own physical storage object. Reuse is safe only after the application checks its tenant, visibility, retention, scan, and business-policy rules.

Recommended integration posture:

- exact payload match: eligible for app-level physical-storage reuse if the app policy allows it,
- chunk overlap: evidence for lineage or review, not automatic reuse,
- derivation match: evidence for parent-child provenance, not automatic read access,
- cross-tenant match: never grant access without explicit app authorization.

## Encryption Boundary

For development, SODL may run with null crypto when `SODL_MASTER_KEY` is unset. Production should set `SODL_MASTER_KEY` and treat null crypto as development-only.

Operational guidance:

- use a strong random key,
- do not commit keys,
- rotate keys through an explicit migration process,
- keep backup and restore plans aligned with key management,
- understand that encrypted physical dedupe depends on encryption strategy and origin boundaries.

## Runtime Storage

Keep runtime paths outside the source checkout:

```text
SODL_BLOB_DIR=/var/lib/sodl/blobs
SODL_DB_PATH=/var/lib/sodl/sodl.sqlite
```

Back up the database and blob directory together. Losing either side can break payload recovery or provenance.

## AI-Specific Risks

AI workflows may add risks beyond ordinary file storage:

- generated artifacts may contain sensitive source data,
- model checkpoints may encode private data,
- dataset lineage may reveal business-sensitive inputs,
- reuse decisions can accidentally cross tenant boundaries,
- prompt or configuration metadata can be sensitive.

Applications should namespace owners and policies carefully. Do not treat artifact similarity as permission to disclose an artifact.

## Public API Guidance

When exposing SODL through a public product API:

- put an application gateway in front of SODL,
- use application-owned object IDs or scoped tokens,
- never expose raw filesystem paths,
- avoid returning internal policy metadata unless needed,
- use idempotency keys for upload completion,
- include audit logs for read and write actions,
- document lifecycle and deletion semantics.

## Security Checklist

- [ ] `SODL_MASTER_KEY` is set in production.
- [ ] SODL binds to localhost or private networking.
- [ ] Runtime storage is outside the Git checkout.
- [ ] Blob and database backups are tested together.
- [ ] Upload limits are enforced before SODL receives bytes.
- [ ] Application authorization wraps all SODL reads.
- [ ] Logs do not contain secrets or plaintext payloads.
- [ ] Public deployments use TLS.
- [ ] Container or binary releases are pinned by version or digest.
- [ ] Dependency audits are part of release workflow.

