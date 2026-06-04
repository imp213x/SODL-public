# Container Deployment

SODL ships a standalone HTTP engine named `sodl-server`. The container image runs that engine as a non-root user and stores durable state under `/data`.

## Build Locally

```bash
docker build -t sodl-server:local .
```

## Run Locally

```bash
docker run --rm \
  -p 7700:7700 \
  -v sodl-data:/data \
  -e SODL_MASTER_KEY=64_hex_chars_from_your_secret_manager \
  sodl-server:local
```

Health check:

```bash
curl http://127.0.0.1:7700/health
```

## Runtime Configuration

| Variable | Default in image | Purpose |
| --- | --- | --- |
| `SODL_LISTEN` | `0.0.0.0:7700` | Bind address inside the container. |
| `SODL_BLOB_DIR` | `/data/blobs` | Content-addressed blob directory. |
| `SODL_DB_PATH` | `/data/sodl.db` | SQLite metadata database. |
| `SODL_MASTER_KEY` | unset | 64-character hex key for production encryption. |
| `RUST_LOG` | `info` | Tracing verbosity. |

Production deployments should mount `/data` on durable storage and set `SODL_MASTER_KEY` from a secret manager. Do not bake secrets into the image.

## GHCR Publishing

The intended public image name is:

```text
ghcr.io/imp213x/sodl-server
```

Publish from a clean release checkout:

```bash
docker build -t ghcr.io/imp213x/sodl-server:0.1.0 .
docker tag ghcr.io/imp213x/sodl-server:0.1.0 ghcr.io/imp213x/sodl-server:latest
docker push ghcr.io/imp213x/sodl-server:0.1.0
docker push ghcr.io/imp213x/sodl-server:latest
```

For CI, use GitHub Actions with `GITHUB_TOKEN` package write permission or a dedicated token with `write:packages`. Keep the image public if downstream apps should pull it without authentication.
