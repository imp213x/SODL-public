# syntax=docker/dockerfile:1

FROM rust:1-bookworm AS builder

WORKDIR /app

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates ./crates

RUN cargo build --locked --release -p sodl-api --bin sodl-server

FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system sodl \
    && useradd --system --gid sodl --home-dir /data --shell /usr/sbin/nologin sodl \
    && mkdir -p /data/blobs \
    && chown -R sodl:sodl /data

COPY --from=builder /app/target/release/sodl-server /usr/local/bin/sodl-server

ENV SODL_LISTEN=0.0.0.0:7700
ENV SODL_BLOB_DIR=/data/blobs
ENV SODL_DB_PATH=/data/sodl.db
ENV RUST_LOG=info

USER sodl
WORKDIR /data

EXPOSE 7700

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:7700/health >/dev/null || exit 1

ENTRYPOINT ["/usr/local/bin/sodl-server"]
