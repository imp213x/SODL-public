from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from sodl_weights.client import SODLClient


def test_sodl_client_blob_and_origin_flow() -> None:
    blobs: dict[str, bytes] = {}
    origins: list[dict[str, object]] = []
    payloads: dict[str, bytes] = {}
    shares: dict[str, dict[str, object]] = {}
    pins: dict[str, dict[str, object]] = {}
    derivations: list[dict[str, object]] = []

    class _Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):  # noqa: N802
            if self.path.startswith("/v1/blobs/"):
                blob_id = self.path.split("/v1/blobs/", 1)[1]
                self.send_response(200 if blob_id in blobs else 404)
                self.end_headers()
                return
            self.send_response(404)
            self.end_headers()

        def do_GET(self):  # noqa: N802
            if self.path == "/v1/health":
                payload = {"status": "ok", "version": "sodl-v1"}
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/v1/origins":
                body = json.dumps({"origins": origins}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/v1/origins/") and self.path.endswith("/lineage-proof"):
                origin_id = self.path.split("/v1/origins/", 1)[1].split("/lineage-proof", 1)[0]
                body = json.dumps(
                    {
                        "origin_id": origin_id,
                        "digest": "proof-digest",
                        "created_at": "2026-03-23T00:00:00Z",
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/v1/origins/") and self.path.endswith("/payload"):
                origin_id = self.path.split("/v1/origins/", 1)[1].split("/payload", 1)[0]
                payload = payloads.get(origin_id, b"")
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if self.path.endswith("/representations"):
                origin_id = self.path.split("/v1/origins/", 1)[1].split("/representations", 1)[0]
                record = next(item for item in origins if item["origin_id"] == origin_id)
                body = json.dumps(
                    {
                        "origin_id": origin_id,
                        "representations": record.get("representations", []),
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/v1/origins/"):
                origin_id = self.path.split("/v1/origins/", 1)[1]
                record = next((item for item in origins if item["origin_id"] == origin_id), None)
                if record is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps(record).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/v1/blobs/"):
                blob_id = self.path.split("/v1/blobs/", 1)[1]
                payload = blobs.get(blob_id)
                if payload is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if self.path.startswith("/v1/shares/"):
                share_id = self.path.split("/v1/shares/", 1)[1]
                share = shares.get(share_id)
                if share is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps(share).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):  # noqa: N802
            if self.path == "/v1/blobs":
                size = int(self.headers.get("Content-Length", "0"))
                payload = self.rfile.read(size)
                blob_id = self.headers.get("X-Blob-Id") or "blake3:auto"
                existed = blob_id in blobs
                blobs[blob_id] = payload
                body = json.dumps(
                    {
                        "blob_id": blob_id,
                        "existed": existed,
                        "size_bytes": len(payload),
                    }
                ).encode("utf-8")
                self.send_response(200 if existed else 201)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/v1/origins":
                size = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(size).decode("utf-8"))
                origin_id = f"origin-{len(origins) + 1}"
                record = {
                    "origin_id": origin_id,
                    "media_kind": payload.get("media_kind", "binary"),
                    "durability": payload.get("durability", "best_effort"),
                    "created_at": "2026-03-20T00:00:00Z",
                    "tombstoned_at": None,
                    "representations": payload.get("representations", []),
                    "owner": payload.get("owner"),
                }
                origins.append(record)
                body = json.dumps(record).encode("utf-8")
                self.send_response(201)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/v1/upload":
                size = int(self.headers.get("Content-Length", "0"))
                payload = self.rfile.read(size)
                assert b'name="meta"' in payload
                assert b'name="file"' in payload
                origin_id = f"origin-{len(origins) + 1}"
                record = {
                    "origin_id": origin_id,
                    "media_kind": "binary",
                    "durability": "best_effort",
                    "created_at": "2026-03-20T00:00:00Z",
                    "tombstoned_at": None,
                    "representations": [{"name": "source", "root_blobs": ["blake3:upload"], "media_kind": "binary"}],
                    "owner": None,
                }
                origins.append(record)
                payloads[origin_id] = b"uploaded-bytes"
                body = json.dumps(record).encode("utf-8")
                self.send_response(201)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/v1/shares":
                size = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(size).decode("utf-8"))
                share_id = f"share-{len(shares) + 1}"
                record = {"share_id": share_id, **payload}
                shares[share_id] = record
                body = json.dumps(record).encode("utf-8")
                self.send_response(201)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.endswith("/verify") and self.path.startswith("/v1/shares/"):
                body = json.dumps({"valid": True}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/v1/derivations":
                size = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(size).decode("utf-8"))
                record = {"derivation_id": f"drv-{len(derivations) + 1}", **payload}
                derivations.append(record)
                body = json.dumps(record).encode("utf-8")
                self.send_response(201)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/v1/pins":
                size = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(size).decode("utf-8"))
                pin_id = f"pin-{len(pins) + 1}"
                record = {"pin_id": pin_id, **payload}
                pins[pin_id] = record
                body = json.dumps(record).encode("utf-8")
                self.send_response(201)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

        def do_DELETE(self):  # noqa: N802
            if self.path.startswith("/v1/blobs/"):
                blob_id = self.path.split("/v1/blobs/", 1)[1]
                blobs.pop(blob_id, None)
                self.send_response(204)
                self.end_headers()
                return
            if self.path.startswith("/v1/origins/"):
                origin_id = self.path.split("/v1/origins/", 1)[1]
                origins[:] = [item for item in origins if item["origin_id"] != origin_id]
                payloads.pop(origin_id, None)
                self.send_response(204)
                self.end_headers()
                return
            if self.path.startswith("/v1/shares/"):
                share_id = self.path.split("/v1/shares/", 1)[1]
                shares.pop(share_id, None)
                self.send_response(204)
                self.end_headers()
                return
            if self.path.startswith("/v1/pins/"):
                pin_id = self.path.split("/v1/pins/", 1)[1]
                pins.pop(pin_id, None)
                self.send_response(204)
                self.end_headers()
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):  # noqa: A003
            return

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = SODLClient(f"http://127.0.0.1:{server.server_port}")
        blob_id = "blake3:test"
        payload = b"hello-sodl"

        uploaded = client.upload(meta={"media_kind": "binary"}, file_bytes=b"uploaded-bytes")
        assert uploaded["origin_id"].startswith("origin-")

        client.put(blob_id, payload)
        assert client.has(blob_id)
        assert client.get(blob_id) == payload
        assert client.health()["status"] == "ok"

        origin = client.create_origin(
            owner="user:test",
            representations=[{"name": "source", "root_blobs": [blob_id], "media_kind": "binary"}],
        )
        payloads[origin["origin_id"]] = payload
        assert origin["origin_id"].startswith("origin-")
        assert client.list_origins()
        assert client.get_origin(origin["origin_id"])["owner"] == "user:test"
        assert client.list_representations(origin["origin_id"])[0]["name"] == "source"
        assert client.get_payload(origin["origin_id"]) == payload
        assert client.lineage_proof(origin["origin_id"])["digest"] == "proof-digest"

        share = client.create_share({"origin_id": origin["origin_id"], "to": "user:peer"})
        assert client.get_share(share["share_id"])["to"] == "user:peer"
        assert client.verify_share(share["share_id"])["valid"] is True

        derivation = client.create_derivation({"origin_id": origin["origin_id"], "kind": "token_hash"})
        assert derivation["derivation_id"].startswith("drv-")

        pin = client.create_pin({"origin_id": origin["origin_id"], "blob_id": blob_id})
        assert pin["pin_id"].startswith("pin-")
        client.release_pin(pin["pin_id"])

        assert client.blob_count() == 2

        client.release_share(share["share_id"])
        client.tombstone_origin(origin["origin_id"])
        assert len(client.list_origins()) == 1

        client.delete(blob_id)
        assert not client.has(blob_id)
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()
