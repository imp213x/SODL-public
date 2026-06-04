"""HTTP client for the SODL REST API.

Provides a remote blob-store shaped interface over the Phase D endpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import uuid

from sodl_weights.store import compute_blob_id


class SodlClientError(RuntimeError):
    """Raised when the remote SODL service returns an unexpected response."""


@dataclass(slots=True)
class _HttpResponse:
    status: int
    body: bytes
    headers: dict[str, str]


class SODLClient:
    """Remote client with a BlobStore-like interface for core blob operations."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = float(timeout)
        self._default_headers = dict(default_headers or {})

    def _request(
        self,
        method: str,
        path: str,
        *,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
        expect_status: set[int] | None = None,
    ) -> _HttpResponse:
        request_headers = dict(self._default_headers)
        request_headers.update(headers or {})
        request = Request(
            f"{self._base_url}{path}",
            data=data,
            headers=request_headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                status = int(getattr(response, "status", 200))
                body = response.read()
                if expect_status and status not in expect_status:
                    raise SodlClientError(f"unexpected status {status} for {method} {path}")
                return _HttpResponse(
                    status=status,
                    body=body,
                    headers={str(key).lower(): str(value) for key, value in response.headers.items()},
                )
        except HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            if expect_status and exc.code in expect_status:
                return _HttpResponse(status=exc.code, body=payload.encode("utf-8"), headers={})
            raise SodlClientError(f"{method} {path} failed with {exc.code}: {payload}") from exc
        except URLError as exc:
            raise SodlClientError(f"{method} {path} failed: {exc}") from exc

    def _json_request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        expect_status: set[int] | None = None,
    ) -> dict[str, Any]:
        response = self._request(
            method,
            path,
            data=json.dumps(payload or {}).encode("utf-8") if payload is not None else None,
            headers={"Content-Type": "application/json"},
            expect_status=expect_status,
        )
        if not response.body:
            return {}
        return json.loads(response.body.decode("utf-8"))

    def _blob_path(self, blob_id: str) -> str:
        return f"/v1/blobs/{quote(blob_id, safe=':')}"

    def put(self, blob_id: str, data: bytes) -> None:
        self._request(
            "POST",
            "/v1/blobs",
            data=bytes(data),
            headers={
                "Content-Type": "application/octet-stream",
                "X-Blob-Id": blob_id,
            },
            expect_status={200, 201},
        )

    def create_blob(self, data: bytes) -> dict[str, Any]:
        response = self._request(
            "POST",
            "/v1/blobs",
            data=bytes(data),
            headers={"Content-Type": "application/octet-stream"},
            expect_status={200, 201},
        )
        return json.loads(response.body.decode("utf-8"))

    def upload(
        self,
        *,
        meta: dict[str, Any],
        file_bytes: bytes,
        filename: str = "upload.bin",
        content_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        boundary = f"sodl-boundary-{uuid.uuid4().hex}"
        meta_payload = json.dumps(meta, separators=(",", ":")).encode("utf-8")
        parts = [
            f"--{boundary}\r\n".encode("utf-8"),
            b'Content-Disposition: form-data; name="meta"\r\n',
            b"Content-Type: application/json\r\n\r\n",
            meta_payload,
            b"\r\n",
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            bytes(file_bytes),
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
        response = self._request(
            "POST",
            "/v1/upload",
            data=b"".join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            expect_status={201},
        )
        return json.loads(response.body.decode("utf-8"))

    def get(self, blob_id: str) -> bytes:
        return self._request("GET", self._blob_path(blob_id), expect_status={200}).body

    def delete(self, blob_id: str) -> None:
        self._request("DELETE", self._blob_path(blob_id), expect_status={204})

    def has(self, blob_id: str) -> bool:
        response = self._request("HEAD", self._blob_path(blob_id), expect_status={200, 404})
        return response.status == 200

    def replica_nodes(self, blob_id: str) -> list[str]:
        return [self._base_url] if self.has(blob_id) else []

    def blob_count(self) -> int:
        origins = self.list_origins()
        unique_blobs = {
            blob_id
            for origin in origins
            for representation in list(origin.get("representations", []) or [])
            for blob_id in list(representation.get("root_blobs", []) or [])
        }
        return len(unique_blobs)

    def health(self) -> dict[str, Any]:
        return self._json_request("GET", "/v1/health", expect_status={200})

    def list_origins(self) -> list[dict[str, Any]]:
        payload = self._json_request("GET", "/v1/origins", expect_status={200})
        return list(payload.get("origins", []) or [])

    def create_origin(
        self,
        *,
        media_kind: str = "binary",
        durability: str = "best_effort",
        owner: str | None = None,
        representations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "media_kind": media_kind,
            "durability": durability,
            "representations": list(representations or []),
        }
        if owner:
            payload["owner"] = owner
        return self._json_request("POST", "/v1/origins", payload=payload, expect_status={201})

    def get_origin(self, origin_id: str) -> dict[str, Any]:
        return self._json_request(
            "GET",
            f"/v1/origins/{quote(origin_id, safe='')}",
            expect_status={200},
        )

    def tombstone_origin(
        self,
        origin_id: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._json_request(
            "DELETE",
            f"/v1/origins/{quote(origin_id, safe='')}",
            payload=payload,
            expect_status={204},
        )

    def list_representations(self, origin_id: str) -> list[dict[str, Any]]:
        payload = self._json_request(
            "GET",
            f"/v1/origins/{quote(origin_id, safe='')}/representations",
            expect_status={200},
        )
        return list(payload.get("representations", []) or [])

    def get_payload(self, origin_id: str) -> bytes:
        return self._request(
            "GET",
            f"/v1/origins/{quote(origin_id, safe='')}/payload",
            expect_status={200},
        ).body

    def lineage_proof(self, origin_id: str) -> dict[str, Any]:
        return self._json_request(
            "GET",
            f"/v1/origins/{quote(origin_id, safe='')}/lineage-proof",
            expect_status={200},
        )

    def create_share(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._json_request("POST", "/v1/shares", payload=payload, expect_status={201})

    def get_share(self, share_id: str) -> dict[str, Any]:
        return self._json_request(
            "GET",
            f"/v1/shares/{quote(share_id, safe='')}",
            expect_status={200},
        )

    def release_share(self, share_id: str) -> None:
        self._request(
            "DELETE",
            f"/v1/shares/{quote(share_id, safe='')}",
            expect_status={204},
        )

    def verify_share(self, share_id: str) -> dict[str, Any]:
        return self._json_request(
            "POST",
            f"/v1/shares/{quote(share_id, safe='')}/verify",
            payload={},
            expect_status={200},
        )

    def create_derivation(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._json_request(
            "POST",
            "/v1/derivations",
            payload=payload,
            expect_status={201},
        )

    def create_pin(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._json_request(
            "POST",
            "/v1/pins",
            payload=payload,
            expect_status={201},
        )

    def release_pin(self, pin_id: str) -> None:
        self._request(
            "DELETE",
            f"/v1/pins/{quote(pin_id, safe='')}",
            expect_status={204},
        )

    def verify_blob_id(self, data: bytes) -> str:
        return compute_blob_id(bytes(data))


class RemoteBlobStore(SODLClient):
    """Alias that matches the blob-store role more explicitly."""


__all__ = [
    "RemoteBlobStore",
    "SODLClient",
    "SodlClientError",
]
