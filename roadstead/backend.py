"""Backend HTTP client for nexus/anvil LLM endpoints.

Maintains persistent httpx.AsyncClient pools per host.  Handles
streaming relay, X-Request-ID injection, and health probing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from .config import EndpointConfig

logger = logging.getLogger(__name__)


@dataclass
class BackendResponse:
    """Result of a backend call (non-streaming)."""
    status_code: int
    body: dict
    duration_s: float
    input_tokens: int
    output_tokens: int


@dataclass
class BackendStreamEvent:
    """One SSE event from a streaming backend call."""
    event_type: str   # "chunk" | "done" | "error"
    data: str         # raw SSE data line
    parsed: dict | None = None  # parsed JSON if applicable


class BackendError(Exception):
    """Backend returned a non-2xx response."""
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"backend error {status_code}: {detail}")


class BackendTimeout(BackendError):
    def __init__(self, detail: str = "timeout") -> None:
        super().__init__(504, detail)


class BackendUnavailable(BackendError):
    def __init__(self, detail: str = "unavailable") -> None:
        super().__init__(503, detail)


class BackendClientPool:
    """Manages httpx.AsyncClient instances for backend connections."""

    def __init__(self) -> None:
        self._clients: dict[str, httpx.AsyncClient] = {}

    def _client_for(self, host: str, port: int) -> httpx.AsyncClient:
        key = f"{host}:{port}"
        if key not in self._clients:
            self._clients[key] = httpx.AsyncClient(
                base_url=f"http://{host}:{port}",
                timeout=httpx.Timeout(connect=5.0, read=300.0, write=10.0, pool=5.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._clients[key]

    async def close(self) -> None:
        for client in self._clients.values():
            try:
                await client.aclose()
            except Exception:
                pass
        self._clients.clear()

    async def call(
        self,
        ep_cfg: EndpointConfig,
        payload: dict,
        payload_type: str,
        request_id: str,
        timeout_s: float = 180.0,
    ) -> BackendResponse:
        """Make a non-streaming backend call."""
        client = self._client_for(ep_cfg.host, ep_cfg.port)
        path = self._path_for(payload_type)
        headers = {"X-Request-ID": request_id}

        t0 = time.monotonic()
        try:
            resp = await asyncio.wait_for(
                client.post(path, json=payload, headers=headers),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            raise BackendTimeout(f"backend {ep_cfg.role} timeout after {timeout_s}s")
        except httpx.ConnectError as exc:
            raise BackendUnavailable(f"backend {ep_cfg.role} unreachable: {exc}")
        except httpx.HTTPError as exc:
            raise BackendError(502, f"backend {ep_cfg.role} http error: {exc}")

        duration = time.monotonic() - t0

        if resp.status_code >= 400:
            raise BackendError(resp.status_code, resp.text[:500])

        try:
            body = resp.json()
        except (json.JSONDecodeError, ValueError):
            body = {"raw": resp.text[:2000]}

        input_tokens = 0
        output_tokens = 0
        usage = body.get("usage") or {}
        if usage:
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

        return BackendResponse(
            status_code=resp.status_code,
            body=body,
            duration_s=duration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def stream(
        self,
        ep_cfg: EndpointConfig,
        payload: dict,
        payload_type: str,
        request_id: str,
        timeout_s: float = 180.0,
    ) -> AsyncIterator[BackendStreamEvent]:
        """Make a streaming backend call.  Yields SSE events."""
        client = self._client_for(ep_cfg.host, ep_cfg.port)
        path = self._path_for(payload_type)
        headers = {"X-Request-ID": request_id}

        try:
            async with client.stream(
                "POST", path, json=payload, headers=headers,
                timeout=httpx.Timeout(connect=5.0, read=timeout_s, write=10.0, pool=5.0),
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise BackendError(response.status_code, body.decode("utf-8", errors="replace")[:500])

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            yield BackendStreamEvent(event_type="done", data=data)
                            break
                        try:
                            parsed = json.loads(data)
                        except json.JSONDecodeError:
                            parsed = None
                        yield BackendStreamEvent(
                            event_type="chunk", data=data, parsed=parsed,
                        )
        except httpx.ConnectError as exc:
            raise BackendUnavailable(f"backend {ep_cfg.role} unreachable: {exc}")
        except asyncio.TimeoutError:
            raise BackendTimeout(f"backend {ep_cfg.role} stream timeout after {timeout_s}s")

    async def probe_props(self, ep_cfg: EndpointConfig) -> dict | None:
        """Probe backend /props for capacity discovery."""
        client = self._client_for(ep_cfg.host, ep_cfg.port)
        try:
            resp = await asyncio.wait_for(
                client.get("/props"),
                timeout=5.0,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    async def probe_health(self, ep_cfg: EndpointConfig) -> bool:
        """Simple health check."""
        client = self._client_for(ep_cfg.host, ep_cfg.port)
        try:
            resp = await asyncio.wait_for(
                client.get("/health"),
                timeout=3.0,
            )
            return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _path_for(payload_type: str) -> str:
        if payload_type == "embedding":
            return "/v1/embeddings"
        if payload_type == "rerank":
            return "/rerank"
        return "/v1/chat/completions"
