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


def _normalize_chat_payload(payload: dict, vllm: bool = False) -> dict:
    """Make an Anthropic/extra_body-shaped chat payload wire-correct for the
    backend.

    The pre-proxy path built requests with ``call_nexus`` +
    ``openai.OpenAI``: the former inlined a top-level ``system`` field as
    the first ``messages`` entry, the latter merged ``extra_body`` keys
    (e.g. GBNF ``grammar``) into the top-level request body. ``ProxyLLMClient``
    does neither, so without this normalization llama-server silently ignores
    both — structured-output call sites (knowledge extract/dedup/relationships,
    temporal, health-verifier) lose their system prompt AND grammar and fall back to
    free-form output that fails JSON parsing.

    For a vLLM backend (``vllm=True``), a top-level ``grammar`` (llama.cpp's
    field) is silently ignored — vLLM enforces GBNF only via
    ``structured_outputs.grammar``. We move it there so the thinker (vLLM-NVFP4)
    actually enforces grammar instead of emitting free-form output.
    """
    if not isinstance(payload, dict):
        return payload
    if "system" not in payload and "extra_body" not in payload and not (
        vllm and "grammar" in payload
    ):
        return payload
    p = dict(payload)
    system = p.pop("system", None)
    if system:
        content = system if isinstance(system, str) else str(system)
        p["messages"] = [{"role": "system", "content": content}, *(p.get("messages") or [])]
    extra_body = p.pop("extra_body", None)
    if isinstance(extra_body, dict):
        p.update(extra_body)
    if vllm and isinstance(p.get("grammar"), str) and p["grammar"].strip():
        so = p.get("structured_outputs")
        so = dict(so) if isinstance(so, dict) else {}
        so.setdefault("grammar", p.pop("grammar"))
        p["structured_outputs"] = so
    return p


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
                timeout=httpx.Timeout(connect=5.0, read=600.0, write=10.0, pool=5.0),
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
        if payload_type == "chat_completion":
            payload = _normalize_chat_payload(
                payload, vllm=(ep_cfg.backend_engine == "vllm"))

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

        # Empty-completion gate (fail-loud). A 2xx with no generated content is a
        # silent failure — backend hiccup, grammar over-constraint that masks all
        # tokens, or an immediate EOS. It must NEVER pass as a valid (empty)
        # result: callers (e.g. knowledge.extract_entities) would record "no
        # entities" and silently drop data. Surface it as an error so the caller
        # retries/handles and WS2 monitoring sees it. Content-based (not token-
        # count) so it holds even when a backend omits usage; tool-call responses
        # legitimately have empty content, so they're exempt.
        if payload_type == "chat_completion":
            choice0 = (body.get("choices") or [{}])[0] or {}
            msg = choice0.get("message") or {}
            if not (msg.get("content") or "").strip() and not msg.get("tool_calls"):
                raise BackendError(
                    502,
                    f"backend {ep_cfg.role} returned empty completion "
                    f"(no content, output_tokens={output_tokens})",
                )

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
        if payload_type == "chat_completion":
            payload = _normalize_chat_payload(
                payload, vllm=(ep_cfg.backend_engine == "vllm"))

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
        except httpx.RemoteProtocolError as exc:
            # Backend dropped the connection — a server-closed keep-alive socket
            # reused from the pool, or a crash mid-stream. httpx normally
            # recovers from the keep-alive case transparently, but if it does
            # surface, map it to a clean BackendUnavailable instead of letting
            # the raw httpx error propagate (parity with call()).
            raise BackendUnavailable(f"backend {ep_cfg.role} disconnected mid-stream: {exc}")
        except (httpx.TimeoutException, asyncio.TimeoutError):
            # stream() bounds reads via httpx.Timeout, which raises
            # httpx.ReadTimeout (a TimeoutException) — not asyncio.TimeoutError.
            raise BackendTimeout(f"backend {ep_cfg.role} stream timeout after {timeout_s}s")
        except httpx.HTTPError as exc:
            raise BackendError(502, f"backend {ep_cfg.role} stream http error: {exc}")

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

    async def call_shadow(
        self,
        shadow_host: str,
        shadow_port: int,
        payload: dict,
        payload_type: str,
        request_id: str,
        timeout_s: float = 300.0,
    ) -> BackendResponse | None:
        """Fire-and-forget shadow call for A/B testing.

        Returns the response on success, None on any failure.
        Never raises — shadow failures must not affect the primary path.
        """
        client = self._client_for(shadow_host, shadow_port)
        path = self._path_for(payload_type)
        headers = {"X-Request-ID": f"shadow-{request_id}"}
        t0 = time.monotonic()
        try:
            resp = await asyncio.wait_for(
                client.post(path, json=payload, headers=headers),
                timeout=timeout_s,
            )
            duration = time.monotonic() - t0
            if resp.status_code >= 400:
                return None
            body = resp.json()
            usage = body.get("usage") or {}
            return BackendResponse(
                status_code=resp.status_code,
                body=body,
                duration_s=duration,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )
        except Exception:
            return None

    @staticmethod
    def _path_for(payload_type: str) -> str:
        if payload_type == "embedding":
            return "/embed"
        if payload_type == "rerank":
            return "/rerank"
        return "/v1/chat/completions"
