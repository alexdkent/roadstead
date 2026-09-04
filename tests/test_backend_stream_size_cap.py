"""P2 audit hardening — a running byte cap on a streamed backend response.

`BackendClientPool.stream()`'s `aiter_lines()` loop had no upper bound: a
wedged or adversarial backend that never stops talking (no `[DONE]`, no
chunk that ends the connection) could grow the proxy's own memory without
bound, one buffered line at a time. `ROADSTEAD_MAX_RESPONSE_BYTES`
(`config.max_response_bytes_from_env`, default 64 MiB) bounds bytes SEEN,
checked line-by-line as they arrive rather than after the whole response has
already been buffered.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from roadstead.backend import (
    BackendClientPool,
    BackendError,
    BackendTimeout,
    BackendUnavailable,
)
from roadstead.config import DEFAULT_MAX_RESPONSE_BYTES, EndpointConfig

EP = EndpointConfig(endpoint_class="chat", role="tier3", host="h", port=1234)


def _pool_with(handler) -> BackendClientPool:
    pool = BackendClientPool()
    client = httpx.AsyncClient(
        base_url=f"http://{EP.host}:{EP.port}",
        transport=httpx.MockTransport(handler),
    )
    pool._clients[EP.backend_url] = (client, 10_000)
    return pool


async def _drain(pool: BackendClientPool) -> list:
    events = []
    async for event in pool.stream(EP, {"messages": []}, "chat_completion", "rid",
                                   timeout_s=5):
        events.append(event)
    return events


def test_an_oversized_stream_is_aborted_not_buffered(monkeypatch):
    monkeypatch.setenv("ROADSTEAD_MAX_RESPONSE_BYTES", "40")

    # No [DONE], no natural end short of the cap — a wedged/adversarial
    # backend, in miniature.
    body = b"".join(f'data: {{"chunk": {i}}}\n\n'.encode() for i in range(50))

    def handler(request):
        return httpx.Response(200, content=body,
                              headers={"content-type": "text/event-stream"})

    with pytest.raises(BackendError) as exc:
        asyncio.run(_drain(_pool_with(handler)))
    # Deterministic, not a transient infra fault: the same request would
    # reproduce the same oversized response, so it must not be classified
    # alongside a dropped connection or a slow backend.
    assert not isinstance(exc.value, (BackendTimeout, BackendUnavailable))
    assert exc.value.status_code == 502
    assert "exceeded" in exc.value.detail
    assert "ROADSTEAD_MAX_RESPONSE_BYTES" in exc.value.detail


def test_a_stream_under_the_cap_is_unaffected():
    # Default cap (64 MiB) — nowhere near what a normal test payload sends.
    body = b'data: {"chunk": 0}\n\ndata: [DONE]\n\n'

    def handler(request):
        return httpx.Response(200, content=body,
                              headers={"content-type": "text/event-stream"})

    events = asyncio.run(_drain(_pool_with(handler)))
    assert [e.event_type for e in events] == ["chunk", "done"]


def test_the_default_is_sixty_four_mebibytes():
    assert DEFAULT_MAX_RESPONSE_BYTES == 64 * 1024 * 1024
