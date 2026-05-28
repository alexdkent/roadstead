"""Exception-mapping tests for BackendClientPool.stream() (WS4b hardening).

The streaming path used to catch only httpx.ConnectError + asyncio.TimeoutError.
But a llama-server keep-alive drop raises httpx.RemoteProtocolError, and a stream
read timeout raises httpx.ReadTimeout (a TimeoutException) — not
asyncio.TimeoutError. Both used to escape as raw httpx errors. stream() must now
map them to clean Backend* exceptions, for parity with call().
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from originfleet.llmproxy.backend import (
    BackendClientPool,
    BackendError,
    BackendTimeout,
    BackendUnavailable,
)
from originfleet.llmproxy.config import EndpointConfig

EP = EndpointConfig(endpoint_class="chat", role="thinker", host="h", port=1234)


def _pool_with(handler) -> BackendClientPool:
    pool = BackendClientPool()
    pool._clients[f"{EP.host}:{EP.port}"] = httpx.AsyncClient(
        base_url=f"http://{EP.host}:{EP.port}",
        transport=httpx.MockTransport(handler),
    )
    return pool


async def _drain(pool: BackendClientPool) -> None:
    async for _ in pool.stream(EP, {"messages": []}, "chat_completion", "rid",
                               timeout_s=5):
        pass


def test_stream_remote_protocol_error_maps_to_unavailable():
    def handler(request):
        raise httpx.RemoteProtocolError("Server disconnected without sending a response.")

    with pytest.raises(BackendError) as exc:
        asyncio.run(_drain(_pool_with(handler)))
    assert isinstance(exc.value, BackendUnavailable)
    assert exc.value.status_code == 503


def test_stream_read_timeout_maps_to_timeout():
    def handler(request):
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(BackendError) as exc:
        asyncio.run(_drain(_pool_with(handler)))
    assert isinstance(exc.value, BackendTimeout)
    assert exc.value.status_code == 504


def test_stream_other_httperror_maps_to_backend_error():
    def handler(request):
        raise httpx.LocalProtocolError("malformed")

    with pytest.raises(BackendError) as exc:
        asyncio.run(_drain(_pool_with(handler)))
    # the catch-all branch — not unavailable/timeout
    assert exc.value.status_code == 502


def test_stream_connect_error_still_unavailable():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(BackendError) as exc:
        asyncio.run(_drain(_pool_with(handler)))
    assert isinstance(exc.value, BackendUnavailable)
    assert exc.value.status_code == 503
