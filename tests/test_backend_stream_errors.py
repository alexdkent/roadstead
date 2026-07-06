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
    client = httpx.AsyncClient(
        base_url=f"http://{EP.host}:{EP.port}",
        transport=httpx.MockTransport(handler),
    )
    # Cache shape is (client, pool size it was built with); a huge size keeps
    # _client_for from retiring this mock for a slot-sized rebuild.
    pool._clients[f"{EP.host}:{EP.port}"] = (client, 10_000)
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


# --- empty-completion gate (non-streaming call) ---

def _json_handler(body):
    def handler(request):
        return httpx.Response(200, json=body)
    return handler


def _call(pool):
    return asyncio.run(pool.call(EP, {"messages": [{"role": "user", "content": "x"}]},
                                 "chat_completion", "rid", timeout_s=5))


def test_call_remote_protocol_error_maps_to_unavailable():
    # Parity with the stream path (2026-07-06 keepalive-disconnect fix): a backend
    # keep-alive drop on the NON-stream call() must be transient (BackendUnavailable
    # → in-proxy retry), not a hard 502 that surfaces to the caller.
    def handler(request):
        raise httpx.RemoteProtocolError("Server disconnected without sending a response.")

    with pytest.raises(BackendError) as exc:
        _call(_pool_with(handler))
    assert isinstance(exc.value, BackendUnavailable)
    assert exc.value.status_code == 503


def test_call_empty_completion_raises():
    body = {"choices": [{"message": {"content": ""}}], "usage": {"completion_tokens": 0}}
    with pytest.raises(BackendError) as exc:
        _call(_pool_with(_json_handler(body)))
    assert exc.value.status_code == 502
    assert "empty completion" in exc.value.detail


def test_call_no_choices_raises():
    with pytest.raises(BackendError):
        _call(_pool_with(_json_handler({"choices": [], "usage": {}})))


def test_call_tool_call_response_ok():
    # empty content but tool_calls present -> valid, must NOT be gated
    body = {"choices": [{"message": {"content": "", "tool_calls": [{"id": "1"}]}}],
            "usage": {"completion_tokens": 5}}
    resp = _call(_pool_with(_json_handler(body)))
    assert resp.status_code == 200


def test_call_nonempty_ok():
    body = {"choices": [{"message": {"content": "hello"}}], "usage": {"completion_tokens": 2}}
    resp = _call(_pool_with(_json_handler(body)))
    assert resp.output_tokens == 2
