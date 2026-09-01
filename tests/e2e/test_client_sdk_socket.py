"""The SDK over a REAL SOCKET — the half `ASGITransport` cannot reach.

`tests/e2e/test_client_sdk_live.py` drives the SDK against the proxy through
`httpx.ASGITransport`, and says so plainly: what it exercises is request
building, envelope parsing and error typing, not httpx's networking. That left
the SDK's own **transport configuration** unobserved — asserted in
`tests/test_client_sdk.py` as a comparison between two constants, which proves
the arithmetic and nothing about the pool.

The keepalive is the one that matters. `docs/api.md` §1.3: the client must retire
an idle socket FIRST, with real margin, because whichever side closes second can
hand the other a socket it has already closed — surfacing as a transport error
for a request that was never attempted. `CLIENT_KEEPALIVE_EXPIRY_S` is 4.5s and
the server's is 30s, and 4.5s is short enough to **watch it happen**.

So this file runs the real proxy under real uvicorn on a real ephemeral socket,
points a client the SDK built ITSELF at it (no injected `httpx.AsyncClient`, or
the thing under test would be the fixture), and observes: a real round trip,
real SSE framing over the wire, connection reuse inside the window, and the
socket actually being retired outside it.
"""
from __future__ import annotations

import asyncio
import dataclasses
import socket
import tempfile
import threading
import time
from typing import Iterator

import pytest
import uvicorn

from roadstead.__main__ import PROXY_SERVER_KEEPALIVE_S, build_app
from roadstead.client import AsyncRoadsteadClient
from roadstead.client._wire import CLIENT_KEEPALIVE_EXPIRY_S
from roadstead.testing import FakeBackend, FakeBackendServer

from .conftest import _repointed_config


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


class _ProxyServer:
    """The real proxy under real uvicorn, on a real socket, in a daemon thread.

    🚨 The service is built and started INSIDE the server thread, by uvicorn's
    lifespan — exactly as production does it. Constructing it on one loop and
    serving it on another is the concurrency violation this repo is built
    around, and a harness that did it would be arming the guard against itself.
    """

    def __init__(self, config) -> None:
        self.host = "127.0.0.1"
        self.port = _free_port()
        self._config = config
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, timeout: float = 20.0) -> "_ProxyServer":
        cfg = self._config

        def _factory():
            app = build_app(cfg)
            svc = app.state.proxy_service

            async def _healthy(ep_cfg):
                return True

            svc._backend.probe_health = _healthy
            return app

        config = uvicorn.Config(
            _factory(), host=self.host, port=self.port,
            log_level="warning", access_log=False,
            timeout_keep_alive=PROXY_SERVER_KEEPALIVE_S,
        )
        self._server = uvicorn.Server(config)
        self._server.install_signal_handlers = lambda: None
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if getattr(self._server, "started", False):
                return self
            time.sleep(0.02)
        raise RuntimeError("proxy did not start")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=30.0)


@pytest.fixture
def live_proxy(fake: FakeBackendServer) -> Iterator[_ProxyServer]:
    with tempfile.TemporaryDirectory() as tmp:
        srv = _ProxyServer(
            _repointed_config(fake.host, fake.port, f"{tmp}/queue.db")).start()
        try:
            yield srv
        finally:
            srv.stop()


def _pool(client: AsyncRoadsteadClient):
    """httpx's live connection pool.

    🚨 Reaching into httpx internals, deliberately and with the alternative in
    mind: the thing being checked IS the transport configuration, and the
    version of this assertion that does not reach in is the constant comparison
    this file exists to replace. It asserts the attributes exist rather than
    skipping when they do not, so an httpx upgrade that moves them fails loudly
    and somebody re-points the test instead of the guard going quietly blind.
    """
    transport = client._client._transport
    assert hasattr(transport, "_pool"), (
        "httpx moved the connection pool — this guard needs re-pointing, not "
        "deleting")
    return transport._pool


async def test_the_sdk_round_trips_over_a_real_socket(live_proxy):
    """Real TCP, real HTTP framing, and a client the SDK built itself.

    No injected `httpx.AsyncClient`: an injected one is the fixture's
    configuration, and the SDK's own is what has never been exercised.
    """
    async with AsyncRoadsteadClient(live_proxy.url) as sdk:
        models = await sdk.models()
        assert {m.endpoint for m in models} >= {"tier1", "tier2", "tier3"}

        result = await sdk.chat(
            intent="fast-chat",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=16)
        assert result.content
        assert result.attribution.endpoint


async def test_a_stream_is_framed_correctly_on_the_wire(live_proxy):
    """SSE over a real socket, not an in-process async generator.

    `ASGITransport` hands the client the app's chunks directly; a real socket
    adds chunked transfer encoding and the framing rules the relay has to get
    right. This is the layer where a malformed `data:` line or a missing blank
    line stops being invisible.
    """
    async with AsyncRoadsteadClient(live_proxy.url) as sdk:
        chunks = [c async for c in sdk.stream(
            intent="fast-chat",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=16)]
    assert chunks
    kinds = [c.get("type") for c in chunks]
    assert "done" in kinds, kinds


async def test_the_sdk_reuses_one_connection_inside_the_keepalive_window(live_proxy):
    """The pool is doing its job at all — the control for the test below.

    Without it, "a new connection appeared" would be indistinguishable from a
    client that never pooled anything.
    """
    async with AsyncRoadsteadClient(live_proxy.url) as sdk:
        await sdk.models()
        first = list(_pool(sdk).connections)
        assert len(first) == 1, first
        await sdk.models()
        await sdk.models()
        again = list(_pool(sdk).connections)
        assert len(again) == 1, again
        assert again[0] is first[0], "the SDK opened a new connection per request"


@pytest.mark.heavy
async def test_the_sdk_retires_an_idle_socket_before_the_server_would(live_proxy):
    """🚨 §1.3's ordering invariant, OBSERVED rather than computed.

    The SDK's `keepalive_expiry` is 4.5s and the proxy's idle timeout is 30s.
    The client must let go first: whichever side closes second can hand the
    other a socket it has already closed, which surfaces as a transport error
    for a request that was never attempted — and that failure is intermittent,
    load-dependent and blamed on the network.

    Waiting out 4.5s of real idle is the whole point. A version of this that
    compared two numbers is what `tests/test_client_sdk.py` already does, and it
    would pass against a client that ignored the limit entirely.
    """
    assert CLIENT_KEEPALIVE_EXPIRY_S + 5.0 <= PROXY_SERVER_KEEPALIVE_S

    async with AsyncRoadsteadClient(live_proxy.url) as sdk:
        await sdk.models()
        before = list(_pool(sdk).connections)
        assert len(before) == 1

        await asyncio.sleep(CLIENT_KEEPALIVE_EXPIRY_S + 1.0)

        await sdk.models()
        after = list(_pool(sdk).connections)
        assert len(after) == 1, after
        assert after[0] is not before[0], (
            "the SDK reused a socket it should have retired — the server would "
            "then be the one to close it, which is the ordering §1.3 forbids")
