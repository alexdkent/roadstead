"""E2E harness fixtures for the LLM proxy (Phase T).

Spins the *real* ``ProxyService`` in-process (via ``build_app``) with its
backends repointed at the programmable :class:`FakeBackendServer` (a real
ephemeral socket). The proxy's own front door is driven over
``httpx.ASGITransport`` from a loopback client (``127.0.0.0/8`` → the ACL's
``internal`` identity, so no 403). North face = real proxy HTTP; south face =
real fake socket. No GPU, no live network.

Fixtures:
  ``fake``   — a running :class:`FakeBackendServer`; ``fake.controller`` steers faults.
  ``proxy``  — a :class:`ProxyHarness` wrapping the started service + an ASGI client.

The parent ``tests/llmproxy/conftest.py`` autouse-stubs the capacity probes to
the class level (no real network). We additionally override the *instance*
``probe_health`` to report healthy so the circuit breaker can never trip on
probe timing during a fast test (it otherwise needs 3 poller rounds + /health
down — ~30s — but we keep the guarantee timing-independent).
"""

from __future__ import annotations

import dataclasses
import tempfile
import types
from typing import Any, AsyncIterator, Dict, Iterator, Optional

import httpx
import pytest
import pytest_asyncio

from roadstead.backend import BackendClientPool
from roadstead.config import EndpointConfig, ProxyConfig
from roadstead.__main__ import build_app

from roadstead.testing import FakeBackend, FakeBackendServer

#: Captured at import, before the parent conftest's autouse fixture replaces it.
_REAL_PREFIX_CACHE = BackendClientPool.probe_prefix_cache


# Loopback client for the ASGI transport → ACL "internal" identity.
_INTERNAL_CLIENT = ("127.0.0.1", 41999)


@pytest.fixture
def fake() -> Iterator[FakeBackendServer]:
    """A running fake backend on a real ephemeral socket."""
    srv = FakeBackendServer(FakeBackend()).start()
    try:
        yield srv
    finally:
        srv.stop()


def _repointed_config(host: str, port: int, queue_db: str) -> ProxyConfig:
    """A ProxyConfig whose every endpoint points at ``host:port`` (the fake),
    with capacity seeded so admission has slots. EndpointConfig objects are
    freshly built (dataclasses.replace) so module-global DEFAULT_ENDPOINTS is
    never mutated across tests."""
    base = ProxyConfig(queue_db_path=queue_db)
    endpoints: Dict[str, EndpointConfig] = {}
    for cls, ep in base.endpoints.items():
        endpoints[cls] = dataclasses.replace(
            ep,
            host=host,
            port=port,
            max_slots=ep.max_slots or 4,
            context_per_slot=ep.context_per_slot or 8192,
        )
    base.endpoints = endpoints
    # Fast poller so any poller-driven behaviour is observable without 10s waits.
    base.poller_interval_s = 0.05
    return base


class ProxyHarness:
    """Wraps a started ProxyService + an ASGI client driving its front door."""

    def __init__(self, app: Any, svc: Any, client: httpx.AsyncClient,
                 fake: FakeBackendServer) -> None:
        self.app = app
        self.svc = svc
        self.client = client
        self.fake = fake

    @property
    def controller(self) -> FakeBackend:
        return self.fake.controller

    def in_flight(self, endpoint: str = "chat") -> int:
        """Live in-flight (dispatched, not-yet-completed) count for an endpoint —
        the slot-accounting invariant reads this before/after each call."""
        return self.svc._scheduler.endpoint_snapshot(endpoint)["in_flight"]

    def total_in_flight(self) -> int:
        return sum(
            self.svc._scheduler.endpoint_snapshot(ep)["in_flight"]
            for ep in self.svc._config.endpoints
        )

    async def chat(self, content: str = "hello", *, model: str = "chat",
                   stream: bool = False, timeout_s: Optional[float] = None,
                   fault: Optional[str] = None, fault_arg: float = 0.0,
                   fault_max_hits: int = 0,
                   extra: Optional[dict] = None) -> httpx.Response:
        # NB: the fault is set on the fake CONTROLLER (server state), not via a
        # request header — the proxy forwards only X-Request-ID to the backend,
        # so a header fault would never reach the fake. Server-state faults do.
        if fault:
            self.controller.set_fault(fault, fault_arg, fault_max_hits)
        body: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 16,
            "stream": stream,
        }
        if timeout_s is not None:
            body["timeout_s"] = timeout_s
        if extra:
            body.update(extra)
        return await self.client.post("/v1/chat/completions", json=body)

    async def stream_frames(self, content: str = "a b c", *, model: str = "chat",
                            fault: Optional[str] = None, fault_arg: float = 0.0,
                            timeout_s: Optional[float] = None) -> list:
        """POST a streaming chat and return the list of raw SSE ``data:`` payloads."""
        if fault:
            self.controller.set_fault(fault, fault_arg)
        body: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 16,
            "stream": True,
        }
        if timeout_s is not None:
            body["timeout_s"] = timeout_s
        frames: list = []
        async with self.client.stream("POST", "/v1/chat/completions", json=body) as resp:
            async for line in resp.aiter_lines():
                line = line.strip()
                if line.startswith("data: "):
                    frames.append(line[len("data: "):])
        return frames


@pytest_asyncio.fixture
async def proxy(fake: FakeBackendServer) -> AsyncIterator[ProxyHarness]:
    with tempfile.TemporaryDirectory() as tmp:
        config = _repointed_config(fake.host, fake.port, f"{tmp}/queue.db")
        app = build_app(config)
        svc = app.state.proxy_service

        async def _healthy(ep_cfg):
            return True

        # Instance override (shadows the parent conftest class stub) so the
        # circuit never trips on probe timing during the test window.
        svc._backend.probe_health = _healthy
        # Same pattern for the prefix-cache scrape: the parent conftest stubs it
        # to "cannot tell" so the unit suite never dials a documentation address,
        # but here every endpoint points at the local fake, which publishes the
        # real counters. Restore the real method so the cache-stats path is
        # exercised end to end rather than mocked away.
        svc._backend.probe_prefix_cache = types.MethodType(
            _REAL_PREFIX_CACHE, svc._backend)

        await svc.startup()
        transport = httpx.ASGITransport(
            app=app, raise_app_exceptions=False, client=_INTERNAL_CLIENT)
        client = httpx.AsyncClient(transport=transport, base_url="http://proxy",
                                   timeout=30.0)
        try:
            yield ProxyHarness(app, svc, client, fake)
        finally:
            await client.aclose()
            await svc.shutdown()
