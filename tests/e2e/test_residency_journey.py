"""JOURNEY — honest availability for an endpoint a host dispatcher loads and
evicts (`roadstead/residency.py`, `Health.endpoint_status`).

    dispatcher evicts a tenant -> its endpoint reads `unloaded` (not healthy,
    not paged) -> the dispatcher stops answering -> the same silence now reads
    `unreachable` / `unhealthy`, never excused -> the tenant comes back ->
    `healthy`. And a request to the evicted endpoint wakes NOTHING.

Not `test_residency.py` (24 units cover the reader and the state matrix against
a faked transport). This one proves the capability is REACHED: a real
``ProxyService`` from ``build_app``, its real poller driving on a 50 ms tick,
and a REAL socket standing in for the dispatcher, read back through the real
``/v1/status`` front door. Two seams are faked and nothing else: the dispatcher
(a tiny ASGI app that records every request it receives) and ``probe_health``
(true only for the endpoints pointed at the live fake backend, false for the two
dead ones — the equivalent of "nothing is listening on the port").

The defect it exists for (2026-09-29): `/v1/status` said `healthy: true` for an
evicted, unlistened-on endpoint whose every call failed `503 unreachable`.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import socket
import tempfile
import threading
import time
from typing import AsyncIterator

import httpx
import pytest
import pytest_asyncio
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from roadstead.__main__ import build_app
from roadstead.config import EndpointConfig, ProxyConfig
from roadstead.residency import ResidencyReader
from roadstead.testing import FakeBackend, FakeBackendServer

from tests.admin_key import enrol_admin

OD = "od-ep"          # on-demand, nothing manages it (the shape that read healthy)
FLASH = "flash-ep"    # plain endpoint that declares a tenant (a failover-eligible one)

_INTERNAL_CLIENT = ("127.0.0.1", 41997)


class FakeDispatcher:
    """A real socket answering ``GET /status`` with a controllable
    ``intended_state``, and RECORDING every request it gets — so a test can
    assert the proxy only ever read it."""

    def __init__(self) -> None:
        self.intent: dict[str, str] = {}
        self.broken = False
        self.requests: list[tuple[str, str]] = []
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()

        async def any_route(request):
            self.requests.append((request.method, request.url.path))
            if request.url.path == "/status" and request.method == "GET" and not self.broken:
                return JSONResponse({"loaded": "x", "intended_state": self.intent})
            return JSONResponse({"error": "no"}, status_code=500 if self.broken else 404)

        self.app = Starlette(routes=[
            Route("/{path:path}", any_route, methods=["GET", "POST", "PUT", "DELETE"])])
        self._server = uvicorn.Server(uvicorn.Config(
            self.app, host="127.0.0.1", port=self.port, log_level="warning",
            access_log=False, lifespan="off"))
        self._server.install_signal_handlers = lambda: None
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "FakeDispatcher":
        self._thread.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not self._server.started:
            time.sleep(0.02)
        return self

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)


async def _wait_until(predicate, timeout_s: float = 8.0, interval_s: float = 0.05):
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval_s)
    return predicate()


@pytest_asyncio.fixture
async def journey(caplog) -> AsyncIterator[dict]:
    caplog.set_level(logging.WARNING, logger="roadstead.health")
    backend = FakeBackendServer(FakeBackend()).start()
    dispatcher = FakeDispatcher().start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            base = ProxyConfig(queue_db_path=f"{tmp}/queue.db")
            endpoints = {
                cls: dataclasses.replace(
                    ep, host=backend.host, port=backend.port,
                    max_slots=ep.max_slots or 4,
                    context_per_slot=ep.context_per_slot or 8192)
                for cls, ep in base.endpoints.items()
            }
            for name, kw in ((OD, dict(on_demand=True, residency_tenant="t-od")),
                             (FLASH, dict(residency_tenant="t-flash"))):
                endpoints[name] = EndpointConfig(
                    endpoint_class=name, role=name, host="127.0.0.1", port=1,
                    max_slots=1, context_per_slot=8192, min_expected_slots=1, **kw)
            base.endpoints = endpoints
            base.poller_interval_s = 0.05
            app = build_app(base)
            svc = app.state.proxy_service
            svc._state.residency = ResidencyReader(
                base.endpoints, dispatcher_url=dispatcher.url)

            async def _probe(ep_cfg):
                return ep_cfg.port == backend.port     # the dead two answer nothing
            svc._backend.probe_health = _probe
            enrol_admin(svc)
            await svc.startup()
            client = httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=app, raise_app_exceptions=False, client=_INTERNAL_CLIENT),
                base_url="http://proxy", timeout=30.0)
            try:
                yield {"svc": svc, "client": client, "dispatcher": dispatcher,
                       "backend": backend, "cfg": base}
            finally:
                await client.aclose()
                await svc.shutdown()
    finally:
        dispatcher.stop()
        backend.stop()


async def _eps(client: httpx.AsyncClient) -> dict:
    resp = await client.get("/v1/status")
    assert resp.status_code == 200
    return resp.json()


async def _wait_state(client, ep: str, state: str) -> dict:
    seen: dict = {}

    async def poll():
        seen.update((await _eps(client))["endpoints"][ep])
        return seen.get("state") == state

    deadline = asyncio.get_event_loop().time() + 8.0
    while asyncio.get_event_loop().time() < deadline:
        if await poll():
            return seen
        await asyncio.sleep(0.05)
    raise AssertionError(f"{ep} never reached state {state!r}; last seen {seen}")


@pytest.mark.asyncio
async def test_evicted_then_unexplained_then_loaded(journey):
    client, disp = journey["client"], journey["dispatcher"]

    # 1. The dispatcher has both tenants evicted; nothing listens on either.
    disp.intent = {"t-od": "evicted", "t-flash": "evicted"}
    for ep in (OD, FLASH):
        snap = await _wait_state(client, ep, "unloaded")
        assert snap["healthy"] is False          # the defect: this read True
        assert snap["residency"] == "evicted"
        assert snap["reachable"] is False
        assert snap["paused"] is False           # down on purpose, not "unexpectedly down"

    # ...and it pages nobody. Wait for a poller pass that ran AFTER the state
    # settled, then read the alerts the proxy publishes.
    await asyncio.sleep(0.3)
    alerts = (await _eps(client))["alerts"]
    assert not [a for a in alerts if a["name"] == "endpoint_paused"
                and (OD in a["detail"] or FLASH in a["detail"])], alerts

    # 2. A request to the evicted endpoint wakes NOTHING: the proxy's only
    # dealings with the dispatcher are reads of /status.
    resp = await client.post("/rs/v1/chat", json={
        "model": OD, "call_site": "journey.evicted", "payload_type": "chat_completion",
        "payload": {"model": OD, "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 8}})
    assert resp.status_code >= 400
    assert disp.requests and set(disp.requests) == {("GET", "/status")}, set(disp.requests)

    # 3. The dispatcher stops answering. The same silence is no longer excused.
    disp.broken = True
    snap = await _wait_state(client, OD, "unreachable")
    assert snap["residency"] == "unknown" and snap["healthy"] is False
    snap = await _wait_state(client, FLASH, "unhealthy")
    assert snap["residency"] == "unknown" and snap["healthy"] is False
    # Positive control for the alert half above: this path DOES page.
    assert await _wait_until(lambda: any(
        a["name"] == "endpoint_paused" and FLASH in a["detail"]
        for a in journey["svc"]._state.alerts))

    # 4. The tenant is loaded (someone's lease): its endpoint answers, and only
    # now — with a probe that answered — does it read healthy.
    disp.broken = False
    disp.intent = {"t-od": "resident", "t-flash": "resident"}
    od_cfg = journey["cfg"].endpoints[OD]
    od_cfg.host, od_cfg.port = journey["backend"].host, journey["backend"].port
    snap = await _wait_state(client, OD, "healthy")
    assert snap["healthy"] is True and snap["residency"] == "resident"
    assert snap["reachable"] is True

    # Still read-only after all of that.
    assert set(disp.requests) == {("GET", "/status")}, set(disp.requests)
