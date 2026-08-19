"""`/readyz` fails CLOSED on the conversational endpoint (§8.0 req 4).

`/health` is fail-OPEN by design — a dead backend degrades its status but must
not 503 the proxy (alert-don't-kill). That is right for liveness and useless for
routing, so go-dark needs a second endpoint that answers the opposite question:
should callers still be dispatching?

These assert against the REAL ASGI app via httpx.ASGITransport rather than a
live proxy, because the house rule is that route registration is a SOURCE fact
or a TEST fact — never something to probe by firing at the live fleet.
"""
from __future__ import annotations

import tempfile

import httpx
import pytest
import pytest_asyncio

from originfleet.llmproxy.__main__ import build_app
from originfleet.llmproxy.config import ProxyConfig
from originfleet.llmproxy.model_catalog import build_endpoint_kwargs

# The proxy ACLs some routes by source IP; use the internal client tuple the
# other proxy tests use so this exercises the real middleware rather than
# tripping an ACL that has nothing to do with readiness.
_INTERNAL_CLIENT = ("127.0.0.1", 41999)


# --------------------------------------------------------------------------
# The flag is wired at all — a key missing from build_endpoint_kwargs()'s
# passthrough tuple is SILENTLY DROPPED from models.yaml, so this is the guard
# that comment asks for.
# --------------------------------------------------------------------------

def test_readiness_critical_flag_reaches_endpoint_config():
    kwargs = build_endpoint_kwargs()
    marked = [c for c, kw in kwargs.items() if kw.get("readiness_critical")]
    assert marked, (
        "no endpoint class carries readiness_critical. Either models.yaml lost "
        "the flag or it was dropped by build_endpoint_kwargs()'s passthrough "
        "tuple — in which case /readyz can never fail closed and would report "
        "READY over a dead conversational backend.")


def test_exactly_one_endpoint_is_conversational():
    """It marks THE conversational lane, singular.

    Two would mean readiness 503s when either is down, which is a broader claim
    than go-dark makes. At the Phase 3 cutover the flag MOVES from
    tier2-analyst to tier2-chat; it is never on both.
    """
    kwargs = build_endpoint_kwargs()
    marked = [c for c, kw in kwargs.items() if kw.get("readiness_critical")]
    assert len(marked) == 1, f"expected exactly one, got {marked}"


def test_the_flag_is_not_on_every_chat_endpoint():
    """Guard the guard: prove it discriminates.

    If `readiness_critical` were derived from `kind: chat` it would cover tier1
    router and tier3 reasoner too, and the tests above would still pass while
    /readyz 503'd the fleet over a long-form authoring outage.
    """
    kwargs = build_endpoint_kwargs()
    assert len(kwargs) > 1, "catalog has one endpoint; this test proves nothing"
    unmarked = [c for c, kw in kwargs.items() if not kw.get("readiness_critical")]
    assert unmarked, "every endpoint is readiness_critical — the flag discriminates nothing"


# --------------------------------------------------------------------------
# Behaviour, against the real app.
# --------------------------------------------------------------------------

@pytest_asyncio.fixture
async def proxy():
    """Real app via `build_app`, no backend sockets — `/readyz` reads state, and
    a fake backend would only add flakiness. Health probing is stubbed healthy
    so the circuit starts closed and each test opens it deliberately."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = ProxyConfig(queue_db_path=f"{tmp}/q.db")
        cfg.poller_interval_s = 3600          # don't let the poller race us
        app = build_app(cfg)
        svc = app.state.proxy_service

        async def _healthy(ep_cfg):
            return True
        svc._backend.probe_health = _healthy

        await svc.startup()
        transport = httpx.ASGITransport(
            app=app, raise_app_exceptions=False, client=_INTERNAL_CLIENT)
        client = httpx.AsyncClient(transport=transport, base_url="http://proxy",
                                   timeout=30.0)
        try:
            yield svc, client
        finally:
            await client.aclose()
            await svc.shutdown()


def _critical(svc) -> str:
    names = [n for n, c in svc._state.config.endpoints.items()
             if getattr(c, "readiness_critical", False)]
    assert names, "no conversational endpoint declared — see models.yaml"
    return names[0]


@pytest.mark.asyncio
async def test_readyz_is_registered_and_green_when_healthy(proxy):
    svc, client = proxy
    r = await client.get("/readyz")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ready"] is True
    assert body["readiness_critical_endpoints"], (
        "readyz reported ready with NO conversational endpoint declared — that "
        "is a config error masquerading as a clean bill of health")


@pytest.mark.asyncio
async def test_readyz_503s_when_the_conversational_backend_is_circuit_open(proxy):
    svc, client = proxy
    name = _critical(svc)
    svc._state.endpoint_health[name]["healthy"] = False
    r = await client.get("/readyz")
    assert r.status_code == 503, r.text
    body = r.json()
    assert body["ready"] is False
    assert body["unready"].get(name) == "circuit_open"


@pytest.mark.asyncio
async def test_readyz_503s_when_the_conversational_backend_is_paused(proxy):
    """An operator drain must read as NOT ready too — the box is deliberately
    down, which is exactly when callers should stop dispatching."""
    svc, client = proxy
    name = _critical(svc)
    svc._state.paused_endpoints.add(name)
    r = await client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["unready"].get(name) == "paused"


@pytest.mark.asyncio
async def test_a_non_conversational_backend_going_down_does_NOT_503(proxy):
    """The discriminating case, and the reason this is a declared flag rather
    than `kind: chat`: tier1/tier3 dying must not make the fleet unready."""
    svc, client = proxy
    name = _critical(svc)
    others = [n for n in svc._state.config.endpoints if n != name]
    assert others, "only one endpoint configured; this test proves nothing"
    for n in others:
        svc._state.endpoint_health[n]["healthy"] = False
    r = await client.get("/readyz")
    assert r.status_code == 200, (
        "every non-conversational endpoint is down and /readyz went 503 — it is "
        f"reacting to more than the conversational lane: {r.text}")


@pytest.mark.asyncio
async def test_health_stays_fail_open_when_readyz_is_closed(proxy):
    """The two endpoints must DISAGREE — that disagreement is the whole design.

    If a future edit made /health 503 on a dead backend, a monitor would start
    restarting a healthy front door over a backend blip.
    """
    svc, client = proxy
    name = _critical(svc)
    svc._state.endpoint_health[name]["healthy"] = False
    ready = await client.get("/readyz")
    health = await client.get("/health")
    assert ready.status_code == 503
    assert health.status_code == 200, (
        "/health must NOT 503 on a dead backend — it is liveness, and killing "
        "the proxy for a backend blip is the failure alert-don't-kill prevents")
    assert health.json()["status"] == "degraded"
