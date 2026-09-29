"""Honest availability for an endpoint the poller cannot see — residency.py and
``Health.endpoint_status``.

THE DEFECT (2026-09-29, measured live): ``/v1/status`` reported an endpoint
``healthy: true`` while its host dispatcher had it EVICTED, nothing was
listening on its port, and every call to it failed ``503 backend ... unreachable``.
Cause: an ``on_demand`` endpoint that is not loaded is skipped by the capacity
poller (``poll_endpoint_once`` returned before probing), and an endpoint nothing
manages never has a lease, so it is "not loaded" forever — never probed, its
circuit breaker sat at its initial ``healthy: True`` for the life of the process,
and ``/v1/status`` projected that default as a fact. The same blind spot as the
2026-07-09 admin-pause one, second instance.

These tests are written to fail on the code before the fix: on it, ``state`` /
``residency`` / ``reachable`` do not exist and the dead on-demand endpoint reads
``healthy: True``.
"""
from __future__ import annotations

import json
import logging
import time

import httpx
import pytest

from roadstead import model_catalog
from roadstead import residency as residency_mod
from roadstead.config import EndpointConfig, ProxyConfig
from roadstead.residency import ResidencyReader
from roadstead.service import ProxyService

OD = "od-ep"      # on-demand, declares a tenant
PLAIN = "flash-ep"  # a plain (failover-eligible) endpoint that declares a tenant
BARE = "bare-ep"    # on-demand, declares NO tenant (nothing to ask a dispatcher)


class _Req:
    class _Client:
        host = "127.0.0.1"

    client = _Client()
    headers: dict = {}


def _reader(endpoints: dict, body=None, *, status=200, url="http://dispatcher.invalid",
            raises: Exception | None = None, calls: list | None = None) -> ResidencyReader:
    r = ResidencyReader(endpoints, dispatcher_url=url)

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append((request.method, request.url.path))
        if raises is not None:
            raise raises
        return httpx.Response(status, json=body)

    r._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return r


def _eps(**tenants: str) -> dict:
    return {name: EndpointConfig(endpoint_class=name, role=name, residency_tenant=t)
            for name, t in tenants.items()}


# ---------------------------------------------------------------- the reader

@pytest.mark.asyncio
async def test_the_three_words_and_the_non_answers():
    body = {"intended_state": {"a": "resident", "b": "evicted", "c": "unmanaged",
                               "d": "some-word-from-the-future"}}
    r = _reader(_eps(**{"ea": "a", "eb": "b", "ec": "c", "ed": "d", "ee": "absent-tenant"}),
                body)
    await r.refresh()
    assert r.state("ea") == "resident"
    assert r.state("eb") == "evicted"
    # Not the dispatcher's to say / not a word we know / not listed: cannot tell.
    assert r.state("ec") == "unknown"
    assert r.state("ed") == "unknown"
    assert r.state("ee") == "unknown"
    # An endpoint that declares nothing is ABSENT, which is not the same as unknown.
    assert r.state("undeclared") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kw", [
    dict(body={"intended_state": {"t": "evicted"}}, status=503),
    dict(body={"no_intended_state_here": 1}),
    dict(body={"intended_state": "not-a-map"}),
    dict(body=None, raises=httpx.ConnectError("refused")),
    dict(body={"intended_state": {"t": "evicted"}}, url=""),
], ids=["http-503", "wrong-shape", "not-a-map", "unreachable", "no-url"])
async def test_a_dispatcher_that_cannot_be_read_is_unknown_never_a_value(kw):
    r = _reader(_eps(e="t"), **kw)
    await r.refresh()
    assert r.state("e") == "unknown"


@pytest.mark.asyncio
async def test_a_failed_read_clears_the_last_answer_rather_than_keeping_it():
    """An old `evicted` must not keep excusing a failure once the dispatcher has
    stopped answering — the failure could be the very thing it was excusing."""
    ok = {"intended_state": {"t": "evicted"}}
    r = _reader(_eps(e="t"), ok)
    await r.refresh()
    assert r.state("e") == "evicted"
    r2 = _reader(_eps(e="t"), raises=httpx.ReadTimeout("slow"))
    r2._intent, r2._read_at = r._intent, r._read_at      # same cache, then a failed read
    await r2.refresh()
    assert r2.state("e") == "unknown"


@pytest.mark.asyncio
async def test_a_reading_that_stopped_refreshing_goes_stale_to_unknown(monkeypatch):
    r = _reader(_eps(e="t"), {"intended_state": {"t": "evicted"}})
    await r.refresh()
    assert r.state("e") == "evicted"
    monkeypatch.setattr(residency_mod, "_STALE_S", 0.0)
    time.sleep(0.01)
    assert r.state("e") == "unknown"


@pytest.mark.asyncio
async def test_no_declared_tenant_means_no_dispatcher_call_at_all():
    calls: list = []
    r = _reader(_eps(e=""), {"intended_state": {}}, calls=calls)
    assert not r.tracks
    await r.refresh()
    assert calls == []


@pytest.mark.asyncio
async def test_the_reader_only_ever_issues_a_GET_of_status():
    """READ-ONLY is the contract: a deployment forbids a request from waking a
    backend, and this reader is what stands next to that dispatcher."""
    calls: list = []
    r = _reader(_eps(e="t"), {"intended_state": {"t": "evicted"}}, calls=calls)
    for _ in range(3):
        await r.refresh()
    assert calls == [("GET", "/status")] * 3


# ------------------------------------------------- catalog -> EndpointConfig

def test_declared_tenant_reaches_endpoint_config():
    entry = model_catalog.EndpointEntry(
        name="probe", provider="p", kind="chat",
        policy={"residency_tenant": "the-tenant"})
    kw = model_catalog.build_endpoint_kwargs(entries=[entry])["probe"]
    assert kw["residency_tenant"] == "the-tenant"
    ep = EndpointConfig(**{k: v for k, v in kw.items()
                           if k in EndpointConfig.__dataclass_fields__})
    assert ep.residency_tenant == "the-tenant"


def test_undeclared_endpoint_has_no_tenant():
    """Absent -> off: nothing to ask, no field on /v1/status."""
    assert EndpointConfig(endpoint_class="x", role="x").residency_tenant == ""


# --------------------------------------------- Health.endpoint_status matrix

def _svc() -> ProxyService:
    cfg = ProxyConfig()
    cfg.endpoints[OD] = EndpointConfig(
        endpoint_class=OD, role=OD, on_demand=True, residency_tenant="t-od",
        host="127.0.0.1", port=1, max_slots=1, min_expected_slots=1)
    cfg.endpoints[PLAIN] = EndpointConfig(
        endpoint_class=PLAIN, role=PLAIN, residency_tenant="t-plain",
        host="127.0.0.1", port=1, max_slots=1, min_expected_slots=1)
    cfg.endpoints[BARE] = EndpointConfig(
        endpoint_class=BARE, role=BARE, on_demand=True,
        host="127.0.0.1", port=1, max_slots=1, min_expected_slots=1)
    svc = ProxyService(cfg)
    svc._state.residency = ResidencyReader(cfg.endpoints,
                                           dispatcher_url="http://dispatcher.invalid")
    return svc


def _set_intent(svc: ProxyService, **intent: str) -> None:
    """What the dispatcher answers. The poller refreshes the reader on every
    pass, so the answer has to come from the (faked) wire, not from poking the
    cache — a poked cache is overwritten by the first iteration."""
    r = svc._state.residency
    r._client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"intended_state": intent})))


async def _status(svc: ProxyService) -> dict:
    return json.loads((await svc.handle_status(_Req())).body)["endpoints"]


@pytest.mark.asyncio
async def test_a_dead_on_demand_endpoint_no_dispatcher_is_never_healthy():
    """THE DEFECT. Not loaded, nothing manages it, nothing answers, and no
    dispatcher to explain it: it was reported healthy. It must read as what it
    is — unreachable — not as a default."""
    svc = _svc()

    async def down(ep_cfg):
        return False
    svc._backend.probe_health = down
    for _ in range(4):
        await svc._poller_iteration()
    snap = (await _status(svc))[BARE]
    assert snap["healthy"] is False
    assert snap["state"] == "unreachable"
    assert snap["reachable"] is False
    assert "residency" not in snap          # declares none: absent, not "unknown"


@pytest.mark.asyncio
async def test_an_on_demand_endpoint_nothing_has_probed_yet_reads_unknown_not_healthy():
    svc = _svc()
    snap = (await _status(svc))[BARE]
    assert snap["state"] == "unknown"
    assert snap["healthy"] is False


@pytest.mark.asyncio
async def test_evicted_and_silent_is_unloaded_expected_and_not_healthy():
    svc = _svc()
    _set_intent(svc, **{"t-od": "evicted", "t-plain": "evicted"})

    async def down(ep_cfg):
        return False
    svc._backend.probe_health = down
    for _ in range(4):
        await svc._poller_iteration()
    snap = await _status(svc)
    for ep in (OD, PLAIN):
        assert snap[ep]["state"] == "unloaded", ep
        assert snap[ep]["healthy"] is False, ep
        assert snap[ep]["residency"] == "evicted"
        assert snap[ep]["reachable"] is False
        # Down on purpose is not the "unexpectedly down" the alert keys on.
        assert snap[ep]["paused"] is False, ep


@pytest.mark.asyncio
async def test_resident_intent_never_makes_a_silent_endpoint_healthy():
    """`intended_state` is an INTENT. A tenant the dispatcher believes resident
    whose process died must read as the fault it is."""
    svc = _svc()
    _set_intent(svc, **{"t-od": "resident", "t-plain": "resident"})

    async def down(ep_cfg):
        return False
    svc._backend.probe_health = down
    for _ in range(4):
        await svc._poller_iteration()
    snap = await _status(svc)
    assert snap[OD]["state"] == "unreachable" and snap[OD]["healthy"] is False
    # The plain one has a breaker: it tripped, and this time it IS a fault.
    assert snap[PLAIN]["state"] == "unhealthy" and snap[PLAIN]["healthy"] is False


@pytest.mark.asyncio
async def test_an_unreadable_dispatcher_never_excuses_a_silent_endpoint():
    svc = _svc()
    svc._state.residency._client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(503, json={"intended_state": {"t-od": "evicted"}})))

    async def down(ep_cfg):
        return False
    svc._backend.probe_health = down
    for _ in range(4):
        await svc._poller_iteration()
    snap = await _status(svc)
    assert snap[OD]["residency"] == "unknown"
    assert snap[OD]["state"] == "unreachable"
    assert snap[PLAIN]["state"] == "unhealthy"


@pytest.mark.asyncio
async def test_a_loaded_endpoint_reads_healthy_and_is_then_discovered():
    """Someone else's lease loaded it: the poller sees it answer and treats it
    as any other endpoint. This is the `healthy loaded` arm the other states
    are distinguished from."""
    svc = _svc()
    _set_intent(svc, **{"t-od": "resident", "t-plain": "resident"})

    async def up(ep_cfg):
        return True
    svc._backend.probe_health = up
    for _ in range(2):
        await svc._poller_iteration()
    snap = await _status(svc)
    for ep in (OD, PLAIN):
        assert snap[ep]["state"] == "healthy" and snap[ep]["healthy"] is True, ep
        assert snap[ep]["reachable"] is True


@pytest.mark.asyncio
async def test_an_evicted_endpoint_that_still_answers_is_not_called_unloaded():
    """Mid-eviction the intent flips before the process stops. The probe decides
    what is THERE; residency only explains a failure."""
    svc = _svc()
    _set_intent(svc, **{"t-od": "evicted", "t-plain": "evicted"})

    async def up(ep_cfg):
        return True
    svc._backend.probe_health = up
    await svc._poller_iteration()
    snap = await _status(svc)
    assert snap[OD]["state"] == "healthy"
    assert snap[OD]["residency"] == "evicted"


@pytest.mark.asyncio
async def test_a_breaker_tripped_while_up_does_not_outlive_an_on_demand_eviction():
    svc = _svc()
    svc._state.endpoint_health[OD] = {
        "healthy": False, "consecutive_failures": 3, "unhealthy_since": 1.0}
    _set_intent(svc, **{"t-od": "evicted"})

    async def down(ep_cfg):
        return False
    svc._backend.probe_health = down
    await svc._poller_iteration()
    assert svc._state.endpoint_health[OD]["healthy"] is True   # nothing left to recover it
    assert (await _status(svc))[OD]["state"] == "unloaded"


# ------------------------------------------------- alarms, breakers, logging

@pytest.mark.asyncio
async def test_an_evicted_endpoint_pages_nobody(caplog):
    """The breaker STILL trips for the plain endpoint (it is what keeps a
    failover target from arming into nothing) but the eviction is not a page:
    no CRITICAL log line, no `endpoint_paused` alert."""
    svc = _svc()
    _set_intent(svc, **{"t-od": "evicted", "t-plain": "evicted"})

    async def down(ep_cfg):
        return False
    svc._backend.probe_health = down
    with caplog.at_level(logging.DEBUG):
        for _ in range(4):
            await svc._poller_iteration()
    assert svc._state.endpoint_health[PLAIN]["healthy"] is False    # still gates admission
    assert not svc._health.endpoint_healthy(PLAIN)
    assert not [r for r in caplog.records
                if r.levelno >= logging.CRITICAL and "UNHEALTHY" in r.getMessage()
                and PLAIN in r.getMessage()]
    assert any(PLAIN in r.getMessage() and "expected" in r.getMessage()
               for r in caplog.records if r.levelno == logging.WARNING)
    svc._health.evaluate_alerts(time.monotonic())
    # (the config's own default endpoints are dead too, and rightly do page)
    paged = [a for a in svc._state.alerts if a["name"] == "endpoint_paused"
             and (PLAIN in a["detail"] or OD in a["detail"])]
    assert not paged, paged


@pytest.mark.asyncio
async def test_a_genuine_crash_still_pages(caplog):
    """Guard the guard: the same setup with the dispatcher saying `resident`
    (or nothing) must still be CRITICAL + `endpoint_paused`."""
    svc = _svc()
    _set_intent(svc, **{"t-plain": "resident"})

    async def down(ep_cfg):
        return False
    svc._backend.probe_health = down
    with caplog.at_level(logging.DEBUG):
        for _ in range(4):
            await svc._poller_iteration()
    assert any(r.levelno == logging.CRITICAL and PLAIN in r.getMessage()
               for r in caplog.records)
    svc._health.evaluate_alerts(time.monotonic())
    assert [a for a in svc._state.alerts
            if a["name"] == "endpoint_paused" and PLAIN in a["detail"]]


@pytest.mark.asyncio
async def test_status_does_not_change_admission_or_wake_anything():
    """Reporting only. An evicted on-demand endpoint still admits (nothing here
    refuses or loads), and the unmanaged endpoint has no lease machinery."""
    svc = _svc()
    _set_intent(svc, **{"t-od": "evicted"})
    assert svc._health.endpoint_healthy(OD) is True
    assert not svc._state.on_demand.manages(OD)


@pytest.mark.asyncio
async def test_the_admin_endpoint_view_carries_the_same_state():
    svc = _svc()
    _set_intent(svc, **{"t-od": "evicted"})

    async def down(ep_cfg):
        return False
    svc._backend.probe_health = down
    await svc._poller_iteration()
    view = svc._management._endpoint_view(OD, model_catalog.load_catalog())
    assert view["health"]["state"] == "unloaded"
    assert view["health"]["healthy"] is False
    assert view["health"]["residency"] == "evicted"
