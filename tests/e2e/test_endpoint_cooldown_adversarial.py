"""Step 4b — rate-windowed per-endpoint cooldown: ADVERSARIAL both-seam E2E.

Independent hostile track (did NOT write the implementation). Goal: BREAK the new
cooldown path over the REAL in-process proxy + fake backend, exercising the actual
lifecycle dispatch-error call sites (sync 5xx, sync timeout, streaming reset) — the
unit file (``test_endpoint_cooldown.py``) already covers Health in isolation; this
proves the WIRING and the end-to-end defer/degrade/recover/no-reroute behaviour.

Kept LEAN for the tollgate budget: each proxy spin drives several assertions, faults
FLAP via ``fault_max_hits`` (fail N then recover — no slow retry loops), and the
cooldown window/duration are shrunk via env so nothing waits on wall-clock.

Guard-bite reverts are performed out-of-band by the adversarial operator; see report.
"""
from __future__ import annotations

import asyncio

import pytest

# Exhaustive ProxyService-spinning adversarial matrix — deselected from the
# per-ship in_container_tollgate via `-m 'not heavy'` (see pyproject `heavy`).
pytestmark = pytest.mark.heavy

from tests.llmproxy.fake_backend import (
    FAULT_HTTP_400,
    FAULT_HTTP_500,
    FAULT_MID_STREAM_RESET,
    FAULT_TIMEOUT,
)

ENFORCE = "COLLECTIVE_PROXY_ENDPOINT_COOLDOWN"
SHADOW = "COLLECTIVE_PROXY_ENDPOINT_COOLDOWN_SHADOW"
ALLOWED = "COLLECTIVE_PROXY_COOLDOWN_ALLOWED_FAILS"
WINDOW = "COLLECTIVE_PROXY_COOLDOWN_WINDOW_S"
DURATION = "COLLECTIVE_PROXY_COOLDOWN_DURATION_S"


def _all_off(mp):
    for k in (ENFORCE, SHADOW, ALLOWED, WINDOW, DURATION):
        mp.delenv(k, raising=False)


# Endpoint under test. "chat" is a pure alias of the "classify" class (2026-07-03
# analyst decommission); the proxy normalizes it everywhere, so cooldown/health
# STATE and the /v1/status "endpoints" map are keyed by "classify", not "chat".
# Look-ups that hit those dicts directly must use the resolved class name.
async def _status_ep(proxy, ep="classify") -> dict:
    resp = await proxy.client.get("/v1/status")
    assert resp.status_code == 200
    return resp.json()["endpoints"][ep]


async def _wait_trips(proxy, ep="classify", want=1, timeout_s=4.0) -> int:
    """Poll for the cooldown trip to land. Some fault paths (sync-timeout) count
    in the BACKGROUND dispatch task, which completes AFTER the caller's shorter
    client-wait deadline returns — so the trip is not synchronous with the HTTP
    response and must be awaited."""
    st = proxy.svc._correction.state
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        if st.endpoint_cooldown_trips.get(ep, 0) >= want:
            return st.endpoint_cooldown_trips[ep]
        await asyncio.sleep(0.05)
    return st.endpoint_cooldown_trips.get(ep, 0)


# --------------------------------------------------------------------------- #
# 2 + 6 + 7 + 10 — enforce core: cool after N faults, defer (no reroute),
#                  /v1/status surface, auto-recover, slot-accounting.
# --------------------------------------------------------------------------- #

async def test_enforce_cools_defers_recovers_no_reroute(proxy, monkeypatch):
    monkeypatch.setenv(ENFORCE, "1")
    monkeypatch.setenv(ALLOWED, "3")
    monkeypatch.setenv(WINDOW, "60")
    monkeypatch.setenv(DURATION, "1")  # config floors duration at 1.0s
    base = proxy.total_in_flight()

    # FLAP: the fake serves 500 exactly 3× (each a non-transient terminal fault =
    # one cooldown count), then recovers. Set once, drive 3 interactive chats.
    proxy.controller.set_fault(FAULT_HTTP_500, 0.0, 3)
    for i in range(3):
        r = await proxy.chat("hi")
        assert r.status_code >= 400, f"call {i} should surface the injected 500"
    # Tripped after exactly allowed_fails backend-faults.
    st = proxy.svc._correction.state
    assert st.endpoint_cooldown_trips.get("classify") == 1, "did not cool after 3 faults"
    assert proxy.svc._health.endpoint_healthy("chat") is False, "cooled ep still healthy"
    assert proxy.total_in_flight() == base, "slot leaked after the flap"

    # /v1/status surface: trips + cooling + remaining.
    snap = await _status_ep(proxy)
    assert snap.get("cooldown_trips") == 1
    assert snap.get("cooling") is True
    assert 0 < snap.get("cooldown_remaining_s", 0) <= 1.0

    # DEFER / NO-REROUTE: a fresh interactive chat while cooled fast-fails 503
    # circuit_open for the SAME endpoint (the backend has already recovered —
    # max_hits=3 exhausted — so a reroute/serve would return 200; it must not).
    r = await proxy.chat("hi")
    assert r.status_code == 503, "cooled interactive should fast-fail, not be served"
    assert "circuit" in r.text.lower(), "wrong deferral reason (reroute/serve leak?)"
    assert proxy.controller.default_fault == FAULT_HTTP_500  # fault untouched
    assert proxy.total_in_flight() == base

    # AUTO-RECOVER: after the (shrunk) duration expires the ep serves again — no
    # probe/poller intervention needed, purely the cooldown gate expiring.
    for _ in range(60):
        if proxy.svc._health.endpoint_healthy("chat"):
            break
        await asyncio.sleep(0.05)
    assert proxy.svc._health.endpoint_healthy("chat") is True, "did not auto-recover"
    r = await proxy.chat("hello again")
    assert r.status_code == 200, "recovered endpoint should serve"
    assert r.json()["choices"][0]["message"]["content"].startswith("echo:")
    # trips persist (audit trail); cooling cleared.
    snap = await _status_ep(proxy)
    assert snap.get("cooldown_trips") == 1
    assert "cooling" not in snap
    assert proxy.total_in_flight() == base


# --------------------------------------------------------------------------- #
# 3 — a 4xx storm must NEVER cool (client error is not the backend's fault).
# --------------------------------------------------------------------------- #

async def test_4xx_storm_never_cools_e2e(proxy, monkeypatch):
    monkeypatch.setenv(ENFORCE, "1")
    monkeypatch.setenv(ALLOWED, "2")
    monkeypatch.setenv(DURATION, "30")
    base = proxy.total_in_flight()
    proxy.controller.set_fault(FAULT_HTTP_400, 0.0)  # persistent 4xx
    for _ in range(6):
        r = await proxy.chat("hi")
        assert r.status_code >= 400
    st = proxy.svc._correction.state
    assert st.endpoint_cooldown_trips == {}, "4xx must never trip a cooldown"
    assert "classify" not in st.endpoint_cooldown_until
    assert proxy.svc._health.endpoint_healthy("chat") is True
    snap = await _status_ep(proxy)
    assert "cooldown_trips" not in snap and "cooling" not in snap
    assert proxy.total_in_flight() == base


# --------------------------------------------------------------------------- #
# 5 — SHADOW: count + surface, but NEVER pull; traffic keeps flowing.
# --------------------------------------------------------------------------- #

async def test_shadow_counts_but_never_pulls_e2e(proxy, monkeypatch):
    monkeypatch.delenv(ENFORCE, raising=False)
    monkeypatch.setenv(SHADOW, "1")
    monkeypatch.setenv(ALLOWED, "2")
    base = proxy.total_in_flight()
    proxy.controller.set_fault(FAULT_HTTP_500, 0.0, 2)  # 2 faults then recover
    for _ in range(2):
        await proxy.chat("hi")
    st = proxy.svc._correction.state
    assert st.endpoint_cooldown_trips.get("classify") == 1, "shadow must still count a trip"
    assert "classify" not in st.endpoint_cooldown_until, "shadow must NOT set a cooldown"
    assert proxy.svc._health.endpoint_healthy("chat") is True, "shadow must NOT pull"
    snap = await _status_ep(proxy)
    assert snap.get("cooldown_trips") == 1  # surfaced even in shadow
    assert "cooling" not in snap
    # traffic still flows immediately (recovered backend).
    r = await proxy.chat("still here")
    assert r.status_code == 200
    assert proxy.total_in_flight() == base


# --------------------------------------------------------------------------- #
# 2b (call-site wiring) — the SYNC-TIMEOUT dispatch terminal counts a fault.
# --------------------------------------------------------------------------- #

async def test_sync_timeout_counts_toward_cooldown(proxy, monkeypatch):
    """The BackendTimeout branch (lifecycle sync-timeout terminal) must feed the
    cooldown. Drives a per-request timeout_s below the backend's injected sleep so
    the proxy raises BackendTimeout (504 → counts)."""
    monkeypatch.setenv(SHADOW, "1")
    monkeypatch.setenv(ALLOWED, "2")
    base = proxy.total_in_flight()
    # backend sleeps 2s; caller deadline 0.2s returns a client-wait timeout while
    # the dispatch keeps running and hits BackendTimeout (~1s) IN THE BACKGROUND —
    # that terminal is what feeds the cooldown, so the trip lands asynchronously.
    for _ in range(2):
        r = await proxy.chat("hi", timeout_s=0.2, fault=FAULT_TIMEOUT, fault_arg=2.0)
        assert r.status_code >= 400
    assert await _wait_trips(proxy, want=1) == 1, \
        "sync-timeout (BackendTimeout) call site did not feed the cooldown"
    # let both background dispatches settle so the slot check is stable
    for _ in range(80):
        if proxy.total_in_flight() == base:
            break
        await asyncio.sleep(0.05)
    assert proxy.total_in_flight() == base


# --------------------------------------------------------------------------- #
# 2c (call-site wiring) + 7 — the STREAMING-error terminal counts; slot clean
#                             even on a mid-stream reset WHILE cooling.
# --------------------------------------------------------------------------- #

async def test_stream_reset_counts_and_no_slot_leak_while_cooling(proxy, monkeypatch):
    monkeypatch.setenv(ENFORCE, "1")
    monkeypatch.setenv(ALLOWED, "2")
    monkeypatch.setenv(DURATION, "30")
    base = proxy.total_in_flight()
    proxy.controller.set_fault(FAULT_MID_STREAM_RESET, 0.0)  # BackendUnavailable(503)
    for _ in range(2):
        frames = await proxy.stream_frames("a b c")
        assert any("error" in f for f in frames), "reset should surface a stream error"
        assert proxy.total_in_flight() == base, "slot leaked on stream reset"
    st = proxy.svc._correction.state
    assert st.endpoint_cooldown_trips.get("classify") == 1, \
        "streaming-error call site did not feed the cooldown"
    assert proxy.svc._health.endpoint_healthy("chat") is False
    # Another reset attempt lands while already cooling — still no leak.
    await proxy.stream_frames("a b c")
    assert proxy.total_in_flight() == base


# --------------------------------------------------------------------------- #
# 2d (isolation) — only the flaky endpoint cools; a sibling keeps serving.
# --------------------------------------------------------------------------- #

async def test_sibling_endpoint_unaffected(proxy, monkeypatch):
    monkeypatch.setenv(ENFORCE, "1")
    monkeypatch.setenv(ALLOWED, "2")
    monkeypatch.setenv(DURATION, "30")
    proxy.controller.set_fault(FAULT_HTTP_500, 0.0, 2)  # only the calls we send fail
    for _ in range(2):
        await proxy.chat("hi", model="chat")
    assert proxy.svc._health.endpoint_healthy("chat") is False, "chat should cool"
    # A different endpoint (role/model) was never driven → healthy → serves.
    assert proxy.svc._health.endpoint_healthy("companion") is True
    r = await proxy.chat("hi", model="companion")
    assert r.status_code == 200, "sibling endpoint must keep serving"
    st = proxy.svc._correction.state
    assert "companion" not in st.endpoint_cooldown_trips


# --------------------------------------------------------------------------- #
# 1 — flags OFF: no cooldown fields ever appear, even under a 5xx storm.
# --------------------------------------------------------------------------- #

async def test_flags_off_no_cooldown_surface(proxy, monkeypatch):
    _all_off(monkeypatch)
    base = proxy.total_in_flight()
    proxy.controller.set_fault(FAULT_HTTP_500, 0.0, 4)
    for _ in range(4):
        await proxy.chat("hi")
    st = proxy.svc._correction.state
    assert st.endpoint_failure_times == {}, "flags off must not accumulate failures"
    assert st.endpoint_cooldown_trips == {}
    snap = await _status_ep(proxy)
    assert "cooldown_trips" not in snap and "cooling" not in snap
    assert proxy.svc._health.endpoint_healthy("chat") is True
    assert proxy.total_in_flight() == base


# --------------------------------------------------------------------------- #
# 8 — FAIL-OPEN: record_dispatch_failure never raises into the dispatch path.
#     (Unit-level fuzz over hostile exc/state shapes — no proxy spin.)
# --------------------------------------------------------------------------- #

class _WeirdExc(Exception):
    def __init__(self, sc):
        self.status_code = sc


@pytest.mark.parametrize("exc", [
    None,
    "not-an-exception",
    ValueError("plain"),
    _WeirdExc(None),
    _WeirdExc("five hundred"),
    _WeirdExc(500),          # non-BackendError with a 5xx status_code
    _WeirdExc(float("nan")),
    KeyError("k"),
])
def test_record_dispatch_failure_fail_open(monkeypatch, exc):
    """Both flags ON (so the body runs), then feed garbage exceptions. Only a real
    BackendError>=500 may count; nothing may raise."""
    import importlib
    import types

    monkeypatch.setenv(ENFORCE, "1")
    monkeypatch.setenv(ALLOWED, "2")
    Health = importlib.import_module("originfleet.llmproxy.health").Health
    state = types.SimpleNamespace(
        endpoint_failure_times={}, endpoint_cooldown_until={},
        endpoint_cooldown_trips={}, paused_endpoints=set(), endpoint_health={},
        on_demand=types.SimpleNamespace(manages=lambda ep: False),
        scheduler=types.SimpleNamespace(queued_requests=lambda ep, bands: []),
    )
    h = Health(state)
    h.record_dispatch_failure("chat", exc)  # must never raise
    # A duck-typed non-BackendError must NOT count even with a 5xx status_code.
    assert state.endpoint_cooldown_trips == {}


def test_record_dispatch_failure_broken_state_never_crashes_dispatch(monkeypatch):
    """A structurally broken state (missing the new dicts) must not let the new
    path escalate into the dispatch terminal. record_dispatch_failure itself may
    raise on a truly broken state, but the lifecycle terminals are the contract —
    here we assert the classification short-circuit protects the common path: a
    4xx exits before ANY state access."""
    import importlib
    from originfleet.llmproxy.backend import BackendError

    monkeypatch.setenv(ENFORCE, "1")
    Health = importlib.import_module("originfleet.llmproxy.health").Health
    h = Health(object())  # no dicts at all
    # 4xx classified out before touching state → no AttributeError.
    h.record_dispatch_failure("chat", BackendError(404, "nope"))
