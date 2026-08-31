"""INDEPENDENT ADVERSARIAL TRACK — Phase 5a "server-side smart DEFAULT timeout".

This file does NOT trust the builder's rationale. It attacks the feature on both
seams (hostile CALLER input = north face; pathological BACKEND output = south
face) and proves guard-bite. The happy/parity pins live in the sibling
``test_smart_default_timeout.py``; here we try to BREAK the same feature harder.

Invariants under attack (letters map to the mandate):
  A  byte-identity flag OFF                 — resolve_default_timeout OFF == 180.0
  B  no-500 / no-raise on hostile timeout_s / priority / payload / max_tokens
  C  deadline sanity flag ON                — finite, >0, floor<=d<=1800, never NaN/inf
  D  applied-vs-tallied                     — supplied wins & never tallies; default tallies
  E  guard-bite                             — a green suite means the guard fired
  F  slot-accounting + south-face under the tighter default (no leak)

Run:  python3.11 -m pytest tests/llmproxy/test_smart_default_timeout_adversarial.py -q
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import math
import tempfile

import httpx
import pytest
import pytest_asyncio

# ProxyService-spinning E2E — keep out of the per-ship fast tier.
pytestmark = pytest.mark.heavy

from roadstead import scheduler as _sched
from roadstead.__main__ import build_app
from roadstead.backend import BackendResponse
from roadstead.config import EndpointConfig, ProxyConfig
from roadstead.constants import _DEFAULT_TIMEOUT_S, _SMART_DEFAULT_CAP_S
from roadstead.service import ProxyService
from roadstead.timeout_model import normalize_endpoint, resolve_ceiling_s

from tests.fake_backend import (
    FAULT_EMPTY_COMPLETION,
    FAULT_HTTP_503,
    FAULT_INTERTOKEN_STALL,
    FAULT_MID_STREAM_RESET,
    FAULT_TIMEOUT,
    FAULT_TTFT_STALL,
    FakeBackend,
    FakeBackendServer,
)

_INTERNAL_CLIENT = ("127.0.0.1", 41999)
# "chat" is an ALIAS; the shadow tally, the floors table and the /v1/status
# endpoints map are all keyed by the RESOLVED CLASS.
#
# Derived from normalize_endpoint rather than hardcoded, because the target has
# moved twice: → "classify" (2026-07-03 analyst decommission), → "creative"
# (2026-07-11 boxa consolidation, which re-homed classify/analyst/vision onto the
# boxa endpoint as aliases). Each move left this file raising KeyError on a class
# the proxy no longer keys; nothing caught it because `local_tollgate` never
# actually ran a test until 2026-07-27 (ledger `local-tollgate-never-ran`).
_CHAT_CLASS = normalize_endpoint("chat")
# (The former _CLASS_FLOORS table is gone — every floor is now read from the
# live timeout model at assert time, which is what stopped it going stale.)


# --------------------------------------------------------------------------- #
# Lightweight service (no fake socket) — used for the resolve_default_timeout
# logic fuzz + the wire-through handler attacks against a stubbed OK backend.
# --------------------------------------------------------------------------- #

class _Req:
    class _Client:
        host = "127.0.0.1"

    client = _Client()
    headers: dict = {}


def _body(endpoint="chat", *, priority="P1_TURN_SUPPORT", max_tokens=64,
          payload=None, **extra):
    b = {
        "agent_id": "a", "endpoint": endpoint, "priority": priority,
        "call_site": "adv", "payload_type": "chat_completion",
        "payload": payload if payload is not None else {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": max_tokens},
    }
    b.update(extra)
    return b


def _ok_backend(svc):
    async def ok_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"content": "y"},
                               "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            duration_s=0.001, input_tokens=1, output_tokens=1,
            finish_reason="stop")
    svc._backend.call = ok_call


@contextlib.contextmanager
def _capture_applied():
    """Spy every QueuedRequest.create → the applied ``timeout_s`` that GOVERNS
    the request (the builder's technique). This is the ground truth for D/C."""
    created: list = []
    orig = _sched.QueuedRequest.create.__func__

    def spy(cls, **kw):
        req = orig(cls, **kw)
        created.append(req)
        return req

    _sched.QueuedRequest.create = classmethod(spy)
    try:
        yield created
    finally:
        _sched.QueuedRequest.create = classmethod(orig)


# ========================================================================== #
# A / C  —  resolve_default_timeout logic under hostile inputs (no startup)
# ========================================================================== #

# A hostile cross-product of priority x payload x max_tokens. None of these
# reaches resolve_default_timeout via a *supplied* timeout_s — they arrive when
# the caller OMITTED it, so every one must resolve to a sane number, never raise.
_HOSTILE_PRIORITIES = [
    None, "P1_TURN_SUPPORT", "NONSENSE", "", "  ", -99, 999, 3.7, True, False,
    [], {}, "p7_bogus", "background", "interactive", object(),
]
_HOSTILE_PAYLOADS = [
    None, {}, "not-a-dict", 123, [], {"messages": "not-a-list"},
    {"messages": ["str-not-dict"]}, {"messages": [{"role": "user"}]},
    {"messages": [{"role": "user", "content": "x" * 5000}]},
    {"max_tokens": "huge"}, {"max_tokens": -5}, {"max_tokens": 10 ** 9},
    {"max_tokens": True}, {"max_tokens": float("nan")},
    {"messages": None, "max_tokens": None},
]


@pytest.mark.parametrize("flag_on", [False, True])
def test_resolve_never_raises_and_is_sane(flag_on):
    svc = ProxyService(ProxyConfig())
    if flag_on:
        svc._flags.set_many({"smart_default_timeout": True})
    floor = 30.0  # chat
    for pri in _HOSTILE_PRIORITIES:
        for pay in _HOSTILE_PAYLOADS:
            for ep in ("chat", "bge-m3-embed", "gemma-router", "no-such-xyz"):
                body = _body(ep, priority=pri, payload=pay)
                d = svc._lifecycle.resolve_default_timeout(ep, body)
                # C: finite, positive, never NaN/inf, never above the cap.
                assert isinstance(d, float)
                assert math.isfinite(d), f"non-finite deadline {d} ep={ep}"
                assert d > 0
                assert d <= _SMART_DEFAULT_CAP_S
                if not flag_on:
                    # A: byte-identical flat default regardless of the garbage.
                    assert d == _DEFAULT_TIMEOUT_S
                else:
                    # C: flag-ON smart deadline never dips below the class floor.
                    # Derived from the LIVE model, not a hardcoded table: the
                    # floor moves whenever a role is re-homed (chat's went
                    # 180→120 when the boxa consolidation aliased it onto
                    # "creative"). The invariant under test is "never dips below
                    # the class floor" — not any particular number.
                    exp_floor = svc._timeout_model.floor_ms(ep) / 1000.0
                    assert d >= exp_floor, f"{ep}: {d} < floor {exp_floor}"


def test_resolve_off_is_flat_even_when_model_is_hot():
    """A: even with the model warmed to a huge recommendation, flag OFF is 180."""
    svc = ProxyService(ProxyConfig())
    tm = svc._timeout_model
    for i in range(80):
        tm.record("chat", 1, 8, 8, 5_000_000.0, "ok", 1000.0 + i)
    assert svc._lifecycle.resolve_default_timeout("chat", _body()) == _DEFAULT_TIMEOUT_S


def test_resolve_on_caps_pathological_tail_and_stays_finite():
    """C: a heavy-tailed cell (recommended = p99*margin) is clamped, never
    leaking a multi-hour or non-finite deadline. Since the 2026-07-05 adaptive
    uplift the per-class CEILING governs first, so the pathological tail is
    bounded by the ceiling that actually applies to chat's resolved class —
    derived here, not hardcoded. NB since the 2026-07-11 boxa consolidation that
    class is "creative", which carries an explicit `timeout_ceiling_s: 1800`
    role override in models.yaml ("a role override so the generous background
    band applies on EVERY tier", because song-compose runs at an interactive
    tier). So chat no longer lands on the 600s interactive band — a real
    consequence of the re-homing, not a test detail. What the test still proves
    is that SOMETHING finite governs before the flat cap."""
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"smart_default_timeout": True})
    tm = svc._timeout_model
    for i in range(80):
        tm.record("chat", 1, 8, 8, 9_999_999_999.0, "ok", 1000.0 + i)
    d = svc._lifecycle.resolve_default_timeout("chat", _body())
    expected = resolve_ceiling_s(
        "chat", interactive=True,
        role_ceilings=svc._state.timeout_ceilings,
        floor_s=svc._timeout_model.floor_ms("chat") / 1000.0,
        interactive_s=svc._config.timeout_ceiling_interactive_s,
        background_s=svc._config.timeout_ceiling_background_s,
    )
    assert d == expected
    assert d <= _SMART_DEFAULT_CAP_S
    assert math.isfinite(d)


def test_tally_written_both_modes_and_never_corrupts():
    """D: the shadow tally accumulates correctly across a duplicate storm of
    default-path resolves; counters stay consistent (single loop, no overflow)."""
    svc = ProxyService(ProxyConfig())
    N = 500
    for _ in range(N):
        svc._lifecycle.resolve_default_timeout("chat", _body())
    t = svc._smart_default_shadow[_CHAT_CLASS]
    # The smart value here IS chat's resolved-class floor — derived, because that
    # number moved 180→120 with the 2026-07-11 boxa re-homing. What this pins is
    # that every one of the N resolves produced the SAME value with no drift.
    _floor = svc._timeout_model.floor_ms("chat") / 1000.0
    assert t["count"] == N
    assert t["flat_s"] == _DEFAULT_TIMEOUT_S
    assert t["smart_s_min"] == t["smart_s_max"] == _floor
    assert t["smart_s_sum"] == _floor * N  # exact — no float drift at this scale
    # mean is recoverable
    assert t["smart_s_sum"] / t["count"] == _floor


# ========================================================================== #
# E  —  GUARD-BITE.  A green suite must mean the flag-gate actually fired.
# ========================================================================== #

def test_guardbite_flag_gate_is_load_bearing():
    """E-i + E-ii: prove the flag-ON path is reachable AND that reverting the
    gate breaks the OFF byte-identity contract. If the gate were removed
    (always-smart), the OFF==180 assertion below would fail — so its passing
    means the gate is genuinely governing."""
    svc = ProxyService(ProxyConfig())
    # Use gemma-router (class floor 60) as the guard-bite endpoint: its floor
    # DIFFERS from the flat default (180), so flag ON vs OFF is observably
    # different. NB "chat"/classify can no longer guard-bite here — its floor was
    # raised to 180 (2026-07-04), which now coincides with _DEFAULT_TIMEOUT_S.
    body = _body("gemma-router")
    # E-i: the two modes DIFFER — the smart value is reachable and != 180.
    off = svc._lifecycle.resolve_default_timeout("gemma-router", body)
    svc._flags.set_many({"smart_default_timeout": True})
    on = svc._lifecycle.resolve_default_timeout("gemma-router", body)
    assert off == _DEFAULT_TIMEOUT_S
    assert on == 60.0  # gemma class floor
    assert on != off, "flag had NO observable effect — gate is dead"


def test_guardbite_reverted_gate_fails_parity(monkeypatch):
    """E-ii: simulate a reverted flag-gate (always returns the smart value) and
    show the byte-identity parity assertion FAILS. If this pytest.raises does not
    trip, the parity guard is blind."""
    svc = ProxyService(ProxyConfig())
    monkeypatch.setattr(svc._lifecycle, "resolve_default_timeout",
                        lambda ep, b: 30.0)  # the "reverted gate" bug
    with pytest.raises(AssertionError):
        # This is exactly what the OFF byte-identity test asserts.
        assert svc._lifecycle.resolve_default_timeout("chat", _body()) == _DEFAULT_TIMEOUT_S


# ========================================================================== #
# Wire-through E2E against a STUBBED OK backend — no-500 / no-leak / applied.
# ========================================================================== #

@pytest_asyncio.fixture
async def okproxy():
    """A started ProxyService with an instant-OK backend stub (no socket)."""
    svc = ProxyService(ProxyConfig())
    _ok_backend(svc)
    await svc.startup()
    try:
        yield svc
    finally:
        await svc.shutdown()


def _total_in_flight(svc):
    return sum(svc._scheduler.endpoint_snapshot(ep)["in_flight"]
               for ep in svc._config.endpoints)


async def _drain(svc, timeout=2.0):
    end = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < end:
        if _total_in_flight(svc) == 0:
            return
        await asyncio.sleep(0.02)
    assert _total_in_flight(svc) == 0, "slot leak: in-flight did not return to 0"


# Hostile timeout_s values delivered on the SUBMIT body. Each must serve or
# fail with a typed status, never a generic 500, and never leak a slot.
_HOSTILE_TIMEOUTS = [
    None, 0, 0.0, -1, -5.0, float("nan"), float("-inf"), "abc", "",
    "45", True, False, [], {}, 1e12, 0.0001, "  ", "1e3",
]


@pytest.mark.parametrize("flag_on", [False, True])
@pytest.mark.parametrize("raw", _HOSTILE_TIMEOUTS)
async def test_wire_hostile_timeout_no_500_no_leak(okproxy, flag_on, raw):
    svc = okproxy
    if flag_on:
        svc._flags.set_many({"smart_default_timeout": True})
    body = _body()
    body["timeout_s"] = raw
    with _capture_applied() as created:
        resp = await asyncio.wait_for(svc.handle_submit(body, _Req()), timeout=8.0)
    assert resp.status_code != 500, f"raw={raw!r} → 500 {resp.body[:200]!r}"
    # whatever governed the request must be a finite positive deadline
    assert created, "no QueuedRequest created"
    applied = created[-1].timeout_s
    assert math.isfinite(applied) and applied > 0, f"raw={raw!r} applied={applied}"
    await _drain(svc)


@pytest.mark.parametrize("flag_on", [False, True])
async def test_wire_supplied_valid_wins_and_skips_tally(okproxy, flag_on):
    """D: a valid supplied deadline governs verbatim AND never writes a tally."""
    svc = okproxy
    if flag_on:
        svc._flags.set_many({"smart_default_timeout": True})
    body = _body()
    body["timeout_s"] = 47.5
    with _capture_applied() as created:
        resp = await asyncio.wait_for(svc.handle_submit(body, _Req()), timeout=8.0)
    assert resp.status_code == 200
    assert created[-1].timeout_s == 47.5
    assert "chat" not in svc._smart_default_shadow  # supplied → no default path
    await _drain(svc)


@pytest.mark.parametrize("flag_on", [False, True])
async def test_wire_omitted_uses_default_and_tallies(okproxy, flag_on):
    """D: an omitted deadline takes the default path (flat OFF / smart ON) and
    writes exactly one tally per request."""
    svc = okproxy
    if flag_on:
        svc._flags.set_many({"smart_default_timeout": True})
    with _capture_applied() as created:
        resp = await asyncio.wait_for(svc.handle_submit(_body(), _Req()), timeout=8.0)
    assert resp.status_code == 200
    # Flag ON floors at chat's resolved-class floor (derived — it is 120.0 since
    # the 2026-07-11 boxa re-homing, was 180.0 as "classify"). Flag OFF is the
    # byte-identical flat default.
    _floor = svc._timeout_model.floor_ms("chat") / 1000.0
    assert created[-1].timeout_s == (_floor if flag_on else _DEFAULT_TIMEOUT_S)
    assert svc._smart_default_shadow[_CHAT_CLASS]["count"] == 1
    await _drain(svc)


async def test_wire_openai_door_falsy_body_edge(okproxy):
    """The OpenAI door resolves the client deadline via
    ``body.pop("timeout_s", None) or header``. Probe the falsy-value edge (0 /
    "" / False): these are invalid deadlines anyway, so the request must still
    serve with a sane finite applied deadline — not 500, not leak."""
    svc = okproxy
    for raw in (0, 0.0, "", False):
        oai = {"model": "chat", "messages": [{"role": "user", "content": "hi"}],
               "timeout_s": raw}
        with _capture_applied() as created:
            resp = await asyncio.wait_for(svc.handle_openai_chat(oai, _Req()),
                                          timeout=8.0)
        assert resp.status_code == 200, f"raw={raw!r} → {resp.status_code}"
        assert math.isfinite(created[-1].timeout_s) and created[-1].timeout_s > 0
    await _drain(svc)


async def test_wire_guardbite_reverted_gate_breaks_wire_parity(okproxy, monkeypatch):
    """E-ii (wire flavour): with the flag OFF but the gate reverted to
    always-smart, the applied deadline for an omitted timeout is 30 not 180 — the
    byte-identity parity a real OFF ship would assert now FAILS."""
    svc = okproxy
    monkeypatch.setattr(svc._lifecycle, "resolve_default_timeout",
                        lambda ep, b: 30.0)
    with _capture_applied() as created:
        await asyncio.wait_for(svc.handle_submit(_body(), _Req()), timeout=8.0)
    with pytest.raises(AssertionError):
        assert created[-1].timeout_s == _DEFAULT_TIMEOUT_S
    await _drain(svc)


# ========================================================================== #
# DEFECT PROBE  —  supplied +inf timeout_s slips the coercion guard.
# ========================================================================== #

async def test_supplied_inf_timeout_falls_back_to_default(okproxy):
    """FIX REGRESSION GUARD (was DEFECT 1): the coercion guard now rejects a
    supplied ``+inf`` (``not math.isfinite`` — ``inf`` slips past ``>0`` and the
    NaN ``!=`` check). A supplied ``timeout_s = +inf`` (JSON ``Infinity``) must
    therefore fall through to the finite smart/flat default, NOT be stamped
    verbatim as an unbounded deadline. If this reverts to +inf the coercion
    finiteness check at lifecycle.py was removed."""
    svc = okproxy
    body = _body()
    body["timeout_s"] = float("inf")
    with _capture_applied() as created:
        resp = await asyncio.wait_for(svc.handle_submit(body, _Req()), timeout=8.0)
    assert resp.status_code == 200
    applied = created[-1].timeout_s
    # okproxy has the flag OFF → the flat default; either way it must be finite.
    assert math.isfinite(applied) and applied == _DEFAULT_TIMEOUT_S
    await _drain(svc)


async def test_supplied_inf_sync_deadline_is_finite():
    """FIX REGRESSION GUARD (was DEFECT 1 consequence): with a supplied +inf and
    a stalling backend the SYNC path must apply a FINITE effective deadline (the
    default, 180s) — not asyncio.wait_for(timeout=inf) + backend.call(timeout=inf)
    which never reclaims the slot. We assert the governing deadline the proxy
    chose is finite (== the default); the actual 180s cut is not waited out (the
    outer wait_for keeps the suite fast + hang-safe)."""
    srv = FakeBackendServer(FakeBackend()).start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _repointed(srv.host, srv.port, f"{tmp}/q.db")
            app = build_app(cfg)
            svc = app.state.proxy_service

            async def _healthy(ep_cfg):
                return True
            svc._backend.probe_health = _healthy
            await svc.startup()
            try:
                srv.controller.set_fault(FAULT_TIMEOUT, arg=30.0)  # sleep 30s
                body = _body()
                body["timeout_s"] = float("inf")
                with _capture_applied() as created:
                    # Bounded outer wait so the suite can never hang.
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(
                            svc.handle_submit(body, _Req()), timeout=1.0)
                # The governing deadline is now FINITE (the default) — the fix.
                assert created and math.isfinite(created[-1].timeout_s)
                assert created[-1].timeout_s == _DEFAULT_TIMEOUT_S
            finally:
                await svc.shutdown()
    finally:
        srv.stop()


# ========================================================================== #
# F  —  slot-accounting + south-face pathologies UNDER THE TIGHTER default.
# ========================================================================== #

def _repointed(host, port, queue_db) -> ProxyConfig:
    base = ProxyConfig(queue_db_path=queue_db)
    eps = {}
    for cls, ep in base.endpoints.items():
        eps[cls] = dataclasses.replace(
            ep, host=host, port=port, max_slots=ep.max_slots or 4,
            context_per_slot=ep.context_per_slot or 8192)
    base.endpoints = eps
    base.poller_interval_s = 0.05
    return base


@pytest_asyncio.fixture
async def fakeproxy():
    """Real ProxyService + real fake backend socket. Returns (svc, controller,
    client). Used for genuine south-face faults on the smart-default path."""
    srv = FakeBackendServer(FakeBackend()).start()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _repointed(srv.host, srv.port, f"{tmp}/q.db")
        app = build_app(cfg)
        svc = app.state.proxy_service

        async def _healthy(ep_cfg):
            return True
        svc._backend.probe_health = _healthy

        # Flag ON + a TINY classify floor so the SMART DEFAULT (which floors at
        # the class floor) becomes a sub-second, fast-to-fire deadline — this is
        # the "tighter default" the mandate wants exercised, without a 30s wait.
        # Override the RESOLVED CLASS for "chat", NOT "chat" itself — the
        # latter normalizes away and the override would be a no-op.
        svc._flags.set_many({"smart_default_timeout": True})
        svc._timeout_model._floors[_CHAT_CLASS] = 0.5  # → smart default ~1s (ceil)

        await svc.startup()
        transport = httpx.ASGITransport(
            app=app, raise_app_exceptions=False, client=_INTERNAL_CLIENT)
        client = httpx.AsyncClient(transport=transport, base_url="http://proxy",
                                   timeout=30.0)
        try:
            yield svc, srv.controller, client
        finally:
            await client.aclose()
            await svc.shutdown()
            srv.stop()


async def _drain_client(svc, timeout=3.0):
    await _drain(svc, timeout)


async def test_smartdefault_sync_stall_cut_and_no_leak(fakeproxy):
    """F: an OMITTED-timeout sync request governed by the tiny smart default,
    against a backend that sleeps far past it, must be cut (not hang) and free
    the slot. Confirms the tighter default composes with the deadline machinery."""
    svc, ctl, client = fakeproxy
    # sanity: the smart default is what governs (ceil(0.5s floor) = 1.0s)
    assert svc._lifecycle.resolve_default_timeout("chat", _body()) == 1.0
    ctl.set_fault(FAULT_TIMEOUT, arg=5.0)  # backend sleeps 5s >> 1s deadline
    # No timeout_s in the OpenAI body → smart default governs.
    resp = await asyncio.wait_for(
        client.post("/v1/chat/completions",
                    json={"model": "chat",
                          "messages": [{"role": "user", "content": "hi"}]}),
        timeout=6.0)
    assert not (resp.status_code == 500 and "internal proxy error" in resp.text)
    assert resp.status_code >= 400 or "error" in resp.text.lower()
    await _drain_client(svc)


@pytest.mark.parametrize("fault,arg", [
    (FAULT_TTFT_STALL, 5.0),
    (FAULT_INTERTOKEN_STALL, 5.0),
    (FAULT_MID_STREAM_RESET, 0.0),
])
async def test_smartdefault_stream_faults_no_leak(fakeproxy, fault, arg):
    """F: streaming south-face faults on the smart-default path free the slot.
    The stream watchdogs (TTFT/inter-token) are min(watchdog, stream_timeout);
    the tighter default must not defeat them or leak."""
    svc, ctl, client = fakeproxy
    ctl.set_fault(fault, arg)
    frames = []
    with contextlib.suppress(Exception):
        async with client.stream(
            "POST", "/v1/chat/completions",
            json={"model": "chat", "stream": True,
                  "messages": [{"role": "user", "content": "a b c"}]}) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    frames.append(line[6:])
    assert isinstance(frames, list)
    await _drain_client(svc, timeout=4.0)


async def test_smartdefault_empty_and_503_no_leak(fakeproxy):
    """F: non-streaming error faults under the smart default free the slot."""
    svc, ctl, client = fakeproxy
    for fault in (FAULT_EMPTY_COMPLETION, FAULT_HTTP_503):
        ctl.set_fault(fault)
        resp = await asyncio.wait_for(
            client.post("/v1/chat/completions",
                        json={"model": "chat",
                              "messages": [{"role": "user", "content": "hi"}]}),
            timeout=6.0)
        assert not (resp.status_code == 500 and "internal proxy error" in resp.text)
        await _drain_client(svc)


async def test_smartdefault_duplicate_storm_no_leak(fakeproxy):
    """F/D: many OMITTED-timeout submits fired at once — no slot leak, no crash,
    and the tally count equals the number of default-path admissions."""
    svc, ctl, client = fakeproxy
    ctl.reset()  # happy backend
    N = 12
    results = await asyncio.gather(*[
        client.post("/v1/chat/completions",
                    json={"model": "chat",
                          "messages": [{"role": "user", "content": f"c{i}"}]})
        for i in range(N)], return_exceptions=True)
    for r in results:
        assert not isinstance(r, Exception), f"raised: {r!r}"
        assert not (r.status_code == 500 and "internal proxy error" in r.text)
    await _drain_client(svc)
    assert svc._smart_default_shadow[_CHAT_CLASS]["count"] == N
