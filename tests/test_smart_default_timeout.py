"""Phase 5a — server-side smart DEFAULT deadline for callers that OMIT timeout_s.

The proxy already applies the timeout model client-side (extend-only, via
``framework.timeout_advice.apply_extend_only``), but two server paths still bake
a flat 180s when the caller supplies no ``timeout_s``: the OpenAI
``/v1/chat/completions`` door and a bare ``/v1/submit``. This closes that gap.

Ships shadow-first behind the ``smart_default_timeout`` runtime flag:
  - OFF (default): the flat ``_DEFAULT_TIMEOUT_S`` (byte-identical), while
    still recording a shadow tally of what a data-driven default WOULD be.
  - ON: the timeout model's class-floored, capped recommendation for the
    ``(endpoint, tier, size)`` (cold model → the per-class floor).

A caller-SUPPLIED ``timeout_s`` ALWAYS wins regardless of the flag. These are
the primary/parity pins; the hostile both-seam matrix lives in the adversarial
track (test_smart_default_timeout_adversarial.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import pytest

from roadstead import scheduler as _sched
from roadstead.backend import BackendResponse
from roadstead.config import ProxyConfig
from roadstead.constants import _DEFAULT_TIMEOUT_S
from roadstead.service import ProxyService


class _Req:
    class _Client:
        host = "127.0.0.1"

    client = _Client()
    headers: dict = {}


def _submit_body(endpoint: str = "chat", *, timeout_s=None, priority="P1_TURN_SUPPORT",
                 max_tokens: int = 100):
    body = {
        "agent_id": "a", "endpoint": endpoint, "priority": priority,
        "call_site": "t", "payload_type": "chat_completion",
        "payload": {"messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": max_tokens},
    }
    if timeout_s is not None:
        body["timeout_s"] = timeout_s
    return body


def _ok_backend(svc):
    async def ok_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"content": "y"}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            duration_s=0.01, input_tokens=1, output_tokens=1, finish_reason="stop")
    svc._backend.call = ok_call


@contextlib.contextmanager
def _capture_applied_timeout():
    """Spy the applied deadline: wrap ``QueuedRequest.create`` (the single point
    that stamps ``timeout_s`` onto the request) and record every created req."""
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


# --------------------------------------------------------------------------
# Core logic — resolve_default_timeout (no startup; deterministic + fast).
# --------------------------------------------------------------------------

def test_shadow_off_returns_flat_default_and_tallies():
    svc = ProxyService(ProxyConfig())
    # Flag OFF (default): flat 180s applied, byte-identical to history.
    # "chat" resolves to the "tier2" endpoint class since the 2026-08-19
    # tier2 split (it was "tier2" on the boxa before) — tallies + floors key
    # off the RESOLVED class, which is exactly why this assertion had to move
    # when the alias did.
    assert svc._lifecycle.resolve_default_timeout("chat", _submit_body()) == _DEFAULT_TIMEOUT_S
    tally = svc._smart_default_shadow["tier2"]
    assert tally["count"] == 1
    assert tally["flat_s"] == _DEFAULT_TIMEOUT_S
    # Cold model (no samples) → the "tier2" per-class floor (120s, held
    # equal to tier2's across the split so the cutover moved aliases and not
    # numbers) is what the smart default WOULD be — recorded even though the
    # flat value is applied.
    assert tally["smart_s_min"] == tally["smart_s_max"] == 120.0
    assert tally["smart_s_sum"] == 120.0


def test_enforce_uses_class_floor_advice_cold():
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"smart_default_timeout": True})
    # chat → tier2 (not on-demand) → the flag's effect is visible: the
    # tier2 floor is 120s, lower than the flat _DEFAULT_TIMEOUT_S (180s).
    assert svc._lifecycle.resolve_default_timeout("chat", _submit_body()) == 120.0
    # embed floor is 15s; tier1 60s — per-class, not a blanket number.
    assert svc._lifecycle.resolve_default_timeout("embed", _submit_body("embed")) == 15.0
    assert svc._lifecycle.resolve_default_timeout("tier1", _submit_body("tier1")) == 60.0


def test_enforce_honors_warm_recommendation_and_cap():
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"smart_default_timeout": True})
    tm = svc._timeout_model
    now = 1000.0
    # Seed the P1 tier2 cell with enough fast samples that p99*margin < floor
    # → the floor still governs (recommendation never drops below the class floor).
    # 🚨 SEED AND READ MUST RESOLVE TO THE SAME CLASS. Before the 2026-08-19 split
    # this seeded "classify" and read "chat" — both normalized to "tier2". They
    # do not any more: "classify" stayed on the analyst and "chat" moved to jetty,
    # so seeding "classify" would now fill a cell this call never reads and the
    # test would silently degrade into the cold-model case it already covers
    # above. "tier2" resolves to tier2, same cell as "chat".
    for i in range(60):
        tm.record("tier2", 1, 8, 8, 500.0, "ok", now + i)
    assert svc._lifecycle.resolve_default_timeout("chat", _submit_body()) == 120.0
    # Now seed a huge-latency cell so p99*margin >> floor, and the CEILING
    # governs the tail. 🔑 THE TWO LANES ANSWER DIFFERENTLY, AND THAT IS THE
    # POINT: `tier2` is the conversational lane and deliberately carries NO
    # `timeout_ceiling_s`, so a P1 turn there gets the INTERACTIVE band, 600s.
    # That is the "no 1,800-second jobs on the chat lane" property, enforced at
    # the timeout layer rather than merely asserted in a plan.
    for i in range(60):
        tm.record("tier2", 1, 8, 8, 5_000_000.0, "ok", now + 100 + i)
    got = svc._lifecycle.resolve_default_timeout("chat", _submit_body())
    assert got == 600.0, "tier2 has no class ceiling → the interactive band"
    # …and the long-form tier DOES carry the override, so the two lanes really
    # do differ rather than both having quietly fallen back to a band.
    for i in range(60):
        tm.record("composer", 1, 8, 8, 5_000_000.0, "ok", now + 200 + i)
    assert svc._lifecycle.resolve_default_timeout(
        "composer", _submit_body("composer")) == 1800.0


def test_tally_records_both_modes_and_accumulates():
    svc = ProxyService(ProxyConfig())
    for _ in range(3):
        svc._lifecycle.resolve_default_timeout("chat", _submit_body())
    svc._flags.set_many({"smart_default_timeout": True})
    svc._lifecycle.resolve_default_timeout("chat", _submit_body())
    tally = svc._smart_default_shadow["tier2"]
    assert tally["count"] == 4                # tallied whether flag on or off
    assert tally["smart_s_sum"] == 120.0 * 4  # mean = sum/count (tier2 floor 120)


def test_guarded_against_bad_priority_and_payload():
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"smart_default_timeout": True})
    # A garbage priority soft-defaults (never raises); a non-dict payload yields
    # est_in=0 and still resolves. Neither may raise or return a non-positive.
    bad = {"agent_id": "a", "endpoint": "chat", "priority": "NONSENSE",
           "call_site": "t", "payload_type": "chat_completion", "payload": "not-a-dict"}
    got = svc._lifecycle.resolve_default_timeout("chat", bad)
    assert got == 120.0  # tier2 floor still applies; no crash


# --------------------------------------------------------------------------
# Wire-through — bare /v1/submit + the OpenAI door apply the resolved default.
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bare_submit_omitted_timeout_applies_default():
    svc = ProxyService(ProxyConfig())
    _ok_backend(svc)
    await svc.startup()
    try:
        with _capture_applied_timeout() as created:
            # Flag OFF → flat 180 applied.
            r = await asyncio.wait_for(svc.handle_submit(_submit_body(), _Req()), timeout=10.0)
            assert r.status_code == 200
            assert created[-1].timeout_s == _DEFAULT_TIMEOUT_S
            # Flag ON → the smart default (tier2 floor 120; "chat" resolves
            # to the jetty tier2 endpoint since the 2026-08-19 split) applied.
            svc._flags.set_many({"smart_default_timeout": True})
            r = await asyncio.wait_for(svc.handle_submit(_submit_body(), _Req()), timeout=10.0)
            assert r.status_code == 200
            assert created[-1].timeout_s == 120.0
        status = json.loads((await svc.handle_status(_Req())).body)
        assert "tier2" in status["reliability"]["smart_default_shadow"]
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_supplied_timeout_wins_in_both_modes_and_skips_tally():
    svc = ProxyService(ProxyConfig())
    _ok_backend(svc)
    await svc.startup()
    try:
        with _capture_applied_timeout() as created:
            for enforce in (False, True):
                svc._flags.set_many({"smart_default_timeout": enforce})
                r = await asyncio.wait_for(
                    svc.handle_submit(_submit_body(timeout_s=45.0), _Req()), timeout=10.0)
                assert r.status_code == 200
                assert created[-1].timeout_s == 45.0  # caller override always wins
        # A supplied deadline never enters the default path → no shadow tally.
        assert "chat" not in svc._smart_default_shadow
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_openai_door_omits_timeout_and_inherits_default():
    svc = ProxyService(ProxyConfig())
    _ok_backend(svc)
    await svc.startup()
    try:
        with _capture_applied_timeout() as created:
            # No timeout_s in the OpenAI body → door omits it → handle_submit
            # default. OFF → 180.
            body = {"model": "tier2",
                    "messages": [{"role": "user", "content": "hi"}]}
            r = await asyncio.wait_for(svc.handle_openai_chat(dict(body), _Req()), timeout=10.0)
            assert r.status_code == 200
            assert created[-1].timeout_s == _DEFAULT_TIMEOUT_S
            # ON → the tier2 smart default (120) — tier2 is a legacy
            # alias resolving to the boxa tier2 endpoint since the 2026-07-11
            # one-model consolidation.
            svc._flags.set_many({"smart_default_timeout": True})
            r = await asyncio.wait_for(svc.handle_openai_chat(dict(body), _Req()), timeout=10.0)
            assert r.status_code == 200
            assert created[-1].timeout_s == 120.0
            # A client-supplied timeout_s in the OpenAI body still wins.
            r = await asyncio.wait_for(
                svc.handle_openai_chat({**body, "timeout_s": 42}, _Req()), timeout=10.0)
            assert r.status_code == 200
            assert created[-1].timeout_s == 42.0
    finally:
        await svc.shutdown()
