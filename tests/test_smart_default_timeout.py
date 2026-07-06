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

from originfleet.llmproxy import scheduler as _sched
from originfleet.llmproxy.backend import BackendResponse
from originfleet.llmproxy.config import ProxyConfig
from originfleet.llmproxy.constants import _DEFAULT_TIMEOUT_S
from originfleet.llmproxy.service import ProxyService


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
    # "chat" fully resolves to "classify" (2026-07-03 analyst decommission
    # made it a pure alias) — tallies + floors key off the resolved class.
    assert svc._lifecycle.resolve_default_timeout("chat", _submit_body()) == _DEFAULT_TIMEOUT_S
    tally = svc._smart_default_shadow["classify"]
    assert tally["count"] == 1
    assert tally["flat_s"] == _DEFAULT_TIMEOUT_S
    # Cold model (no samples) → the "classify" per-class floor (180s, raised
    # 45→180 in the 2026-07-04 nexus loadout) is what the smart default WOULD
    # be — recorded even though the flat value is applied.
    assert tally["smart_s_min"] == tally["smart_s_max"] == 180.0
    assert tally["smart_s_sum"] == 180.0


def test_enforce_uses_class_floor_advice_cold():
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"smart_default_timeout": True})
    # chat/classify (not on-demand) → the flag's effect is visible: the classify
    # floor is 180s (raised 45→180, 2026-07-04) — coincidentally equal to the flat
    # _DEFAULT_TIMEOUT_S here, so assert against the floor constant directly below.
    assert svc._lifecycle.resolve_default_timeout("chat", _submit_body()) == 180.0
    # embed floor is 15s; gemma 60s — per-class, not a blanket number.
    assert svc._lifecycle.resolve_default_timeout("bge-m3-embed", _submit_body("bge-m3-embed")) == 15.0
    assert svc._lifecycle.resolve_default_timeout("gemma-router", _submit_body("gemma-router")) == 60.0


def test_enforce_honors_warm_recommendation_and_cap():
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"smart_default_timeout": True})
    tm = svc._timeout_model
    now = 1000.0
    # Seed the P1 classify cell with enough fast samples that p99*margin < floor →
    # the floor still governs (recommendation never drops below the class floor).
    for i in range(60):
        tm.record("classify", 1, 8, 8, 500.0, "ok", now + i)
    assert svc._lifecycle.resolve_default_timeout("chat", _submit_body()) == 180.0
    # Now seed a huge-latency cell so p99*margin >> floor. The per-class CEILING
    # now governs the tail (2026-07-05 adaptive-timeout uplift): classify is an
    # interactive-tier class, so the recommendation is bounded by the interactive
    # ceiling (600s) — which is tighter than, and reached before, the flat
    # _SMART_DEFAULT_CAP_S (1800s) fallback.
    for i in range(60):
        tm.record("classify", 1, 8, 8, 5_000_000.0, "ok", now + 100 + i)
    got = svc._lifecycle.resolve_default_timeout("chat", _submit_body())
    assert got == svc._config.timeout_ceiling_interactive_s  # 600s class ceiling


def test_tally_records_both_modes_and_accumulates():
    svc = ProxyService(ProxyConfig())
    for _ in range(3):
        svc._lifecycle.resolve_default_timeout("chat", _submit_body())
    svc._flags.set_many({"smart_default_timeout": True})
    svc._lifecycle.resolve_default_timeout("chat", _submit_body())
    tally = svc._smart_default_shadow["classify"]
    assert tally["count"] == 4                # tallied whether flag on or off
    assert tally["smart_s_sum"] == 180.0 * 4  # mean = sum/count (classify floor 180)


def test_guarded_against_bad_priority_and_payload():
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"smart_default_timeout": True})
    # A garbage priority soft-defaults (never raises); a non-dict payload yields
    # est_in=0 and still resolves. Neither may raise or return a non-positive.
    bad = {"agent_id": "a", "endpoint": "chat", "priority": "NONSENSE",
           "call_site": "t", "payload_type": "chat_completion", "payload": "not-a-dict"}
    got = svc._lifecycle.resolve_default_timeout("chat", bad)
    assert got == 180.0  # classify floor still applies; no crash


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
            # Flag ON → the smart default (classify floor 180, "chat" resolves
            # there since analyst's decommission made it a pure alias) applied.
            svc._flags.set_many({"smart_default_timeout": True})
            r = await asyncio.wait_for(svc.handle_submit(_submit_body(), _Req()), timeout=10.0)
            assert r.status_code == 200
            assert created[-1].timeout_s == 180.0
        status = json.loads((await svc.handle_status(_Req())).body)
        assert "classify" in status["reliability"]["smart_default_shadow"]
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
            body = {"model": "qwen-analyst",
                    "messages": [{"role": "user", "content": "hi"}]}
            r = await asyncio.wait_for(svc.handle_openai_chat(dict(body), _Req()), timeout=10.0)
            assert r.status_code == 200
            assert created[-1].timeout_s == _DEFAULT_TIMEOUT_S
            # ON → the classify smart default (180) — qwen-analyst is a legacy
            # alias resolving to classify since the 2026-07-03 decommission.
            svc._flags.set_many({"smart_default_timeout": True})
            r = await asyncio.wait_for(svc.handle_openai_chat(dict(body), _Req()), timeout=10.0)
            assert r.status_code == 200
            assert created[-1].timeout_s == 180.0
            # A client-supplied timeout_s in the OpenAI body still wins.
            r = await asyncio.wait_for(
                svc.handle_openai_chat({**body, "timeout_s": 42}, _Req()), timeout=10.0)
            assert r.status_code == 200
            assert created[-1].timeout_s == 42.0
    finally:
        await svc.shutdown()
