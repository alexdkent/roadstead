"""The interactive band's invariants.

Two defects this pins, both of which were live on one registration in the origin
fleet between 2026-08-23 and 2026-08-24:

1. **A user-facing chat caller was in the BACKGROUND band.** It was registered
   `P3_INGESTION` while it was still framed as an evaluation surface. After it
   became the chat brain behind a phone app and a web UI, that put every turn a
   human waits on behind ingestion/hygiene batch work and excluded it from
   `fast_path_reserve_slots`. Measured at the time: `tier3` p95 background wait
   583,017 ms, one trivial call 254.9 s.

2. **A deadline floor ABOVE its own ceiling.** The floor was
   `_SMART_DEFAULT_CAP_S` (1800 s) — the BACKGROUND cap — which is 3x the
   interactive ceiling (600 s). Correct while the caller was background;
   incoherent the moment it moved, and nothing failed on it.

🚨 The subject changed on 2026-09-01, the doctrine did not. These used to assert
about specific fleet hosts by address, which is both scrub item S1 and a test
that could only ever protect one deployment. Defect (2) is now a **load-time
guard in `identity.py`** that fires for anybody's registration, in either
registry, and this file drives that guard — including the direction that must
NOT fire, since a check that reports everything protects nothing.
"""
from __future__ import annotations

import asyncio

import pytest

from roadstead import hooks
from roadstead.backend import BackendResponse
from roadstead.config import LLMPriority, PriorityBand, ProxyConfig, priority_to_band
from roadstead.constants import _INTERACTIVE_CEILING_S, _SMART_DEFAULT_CAP_S
from roadstead.identity import KeyRegistry
from roadstead.service import ProxyService

_INTERACTIVE = (LLMPriority.P0_REALTIME, LLMPriority.P1_TURN_SUPPORT)


@pytest.fixture
def degradations():
    """Capture what the package reports out through its integration seam."""
    seen: list[dict] = []

    def sink(*, component, reason, impact, **fields):
        seen.append({"component": component, "reason": reason,
                     "impact": impact, **fields})

    hooks.set_degradation_sink(sink)
    try:
        yield seen
    finally:
        hooks.set_degradation_sink(None)


# --------------------------------------------------------------------------
# The band mapping itself.
# --------------------------------------------------------------------------

def test_the_interactive_band_is_p0_and_p1_only():
    """The premise both defects rest on. If this mapping ever widens, "put the
    chat brain in the interactive band" stops meaning what it meant."""
    assert {p for p in LLMPriority
            if priority_to_band(p) is PriorityBand.INTERACTIVE} == set(_INTERACTIVE)


def test_the_interactive_ceiling_has_one_source_of_truth():
    """`config` must derive its default from `constants`, not restate the number.

    Two literals drift silently; the floor and the ceiling then disagree and
    nothing fails until a request is scheduled.
    """
    assert ProxyConfig().timeout_ceiling_interactive_s == _INTERACTIVE_CEILING_S


# --------------------------------------------------------------------------
# Defect 2, generically: a floor above its own ceiling is reported at LOAD.
# --------------------------------------------------------------------------

def test_an_interactive_key_floored_above_its_ceiling_is_reported(degradations):
    keys = KeyRegistry()
    keys.register(secret="k", agent_id="chat-brain",
                  priority=LLMPriority.P1_TURN_SUPPORT,
                  min_timeout_s=_SMART_DEFAULT_CAP_S, key_id="chat-brain-key")
    assert len(degradations) == 1, (
        "an interactive identity floored at the BACKGROUND cap was accepted in "
        "silence — this is the exact shape that was live for a day")
    reported = degradations[0]
    assert reported["component"] == "identity"
    assert reported["agent_id"] == "chat-brain"
    assert reported["min_timeout_s"] == _SMART_DEFAULT_CAP_S
    assert reported["interactive_ceiling_s"] == _INTERACTIVE_CEILING_S
    # 🚨 It REPORTS, it does not refuse and it does not clamp. Refusing to start
    # over a policy typo is worse than serving with a loud line, and clamping
    # would hide the mistake being made.
    assert keys.resolve("k").min_timeout_s == _SMART_DEFAULT_CAP_S


@pytest.mark.parametrize(
    "priority,floor",
    [
        # Background caller, background floor — the case the floor was BUILT for.
        (LLMPriority.P3_INGESTION, _SMART_DEFAULT_CAP_S),
        # Interactive caller, a floor at its ceiling — coherent, on the boundary.
        (LLMPriority.P1_TURN_SUPPORT, _INTERACTIVE_CEILING_S),
        # Interactive caller, no floor at all — the common case.
        (LLMPriority.P1_TURN_SUPPORT, None),
    ],
)
def test_coherent_registrations_are_not_reported(degradations, priority, floor):
    """The counterweight. A guard that fires on the background floor — which is
    the floor's whole purpose — would be noise an operator learns to ignore, and
    then it protects nothing."""
    KeyRegistry().register(secret="k", agent_id="a", priority=priority,
                           min_timeout_s=floor)
    assert degradations == []


def test_the_two_caps_are_still_three_ceilings_apart():
    """Why the mistake was available at all: the background cap is 3x the
    interactive ceiling, so leaving a promoted caller's floor alone yields a
    floor that can never bind."""
    assert _SMART_DEFAULT_CAP_S > _INTERACTIVE_CEILING_S
    assert _SMART_DEFAULT_CAP_S == 1800.0 and _INTERACTIVE_CEILING_S == 600.0


# --------------------------------------------------------------------------
# Defect 1, generically: a registration's band reaches the scheduler.
# --------------------------------------------------------------------------

class _Req:
    def __init__(self, host="203.0.113.10", headers=None):
        class _C:
            pass

        _C.host = host
        self.client = _C()
        self.headers = headers or {}
        self.method = "POST"
        self.query_params: dict = {}


def _body():
    # NO `priority` — the whole point is that the registration supplies it.
    return {"endpoint": "chat", "call_site": "t", "payload_type": "chat_completion",
            "payload": {"messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 8}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spec,expected_band",
    [("chat-brain:P1_TURN_SUPPORT", PriorityBand.INTERACTIVE),
     ("batch-harness:P3_INGESTION", PriorityBand.BACKGROUND)],
)
async def test_a_registrations_band_reaches_the_queued_request(
    monkeypatch, spec, expected_band,
):
    """A registered default priority that never reaches the scheduler is a
    policy nobody can see is wrong — which is how defect 1 survived. Assert the
    band on the QUEUED REQUEST, not on the registry that declared it."""
    monkeypatch.setenv("ROADSTEAD_API_KEYS", f"kk={spec}")
    monkeypatch.delenv("ROADSTEAD_ACL", raising=False)
    svc = ProxyService(ProxyConfig())

    async def ok_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"content": "y"}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            duration_s=0.01, input_tokens=1, output_tokens=1, finish_reason="stop")

    svc._backend.call = ok_call
    from roadstead import scheduler as _sched
    created: list = []
    orig = _sched.QueuedRequest.create.__func__
    _sched.QueuedRequest.create = classmethod(
        lambda cls, **kw: created.append(orig(cls, **kw)) or created[-1])
    await svc.startup()
    try:
        r = await asyncio.wait_for(
            svc.handle_submit(_body(), _Req(headers={"X-API-Key": "kk"})),
            timeout=10.0)
        assert r.status_code == 200
        assert created[-1].band is expected_band
        assert created[-1].agent_id == spec.split(":")[0]
    finally:
        _sched.QueuedRequest.create = classmethod(orig)
        await svc.shutdown()
