"""A FORCED-reasoning endpoint's declared `policy.reasoning_effort` must reach
the wire — and the switch must ride with it.

WHY THIS EXISTS. tier2-flash (Qwen3.8-Flash-Next on halogen) shipped declaring
`policy.reasoning_effort: low` and ran EVERY request at its template's own
default, `xhigh` — the model's maximum. The declaration was inert: the only
injection site in the proxy was `Correction.apply_thinking`, which returns early
unless the CALLER sent a `thinking:` control field. That models the vLLM world,
where reasoning is OFF until somebody asks for it. A forced-reasoning endpoint is
the opposite world — the model reasons on every request and nobody opts in — so
an effort that only arrives via an opt-in never arrives at all.

Measured cost before the fix (2026-09-14, live):
    ROADSTEAD_TRUNCATION model=tier2-flash agent=gateway-external
    max_tokens=3584 output_tokens=3584          # 100.6s, finish_reason=length,
                                                # ZERO content — all of it reasoning
And through the proxy, same endpoint, same trivial prompt:
    no kwargs                                  -> 339 chars of reasoning
    {enable_thinking: true, reasoning_effort: "xhigh"} -> 339   # baseline IS xhigh
    {enable_thinking: true, reasoning_effort: "minimal"} -> 104
    {enable_thinking: false}                   ->   0
⇒ the effort key DOES work over `chat_template_kwargs`; nothing was sending it.

🚨 THE SWITCH MUST TRAVEL IN THE SAME OBJECT. Per-request `chat_template_kwargs`
merge KEY-BY-KEY over the server's launch default, so a lone effort key can
render against whatever the template defaults its switch to — dropping the effort
silently, or disabling reasoning on a reasoning tier. An endpoint declaring an
effort and NO `thinking_kwargs` is therefore SKIPPED and reported here, never
guessed at; the reconcile gate fails that combination separately.
"""
from __future__ import annotations

import importlib
import types

import pytest

config = importlib.import_module("roadstead.config")
correction = importlib.import_module("roadstead.correction")
C = correction.Correction


def _req(payload, *, stream=False, ptype="chat_completion", endpoint="tier2-flash"):
    r = types.SimpleNamespace()
    r.payload = payload; r.stream = stream; r.payload_type = ptype
    r.endpoint = endpoint; r.request_id = "r1"; r.call_site = "test"
    return r


def _mock(*, forces=True, effort="low", switch=("enable_thinking",),
          budget=0, endpoint="tier2-flash"):
    """Mock `self` for apply_forced_reasoning_budget. Each knob is a separate
    argument so every TERM of the compound guard can be sabotaged alone."""
    ep = types.SimpleNamespace(backend_engine="halogen", forces_reasoning=forces,
                               reasoning_effort=effort, thinking_kwargs=switch,
                               forced_reasoning_budget=budget)
    state = types.SimpleNamespace(config=types.SimpleNamespace(endpoints={endpoint: ep}))
    m = types.SimpleNamespace(state=state)
    m.apply_forced_reasoning_budget = C.apply_forced_reasoning_budget.__get__(m, C)
    return m


# ── the fix itself ────────────────────────────────────────────────────────────

def test_declared_effort_is_injected_with_its_switch_in_one_object():
    """THE REGRESSION. Before the fix this payload reached the backend with no
    `chat_template_kwargs` at all and the model reasoned at `xhigh`."""
    m = _mock()
    req = _req({"max_tokens": 3584, "messages": []})
    m.apply_forced_reasoning_budget(req)
    ck = req.payload["chat_template_kwargs"]
    assert ck["reasoning_effort"] == "low", (
        "the endpoint's declared effort did not reach the payload — this is the "
        "exact state that burned 3,584 tokens on reasoning and returned nothing")
    assert ck["enable_thinking"] is True, (
        "the effort was sent WITHOUT its switch; chat_template_kwargs merge "
        "key-by-key, so a lone effort can be dropped or can disable reasoning")


def test_effort_is_injected_even_with_no_max_tokens():
    """An uncapped caller still REASONS. The budget half has nothing to pad, but
    returning early for the whole method (what it used to do) left those callers
    at `xhigh` too."""
    m = _mock()
    req = _req({"messages": []})
    m.apply_forced_reasoning_budget(req)
    assert req.payload["chat_template_kwargs"]["reasoning_effort"] == "low"
    assert "max_tokens" not in req.payload, "nothing to pad, so nothing added"


def test_streaming_gets_it_too():
    """The forced case is a property of the ENDPOINT, not of the response shape."""
    m = _mock()
    req = _req({"max_tokens": 500, "messages": []}, stream=True)
    m.apply_forced_reasoning_budget(req)
    assert req.payload["chat_template_kwargs"]["reasoning_effort"] == "low"


# ── the caller always wins ────────────────────────────────────────────────────

def test_caller_pin_wins_on_effort():
    m = _mock()
    req = _req({"max_tokens": 500, "messages": [],
                "chat_template_kwargs": {"reasoning_effort": "xhigh"}})
    m.apply_forced_reasoning_budget(req)
    assert req.payload["chat_template_kwargs"]["reasoning_effort"] == "xhigh", (
        "a per-endpoint DEFAULT overrode a caller that explicitly asked")


def test_caller_pin_wins_on_the_switch_so_reasoning_can_be_turned_off():
    """`enable_thinking: false` measured 0 chars of reasoning on this endpoint.
    A caller that deliberately turns reasoning off must keep it off."""
    m = _mock()
    req = _req({"max_tokens": 500, "messages": [],
                "chat_template_kwargs": {"enable_thinking": False}})
    m.apply_forced_reasoning_budget(req)
    ck = req.payload["chat_template_kwargs"]
    assert ck["enable_thinking"] is False
    assert ck["reasoning_effort"] == "low", "the effort default still applies"


def test_other_caller_kwargs_are_preserved():
    m = _mock()
    req = _req({"max_tokens": 500, "messages": [],
                "chat_template_kwargs": {"preserve_thinking": True}})
    m.apply_forced_reasoning_budget(req)
    assert req.payload["chat_template_kwargs"]["preserve_thinking"] is True


# ── sabotage each TERM of the compound guard, separately ──────────────────────

def test_no_injection_when_the_endpoint_does_not_force_reasoning():
    """tier2-analyst declares `low` and is NOT forced — its effort correctly
    reaches the model through the opt-in path instead. Injecting here would
    switch reasoning ON for an endpoint that serves it OFF by default."""
    m = _mock(forces=False)
    req = _req({"max_tokens": 500, "messages": []})
    m.apply_forced_reasoning_budget(req)
    assert "chat_template_kwargs" not in req.payload
    assert req.payload["max_tokens"] == 500, "budget must not move either"


def test_no_injection_when_no_effort_is_declared():
    """An endpoint that forces reasoning but declares no effort keeps the
    template's default — we do not invent a rung for it."""
    m = _mock(effort="")
    req = _req({"max_tokens": 500, "messages": []})
    m.apply_forced_reasoning_budget(req)
    assert "chat_template_kwargs" not in req.payload
    assert req.payload["max_tokens"] == 500 + config.forced_reasoning_budget(), (
        "the budget half is independent and must still fire")


def test_effort_without_a_switch_injects_NOTHING_and_says_so(caplog):
    """THE DEFECT SHAPE ITSELF. A lone effort key is not safe to send, so this
    refuses rather than guessing — and must be LOUD, because a silent skip here
    is indistinguishable from the bug we just fixed."""
    m = _mock(switch=())
    req = _req({"max_tokens": 500, "messages": []})
    with caplog.at_level("WARNING"):
        m.apply_forced_reasoning_budget(req)
    assert "chat_template_kwargs" not in req.payload
    assert "ROADSTEAD_UNREACHABLE_REASONING_EFFORT" in caplog.text, (
        "an unreachable declaration was skipped SILENTLY — that is the original "
        "defect wearing a different hat")


def test_non_chat_payloads_are_untouched():
    m = _mock()
    req = _req({"max_tokens": 500, "prompt": "hi"}, ptype="completion")
    m.apply_forced_reasoning_budget(req)
    assert "chat_template_kwargs" not in req.payload


# ── the per-endpoint budget ───────────────────────────────────────────────────

def test_declared_budget_replaces_the_global():
    """1536 is derived from creative/Trinity-Mini at ~350-500 reasoning tokens.
    An endpoint reasoning an order of magnitude harder needs its own number."""
    m = _mock(budget=6000)
    req = _req({"max_tokens": 2000, "messages": []})
    m.apply_forced_reasoning_budget(req)
    assert req.payload["max_tokens"] == 2000 + 6000
    assert config.forced_reasoning_budget() != 6000, "test would be vacuous"


def test_undeclared_budget_falls_back_to_the_global():
    m = _mock(budget=0)
    req = _req({"max_tokens": 2000, "messages": []})
    m.apply_forced_reasoning_budget(req)
    assert req.payload["max_tokens"] == 2000 + config.forced_reasoning_budget()


def test_declared_budget_reaches_endpoint_config():
    """The knob is worthless if `policy.forced_reasoning_budget` never survives
    catalog load — a dropped policy key looks exactly like one that isn't
    load-bearing."""
    from roadstead.model_catalog import _POLICY_PASSTHROUGH
    assert "forced_reasoning_budget" in _POLICY_PASSTHROUGH
    assert hasattr(config.EndpointConfig, "forced_reasoning_budget") or \
        "forced_reasoning_budget" in getattr(config.EndpointConfig, "__annotations__", {})


# ── the caller's pin in the OpenAI spellings (2026-09-16) ─────────────────────
#
# THE REGRESSION. "A caller's own pin wins" was checked against `ck` only. An
# agent harness speaking plain OpenAI sends a TOP-LEVEL `reasoning_effort`, so the
# declared default was injected BESIDE it and the backend refused the pair on
# every request, live:
#     400 conflicting reasoning_effort: 'medium' at the top level and 'low' in
#         chat_template_kwargs; they are the same control under two names
# and the OpenRouter shape `reasoning: {effort}` drew the same 400.

def _one_effort_name(payload):
    """How many places carry an effort — the backend refuses more than one."""
    n = 0
    if "reasoning_effort" in (payload.get("chat_template_kwargs") or {}):
        n += 1
    if "reasoning_effort" in payload:
        n += 1
    if isinstance(payload.get("reasoning"), dict) and "effort" in payload["reasoning"]:
        n += 1
    return n


def test_top_level_caller_effort_wins_and_is_sent_under_one_name():
    m = _mock()
    req = _req({"max_tokens": 500, "messages": [], "reasoning_effort": "medium"})
    m.apply_forced_reasoning_budget(req)
    p = req.payload
    assert _one_effort_name(p) == 1, (
        "the default was injected beside the caller's top-level effort — the "
        "exact payload the backend 400s as 'the same control under two names'")
    assert p["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "medium"}


def test_reasoning_object_effort_wins_and_its_other_fields_survive():
    m = _mock()
    req = _req({"max_tokens": 500, "messages": [],
                "reasoning": {"effort": "high", "max_tokens": 900}})
    m.apply_forced_reasoning_budget(req)
    p = req.payload
    assert _one_effort_name(p) == 1
    assert p["chat_template_kwargs"]["reasoning_effort"] == "high"
    assert p["reasoning"] == {"max_tokens": 900}, "only `effort` moves"


def test_reasoning_object_holding_only_effort_is_removed_whole():
    m = _mock()
    req = _req({"max_tokens": 500, "messages": [], "reasoning": {"effort": "high"}})
    m.apply_forced_reasoning_budget(req)
    assert "reasoning" not in req.payload


def test_effort_none_turns_the_switch_off():
    """`none` is the OpenAI spelling of thinking OFF, not a rung — and not a
    value a template's effort ladder accepts."""
    m = _mock()
    req = _req({"max_tokens": 500, "messages": [], "reasoning_effort": "none"})
    m.apply_forced_reasoning_budget(req)
    ck = req.payload["chat_template_kwargs"]
    assert ck["enable_thinking"] is False
    assert ck.get("reasoning_effort") != "none"
    assert _one_effort_name(req.payload) <= 1


def test_a_conflict_the_caller_wrote_itself_is_left_as_sent():
    """Two DIFFERENT values from the caller is the caller's own conflict; the
    backend's 400 names it accurately, so the proxy must not pick a winner."""
    m = _mock()
    req = _req({"max_tokens": 500, "messages": [], "reasoning_effort": "medium",
                "reasoning": {"effort": "high"}})
    m.apply_forced_reasoning_budget(req)
    p = req.payload
    assert p["reasoning_effort"] == "medium" and p["reasoning"] == {"effort": "high"}
    assert "reasoning_effort" not in p["chat_template_kwargs"], (
        "a third value was added to a caller's two")


def test_no_caller_effort_still_gets_the_declared_default():
    """Sabotage guard for the fold itself: the fix must not stop the default."""
    m = _mock()
    req = _req({"max_tokens": 500, "messages": [], "reasoning": {"max_tokens": 900}})
    m.apply_forced_reasoning_budget(req)
    assert req.payload["chat_template_kwargs"]["reasoning_effort"] == "low"
    assert req.payload["reasoning"] == {"max_tokens": 900}
