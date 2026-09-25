"""``policy.thinking_effort`` — mapping the `thinking:` opt-in to a reasoning
EFFORT on an endpoint with NO thinking switch at all.

WHY THIS EXISTS. ``Correction.apply_thinking`` bails when an endpoint's
``thinking_kwargs`` is empty, on the assumption that "empty" means "we
haven't measured this model's switch yet". GLM-5.3-Flash (tier3, cutover
2026-09-25) collapses a DIFFERENT case onto the same bail: it has no switch
to declare because it always reasons, and its chat template's only lever is
``chat_template_kwargs.reasoning_effort`` (``"low"``/``"high"``, anything
else renders ``"max"``; server default ``"low"``). Without this field, a
caller's ``thinking: true`` on such an endpoint was a SILENT no-op — see
``docs/ledger.md`` "`thinking: true` was a silent no-op on an always-thinking
model with no switch".

Uses the same lightweight mock-``self`` pattern as
``tests/test_thinking_option.py`` / ``tests/test_reasoning_replay.py``: bind
the real ``Correction.apply_thinking`` to a ``types.SimpleNamespace`` carrying
just the ``.state`` fields it reads.
"""
from __future__ import annotations

import importlib
import types

correction = importlib.import_module("roadstead.correction")
model_catalog = importlib.import_module("roadstead.model_catalog")

C = correction.Correction


def _req(payload, *, stream=False, ptype="chat_completion", endpoint="tier3",
         rid="r1", call_site="test"):
    r = types.SimpleNamespace()
    r.payload = payload; r.stream = stream; r.payload_type = ptype
    r.endpoint = endpoint; r.request_id = rid; r.call_site = call_site
    return r


def test_thinking_effort_is_in_the_policy_passthrough_allowlist():
    assert "thinking_effort" in model_catalog._POLICY_PASSTHROUGH


def test_thinking_effort_reaches_endpoint_config():
    from roadstead.config import EndpointConfig

    entry = model_catalog.EndpointEntry(
        name="probe3", provider="p", kind="chat",
        policy={"thinking_effort": "high"},
    )
    kw = model_catalog.build_endpoint_kwargs(entries=[entry])["probe3"]
    assert kw["thinking_effort"] == "high"
    ep = EndpointConfig(**{k: v for k, v in kw.items()
                           if k in EndpointConfig.__dataclass_fields__})
    assert ep.thinking_effort == "high"


def _thinking_mock_self(*, thinking_kwargs=(), thinking_effort="",
                        reasoning_effort="", engine="vllm"):
    ep = types.SimpleNamespace(
        backend_engine=engine, thinking_kwargs=tuple(thinking_kwargs),
        thinking_effort=thinking_effort, reasoning_effort=reasoning_effort,
        thinking_reasoning_budget=0)
    state = types.SimpleNamespace()
    state.config = types.SimpleNamespace(endpoints={"tier3": ep})
    state.thinking_active = {}
    m = types.SimpleNamespace(state=state)
    for pub in ("apply_thinking", "thinking_allowed_keys", "extract_grammar"):
        setattr(m, pub, getattr(C, pub).__get__(m, C))
    return m


def test_thinking_effort_maps_opt_in_on_a_switchless_endpoint():
    """The GLM shape: no thinking switch at all, only an effort word. Without
    `thinking_effort` declared, `thinking: true` on such an endpoint is a
    silent no-op (see the next test) — this is the fix."""
    m = _thinking_mock_self(thinking_kwargs=(), thinking_effort="high")
    p = {"messages": [], "max_tokens": 800, "thinking": True}
    m.apply_thinking(_req(p))
    ck = p.get("chat_template_kwargs", {})
    assert ck.get("reasoning_effort") == "high"
    # No switch key was injected — there is none to inject.
    assert set(ck) == {"reasoning_effort"}
    assert p["max_tokens"] > 800  # headroom applied, same as a switch-bearing endpoint


def test_without_thinking_effort_a_switchless_endpoint_stays_a_noop():
    """REGRESSION GUARD for the defect this field fixes (docs/ledger.md): an
    always-thinking endpoint with thinking_kwargs=() and NO thinking_effort
    declared must still be a silent no-op, exactly as it was before this
    field existed."""
    m = _thinking_mock_self(thinking_kwargs=(), thinking_effort="")
    p = {"messages": [], "max_tokens": 800, "thinking": True}
    m.apply_thinking(_req(p))
    assert "chat_template_kwargs" not in p
    assert p["max_tokens"] == 800


def test_callers_own_effort_wins_over_thinking_effort():
    m = _thinking_mock_self(thinking_kwargs=(), thinking_effort="high")
    p = {"messages": [], "max_tokens": 800, "thinking": True,
         "reasoning_effort": "low"}
    m.apply_thinking(_req(p))
    assert p["chat_template_kwargs"]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in p  # folded, alias removed


def test_thinking_effort_does_not_change_a_switch_bearing_endpoint():
    """An endpoint that DOES declare a switch keeps using `reasoning_effort`,
    not `thinking_effort` — the two fields target different endpoint shapes
    and a switch-bearing endpoint declaring only the switch-based field must
    behave exactly as it always did."""
    m = _thinking_mock_self(thinking_kwargs=("enable_thinking",),
                            reasoning_effort="low", thinking_effort="")
    p = {"messages": [], "max_tokens": 800, "thinking": True}
    m.apply_thinking(_req(p))
    ck = p["chat_template_kwargs"]
    assert ck["enable_thinking"] is True
    assert ck["reasoning_effort"] == "low"


def test_thinking_effort_streaming_applies_the_same_as_sync():
    m = _thinking_mock_self(thinking_kwargs=(), thinking_effort="high")
    p = {"messages": [], "max_tokens": 800, "thinking": True}
    m.apply_thinking(_req(p, stream=True))
    assert p["chat_template_kwargs"]["reasoning_effort"] == "high"
    assert m.state.thinking_active == {}  # streaming never registers for finalize
