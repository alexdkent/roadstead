"""Per-endpoint reasoning TUNING — effort, an absolute cap, and headroom.

WHY THIS EXISTS. Turning thinking on is not the same as making it usable, and
before 2026-09-07 the proxy could only do the first. Three gaps, each measured:

  1. EFFORT. A chat template may carry its own effort default, so "thinking on"
     is not a neutral toggle. Qwen3.8 renders ``reasoning_effort|default('xhigh')``
     — its MAXIMUM. Verified against a live ``/apply-template``:
     ``{enable_thinking:true}`` alone renders BYTE-IDENTICAL to that same call
     with ``reasoning_effort:"xhigh"``. Measured N=2 at ``max_tokens`` 4000 on
     that endpoint: unset/xhigh returned ``finish_reason=length`` with ZERO
     answer chars 2/2, all 4,000 tokens spent on ~15k chars of reasoning, while
     ``low`` returned a complete answer in half the wall clock. It is a prompt
     prefix, not a bound — ``low`` gave a LONGER answer, not a truncated one.

  2. THE CAP'S WIRE NAME. ``thinking_budget_ratio`` emits vLLM's
     ``thinking_token_budget``. llama.cpp reads ``reasoning_budget_tokens`` and
     ignores the vLLM spelling silently, so every llama.cpp stanza's ratio was a
     documented no-op — a declared cap that bounded nothing and read as applied.

  3. THE CAP'S SHAPE. A RATIO only bites when reasoning would otherwise eat the
     whole allowance. Where natural reasoning already lands well under it, the
     ratio computes a number ABOVE the tail it was meant to bound and is inert.
     Measured N=2, ``max_tokens`` 4000, Qwen3.6-35B-A3B on llama.cpp: natural
     reasoning ~2,000 tokens, so 0.6 (=2,400) never bound and a declared 2000
     barely did; only a declared 512 moved anything — reasoning 8,595 -> 1,858
     chars, wall time 31s -> 14s, answer 5,082 -> 4,425 chars.

Every one of the three is "absent => inject nothing", so an undeclared endpoint
behaves exactly as it did before this file existed. That is the property most of
these tests are really pinning: the failure mode here is not a wrong value, it is
a declaration that silently does nothing.
"""
from __future__ import annotations

import types

import pytest

from roadstead import model_catalog
from roadstead.providers import LLAMACPP, VLLM
from roadstead.providers.payload import (
    _THINKING_BUDGET_FLOOR,
    _apply_thinking_token_budget,
)


def _payload(**over):
    p = {"model": "m", "max_tokens": 4000,
         "messages": [{"role": "user", "content": "hi"}],
         "chat_template_kwargs": {"enable_thinking": True}}
    p.update(over)
    return p


# ---------------------------------------------------------------- the wire name

def test_llamacpp_and_vllm_do_not_share_a_spelling():
    """The two engines name the same concept differently, and getting it wrong
    is SILENT on llama.cpp — it ignores the unknown field, so a declared cap
    reads as applied and bounds nothing. This is the defect that made every
    llama.cpp `thinking_budget_ratio` a no-op."""
    assert LLAMACPP.descriptor.reasoning_budget_field == "reasoning_budget_tokens"
    assert VLLM.descriptor.reasoning_budget_field == "thinking_token_budget"


@pytest.mark.parametrize("provider,field", [
    (LLAMACPP, "reasoning_budget_tokens"),
    (VLLM, "thinking_token_budget"),
])
def test_each_provider_emits_its_own_spelling(provider, field):
    """Through the real entry point, not the helper — the payload that reaches
    the wire must carry the name THAT engine reads, and not the other one."""
    out = provider.prepare_chat_payload(
        _payload(), model_id="m", reasoning_budget_tokens=512)
    assert out[field] == 512
    other = ({"reasoning_budget_tokens", "thinking_token_budget"} - {field}).pop()
    assert other not in out, (
        f"{provider.descriptor.name} emitted {other}, which it does not read")


def test_an_engine_with_no_cap_parameter_gets_nothing():
    """`field=None` means the engine has no such parameter. Injecting a
    plausible-looking key anyway is how a cap comes to bound nothing."""
    p = _payload()
    _apply_thinking_token_budget(p, 0.6, field=None, absolute=512)
    assert "reasoning_budget_tokens" not in p
    assert "thinking_token_budget" not in p


# ---------------------------------------------------------------- absolute vs ratio

def test_an_absolute_below_the_ratio_floor_survives():
    """🚨 THE REGRESSION THIS FILE EXISTS FOR. `_THINKING_BUDGET_FLOOR` is 2000
    and was derived for the RATIO path on a long-form reasoner. Applying it to a
    declared absolute would raise the one value measured to work (512) to a value
    measured NOT to bind (2000) — leaving the config looking tuned and the
    endpoint behaving exactly as it did untuned."""
    assert 512 < _THINKING_BUDGET_FLOOR, "the floor no longer makes this a real risk"
    p = _payload()
    _apply_thinking_token_budget(
        p, 0.0, field="reasoning_budget_tokens", absolute=512)
    assert p["reasoning_budget_tokens"] == 512


def test_an_absolute_beats_a_ratio_that_would_not_bind():
    """Both declared: the absolute wins. A 0.6 ratio of 4000 is 2400 — above the
    ~2,000-token tail it was supposed to bound, i.e. inert."""
    p = _payload()
    _apply_thinking_token_budget(
        p, 0.6, field="reasoning_budget_tokens", absolute=512)
    assert p["reasoning_budget_tokens"] == 512


def test_the_ratio_path_is_unchanged_when_no_absolute_is_declared():
    """Nothing above may disturb the existing reasoner behaviour."""
    p = _payload(max_tokens=12000)
    _apply_thinking_token_budget(p, 0.6, field="thinking_token_budget")
    assert p["thinking_token_budget"] == 7200


def test_an_absolute_that_would_leave_no_answer_room_is_refused():
    """The whole point is answer headroom. A cap at or above max_tokens gives the
    reasoning the entire allowance, which is the failure being prevented."""
    p = _payload(max_tokens=500)
    _apply_thinking_token_budget(
        p, 0.0, field="reasoning_budget_tokens", absolute=500)
    assert "reasoning_budget_tokens" not in p


def test_a_caller_that_set_its_own_cap_is_never_overridden():
    p = _payload()
    p["reasoning_budget_tokens"] = 99
    _apply_thinking_token_budget(
        p, 0.0, field="reasoning_budget_tokens", absolute=512)
    assert p["reasoning_budget_tokens"] == 99


def test_no_cap_when_thinking_is_off():
    """A reasoning cap on a request that will not reason is meaningless, and
    would be a payload key the endpoint never asked for."""
    p = _payload(chat_template_kwargs={"enable_thinking": False})
    _apply_thinking_token_budget(
        p, 0.0, field="reasoning_budget_tokens", absolute=512)
    assert "reasoning_budget_tokens" not in p


def test_absent_declaration_leaves_the_payload_byte_identical():
    """The feature-off default, asserted as EQUALITY rather than as the absence
    of one key — a new mechanism must not perturb an undeclared endpoint at all."""
    before = _payload()
    after = LLAMACPP.prepare_chat_payload(dict(before), model_id="m")
    assert after == before


# ---------------------------------------------------------------- catalog plumbing

@pytest.mark.parametrize("key", [
    "reasoning_effort",
    "reasoning_budget_tokens",
    "thinking_reasoning_budget",
])
def test_the_policy_key_is_not_silently_dropped(key):
    """A `policy:` key absent from the passthrough tuple is discarded WITHOUT
    RAISING — models.yaml keeps the value, EndpointConfig keeps the default, and
    the tuning simply never applies while the config looks correct. Every other
    declaration in that tuple carries a test exactly like this one."""
    assert key in model_catalog._POLICY_PASSTHROUGH, (
        f"build_endpoint_kwargs no longer copies {key}; any stanza declaring it "
        f"would be silently dropped — add it to model_catalog._POLICY_PASSTHROUGH")


def test_a_declared_stanza_reaches_endpoint_config():
    """End to end through the real builder: a synthetic entry, so this stays true
    for whichever endpoint next needs the knobs rather than pinning today's."""
    from roadstead.config import EndpointConfig

    entry = model_catalog.EndpointEntry(
        name="probe", provider="p", kind="chat",
        policy={"reasoning_effort": "low",
                "reasoning_budget_tokens": 512,
                "thinking_reasoning_budget": 2000},
    )
    kw = model_catalog.build_endpoint_kwargs(entries=[entry])["probe"]
    assert kw["reasoning_effort"] == "low"
    assert kw["reasoning_budget_tokens"] == 512
    assert kw["thinking_reasoning_budget"] == 2000
    # And they must be REAL fields, not kwargs EndpointConfig rejects.
    ep = EndpointConfig(**{k: v for k, v in kw.items()
                           if k in EndpointConfig.__dataclass_fields__})
    assert ep.reasoning_effort == "low"
    assert ep.reasoning_budget_tokens == 512
    assert ep.thinking_reasoning_budget == 2000


# ---------------------------------------------------------------- apply_thinking

def _optin_self(*, thinking_kwargs=("enable_thinking",), reasoning_effort="",
                thinking_reasoning_budget=0, endpoint="tier2"):
    """Same shape as test_thinking_kwargs_are_family_aware's fixture — a
    namespace endpoint, so this also pins that the new reads stay defensive."""
    from roadstead.correction import Correction

    ep = types.SimpleNamespace(
        backend_engine="llama.cpp", thinking_kwargs=tuple(thinking_kwargs),
        reasoning_effort=reasoning_effort,
        thinking_reasoning_budget=thinking_reasoning_budget)
    state = types.SimpleNamespace(
        config=types.SimpleNamespace(endpoints={endpoint: ep}),
        thinking_active={})
    m = types.SimpleNamespace(state=state)
    for name in ("thinking_allowed_keys", "apply_thinking", "extract_grammar"):
        setattr(m, name, getattr(Correction, name).__get__(m, Correction))
    return m


def _req(payload, endpoint="tier2"):
    return types.SimpleNamespace(
        payload=payload, stream=False, payload_type="chat_completion",
        endpoint=endpoint, request_id="r1", call_site="test")


def test_declared_effort_rides_in_the_same_object_as_the_switch():
    """🚨 THE ONE THAT MATTERS. Per-request `chat_template_kwargs` merge
    KEY-BY-KEY over the server's launch default, so an effort that arrives
    WITHOUT the switch renders with thinking OFF and the effort dropped —
    verified against a live `/apply-template`. Asserting the whole dict, rather
    than the presence of the effort key, is what pins them together."""
    m = _optin_self(reasoning_effort="low")
    p = {"messages": [], "max_tokens": 800, "thinking": True}
    m.apply_thinking(_req(p))
    assert p["chat_template_kwargs"] == {
        "enable_thinking": True, "reasoning_effort": "low"}


def test_no_declaration_injects_no_effort():
    """The feature-off default. Absent declaration => the template's own default,
    which is the pre-2026-09-07 behaviour exactly."""
    m = _optin_self()
    p = {"messages": [], "max_tokens": 800, "thinking": True}
    m.apply_thinking(_req(p))
    assert p["chat_template_kwargs"] == {"enable_thinking": True}


def test_a_callers_own_effort_wins():
    """A per-endpoint DEFAULT for a caller that said nothing — never an override
    of a caller that did. Same rule the switch keys follow."""
    m = _optin_self(reasoning_effort="low")
    p = {"messages": [], "max_tokens": 800, "thinking": True,
         "chat_template_kwargs": {"reasoning_effort": "xhigh"}}
    m.apply_thinking(_req(p))
    assert p["chat_template_kwargs"]["reasoning_effort"] == "xhigh"
    assert p["chat_template_kwargs"]["enable_thinking"] is True


def test_a_declared_headroom_replaces_the_global():
    """8000 is one number for every endpoint and was sized for a long-form
    reasoner. On a slow endpoint its unused allowance is also unbudgeted WALL
    CLOCK, because the deadline was resolved from the caller's max_tokens before
    this inflation happens."""
    m = _optin_self(thinking_reasoning_budget=2000)
    p = {"messages": [], "max_tokens": 1400, "thinking": True}
    m.apply_thinking(_req(p))
    assert p["max_tokens"] == 3400, (
        "the declared headroom must replace the flat global, not add to it")


def test_an_undeclared_headroom_still_gets_the_global():
    m = _optin_self()
    p = {"messages": [], "max_tokens": 1400, "thinking": True}
    m.apply_thinking(_req(p))
    assert p["max_tokens"] == 9400


def test_the_int_form_is_capped_by_the_declared_headroom_not_the_global():
    """The int form may only ever ask for LESS. Once an endpoint declares its own
    ceiling, that is the ceiling a caller is capped against — otherwise a caller
    could talk a tuned endpoint back up to the untuned global."""
    m = _optin_self(thinking_reasoning_budget=2000)
    p = {"messages": [], "max_tokens": 1400, "thinking": 6000}
    m.apply_thinking(_req(p))
    assert p["max_tokens"] == 3400
