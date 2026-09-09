"""The proxy's reasoning CAP — `thinking_token_budget` injection.

WHY THIS EXISTS. Measured on tier3 (DeepSeek-V4-Flash) 2026-08-23, N=2 per arm,
interleaved, streaming: with no cap, the model's natural reasoning length is BIMODAL.
Sometimes it terminates around 7-8k tokens and answers cleanly; sometimes it runs past a
12,000-token ceiling and returns `content: ""` with finish=length — 8 and 13 minutes of
compute for nothing. `empty_completion_error` already names that failure precisely; this
is the mechanism that stops producing it.

The cap COMPLEMENTS `Correction.apply_thinking`, which ADDS `thinking_reasoning_budget()`
headroom to `max_tokens`. That makes the pie bigger so reasoning does not truncate the
answer; this slices the pie so reasoning cannot eat all of it. They also fire for
different callers — the opt-in needs the top-level `thinking` control field, while dsh
sets `chat_template_kwargs` directly and never sends it.

🚨 The declaration is the dangerous part. vLLM honours `thinking_token_budget` only when
the server was launched with `--reasoning-config`, and 400s the WHOLE REQUEST when it was
not. So a wrong `thinking_budget_ratio` does not degrade an endpoint — it breaks every
thinking call to it. Hence the bidirectional pin against the vendored serve script.
"""
from __future__ import annotations

import pytest

from roadstead.providers import VLLM
from roadstead.providers.payload import (
    _THINKING_BUDGET_CEILING,
    _THINKING_BUDGET_FLOOR,
    _apply_thinking_token_budget,
)


def _payload(**over):
    p = {"model": "tier3", "max_tokens": 12000,
         "messages": [{"role": "user", "content": "hi"}],
         "chat_template_kwargs": {"thinking": True}}
    p.update(over)
    return p


# ---------------------------------------------------------------- the default is OFF

def test_an_endpoint_that_does_not_declare_support_gets_nothing():
    """The default must be inert. Sending this parameter to a server without
    `--reasoning-config` returns 400 for the entire request, so silence is the only
    safe behaviour for every endpoint that has not declared the launch flag."""
    p = _payload()
    _apply_thinking_token_budget(p, 0.0, field="thinking_token_budget")
    assert "thinking_token_budget" not in p


def test_normalize_does_not_inject_without_a_declared_ratio():
    """Same thing through the real entry point, since the early-return short-circuit
    in `_normalize_chat_payload` is a separate way to skip it."""
    out = VLLM.prepare_chat_payload(_payload(), model_id="tier3")
    assert "thinking_token_budget" not in out


# ---------------------------------------------------------------- the injection

def test_a_declared_endpoint_gets_a_budget_proportional_to_max_tokens():
    p = _payload(max_tokens=12000)
    _apply_thinking_token_budget(p, 0.6, field="thinking_token_budget")
    assert p["thinking_token_budget"] == 7200
    # And the answer keeps the rest — that is the entire point.
    assert p["thinking_token_budget"] < p["max_tokens"]


def test_it_reaches_the_wire_through_normalize():
    out = VLLM.prepare_chat_payload(_payload(), model_id="tier3",
                                  thinking_budget_ratio=0.6)
    assert out["thinking_token_budget"] == 7200


@pytest.mark.parametrize("key", ["thinking", "enable_thinking"])
def test_either_spelling_of_the_thinking_switch_counts(key):
    """The two live vLLM templates spell the switch differently — DeepSeek-V4 uses
    `thinking`, the Qwen tiers `enable_thinking`. Reading only one would make the cap a
    silent no-op for half the fleet, which is the failure shape that reads as
    'the budget didn't work today'."""
    p = _payload(chat_template_kwargs={key: True})
    _apply_thinking_token_budget(p, 0.6, field="thinking_token_budget")
    assert p["thinking_token_budget"] == 7200


# ---------------------------------------------------------------- when NOT to inject

def test_thinking_off_gets_no_budget():
    p = _payload(chat_template_kwargs={"thinking": False})
    _apply_thinking_token_budget(p, 0.6, field="thinking_token_budget")
    assert "thinking_token_budget" not in p


def test_a_caller_that_set_its_own_budget_is_never_overridden():
    """Callers declare intent; the proxy supplies the number only when they didn't."""
    p = _payload(thinking_token_budget=1234)
    _apply_thinking_token_budget(p, 0.6, field="thinking_token_budget")
    assert p["thinking_token_budget"] == 1234


def test_a_tiny_max_tokens_is_left_alone():
    """Below the floor there is no sane split: a 2000-token cap on a 1500-token
    allowance leaves nothing for an answer, which is the failure being prevented."""
    p = _payload(max_tokens=1500)
    _apply_thinking_token_budget(p, 0.6, field="thinking_token_budget")
    assert "thinking_token_budget" not in p


def test_a_missing_max_tokens_is_left_alone():
    p = _payload()
    p.pop("max_tokens")
    _apply_thinking_token_budget(p, 0.6, field="thinking_token_budget")
    assert "thinking_token_budget" not in p


# ---------------------------------------------------------------- floor and ceiling

def test_the_floor_keeps_the_cap_clear_of_the_tool_call_corruption_zone():
    """vLLM #44676: forced reasoning-end tokens land inside tool-call JSON at ~256
    tokens in ~75% of runs. Our agents make dozens of tool calls per task, so the cap
    must never be computed down into that zone by a small max_tokens."""
    p = _payload(max_tokens=4000)
    _apply_thinking_token_budget(p, 0.05, field="thinking_token_budget")      # would compute to 200
    assert p["thinking_token_budget"] == _THINKING_BUDGET_FLOOR
    assert _THINKING_BUDGET_FLOOR > 1024       # above Anthropic's documented minimum


def test_the_ceiling_stops_at_the_point_published_curves_turn_down():
    p = _payload(max_tokens=200000)
    _apply_thinking_token_budget(p, 0.9, field="thinking_token_budget")
    assert p["thinking_token_budget"] == _THINKING_BUDGET_CEILING


def test_a_budget_that_would_leave_no_answer_room_is_refused():
    """Clamping to the floor must not itself produce the starvation it prevents."""
    p = _payload(max_tokens=_THINKING_BUDGET_FLOOR)
    _apply_thinking_token_budget(p, 0.9, field="thinking_token_budget")
    assert "thinking_token_budget" not in p


# ------------------------------------------------------- the declaration is WIRED

def test_the_models_yaml_key_reaches_endpoint_config():
    """A key missing from `model_catalog`'s mapping tuple is SILENTLY DROPPED.

    That file says so in its own comment, and every other declaration there
    (`disable_any_whitespace`, `readiness_critical`, `failover_dwell_s`) carries a
    test exactly like this one, because the failure is invisible: models.yaml keeps
    the value, `EndpointConfig` keeps the default, and the cap simply never applies
    while the config looks correct.
    """
    from roadstead import config

    ep = config.DEFAULT_ENDPOINTS[config.normalize_endpoint("tier3")]
    assert hasattr(ep, "thinking_budget_ratio"), (
        "EndpointConfig has no thinking_budget_ratio field (config.py)")
    assert ep.thinking_budget_ratio == pytest.approx(0.6), (
        f"models.yaml declares reasoner.policy.thinking_budget_ratio but "
        f"EndpointConfig sees {ep.thinking_budget_ratio!r} — the key is not in "
        "model_catalog.py's mapping tuple and is being dropped"
    )


def test_no_other_endpoint_declares_a_ratio_it_cannot_honour():
    """Only endpoints whose server runs `--reasoning-config` may declare a ratio.

    vLLM 400s the entire request when the parameter arrives without that flag, so a
    stray declaration on tier2 would not degrade it — it would break every thinking
    call to it. Today exactly one backend is launched with the flag.
    """
    from roadstead import config

    declared = {name for name, ep in config.DEFAULT_ENDPOINTS.items()
                if getattr(ep, "thinking_budget_ratio", 0)}
    # Aliases of the same endpoint resolve to the same object, so compare the set of
    # distinct ROLES rather than of names.
    # models.yaml keys the block `reasoner`; the resolved endpoint is `tier3`
    # with role `tier3`. Assert on the RESOLVED role, since that is what the
    # dispatch path actually reads.
    roles = {config.DEFAULT_ENDPOINTS[n].role for n in declared}
    assert roles == {"tier3"}, (
        f"roles declaring a reasoning cap: {roles} (via {sorted(declared)}). "
        "Every one of them must be "
        "launched with --reasoning-config; see the vendored serve scripts."
    )


# ---------------------------------------------------------------- tool turns
#
# 🚨 A BUDGET THAT CUTS REASONING SHORT CORRUPTS THE TOOL-CALL CHANNEL.
# NOT "any binding budget" — measured dose-response, n=4, natural reasoning
# ~210 tok: budget 64 -> 0/4 tool calls, 128 -> 0/4, 256 -> 3/4, 512 -> 4/4,
# none -> 4/4. A budget above the turn's natural reasoning never binds and is
# harmless; the failure is confined to budgets at or below it. Evidence:
#
#   * Forced to bind on a simple tool turn (budget 50): arguments stayed clean
#     0/9, but 3 of 12 turns emitted NO TOOL CALL AT ALL where the control
#     emitted 12/12.
#   * Agentic coding, 2026-09-09, `pi` on tier3 driving a multi-file build:
#     13 draws across budgets 2,048 / 8,192 / 16,384 wrote files in 2. Output
#     tokens pinned to the budget every time (2,048 -> out 2,218; 16,384 -> out
#     16,513), i.e. reasoning consumed the whole allowance, the forced
#     `reasoning_end_str` fired, and the tool call died with it. With thinking
#     OFF the same harness went 3/3 with a perfect acceptance score.
#
# WHY THE EARLIER PROBE SAID THIS COULD NOT HAPPEN. It measured natural
# reasoning on a tool turn at 70-234 tokens against a ~3,600-token budget and
# concluded the ratio "cannot engage". That is true of a SIMPLE tool turn and
# false of an agentic build, where reasoning reaches the full budget. The
# binding point is a property of the WORKLOAD SHAPE, not of the endpoint — so
# "it never binds here" is not a safety argument, it is a statement about the
# workload that was sampled.
#
# Hence: on a turn that declares tools, inject nothing. The upstream failures a
# binding budget triggers are open with no fix (vLLM #39697 leaks
# `reasoning_end_str` into `content`; #44676 corrupts tool-call arguments), and
# a suppressed tool call is silent — the caller gets a confident answer and no
# side effect.


def test_tools_present_suppresses_ratio_injection():
    """A tool-calling turn gets NO ratio-derived budget.

    The safe budget is a function of the TURN's natural reasoning length, which
    is not knowable before generating it — so any fixed number is a gamble whose
    loss (a silently suppressed tool call) is worse than the runaway-reasoning
    tail the cap exists to bound."""
    p = _payload(tools=[{"type": "function",
                         "function": {"name": "write", "parameters": {}}}])
    _apply_thinking_token_budget(p, 0.6, field="thinking_token_budget")
    assert "thinking_token_budget" not in p, (
        "a tool-calling turn must not receive a ratio-derived reasoning budget: "
        "a binding budget suppresses the tool call outright"
    )


def test_tools_present_suppresses_absolute_injection():
    """Same for an operator-declared absolute — the harm is WHERE THE CUT LANDS,
    not the provenance of the number."""
    p = _payload(tools=[{"type": "function",
                         "function": {"name": "write", "parameters": {}}}])
    _apply_thinking_token_budget(p, 0.0, field="thinking_token_budget",
                                 absolute=4000)
    assert "thinking_token_budget" not in p


def test_no_tools_still_injects():
    """CONTROL — the guard must be narrow. Without tools the cap still applies,
    because the bimodal-reasoning tail it exists to bound is real."""
    p = _payload()
    _apply_thinking_token_budget(p, 0.6, field="thinking_token_budget")
    assert p["thinking_token_budget"] > 0


def test_empty_tools_list_still_injects():
    """`tools: []` is not a tool turn — an empty list means the caller declared
    no tools, so nothing is suppressed."""
    p = _payload(tools=[])
    _apply_thinking_token_budget(p, 0.6, field="thinking_token_budget")
    assert p["thinking_token_budget"] > 0


def test_caller_supplied_budget_on_a_tool_turn_is_STILL_honoured():
    """🚨 THE KNOWN GAP, PINNED SO IT IS NOT MISREAD AS COVERED.

    The tool-turn guard sits BELOW the `field in p` early return, so a caller
    that sends its OWN `thinking_token_budget` still gets it on a tool turn —
    "caller declared intent; never override it" wins.

    That is deliberate but it is NOT harmless, and it is exactly the shape that
    was measured failing: `pi` sends its own budget (1,024 / 2,048 / 8,192 /
    16,384 depending on `--thinking`), so THIS FIX DOES NOT COVER pi. It covers
    ratio-driven callers — dsh's `coder` and `code-reviewer`, which send no
    budget of their own.

    Whether the proxy should override a caller's budget on a tool turn is an
    open policy question: it contradicts caller-intent, and the alternative is a
    silently suppressed tool call. Left to the operator; this test exists so the
    gap is visible rather than assumed shut.
    """
    p = _payload(tools=[{"type": "function",
                         "function": {"name": "write", "parameters": {}}}],
                 thinking_token_budget=8192)
    _apply_thinking_token_budget(p, 0.6, field="thinking_token_budget")
    assert p["thinking_token_budget"] == 8192, (
        "caller-supplied budgets are still honoured verbatim on tool turns — "
        "if this changed, the pi coverage gap closed and the docstring is stale"
    )


def test_guard_is_REACHED_through_the_real_provider_call_site():
    """A unit test proves the helper works; this proves the provider actually
    calls it that way. The guard is worthless if `prepare_chat_payload` never
    reaches it."""
    payload = {"model": "tier3", "max_tokens": 12000,
               "messages": [{"role": "user", "content": "hi"}],
               "chat_template_kwargs": {"thinking": True},
               "tools": [{"type": "function",
                          "function": {"name": "write", "parameters": {}}}]}
    out = VLLM.prepare_chat_payload(payload, model_id="tier3",
                                    thinking_budget_ratio=0.6,
                                    thinking_kwargs=("thinking",))
    assert "thinking_token_budget" not in out, (
        "the ratio was injected through the provider path despite tools being "
        "declared — the guard is not on the path that production uses"
    )


def test_provider_call_site_still_injects_without_tools():
    """CONTROL for the above — same call, no tools, budget present."""
    payload = {"model": "tier3", "max_tokens": 12000,
               "messages": [{"role": "user", "content": "hi"}],
               "chat_template_kwargs": {"thinking": True}}
    out = VLLM.prepare_chat_payload(payload, model_id="tier3",
                                    thinking_budget_ratio=0.6,
                                    thinking_kwargs=("thinking",))
    assert out.get("thinking_token_budget", 0) > 0
