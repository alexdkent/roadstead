"""``policy.reasoning_effort_map`` — the "no accidental MAX" guard (2026-09-26).

Measured on GLM-5.3-Flash (tier3): its chat template renders only
low/high/max, and buckets "medium" — and any unrecognised word, typo
included — into MAX, its most expensive rung, SILENTLY. Operator rule:
"max must be an operator decision or a decision at time of wiring — callers
must not default to max." `Correction.apply_reasoning_effort_map` is the
endpoint's own statement of which words its template understands and what to
do with everything else: a declared word is translated (verbatim, whatever
the operator wrote); an undeclared one is REMOVED so the engine's own
server-side default applies, rather than an unbucketed word landing wherever
the template's fallback happens to be.

Uses the same lightweight mock-``self`` pattern as
``tests/test_thinking_effort.py`` / ``tests/test_thinking_option.py``: bind
the real ``Correction`` methods to a ``types.SimpleNamespace`` carrying just
the ``.state`` fields they read.
"""
from __future__ import annotations

import importlib
import types

correction = importlib.import_module("roadstead.correction")
model_catalog = importlib.import_module("roadstead.model_catalog")

C = correction.Correction


def _req(payload, *, endpoint="tier3", rid="r1", agent_id="chat-agent",
         stream=False, ptype="chat_completion", call_site="test"):
    r = types.SimpleNamespace()
    r.payload = payload; r.stream = stream; r.payload_type = ptype
    r.endpoint = endpoint; r.request_id = rid; r.agent_id = agent_id
    r.call_site = call_site
    return r


def _mock_self(*, reasoning_effort_map=None, thinking_kwargs=(),
               reasoning_effort="", thinking_effort="", engine="vllm"):
    ep = types.SimpleNamespace(
        backend_engine=engine, thinking_kwargs=tuple(thinking_kwargs),
        reasoning_effort=reasoning_effort, thinking_effort=thinking_effort,
        thinking_reasoning_budget=0,
        reasoning_effort_map=dict(reasoning_effort_map or {}))
    state = types.SimpleNamespace()
    state.config = types.SimpleNamespace(endpoints={"tier3": ep})
    state.thinking_active = {}
    state.reasoning_effort_remaps = {}
    state.reasoning_effort_remap_logged = set()
    m = types.SimpleNamespace(state=state)
    for pub in ("apply_thinking", "apply_reasoning_effort_map",
               "_remap_one_effort_field", "thinking_allowed_keys",
               "extract_grammar"):
        setattr(m, pub, getattr(C, pub).__get__(m, C))
    return m


# --------------------------------------------------------------------------- #
# Plumbing: the policy key reaches EndpointConfig like every other one.
# --------------------------------------------------------------------------- #

def test_reasoning_effort_map_is_in_the_policy_passthrough_allowlist():
    assert "reasoning_effort_map" in model_catalog._POLICY_PASSTHROUGH


def test_declared_map_reaches_endpoint_config():
    from roadstead.config import EndpointConfig

    entry = model_catalog.EndpointEntry(
        name="probe4", provider="p", kind="chat",
        policy={"reasoning_effort_map": {"medium": "high", "max": "max"}},
    )
    kw = model_catalog.build_endpoint_kwargs(entries=[entry])["probe4"]
    assert kw["reasoning_effort_map"] == {"medium": "high", "max": "max"}
    ep = EndpointConfig(**{k: v for k, v in kw.items()
                           if k in EndpointConfig.__dataclass_fields__})
    assert ep.reasoning_effort_map == {"medium": "high", "max": "max"}


def test_endpoint_config_ships_no_default_map():
    """{} ⇒ undeclared ⇒ off, the same contract every other policy key here
    follows."""
    from roadstead.config import EndpointConfig
    ep = EndpointConfig(endpoint_class="x", role="x")
    assert ep.reasoning_effort_map == {}


# --------------------------------------------------------------------------- #
# The three semantics, in chat_template_kwargs.
# --------------------------------------------------------------------------- #

def test_medium_is_remapped_to_high_in_chat_template_kwargs():
    m = _mock_self(reasoning_effort_map={"medium": "high"})
    p = {"messages": [], "chat_template_kwargs": {"medium_irrelevant": 1,
                                                  "reasoning_effort": "medium"}}
    m.apply_reasoning_effort_map(_req(p))
    assert p["chat_template_kwargs"]["reasoning_effort"] == "high"


def test_a_typo_is_removed_in_chat_template_kwargs():
    m = _mock_self(reasoning_effort_map={"low": "low", "high": "high"})
    p = {"messages": [], "chat_template_kwargs": {"reasoning_effort": "mediumm"}}
    m.apply_reasoning_effort_map(_req(p))
    assert "reasoning_effort" not in p["chat_template_kwargs"], (
        "an undeclared word must be REMOVED, not forwarded — this is what "
        "stops a typo becoming the template's own (possibly maximal) fallback")


def test_max_kept_when_the_map_says_max_to_max():
    m = _mock_self(reasoning_effort_map={"max": "max"})
    p = {"messages": [], "chat_template_kwargs": {"reasoning_effort": "max"}}
    m.apply_reasoning_effort_map(_req(p))
    assert p["chat_template_kwargs"]["reasoning_effort"] == "max"


def test_the_key_comparison_is_case_insensitive_and_stripped():
    m = _mock_self(reasoning_effort_map={"Medium": "high"})
    p = {"messages": [], "chat_template_kwargs": {"reasoning_effort": "  MEDIUM "}}
    m.apply_reasoning_effort_map(_req(p))
    assert p["chat_template_kwargs"]["reasoning_effort"] == "high", (
        "the mapped VALUE is sent verbatim as declared; only the KEY lookup "
        "is case/space insensitive")


# --------------------------------------------------------------------------- #
# The same three semantics, at the top-level OpenAI spelling — the location
# apply_thinking never touches for a caller who never opted into `thinking:`.
# --------------------------------------------------------------------------- #

def test_medium_is_remapped_to_high_at_the_top_level():
    m = _mock_self(reasoning_effort_map={"medium": "high"})
    p = {"messages": [], "reasoning_effort": "medium"}
    m.apply_reasoning_effort_map(_req(p))
    assert p["reasoning_effort"] == "high"


def test_a_typo_is_removed_at_the_top_level():
    m = _mock_self(reasoning_effort_map={"low": "low", "high": "high"})
    p = {"messages": [], "reasoning_effort": "mediumm"}
    m.apply_reasoning_effort_map(_req(p))
    assert "reasoning_effort" not in p


def test_both_locations_are_normalized_independently_in_one_call():
    """A payload carrying an effort in BOTH places (a malformed caller, or one
    hedging its bets across API dialects) gets each one checked on its own —
    neither location's outcome depends on the other's."""
    m = _mock_self(reasoning_effort_map={"medium": "high", "low": "low"})
    p = {
        "messages": [],
        "reasoning_effort": "medium",
        "chat_template_kwargs": {"reasoning_effort": "typo"},
    }
    m.apply_reasoning_effort_map(_req(p))
    assert p["reasoning_effort"] == "high"
    assert "reasoning_effort" not in p["chat_template_kwargs"]


# --------------------------------------------------------------------------- #
# Absence — a total no-op, at both call sites this method can be reached from.
# --------------------------------------------------------------------------- #

def test_an_undeclared_endpoint_is_completely_untouched():
    m = _mock_self(reasoning_effort_map={})  # {} = undeclared
    p = {
        "messages": [],
        "reasoning_effort": "medium",
        "chat_template_kwargs": {"reasoning_effort": "anything at all"},
    }
    before = {"reasoning_effort": p["reasoning_effort"],
             "ck": dict(p["chat_template_kwargs"])}
    m.apply_reasoning_effort_map(_req(p))
    assert p["reasoning_effort"] == before["reasoning_effort"]
    assert p["chat_template_kwargs"] == before["ck"]
    assert m.state.reasoning_effort_remaps == {}, (
        "an undeclared endpoint must not even be counted")


def test_absent_effort_fields_are_a_noop_even_when_the_map_is_declared():
    m = _mock_self(reasoning_effort_map={"medium": "high"})
    p = {"messages": [], "max_tokens": 100}
    m.apply_reasoning_effort_map(_req(p))
    assert p == {"messages": [], "max_tokens": 100}


def test_a_non_chat_completion_payload_type_is_untouched():
    m = _mock_self(reasoning_effort_map={"medium": "high"})
    p = {"messages": [], "reasoning_effort": "medium"}
    m.apply_reasoning_effort_map(_req(p, ptype="embedding"))
    assert p["reasoning_effort"] == "medium"


def test_the_callers_payload_dict_is_not_mutated_in_place():
    """`chat_template_kwargs` must be a fresh dict when written back, same
    corpus-capture contract every other correction here follows."""
    m = _mock_self(reasoning_effort_map={"medium": "high"})
    ck = {"reasoning_effort": "medium"}
    p = {"messages": [], "chat_template_kwargs": ck}
    m.apply_reasoning_effort_map(_req(p))
    assert ck["reasoning_effort"] == "medium", "the original ck dict is untouched"
    assert p["chat_template_kwargs"] is not ck
    assert p["chat_template_kwargs"]["reasoning_effort"] == "high"


# --------------------------------------------------------------------------- #
# The operator surface: a counter, keyed by endpoint/from->to.
# --------------------------------------------------------------------------- #

def test_a_remap_is_counted_by_endpoint_from_and_to():
    m = _mock_self(reasoning_effort_map={"medium": "high"})
    p = {"messages": [], "reasoning_effort": "medium"}
    m.apply_reasoning_effort_map(_req(p, endpoint="tier3"))
    row = m.state.reasoning_effort_remaps["tier3|medium->high"]
    assert row == {"endpoint": "tier3", "from": "medium", "to": "high", "count": 1}


def test_a_removal_is_counted_with_a_null_to():
    m = _mock_self(reasoning_effort_map={"low": "low"})
    p = {"messages": [], "reasoning_effort": "mediumm"}
    m.apply_reasoning_effort_map(_req(p, endpoint="tier3"))
    row = m.state.reasoning_effort_remaps["tier3|mediumm->"]
    assert row["to"] is None
    assert row["from"] == "mediumm"
    assert row["count"] == 1


def test_the_counter_accumulates_across_repeated_requests():
    m = _mock_self(reasoning_effort_map={"medium": "high"})
    m.apply_reasoning_effort_map(_req({"messages": [], "reasoning_effort": "medium"}))
    m.apply_reasoning_effort_map(_req({"messages": [], "reasoning_effort": "medium"}))
    row = m.state.reasoning_effort_remaps["tier3|medium->high"]
    assert row["count"] == 2


def test_the_log_dedup_set_is_keyed_by_caller_and_value():
    m = _mock_self(reasoning_effort_map={"medium": "high"})
    m.apply_reasoning_effort_map(
        _req({"messages": [], "reasoning_effort": "medium"}, agent_id="agent-a"))
    m.apply_reasoning_effort_map(
        _req({"messages": [], "reasoning_effort": "medium"}, agent_id="agent-b"))
    assert m.state.reasoning_effort_remap_logged == {
        ("agent-a", "medium"), ("agent-b", "medium")}


# --------------------------------------------------------------------------- #
# /v1/status carries the counter.
# --------------------------------------------------------------------------- #

def test_status_field_name_is_reasoning_effort_remaps():
    import inspect
    from roadstead import http_handlers
    src = inspect.getsource(http_handlers.ProxyHttpHandlers.handle_status)
    assert '"reasoning_effort_remaps": self.state.reasoning_effort_remaps' in src


# --------------------------------------------------------------------------- #
# Interaction with the EXISTING effort-injection policy (apply_thinking /
# apply_forced_reasoning_budget): the map is the LAST word, not a substitute.
# --------------------------------------------------------------------------- #

def test_the_map_overrides_an_endpoints_own_stale_declared_default():
    """The exact shape of the GLM incident: the endpoint's OWN configured
    `reasoning_effort` default is itself a word the map does not recognise
    (declared before the map existed, or simply wrong) — apply_thinking
    injects it as usual, and the map — running AFTER, as the final gate —
    still catches it. The guard protects against a stale operator
    declaration, not only a caller's typo."""
    m = _mock_self(thinking_kwargs=("thinking",), reasoning_effort="medium",
                  reasoning_effort_map={"low": "low", "high": "high"})
    p = {"messages": [], "max_tokens": 800, "thinking": True}
    m.apply_thinking(_req(p))
    assert p["chat_template_kwargs"]["reasoning_effort"] == "medium", (
        "apply_thinking must still inject the declared default — this test "
        "would be vacuous if it didn't")
    m.apply_reasoning_effort_map(_req(p))
    assert "reasoning_effort" not in p["chat_template_kwargs"], (
        "the map runs LAST and must still remove an unrecognised word even "
        "when it came from the endpoint's OWN declared default, not a caller")


def test_the_map_translates_an_endpoints_own_declared_default_when_it_is_named():
    """The harmonious case: the operator's default IS in the map, so the two
    policies compose rather than fight — apply_thinking's injected value is
    itself translated to the template's real spelling."""
    m = _mock_self(thinking_kwargs=("thinking",), reasoning_effort="medium",
                  reasoning_effort_map={"medium": "high"})
    p = {"messages": [], "max_tokens": 800, "thinking": True}
    m.apply_thinking(_req(p))
    m.apply_reasoning_effort_map(_req(p))
    assert p["chat_template_kwargs"]["reasoning_effort"] == "high"


def test_a_switch_bearing_endpoint_with_no_thinking_opt_in_still_gets_the_top_level_gate():
    """The gap neither existing injection site covers: a plain caller on a
    switch-bearing endpoint that never sends `thinking:` at all. apply_thinking
    bails immediately (no `want`), so chat_template_kwargs is never even
    built — only the top-level field this method also checks is at risk."""
    m = _mock_self(thinking_kwargs=("enable_thinking",),
                  reasoning_effort_map={"high": "high"})
    p = {"messages": [], "reasoning_effort": "medium"}
    m.apply_thinking(_req(p))
    assert "chat_template_kwargs" not in p, (
        "apply_thinking must have been a no-op here — this test would be "
        "vacuous otherwise")
    m.apply_reasoning_effort_map(_req(p))
    assert "reasoning_effort" not in p
