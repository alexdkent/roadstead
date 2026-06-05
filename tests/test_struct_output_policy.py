"""Tests for the proxy-centralized structured-output policy layer:
the reason-then-constrain transform (grammar.py) + the per-call_site policy
registry (config.py). Self-contained — loads the modules by path so it runs
with just the stdlib + pytest (the proxy package's __init__ pulls heavy deps).

The service-level wiring (_apply_struct_policy / _finalize_struct_output) is
exercised by the in-container suite; here we lock down the pure logic + the
end-to-end inject→strip→verify roundtrip that the wiring depends on, and the
critical safety property: the registry ships OFF for every call_site.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_C2 = Path(__file__).resolve().parents[2]              # <repo>/originfleet


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_C2 / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                                 # dataclass needs it registered
    spec.loader.exec_module(mod)
    return mod


grammar = _load("_sop_grammar", "originfleet/llmproxy/grammar.py")
config = _load("_sop_config", "originfleet/llmproxy/config.py")

PROPOSAL = _REPO_C2 / "originfleet/agents/forum-agent/grammars/proposal.gbnf"


# --- policy registry: the critical safety property is "ships OFF" ------------

def test_registry_ships_off_for_every_call_site():
    for cs, pol in config.STRUCTURED_OUTPUT_POLICY.items():
        assert pol.mode == config.StructMode.OFF, f"{cs} must ship OFF"
        assert pol.call_site == cs
        assert pol.kind == config.StructKind.JUDGMENT


def test_policy_lookup_unknown_is_none():
    assert config.struct_policy_for("some.random_call_site") is None
    assert config.struct_policy_for("forum-agent.proposal_emitter") is not None


def test_six_judgment_call_sites_present():
    for cs in ("forum-agent.proposal_emitter", "knowledge.dedup_check",
               "knowledge.hygiene_monthly_merge", "temporal.trip_resolve",
               "temporal.trip_business_classify", "mail-agent.calendar_flight_extract"):
        assert cs in config.STRUCTURED_OUTPUT_POLICY


# --- strip_top_field ---------------------------------------------------------

def test_strip_top_field_removes_reason():
    out, removed = grammar.strip_top_field('{"reason":"because X","merge":true}', "reason")
    assert removed
    import json
    assert json.loads(out) == {"merge": True}


def test_strip_top_field_absent_is_noop():
    out, removed = grammar.strip_top_field('{"merge":true}', "reason")
    assert not removed and out == '{"merge":true}'


def test_strip_top_field_bad_json_is_noop():
    out, removed = grammar.strip_top_field("not json", "reason")
    assert not removed and out == "not json"


# --- verify_conformance ------------------------------------------------------

def test_verify_conformance_ok():
    g = PROPOSAL.read_text()
    ok, why = grammar.verify_conformance(
        '{"action":"comment","params":{},"why":"x"}', g, forbid_field="reason")
    assert ok, why


def test_verify_conformance_rejects_fenced():
    g = PROPOSAL.read_text()
    ok, why = grammar.verify_conformance('```json\n{"action":"comment"}', g)
    assert not ok and "fenced" in why


def test_verify_conformance_rejects_leaked_field():
    g = PROPOSAL.read_text()
    ok, why = grammar.verify_conformance(
        '{"reason":"leaked","action":"comment","params":{},"why":"x"}', g,
        forbid_field="reason")
    assert not ok and "reason" in why


def test_verify_conformance_rejects_unexpected_keys():
    g = PROPOSAL.read_text()
    ok, why = grammar.verify_conformance('{"bogus":1}', g)
    assert not ok and "unexpected_keys" in why


def test_verify_conformance_rejects_non_json():
    g = PROPOSAL.read_text()
    ok, why = grammar.verify_conformance("Here is my answer: comment", g)
    assert not ok and "not_json" in why


# --- the full roundtrip the proxy relies on ----------------------------------

def test_inject_then_strip_then_verify_roundtrip():
    """Inject reason → model emits {reason, ...decision} → strip → the result
    validates against the caller's ORIGINAL grammar (no leaked reason)."""
    original = PROPOSAL.read_text()
    res = grammar.inject_reason_field(original, max_chars=400)
    assert res.injected
    # the model's output under the INJECTED grammar (reason first, then payload)
    model_output = '{"reason":"this submolt fits and the draft is on-thesis","action":"comment","params":{"text":"nice"},"why":"on topic"}'
    stripped, removed = grammar.strip_top_field(model_output, res.field)
    assert removed
    ok, why = grammar.verify_conformance(stripped, original, forbid_field=res.field)
    assert ok, why
    import json
    assert "reason" not in json.loads(stripped)
    assert json.loads(stripped)["action"] == "comment"
