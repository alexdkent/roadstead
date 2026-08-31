"""Tests for the egress conformance check (grammar.verify_conformance) — the
core of the SHADOW silent-drop detector (service._shadow_egress_detect).

It catches the failure mode the proxy must surface: a backend that silently
DROPPED the grammar and ran free-form (markdown fence / non-JSON / wrong keys).
Self-contained — loads grammar.py by path so it runs with just the stdlib +
pytest (the proxy package's __init__ pulls heavy deps).

(The proxy-centralized reason-injection layer this file used to also cover was
removed 2026-06-14 — see config.py. verify_conformance is independent and stays.)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_C2 = Path(__file__).resolve().parents[1]              # <repo>/originfleet


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_C2 / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                                 # dataclass needs it registered
    spec.loader.exec_module(mod)
    return mod


grammar = _load("_sop_grammar", "originfleet/llmproxy/grammar.py")

PROPOSAL = _REPO_C2 / "originfleet/agents/forum-agent/grammars/proposal.gbnf"


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
