"""Tests for the egress conformance check (grammar.verify_conformance) — the
core of the SHADOW silent-drop detector (service._shadow_egress_detect).

It catches the failure mode the proxy must surface: a backend that silently
DROPPED the grammar and ran free-form (markdown fence / non-JSON / wrong keys).
(The proxy-centralized reason-injection layer this file used to also cover was
removed 2026-06-14 — see config.py. verify_conformance is independent and stays.)

The grammar is now ``tests/corpus/grammars/proposal.gbnf``, vendored on
2026-08-31. It previously read a fleet agent's own grammar file out of the
monorepo, which is what made this file unrunnable standalone — and was never
the point: these tests are about conformance CHECKING, not about that agent.
The vendored fixture reproduces all five outcomes below exactly.

The by-path module loading that used to sit here is gone too. It existed to
import ``grammar.py`` without pulling the package's heavy dependencies; the
package now installs cleanly, so a plain import is both simpler and a better
test — it exercises the module as callers actually get it.
"""
from __future__ import annotations

from pathlib import Path

from roadstead import grammar

PROPOSAL = Path(__file__).resolve().parents[0] / "corpus" / "grammars" / "proposal.gbnf"


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


def test_the_vendored_grammar_is_the_shape_these_tests_assume():
    """Guard the fixture, not just the code. Every assertion above depends on
    ``proposal.gbnf`` having an object root with exactly these three keys — if
    someone edits it, the tests below would start passing for the wrong reason
    (an unexpected-keys check against the wrong key set still 'fails' correctly,
    and the ok case would start failing mysteriously)."""
    assert PROPOSAL.exists(), f"vendored fixture missing at {PROPOSAL}"
    assert grammar.root_object_keys(PROPOSAL.read_text()) == [
        "action", "params", "why"]
    res = grammar.normalize_and_validate(PROPOSAL.read_text())
    assert res.ok, [e.code for e in res.errors]


def test_non_object_root_only_gets_the_fence_and_forbid_checks():
    """The other branch of verify_conformance. A bare-token root cannot be
    structurally validated, so it must return ok rather than guessing — but the
    two checks that DO apply must still fire."""
    g = (PROPOSAL.parent / "freeform_string.gbnf").read_text()
    assert grammar.root_object_keys(g) == []
    ok, why = grammar.verify_conformance('"just prose"', g)
    assert ok and why == "ok_non_object_root"
    ok, why = grammar.verify_conformance('"has reason inside"', g,
                                         forbid_field="reason")
    assert not ok and "reason" in why
    ok, why = grammar.verify_conformance("```json", g)
    assert not ok and "fenced" in why
