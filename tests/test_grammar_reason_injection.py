"""Tests for the CRANE-style reason-field injection in llmproxy/grammar.py.

Self-contained: loads grammar.py and the real production .gbnf files by path,
so it runs anywhere with just the stdlib + pytest (no package import / proxy
deps). The strongest assertion is that every injected grammar still passes the
proxy's own normalize_and_validate() — i.e. the backend would accept it rather
than silently drop it into free-form output.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# --- load grammar.py directly (no package import, no proxy deps) -------------
_REPO_C2 = Path(__file__).resolve().parents[2]              # <repo>/originfleet
_GRAMMAR_PY = _REPO_C2 / "originfleet" / "llmproxy" / "grammar.py"
_spec = importlib.util.spec_from_file_location("_llmproxy_grammar", _GRAMMAR_PY)
grammar = importlib.util.module_from_spec(_spec)
sys.modules["_llmproxy_grammar"] = grammar                  # dataclass needs the module registered
_spec.loader.exec_module(grammar)

inject_reason_field = grammar.inject_reason_field
normalize_and_validate = grammar.normalize_and_validate
root_object_keys = grammar.root_object_keys

# --- the real judgment grammars + their first (decision) key ----------------
_GRAMMARS = _REPO_C2 / "originfleet" / "agents"
JUDGMENT = {
    "merge_review":          (_GRAMMARS / "knowledge/grammars/merge_review.gbnf",          "merge"),
    "dedup_check":           (_GRAMMARS / "knowledge/grammars/dedup_check.gbnf",            "match"),
    "trip_resolve":          (_GRAMMARS / "temporal/grammars/trip_resolve.gbnf",           "is_trip"),
    "trip_business_classify":(_GRAMMARS / "temporal/grammars/trip_business_classify.gbnf", "business"),
    "flight_itinerary":      (_GRAMMARS / "mail-agent/grammars/flight_itinerary.gbnf",   "has_itinerary"),
    "proposal":              (_GRAMMARS / "forum-agent/grammars/proposal.gbnf",                "action"),
}
BARE_ENUM = _GRAMMARS / "knowledge/grammars/classify_document.gbnf"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", sorted(JUDGMENT))
@pytest.mark.parametrize("max_chars", [400, None])  # bounded (vLLM) + unbounded (llama.cpp)
def test_inject_is_valid_and_reason_first(name, max_chars):
    path, decision_key = JUDGMENT[name]
    src = _read(path)
    res = inject_reason_field(src, max_chars=max_chars)

    assert res.injected, f"{name}: expected injection into an object-root grammar"
    assert res.field == "reason"

    # The transformed grammar must still be accepted by the proxy's authority —
    # otherwise the backend would silently drop it (the failure mode we guard).
    result = normalize_and_validate(res.grammar)
    assert result.ok, f"{name}: injected grammar invalid: {result.error_payload()}"

    # `reason` is present and is the FIRST top-level key, before the decision.
    keys = root_object_keys(res.grammar)
    assert "reason" in keys, f"{name}: reason key missing from root ({keys})"
    assert keys[0] == "reason", f"{name}: reason not first ({keys})"
    assert decision_key in keys and keys.index("reason") < keys.index(decision_key)

    # Support rules emitted; bound matches the requested mode.
    assert "reasonstr ::=" in res.grammar
    assert ("rchar*" in res.grammar) == (max_chars is None)
    if max_chars is not None:
        assert "rchar{0,%d}" % max_chars in res.grammar


def test_bare_enum_root_is_skipped():
    src = _read(BARE_ENUM)                       # classify_document: root is a bare enum
    res = inject_reason_field(src)
    assert not res.injected
    assert res.grammar == src                    # untouched


def test_injection_is_idempotent():
    src = _read(JUDGMENT["merge_review"][0])
    once = inject_reason_field(src)
    twice = inject_reason_field(once.grammar)
    assert once.injected and not twice.injected   # second pass sees the field, skips
    assert twice.grammar == once.grammar


def test_merge_review_full_key_order():
    src = _read(JUDGMENT["merge_review"][0])
    res = inject_reason_field(src)
    assert root_object_keys(res.grammar) == ["reason", "merge", "rationale", "confidence"]


def test_rule_name_collision_is_avoided():
    # Synthetic object-root grammar that already defines `reasonstr`.
    src = (
        'root ::= "{" ws "\\"x\\"" ws ":" ws reasonstr ws "}"\n'
        'reasonstr ::= "\\"" rchar* "\\""\n'
        'rchar ::= [^"\\\\\\x00-\\x1f]\n'
        'ws ::= [ \\t\\n\\r]*\n'
    )
    res = inject_reason_field(src)
    assert res.injected
    assert "reasonstr2 ::=" in res.grammar        # collision-free name chosen
    assert normalize_and_validate(res.grammar).ok


def test_grammar_with_no_root_is_skipped():
    res = inject_reason_field('foo ::= "a" | "b"\n')
    assert not res.injected
