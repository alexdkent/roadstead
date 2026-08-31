"""WS1 — proxy GBNF authority: validate, normalize (safe), fail-loud.

The proxy must never dispatch a grammar that llama-server would silently
drop (parse-fail → unconstrained 200). These tests lock the two production
failure classes (newline mid-sequence, repetition over threshold), the
semantics-preserving normalization, and the fail-loud contract.
"""

from __future__ import annotations

import pathlib

import pytest

from roadstead.grammar import (
    MAX_REPETITION_THRESHOLD,
    normalize,
    normalize_and_validate,
    validate,
)

# A multi-line rule body with mid-sequence newlines — the extract_entities
# failure class. Valid language, but newlines are outside parens / not after |.
MULTILINE = (
    'root ::= "{" ws\n'
    '         "\\"a\\"" ws ":" ws string ws ","\n'
    '         ws "\\"b\\"" ws ":" ws string\n'
    '         ws "}"\n'
    'string ::= "\\"" "x" "\\""\n'
    'ws ::= [ \\t\\n]*\n'
)

# A grammar with the {0,2000} repetition (== MAX_REPETITION_THRESHOLD).
OVER_THRESHOLD = (
    'root ::= item ( ws "," ws item ){0,2000}\n'
    'item ::= "x"\n'
    'ws ::= [ \\t]*\n'
)

CLEAN = (
    'root ::= "[" ws ( item ( ws "," ws item )* )? ws "]"\n'
    'item ::= "\\"" "x" "\\""\n'
    'ws ::= [ \\t\\n]*\n'
)


# --- normalization: multi-line → parenthesized, semantics-preserving ---

def test_normalize_wraps_multiline_rule_in_parens():
    out = normalize(MULTILINE)
    # the root rule's body should now be wrapped so its newlines are legal
    root_block = out.split("string ::=")[0]
    assert "::= (" in root_block
    assert root_block.rstrip().endswith(")")


def test_normalize_leaves_singleline_untouched():
    src = 'root ::= "a" | "b"\n'
    assert normalize(src) == src


def test_normalize_idempotent():
    once = normalize(MULTILINE)
    twice = normalize(once)
    assert once == twice


def test_normalized_multiline_validates_clean():
    res = normalize_and_validate(MULTILINE)
    assert res.ok, res.error_payload()
    assert res.normalized is True


# --- validation: repetition threshold (fail-loud, not auto-fixed) ---

def test_repetition_over_threshold_fails():
    res = normalize_and_validate(OVER_THRESHOLD)
    assert not res.ok
    codes = [e.code for e in res.errors]
    assert "repetition_over_threshold" in codes


def test_repetition_just_under_threshold_ok():
    src = OVER_THRESHOLD.replace("{0,2000}", "{0,50}")
    res = normalize_and_validate(src)
    assert res.ok, res.error_payload()


def test_threshold_constant_matches_llamacpp():
    assert MAX_REPETITION_THRESHOLD == 2000


# --- validation: structural ---

def test_undefined_rule_fails():
    src = 'root ::= missing\n'
    errs = validate(src)
    assert any(e.code == "undefined_rule" for e in errs)


def test_underscore_rule_name_fails():
    src = 'root ::= my_rule\nmy_rule ::= "x"\n'
    errs = validate(src)
    assert any(e.code == "bad_rule_name" for e in errs)


def test_no_root_fails():
    src = 'start ::= "x"\n'
    errs = validate(src)
    assert any(e.code == "no_root" for e in errs)


def test_clean_grammar_passes():
    res = normalize_and_validate(CLEAN)
    assert res.ok, res.error_payload()


# --- tokenizer robustness: operators/# inside literals must not fool it ---

def test_hash_inside_string_not_comment():
    src = 'root ::= "a#b"\n'
    # `#b"` must not be treated as a comment that eats the closing quote
    res = normalize_and_validate(src)
    assert res.ok, res.error_payload()


def test_repeat_braces_inside_charclass_ok():
    src = 'root ::= [{}]+\n'
    res = normalize_and_validate(src)
    assert res.ok, res.error_payload()


# --- char-class \- escape: b9357 rejects it; normalize → literal hyphen ---

def test_backslash_hyphen_normalized_to_literal():
    # `\-` at end of class → drop backslash (already literal position)
    src = 'root ::= [a-z\\-]+\n'
    res = normalize_and_validate(src)
    assert res.ok, res.error_payload()
    assert "\\-" not in res.grammar
    assert res.normalized


def test_backslash_hyphen_midclass_moved_to_end():
    # `\-` in the middle → move literal hyphen to end (preserve intent,
    # avoid creating an unintended range)
    src = 'root ::= [a\\-z]+\n'
    res = normalize_and_validate(src)
    assert res.ok, res.error_payload()
    assert "\\-" not in res.grammar
    # must NOT have become a range a-z; the hyphen is literal at the end
    assert "[az-]" in res.grammar


def test_unknown_escape_other_than_hyphen_fails():
    src = 'root ::= [a\\q]+\n'
    errs = validate(src)
    assert any(e.code == "unknown_escape" for e in errs)


# --- corpus guard: every vendored grammar must normalize to valid ---
#
# This used to rglob the monorepo for *.gbnf and assert >=13 production agent
# grammars normalised cleanly. That scanned a private system's agent tree, which
# is not something Roadstead has any business knowing about — and it is not what
# the assertion is FOR. What it actually guards is that the validator stays
# usable against grammars of the shapes real callers send. So the corpus is now
# vendored in tests/corpus/grammars/, covering those shapes deliberately rather
# than by accident of what happened to be in the fleet that day.

CORPUS = pathlib.Path(__file__).resolve().parents[0] / "corpus" / "grammars"

#: Every shape the corpus is meant to cover. Named, so a deleted fixture fails
#: loudly here instead of quietly shrinking the sweep below.
EXPECTED_FIXTURES = {
    "proposal.gbnf",          # object root, closed enum, nested object
    "extract_entities.gbnf",  # array of objects
    "scored_answer.gbnf",     # mixed value types + an optional trailing field
    "bounded_list.gbnf",      # a legal bounded repetition
    "verdict_enum.gbnf",      # bare enum root
    "freeform_string.gbnf",   # genuinely non-object root
}


def test_the_corpus_is_present_and_complete():
    """🚨 Refuse to pass on an empty set.

    ``docs/ledger.md``, "a green suite that ran nothing": a structural check
    that computes its paths wrongly finds zero files and reports success because
    its loop body never runs. The sweep below is exactly that shape, so its
    input is asserted first — by NAME, not merely by count, so swapping one
    fixture for a duplicate of another cannot keep the count up while dropping
    a shape."""
    assert CORPUS.is_dir(), f"vendored grammar corpus missing at {CORPUS}"
    found = {p.name for p in CORPUS.glob("*.gbnf")}
    assert found == EXPECTED_FIXTURES, (
        f"corpus drift — missing: {sorted(EXPECTED_FIXTURES - found)}, "
        f"unexpected: {sorted(found - EXPECTED_FIXTURES)}")


def test_all_vendored_grammars_normalize_to_valid():
    """Every fixture must pass the proxy's normalize+validate. If a new or
    edited grammar can't be made b9357-valid, this fails — fix the grammar,
    don't loosen the validator."""
    gbnfs = sorted(CORPUS.glob("*.gbnf"))
    assert len(gbnfs) == len(EXPECTED_FIXTURES)  # see the guard above
    failures = []
    for f in gbnfs:
        res = normalize_and_validate(f.read_text())
        if not res.ok:
            failures.append((f.name, [e.code for e in res.errors]))
    assert not failures, f"grammars failed proxy validation: {failures}"


def test_normalization_is_idempotent_across_the_corpus():
    """normalize() runs on every dispatched grammar, so a second pass must be a
    no-op — otherwise a re-validated request could drift from what was sent."""
    for f in sorted(CORPUS.glob("*.gbnf")):
        once = normalize(f.read_text())
        assert normalize(once) == once, f"{f.name} is not normalize-stable"


# --- error payload shape (what the caller receives on fail-loud) ---

def test_error_payload_shape():
    res = normalize_and_validate(OVER_THRESHOLD)
    p = res.error_payload()
    assert p["error"] == "grammar_invalid"
    assert "detail" in p and p["errors"]
