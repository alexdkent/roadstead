"""StructuredBlankRunDetector — the structured CONTENT blank-run arm.

Calibrated against 2,453 real tier3 (GLM-5.3-Flash) structured completions
(2 days): the longest LEGITIMATE whitespace run in content was 37 chars, every
runaway measured >= 7,194. The four synthetic shapes below reproduce the real
runaway character-class MIXES (irregular — no fixed `stop` string catches
them), at a scale small enough to run fast in CI while still clearing the
default 512-char threshold by a wide margin.
"""
import json

import pytest

from roadstead.correction import (
    StructuredBlankRunDetector,
    strip_trailing_blank_chars,
)

_DEFAULT_THRESHOLD = 512


def _armed(threshold: int = _DEFAULT_THRESHOLD) -> StructuredBlankRunDetector:
    return StructuredBlankRunDetector(threshold=threshold)


def _feed_all(det: StructuredBlankRunDetector, text: str, chunk: int = 40):
    """Feed `text` in small pieces (the real streaming shape) and return the
    first verdict, or None if it never fired."""
    for i in range(0, len(text), chunk):
        v = det.feed(text[i:i + chunk])
        if v is not None:
            return v
    return None


# --------------------------------------------------------------------------- #
# the four measured runaway shapes
# --------------------------------------------------------------------------- #

def test_fires_on_shape_one_right_after_opening_brace():
    """'  \\n\\n\\n\\n  \\n\\n  \\n\\n  \\n\\n …' (measured 7,194 chars, starts
    right after the opening '{')."""
    unit = "  \n\n\n\n  \n\n  \n\n  \n\n"
    body = "{" + unit * (7200 // len(unit))
    v = _feed_all(_armed(), body)
    assert v is not None
    assert v["run_chars"] >= _DEFAULT_THRESHOLD


def test_fires_on_shape_two_mid_object_after_a_string_value():
    """'\\n   \\n   \\n   \\n\\n\\n\\n\\n   \\n\\n\\n\\n\\n   …' (measured
    14,636 / 12,060 chars, mid-object after a complete string value)."""
    unit = "\n   \n   \n   \n\n\n\n\n   \n\n\n\n\n   "
    body = '{"a": "done", "b": ' + unit * (12100 // len(unit))
    v = _feed_all(_armed(), body)
    assert v is not None


def test_fires_on_shape_three_mid_object():
    """' \\n\\n \\n  \\n \\n\\n\\n\\n\\n\\n\\n\\n…' (measured 22,404 chars,
    mid-object)."""
    unit = " \n\n \n  \n \n\n\n\n\n\n\n\n"
    body = '{"a": 1, "b": ' + unit * (22400 // len(unit))
    v = _feed_all(_armed(), body)
    assert v is not None


def test_fires_on_shape_four_after_a_complete_closed_object():
    """'\\t\\r\\t\\r\\t\\r…' (measured 11,769 chars) AFTER a complete, CLOSED
    JSON object — the grammar allows trailing whitespace and the object was
    already done. The detector doesn't know that; it only sees the run."""
    body = '{"a": [1, 2, 3]}' + ("\t\r" * (11800 // 2))
    v = _feed_all(_armed(), body)
    assert v is not None


# --------------------------------------------------------------------------- #
# the legitimate case — must NOT fire
# --------------------------------------------------------------------------- #

def test_does_not_fire_on_legitimate_37_char_indentation_run():
    """The measured legitimate ceiling: a 37-char indentation run inside
    otherwise-normal structured content, at the DEFAULT threshold (512) —
    this must never fire, or every legitimate long body id at risk."""
    body = '{"a": 1,' + (" " * 37) + '"b": 2}'
    assert _feed_all(_armed(), body) is None


def test_a_37_char_run_does_fire_at_a_tighter_threshold():
    """Proves the detector actually counts the run length (not a no-op) —
    the same 37-char run fires once the threshold is set below it."""
    body = '{"a": 1,' + (" " * 37) + '"b": 2}'
    assert _feed_all(_armed(threshold=30), body) is not None


# --------------------------------------------------------------------------- #
# detector mechanics
# --------------------------------------------------------------------------- #

def test_inert_when_threshold_is_zero():
    det = StructuredBlankRunDetector()
    assert det.armed is False
    assert det.feed(" " * 10000) is None


def test_run_is_trailing_not_cumulative():
    """Two separate 300-char blank runs, split by real content, must NOT sum
    to 600 and fire at a 512 threshold — each run resets on a non-blank char."""
    det = _armed(threshold=512)
    assert det.feed(" " * 300) is None
    assert det.feed("x") is None  # resets the run
    assert det.feed(" " * 300) is None


def test_idempotent_after_firing():
    det = _armed(threshold=10)
    first = det.feed(" " * 20)
    assert first is not None
    # Continuing to feed after the verdict must never raise or replace it.
    assert det.feed(" " * 20) is None
    assert det.verdict == first


def test_fail_open_on_non_string_delta():
    det = _armed(threshold=10)
    assert det.feed(None) is None
    assert det.feed(123) is None
    assert det.feed([]) is None
    # And the run counter must be untouched by the bad input.
    assert det.feed(" " * 9) is None
    assert det.feed(" ") is not None


def test_verdict_carries_evidence():
    """Fires the INSTANT the run reaches threshold — mid-delta, not after the
    whole delta is consumed, which is what minimizes wasted decode."""
    det = _armed(threshold=50)
    v = det.feed(" " * 60)
    assert v == {"run_chars": 50, "threshold": 50, "content_chars": 60}


# --------------------------------------------------------------------------- #
# strip_trailing_blank_chars — the salvage check's own strip
# --------------------------------------------------------------------------- #

def test_strip_trailing_blank_chars_only_strips_the_tail():
    stripped = strip_trailing_blank_chars('  {"a": 1}' + "\t\r" * 50)
    assert stripped == '  {"a": 1}'
    assert json.loads(stripped) == {"a": 1}


def test_strip_trailing_blank_chars_leaves_leading_whitespace():
    """Leading whitespace must survive the strip — json.loads tolerates it,
    and quietly removing it would accept a body that never had a leading
    '{' at all."""
    assert strip_trailing_blank_chars("   " + "x") == "   x"


def test_strip_trailing_blank_chars_is_total():
    assert strip_trailing_blank_chars("") == ""
    assert strip_trailing_blank_chars(None) == ""  # type: ignore[arg-type]
