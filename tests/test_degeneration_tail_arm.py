"""Unit tests for the egress degeneration guard's TAIL arm
(`_degen_tail` / `_blank_ratio` / `_distinct_gram_ratio` / `_degenerate_text_arm`
/ `_is_degenerate_text`).

The word-shingle arm (see test_degeneration_guard.py) is blind to a measured
tier3 shape: a valid JSON prefix followed by a loop of pure/near-pure
whitespace until max_tokens. `str.split()` yields zero words on a blank tail
(never reaches `_DEGEN_MIN_WORDS`), and even when it doesn't, a good prefix
dilutes a whole-body ratio — so this second arm looks only at the TAIL
(`_DEGEN_TAIL_CHARS`) at the character level. Calibrated over 12,899 real
completions (`proxy_completions`, bodies >= 200 chars, warm-up pings
excluded) — see the `_DEGEN_TAIL_*` comment block in roadstead/correction.py
for the measured distribution these thresholds sit inside.

Pinned here:
  - every measured degenerate shape (60000 spaces; a blank+newline loop; 500
    zero-width spaces; a mixed-whitespace loop; the a metadata-resolution caller semantic loop)
    is flagged, and by the arm that should catch it;
  - the highest measured legitimate blank ratio (~0.31) and a legitimate body
    near the measured distinct-gram floor (~0.192, long-form prose) both
    pass — the real margins, not a trivially-safe value;
  - ordinary varied prose passes;
  - a body under `_DEGEN_TAIL_MIN_CHARS` is never judged by this arm, however
    degenerate its tail;
  - the helpers are total (never raise) on None / '' / a body shorter than the
    gram.
"""

from __future__ import annotations

import itertools
import random
import string

from roadstead import correction


# --------------------------------------------------------------------------- #
# measured degenerate shapes
# --------------------------------------------------------------------------- #

def test_pure_spaces_60000():
    assert correction._is_degenerate_text(" " * 60000)
    assert correction._degenerate_text_arm(" " * 60000) == "blank_ratio"


def test_blank_newline_loop():
    text = "  \n\n" * 500
    assert correction._is_degenerate_text(text)
    assert correction._degenerate_text_arm(text) == "blank_ratio"


def test_zero_width_space_loop():
    # U+200B repeated — invisible, and str.split() treats the WHOLE run as a
    # single "word", so the word-shingle arm cannot see it at all.
    text = "​" * 500
    assert correction._is_degenerate_text(text)
    assert correction._degenerate_text_arm(text) == "blank_ratio"


def test_mixed_whitespace_loop():
    unit = "        \t\t                    "
    text = unit * 200
    assert correction._is_degenerate_text(text)
    assert correction._degenerate_text_arm(text) == "blank_ratio"


def test_semantic_loop_a_metadata_resolution_caller_shape():
    # ~0.06 distinct-24-gram ratio: a ~20-word sentence fragment repeating.
    # The real shape: a metadata-resolution caller, measured at 0.058.
    unit = ("me try that. responseI'll search for the release-group with "
            "the title 'Spring'... Let me try ")
    text = unit * 40
    ratio = correction._distinct_gram_ratio(text)
    assert ratio <= 0.08, f"fixture drifted off its target ratio: {ratio}"
    assert correction._is_degenerate_text(text)
    assert correction._degenerate_text_arm(text) == "distinct_gram"


# --------------------------------------------------------------------------- #
# negative cases — the real margins, not trivially-safe values
# --------------------------------------------------------------------------- #

def test_ordinary_prose_tail_passes():
    prose = (
        "The proxy resumed normal operation once the endpoint recovered from its "
        "brief outage. Every queued request drained within the configured "
        "deadline, and the health check returned green across all three "
        "replicas within about ninety seconds. An operator watching the "
        "dashboard saw queue depth fall steadily from its peak. Backend "
        "latency settled back near its usual baseline by the time the alert "
        "cleared. Nothing about the incident required a manual restart or "
        "any config change at all. The on-call engineer noted the timeline "
        "in the postmortem doc for later review. A follow-up ticket was "
        "filed to add a slightly earlier warning threshold. Traffic from "
        "the affected callers resumed without any visible client-side "
        "errors. The cache warmed back up within a couple of minutes of "
        "the recovery. Nobody paged twice, which the team took as a small "
        "but genuine win."
    )
    assert len(prose) >= correction._DEGEN_TAIL_MIN_CHARS
    assert not correction._is_degenerate_text(prose)
    assert correction._degenerate_text_arm(prose) is None


def test_highest_measured_legitimate_blank_ratio_passes():
    """~0.31 blank ratio — the highest LEGITIMATE value measured across the
    12,899-completion corpus (degenerate bodies sit at 1.00, a wide gap below
    the 0.90 threshold). Built from short, varied, non-repeating tokens (a
    tag/ID-list shape) so distinct-gram stays at 1.0 and only the blank-ratio
    arm is exercised."""
    random.seed(42)
    letters = string.ascii_lowercase
    codes2 = [a + b for a in letters for b in letters]
    codes3 = [a + b + ch for a in letters for b in letters for ch in "xyz"]
    pool = codes2 * 5 + codes3
    random.shuffle(pool)
    text = " ".join(itertools.islice(itertools.cycle(pool), 900))
    ratio = correction._blank_ratio(text)
    assert 0.25 <= ratio <= 0.35, f"fixture drifted off its target ratio: {ratio}"
    assert not correction._is_degenerate_text(text)


def test_legitimate_body_near_the_distinct_gram_floor_passes():
    """~0.192 distinct-24-gram ratio — the nearest LEGITIMATE body measured
    (long-form prose: real cinematic-description prose with recurring
    shot-transition phrasing, "The camera holds wide, then slowly pulls back
    as the song ends…"). Built to land NEAR that value, not trivially above
    it — that is the real margin over `_DEGEN_TAIL_MIN_DISTINCT`."""
    phrases = [
        "The camera holds wide, then slowly pulls back as the song ends.",
        "A soft light drifts across the empty stage, and the crowd falls silent.",
        "The frame tightens on her hands, steady on the worn guitar strings.",
        "Rain streaks the window as the melody fades into the hallway.",
        "The dancers scatter into shadow while the last chord rings out.",
        "A single spotlight narrows on the drummer's closing cymbal crash.",
    ]
    text = " ".join(itertools.islice(itertools.cycle(phrases), 40))
    ratio = correction._distinct_gram_ratio(text)
    assert 0.15 <= ratio <= 0.25, f"fixture drifted off its target ratio: {ratio}"
    assert not correction._is_degenerate_text(text)


def test_short_bodies_never_flagged_by_the_tail_arm():
    # Below _DEGEN_TAIL_MIN_CHARS — a blank tail this short still isn't judged
    # by this arm (and is too short to trip the word-shingle arm either).
    short_blank = " " * (correction._DEGEN_TAIL_MIN_CHARS - 1)
    assert not correction._is_degenerate_text(short_blank)
    assert correction._degenerate_text_arm(short_blank) is None


# --------------------------------------------------------------------------- #
# helper totality
# --------------------------------------------------------------------------- #

def test_helpers_are_total_on_none_and_empty():
    for fn in (correction._degen_tail, correction._blank_ratio,
               correction._distinct_gram_ratio):
        fn(None)
        fn("")


def test_helpers_are_total_on_a_body_shorter_than_the_gram():
    short = "x" * (correction._DEGEN_TAIL_GRAM - 1)
    assert correction._degen_tail(short) == short
    assert correction._blank_ratio(short) == 0.0
    assert correction._distinct_gram_ratio(short) == 1.0  # "diverse", not degenerate


def test_degen_tail_windows_to_the_last_n_chars():
    text = "a" * 5000 + "b" * 100
    tail = correction._degen_tail(text)
    assert len(tail) == correction._DEGEN_TAIL_CHARS
    assert tail.endswith("b" * 100)


# --- threshold recalibration + evidence reporting (2026-09-06, second pass) ---
#
# These pin the two changes made after the first implementation: the
# distinct-gram threshold moved 0.10 -> 0.08 on new measurement, and the
# verdict now carries its EVIDENCE so a caller can check the claim.

import pytest  # noqa: E402

from roadstead.correction import (  # noqa: E402
    _DEGEN_TAIL_MIN_DISTINCT,
    _degeneracy_evidence,
    _fmt_evidence,
)


def _tuned(target: float, n: int = 2400) -> str:
    """A body whose tail distinct-24-gram ratio lands near `target`."""
    import random
    rng = random.Random(1234)
    words = [f"w{rng.randrange(int(4 + 300 * target))}" for _ in range(n)]
    return "start " + " ".join(words)


def test_threshold_is_pinned_at_the_measured_value():
    """0.08, not 0.10 and NOT the 0.15 an older doc recommends: that caller's
    measured WORST legitimate body is 0.141, so 0.15 would flag real work."""
    assert _DEGEN_TAIL_MIN_DISTINCT == 0.08
    assert _DEGEN_TAIL_MIN_DISTINCT < 0.141, (
        "threshold must sit below the nearest legitimate caller's measured worst case"
    )
    assert _DEGEN_TAIL_MIN_DISTINCT > 0.058, (
        "threshold must still catch the measured semantic loop at 0.058"
    )


def test_nearest_legitimate_body_is_not_flagged():
    """A body at the nearest legitimate caller's measured floor (0.141) must PASS.
    A false positive tells that caller 'do not raise your budget', which would
    break its legitimate 25,920 -> 46,656 truncation-recovery escalation."""
    from roadstead.correction import _distinct_gram_ratio, _is_degenerate_text
    body = _tuned(0.141)
    ratio = _distinct_gram_ratio(body)
    assert ratio > _DEGEN_TAIL_MIN_DISTINCT, f"built body measured {ratio}"
    assert not _is_degenerate_text(body)


def test_evidence_reports_measured_values_for_a_real_loop():
    ev = _degeneracy_evidence(" " * 60000)
    assert ev["measurable"] is True
    assert ev["blank_ratio"] == 1.0
    assert ev["distinct_gram"] is not None
    rendered = _fmt_evidence(ev)
    assert "blank_ratio=1.000" in rendered
    assert "n/a" not in rendered


@pytest.mark.parametrize("body", [None, "", "x" * 50])
def test_unmeasurable_evidence_is_none_never_zero(body):
    """🚨 The DECISION fails open; the EVIDENCE must refuse to assert.
    A blank_ratio of 0.00 on an unreadable body would say 'measured, healthy'
    — a confidently wrong number is worse than the label it replaced, because
    it looks checkable."""
    ev = _degeneracy_evidence(body)
    assert ev["measurable"] is False
    assert ev["blank_ratio"] is None
    assert ev["distinct_gram"] is None
    rendered = _fmt_evidence(ev)
    assert "blank_ratio=n/a" in rendered and "distinct_24gram=n/a" in rendered
    assert "0.00" not in rendered


def test_evidence_helpers_are_total():
    for bad in (None, "", 12345, object()):
        ev = _degeneracy_evidence(bad)
        assert isinstance(ev, dict) and "measurable" in ev
        assert isinstance(_fmt_evidence(ev), str)
    assert isinstance(_fmt_evidence({}), str)
    assert isinstance(_fmt_evidence(None), str)
