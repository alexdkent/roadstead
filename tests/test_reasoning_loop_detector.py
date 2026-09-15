"""ReasoningLoopDetector — the long-period reasoning-loop arm.

Calibrated 2026-09-15 against 14 real tier3 (DeepSeek-V4-Flash) traces captured at
the top reasoning rung: 5 runaway loops (which burned the whole max_tokens and
returned ZERO answer chars, one for 1h04m) and 9 healthy bodies.
"""
import pytest

from roadstead.correction import (
    ReasoningLoopDetector,
    distinct_gram_ratio_over,
    _REASONING_LOOP_MAX_DISTINCT,
    _REASONING_LOOP_WINDOW_CHARS,
    _DEGEN_TAIL_CHARS,
)

# The measured corpus, as numbers rather than 2 MB of fixtures. Each value is the
# distinct-24-gram ratio over a 20,000-char window, at the checkpoint where the
# detector would decide. Recomputing these needs the traces; pinning them here
# means a threshold change has to CONFRONT them rather than route around them.
RUNAWAY_AT_FIRE = [0.282, 0.274, 0.289, 0.276, 0.081]   # worst (highest) = 0.289
HEALTHY_FLOOR = [0.904, 0.935, 0.921, 0.933, 0.909, 0.891]  # lowest = 0.891
# The SAME five runaways measured by the incumbent 2,000-char arm — all read as
# healthy, which is why this arm exists rather than a retuned threshold.
RUNAWAY_AT_INCUMBENT_WINDOW = [0.709, 0.898, 0.956, 0.857, 0.507]


def _armed(**kw):
    base = dict(window=_REASONING_LOOP_WINDOW_CHARS, min_chars=20000,
                max_distinct=_REASONING_LOOP_MAX_DISTINCT, check_every=4000)
    base.update(kw)
    return ReasoningLoopDetector(**base)


def _loop_text(n_cycles: int, period: int = 2800) -> str:
    """A long-period verbatim cycle — the measured tier3 shape (period
    2,292-2,957 chars, repeated 6-8x)."""
    unit = "".join(f"Need maybe candidate transition {i} in size 4 table. Good. "
                   for i in range(period // 56))
    return unit * n_cycles


def _diverse_text(n: int) -> str:
    import random
    r = random.Random(11)
    words = [f"{r.choice('abcdefghijklmnop')}{r.randrange(10**6)}" for _ in range(n // 8)]
    return " ".join(words)


def _stream(det, text, chunk=200):
    for i in range(0, len(text), chunk):
        if det.feed(text[i:i + chunk]):
            return i + chunk
    return None


def test_fires_on_a_long_period_loop():
    assert _stream(_armed(), _loop_text(40)) is not None


def test_does_not_fire_on_diverse_reasoning():
    assert _stream(_armed(), _diverse_text(300_000)) is None


def test_disarmed_by_default_is_completely_inert():
    """Absent declaration => byte-identical behaviour to before this existed."""
    det = ReasoningLoopDetector()
    assert det.armed is False
    assert _stream(det, _loop_text(40)) is None
    assert det.verdict is None


@pytest.mark.parametrize("kw", [
    {"window": 0}, {"min_chars": 0}, {"max_distinct": 0.0}, {"max_distinct": 1.0},
])
def test_a_partial_declaration_is_inert_not_half_armed(kw):
    """Every threshold must be positively declared. A partial declaration must
    disarm rather than silently fall back to a default — a half-configured guard
    that fires on a default nobody chose is worse than one that does not run."""
    det = _armed(**kw)
    assert det.armed is False
    assert _stream(det, _loop_text(40)) is None


def test_survives_buffer_trims__regression():
    """🚨 REGRESSION (bug shipped and caught in review, 2026-09-15). `_seen` and
    the buffer length were ONE counter. The window trim reset it, so it never
    again reached `_next_check` and the detector went permanently silent after
    the first trim — it caught 1 of 5 known loops while reporting ZERO false
    positives, which is indistinguishable from a working guard.

    This text is >10x the window, so it forces many trims BEFORE the loop
    establishes. The pre-fix code returns None here."""
    det = _armed()
    text = _diverse_text(260_000) + _loop_text(40)
    assert _stream(det, text) is not None, "detector went silent after a buffer trim"


def test_the_incumbent_window_is_blind_to_this_shape():
    """The reason this is a new arm and not a retuned threshold: at the egress
    arm's 2,000-char window every one of these loops reads HEALTHY, because the
    window is SMALLER than the 2,292-2,957-char cycle period."""
    assert _DEGEN_TAIL_CHARS < 2292
    loop = _loop_text(40)
    assert distinct_gram_ratio_over(loop, _DEGEN_TAIL_CHARS) > _REASONING_LOOP_MAX_DISTINCT
    assert distinct_gram_ratio_over(loop, _REASONING_LOOP_WINDOW_CHARS) <= _REASONING_LOOP_MAX_DISTINCT
    assert min(RUNAWAY_AT_INCUMBENT_WINDOW) > _REASONING_LOOP_MAX_DISTINCT


def test_threshold_separates_the_measured_corpus_with_margin():
    """The calibration, pinned. Worst runaway 0.289 < 0.40 < lowest healthy 0.891."""
    assert max(RUNAWAY_AT_FIRE) < _REASONING_LOOP_MAX_DISTINCT < min(HEALTHY_FLOOR)
    assert _REASONING_LOOP_MAX_DISTINCT - max(RUNAWAY_AT_FIRE) >= 0.10
    assert min(HEALTHY_FLOOR) - _REASONING_LOOP_MAX_DISTINCT >= 0.10
    # Deliberately nearer the runaway end than the midpoint: a false positive
    # destroys a caller's legitimate long deliberation; a false negative costs
    # one more runaway the max_tokens ceiling ends anyway.
    midpoint = (max(RUNAWAY_AT_FIRE) + min(HEALTHY_FLOOR)) / 2
    assert _REASONING_LOOP_MAX_DISTINCT < midpoint


def test_cannot_judge_a_body_shorter_than_min_chars():
    """Safety property, not performance: production reasoning on this lane
    averages ~413 output tokens (~1,800 chars), an order of magnitude under
    min_chars, so ordinary traffic is structurally out of reach."""
    det = _armed()
    assert _stream(det, _loop_text(40)[:19_000]) is None


def test_is_fail_open_and_idempotent():
    det = _armed()
    for bad in (None, "", 0, b"bytes", object()):
        assert det.feed(bad) is None          # never raises
    fired_at = _stream(det, _loop_text(40))
    assert fired_at is not None
    first = det.verdict
    assert det.feed("more text") is None      # idempotent after firing
    assert det.verdict is first


def test_memory_is_bounded_by_the_window():
    det = _armed()
    _stream(det, _diverse_text(400_000))
    assert len("".join(det._buf)) <= _REASONING_LOOP_WINDOW_CHARS * 2
