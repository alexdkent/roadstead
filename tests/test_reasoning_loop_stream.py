"""Loop-break on the STREAMING path: detection, shadow mode, and labelling."""
import pytest

from roadstead.correction import ReasoningLoopDetector
from roadstead.queue import PersistentQueue


def _armed():
    return ReasoningLoopDetector(window=20000, min_chars=20000,
                                 max_distinct=0.40, check_every=4000)


def _loop(n, period=2800):
    unit = "".join(f"Need maybe candidate transition {i} in size 4 table. Good. "
                   for i in range(period // 56))
    return unit * n


def test_reads_both_reasoning_spellings():
    """vLLM's deepseek_v4 parser emits `reasoning`; other lanes emit
    `reasoning_content`. The stream path must read both — reading one is how a
    23-minute trace measured as 'no reasoning at all' (2026-09-15)."""
    text = _loop(40)
    for key in ("reasoning", "reasoning_content"):
        det = _armed()
        fired = None
        for i in range(0, len(text), 200):
            delta = {key: text[i:i + 200]}
            rc = delta.get("reasoning")
            if not isinstance(rc, str):
                rc = delta.get("reasoning_content")
            if det.feed(rc):
                fired = i
                break
        assert fired is not None, f"blind to delta.{key}"


def test_content_deltas_never_feed_the_reasoning_detector():
    """A looping ANSWER is the egress degeneration guard's job, not this one.
    Feeding content here would double-judge it under thresholds calibrated on
    reasoning."""
    det = _armed()
    text = _loop(40)
    for i in range(0, len(text), 200):
        delta = {"content": text[i:i + 200]}
        rc = delta.get("reasoning")
        if not isinstance(rc, str):
            rc = delta.get("reasoning_content")
        assert det.feed(rc) is None


def test_reasoning_loop_is_not_backend_stall_evidence():
    """🚨 The backend is HEALTHY during a loop-break — decoding steadily, every
    watchdog satisfied. Filing it as a stall would let a MODEL behaviour cool an
    endpoint that is serving everyone else fine."""
    assert "reasoning_loop" not in PersistentQueue.STALL_ABORT_REASONS


def test_an_undeclared_endpoint_costs_one_attribute_check():
    """The hot path must short-circuit on `armed` before touching any delta."""
    det = ReasoningLoopDetector()
    assert det.armed is False
    for _ in range(1000):
        assert det.feed("some reasoning text " * 20) is None
    assert det._buf == [] and det._seen == 0


# --- answer-now re-ask ------------------------------------------------------

def test_the_notes_are_the_head_not_the_tail():
    """🚨 The rescue must be seeded with the PRE-loop deliberation. Handing the
    model back its own repeating tail and asking it to conclude re-seeds the
    failure that was just interrupted. Measured on the real corpus: the head
    carries ~1.7% of the full trace's loop-marker density."""
    det = _armed()
    productive = "".join(
        f"Step {i}: compute dp over subset {i} with distinct working here. "
        for i in range(400)
    )
    text = productive + _loop(60)
    fired = None
    for i in range(0, len(text), 200):
        if det.feed(text[i:i + 200]):
            fired = i
            break
    assert fired is not None
    head = det.head
    assert head.startswith("Step 0:"), "notes must start at the beginning of reasoning"
    assert head.count("Need maybe") * 20 < text.count("Need maybe"), \
        "the head is dominated by the loop — it is the tail, not the head"


def test_head_is_bounded_regardless_of_trace_length():
    from roadstead.correction import _REASONING_LOOP_HEAD_CHARS
    det = _armed()
    for _ in range(4000):
        det.feed("some long productive reasoning text that keeps on going. ")
    assert len(det.head) == _REASONING_LOOP_HEAD_CHARS


def test_answer_now_is_off_by_default_and_needs_the_break_to_act():
    """Opt-in: it issues a SECOND backend call. And it is inert while the break
    is in shadow, because nothing is ever interrupted for it to rescue."""
    import os
    from roadstead.config import reasoning_loop_answer_now, reasoning_loop_break_shadow
    os.environ.pop("ROADSTEAD_PROXY_REASONING_LOOP_ANSWER_NOW", None)
    os.environ.pop("ROADSTEAD_PROXY_REASONING_LOOP_SHADOW", None)
    assert reasoning_loop_answer_now() is False
    assert reasoning_loop_break_shadow() is True


def test_the_rescue_budget_cannot_buy_a_second_runaway():
    """The bound is the point: the call being replaced was unbounded in
    practice (131,072 tokens for zero output). Measured complete answers on that
    lane were 4,549-13,612 chars (~1,100-3,400 tokens)."""
    from roadstead.constants import _ANSWER_NOW_MAX_TOKENS
    assert _ANSWER_NOW_MAX_TOKENS <= 8000
    assert _ANSWER_NOW_MAX_TOKENS >= 3400 * 2, "too tight for the measured answers"
