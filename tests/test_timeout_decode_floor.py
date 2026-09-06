"""The DECODE-RATE floor: ``recommended`` must never sit below the physical
decode time of the output being asked for.

Sibling of ``test_timeout_model_monotonic.py`` (regression ledger
`llm-output-budget-starvation`, an extraction leftover — this repo has no
access to that ledger, but the reasoning it points at is live and the code
comment above ``decode_rate_floor_ms`` in ``timeout_model.py`` cross-references
it). That guard stopped the advice INVERTING as ``est_out`` grew. This is the
same trap one rung up: the advice can be ABSOLUTELY too small, and no amount of
empirical sampling can fix it, because a call that always times out can never
produce the ``status == "ok"`` samples ``record()`` requires to correct it.

THE DEFECT
==========
Measured on a fleet instance (2026-09-06), ``proxy_completions`` (status='ok',
output_tokens>=500, last 14 days): a dense 27B endpoint ("tier2-analyst" here)
decodes at ~26.6 tok/s and has never — in that window — completed an output
larger than 9,650 tokens. A caller asked it for 12,000 output tokens (~451s of
decode alone) and ``advise()`` returned 352s: a deadline the call could not
meet at ANY load, because nothing but decode time stood between it and the
timeout. Six consecutive jobs died there. The 12k bucket cannot fill itself,
because every one of those jobs is a ``status != "ok"`` sample ``record()``
discards.

THE FIX
=======
``decode_rate_floor_ms(est_out, decode_tok_s)`` = ``(est_out / decode_tok_s) *
margin``, bounded by ``_BACKGROUND_CEILING_S``. ``TimeoutModel.advise()``
applies ``max(recommended, floor)`` with it after the monotonicity guard's
lattice walk. The rate comes from ``models.yaml`` ``token_speed``
(``model_catalog.build_class_decode_rates``) — a fleet measurement, never a
number invented here — so a class nobody profiled gets floor 0.0, i.e. no
floor, i.e. unchanged behaviour.
"""
from __future__ import annotations

import pytest

from roadstead.timeout_model import (
    TimeoutModel,
    _BACKGROUND_CEILING_S,
    _DECODE_FLOOR_MARGIN,
    decode_rate_floor_ms,
)

_NOW = 1000.0
_TIER2_ANALYST_TOK_S = 26.6  # measured, see module docstring


def _model(**kw) -> TimeoutModel:
    return TimeoutModel(margin=1.0, min_samples=30,
                         floors={"tier2-analyst": 120.0}, **kw)


def _seed_the_reported_shape(m: TimeoutModel) -> None:
    """40 completed 9,000-token calls at exactly the reported latency.

    9,000 and 12,000 share the top out-bucket (``_OUT_EDGES[-1]`` is 8,192),
    so the query resolves at ``source == "cell"`` and ``recommended_ms`` is
    exactly the seeded p99 x margin(1.0) = 352,000ms — the live number,
    reproduced rather than asserted against a magic constant."""
    for _ in range(40):
        m.record("tier2-analyst", 3, 4_000, 9_000, 352_000.0, "ok", _NOW)


# ---------------------------------------------------------------------------
# The reported case
# ---------------------------------------------------------------------------

def test_the_reported_case_reproduces_a_sub_decode_recommendation_unfloored():
    """Guard on the guard: prove the empirical answer really is ~352s without
    the floor, so the test below cannot pass vacuously. Queries ``advise()``
    directly (not a mock of it) with no ``decode_rates`` configured — this is
    the "layer below the floor" the samples exercise, not the floor itself."""
    m = _model()
    _seed_the_reported_shape(m)
    unfloored = m.advise("tier2-analyst", 3, 4_000, 12_000)
    assert unfloored["source"] == "cell", unfloored["source"]
    assert unfloored["recommended_ms"] == 352_000.0
    assert "decode_floor_applied_ms" not in unfloored


def test_the_reported_case_is_floored_above_pure_decode_time():
    """With the rate declared, the same samples must not be advised a
    deadline the call cannot physically meet."""
    m = _model(decode_rates={"tier2-analyst": _TIER2_ANALYST_TOK_S})
    _seed_the_reported_shape(m)
    advice = m.advise("tier2-analyst", 3, 4_000, 12_000)

    pure_decode_s = 12_000 / _TIER2_ANALYST_TOK_S
    assert pure_decode_s == pytest.approx(451.1, abs=1.0)

    assert advice["recommended_timeout_s"] >= 451, (
        f"floored recommendation ({advice['recommended_timeout_s']}s) is below "
        f"the ~{pure_decode_s:.0f}s of pure decode the call needs")
    # The floor actually did the lifting, not a coincidentally-large sample.
    assert advice["recommended_ms"] > 352_000.0
    assert advice.get("decode_floor_applied_ms") == advice["recommended_ms"]


# ---------------------------------------------------------------------------
# Compatibility guarantee: absent rate changes nothing
# ---------------------------------------------------------------------------

def test_absent_rate_is_byte_identical_to_before_the_floor_existed():
    """No ``decode_rates`` at all, and an explicit empty ``{}``, must produce
    the exact same ``advise()`` output as each other AND as a model built
    before this feature existed would have — this is the whole reason the
    rate is opt-in per class rather than defaulted."""
    m_default = _model()
    m_empty = _model(decode_rates={})
    _seed_the_reported_shape(m_default)
    _seed_the_reported_shape(m_empty)

    for est_out in (60, 500, 2_048, 9_000, 12_000, 50_000):
        a = m_default.advise("tier2-analyst", 3, 4_000, est_out)
        b = m_empty.advise("tier2-analyst", 3, 4_000, est_out)
        assert a == b, f"decode_rates={{}} diverged from decode_rates=None at est_out={est_out}"
        assert "decode_floor_applied_ms" not in a


# ---------------------------------------------------------------------------
# The floor only ever RAISES
# ---------------------------------------------------------------------------

def test_the_floor_does_not_lower_a_higher_empirical_recommendation():
    """A genuinely slow, well-sampled cell already above the decode floor
    must win untouched — it carries load/tail information the floor does
    not, and the brief is explicit: max(), never a substitution."""
    m = _model(decode_rates={"tier2-analyst": 200.0})  # a FAST rate
    # p99 x margin(1.0) = 700_000ms — far above what 12,000 tokens at 200
    # tok/s would floor to (75,000ms).
    for _ in range(40):
        m.record("tier2-analyst", 3, 4_000, 9_000, 700_000.0, "ok", _NOW)
    advice = m.advise("tier2-analyst", 3, 4_000, 12_000)
    assert advice["recommended_ms"] == 700_000.0
    assert "decode_floor_applied_ms" not in advice


# ---------------------------------------------------------------------------
# The floor is self-bounding at the background ceiling
# ---------------------------------------------------------------------------

def test_the_floor_is_bounded_by_the_background_ceiling_for_an_absurd_ask():
    """Three of ``advise()``'s five call sites bypass the downstream
    ``resolve_ceiling_s`` entirely, so the floor has to bound itself."""
    m = _model(decode_rates={"tier2-analyst": 1.0})  # deliberately slow
    advice = m.advise("tier2-analyst", 3, 4_000, 10_000_000)  # absurd est_out
    assert advice["recommended_timeout_s"] == int(_BACKGROUND_CEILING_S)
    assert advice["decode_floor_applied_ms"] == _BACKGROUND_CEILING_S * 1000.0


def test_decode_rate_floor_ms_is_pure_and_bounded():
    """Direct unit coverage of the primitive, independent of TimeoutModel."""
    # the sanity-check numbers from the brief: est_out=500 @ 25 tok/s.
    small = decode_rate_floor_ms(500, 25.0)
    assert small == 500 / 25.0 * _DECODE_FLOOR_MARGIN * 1000.0
    assert small == 25_000.0  # 25s — well below any class floor (tier2 is 120s)

    # absent rate / absent estimate -> no floor.
    assert decode_rate_floor_ms(500, 0.0) == 0.0
    assert decode_rate_floor_ms(0, 25.0) == 0.0
    assert decode_rate_floor_ms(-5, 25.0) == 0.0

    # bounded at the background ceiling regardless of how absurd the ask is.
    huge = decode_rate_floor_ms(50_000_000, 1.0)
    assert huge == _BACKGROUND_CEILING_S * 1000.0


# ---------------------------------------------------------------------------
# Small asks keep their fail-fast behaviour
# ---------------------------------------------------------------------------

def test_small_asks_stay_below_the_class_floor_and_are_unaffected():
    """At est_out=500 on a 25 tok/s model the decode floor is ~25s, which
    must sit BELOW the tier2 class floor (120s) — only large asks may move.
    A cold model (no samples) answers from the class floor either way."""
    m = TimeoutModel(min_samples=30, floors={"tier2": 120.0},
                      decode_rates={"tier2": 25.0})
    advice = m.advise("tier2", 3, 1_000, 500)
    assert advice["source"] == "floor"
    assert advice["recommended_timeout_s"] == 120
    assert "decode_floor_applied_ms" not in advice, (
        "a small ask moved the recommendation — the class floor should have "
        "already been above the decode floor")


# ---------------------------------------------------------------------------
# Interaction with the monotonicity guard
# ---------------------------------------------------------------------------

def test_interaction_with_the_monotonicity_guard_result_is_the_max():
    """Both mechanisms active at once: the monotonicity guard still holds
    (advice never shrinks as est_out grows), and wherever the decode floor
    exceeds the guard's lattice-walked answer, the floor's number wins."""
    m = TimeoutModel(margin=1.5, min_samples=30, floors={"tier3": 60.0},
                      decode_rates={"tier3": 27.9})  # measured "thinker" rate
    # A well-sampled low-output cell, and a sparse high-output one that would
    # otherwise fall through to a faster aggregate (the exact starvation
    # shape from test_timeout_model_monotonic.py).
    for _ in range(1_000):
        m.record("tier3", 3, 400, 300, 7_000.0, "ok", _NOW)
    for _ in range(35):
        m.record("tier3", 3, 400, 6_000, 300_000.0, "ok", _NOW)

    prev = 0.0
    saw_decode_floor = False
    for est_out in (100, 300, 1_000, 6_000, 9_000, 32_000, 100_000):
        advice = m.advise("tier3", 3, 400, est_out)
        rec = advice["recommended_ms"]
        assert rec >= prev - 1e-6, (
            f"monotonicity broken at est_out={est_out}: {rec} < {prev}")
        prev = rec
        if advice.get("decode_floor_applied_ms") is not None:
            saw_decode_floor = True
            assert rec == advice["decode_floor_applied_ms"]

    assert saw_decode_floor, (
        "fixture never exercised the decode floor — test would pass even "
        "with it disabled")


# ---------------------------------------------------------------------------
# The catalog seam — models.yaml `token_speed` reaches the class map
# ---------------------------------------------------------------------------

def test_token_speed_is_parsed_and_filtered_like_every_other_class_field():
    """Mirrors ``test_the_runtime_overlay_is_seen_by_every_caller`` in
    ``test_catalog_writes.py``: a candidate build (``overlay=``) never touches
    the cached/served catalog, so this needs no cleanup. A class that omits
    ``token_speed`` must not appear in the map at all — that absence is what
    makes "no rate declared" the same thing as "no floor" downstream."""
    from roadstead.model_catalog import build_class_decode_rates, load_catalog

    cat = load_catalog(overlay={"endpoints": {"ghost": {
        "provider": "small-box", "kind": "chat", "status": "active",
        "slots": 1, "context_per_slot": 2048, "token_speed": 26.6,
    }}})
    rates = build_class_decode_rates(cat)
    assert rates["ghost"] == 26.6
    # tier1 in the shipped example declares no token_speed.
    assert "tier1" not in rates
