"""The SIZING inputs to timeout advice: stretch curve, input buckets, and the
"caller declared no output budget" case.

Companion to ``test_timeout_model.py`` (primitives + ladder) and
``test_timeout_model_monotonic.py`` (the out-axis monotonicity guard). This file
covers the three defects that made a LONG-CONTEXT call get a deadline far
shorter than the work it was asked to do, plus the monotonicity invariant on the
INPUT axis that the finer buckets newly make reachable.

D1 — ``size_stretch`` saturated at ~82K
    ``size_max=4.0`` measured in linear multiples of a 16,384-token top edge
    caps the multiplier at 3.0x and stops growing at 5x the top edge. tier3
    serves 700K context; the constant was tuned when the ceiling was 128K.

D2 — ``_IN_EDGES`` collapsed 17K..700K into ONE cell
    whose p99 is dominated by ~20K-token calls, so the class floor always won and
    the model could never learn long-context latency. Live: ``recommended_ms``
    of exactly 180000.0 (the thinker floor) for 123K-token prompts.

D3 — ``est_out=0`` meant "the shortest output distribution"
    Every stock OpenAI client omits ``max_tokens``. The out-axis monotonicity
    guard only lifts from LOWER out-buckets and 0 is already the lowest, so it
    structurally could not help.

MONOTONICITY — advice must never SHRINK as the caller asks for more input or
more output. D2 makes the input axis a real axis, so the guard has to cover it;
without that, the sparse top in-bucket falls through to the ``tier`` aggregate
(small fast calls) and inverts exactly the way the out axis did in
``llm-output-budget-starvation``.
"""

from __future__ import annotations

from roadstead.timeout_model import (
    TimeoutModel,
    _IN_EDGES,
    _bucket,
    size_stretch,
)

_NOW = 1000.0


def _model(**kw) -> TimeoutModel:
    return TimeoutModel(margin=1.5, min_samples=30, floors={"thinker": 180.0}, **kw)


def _fill(m: TimeoutModel, *, n: int, in_tokens: int, out_tokens: int,
          latency_ms: float, pri: int = 3) -> None:
    for _ in range(n):
        m.record("thinker", pri, in_tokens, out_tokens, latency_ms, "ok", _NOW)


# ---------------------------------------------------------------------------
# D1 — the stretch curve
# ---------------------------------------------------------------------------

def test_stretch_keeps_growing_far_past_the_old_82k_saturation():
    """The old curve was flat from 5x the 16K top edge (81,920) upward, so a
    700K prompt was advised exactly what an 82K one was."""
    at82k = size_stretch(81_920)
    assert size_stretch(128_000) > at82k
    assert size_stretch(300_000) > size_stretch(128_000)
    assert size_stretch(700_000) > size_stretch(300_000)


def test_stretch_at_the_largest_served_context_is_sane():
    """tier3 serves 700K. Prefill there measures ~729 tok/s (vs ~1426 tok/s at
    213K), i.e. ~16 minutes of prefill alone — the multiplier has to be able to
    lift the 180s thinker floor into that territory, and no further than the
    per-caller ceiling would allow anyway."""
    s = size_stretch(700_000)
    assert 8.0 <= s <= 12.0, s


def test_stretch_is_never_below_what_the_old_curve_gave():
    """The old linear term was fine below its clamp — it was only the clamp that
    was wrong. Nothing in the 16K..82K band may get a SHORTER deadline than it
    gets today; this fix only ever widens."""
    for n in (16_384, 20_000, 32_768, 49_152, 65_536, 81_920):
        legacy = 1.0 + 0.5 * min((n - 16_384) / 16_384, 4.0)
        assert size_stretch(n) >= legacy - 1e-9, f"{n}: {size_stretch(n)} < {legacy}"


def test_stretch_is_monotonic_and_neutral_below_the_reference():
    assert size_stretch(0) == 1.0
    assert size_stretch(1_000) == 1.0
    assert size_stretch(16_384) == 1.0
    prev = 0.0
    for n in (16_384, 17_000, 24_000, 32_768, 65_536, 100_000, 131_072,
              200_000, 262_144, 400_000, 700_000, 1_000_000):
        cur = size_stretch(n)
        assert cur >= prev - 1e-9, f"stretch DROPPED at {n}"
        prev = cur


# ---------------------------------------------------------------------------
# D2 — long context needs its own cell
# ---------------------------------------------------------------------------

def test_long_context_resolves_to_a_long_context_cell_not_the_17k_catch_all():
    """Reproduces the live shape: plentiful, fast ~22K-token traffic plus a
    small population of genuinely slow 120K-token calls.

    The fast traffic is recorded LAST and exceeds ``max_samples_per_cell``, so
    under one shared >16K bucket the reservoir holds nothing but fast samples
    and p99*margin lands under the floor — which is exactly what the live
    timeout events showed (``recommended_ms`` == 180000.0, the thinker floor,
    for 123K-token prompts)."""
    m = _model()
    _fill(m, n=35, in_tokens=120_000, out_tokens=400, latency_ms=240_000.0)
    # Two fast in-buckets, matching the live ratio (61,319 sub-16K calls against
    # ~800 long-context ones). It has to be this lopsided: the p99 of the
    # coarsened `tier` pool is a LONG-CONTEXT sample as soon as the slow calls
    # are more than 1% of it, and then every short-prompt request inherits a
    # long-context deadline through the fallback ladder.
    _fill(m, n=4_000, in_tokens=8_000, out_tokens=400, latency_ms=20_000.0)
    _fill(m, n=4_000, in_tokens=22_000, out_tokens=400, latency_ms=20_000.0)

    assert _bucket(22_000, _IN_EDGES) != _bucket(120_000, _IN_EDGES), (
        "17K and 120K still share one input bucket — the model cannot learn "
        "long-context latency")

    long_ctx = m.advise("thinker", 3, 120_000, 400)
    short_ctx = m.advise("thinker", 3, 22_000, 400)

    assert long_ctx["source"] == "cell", long_ctx["source"]
    assert long_ctx["recommended_ms"] > 180_000.0, (
        "long-context advice is still pinned to the class floor")
    assert long_ctx["recommended_ms"] > short_ctx["recommended_ms"]


def test_the_input_buckets_reach_the_largest_served_context():
    """A 700K prompt and a 130K prompt must not be the same cell — tier3 serves
    700K and the difference is ~14 minutes of prefill."""
    assert _bucket(130_000, _IN_EDGES) != _bucket(700_000, _IN_EDGES)
    assert _IN_EDGES == sorted(set(_IN_EDGES)), "edges must be strictly ascending"


# ---------------------------------------------------------------------------
# D3 — a caller that omits max_tokens
# ---------------------------------------------------------------------------

def test_missing_output_budget_is_unknown_not_the_shortest_output_bucket():
    """Every stock OpenAI client omits ``max_tokens`` (the ``pool`` CLI sends
    ``{messages, model, tools, stream, stream_options}``), which arrives here as
    est_out=0 — the SHORTEST-output distribution, and the lowest bucket the
    out-axis guard can lift from."""
    m = _model()
    _fill(m, n=1_000, in_tokens=4_000, out_tokens=60, latency_ms=5_000.0)
    _fill(m, n=300, in_tokens=4_000, out_tokens=3_000, latency_ms=400_000.0)

    unknown = m.advise("thinker", 3, 4_000, 0)
    tiny = m.advise("thinker", 3, 4_000, 60)

    assert unknown["recommended_ms"] > tiny["recommended_ms"], (
        "a caller that declared no output budget was advised off the "
        "shortest-output distribution")
    assert unknown.get("out_bucket_unknown_resolved_to") is not None, (
        "the substitution left no trace — a shadow report cannot tell an "
        "unknown-budget call from a genuinely tiny one")


def test_unknown_output_does_not_inflate_an_endpoint_whose_outputs_are_tiny():
    """The policy is empirical, not a blanket lift: rerank/embed genuinely
    produce zero output tokens, and their observed distribution says so."""
    m = TimeoutModel(margin=1.5, min_samples=30, floors={"rerank": 10.0})
    for _ in range(200):
        m.record("rerank", 1, 500, 0, 2_000.0, "ok", _NOW)
    a = m.advise("rerank", 1, 500, 0)
    assert a["source"] == "cell", a["source"]
    assert a["recommended_ms"] == 10_000.0


# ---------------------------------------------------------------------------
# Monotonicity — BOTH axes
# ---------------------------------------------------------------------------

def test_advice_never_shrinks_as_the_prompt_grows():
    """The input-axis twin of ``llm-output-budget-starvation``.

    Finer input buckets create sparse HIGH cells. A sparse cell falls through to
    the ``tier`` aggregate, which is dominated by small fast calls — so asking
    for MORE input would be advised LESS time. That inversion is what burned
    ~12.5 hours of thinker time on the output axis."""
    m = _model()
    # a well-sampled, genuinely slow 120K cell
    _fill(m, n=40, in_tokens=120_000, out_tokens=400, latency_ms=240_000.0)
    # a barely-sampled 400K cell — below min_samples, so it falls back
    _fill(m, n=4, in_tokens=400_000, out_tokens=400, latency_ms=300_000.0)
    # ...and a mountain of small fast traffic for a fallback to land on. It has
    # to be big enough that the slow samples sit BELOW the p99 index of the
    # merged pool (44 slow in 6044 is 0.7%), or the fallback accidentally
    # inherits a slow number and the inversion never shows.
    for in_tokens in (500, 2_000, 8_000, 22_000):
        _fill(m, n=1_500, in_tokens=in_tokens, out_tokens=400, latency_ms=7_000.0)

    prev = 0.0
    for est_in in (500, 2_000, 8_000, 22_000, 40_000, 80_000, 120_000,
                   200_000, 400_000, 700_000):
        rec = m.advise("thinker", 3, est_in, 400)["recommended_ms"]
        assert rec >= prev - 1e-6, (
            f"advice DROPPED at est_in={est_in}: {rec:.0f}ms after {prev:.0f}ms")
        prev = rec


def test_advice_is_monotonic_across_the_full_in_x_out_grid():
    """Neither axis alone: the guarantee is over the whole lattice, because a
    caller can raise both at once (a truncation retry on a long prompt)."""
    m = _model()
    _fill(m, n=40, in_tokens=120_000, out_tokens=400, latency_ms=240_000.0)
    _fill(m, n=40, in_tokens=8_000, out_tokens=6_000, latency_ms=300_000.0)
    for in_tokens in (500, 8_000, 22_000, 120_000):
        _fill(m, n=1_000, in_tokens=in_tokens, out_tokens=100, latency_ms=7_000.0)

    ins = [1, 2_000, 8_000, 22_000, 40_000, 120_000, 300_000, 700_000]
    outs = [1, 100, 400, 1_000, 4_000, 6_000, 9_000, 32_000]
    grid = {
        (i, o): m.advise("thinker", 3, i, o)["recommended_ms"]
        for i in ins for o in outs
    }
    for ii, i in enumerate(ins):
        for oi, o in enumerate(outs):
            if ii:
                assert grid[(i, o)] >= grid[(ins[ii - 1], o)] - 1e-6, (
                    f"advice dropped growing input {ins[ii-1]}->{i} at out={o}")
            if oi:
                assert grid[(i, o)] >= grid[(i, outs[oi - 1])] - 1e-6, (
                    f"advice dropped growing output {outs[oi-1]}->{o} at in={i}")


def test_the_in_axis_lift_is_reported():
    """When a sparse long-context cell inherits from a shorter one, say so —
    otherwise the only symptom is a number that happens to look fine."""
    m = _model()
    _fill(m, n=40, in_tokens=120_000, out_tokens=400, latency_ms=240_000.0)
    _fill(m, n=4, in_tokens=400_000, out_tokens=400, latency_ms=300_000.0)
    # ...and a mountain of small fast traffic for a fallback to land on. It has
    # to be big enough that the slow samples sit BELOW the p99 index of the
    # merged pool (44 slow in 6044 is 0.7%), or the fallback accidentally
    # inherits a slow number and the inversion never shows.
    for in_tokens in (500, 2_000, 8_000, 22_000):
        _fill(m, n=1_500, in_tokens=in_tokens, out_tokens=400, latency_ms=7_000.0)

    big = m.advise("thinker", 3, 400_000, 400)
    assert big.get("monotonic_lift_from_in_bucket") is not None, big


def test_the_guard_is_bounded_work():
    """The guard re-queries every LOWER (in, out) cell on every advice call;
    the coarse ladder levels are memoised, but the lattice itself must stay
    small."""
    from roadstead.timeout_model import _OUT_EDGES

    assert (len(_IN_EDGES) + 1) * (len(_OUT_EDGES) + 1) <= 64, (
        "the in x out lattice grew past a sane per-request lookup budget")
