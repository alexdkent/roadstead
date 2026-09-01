"""TimeoutModel.advise must never shrink as the caller asks for MORE output.

LEDGER: ``llm-output-budget-starvation``.

THE DEFECT
==========
``advise()`` walks a coarsening fallback ladder (cell → tier_out → tier →
endpoint) and takes the FIRST level with enough samples. That is the right shape
for a cold cell — but it silently inverts the advice at the top end:

  * a mid-sized request lands in a well-populated ``tier_out`` out-bucket and is
    advised off calls that genuinely produce that much output;
  * a LARGER request lands in a sparse high out-bucket, falls through to
    ``tier``, and is advised off an aggregate dominated by small, fast calls.

Measured on the live proxy, ``tier3`` @ P3_INGESTION (2026-07-29):

    est_out 8192  ->  459s   (source=tier_out, n=107)
    est_out 9000  ->  227s   (source=tier,     n=8682)

Asking for 10% more output HALVED the deadline.

WHY IT MATTERED
===============
kv4's truncation recovery re-derives its deadline every time it bumps
``max_tokens`` — correct, per the LLM-timeout doctrine. So the retry that was
supposed to rescue a truncated ``grader_coverage`` call (6000 -> 9000) came back
with a deadline SHORTER than the 276s the 6000-token attempt had already spent,
and 136 of 143 such retries were killed at the deadline having produced ZERO
output tokens.

And it could not self-heal: ``record()`` admits only ``status == "ok"`` samples,
so a bucket whose calls always time out can never gather the samples that would
correct its advice. Sparse stays sparse, forever.

THE GUARD
=========
An advice for N output tokens is never lower than the advice for fewer. It can
only ever RAISE a deadline, and the caller-side ceiling still bounds the result.
"""
from __future__ import annotations

import pytest

from roadstead.timeout_model import TimeoutModel, _OUT_EDGES


def _model(**kw) -> TimeoutModel:
    return TimeoutModel(margin=1.5, min_samples=30, floors={"tier3": 60.0}, **kw)


def _fill(m: TimeoutModel, *, out_tokens: int, n: int, latency_ms: float,
          in_tokens: int = 400, pri: int = 3, now: float = 1000.0) -> None:
    for _ in range(n):
        m.record("tier3", pri, in_tokens, out_tokens, latency_ms, "ok", now)


def _live_shape() -> TimeoutModel:
    """The measured live condition, reproduced.

    Two properties have to hold together or the inversion does not appear, and
    both are true of the real `tier3` P3 distribution:

      * the mid out-bucket (2048..8191) is well-sampled and SLOW — those are real
        multi-thousand-token generations;
      * the ``tier`` pool those calls fall back onto is overwhelmingly small, fast
        traffic, so its p99 sits BELOW the mid bucket's. The fast traffic has to
        be spread across several in-buckets because each cell's reservoir is
        capped at ``max_samples_per_cell`` (2000) — one cell alone cannot outvote
        the slow samples, which is why a naive fixture silently fails to
        reproduce this.

    Nothing is recorded in the top out-bucket (>= 8192): that is the whole point,
    and it is self-perpetuating in production because ``record()`` admits only
    status=="ok" samples and those calls only ever time out.
    """
    m = _model()
    _fill(m, out_tokens=6000, n=30, latency_ms=300_000.0)          # slow, real generations
    for in_tokens in (100, 2000, 8000, 20000):                     # all four in-buckets
        for out_tokens in (100, 300):                              # out-buckets 0 and 1
            _fill(m, out_tokens=out_tokens, n=1000, latency_ms=7_000.0,
                  in_tokens=in_tokens)
    return m


def test_the_exact_live_inversion_no_longer_happens() -> None:
    m = _live_shape()
    mid = m.advise("tier3", 3, 400, 6000)["recommended_ms"]
    big = m.advise("tier3", 3, 400, 9000)["recommended_ms"]
    assert big >= mid, (
        f"asking for MORE output was advised LESS time ({big:.0f}ms < {mid:.0f}ms) "
        "— this is the inversion that made every truncation retry a guaranteed "
        "timeout")


def test_the_fixture_really_does_reproduce_the_inversion() -> None:
    """Guard on the guard: prove the raw ladder WOULD invert here, so the test
    above cannot pass vacuously. ``_advise_at`` is the pre-guard behaviour."""
    m = _live_shape()
    floor = m.floor_ms("tier3")
    raw_mid = m._advise_at("tier3", 3, 0, 3, floor)[3]   # out-bucket 3 (2048..8191)
    raw_big = m._advise_at("tier3", 3, 0, 4, floor)[3]   # out-bucket 4 (>= 8192)
    assert raw_big < raw_mid, (
        f"fixture does not reproduce the defect (raw {raw_big:.0f} >= {raw_mid:.0f}) "
        "— the monotonicity test above would pass even with the guard removed")


def test_advice_is_monotonic_across_every_out_bucket_boundary() -> None:
    m = _live_shape()

    # walk each bucket edge, and a point either side of it
    probes = [1]
    for e in _OUT_EDGES:
        probes += [e - 1, e, e + 1]
    probes += [_OUT_EDGES[-1] * 4]

    prev = 0.0
    for eo in sorted(set(probes)):
        rec = m.advise("tier3", 3, 400, eo)["recommended_ms"]
        assert rec >= prev - 1e-6, (
            f"advice DROPPED at est_out={eo}: {rec:.0f}ms after {prev:.0f}ms")
        prev = rec


def test_the_lift_is_reported_so_a_starved_bucket_is_visible() -> None:
    """When the guard fires it says so — otherwise the only symptom of a sparse
    high bucket is a number that happens to look fine."""
    m = _live_shape()

    big = m.advise("tier3", 3, 400, 9000)
    assert "monotonic_lift_from_out_bucket" in big, (
        "the guard lifted the advice but left no trace — a dashboard cannot tell "
        "a genuinely fast high bucket from a starved one")

    mid = m.advise("tier3", 3, 400, 6000)
    assert "monotonic_lift_from_out_bucket" not in mid, (
        "the guard must be silent when it changes nothing")


def test_a_genuinely_well_sampled_high_bucket_is_left_alone() -> None:
    """The guard raises a starved bucket; it must not override real evidence that
    a big request is legitimately slower."""
    m = _model()
    _fill(m, out_tokens=6000, n=100, latency_ms=100_000.0)
    _fill(m, out_tokens=12000, n=100, latency_ms=400_000.0)

    big = m.advise("tier3", 3, 400, 12000)
    assert big["source"] in ("cell", "tier_out"), big["source"]
    assert big["recommended_ms"] == pytest.approx(400_000.0 * 1.5, rel=0.15)
    assert "monotonic_lift_from_out_bucket" not in big


def test_the_floor_still_applies_on_a_completely_cold_model() -> None:
    m = _model()
    a = m.advise("tier3", 3, 400, 9000)
    assert a["recommended_ms"] == pytest.approx(60_000.0)
    assert a["sample_count"] == 0


def test_reported_source_and_sample_count_describe_the_requested_bucket() -> None:
    """The lift changes the NUMBER; it must not misreport where the request
    actually landed, or the shadow report stops being diagnostic."""
    m = _live_shape()

    big = m.advise("tier3", 3, 400, 9000)
    assert big["source"] == "tier", big["source"]
    assert big["sample_count"] == 8030, big["sample_count"]
    # ...but the recommendation came from the slower, better-supported bucket
    assert big["recommended_ms"] > 300_000.0


def test_the_guard_is_bounded_work() -> None:
    """<= len(_OUT_EDGES) extra lookups — this runs on every advice call."""
    assert len(_OUT_EDGES) <= 8, (
        "the monotonicity guard re-queries every LOWER out-bucket; growing "
        "_OUT_EDGES makes advise() proportionally more expensive")
