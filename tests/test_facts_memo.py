"""`facts()` on the hot path — memoised, and still correct.

`/rs/v1/chat` calls `EnrichedApi.facts()` on **every** request to resolve one
intent, and `facts()` called `timeout_model.advise` once per endpoint to fill a
single field. N ladder walks, per request, to answer a question about the whole
fleet — for a number that is a median over thousands of samples and cannot
meaningfully move between two requests a millisecond apart.

🚨 The field is not droppable and not optional. `prefer=latency` and
`prefer=balanced` rank on `typical_ms`, and an endpoint with no samples sorts as
SLOW (`intent.py` doctrine) — so a `facts()` that omitted it would silently
re-rank every intent-routed request rather than merely lose a display field.
Memoising keeps it correct for every caller; a flag would have made some callers
lose it.
"""

from __future__ import annotations

import pytest

from roadstead.config import ProxyConfig
from roadstead.enriched import EnrichedApi
from roadstead.service import ProxyService


@pytest.fixture
def svc(tmp_path):
    return ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db")))


def _count_advise(svc):
    calls = []
    real = svc._state.timeout_model.advise

    def counting(*a, **k):
        calls.append(a[0])
        return real(*a, **k)

    svc._state.timeout_model.advise = counting
    return calls


def test_a_burst_of_requests_walks_the_ladder_once(svc):
    """🚨 The point. Before this, twenty requests against a five-endpoint
    catalog cost a hundred `advise` calls; now they cost five."""
    calls = _count_advise(svc)
    first = svc._enriched.facts()
    per_pass = len(calls)
    assert per_pass == len(first), "the first pass should price every endpoint"

    for _ in range(19):
        svc._enriched.facts()
    assert len(calls) == per_pass, (
        f"{len(calls)} advise() calls for 20 facts() — the memo is not holding")


def test_typical_ms_is_still_reported_and_still_the_learned_median(svc):
    """🚨 Not dropped, and not zeroed. The whole risk of this change is a field
    that quietly becomes 0.0 — which `intent.py` reads as SLOW, so every
    `prefer=latency` request would re-rank without a single test failing on the
    ranking itself."""
    svc._state.timeout_model.advise = lambda ep, pri, ei, eo: {
        "median_ms": 1234.0, "sample_count": 99}
    facts = svc._enriched.facts()
    assert facts, "the catalog produced no facts"
    assert all(f.typical_ms == 1234.0 for f in facts), [
        (f.endpoint, f.typical_ms) for f in facts]


def test_a_readout_failure_costs_the_field_and_not_the_listing(svc):
    """A model readout must never 500 a listing. It falls back to 0.0, which is
    the same value an endpoint with no samples reports."""
    def boom(*a, **k):
        raise RuntimeError("timeout model is having a day")

    svc._state.timeout_model.advise = boom
    facts = svc._enriched.facts()
    assert facts
    assert all(f.typical_ms == 0.0 for f in facts)


def test_the_memo_expires(svc, monkeypatch):
    """A memo that never expired would pin the fleet's latency picture to
    whatever it was at boot — and at boot there are no samples at all, so every
    endpoint would rank as SLOW forever."""
    # 🚨 Patch the clock BEFORE priming. Priming under the real clock stamps a
    # real monotonic and every fake reading is then in the past, so the memo
    # looks eternally fresh and the test passes for the wrong reason.
    now = [1000.0]
    monkeypatch.setattr("roadstead.enriched.time.monotonic", lambda: now[0])
    calls = _count_advise(svc)
    svc._enriched.facts()
    first = len(calls)
    assert first > 0

    svc._enriched.facts()
    assert len(calls) == first, "the memo did not hold within its TTL"

    # 🚨 A FIXED advance, not `TTL + 1`. Advancing by the TTL moves the
    # goalposts with the thing under test: a mutation setting the TTL to a
    # billion seconds made the test wait a billion fake seconds and pass. The
    # bound below is the other half of the same assertion — the memo must expire
    # AND the window must be short enough that a fleet's latency picture is
    # never more than a moment stale.
    assert EnrichedApi._TYPICAL_MS_TTL_S <= 5.0, (
        "the memo window is long enough to pin the fleet's latency picture; "
        "`prefer=latency` would rank on numbers from minutes ago")
    now[0] += 6.0
    svc._enriched.facts()
    assert len(calls) > first, "the memo never expires"


def test_an_endpoint_that_appears_later_is_priced(svc):
    """The memo is keyed by endpoint, and the fleet is not fixed: a deployment
    (or a test) may add an entry to `config.endpoints` that the catalog never
    named, and `facts()` reports the UNION. A memo keyed only on time would
    report 0.0 for it until the TTL lapsed — i.e. rank a brand-new endpoint as
    SLOW for a second, on the one path that routes by latency."""
    from roadstead.config import EndpointConfig

    svc._enriched.facts()                       # prime
    svc._state.config.endpoints["latecomer"] = EndpointConfig(
        endpoint_class="latecomer", role="latecomer", host="192.0.2.9", port=1)
    calls = _count_advise(svc)
    names = [f.endpoint for f in svc._enriched.facts()]
    assert "latecomer" in names
    assert "latecomer" in calls, (
        "a newly-routed endpoint was served from a memo that never priced it")
