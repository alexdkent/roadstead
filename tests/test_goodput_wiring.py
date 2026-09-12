"""Goodput collapse — the WIRING, as opposed to the rule.

`tests/test_goodput.py` holds the rule to its measured precision against a fleet
that does not exist. This file holds everything BETWEEN that rule and a caller:
the catalog keys that carry the thresholds, the scraper that reads the counters,
the poller that samples, the single enforcement gate, and the surfaces an
operator reads during the dark soak.

The failure mode all of it guards against is the one this repository keeps
finding: a knob that is silently dropped, a detector that never runs, a gate that
looks armed and is not. Each of those looks exactly like a feature that was never
load-bearing.
"""

from __future__ import annotations

import types

import pytest

from roadstead import model_catalog
from roadstead.backend import BackendClientPool
from roadstead.config import EndpointConfig
from roadstead.flags import DEFAULT_FLAGS, RuntimeFlags
from roadstead.goodput import GoodputMonitor, GoodputThresholds, Verdict
from roadstead.health import Health
from roadstead.observability import check_alerts
from roadstead.queue import PersistentQueue

# The measured fleet rule, as a `models.yaml` policy block would spell it. Every
# number is one fleet's hardware measurement — see test_goodput.py's docstring.
FLEET_POLICY = {
    "goodput_min_running": 3,
    "goodput_sustain_evaluations": 3,
    "goodput_max_iteration_rate": 1.0,
    "goodput_max_generation_tps": 2.0,
    "goodput_max_prefill_tps": 600.0,
    "goodput_recovery_evaluations": 2,
    "goodput_max_hold_s": 300.0,
}


# --------------------------------------------------------------------------- #
# 1. The catalog keys. A policy key absent from _POLICY_PASSTHROUGH is SILENTLY
#    DROPPED, which looks identical to a threshold that never mattered.
# --------------------------------------------------------------------------- #

def _goodput_passthrough_keys() -> list[str]:
    return [k for k in model_catalog._POLICY_PASSTHROUGH if k.startswith("goodput_")]


def test_every_goodput_policy_key_reaches_endpoint_config():
    """All seven, end to end through the REAL builder.

    Parametrised over `_POLICY_PASSTHROUGH` itself rather than a hand-kept list,
    so a key added to one and not the other cannot pass: an entry here with no
    `EndpointConfig` field fails on the constructor, and a field with no entry
    fails the count assertion below.
    """
    keys = _goodput_passthrough_keys()
    assert len(keys) == 7, f"expected seven goodput policy keys, got {keys}"
    assert set(keys) == set(FLEET_POLICY), (
        "this test's policy block and the passthrough list have diverged — one of "
        "them is describing a key that is being dropped")

    entry = model_catalog.EndpointEntry(
        name="probe", provider="p", kind="chat", policy=dict(FLEET_POLICY))
    kw = model_catalog.build_endpoint_kwargs(entries=[entry])["probe"]
    for key, value in FLEET_POLICY.items():
        assert kw.get(key) == value, f"{key} was dropped by the catalog"

    # …and they must be REAL fields, not kwargs EndpointConfig rejects.
    ep = EndpointConfig(**{k: v for k, v in kw.items()
                           if k in EndpointConfig.__dataclass_fields__})
    for key, value in FLEET_POLICY.items():
        assert getattr(ep, key) == value


def test_endpoint_config_ships_no_threshold_default():
    """🚨 Absent ⇒ the detector does not run.

    A default here would put one fleet's hardware constants in every deployment
    AND make "nobody configured it" indistinguishable from "it is watching".
    """
    ep = EndpointConfig(endpoint_class="x", role="x")
    for key in _goodput_passthrough_keys():
        assert getattr(ep, key) in (0, 0.0), f"{key} ships a default"
    health = Health(_state())
    assert not health.goodput_thresholds(ep).armed


def test_a_declared_stanza_arms_the_detector():
    entry = model_catalog.EndpointEntry(
        name="probe", provider="p", kind="chat", policy=dict(FLEET_POLICY))
    kw = model_catalog.build_endpoint_kwargs(entries=[entry])["probe"]
    ep = EndpointConfig(**{k: v for k, v in kw.items()
                           if k in EndpointConfig.__dataclass_fields__})
    t = Health(_state()).goodput_thresholds(ep)
    assert t.armed
    assert t.min_running == 3
    assert t.sustain_evaluations == 3
    assert t.max_iteration_rate == 1.0
    assert t.max_generation_tps_per_request == 2.0
    assert t.max_prefill_tps == 600.0
    assert t.effective_recovery_evaluations == 2
    assert t.effective_max_hold_s == 300.0


def test_an_undeclared_recovery_bound_still_bounds_the_hold():
    """The recovery numbers are NOT detection thresholds, and they DO carry
    defaults — a partially-configured endpoint must still be unable to latch a
    trip forever. No value here can make the detector fire."""
    partial = GoodputThresholds(min_running=3, sustain_evaluations=3,
                                max_prefill_tps=600.0)
    assert partial.effective_recovery_evaluations >= 2
    assert 0 < partial.effective_max_hold_s < float("inf")


# --------------------------------------------------------------------------- #
# 2. The scraper. Two counters added to a payload four tests already pin.
# --------------------------------------------------------------------------- #

def _probe_returning(text: str, status: int = 200) -> BackendClientPool:
    pool = BackendClientPool.__new__(BackendClientPool)

    class _Resp:
        status_code = status

        @property
        def text(self):
            return text

    class _Client:
        async def get(self, path):
            return _Resp()

    pool._client_for = lambda base_url, min_pool=0: _Client()
    return pool


#: 🚨 Captured UNBOUND at import, before the autouse `_no_network_probes`
#: fixture stubs it on the class (it moved into `STUBBED_PROBES` on 2026-09-12
#: because the poller reaches it now). A bound call here would exercise the STUB
#: and every parser assertion below would pass for the wrong reason.
_REAL_COUNTERS = BackendClientPool.probe_progress_counters

_EP = EndpointConfig(endpoint_class="x", role="x")

#: The real vLLM shape, with the iteration HISTOGRAM family as the engine emits
#: it: `_bucket` lines carrying `le=` labels, then `_sum`, then `_created`, and
#: only then the `_count` that is the actual iteration counter.
_VLLM_FULL = """# HELP vllm:prompt_tokens_total x
vllm:prompt_tokens_total{engine="0",model_name="m"} 7815245.0
vllm:generation_tokens_total{engine="0",model_name="m"} 358688.0
vllm:num_requests_running{engine="0",model_name="m"} 6.0
# TYPE vllm:iteration_tokens_total histogram
vllm:iteration_tokens_total_bucket{engine="0",le="1.0",model_name="m"} 900000.0
vllm:iteration_tokens_total_bucket{engine="0",le="8.0",model_name="m"} 900000.0
vllm:iteration_tokens_total_bucket{engine="0",le="+Inf",model_name="m"} 900000.0
vllm:iteration_tokens_total_sum{engine="0",model_name="m"} 41000000.0
vllm:iteration_tokens_total_created{engine="0",model_name="m"} 1.757e+09
vllm:iteration_tokens_total_count{engine="0",model_name="m"} 123456.0
"""

_LLAMACPP_FULL = """llamacpp:prompt_tokens_total 1.90511e+06
llamacpp:tokens_predicted_total 0
llamacpp:n_decode_total 196408
llamacpp:requests_processing 4
"""


@pytest.mark.asyncio
async def test_iteration_counter_is_matched_on_its_full_name_not_the_family():
    """🚨 THE TRAP, and it is the kind that looks like it works.

    `vllm:iteration_tokens_total` is a HISTOGRAM. A `startswith` on the family
    prefix — which is how the two token counters beside it are correctly matched —
    would fold three `_bucket` lines plus `_sum` plus `_created` into one number.
    That number rises monotonically and graphs beautifully. It is garbage, and the
    iteration clause built on it would be measuring nothing.

    Here the buckets are 900,000 each and `_sum` is 41,000,000 against a true
    `_count` of 123,456, so a prefix match cannot coincidentally agree.
    """
    got = await _REAL_COUNTERS(_probe_returning(_VLLM_FULL), _EP)
    assert got is not None
    assert got["iterations"] == 123456, (
        f"iterations must come from `_count` alone; got {got['iterations']} "
        f"(2,741,000 would be the whole family summed)")
    assert got["running"] == 6
    assert got["prompt"] == 7815245
    assert got["generation"] == 358688


@pytest.mark.asyncio
async def test_llamacpp_yields_running_and_no_iteration_counter():
    """llama.cpp has no iteration counter, and `n_decode_total` must not be
    mapped to both clauses.

    One number wearing two clause names would make the iteration clause a
    duplicate of the generation clause while reading, in the verdict, as
    independent evidence — a conjunction with a term that cannot disagree.
    """
    got = await _REAL_COUNTERS(_probe_returning(_LLAMACPP_FULL), _EP)
    assert got is not None
    assert got["iterations"] is None, (
        "absent, not 0 — a detector that reads absence as 0 concludes the engine "
        "has stopped")
    assert got["running"] == 4
    assert got["generation"] == 196408   # n_decode_total, not tokens_predicted
    assert got["prompt"] == 1905110


@pytest.mark.asyncio
async def test_an_engine_with_only_running_still_discriminates():
    """A backend publishing ONE of the four is not "cannot discriminate".

    The None-means-blind contract is per KEY, not per scrape: an engine that
    publishes occupancy and nothing else can still be configured for no clauses
    at all, and must not be confused with an unreachable backend.

    🚨 **AND EVERY ABSENT KEY IS `None`, INCLUDING `prompt` AND `generation`.**
    This assertion read `{"prompt": 0, "generation": 0, ...}` for a day, and that
    zero was a latching break of `goodput.py`'s central invariant — see
    `test_an_absent_prompt_counter_does_not_read_as_a_stopped_engine` below for
    the end-to-end consequence. The test that blessed the defect sat eighteen
    lines under a docstring correctly explaining why it would be one.
    """
    got = await _REAL_COUNTERS(_probe_returning(
        'vllm:num_requests_running{engine="0"} 2.0\n'), _EP)
    assert got == {"prompt": None, "generation": None,
                   "iterations": None, "running": 2}


@pytest.mark.asyncio
async def test_a_data_parallel_backend_has_its_engines_SUMMED():
    """Summing across DIFFERENT label sets is correct and deliberate.

    vLLM data parallelism publishes one series per engine; taking the last would
    silently track whichever sorted last. This is the control for the duplicate
    case below — without it, "treat a repeat as malformed" could be implemented as
    "never sum", and that would quietly halve a DP backend's counters.
    """
    body = ('vllm:num_requests_running{engine="0",model_name="m"} 3.0\n'
            'vllm:num_requests_running{engine="1",model_name="m"} 4.0\n'
            'vllm:prompt_tokens_total{engine="0",model_name="m"} 100.0\n'
            'vllm:prompt_tokens_total{engine="1",model_name="m"} 200.0\n')
    got = await _REAL_COUNTERS(_probe_returning(body), _EP)
    assert got["running"] == 7, "two ENGINES must sum"
    assert got["prompt"] == 300


@pytest.mark.asyncio
async def test_a_duplicated_identical_series_is_absence_not_a_doubled_gauge():
    """🚨 THE SAME "EASIER TO FIRE" CLASS AS THE `or 0` COALESCE.

    Two identical `requests_processing 6` lines summed give `running: 12`, which
    pushes occupancy PAST its gate and shrinks the `generation_tps / running_mean`
    denominator at the same time — both in the direction of firing, on an endpoint
    whose exposition nobody parsed correctly. A repeat of the same series with the
    SAME label set is malformed exposition, not data parallelism, and it gets the
    answer every unmeasurable input gets: absence.
    """
    body = ("llamacpp:requests_processing 6\n"
            "llamacpp:requests_processing 6\n"
            "llamacpp:prompt_tokens_total 100\n")
    got = await _REAL_COUNTERS(_probe_returning(body), _EP)
    assert got["running"] is None, (
        f"a duplicated gauge read as {got['running']} — 12 is the doubled value "
        f"that trips the occupancy gate")
    assert got["prompt"] == 100, "the OTHER counters must survive intact"


@pytest.mark.asyncio
async def test_a_malformed_key_cannot_be_revived_by_a_later_good_line():
    """Stickiness, so the verdict does not depend on line ORDER.

    Without it, `dup, dup, good` reads as malformed and `good, dup, dup` … also
    has to, or the same body means two different things depending on how the
    engine chose to order its exposition.
    """
    body = ("llamacpp:requests_processing 6\n"
            "llamacpp:requests_processing 6\n"
            "llamacpp:requests_processing 6\n")
    got = await _REAL_COUNTERS(_probe_returning(body), _EP)
    assert got is None or got["running"] is None


@pytest.mark.asyncio
async def test_an_infinite_value_drops_only_its_own_counter():
    """`int(float("+Inf"))` raises OverflowError, NOT ValueError.

    Uncaught it escaped the per-line handler to the function-level one and
    discarded the WHOLE scrape — `running` and every counter already parsed
    included. The direction was safe (absence) and the blast radius was wrong, and
    it contradicted `probe_progress_counters`' own documented promise that a bad
    line drops only its own counter. `+Inf` is ordinary exposition: every histogram
    publishes an `le="+Inf"` bucket.
    """
    body = ("llamacpp:requests_processing 6\n"
            "llamacpp:prompt_tokens_total +Inf\n"
            "llamacpp:n_decode_total 500\n")
    got = await _REAL_COUNTERS(_probe_returning(body), _EP)
    assert got is not None, "one bad line must not discard the scrape"
    assert got["running"] == 6, "the counter parsed BEFORE the bad line survives"
    assert got["generation"] == 500, "and the one parsed after it"
    assert got["prompt"] is None, "only the unreadable counter is absent"


@pytest.mark.asyncio
async def test_a_negative_counter_is_absence_not_a_measurement():
    """A cumulative token counter and a concurrency gauge are non-negative.

    Accepted as measured, a negative yields a 0.0 rate — which satisfies every
    `<` clause in the rule. Fixed as part of the duplicate work above because the
    check is one line in the same helper; flagged separately in the report because
    it was raised as report-only.

    Distinct from a counter going BACKWARDS across samples, which is a backend
    RESTART and `goodput.py` handles on its own (`counter_reset`).
    """
    body = ("llamacpp:requests_processing 6\n"
            "llamacpp:prompt_tokens_total -5\n")
    got = await _REAL_COUNTERS(_probe_returning(body), _EP)
    assert got["prompt"] is None
    assert got["running"] == 6


@pytest.mark.asyncio
async def test_a_backend_with_no_counters_at_all_is_still_none():
    got = await _REAL_COUNTERS(_probe_returning("# nothing\nfoo 1\n"), _EP)
    assert got is None


# --------------------------------------------------------------------------- #
# 2c. 🚨 REGRESSION — the invariant broken by the SCRAPER, not the monitor
#
# Every "configured-but-absent" test in this repo used `iterations` or `running`,
# the two keys the scraper could already return as None, or stubbed the whole
# scrape to None. NOTHING passed an absent `prompt` or an absent `generation`
# into the monitor — and those are precisely the two the scraper coalesced to 0.
# The unit proved the unit; the PATH never reached it. These tests drive the real
# scraper into the real monitor, which is the only reading that could have caught
# it.
# --------------------------------------------------------------------------- #

#: The llama.cpp 3-clause configuration `goodput.py` explicitly blesses:
#: occupancy + generation + prefill, iteration clause undeclared because the
#: engine publishes no iteration counter. This is the SUPPORTED shape, not a
#: contrived one — which is what made the defect serious.
LLAMACPP_POLICY = dict(FLEET_POLICY)
LLAMACPP_POLICY["goodput_max_iteration_rate"] = 0.0


def _armed_llamacpp() -> EndpointConfig:
    kw = dict(endpoint_class="tier3", role="reasoner")
    kw.update(LLAMACPP_POLICY)
    return EndpointConfig(**kw)


class _ScrapingBackend:
    """A backend whose `/metrics` body a test writes, scraped by the REAL parser.

    The point is to leave NOTHING between the wire and the verdict stubbed: the
    defect this file now guards lived in the parser's return statement, so a test
    that hands the monitor a hand-built dict cannot see it.
    """

    def __init__(self, body: str) -> None:
        self._pool = _probe_returning(body)

    async def probe_progress_counters(self, ep_cfg):
        return await _REAL_COUNTERS(self._pool, ep_cfg)


async def _drive(body: str, ep_cfg: EndpointConfig, ticks: int = 6):
    """Real scrape → real observe/evaluate → real latch, N poller ticks."""
    state = _state()
    state.backend = _ScrapingBackend(body)
    health = Health(state)
    for i in range(ticks):
        await health.sample_goodput("tier3", ep_cfg, now=20_000.0 + i * 10.0)
    return state, health


@pytest.mark.asyncio
async def test_an_absent_prompt_counter_does_not_read_as_a_stopped_engine():
    """🚨 THE REGRESSION. A llama.cpp endpoint publishing ONLY occupancy.

    `/metrics` carries `llamacpp:requests_processing 6` and nothing else, against
    a config whose prefill and generation clauses are both declared. Before the
    fix the scraper returned `prompt: 0, generation: 0`, so:

        verdict COLLAPSED, clauses_held (occupancy, generation, prefill),
        reason '', collapsed_endpoints {'tier3'}, trips 1,
        endpoint_healthy False when armed — and
        roadstead_endpoint_goodput_unknown reading 0.

    Two of the four clauses were satisfied by counters NOBODY HAD MEASURED, the
    breaker latched, and the metric whose entire job is to make that visible said
    everything was fine. `goodput.py` was correct throughout; its INPUT was not.
    """
    state, health = await _drive("llamacpp:requests_processing 6\n",
                                 _armed_llamacpp())

    verdict = state.goodput_verdicts["tier3"]
    assert verdict.outcome is Verdict.UNKNOWN, (
        f"a configured counter the engine does not publish read as "
        f"{verdict.outcome.value} with clauses {verdict.clauses_held}")
    assert verdict.reason, "UNKNOWN must name WHICH blindness it was"
    assert "generation" in verdict.reason and "prompt" in verdict.reason, (
        f"the reason must name the missing counters; got {verdict.reason!r}")
    assert state.collapsed_endpoints == set(), "the breaker latched while blind"
    assert state.goodput.trips("tier3") == 0
    # The blindness signal is the one that must speak.
    assert health.goodput_snapshot("tier3", 20_060.0)["verdict"] == "unknown"


@pytest.mark.asyncio
async def test_an_absent_generation_counter_alone_is_also_unknown():
    """The same defect from the other side: prefill measurable, generation not.

    Separated from the test above because the two clauses read DIFFERENT keys, and
    a fix that restored absence for one of them would leave the other latching.
    """
    body = ("llamacpp:requests_processing 6\n"
            "llamacpp:prompt_tokens_total 80000000\n")
    state, _ = await _drive(body, _armed_llamacpp())

    verdict = state.goodput_verdicts["tier3"]
    assert verdict.outcome is Verdict.UNKNOWN
    assert "generation" in verdict.reason, verdict.reason
    assert "prompt" not in verdict.reason, (
        f"prompt WAS published — naming it would send the operator after the "
        f"wrong counter: {verdict.reason!r}")
    assert state.collapsed_endpoints == set()


@pytest.mark.asyncio
async def test_a_fully_published_llamacpp_body_still_reaches_a_verdict():
    """🚨 THE POSITIVE CONTROL for the two tests above.

    Without it they pass on a detector that can only ever say UNKNOWN — a null
    from an instrument that cannot produce a positive is not a result. Same
    3-clause config, same driver, a COMPLETE body: this one must collapse.
    """
    body = ("llamacpp:requests_processing 6\n"
            "llamacpp:prompt_tokens_total 80000000\n"
            "llamacpp:n_decode_total 9000000\n")
    state, _ = await _drive(body, _armed_llamacpp())

    verdict = state.goodput_verdicts["tier3"]
    assert verdict.outcome is Verdict.COLLAPSED, verdict
    assert set(verdict.clauses_held) == {"occupancy", "generation", "prefill"}
    assert "iterations" not in verdict.clauses_held, (
        "the iteration clause is undeclared here and must not be evaluated")
    assert state.collapsed_endpoints == {"tier3"}


@pytest.mark.parametrize("label, body", [
    # ⚠️ Every one of these was verified against the REAL parser to produce a
    # DROPPED counter, and `except ValueError: continue` is why. Before the fix
    # each became a 0 and therefore a satisfied clause.
    ("trailing space", "llamacpp:requests_processing 6\n"
                       "llamacpp:prompt_tokens_total 80000000 \n"
                       "llamacpp:n_decode_total 9000000\n"),
    ("tab separator", "llamacpp:requests_processing 6\n"
                      "llamacpp:prompt_tokens_total\t80000000\n"
                      "llamacpp:n_decode_total 9000000\n"),
    ("NaN value", "llamacpp:requests_processing 6\n"
                  "llamacpp:prompt_tokens_total NaN\n"
                  "llamacpp:n_decode_total 9000000\n"),
    # `int(float("+Inf"))` raises OverflowError, not ValueError — caught per line
    # alongside it since 2026-09-12. Uncaught it discarded the whole scrape.
    ("+Inf value", "llamacpp:requests_processing 6\n"
                   "llamacpp:prompt_tokens_total +Inf\n"
                   "llamacpp:n_decode_total 9000000\n"),
    ("overflowing exponent", "llamacpp:requests_processing 6\n"
                             "llamacpp:prompt_tokens_total 1e400\n"
                             "llamacpp:n_decode_total 9000000\n"),
    # A negative cumulative counter is physically impossible, and read as measured
    # it yields a 0.0 rate — which satisfies every `<` clause.
    ("negative counter", "llamacpp:requests_processing 6\n"
                         "llamacpp:prompt_tokens_total -5\n"
                         "llamacpp:n_decode_total 9000000\n"),
    # 🚨 THE ONE THAT MATTERS MOST. A wedged engine is exactly when `/metrics` is
    # slow and a body is most likely to arrive SHORT — so this failure mode was
    # CORRELATED with the condition being detected, which is the worst kind: the
    # detector would have been most confidently wrong precisely when it fired.
    ("body truncated after the gauge", "llamacpp:requests_processing 6\n"
                                       "llamacpp:prompt_tok"),
    # 🚨 A DUPLICATED series is the one malformed shape that fails in the
    # EASIER-TO-FIRE direction rather than towards absence: summed it inflates the
    # gauge, raising occupancy past its gate AND shrinking the per-request
    # denominator. Included here so it is covered by the same end-to-end
    # "must be UNKNOWN, must not latch" assertions as the rest.
    ("duplicated gauge", "llamacpp:requests_processing 6\n"
                         "llamacpp:requests_processing 6\n"
                         "llamacpp:prompt_tokens_total 80000000\n"
                         "llamacpp:n_decode_total 9000000\n"),
])
@pytest.mark.asyncio
async def test_a_malformed_metrics_line_is_blindness_not_a_stopped_engine(label, body):
    state, _ = await _drive(body, _armed_llamacpp())
    verdict = state.goodput_verdicts["tier3"]
    assert verdict.outcome is Verdict.UNKNOWN, (
        f"{label}: a line the parser cannot read became "
        f"{verdict.outcome.value} — a dropped counter must be absence, not zero")
    assert verdict.reason, f"{label}: UNKNOWN with no reason"
    assert state.collapsed_endpoints == set(), f"{label}: the breaker latched"


@pytest.mark.asyncio
async def test_the_blindness_metric_speaks_when_a_counter_is_absent():
    """The metric, not just the verdict — it is what a TSDB rule fires on.

    Before the fix `roadstead_endpoint_goodput_unknown` read **0** on the body
    below while the breaker latched, so every external consumer of this feature
    would have seen a clean, confident collapse.
    """
    from roadstead.goodput import Verdict as V

    state, _ = await _drive("llamacpp:requests_processing 6\n", _armed_llamacpp())
    verdict = state.goodput_verdicts["tier3"]
    # The emitter's two branches, asserted on the verdict they read.
    assert verdict.evaluable is False, "unknown must not publish a collapsed series"
    assert verdict.outcome is V.UNKNOWN


# --------------------------------------------------------------------------- #
# 2b. The FAKE's /metrics, so the e2e journey is standing on a real payload
# --------------------------------------------------------------------------- #

def _fake_metrics(**knobs) -> str:
    """Render `roadstead.testing`'s `/metrics` without a socket."""
    from starlette.testclient import TestClient

    from roadstead.testing import FakeBackend
    from roadstead.testing.fake_backend import make_fake_app

    controller = FakeBackend(**knobs)
    with TestClient(make_fake_app(controller)) as client:
        resp = client.get("/metrics")
    assert resp.status_code == 200
    return resp.text


@pytest.mark.asyncio
async def test_the_fake_serves_the_histogram_trap_by_default():
    """🚨 The fake must LAY the trap, not merely be capable of laying it.

    `emit_iteration_histogram_siblings` defaults True precisely so the
    `_bucket`/`_sum`/`_created` lines are there for anyone who reaches for the
    fake, rather than opted into by whoever remembered the trap existed. If this
    goes red, every e2e assertion about the iteration clause is being made
    against a payload a prefix matcher would also have got right.
    """
    text = _fake_metrics(engine="vllm", iteration_tokens_count=4_000,
                         generation_tokens_total=90_000,
                         prompt_tokens_total=800_000, num_requests_running=6)
    assert "vllm:iteration_tokens_total_bucket{" in text
    assert 'le="+Inf"' in text
    assert "vllm:iteration_tokens_total_sum{" in text
    assert "vllm:iteration_tokens_total_created{" in text

    got = await _REAL_COUNTERS(_probe_returning(text), _EP)
    assert got["iterations"] == 4_000, (
        f"got {got['iterations']} — the siblings were summed in")
    assert got["running"] == 6


@pytest.mark.asyncio
async def test_the_fake_in_llamacpp_shape_publishes_no_iteration_counter():
    """The engine split, through the fake: llama.cpp exposes occupancy and the
    two token counters, and NO iteration counter — plus a frozen
    `tokens_predicted_total`, the measured trap the generation mapping avoids."""
    text = _fake_metrics(engine="llama.cpp", iteration_tokens_count=4_000,
                         generation_tokens_total=196_408,
                         prompt_tokens_total=1_905_110, num_requests_running=4)
    assert "vllm:iteration_tokens_total" not in text
    assert "llamacpp:tokens_predicted_total 0" in text

    got = await _REAL_COUNTERS(_probe_returning(text), _EP)
    assert got["iterations"] is None, (
        "`n_decode_total` must not be mapped to the iteration clause as well — "
        "one number wearing two clause names is a conjunction with a term that "
        "cannot disagree")
    assert got["generation"] == 196_408
    assert got["running"] == 4


def test_the_fake_publishes_nothing_when_the_knobs_are_unset():
    """Default FakeBackend keeps its pre-2026-09-12 payload, so no existing test
    that reads `/metrics` sees a counter it never asked for."""
    text = _fake_metrics()
    assert "prefix_cache_hits_total" in text
    for name in ("num_requests_running", "requests_processing",
                 "iteration_tokens_total", "n_decode_total",
                 "prompt_tokens_total"):
        assert name not in text, f"the fake volunteered {name}"


# --------------------------------------------------------------------------- #
# 3. The poller + the single enforcement gate
# --------------------------------------------------------------------------- #

def _state(**over):
    """A ProxyState stub carrying everything `endpoint_healthy` and
    `sample_goodput` read — and nothing production does not have."""
    state = types.SimpleNamespace(
        endpoint_failure_times={}, endpoint_cooldown_until={},
        endpoint_cooldown_trips={}, cooldown_best_effort_skips={},
        paused_endpoints=set(), endpoint_health={},
        collapsed_endpoints=set(), goodput=GoodputMonitor(),
        goodput_verdicts={}, flags=RuntimeFlags(None),
        on_demand=types.SimpleNamespace(manages=lambda ep: False),
        scheduler=types.SimpleNamespace(queued_requests=lambda ep, bands: []),
        dispatch_event=types.SimpleNamespace(set=lambda: None),
        failover=None,
    )
    for k, v in over.items():
        setattr(state, k, v)
    return state


def _armed_ep(**over) -> EndpointConfig:
    kw = dict(endpoint_class="tier3", role="reasoner")
    kw.update({k: v for k, v in FLEET_POLICY.items()})
    kw.update(over)
    return EndpointConfig(**kw)


class _Counters:
    """A backend stub whose `/metrics` answer a test drives directly."""

    def __init__(self, sequence):
        self._seq = list(sequence)
        self.calls = 0

    async def probe_progress_counters(self, ep_cfg):
        self.calls += 1
        return self._seq[min(self.calls - 1, len(self._seq) - 1)]


def _wedge_counters(n):
    return [{"iterations": 500_000, "generation": 9_000_000,
             "prompt": 80_000_000, "running": 6} for _ in range(n)]


def _healthy_counters(n):
    return [{"iterations": 500_000 + i * 300, "generation": 9_000_000 + i * 1_200,
             "prompt": 80_000_000 + i * 12_000, "running": 6} for i in range(n)]


async def _poll(health, ep_cfg, times):
    """Drive N poller passes at the real 10 s cadence.

    Passes `now` rather than patching `time.monotonic`: the detector needs a
    MEASURED window and no test can produce one at wall-clock speed, but patching
    the module clock patches it for the EVENT LOOP too — which broke every real
    scrape in the e2e journey and read as a blind backend.
    """
    base = 10_000.0
    for i in range(times):
        await health.sample_goodput("tier3", ep_cfg, now=base + i * 10.0)


@pytest.mark.asyncio
async def test_the_poller_samples_and_trips():
    state = _state()
    state.backend = _Counters(_wedge_counters(8))
    health = Health(state)
    await _poll(health, _armed_ep(), 6)

    assert "tier3" in state.collapsed_endpoints, "the poller never tripped"
    assert state.goodput_verdicts["tier3"].outcome is Verdict.COLLAPSED
    assert state.goodput.trips("tier3") == 1


@pytest.mark.asyncio
async def test_an_unconfigured_endpoint_is_never_even_scraped():
    """Absent ⇒ off means off all the way down: no `/metrics` GET, no samples,
    no verdict published. A detector that costs a scrape on every endpoint that
    does not use it is a detector nobody leaves on."""
    state = _state()
    state.backend = _Counters(_wedge_counters(8))
    health = Health(state)
    await _poll(health, EndpointConfig(endpoint_class="tier3", role="r"), 6)

    assert state.backend.calls == 0
    assert state.goodput_verdicts == {}
    assert state.collapsed_endpoints == set()


@pytest.mark.asyncio
async def test_a_failing_scrape_never_trips():
    """🚨 THE INVARIANT, through the poller.

    `probe_progress_counters` returning None on every tick is an unreachable
    backend. It must publish UNKNOWN forever and never latch — the conjunction
    gets EASIER to satisfy the less we can see, so this is the path that turns a
    guard into the outage it was built to prevent.
    """
    state = _state()
    state.backend = _Counters([None] * 10)
    health = Health(state)
    await _poll(health, _armed_ep(), 8)

    assert state.collapsed_endpoints == set()
    assert state.goodput_verdicts["tier3"].outcome is Verdict.UNKNOWN
    assert state.goodput.trips("tier3") == 0


@pytest.mark.asyncio
async def test_a_raising_probe_never_disturbs_the_poller():
    """Fail-open: a detector that can wedge the capacity poller is worse than no
    detector."""
    class _Boom:
        async def probe_progress_counters(self, ep_cfg):
            raise RuntimeError("boom")

    state = _state()
    state.backend = _Boom()
    health = Health(state)
    # poll_endpoint_once guards it; sample_goodput itself is allowed to raise.
    with pytest.raises(RuntimeError):
        await health.sample_goodput("tier3", _armed_ep())
    assert state.collapsed_endpoints == set()


@pytest.mark.asyncio
async def test_recovery_clears_the_trip_through_the_poller():
    state = _state()
    state.backend = _Counters([*_wedge_counters(6), *_healthy_counters(6)])
    health = Health(state)
    await _poll(health, _armed_ep(), 12)

    assert "tier3" not in state.collapsed_endpoints, "never recovered"
    assert state.goodput.trips("tier3") == 1
    assert state.goodput_verdicts["tier3"].outcome is Verdict.HEALTHY


@pytest.mark.asyncio
async def test_the_flag_is_the_only_gate_and_it_is_off_by_default():
    """🚨 SHIP DARK, and prove the dark half separately from the armed half.

    The set is populated as an OBSERVATION whether or not enforcement is armed —
    that pair (`tripped: true, enforced: false`) IS the soak report. So the flag
    has to bite in `endpoint_healthy` or nowhere, and this is the test that says
    which.
    """
    assert DEFAULT_FLAGS["goodput_collapse_enforce"] is False

    state = _state()
    state.backend = _Counters(_wedge_counters(8))
    health = Health(state)
    await _poll(health, _armed_ep(), 6)

    # Latched, and the caller sees nothing.
    assert "tier3" in state.collapsed_endpoints
    assert health.endpoint_healthy("tier3") is True, (
        "with the flag off a latched endpoint must stay healthy to every caller")

    state.flags.set_many({"goodput_collapse_enforce": True})
    assert health.endpoint_healthy("tier3") is False, (
        "the flag flip must be the whole difference — no redeploy, no resample")


#: Every read of the enforce flag: (module, enclosing function) → (count, why).
#:
#: 🚨 This exists because the code ASSERTED "this is the only place the enforce
#: flag is read" in a comment and shipped no guard for it — and the claim was
#: false by a factor of six. A property worth writing in a comment is worth a
#: test; that is this repo's own rule, and this feature broke it. Idiom from
#: `test_identity_delegation.py::test_the_grant_is_read_in_exactly_one_place`.
#:
#: COUNTS, not just keys. The first cut of this guard pinned a bare total and
#: immediately caught that a fix landed minutes earlier had taken the total from
#: five to six — which is the whole argument for the counts: a read ADDED to a
#: function that already reads the flag is invisible to a key-set comparison, and
#: `sample_goodput` is precisely where an enforcement side effect would land.
_EXPECTED_FLAG_READS = {
    ("roadstead/health.py", "endpoint_healthy"): (
        1, "THE GATE — the only read that decides whether traffic is refused"),
    ("roadstead/health.py", "sample_goodput"): (
        3, "gates `fast_fail_interactive` (releasing the queued cohort), gates the "
           "post-recovery `dispatch_event.set()`, and picks the SHADOW suffix on "
           "the trip log. The first two change behaviour; both are downstream of a "
           "trip this same function just recorded"),
    ("roadstead/health.py", "goodput_snapshot"): (
        1, "the read-only `enforced` field on /v1/status — what makes the dark "
           "soak readable"),
    ("roadstead/lifecycle.py", "handle_submit"): (
        1, "re-words a refusal ALREADY decided by endpoint_healthy — nested inside "
           "`if not self.health.endpoint_healthy(...)`, so it cannot refuse "
           "anything on its own"),
}


def test_the_enforce_flag_is_read_only_in_named_places():
    """🚨 AST, over the whole package, for `goodput_collapse_enforce`.

    An unnamed read is how a feature shipped dark stops being dark. The table
    above is a literal, so a NEW read — in a new function OR an extra one in a
    function already listed — fails here with the message asking which of the two
    things it is: a second DECISION (it must not exist; route it through
    `Health.endpoint_healthy`) or REPORTING (say so in the table).

    Keyed by ENCLOSING FUNCTION rather than line number, so reformatting cannot
    redden it while moving a read into a new function does.

    ⚠️ **WHAT THIS GUARD CANNOT SEE, stated so nobody mistakes it for airtight:**
    it matches STRING LITERALS, so a read that never spells the key as one is
    invisible — `"goodput_collapse_" + "enforce"`, an f-string, a `getattr`, a
    module-level alias, or importing the name from elsewhere. Reaching that hole
    takes deliberate obfuscation rather than an ordinary mistake, and closing it
    would need dataflow analysis this repo does not have, so it is recorded here
    instead of chased. The same limitation applies to
    `test_identity_delegation.py::test_the_grant_is_read_in_exactly_one_place`,
    which this is modelled on.
    """
    import ast
    import pathlib

    root = pathlib.Path(__import__("roadstead").__file__).resolve().parent.parent
    found: dict[tuple[str, str], int] = {}
    for path in sorted((root / "roadstead").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        # Map each node to its nearest enclosing function by walking down.
        stack: list[tuple[ast.AST, str]] = [(tree, "<module>")]
        while stack:
            node, fname = stack.pop()
            for child in ast.iter_child_nodes(node):
                name = (child.name
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        else fname)
                if (isinstance(child, ast.Constant)
                        and child.value == "goodput_collapse_enforce"):
                    key = (str(path.relative_to(root)), fname)
                    found[key] = found.get(key, 0) + 1
                stack.append((child, name))

    # `flags.py` DECLARES the flag; that is not a read of it.
    found.pop(("roadstead/flags.py", "<module>"), None)

    expected = {k: n for k, (n, _) in _EXPECTED_FLAG_READS.items()}
    unexpected = sorted(set(found) - set(expected))
    assert not unexpected, (
        "a NEW read of `goodput_collapse_enforce` appeared at "
        f"{unexpected}. If it DECIDES whether a request is refused or deferred, "
        "it must not exist — route it through `Health.endpoint_healthy`, which is "
        "the one gate every admission path already funnels through. If it only "
        "REPORTS, add it to _EXPECTED_FLAG_READS with what it reports.")
    stale = sorted(set(expected) - set(found))
    assert not stale, (
        f"_EXPECTED_FLAG_READS lists {stale}, which no longer reads the flag — "
        "drop the entry so this list keeps meaning something. A stale allowlist "
        "is how the guard goes quiet.")
    assert found == expected, (
        "the NUMBER of reads changed inside a function that already read the "
        f"flag:\n  found    {dict(sorted(found.items()))}\n  expected "
        f"{dict(sorted(expected.items()))}\n"
        "An extra read in an existing function is invisible to a key-set check "
        "and is exactly where a new enforcement side effect lands. Say what it "
        "does in _EXPECTED_FLAG_READS and bump the count.")


@pytest.mark.asyncio
async def test_a_sibling_endpoint_is_unaffected():
    state = _state()
    state.backend = _Counters(_wedge_counters(8))
    health = Health(state)
    await _poll(health, _armed_ep(), 6)
    state.flags.set_many({"goodput_collapse_enforce": True})

    assert health.endpoint_healthy("tier3") is False
    assert health.endpoint_healthy("tier2") is True, (
        "the breaker is per endpoint — a set, not a global")


@pytest.mark.asyncio
async def test_an_operator_pause_still_wins():
    """A paused endpoint reads unhealthy regardless, and the poller skips it.
    The goodput arm must not be able to make a drained endpoint read healthy."""
    state = _state(paused_endpoints={"tier3"})
    health = Health(state)
    assert health.endpoint_healthy("tier3") is False


# --------------------------------------------------------------------------- #
# 4. The surfaces an operator reads during the soak
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_the_status_snapshot_carries_the_numbers():
    state = _state()
    state.backend = _Counters(_wedge_counters(8))
    health = Health(state)
    await _poll(health, _armed_ep(), 6)

    snap = health.goodput_snapshot("tier3", 10_060.0)
    assert snap["verdict"] == "collapsed"
    assert snap["tripped"] is True
    assert snap["enforced"] is False       # the soak pair
    assert snap["trips"] == 1
    assert snap["samples"] >= 2
    assert snap["window_s"] > 0
    assert snap["clauses_held"] == ["occupancy", "iterations", "generation",
                                    "prefill"]
    assert snap["running_min"] == 6
    assert snap["prefill_tps"] == 0.0
    assert "tripped_for_s" in snap


def test_an_unevaluated_endpoint_publishes_nothing():
    """The sample-floor precedent: absence means "not watched", never "watched
    and fine". A zeroed dict here would read as verified-healthy."""
    health = Health(_state())
    assert health.goodput_snapshot("tier3", 0.0) is None


@pytest.mark.asyncio
async def test_an_unknown_verdict_names_its_blindness():
    state = _state()
    state.backend = _Counters([None] * 8)
    health = Health(state)
    await _poll(health, _armed_ep(), 4)

    snap = health.goodput_snapshot("tier3", 10_040.0)
    assert snap["verdict"] == "unknown"
    assert snap["reason"].startswith("counters_absent:")
    assert snap["tripped"] is False


# --------------------------------------------------------------------------- #
# 5. Alerts
# --------------------------------------------------------------------------- #

def _alerts(goodput: dict | None, **snap):
    base = {"max_slots": 4, "in_flight": 0, "queued": 0}
    base.update(snap)
    if goodput is not None:
        base["goodput"] = goodput
    return {a.name: a for a in check_alerts(
        endpoint_snapshots={"tier3": base}, agent_budgets=[],
        metrics=_NoMetrics(), cost_model_samples={}, queue_wal_size=0, now=0.0)}


class _NoMetrics:
    """The RollingMetrics surface `check_alerts` reads, and only that.

    Deliberately not a mock that answers anything: a method this stub is missing
    means `check_alerts` grew a new data dependency, and that should surface here
    as an error rather than as a silently different verdict.
    """

    _window_s = 300.0

    def count(self, **kw):
        return 0

    def premature_background_by_call_site(self, **kw):
        return {}

    def per_agent_consumed(self, now):
        return {}


def test_a_shadow_trip_warns_and_an_enforced_trip_is_critical():
    shadow = _alerts({"tripped": True, "enforced": False, "verdict": "collapsed",
                      "clauses_held": ["occupancy"], "tripped_for_s": 30})
    assert shadow["endpoint_goodput_collapse"].severity == "WARNING"
    assert "SHADOW ONLY" in shadow["endpoint_goodput_collapse"].detail

    armed = _alerts({"tripped": True, "enforced": True, "verdict": "collapsed",
                     "clauses_held": ["occupancy"], "tripped_for_s": 30})
    assert armed["endpoint_goodput_collapse"].severity == "CRITICAL"
    assert "SHEDDING" in armed["endpoint_goodput_collapse"].detail


def test_a_configured_but_blind_detector_complains():
    """🚨 Otherwise a dead guard is indistinguishable from a quiet healthy one.

    Refusing a verdict is the SAFE behaviour, which is exactly why it needs its
    own alarm: an endpoint that declares a threshold whose counter its engine
    never publishes will never fire and never complain.
    """
    alerts = _alerts({"tripped": False, "enforced": False, "verdict": "unknown",
                      "reason": "counters_absent:iterations"})
    assert "endpoint_goodput_blind" in alerts
    assert "counters_absent:iterations" in alerts["endpoint_goodput_blind"].detail
    assert "endpoint_goodput_collapse" not in alerts


def test_a_healthy_endpoint_raises_neither():
    alerts = _alerts({"tripped": False, "enforced": False, "verdict": "healthy"})
    assert "endpoint_goodput_collapse" not in alerts
    assert "endpoint_goodput_blind" not in alerts


def test_an_unconfigured_endpoint_raises_neither():
    alerts = _alerts(None)
    assert "endpoint_goodput_collapse" not in alerts
    assert "endpoint_goodput_blind" not in alerts


def test_endpoint_stalled_still_fires_on_its_own_signature():
    """🚨 The nearest neighbour STAYS, and this proves it by firing it.

    `endpoint_stalled` was the closest thing to this feature and could not see the
    2026-09-11/12 wedge: it keys on COMPLETED timeouts and requires `queued == 0`,
    and a collapse that fills every slot makes requests queue. The two detect
    different faults from different evidence, so the test that matters is not
    "the string still appears in the file" — it is that its condition still
    triggers under its own signature, with no goodput data present at all.
    """
    class _Timeouts(_NoMetrics):
        def count(self, **kw):
            # Non-premature, non-best-effort timeouts: its exact predicate.
            return 9

    fired = {a.name: a for a in check_alerts(
        endpoint_snapshots={"tier3": {"max_slots": 4, "in_flight": 1, "queued": 0}},
        agent_budgets=[], metrics=_Timeouts(), cost_model_samples={},
        queue_wal_size=0, now=0.0)}
    assert "endpoint_stalled" in fired, (
        "the pre-existing stall alert was weakened or deleted — it is a different "
        "fault from goodput collapse, not a duplicate of it")
    assert "endpoint_goodput_collapse" not in fired


def test_a_queueing_collapse_is_exactly_what_endpoint_stalled_cannot_see():
    """The complementarity, asserted rather than claimed.

    Same endpoint, same timeout burst, but `queued > 0` — the incident's actual
    shape. `endpoint_stalled` goes silent; the goodput condition speaks. That gap
    is the whole reason this feature exists rather than a threshold tweak to the
    older alert.
    """
    class _Timeouts(_NoMetrics):
        def count(self, **kw):
            return 9

    snap = {"max_slots": 4, "in_flight": 4, "queued": 11,
            "goodput": {"tripped": True, "enforced": False, "verdict": "collapsed",
                        "clauses_held": ["occupancy", "iterations", "generation",
                                         "prefill"], "tripped_for_s": 40}}
    fired = {a.name for a in check_alerts(
        endpoint_snapshots={"tier3": snap}, agent_budgets=[], metrics=_Timeouts(),
        cost_model_samples={}, queue_wal_size=0, now=0.0)}
    assert "endpoint_stalled" not in fired, (
        "queued > 0 is its documented blind spot; if it fires here the two alerts "
        "have become duplicates and one should go")
    assert "endpoint_goodput_collapse" in fired


# --------------------------------------------------------------------------- #
# 6. abort_reason
# --------------------------------------------------------------------------- #

def test_a_goodput_refusal_is_a_backend_fault():
    """A refusal at the door IS the substrate dying under a caller who did
    nothing wrong — the one question `STALL_ABORT_REASONS` answers.

    The distinction is WHOSE FAULT, not who acted: `hard_cap` is also
    proxy-initiated and stays out, because it cuts off work that was HEALTHY.
    """
    assert "goodput_collapse" in PersistentQueue.STALL_ABORT_REASONS
    assert "stall" in PersistentQueue.STALL_ABORT_REASONS
    assert "ttft" in PersistentQueue.STALL_ABORT_REASONS


@pytest.mark.parametrize("reason", [
    "hard_cap", "caller_deadline",
    # The five added 2026-09-12 to fill the NULL hole. Every one of them is a
    # bound WE or the CALLER chose expiring, so counting them would let a
    # capacity decision or an under-budgeting caller masquerade as a backend
    # failure — the exact inversion the lookup exists to prevent.
    "client_deadline", "sse_consumer_deadline", "admission_expiry",
    "queue_deadline_exhausted", "backend_transport_deadline",
])
def test_a_deadline_is_not_a_backend_fault(reason):
    assert reason not in PersistentQueue.STALL_ABORT_REASONS


def test_every_timeout_site_names_a_reason():
    """🚨 THE HOLE THAT MADE THE INCIDENT INVISIBLE.

    219 of 226 real timeout events carried `abort_reason` NULL, so the
    `by_abort_reason` rollup was blind to 97% of the population. Every
    `record_timeout_event` call site now passes one; this asserts it structurally
    rather than trusting the five edits, because the next site added will be
    written by somebody who never read the incident.
    """
    import ast
    import pathlib

    src = pathlib.Path(
        __import__("roadstead.lifecycle", fromlist=["x"]).__file__).read_text()
    tree = ast.parse(src)
    sites = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "record_timeout_event"):
            sites.append(node)
    assert len(sites) >= 6, f"expected every call site to be found, got {len(sites)}"
    missing = [n.lineno for n in sites
               if not any(k.arg == "abort_reason" for k in n.keywords)]
    assert not missing, (
        f"record_timeout_event called with no abort_reason at lifecycle.py lines "
        f"{missing} — a NULL there is a timeout nobody can classify, which is how "
        f"a three-day endpoint wedge stayed invisible")


def test_the_reasons_are_all_distinct():
    """Two sites sharing a reason is the same blindness in a smaller form."""
    import ast
    import pathlib

    src = pathlib.Path(
        __import__("roadstead.lifecycle", fromlist=["x"]).__file__).read_text()
    reasons = []
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "record_timeout_event"):
            for kw in node.keywords:
                if kw.arg == "abort_reason" and isinstance(kw.value, ast.Constant):
                    reasons.append(kw.value.value)
    assert len(reasons) == len(set(reasons)), f"duplicate reasons: {reasons}"
