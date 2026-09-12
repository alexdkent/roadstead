"""Endpoint-level goodput collapse — the monitor, held to its measured precision.

`goodput.py` is pure on purpose, which is what makes a precision claim testable
at all: every assertion below runs in microseconds against a fleet that does not
exist, so the rule can be sabotaged clause by clause instead of trusted.

WHAT THIS FILE IS ORGANISED AROUND. A compound guard passes its own suite with a
clause silently dropped — the conjunction is satisfied by the remaining terms and
nothing goes red. So the load-bearing half of this file is the `test_sabotage_*`
set: one test per clause, each written to FAIL if that clause is deleted from the
rule. The matrix, VERIFIED by deleting each term one at a time and watching which
test goes red:

    clause / safety term          | the test that dies without it
    ------------------------------|----------------------------------------------
    occupancy                     | A PAIR — they pin DIFFERENT things, see below
      …no blind alert when idle    |  test_sabotage_occupancy_an_idle_engine_is_not_collapsed
      …no FALSE TRIP               |  test_sabotage_occupancy_just_below_the_gate_is_not_collapsed
    iterations                    | test_sabotage_iterations_a_turning_loop_is_not_collapsed
    generation                    | test_sabotage_generation_a_producing_engine_is_not_collapsed
    prefill                       | test_sabotage_prefill_the_huge_prompt_caller_is_not_collapsed
    counter-reset guard           | test_a_counter_reset_is_unknown_and_drops_the_stale_samples
    absent-counter ⇒ UNKNOWN      | test_a_configured_counter_the_engine_never_publishes_is_unknown
    blindness does not accumulate | test_an_unknown_between_collapses_resets_the_sustain_counter
    absolute max hold             | test_the_absolute_max_hold_clears_even_while_blind

⚠️ **THE OCCUPANCY ROW NEEDS BOTH TESTS, AND CITING ONLY THE FIRST OVERSTATES IT.**
Deleting the occupancy term makes the idle case (`running=0`) go red with UNKNOWN,
not COLLAPSED — tokens-per-busy-slot is undefined when no slot is busy. So that
test alone pins "an idle proxy raises no blindness alert", which is a real property
and NOT "an idle proxy cannot trip the breaker". The `running=2` sibling is the one
that reddens with a COLLAPSED verdict, and it is what pins the false trip.

⚠️ **Two entries in that table were earned the hard way, and both were green over
the defect first.** `test_sabotage_occupancy_an_idle_engine_is_not_collapsed` found
a real bug (an idle armed endpoint returned UNKNOWN, which would have fired the
blindness alert all night on a healthy proxy). And
`test_an_unknown_between_collapses_resets_the_sustain_counter` passed with the
reset it guards DELETED: it drove the sample ring, and the absent sample lingered
inside the window so no collapsed verdict ever followed the blind one. It drives
the latch directly now, with `test_the_latch_DOES_trip_on_an_uninterrupted_run`
as its positive control — because a null result from an instrument that cannot
produce a positive is not a result.

🚨 **AND THE WORST BREAK OF THE INVARIANT WAS INVISIBLE TO THIS WHOLE FILE.**
Everything here hands the monitor a dict directly, so nothing in it could see that
`backend.probe_progress_counters` coalesced an absent `prompt`/`generation` to
integer 0. Two of the four clauses read exactly those keys, so a MISSING counter
satisfied them: the verdict came back COLLAPSED with an empty `reason`, the breaker
LATCHED, and `roadstead_endpoint_goodput_unknown` read 0. `goodput.py` was correct
throughout; its PRODUCER was not, and "the unit proved the unit, the path never
reached it" is the whole lesson. The regression tests drive the real scraper into
the real monitor — `tests/test_goodput_wiring.py` section 2c.

🚨 EVERY THRESHOLD IN THIS FILE IS A FLEET MEASUREMENT, NOT A DEFAULT. The
numbers (`running >= 3`, `iterations < 1.0/s`, `generation < 2.0/s/req`,
`prefill < 600/s`, sustain 3) were measured on ONE fleet's hardware over 4,264
minutes of engine counters, and they live here and in `docs/api.md` as a worked
example. `roadstead` ships none of them: `GoodputThresholds()` is off, and
`test_no_threshold_ships_as_a_default` is the guard on that.
"""

from __future__ import annotations

import pytest

from roadstead.goodput import (
    CLAUSE_GENERATION,
    CLAUSE_ITERATIONS,
    CLAUSE_OCCUPANCY,
    CLAUSE_PREFILL,
    GoodputMonitor,
    GoodputVerdict,
    GoodputThresholds,
    LatchAction,
    Verdict,
)

# --------------------------------------------------------------------------- #
# The measured rule, as ONE fleet declared it. See the module docstring.
# --------------------------------------------------------------------------- #

FLEET = GoodputThresholds(
    min_running=3,
    sustain_evaluations=3,
    max_iteration_rate=1.0,
    max_generation_tps_per_request=2.0,
    max_prefill_tps=600.0,
)

#: The poller's real cadence, so every window below spans what production spans.
TICK = 10.0


def _feed(mon: GoodputMonitor, endpoint: str, samples: list[dict | None], *,
          t0: float = 1000.0, tick: float = TICK) -> float:
    """Observe a sequence at the poller cadence. Returns the last timestamp."""
    ts = t0
    for i, counters in enumerate(samples):
        ts = t0 + i * tick
        mon.observe(endpoint, ts, counters)
    return ts


def _wedge(n: int = 3, *, running: int = 6) -> list[dict]:
    """The incident, as counters: slots busy, nothing advancing.

    Every cumulative counter frozen — the scheduler step sat at ~7 s filling
    constrained-decoding bitmasks, so the loop barely turned and no tokens came
    out of it. `/health` said `healthy: true` throughout.
    """
    return [{"iterations": 500_000, "generation": 9_000_000,
             "prompt": 80_000_000, "running": running} for _ in range(n)]


def _healthy(n: int = 3, *, running: int = 6) -> list[dict]:
    """A busy engine doing real work: loop turning, tokens coming out."""
    out = []
    for i in range(n):
        out.append({
            "iterations": 500_000 + i * 300,      # 30/s
            "generation": 9_000_000 + i * 1_200,  # 120/s -> 20/s/req at running=6
            "prompt": 80_000_000 + i * 12_000,    # 1,200/s
            "running": running,
        })
    return out


# --------------------------------------------------------------------------- #
# A clean trip, and the evidence it carries
# --------------------------------------------------------------------------- #

def test_a_wedged_engine_is_collapsed():
    mon = GoodputMonitor()
    now = _feed(mon, "tier3", _wedge(3))
    v = mon.evaluate("tier3", FLEET, now)

    assert v.outcome is Verdict.COLLAPSED
    assert set(v.clauses_held) == {
        CLAUSE_OCCUPANCY, CLAUSE_ITERATIONS, CLAUSE_GENERATION, CLAUSE_PREFILL}
    assert v.clauses_cleared == ()
    # The verdict has to carry its own evidence: a soak that cannot read the
    # rates cannot decide whether to arm enforcement.
    assert v.samples == 3
    assert v.window_s == pytest.approx(20.0)
    assert v.running_min == 6
    assert v.iteration_rate == 0.0
    assert v.generation_tps_per_request == 0.0
    assert v.prefill_tps == 0.0


def test_a_busy_working_engine_is_healthy():
    mon = GoodputMonitor()
    now = _feed(mon, "tier3", _healthy(3))
    v = mon.evaluate("tier3", FLEET, now)

    assert v.outcome is Verdict.HEALTHY
    # Occupancy HELD (it is busy) and every progress clause cleared. The split is
    # the point: occupancy holding is normal, and on its own means nothing.
    assert v.clauses_held == (CLAUSE_OCCUPANCY,)
    assert set(v.clauses_cleared) == {
        CLAUSE_ITERATIONS, CLAUSE_GENERATION, CLAUSE_PREFILL}


# --------------------------------------------------------------------------- #
# SABOTAGE — one test per clause. Each FAILS if its clause is deleted.
# --------------------------------------------------------------------------- #

def test_sabotage_occupancy_an_idle_engine_is_not_collapsed():
    """🚨 THE ONE MEASURABLY LOAD-BEARING TERM.

    An IDLE engine trivially satisfies all three progress clauses: nothing is
    running, so nothing is being produced. Ablation over the same 4,264 minutes
    put precision at **0.425 (415 false positives)** without this gate, against
    1.000 with it — every other clause ablation left precision at 0.997-1.000.

    Delete the occupancy term and this test goes red: `running=0` with every
    counter frozen is exactly a quiet proxy at 04:00.

    ⚠️ This test found a real defect on the way in. Falling through to the
    progress clauses at `running=0` makes tokens-per-busy-slot undefined, so the
    verdict came back UNKNOWN — which is not merely imprecise: it would have
    published `roadstead_endpoint_goodput_unknown=1` and fired
    `endpoint_goodput_blind` continuously on every idle armed endpoint, calling a
    known-good state unmeasured. The occupancy gate now short-circuits, which is
    safe in the one direction that matters (the conjunction is already refuted).
    The `running=2` case below is the one that dies with a COLLAPSED verdict.
    """
    mon = GoodputMonitor()
    now = _feed(mon, "tier3", _wedge(3, running=0))
    v = mon.evaluate("tier3", FLEET, now)

    assert v.outcome is Verdict.HEALTHY, (
        "an IDLE engine must never read as collapsed — this is the 0.425-precision "
        "failure, and it is what the occupancy gate is for")
    assert v.clauses_cleared == (CLAUSE_OCCUPANCY,)
    assert v.running_min == 0
    # An idle endpoint is HEALTHY, not blind. The distinction is what keeps the
    # blindness alert meaningful.
    assert v.outcome is not Verdict.UNKNOWN and not v.reason


def test_sabotage_occupancy_just_below_the_gate_is_not_collapsed():
    """The boundary, so the gate cannot be widened without a red test.

    `running=2` against `min_running=3` on an otherwise perfectly wedged engine.
    A gate rewritten `> 0` or `>= 1` passes every other test in this file.
    """
    mon = GoodputMonitor()
    now = _feed(mon, "tier3", _wedge(3, running=2))
    assert mon.evaluate("tier3", FLEET, now).outcome is Verdict.HEALTHY


def test_sabotage_iterations_a_turning_loop_is_not_collapsed():
    """The loop IS turning: the engine is scheduling, it just is not emitting.

    Mechanism, not measured precision — ablating this clause left precision at
    0.997-1.000, and the honest reason it exists is that "loop stopped" and "loop
    turning, producing nothing" are different faults with different fixes.

    Delete the iterations term and this goes red: everything else here reads as
    a wedge.
    """
    mon = GoodputMonitor()
    # 30 iterations/s — well above the 1.0/s ceiling — and nothing else moving.
    samples = [{"iterations": 500_000 + i * 300, "generation": 9_000_000,
                "prompt": 80_000_000, "running": 6} for i in range(3)]
    now = _feed(mon, "tier3", samples)
    v = mon.evaluate("tier3", FLEET, now)

    assert v.outcome is Verdict.HEALTHY
    assert v.clauses_cleared == (CLAUSE_ITERATIONS,)
    assert v.iteration_rate == pytest.approx(30.0)


def test_sabotage_generation_a_producing_engine_is_not_collapsed():
    """Tokens ARE coming out, and the other two clauses say otherwise.

    Prefill frozen and the loop barely turning while generation runs at
    20 tok/s/request is not a shape any real engine holds for long — which is the
    point of a sabotage case. It exists to make the generation term load-bearing
    in the suite, not to model a workload.
    """
    mon = GoodputMonitor()
    samples = [{"iterations": 500_000, "generation": 9_000_000 + i * 1_200,
                "prompt": 80_000_000, "running": 6} for i in range(3)]
    now = _feed(mon, "tier3", samples)
    v = mon.evaluate("tier3", FLEET, now)

    assert v.outcome is Verdict.HEALTHY
    assert v.clauses_cleared == (CLAUSE_GENERATION,)
    assert v.generation_tps_per_request == pytest.approx(20.0)


def test_sabotage_prefill_the_huge_prompt_caller_is_not_collapsed():
    """🚨 THE FALSE-POSITIVE CONTROL, FROM REAL TRAFFIC — the `hermes` caller.

    Measured, not invented. `hermes` sends **31-37k-token prompts** and gets
    **2-3 output tokens** back. Six such requests concurrently read **0.42
    generation tokens/sec/busy-slot** — deep inside the wedge band, below the
    2.0 ceiling — while being perfectly healthy, because almost all of the work
    is PREFILL. On a whole-request average it is indistinguishable from the
    incident by occupancy and generation alone.

    What separates them is the prefill rate: hermes drives **1,209 prompt
    tokens/sec**, while wedge minutes never exceeded **247.5** and healthy busy
    minutes had a median of 992. This clause is the only thing standing between
    that caller and a shed endpoint.

    Delete the prefill term and this test goes red. It is also the reason
    `CostModel.decode_tps` was rejected as the signal: a prefill-dominated
    workload inflates it, and a decode-rate guard built on it was already
    removed from `scheduler._admit` once for exactly this.
    """
    mon = GoodputMonitor()
    samples = []
    for i in range(3):
        samples.append({
            # The loop IS turning slowly in wall-clock terms during a long
            # prefill, so leave it under the ceiling: this control must be saved
            # by PREFILL, not rescued by a second clause.
            "iterations": 500_000 + i * 4,          # 0.4/s — below 1.0
            "generation": 9_000_000 + int(i * 25),  # 2.5/s over 6 slots = 0.42/s/req
            "prompt": 80_000_000 + i * 12_090,      # 1,209/s of real prefill
            "running": 6,
        })
    now = _feed(mon, "tier3", samples)
    v = mon.evaluate("tier3", FLEET, now)

    assert v.generation_tps_per_request == pytest.approx(0.42, abs=0.01), (
        "the control has to actually sit inside the wedge band, or it proves "
        "nothing about the prefill clause")
    assert v.iteration_rate == pytest.approx(0.4)
    assert v.prefill_tps == pytest.approx(1209.0)
    assert v.outcome is Verdict.HEALTHY, (
        "hermes is HEALTHY: huge prompts, tiny outputs, and 1,209 prompt "
        "tokens/sec of real prefill work")
    assert v.clauses_cleared == (CLAUSE_PREFILL,), (
        "prefill must be the ONLY clause clearing this — if another one also "
        "cleared, the test would stay green with the prefill clause deleted")


def test_the_wedge_and_the_prefill_control_differ_only_in_prefill():
    """The two populations, side by side, so the discriminator is explicit.

    Both are six busy slots producing almost no output tokens. One is the
    incident and one is a healthy caller, and the ONLY term that tells them
    apart is prefill — which is the whole argument for keeping that clause.
    """
    mon = GoodputMonitor()
    wedge_now = _feed(mon, "wedge", _wedge(3))
    wedge = mon.evaluate("wedge", FLEET, wedge_now)

    ctrl_samples = [{"iterations": 500_000 + i * 4,
                     "generation": 9_000_000 + int(i * 25),
                     "prompt": 80_000_000 + i * 12_090,
                     "running": 6} for i in range(3)]
    ctrl_now = _feed(mon, "hermes", ctrl_samples)
    ctrl = mon.evaluate("hermes", FLEET, ctrl_now)

    assert wedge.running_min == ctrl.running_min == 6
    assert wedge.generation_tps_per_request < FLEET.max_generation_tps_per_request
    assert ctrl.generation_tps_per_request < FLEET.max_generation_tps_per_request
    assert wedge.prefill_tps < FLEET.max_prefill_tps < ctrl.prefill_tps
    assert (wedge.outcome, ctrl.outcome) == (Verdict.COLLAPSED, Verdict.HEALTHY)


# --------------------------------------------------------------------------- #
# THE INVARIANT — blindness must never read as collapse
# --------------------------------------------------------------------------- #

def test_too_few_samples_is_unknown_not_collapsed():
    """One sample is not a rate. A fresh process must not shed traffic."""
    mon = GoodputMonitor()
    now = _feed(mon, "tier3", _wedge(1))
    v = mon.evaluate("tier3", FLEET, now)
    assert v.outcome is Verdict.UNKNOWN
    assert v.reason == "too_few_samples"


def test_a_window_too_short_to_measure_is_unknown():
    """Two samples 0.5 s apart divide a tiny delta by a tiny dt.

    Without the span floor, a pair of near-simultaneous samples on a working
    engine can compute an arbitrarily small rate and look wedged.
    """
    mon = GoodputMonitor()
    now = _feed(mon, "tier3", _wedge(2), tick=0.5)
    v = mon.evaluate("tier3", FLEET, now)
    assert v.outcome is Verdict.UNKNOWN
    assert v.reason == "window_too_short"


def test_a_counter_reset_is_unknown_and_drops_the_stale_samples():
    """A backend RESTART makes a cumulative counter go backwards.

    Two wrong answers are available and both are worse than refusing: a negative
    rate (which satisfies every `<` clause, so the breaker trips on a restart)
    and a huge one. The stale samples must also go, or the next evaluation
    straddles the restart too.
    """
    mon = GoodputMonitor()
    pre = _healthy(3)
    restarted = {"iterations": 12, "generation": 40, "prompt": 900, "running": 6}
    now = _feed(mon, "tier3", [*pre, restarted])
    v = mon.evaluate("tier3", FLEET, now)

    assert v.outcome is Verdict.UNKNOWN
    assert v.reason == "counter_reset"
    # The straddling history is gone; only the post-restart sample survives, so
    # the very next evaluation is "too few samples", never a negative rate.
    assert mon.evaluate("tier3", FLEET, now).reason == "too_few_samples"


def test_a_configured_counter_the_engine_never_publishes_is_unknown():
    """🚨 THE INVARIANT IN ITS SHARPEST FORM.

    A llama.cpp endpoint publishes no iteration counter. If the operator
    nonetheless declares `max_iteration_rate`, the conjunction has a term that
    cannot be evaluated — and because it is a CONJUNCTION, silently skipping it
    makes firing EASIER. It must be UNKNOWN, and specifically not COLLAPSED:
    every OTHER clause here holds, so a permissive reading trips the breaker on
    an endpoint whose instrument is simply missing.
    """
    mon = GoodputMonitor()
    samples = [{"iterations": None, "generation": 9_000_000,
                "prompt": 80_000_000, "running": 6} for _ in range(3)]
    now = _feed(mon, "tier3", samples)
    v = mon.evaluate("tier3", FLEET, now)

    assert v.outcome is Verdict.UNKNOWN, (
        "a missing instrument must not be able to trip the breaker")
    assert v.outcome is not Verdict.HEALTHY, (
        "…and must not read as verified-healthy either")
    assert v.reason == "counters_absent:iterations"


def test_the_same_endpoint_evaluates_fine_without_that_clause_declared():
    """The other half of the case above: llama.cpp CAN run the rule.

    Leave `max_iteration_rate` undeclared and the endpoint evaluates the two
    clauses its engine can support. That is why "which clauses" is a
    configuration decision rather than an engine one.
    """
    llamacpp = GoodputThresholds(
        min_running=3, sustain_evaluations=3,
        max_generation_tps_per_request=2.0, max_prefill_tps=600.0)
    assert llamacpp.progress_clauses == (CLAUSE_GENERATION, CLAUSE_PREFILL)

    mon = GoodputMonitor()
    samples = [{"iterations": None, "generation": 9_000_000,
                "prompt": 80_000_000, "running": 6} for _ in range(3)]
    now = _feed(mon, "tier3", samples)
    assert mon.evaluate("tier3", llamacpp, now).outcome is Verdict.COLLAPSED


def test_a_failed_scrape_is_unknown_not_collapsed():
    """`probe_progress_counters` returning None is an unreachable backend.

    Recorded as an ABSENT sample rather than dropped: dropping it would leave a
    stale window standing and let a dead backend keep reading healthy.
    """
    mon = GoodputMonitor()
    now = _feed(mon, "tier3", [None, None, None])
    v = mon.evaluate("tier3", FLEET, now)
    assert v.outcome is Verdict.UNKNOWN
    assert v.reason.startswith("counters_absent:")


def test_a_counter_that_blinked_out_mid_window_is_unknown():
    """Absent at the ENDPOINTS is not the only way to be blind.

    A first/last delta across a gap nobody measured is not an observation, so ANY
    absent sample in the window disqualifies the key — not just the two the rate
    is computed from.
    """
    mon = GoodputMonitor()
    good = _wedge(1)[0]
    blind = dict(good, running=None)
    now = _feed(mon, "tier3", [good, blind, good])
    v = mon.evaluate("tier3", FLEET, now)
    assert v.outcome is Verdict.UNKNOWN
    assert "running" in v.reason


def test_an_unconfigured_endpoint_is_unknown_and_never_collapses():
    """Absent thresholds mean the detector does not run.

    The public default has to be OFF: this repository ships mechanism, and the
    five numbers are one fleet's hardware measurements.
    """
    mon = GoodputMonitor()
    now = _feed(mon, "tier3", _wedge(4))
    v = mon.evaluate("tier3", GoodputThresholds(), now)
    assert v.outcome is Verdict.UNKNOWN
    assert v.reason == "not_configured"


def test_no_threshold_ships_as_a_default():
    """🚨 The guard on "absent ⇒ feature off".

    A default that arms the detector would put one fleet's hardware constants in
    every deployment, and — worse — make "we never configured it" look exactly
    like "it is watching".
    """
    bare = GoodputThresholds()
    assert not bare.armed
    assert bare.min_running == 0
    assert bare.sustain_evaluations == 0
    assert bare.progress_clauses == ()
    assert bare.max_iteration_rate is None
    assert bare.max_generation_tps_per_request is None
    assert bare.max_prefill_tps is None


@pytest.mark.parametrize("partial, why", [
    (GoodputThresholds(sustain_evaluations=3, max_prefill_tps=600.0),
     "no occupancy gate — the one measurably load-bearing term"),
    (GoodputThresholds(min_running=3, max_prefill_tps=600.0),
     "no sustain count — a single evaluation is one sample pair"),
    (GoodputThresholds(min_running=3, sustain_evaluations=3),
     "no progress clause — occupancy alone fires on every busy endpoint"),
])
def test_a_partial_configuration_does_not_arm(partial, why):
    assert not partial.armed, why


# --------------------------------------------------------------------------- #
# The latch: sustain, recovery, and the absolute ceiling
# --------------------------------------------------------------------------- #

def _step(mon, verdicts_needed: int, *, endpoint="tier3", tripped=False,
          thresholds=FLEET, t0=2000.0, samples=None):
    """Drive `sustain` collapsed evaluations and return the actions seen."""
    actions = []
    ts = t0
    for _ in range(verdicts_needed):
        ts += TICK
        mon.observe(endpoint, ts, (samples or _wedge(1))[0])
        v = mon.evaluate(endpoint, thresholds, ts)
        actions.append(mon.step(endpoint, v, tripped=tripped,
                                thresholds=thresholds, now=ts))
        if actions[-1] is LatchAction.TRIP:
            tripped = True
    return actions, ts


def test_the_trip_needs_the_sustain_count_not_one_evaluation():
    mon = GoodputMonitor()
    actions, _ = _step(mon, 5)
    # Two evaluations are UNKNOWN (ring filling / window too short), then three
    # COLLAPSED in a row earn the trip. The count is what matters: exactly one
    # TRIP, and it is not the first action.
    assert actions.count(LatchAction.TRIP) == 1
    assert actions[0] is LatchAction.NONE
    assert mon.trips("tier3") == 1


def _verdict(outcome: Verdict) -> GoodputVerdict:
    """A bare verdict, for driving the LATCH without going through the ring.

    The sustain/recovery property is about the sequence of OUTCOMES, so feeding
    them directly is the honest unit. ⚠️ It is also the only way to state it:
    the first cut of the test below drove the ring instead, and an absent sample
    lingering inside a wide rate window made every LATER evaluation UNKNOWN too —
    so the scenario never delivered a collapsed verdict after the blind one and
    passed with the reset deleted. See the positive control below.
    """
    return GoodputVerdict(outcome, samples=3, window_s=20.0)


def test_an_unknown_between_collapses_resets_the_sustain_counter():
    """🚨 Blindness cannot ACCUMULATE towards a trip.

    The conjunction gets easier to satisfy the less we can see, so a run of
    collapsed evaluations broken by blindness is not a sustained collapse:
    collapsed, collapsed, BLIND, collapsed, collapsed must not trip a sustain of 3.
    """
    mon = GoodputMonitor()
    sequence = [Verdict.COLLAPSED, Verdict.COLLAPSED, Verdict.UNKNOWN,
                Verdict.COLLAPSED, Verdict.COLLAPSED]
    for i, outcome in enumerate(sequence):
        action = mon.step("tier3", _verdict(outcome), tripped=False,
                          thresholds=FLEET, now=4000.0 + i * TICK)
        assert action is not LatchAction.TRIP, (
            f"tripped at step {i} ({outcome}) — a collapse run interrupted by "
            f"blindness is not sustained, and a conjunction nobody could evaluate "
            f"must never count towards firing")
    assert mon.trips("tier3") == 0


def test_the_latch_DOES_trip_on_an_uninterrupted_run():
    """🚨 THE POSITIVE CONTROL for the test above.

    Without it, that test passes on a latch that can never trip at all — which is
    exactly how its first cut passed with the reset it was guarding deleted. Same
    length of collapsed run, no blind tick in the middle: this one must trip.
    """
    mon = GoodputMonitor()
    actions = [
        mon.step("tier3", _verdict(Verdict.COLLAPSED), tripped=False,
                 thresholds=FLEET, now=4000.0 + i * TICK)
        for i in range(3)
    ]
    assert actions[-1] is LatchAction.TRIP, actions
    assert mon.trips("tier3") == 1


def test_recovery_clears_the_trip():
    mon = GoodputMonitor()
    actions, ts = _step(mon, 5)
    assert LatchAction.TRIP in actions

    cleared = None
    healthy = _healthy(6)
    for i, counters in enumerate(healthy):
        ts += TICK
        mon.observe("tier3", ts, counters)
        v = mon.evaluate("tier3", FLEET, ts)
        action = mon.step("tier3", v, tripped=cleared is None,
                          thresholds=FLEET, now=ts)
        if action is LatchAction.CLEAR:
            cleared = i
            break
    assert cleared is not None, "a recovered endpoint must clear"
    assert mon.tripped_for_s("tier3", ts) is None


def test_an_unknown_is_not_recovery_evidence():
    """Blindness does not clear a trip either — it is not evidence of anything.

    The absolute ceiling below is what stops that from being a latch.
    """
    mon = GoodputMonitor()
    _, ts = _step(mon, 5)
    for _ in range(20):
        ts += TICK
        mon.observe("tier3", ts, None)
        v = mon.evaluate("tier3", FLEET, ts)
        assert v.outcome is Verdict.UNKNOWN
        action = mon.step("tier3", v, tripped=True, thresholds=FLEET, now=ts)
        assert action is not LatchAction.CLEAR, (
            "a blind tick is not a healthy tick")


def test_the_absolute_max_hold_clears_even_while_blind():
    """🚨 THERE MUST BE NO WAY TO STAY TRIPPED FOREVER.

    A latched trip on a false positive is an outage we caused, and the test
    above establishes that blindness alone will never clear one. The ceiling is
    what closes that: it fires regardless of the verdict, including UNKNOWN.

    It is not a guess at how long a wedge lasts — a wedge that is still wedged
    re-trips within `sustain` poller ticks, which is cheap. Holding on a claim
    nobody has re-confirmed is not.
    """
    thresholds = GoodputThresholds(
        min_running=3, sustain_evaluations=3, max_iteration_rate=1.0,
        max_generation_tps_per_request=2.0, max_prefill_tps=600.0,
        max_hold_s=60.0)
    mon = GoodputMonitor()
    actions, ts = _step(mon, 5, thresholds=thresholds)
    assert LatchAction.TRIP in actions

    # Blind from here on, so nothing but the ceiling can clear it.
    cleared_at = None
    for _ in range(20):
        ts += TICK
        mon.observe("tier3", ts, None)
        v = mon.evaluate("tier3", thresholds, ts)
        if mon.step("tier3", v, tripped=True, thresholds=thresholds,
                    now=ts) is LatchAction.CLEAR:
            cleared_at = ts
            break
    assert cleared_at is not None, "the absolute ceiling did not fire"


def test_a_still_wedged_endpoint_re_trips_after_the_ceiling_releases_it():
    """The ceiling is a release, not an amnesty.

    This is what makes bounding the hold safe: if the wedge is still there, the
    rule re-earns the trip from fresh evidence within `sustain` ticks.
    """
    thresholds = GoodputThresholds(
        min_running=3, sustain_evaluations=3, max_iteration_rate=1.0,
        max_generation_tps_per_request=2.0, max_prefill_tps=600.0,
        max_hold_s=30.0)
    mon = GoodputMonitor()
    ts = 5000.0
    tripped = False
    trips_seen = 0
    clears_seen = 0
    for _ in range(30):
        ts += TICK
        mon.observe("tier3", ts, _wedge(1)[0])
        v = mon.evaluate("tier3", thresholds, ts)
        action = mon.step("tier3", v, tripped=tripped, thresholds=thresholds,
                          now=ts)
        if action is LatchAction.TRIP:
            tripped, trips_seen = True, trips_seen + 1
        elif action is LatchAction.CLEAR:
            tripped, clears_seen = False, clears_seen + 1
    assert trips_seen >= 2, "a persistent wedge must re-trip after a ceiling release"
    assert clears_seen >= 1


def test_forget_drops_everything_about_an_endpoint():
    mon = GoodputMonitor()
    _step(mon, 5)
    assert mon.trips("tier3") == 1
    mon.forget("tier3")
    assert mon.trips("tier3") == 0
    assert mon.evaluate("tier3", FLEET, 9999.0).reason == "no_samples"


# --------------------------------------------------------------------------- #
# Arithmetic details that produce a wrong number rather than an error
# --------------------------------------------------------------------------- #

def test_occupancy_uses_the_MINIMUM_running_across_the_window():
    """A momentary spike must not arm the gate.

    Samples at running 0, 0, 6 average to 2 and peak at 6; neither reading is
    "the engine was busy for this window". The minimum is, and it is the
    conservative one — which is the right bias for a term whose job is to stop
    an idle engine from tripping the breaker.
    """
    mon = GoodputMonitor()
    samples = [dict(_wedge(1)[0], running=r) for r in (0, 0, 6)]
    now = _feed(mon, "tier3", samples)
    v = mon.evaluate("tier3", FLEET, now)
    assert v.running_min == 0
    assert v.running_mean == pytest.approx(2.0)
    assert v.outcome is Verdict.HEALTHY


def test_the_per_request_divisor_is_mean_running_not_a_slot_count():
    """Tokens produced over an interval are divided by the occupancy DURING it.

    Using `max_slots` would answer a different question (utilisation), and using
    the instantaneous last reading would divide an interval's tokens by a moment.
    """
    mon = GoodputMonitor()
    samples = []
    for i in range(3):
        samples.append({"iterations": 500_000, "generation": 9_000_000 + i * 40,
                        "prompt": 80_000_000, "running": 4 if i else 8})
    now = _feed(mon, "tier3", samples)
    v = mon.evaluate("tier3", FLEET, now)
    # 80 tokens over 20 s = 4.0 tok/s; mean running = (8+4+4)/3 = 5.33
    assert v.generation_tps == pytest.approx(4.0)
    assert v.running_mean == pytest.approx(5.33, abs=0.01)
    assert v.generation_tps_per_request == pytest.approx(0.75, abs=0.01)


def test_samples_outside_the_rate_window_are_not_used():
    """The window is TRAILING, so stale history cannot dilute a fresh wedge.

    A long averaging window would blend the healthy work that preceded a collapse
    into the rate and delay detection — and the value of this detector is that it
    fired two minutes after the first timeout.
    """
    mon = GoodputMonitor(rate_window_s=25.0)
    now = _feed(mon, "tier3", [*_healthy(3), *_wedge(3)])
    v = mon.evaluate("tier3", FLEET, now)
    assert v.samples == 3, "only the trailing window may contribute"
    assert v.outcome is Verdict.COLLAPSED
