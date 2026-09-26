"""The chunked-prefill busy clause (`goodput.CLAUSE_ENGINE_IDLE`) — rule + wiring.

`tests/test_goodput.py` and `tests/test_goodput_wiring.py` hold the original
four-clause rule to its measured precision and prove its wiring. This file is
the same split for the fifth, OPTIONAL clause, added 2026-09-26 after a fleet
measurement: vLLM only emits its iteration/token counters on an
output-producing scheduler step, so during a long CHUNKED PREFILL every one of
those four clauses reads exactly like the original wedge on a perfectly
healthy engine. GPU power, read from an exporter with no connection to the
inference engine, was the one signal that told the two apart. See the
"CHUNKED PREFILL IS A SECOND, UNRELATED WAY TO GO BLIND" section of
`goodput.py`'s module docstring.

Kept in its own file, not folded into the two above, so it is obvious on sight
that `test_goodput.py`/`test_goodput_wiring.py`'s FLEET/FLEET_POLICY constants
are UNCHANGED and every one of their assertions still describes the original
four-clause rule with the busy clause undeclared — those two files staying
green, unmodified in substance (`tests/test_goodput_wiring.py`'s two
passthrough-count tests were widened, not weakened — see their own diff), is
itself the proof of requirement 6: undeclared behaves byte-for-byte as before.

🚨 EVERY BUSY NUMBER IN THIS FILE IS A FLEET MEASUREMENT, NOT A DEFAULT — same
rule as `test_goodput.py`. `busy_min=20.0` and the readings used below (60.0
inside the healthy decode/prefill band, 10.0 inside the measured 5-18W wedge
band) are a worked example; `roadstead` ships none of them
(`GoodputThresholds().busy_min == 0.0`, covered by
`test_goodput_wiring.py::test_no_threshold_ships_as_a_default`'s sibling for
this field).
"""

from __future__ import annotations

import json
import types

import pytest

from roadstead.backend import BackendClientPool
from roadstead.config import EndpointConfig
from roadstead.flags import RuntimeFlags
from roadstead.goodput import (
    CLAUSE_ENGINE_IDLE,
    CLAUSE_GENERATION,
    CLAUSE_ITERATIONS,
    CLAUSE_OCCUPANCY,
    CLAUSE_PREFILL,
    GoodputMonitor,
    GoodputThresholds,
    LatchAction,
    Verdict,
)
from roadstead.health import Health

#: 🚨 Captured UNBOUND at import, before the autouse `_no_network_probes`
#: fixture stubs `probe_busy_gauge` on the CLASS (see tests/conftest.py's
#: STUBBED_PROBES — it now covers this probe too). A bound call made from
#: inside a test body would exercise the STUB, not the parser, and every
#: assertion in this file's parser section would pass for the wrong reason —
#: same trap `test_goodput_wiring.py` documents for `_REAL_COUNTERS`.
_REAL_BUSY_GAUGE = BackendClientPool.probe_busy_gauge

# --------------------------------------------------------------------------- #
# The measured fleet rule, WITH the busy floor declared — same shape as
# test_goodput.py's FLEET, plus busy_min.
# --------------------------------------------------------------------------- #

BUSY_FLEET = GoodputThresholds(
    min_running=3,
    sustain_evaluations=3,
    max_iteration_rate=1.0,
    max_generation_tps_per_request=2.0,
    max_prefill_tps=600.0,
    busy_min=20.0,
)

TICK = 10.0

#: RFC 5737 documentation address — never a real, dialable host. See
#: test_scrub_sweep.py.
BUSY_URL = "http://192.0.2.10:9400/metrics"
BUSY_METRIC = "gpu_power_watts"


def _feed(mon: GoodputMonitor, endpoint: str, samples: list[dict | None], *,
          t0: float = 1000.0, tick: float = TICK) -> float:
    ts = t0
    for i, counters in enumerate(samples):
        ts = t0 + i * tick
        mon.observe(endpoint, ts, counters)
    return ts


def _drive(mon: GoodputMonitor, endpoint: str, thresholds: GoodputThresholds,
           rows: list[dict | None], *, t0: float = 2000.0, tick: float = TICK):
    """Observe+evaluate+step one row at a time. Returns (actions, verdicts)."""
    ts = t0
    actions: list[LatchAction] = []
    verdicts = []
    tripped = False
    for row in rows:
        ts += tick
        mon.observe(endpoint, ts, row)
        v = mon.evaluate(endpoint, thresholds, ts)
        verdicts.append(v)
        a = mon.step(endpoint, v, tripped=tripped, thresholds=thresholds, now=ts)
        if a is LatchAction.TRIP:
            tripped = True
        elif a is LatchAction.CLEAR:
            tripped = False
        actions.append(a)
    return actions, verdicts


def _wedge_shaped(n: int = 3, *, running: int = 6,
                   busy: float | None = None) -> list[dict]:
    """Every cumulative counter frozen, occupancy busy — the shape a REAL wedge
    and a healthy chunked-prefill step both produce. `busy=None` omits the key
    entirely (an undeclared clause, or a scrape that failed outright)."""
    out = []
    for _ in range(n):
        row = {"iterations": 500_000, "generation": 9_000_000,
               "prompt": 80_000_000, "running": running}
        if busy is not None:
            row["busy"] = busy
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# THE CONTROL — chunked prefill must never read as collapse
# --------------------------------------------------------------------------- #

def test_sabotage_engine_idle_a_chunked_prefill_step_is_not_collapsed():
    """🚨 THE CONTROL THIS CLAUSE EXISTS FOR.

    Every counter the first four clauses read is EXACTLY the wedge shape
    (occupancy busy, iterations/generation/prefill all frozen) — a long vLLM
    chunked-prefill step produces that shape on a perfectly healthy engine.
    GPU power is what tells the two apart: measured on one fleet, real wedges
    read 5-18W and idle reads 12W, while decode reads 33-48W and cold prefill
    reads 58-64W. A busy reading inside the decode/prefill band must clear
    this clause on every evaluation, and the endpoint must never trip.

    SABOTAGED by hand 2026-09-26 (forced ENGINE_IDLE to always HOLD regardless
    of `busy_max`, mirroring how the other four clause sabotages are verified):
    this test failed as designed (COLLAPSED instead of HEALTHY,
    `clauses_cleared=()`), along with four siblings that also depend on the
    clause clearing correctly (`test_the_wedge_and_the_chunked_prefill_
    control_differ_only_in_busy`, both single-sample MAX-rule tests, and the
    end-to-end `test_a_well_formed_busy_reading_reaches_health_and_clears_it`).
    `test_goodput.py` and `test_goodput_wiring.py` stayed fully green
    throughout — the sabotage could not touch the original four-clause rule.
    Reverted immediately after.
    """
    mon = GoodputMonitor()
    actions, verdicts = _drive(mon, "tier3", BUSY_FLEET,
                               _wedge_shaped(8, busy=60.0))
    assert LatchAction.TRIP not in actions, (
        "chunked prefill tripped the breaker — the busy clause failed to save "
        "the control it exists for")
    settled = [v for v in verdicts if v.outcome is not Verdict.UNKNOWN]
    assert settled, "the ring/window never settled — the control proves nothing"
    assert all(v.outcome is Verdict.HEALTHY for v in settled)
    assert all(CLAUSE_ENGINE_IDLE in v.clauses_cleared for v in settled)
    assert all(v.busy_max == pytest.approx(60.0) for v in settled)


def test_a_real_wedge_with_low_busy_still_trips():
    """The other half of the control: a GENUINE wedge (busy inside the
    measured 5-18W band) must still collapse and trip after `sustain`."""
    mon = GoodputMonitor()
    actions, verdicts = _drive(mon, "tier3", BUSY_FLEET,
                               _wedge_shaped(8, busy=10.0))
    assert LatchAction.TRIP in actions
    collapsed = [v for v in verdicts if v.outcome is Verdict.COLLAPSED]
    assert collapsed
    assert set(collapsed[0].clauses_held) == {
        CLAUSE_OCCUPANCY, CLAUSE_ITERATIONS, CLAUSE_GENERATION, CLAUSE_PREFILL,
        CLAUSE_ENGINE_IDLE}
    assert collapsed[0].busy_max == pytest.approx(10.0)


def test_the_wedge_and_the_chunked_prefill_control_differ_only_in_busy():
    """Side by side: same frozen counters, same occupancy — only the external
    busy reading tells the two populations apart."""
    mon = GoodputMonitor()
    wedge_now = _feed(mon, "wedge", _wedge_shaped(3, busy=10.0))
    wedge = mon.evaluate("wedge", BUSY_FLEET, wedge_now)

    ctrl_now = _feed(mon, "prefill", _wedge_shaped(3, busy=60.0))
    ctrl = mon.evaluate("prefill", BUSY_FLEET, ctrl_now)

    assert wedge.running_min == ctrl.running_min == 6
    assert wedge.iteration_rate == ctrl.iteration_rate == 0.0
    assert wedge.busy_max < BUSY_FLEET.busy_min <= ctrl.busy_max
    assert (wedge.outcome, ctrl.outcome) == (Verdict.COLLAPSED, Verdict.HEALTHY)


# --------------------------------------------------------------------------- #
# THE MAX RULE — one busy sample anywhere in the window is enough
# --------------------------------------------------------------------------- #

def test_a_single_high_busy_sample_inside_the_window_clears_the_clause():
    """A flicker to a busy reading mid-window is proof the engine did real
    work at SOME point in it — MIN would ask a different, wrong question
    (whether it was busy for the WHOLE window) and would fail this exact
    chunked-prefill case the moment the step ends and generation resumes."""
    mon = GoodputMonitor()
    rows = (_wedge_shaped(1, busy=10.0) + _wedge_shaped(1, busy=60.0)
            + _wedge_shaped(1, busy=10.0))
    now = _feed(mon, "tier3", rows)
    v = mon.evaluate("tier3", BUSY_FLEET, now)
    assert v.busy_max == pytest.approx(60.0), (
        "the window's MAX must win even though two of three samples are low")
    assert CLAUSE_ENGINE_IDLE in v.clauses_cleared
    assert v.outcome is Verdict.HEALTHY


def test_a_single_low_busy_sample_does_not_clear_a_run_of_high_ones():
    """The mirror case: MAX means one low sample cannot rescue a genuine wedge
    that happened to flicker busy for one tick — a single high reading is
    proof of real work, but a single LOW reading is not proof of a wedge on
    its own (min_running/iterations/generation/prefill still decide that)."""
    mon = GoodputMonitor()
    rows = (_wedge_shaped(1, busy=60.0) + _wedge_shaped(1, busy=10.0)
            + _wedge_shaped(1, busy=60.0))
    now = _feed(mon, "tier3", rows)
    v = mon.evaluate("tier3", BUSY_FLEET, now)
    assert v.busy_max == pytest.approx(60.0)
    assert CLAUSE_ENGINE_IDLE in v.clauses_cleared
    assert v.outcome is Verdict.HEALTHY


# --------------------------------------------------------------------------- #
# THE INVARIANT — blindness must never read as collapse, for this clause too
# --------------------------------------------------------------------------- #

def test_a_declared_busy_clause_with_no_reading_at_all_is_unknown():
    """Same invariant as the other four clauses: declared, but the sample
    carries no `busy` key (the scrape failed outright) — UNKNOWN, never
    COLLAPSED, and it must not be able to trip."""
    mon = GoodputMonitor()
    now = _feed(mon, "tier3", _wedge_shaped(3, busy=None))
    v = mon.evaluate("tier3", BUSY_FLEET, now)
    assert v.outcome is Verdict.UNKNOWN
    assert v.reason == "counters_absent:busy"


def test_a_busy_reading_missing_from_just_one_sample_mid_window_is_unknown():
    """Absent at an ENDPOINT of the window is not the only way to be blind —
    ANY absent sample disqualifies the key, same rule as the other clauses."""
    mon = GoodputMonitor()
    rows = (_wedge_shaped(1, busy=60.0) + _wedge_shaped(1, busy=None)
            + _wedge_shaped(1, busy=60.0))
    now = _feed(mon, "tier3", rows)
    v = mon.evaluate("tier3", BUSY_FLEET, now)
    assert v.outcome is Verdict.UNKNOWN
    assert "busy" in v.reason


def test_an_unknown_busy_tick_never_accumulates_towards_a_trip():
    """Blindness on the busy clause resets the sustain counter exactly like
    blindness on any other clause — the conjunction must not get easier to
    satisfy the less this probe can see."""
    mon = GoodputMonitor()
    rows = [*_wedge_shaped(2, busy=10.0), *_wedge_shaped(1, busy=None),
            *_wedge_shaped(2, busy=10.0)]
    actions, verdicts = _drive(mon, "tier3", BUSY_FLEET, rows)
    assert LatchAction.TRIP not in actions, (
        "a collapse run interrupted by busy-clause blindness must not trip")
    assert any(v.outcome is Verdict.UNKNOWN for v in verdicts)


def test_undeclared_busy_min_leaves_the_original_rule_untouched():
    """🚨 The requirement-6 guard, at the PURE level: a threshold object that
    never sets `busy_min` must evaluate identically to the original
    four-clause rule — `progress_clauses` never grows the fifth entry."""
    original = GoodputThresholds(
        min_running=3, sustain_evaluations=3, max_iteration_rate=1.0,
        max_generation_tps_per_request=2.0, max_prefill_tps=600.0)
    assert original.progress_clauses == (
        CLAUSE_ITERATIONS, CLAUSE_GENERATION, CLAUSE_PREFILL)
    assert not original.armed or CLAUSE_ENGINE_IDLE not in original.progress_clauses

    mon = GoodputMonitor()
    # The identical wedge rows used throughout this file, but with NO "busy"
    # key at all — exactly what a caller that never declared the clause sends.
    now = _feed(mon, "tier3", _wedge_shaped(3, busy=None))
    v = mon.evaluate("tier3", original, now)
    assert v.outcome is Verdict.COLLAPSED, (
        "an undeclared busy clause must not change the original verdict")
    assert CLAUSE_ENGINE_IDLE not in v.clauses_held
    assert CLAUSE_ENGINE_IDLE not in v.clauses_cleared
    assert v.busy_max is None


def test_no_busy_default_ships():
    bare = GoodputThresholds()
    assert bare.busy_min == 0.0
    assert CLAUSE_ENGINE_IDLE not in bare.progress_clauses


# --------------------------------------------------------------------------- #
# `backend.probe_busy_gauge` — the parser, mirroring the hardening in
# `probe_progress_counters` (see that method's own docstring for the exact
# bugs each case guards against: the `or 0` seam bug, the sticky-malformed
# rule, negative/NaN/Inf, and a duplicated identical-label series).
# --------------------------------------------------------------------------- #

def _busy_pool(text: str, status: int = 200) -> BackendClientPool:
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


@pytest.mark.asyncio
async def test_probe_busy_gauge_reads_the_named_metric():
    pool = _busy_pool('gpu_power_watts{gpu="0"} 42.5\n')
    assert await _REAL_BUSY_GAUGE(pool, BUSY_URL, BUSY_METRIC) == pytest.approx(42.5)


@pytest.mark.asyncio
async def test_probe_busy_gauge_matches_the_full_name_not_a_prefix():
    """Same reasoning as `vllm:iteration_tokens_total_count` in
    `probe_progress_counters`: a prefix match would also catch an unrelated
    sibling series and silently change what is being read."""
    pool = _busy_pool('gpu_power_watts_max{gpu="0"} 999.0\n')
    assert await _REAL_BUSY_GAUGE(pool, BUSY_URL, BUSY_METRIC) is None


@pytest.mark.asyncio
async def test_probe_busy_gauge_takes_the_max_across_different_labels():
    """Multi-GPU exporter: different label sets are real and are MAXED, never
    summed — this is a LEVEL, not accumulated work."""
    pool = _busy_pool('gpu_power_watts{gpu="0"} 12.0\n'
                      'gpu_power_watts{gpu="1"} 45.0\n')
    assert await _REAL_BUSY_GAUGE(pool, BUSY_URL, BUSY_METRIC) == pytest.approx(45.0)


@pytest.mark.parametrize("label, body", [
    ("duplicated identical labels",
     'gpu_power_watts{gpu="0"} 12.0\ngpu_power_watts{gpu="0"} 12.0\n'),
    ("negative reading", 'gpu_power_watts{gpu="0"} -5.0\n'),
    ("NaN reading", 'gpu_power_watts{gpu="0"} NaN\n'),
    ("+Inf reading", 'gpu_power_watts{gpu="0"} +Inf\n'),
    ("unparseable value", 'gpu_power_watts{gpu="0"} not-a-number\n'),
    ("metric absent from the body", 'some_other_metric 1.0\n'),
])
@pytest.mark.asyncio
async def test_probe_busy_gauge_is_absent_not_zero_on_a_malformed_reading(label, body):
    pool = _busy_pool(body)
    assert await _REAL_BUSY_GAUGE(pool, BUSY_URL, BUSY_METRIC) is None, label


@pytest.mark.asyncio
async def test_probe_busy_gauge_is_none_on_a_non_200():
    pool = _busy_pool("gpu_power_watts 42.0\n", status=503)
    assert await _REAL_BUSY_GAUGE(pool, BUSY_URL, BUSY_METRIC) is None


@pytest.mark.asyncio
async def test_probe_busy_gauge_is_none_on_an_unparseable_or_empty_url():
    pool = BackendClientPool.__new__(BackendClientPool)
    assert await _REAL_BUSY_GAUGE(pool, "not a url", BUSY_METRIC) is None
    assert await _REAL_BUSY_GAUGE(pool, "", BUSY_METRIC) is None


# --------------------------------------------------------------------------- #
# Wiring: health.sample_goodput — the second, independent GET
# --------------------------------------------------------------------------- #

def _state(**over):
    """Mirrors test_goodput_wiring.py's `_state()` — everything
    `endpoint_healthy`/`sample_goodput` read, nothing production does not
    have."""
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


def _busy_armed_ep(**over) -> EndpointConfig:
    kw = dict(
        endpoint_class="tier3", role="reasoner",
        goodput_min_running=3, goodput_sustain_evaluations=3,
        goodput_max_iteration_rate=1.0, goodput_max_generation_tps=2.0,
        goodput_max_prefill_tps=600.0,
        goodput_busy_probe_url=BUSY_URL, goodput_busy_probe_metric=BUSY_METRIC,
        goodput_busy_min=20.0,
    )
    kw.update(over)
    return EndpointConfig(**kw)


class _CountingBackend:
    """Progress counters from a fixed sequence; counts BOTH probes so a test
    can assert the busy GET happens only when the clause is declared."""

    def __init__(self, progress_seq):
        self._seq = list(progress_seq)
        self.progress_calls = 0
        self.busy_calls = 0
        self.busy_value: float | None = 60.0

    async def probe_progress_counters(self, ep_cfg):
        self.progress_calls += 1
        return self._seq[min(self.progress_calls - 1, len(self._seq) - 1)]

    async def probe_busy_gauge(self, url, metric):
        self.busy_calls += 1
        return self.busy_value


def _wedge_counters(n):
    return [{"iterations": 500_000, "generation": 9_000_000,
             "prompt": 80_000_000, "running": 6} for _ in range(n)]


@pytest.mark.asyncio
async def test_an_undeclared_busy_clause_never_makes_the_second_get():
    """🚨 Absent means off, all the way down: no extra GET, no `busy` key
    recorded — identical to the pre-busy-clause poller cost."""
    state = _state()
    backend = _CountingBackend(_wedge_counters(8))
    state.backend = backend
    health = Health(state)
    ep = EndpointConfig(  # the ORIGINAL four-clause config, busy undeclared
        endpoint_class="tier3", role="reasoner",
        goodput_min_running=3, goodput_sustain_evaluations=3,
        goodput_max_iteration_rate=1.0, goodput_max_generation_tps=2.0,
        goodput_max_prefill_tps=600.0)
    for i in range(6):
        await health.sample_goodput("tier3", ep, now=10_000.0 + i * 10.0)
    assert backend.progress_calls == 6
    assert backend.busy_calls == 0, "an undeclared busy clause cost an extra GET"
    assert state.goodput_verdicts["tier3"].busy_max is None


@pytest.mark.asyncio
async def test_a_declared_busy_clause_makes_exactly_one_extra_get_per_tick():
    state = _state()
    backend = _CountingBackend(_wedge_counters(8))
    state.backend = backend
    health = Health(state)
    for i in range(6):
        await health.sample_goodput("tier3", _busy_armed_ep(), now=10_000.0 + i * 10.0)
    assert backend.progress_calls == 6
    assert backend.busy_calls == 6


@pytest.mark.asyncio
async def test_a_busy_probe_failure_does_not_blank_the_progress_counters():
    """The two probes are independent sources. Declare occupancy + generation
    + busy (skip iteration/prefill to keep the missing-key list unambiguous):
    a busy scrape failure must make ONLY `busy` read absent, never `running`
    or `generation` too."""
    state = _state()
    backend = _CountingBackend(_wedge_counters(8))
    backend.busy_value = None  # the busy probe fails every tick
    state.backend = backend
    health = Health(state)
    ep = _busy_armed_ep(goodput_max_iteration_rate=0.0, goodput_max_prefill_tps=0.0)
    for i in range(6):
        await health.sample_goodput("tier3", ep, now=10_000.0 + i * 10.0)
    verdict = state.goodput_verdicts["tier3"]
    assert verdict.outcome is Verdict.UNKNOWN
    assert verdict.reason == "counters_absent:busy", (
        f"a failed busy probe must not also blank running/generation; "
        f"got {verdict.reason!r}")
    assert state.collapsed_endpoints == set()


@pytest.mark.asyncio
async def test_a_progress_probe_failure_does_not_blank_busy():
    """The mirror case: the progress scrape fails outright (`None`), the busy
    probe succeeds — `busy` must still be recorded, not wiped by the other
    probe's failure."""
    state = _state()

    class _ProgressFails:
        async def probe_progress_counters(self, ep_cfg):
            return None

        async def probe_busy_gauge(self, url, metric):
            return 60.0

    state.backend = _ProgressFails()
    health = Health(state)
    ep = _busy_armed_ep(goodput_max_iteration_rate=0.0,
                        goodput_max_generation_tps=0.0, goodput_max_prefill_tps=0.0)
    for i in range(6):
        await health.sample_goodput("tier3", ep, now=10_000.0 + i * 10.0)
    verdict = state.goodput_verdicts["tier3"]
    assert verdict.outcome is Verdict.UNKNOWN
    assert verdict.reason == "counters_absent:running", (
        f"the busy reading must survive a totally-failed progress scrape; "
        f"got {verdict.reason!r}")


# --------------------------------------------------------------------------- #
# Wiring: the REAL parser, driven end to end through health.sample_goodput —
# mirrors test_goodput_wiring.py section 2c, which exists because a hand-built
# dict cannot see a seam bug that lives in the scraper's return statement.
# --------------------------------------------------------------------------- #

class _BusyScrapingBackend:
    """Fixed, healthy progress counters; the busy probe scrapes a REAL
    `/metrics` body through the real parser — nothing between the wire and
    the verdict is stubbed for the busy half."""

    def __init__(self, busy_body: str) -> None:
        self._pool = _busy_pool(busy_body)

    async def probe_progress_counters(self, ep_cfg):
        return {"iterations": 500_000, "generation": 9_000_000,
                "prompt": 80_000_000, "running": 6}

    async def probe_busy_gauge(self, url, metric):
        return await _REAL_BUSY_GAUGE(self._pool, url, metric)


async def _drive_real(busy_body: str, ticks: int = 6):
    state = _state()
    state.backend = _BusyScrapingBackend(busy_body)
    health = Health(state)
    for i in range(ticks):
        await health.sample_goodput("tier3", _busy_armed_ep(),
                                    now=20_000.0 + i * 10.0)
    return state


@pytest.mark.asyncio
async def test_a_malformed_busy_reading_reaches_health_as_unknown_not_collapsed():
    """THE REGRESSION SHAPE, for the busy probe: healthy-looking progress
    counters plus a MALFORMED busy body must not read as collapsed just
    because a naive parse coalesced the bad line to something falsy-but-
    numeric."""
    state = await _drive_real('gpu_power_watts{gpu="0"} NaN\n')
    verdict = state.goodput_verdicts["tier3"]
    assert verdict.outcome is Verdict.UNKNOWN, (
        f"a NaN busy reading read as {verdict.outcome.value}")
    assert "busy" in verdict.reason
    assert state.collapsed_endpoints == set()
    assert state.goodput.trips("tier3") == 0


@pytest.mark.asyncio
async def test_a_well_formed_busy_reading_reaches_health_and_clears_it():
    """The positive control for the test above — without it, that test could
    pass on a busy probe that always returns None."""
    state = await _drive_real('gpu_power_watts{gpu="0"} 60.0\n')
    verdict = state.goodput_verdicts["tier3"]
    assert verdict.outcome is Verdict.HEALTHY
    assert CLAUSE_ENGINE_IDLE in verdict.clauses_cleared
    assert verdict.busy_max == pytest.approx(60.0)


# --------------------------------------------------------------------------- #
# Wiring: /v1/status + the Prometheus gauge, through the REAL ProxyService
# --------------------------------------------------------------------------- #

class _LoopReq:
    def __init__(self) -> None:
        class _C:
            host = "127.0.0.1"

        self.client = _C()
        self.headers: dict = {}


@pytest.mark.asyncio
async def test_busy_max_reaches_status_and_the_prometheus_gauge():
    from roadstead.config import ProxyConfig
    from roadstead.service import ProxyService

    svc = ProxyService(ProxyConfig())
    ep_name = next(iter(svc._config.endpoints))
    ep_cfg = svc._config.endpoints[ep_name]
    for key, value in dict(
        goodput_min_running=3, goodput_sustain_evaluations=3,
        goodput_max_iteration_rate=1.0, goodput_max_generation_tps=2.0,
        goodput_max_prefill_tps=600.0,
        goodput_busy_probe_url=BUSY_URL, goodput_busy_probe_metric=BUSY_METRIC,
        goodput_busy_min=20.0,
    ).items():
        setattr(ep_cfg, key, value)

    async def _fake_progress(ep_cfg):
        return {"iterations": 500_000, "generation": 9_000_000,
                "prompt": 80_000_000, "running": 6}

    async def _fake_busy(url, metric):
        return 60.5

    svc._backend.probe_progress_counters = _fake_progress
    svc._backend.probe_busy_gauge = _fake_busy

    for i in range(6):
        await svc._health.sample_goodput(ep_name, ep_cfg, now=10_000.0 + i * 10.0)

    status_resp = await svc.handle_status(_LoopReq())
    body = json.loads(status_resp.body)
    gp = body["endpoints"][ep_name]["goodput"]
    assert gp["busy_max"] == pytest.approx(60.5)

    metrics_resp = await svc.handle_prometheus_metrics(_LoopReq())
    text = metrics_resp.body.decode()
    assert f'roadstead_endpoint_goodput_busy_max{{endpoint="{ep_name}"}} 60.5' in text
