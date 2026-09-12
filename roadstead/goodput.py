"""Endpoint-level goodput-collapse detection — "the ENDPOINT is sick", not "this
request is stuck".

WHAT HAPPENED. On 2026-09-11 and 2026-09-12 a vLLM reasoning backend wedged: its
scheduler step ballooned to ~7 s (stuck filling constrained-decoding bitmasks),
so its slots filled with requests producing almost nothing. It answered
``/health`` with ``healthy: true`` throughout, because the HTTP server was fine —
only the engine was not. Callers hit their deadlines and retried, which refilled
the slots, which is what made it self-sustaining rather than transient. 226
timeout events over three days, 219 of them (97%) carrying no ``abort_reason``
at all, so the pattern was invisible by construction.

The per-request stall watchdog (``lifecycle._stream_progress_probe``) worked
throughout, and could not have helped: its conclusion is always "THIS request is
stuck", never "this endpoint is sick". This module is the endpoint-level verdict
the watchdog cannot reach.

WHAT THE SIGNAL IS, AND WHY IT IS THE BACKEND'S OWN COUNTERS. The rule below was
measured over 4,264 minutes of real engine counters against independent ground
truth (the proxy's own ``proxy_timeouts`` rows, ±10 min). Per endpoint, sampled
on the capacity poller's timer:

    occupancy:   running      >= N          (slots are BUSY)
    iterations:  d(iterations)/dt  < R      (the scheduler loop is not turning)
    generation:  d(generation)/dt / running < G   (turning, emitting nothing)
    prefill:     d(prompt)/dt          < P      (and not doing prefill either)
    sustained for S consecutive evaluations

Measured precision 1.000 (294 firings, none outside a real timeout window) and
60.7% harm-minute recall, firing two minutes after the first timeout of the
second day with 42 of that day's 43 timeout events still to come.

🚨 **ONE TERM IS MEASURABLY LOAD-BEARING AND THE OTHER THREE ARE NOT — say it
honestly.** Ablation over the same 4,264 minutes: removing the OCCUPANCY gate
collapses precision to 0.425 (415 false positives), because an idle engine
trivially satisfies every progress clause. Removing any single PROGRESS clause
leaves precision at 0.997-1.000. So the triple conjunction is justified by
MECHANISM — loop not turning / loop turning but emitting nothing / prefill-only —
and NOT by a measured precision gain over one clause. Do not write that it is.

🚨 **THE FALSE POSITIVE THAT SHAPED THE RULE.** One real caller sends 31-37k-token
prompts and gets 2-3 output tokens back. Six such requests concurrently read
**0.42 generation tokens/sec/busy-slot** — deep inside the wedge band — while
being perfectly healthy, because almost all the work is prefill. What separates
them is that the engine is doing **1,209 prompt tokens/sec** of real prefill.
Wedge minutes never exceeded **247.5** prompt tokens/sec; healthy busy minutes
had a median of **992**. The prefill clause exists for exactly that caller, and
``tests/test_goodput.py`` keeps it as a named control.

WHAT WAS REJECTED, AND WHY NOT TO REACH FOR IT AGAIN:

  * ``QueuedRequest.estimated_remaining_s`` — written once at dispatch and never
    updated. During the incident it read ~258 s remaining on a request that then
    ran for 1000+ s.
  * ``CostModel.decode_tps`` — an EWMA over COMPLETED requests. A decode_tps
    degradation guard already existed in ``scheduler._admit`` and was REMOVED
    because prefill-dominated workloads inflate it; re-adding it is the main
    hazard here, and the caller above is why.
  * proxy-side per-request output-token counting — the count exists only as a
    local in ``lifecycle.execute_streaming``, set from a final ``usage`` frame.
    Non-streaming requests are structurally unobservable (one awaited POST), so
    it would be blind to the synchronous majority.

🚨 **THE NON-NEGOTIABLE INVARIANT: BLINDNESS MUST NEVER READ AS COLLAPSE.** The
rule is a CONJUNCTION, so a clause that cannot be evaluated makes firing EASIER,
not harder — the classic way a guard turns into the outage it was built to
prevent. Therefore **every clause the endpoint has configured must be evaluable,
or the verdict is UNKNOWN**: a configured counter absent from ``/metrics``, a
failed scrape, a counter reset, too few samples, too short a window. UNKNOWN is a
third outcome, never folded into either of the other two, and it is published as
its own metric so a permanently-blind endpoint is visible rather than reading as
verified-healthy.

NO PUBLIC THRESHOLD DEFAULTS. The five numbers above are fleet-hardware
measurements, not mechanism, and this repository's rule is that mechanism is
public and measurements are private: **absent thresholds mean the detector does
not run at all** for that endpoint. They arrive per-endpoint from ``models.yaml``
``policy.goodput_*``. The numbers in the tests and in ``docs/operations.md`` are
labelled as measured on ONE fleet, and are worked examples rather than defaults.

PURITY. No I/O, no async, no clock of its own — ``observe`` and ``evaluate`` both
take the timestamp. ``tests/test_pure_modules.py`` holds this module to that, so
a precision claim can be measured in microseconds against a fleet that does not
exist.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

# --------------------------------------------------------------------------- #
# Mechanism constants. These are NOT thresholds — no value here can arm the
# detector or move its firing point; they bound the arithmetic and the latch.
# --------------------------------------------------------------------------- #

#: Samples retained per endpoint. At the poller's 10 s cadence this is ~60 s of
#: history, which is more than any evaluation window below needs — the surplus
#: exists so a single slow poller tick cannot empty the ring.
_RING_CAPACITY = 6

#: Trailing seconds an evaluation draws its first/last pair from. Deliberately
#: SHORT (2-3 poller ticks): a long averaging window dilutes the wedge with the
#: healthy work that preceded it, and the incident's value is early detection.
#: Time depth comes from requiring S consecutive collapsed evaluations, not from
#: widening this.
_RATE_WINDOW_S = 25.0

#: Floors below which a rate is arithmetic rather than observation. Two samples
#: is the minimum a delta needs; the 5 s span keeps a pair of near-simultaneous
#: samples from dividing a small delta by a tiny dt and inventing a huge rate.
_MIN_SAMPLES = 2
_MIN_WINDOW_S = 5.0

#: Consecutive HEALTHY evaluations required to clear a trip, when the endpoint
#: does not declare its own. Deliberately small: a false trip is an outage, and
#: the absolute ceiling below — not this number — is what actually bounds a hold.
#: More than one so a single lucky sample cannot un-trip a real wedge.
_RECOVERY_EVALUATIONS = 2

#: Absolute ceiling on a single hold, seconds. 🚨 This is what makes a latched
#: false positive impossible rather than unlikely: there must be NO path that
#: stays tripped forever. It is not a guess at how long a wedge lasts — a wedge
#: that is still wedged re-trips within S poller ticks, which is cheap, whereas
#: holding on a claim nobody has re-confirmed is an outage we caused.
_MAX_HOLD_S = 300.0

#: The four counter keys this module reads, as ``backend.probe_progress_counters``
#: publishes them. ``prompt``/``generation``/``iterations`` are CUMULATIVE (rates
#: are deltas); ``running`` is a GAUGE (read as a level).
CUMULATIVE_KEYS = ("iterations", "generation", "prompt")
GAUGE_KEYS = ("running",)


class Verdict(str, Enum):
    """Three-valued by construction — see the invariant in the module docstring.

    ``str`` mixin so the value lands in a JSON payload and a metric label
    without a conversion at each site.
    """

    HEALTHY = "healthy"
    COLLAPSED = "collapsed"
    #: Could not tell. NOT a synonym for either of the others, and published as
    #: its own signal.
    UNKNOWN = "unknown"


class LatchAction(str, Enum):
    """What the latch wants done to the breaker set this tick."""

    NONE = "none"
    TRIP = "trip"
    CLEAR = "clear"


#: Clause names, used in the verdict's evidence and in log lines. One per term of
#: the conjunction; ``occupancy`` is the measured load-bearing one.
CLAUSE_OCCUPANCY = "occupancy"
CLAUSE_ITERATIONS = "iterations"
CLAUSE_GENERATION = "generation"
CLAUSE_PREFILL = "prefill"


@dataclass(frozen=True)
class GoodputThresholds:
    """Per-endpoint detector configuration. Injected, never defaulted to a fleet
    measurement — see "NO PUBLIC THRESHOLD DEFAULTS" above.

    0 / ``None`` means "not declared" for every field, and the consequences
    differ by field on purpose:

      * ``min_running`` undeclared  -> the detector is OFF for this endpoint.
        The occupancy gate is the one term ablation showed to be load-bearing
        (precision 1.000 -> 0.425 without it), so a configuration that omits it
        is not a weaker detector, it is a broken one.
      * ``sustain_evaluations`` undeclared -> also OFF. A single evaluation is
        one sample pair; the measured rule is "sustained", and shipping a public
        default here would be shipping a measured number.
      * every PROGRESS threshold undeclared -> OFF. Occupancy alone fires on
        every busy endpoint.
      * ONE progress threshold undeclared -> that clause is simply not part of
        the conjunction for this endpoint. This is how a llama.cpp endpoint runs
        the rule at all: it publishes no iteration counter, so it declares no
        ``max_iteration_rate`` and the other two clauses decide. Declaring a
        threshold whose counter the engine never publishes does NOT make the
        detector stricter — it makes every verdict UNKNOWN, which the blindness
        metric is there to surface.
    """

    #: Occupancy gate: the engine must have at least this many requests running
    #: for a progress collapse to mean anything. 0 = detector off.
    min_running: int = 0
    #: Consecutive collapsed evaluations before the breaker trips. 0 = off.
    sustain_evaluations: int = 0
    #: Scheduler iterations per second, below which the loop is not turning.
    max_iteration_rate: float | None = None
    #: Generation tokens per second PER BUSY SLOT, below which the loop is
    #: turning and emitting nothing.
    max_generation_tps_per_request: float | None = None
    #: Prompt tokens per second, below which the engine is not doing prefill
    #: either. 🚨 This clause is what keeps a legitimately prefill-heavy caller
    #: out of the wedge band — see the module docstring.
    max_prefill_tps: float | None = None
    #: Consecutive healthy evaluations required to clear. 0 = the module default.
    recovery_evaluations: int = 0
    #: Absolute ceiling on one hold, seconds. 0 = the module default.
    max_hold_s: float = 0.0

    @property
    def progress_clauses(self) -> tuple[str, ...]:
        """The progress clauses this endpoint evaluates, in rule order."""
        out = []
        if self.max_iteration_rate is not None and self.max_iteration_rate > 0:
            out.append(CLAUSE_ITERATIONS)
        if (self.max_generation_tps_per_request is not None
                and self.max_generation_tps_per_request > 0):
            out.append(CLAUSE_GENERATION)
        if self.max_prefill_tps is not None and self.max_prefill_tps > 0:
            out.append(CLAUSE_PREFILL)
        return tuple(out)

    @property
    def armed(self) -> bool:
        """True when this endpoint runs the detector at all."""
        return (self.min_running >= 1
                and self.sustain_evaluations >= 1
                and bool(self.progress_clauses))

    @property
    def effective_recovery_evaluations(self) -> int:
        return self.recovery_evaluations or _RECOVERY_EVALUATIONS

    @property
    def effective_max_hold_s(self) -> float:
        return self.max_hold_s or _MAX_HOLD_S


@dataclass(frozen=True)
class GoodputVerdict:
    """One evaluation, carrying every number that produced it.

    A verdict that says only "collapsed" is unreviewable: the soak that decides
    whether to arm this feature has to read the rates, the window they came from
    and which clauses held. ``reason`` is populated for UNKNOWN and names WHICH
    blindness it was, because "a counter the engine never publishes" and "the
    ring is still filling after a restart" call for opposite responses.
    """

    outcome: Verdict
    samples: int = 0
    window_s: float = 0.0
    #: Minimum ``running`` across the window — the conservative reading of the
    #: occupancy gate, so a momentary spike cannot arm it.
    running_min: float | None = None
    #: Mean ``running`` across the window — the honest denominator for tokens
    #: produced over that interval.
    running_mean: float | None = None
    iteration_rate: float | None = None
    generation_tps: float | None = None
    generation_tps_per_request: float | None = None
    prefill_tps: float | None = None
    #: Clauses whose collapse condition HELD this evaluation.
    clauses_held: tuple[str, ...] = ()
    #: Clauses that did NOT hold — on a HEALTHY verdict this is the evidence of
    #: health, and it is the field that names the prefill clause when a
    #: prefill-heavy caller is correctly cleared.
    clauses_cleared: tuple[str, ...] = ()
    reason: str = ""

    @property
    def evaluable(self) -> bool:
        return self.outcome is not Verdict.UNKNOWN


@dataclass
class _Sample:
    ts: float
    #: ``None`` for a key the scrape could not supply. Kept per-key rather than
    #: dropping the sample, because an engine that publishes three of the four
    #: counters can still evaluate the three clauses it configured.
    values: dict[str, int | None] = field(default_factory=dict)


@dataclass
class _LatchState:
    consecutive_collapsed: int = 0
    consecutive_healthy: int = 0
    #: Monotonic timestamp of the trip currently held, 0 when not tripped.
    tripped_at: float = 0.0
    #: Trips since boot, for the dark-soak report. Accrues whether or not
    #: enforcement is armed — the point of a dark ship is to read this first.
    trips: int = 0


class GoodputMonitor:
    """Per-endpoint sample rings, the verdict function, and the trip latch.

    Holds no opinion about enforcement: ``step`` is told whether the endpoint is
    currently tripped and answers with what it wants done. The authoritative
    record of trippedness is the caller's set (``ProxyState.collapsed_endpoints``)
    so there is exactly one, rather than a copy here to drift out of step with it.
    """

    def __init__(self, ring_capacity: int = _RING_CAPACITY,
                 rate_window_s: float = _RATE_WINDOW_S) -> None:
        self._ring_capacity = max(_MIN_SAMPLES, int(ring_capacity))
        self._rate_window_s = float(rate_window_s)
        self._rings: dict[str, deque[_Sample]] = {}
        self._latches: dict[str, _LatchState] = {}

    # ----------------------------------------------------------- observation

    def observe(self, endpoint: str, ts: float,
                counters: dict[str, int | None] | None) -> None:
        """Record one scrape.

        ``counters=None`` is a FAILED scrape, and it is recorded as a sample
        whose every value is absent rather than dropped: a scrape that keeps
        failing must push the endpoint towards UNKNOWN, not leave a stale
        healthy-looking window in place.
        """
        ring = self._rings.setdefault(endpoint, deque(maxlen=self._ring_capacity))
        values: dict[str, int | None] = {}
        for key in CUMULATIVE_KEYS + GAUGE_KEYS:
            raw = None if counters is None else counters.get(key)
            values[key] = None if raw is None else int(raw)
        ring.append(_Sample(ts=float(ts), values=values))

    def forget(self, endpoint: str) -> None:
        """Drop everything known about an endpoint (it was removed, or paused
        long enough that its history is a lie)."""
        self._rings.pop(endpoint, None)
        self._latches.pop(endpoint, None)

    def trips(self, endpoint: str) -> int:
        latch = self._latches.get(endpoint)
        return latch.trips if latch else 0

    def tripped_for_s(self, endpoint: str, now: float) -> float | None:
        latch = self._latches.get(endpoint)
        if latch is None or not latch.tripped_at:
            return None
        return max(0.0, now - latch.tripped_at)

    # -------------------------------------------------------------- verdict

    def evaluate(self, endpoint: str, thresholds: GoodputThresholds,
                 now: float) -> GoodputVerdict:
        """Judge one endpoint from the samples inside the trailing window.

        Never raises on a hostile sample sequence: every path that cannot
        produce all the numbers the endpoint's clauses need returns UNKNOWN with
        a reason, and UNKNOWN is the only outcome that costs nothing.
        """
        if not thresholds.armed:
            return GoodputVerdict(Verdict.UNKNOWN, reason="not_configured")

        ring = self._rings.get(endpoint)
        if not ring:
            return GoodputVerdict(Verdict.UNKNOWN, reason="no_samples")

        window = [s for s in ring if (now - s.ts) <= self._rate_window_s]
        if len(window) < _MIN_SAMPLES:
            return GoodputVerdict(Verdict.UNKNOWN, samples=len(window),
                                  reason="too_few_samples")
        span = window[-1].ts - window[0].ts
        if span < _MIN_WINDOW_S:
            return GoodputVerdict(Verdict.UNKNOWN, samples=len(window),
                                  window_s=round(span, 3),
                                  reason="window_too_short")

        # A counter that went BACKWARDS is a backend restart, not a negative
        # rate. Drop the history that straddles it — keeping the newest sample so
        # the next tick has a baseline — and report blindness for this tick.
        if self._reset_detected(window):
            newest = ring[-1]
            ring.clear()
            ring.append(newest)
            return GoodputVerdict(Verdict.UNKNOWN, samples=len(window),
                                  window_s=round(span, 3), reason="counter_reset")

        needed = [CLAUSE_OCCUPANCY, *thresholds.progress_clauses]
        missing = self._missing_counters(window, needed)
        if missing:
            return GoodputVerdict(
                Verdict.UNKNOWN, samples=len(window), window_s=round(span, 3),
                reason="counters_absent:" + ",".join(missing))

        runnings = [float(s.values["running"]) for s in window]  # type: ignore[arg-type]
        running_min = min(runnings)
        running_mean = sum(runnings) / len(runnings)

        iteration_rate = self._rate(window, "iterations", span)
        prefill_tps = self._rate(window, "prompt", span)
        generation_tps = self._rate(window, "generation", span)
        generation_per_req = (
            None if generation_tps is None or running_mean <= 0
            else generation_tps / running_mean)

        if running_min < thresholds.min_running:
            # 🚨 SHORT-CIRCUIT ON THE OCCUPANCY GATE, and it is safe in the one
            # direction that matters: the conjunction is already REFUTED, so
            # declining to evaluate the remaining clauses can only make firing
            # HARDER. That is the exact opposite of skipping a clause we CANNOT
            # evaluate, which is what UNKNOWN is for.
            #
            # It is also the only reading that is correct for an IDLE endpoint.
            # With no slots busy there is no tokens-per-busy-slot to compute, so
            # falling through would hit the `value is None` arm below and publish
            # UNKNOWN — firing the blindness alert all night on a perfectly
            # healthy quiet proxy, and calling a known-good state unmeasured.
            # Found by `test_sabotage_occupancy_an_idle_engine_is_not_collapsed`.
            return GoodputVerdict(
                Verdict.HEALTHY,
                samples=len(window), window_s=round(span, 3),
                running_min=running_min, running_mean=round(running_mean, 2),
                iteration_rate=(None if iteration_rate is None
                                else round(iteration_rate, 4)),
                generation_tps=(None if generation_tps is None
                                else round(generation_tps, 3)),
                generation_tps_per_request=(None if generation_per_req is None
                                            else round(generation_per_req, 3)),
                prefill_tps=None if prefill_tps is None else round(prefill_tps, 2),
                clauses_cleared=(CLAUSE_OCCUPANCY,),
            )

        held: list[str] = [CLAUSE_OCCUPANCY]
        cleared: list[str] = []
        for clause, value, ceiling in (
            (CLAUSE_ITERATIONS, iteration_rate, thresholds.max_iteration_rate),
            (CLAUSE_GENERATION, generation_per_req,
             thresholds.max_generation_tps_per_request),
            (CLAUSE_PREFILL, prefill_tps, thresholds.max_prefill_tps),
        ):
            if clause not in thresholds.progress_clauses:
                continue
            if value is None:
                # The counter was present in every sample (checked above) and the
                # rate is still not computable — only the per-request divisor can
                # do that. Blind, not healthy.
                return GoodputVerdict(
                    Verdict.UNKNOWN, samples=len(window), window_s=round(span, 3),
                    running_min=running_min, running_mean=round(running_mean, 2),
                    reason=f"rate_unavailable:{clause}")
            (held if value < ceiling else cleared).append(clause)

        collapsed = len(held) == len(needed)
        return GoodputVerdict(
            Verdict.COLLAPSED if collapsed else Verdict.HEALTHY,
            samples=len(window),
            window_s=round(span, 3),
            running_min=running_min,
            running_mean=round(running_mean, 2),
            iteration_rate=None if iteration_rate is None else round(iteration_rate, 4),
            generation_tps=None if generation_tps is None else round(generation_tps, 3),
            generation_tps_per_request=(
                None if generation_per_req is None else round(generation_per_req, 3)),
            prefill_tps=None if prefill_tps is None else round(prefill_tps, 2),
            clauses_held=tuple(held),
            clauses_cleared=tuple(cleared),
        )

    # ----------------------------------------------------------------- latch

    def step(self, endpoint: str, verdict: GoodputVerdict, *, tripped: bool,
             thresholds: GoodputThresholds, now: float) -> LatchAction:
        """Fold one verdict into the sustain/recovery counters.

        UNKNOWN resets BOTH counters. That is the invariant in its operational
        form: blindness cannot accumulate towards a trip (the conjunction would
        otherwise be easier to satisfy the less we can see), and it is not
        evidence of recovery either. The absolute ceiling is what stops a hold
        that has gone blind from lasting forever.
        """
        latch = self._latches.setdefault(endpoint, _LatchState())

        # Checked FIRST, so it applies under every verdict including UNKNOWN.
        if tripped and latch.tripped_at:
            if (now - latch.tripped_at) >= thresholds.effective_max_hold_s:
                latch.tripped_at = 0.0
                latch.consecutive_collapsed = 0
                latch.consecutive_healthy = 0
                return LatchAction.CLEAR

        if verdict.outcome is Verdict.COLLAPSED:
            latch.consecutive_healthy = 0
            latch.consecutive_collapsed = min(
                latch.consecutive_collapsed + 1, thresholds.sustain_evaluations)
            if (not tripped
                    and latch.consecutive_collapsed >= thresholds.sustain_evaluations):
                latch.tripped_at = now
                latch.trips += 1
                latch.consecutive_collapsed = 0
                return LatchAction.TRIP
            return LatchAction.NONE

        if verdict.outcome is Verdict.HEALTHY:
            latch.consecutive_collapsed = 0
            latch.consecutive_healthy += 1
            if (tripped
                    and latch.consecutive_healthy
                    >= thresholds.effective_recovery_evaluations):
                latch.tripped_at = 0.0
                latch.consecutive_healthy = 0
                return LatchAction.CLEAR
            return LatchAction.NONE

        latch.consecutive_collapsed = 0
        latch.consecutive_healthy = 0
        return LatchAction.NONE

    # ------------------------------------------------------------- internals

    @staticmethod
    def _reset_detected(window: list[_Sample]) -> bool:
        for key in CUMULATIVE_KEYS:
            prev = None
            for sample in window:
                cur = sample.values.get(key)
                if cur is None:
                    prev = None
                    continue
                if prev is not None and cur < prev:
                    return True
                prev = cur
        return False

    @staticmethod
    def _missing_counters(window: list[_Sample], clauses: list[str]) -> list[str]:
        """Counter keys a configured clause needs and some sample lacks.

        ANY absent sample disqualifies the key, not just the endpoints of the
        window: a counter that blinked out mid-window gives a first/last delta
        that spans a gap nobody measured.
        """
        needed_keys: list[str] = []
        for clause in clauses:
            if clause == CLAUSE_OCCUPANCY:
                needed_keys.append("running")
            elif clause == CLAUSE_ITERATIONS:
                needed_keys.append("iterations")
            elif clause == CLAUSE_GENERATION:
                needed_keys.extend(("generation", "running"))
            elif clause == CLAUSE_PREFILL:
                needed_keys.append("prompt")
        missing = []
        for key in dict.fromkeys(needed_keys):
            if any(s.values.get(key) is None for s in window):
                missing.append(key)
        return missing

    @staticmethod
    def _rate(window: list[_Sample], key: str, span: float) -> float | None:
        if span <= 0:
            return None
        first = window[0].values.get(key)
        last = window[-1].values.get(key)
        if first is None or last is None:
            return None
        return (last - first) / span
