"""Goodput collapse through the REAL front door — trip, shed, recover, and dark.

The unit files prove the rule (`tests/test_goodput.py`) and the wiring
(`tests/test_goodput_wiring.py`). This file proves the thing neither can: that a
latched endpoint actually changes what a caller gets from `POST
/v1/chat/completions`, over the real `ProxyService` and the real fake backend
socket — and that with the flag OFF, nothing a caller can see changes at all.

Modelled on `test_endpoint_cooldown_adversarial.py`, which is the closest
precedent (trip → defer → auto-recover) and carries both analogues this needs:
a flags-off surface test and a sibling-endpoint isolation test.

The engine counters are driven by mutating the fake's CONTROLLER state, not by
headers — the proxy forwards only `X-Request-ID` to a backend, so a header-driven
fault never reaches the fake through the proxy (`tests/e2e/conftest.py`).
"""

from __future__ import annotations

import asyncio

import pytest

from roadstead.testing import FakeBackend, FakeBackendServer
from roadstead.timeout_model import normalize_endpoint


@pytest.fixture
def fake():
    """Override the shared e2e fake with a **vLLM** one.

    Two reasons, and the first is not convenience. The backend that wedged on
    2026-09-11/12 was vLLM, and vLLM is the only engine family that publishes an
    iteration counter — so the four-clause rule can only be exercised end to end
    against this shape. The default e2e fake is `llama.cpp`, which publishes no
    `vllm:iteration_tokens_total_count` and would leave the iteration clause
    permanently UNKNOWN here: the journey would still pass, by never evaluating
    the term it claims to test.

    (`tests/test_goodput_wiring.py` covers the llama.cpp shape — occupancy plus
    the two token counters, iteration clause undeclared — which is the other
    configuration this feature has to support.)
    """
    srv = FakeBackendServer(FakeBackend(engine="vllm")).start()
    try:
        yield srv
    finally:
        srv.stop()

# Spins a real ProxyService; deselected from the fast tollgate like its siblings.
pytestmark = pytest.mark.heavy

FLAG = "goodput_collapse_enforce"

# The endpoint under test. "chat" is an ALIAS — health/breaker state and the
# /v1/status map are keyed by the RESOLVED class, so derive it rather than
# hardcoding a name that has moved three times in this repo's history.
EP_ALIAS = "chat"
EP_CLASS = normalize_endpoint(EP_ALIAS)

#: The measured fleet rule. 🚨 Every number is ONE fleet's hardware measurement
#: (4,264 minutes of engine counters); Roadstead ships none of them as defaults,
#: which is why this test has to declare them itself. `sustain=2` is the one
#: departure: the trip count is pinned by the unit tests, and a shorter sustain
#: keeps the number of poller ticks this journey has to wait for small.
POLICY = dict(
    goodput_min_running=3,
    goodput_sustain_evaluations=2,
    goodput_max_iteration_rate=1.0,
    goodput_max_generation_tps=2.0,
    goodput_max_prefill_tps=600.0,
    goodput_recovery_evaluations=2,
    goodput_max_hold_s=300.0,
)


def _arm(proxy, ep: str = EP_CLASS) -> None:
    """Declare the thresholds on ONE endpoint's live config.

    Mutating the EndpointConfig is the same thing `models.yaml` `policy.*` does
    through the catalog — pinned by
    `test_goodput_wiring.py::test_every_goodput_policy_key_reaches_endpoint_config`,
    so this shortcut cannot diverge from the real path without that test failing.
    """
    cfg = proxy.svc._config.endpoints[ep]
    for key, value in POLICY.items():
        setattr(cfg, key, value)


def _wedged(controller) -> None:
    """Slots busy, every cumulative counter frozen: the incident."""
    controller.num_requests_running = 6
    controller.iteration_tokens_count = 500_000
    controller.generation_tokens_total = 9_000_000
    controller.prompt_tokens_total = 80_000_000


def _working(controller, tick: int) -> None:
    """A busy engine doing real work — the loop turning, tokens coming out."""
    controller.num_requests_running = 6
    controller.iteration_tokens_count = 500_000 + tick * 3_000
    controller.generation_tokens_total = 9_000_000 + tick * 12_000
    controller.prompt_tokens_total = 80_000_000 + tick * 120_000


async def _poll(proxy, ep: str = EP_CLASS, *, ticks: int = 1,
                advance: float = 10.0) -> None:
    """Drive N goodput poller passes at the real 10 s cadence.

    The poller's own interval is shrunk to 0.05 s by the e2e fixture, which is
    useless here: the detector needs a MEASURED window, and samples 0.05 s apart
    are below its span floor by design. So this calls the sampler directly with a
    stepped timestamp — the same code path the poller takes, on the cadence
    production runs it at.

    🚨 It passes `now` rather than patching the clock. Patching
    `health.time.monotonic` patches the module `time`, which the EVENT LOOP also
    reads — the first cut of this file did exactly that and every scrape against
    the real fake socket came back `None`, which the detector then correctly
    reported as a blind backend. A test harness that breaks the transport and
    then asserts on what the detector concluded is measuring itself.
    """
    cfg = proxy.svc._config.endpoints[ep]
    base = getattr(_poll, "_t", 100_000.0)
    for _ in range(ticks):
        base += advance
        await proxy.svc._health.sample_goodput(ep, cfg, now=base)
    _poll._t = base


async def _status_ep(proxy, ep: str = EP_CLASS) -> dict:
    resp = await proxy.client.get("/v1/status")
    assert resp.status_code == 200
    return resp.json()["endpoints"][ep]


async def _metrics(proxy) -> str:
    resp = await proxy.client.get("/metrics")
    assert resp.status_code == 200
    return resp.text


# --------------------------------------------------------------------------- #
# 1 — ARMED: a tripped endpoint FAST-FAILS rather than queueing
# --------------------------------------------------------------------------- #

async def test_a_tripped_endpoint_fast_fails_instead_of_queueing(proxy):
    """The whole point of the feature, at the front door.

    Before the trip the same request is served 200 by the same backend; after it,
    an interactive caller gets a deferrable 503 IMMEDIATELY instead of waiting out
    its deadline on an engine producing nothing. Breaking that wait is what breaks
    the retry amplifier.
    """
    _arm(proxy)
    proxy.svc._state.flags.set_many({FLAG: True})
    _wedged(proxy.controller)

    # Control: the backend is happy to answer. The wedge is a COUNTER story, not
    # a completion story — which is exactly why `/health` could not see it.
    ok = await proxy.chat("hi", model=EP_ALIAS)
    assert ok.status_code == 200, "the backend answers fine — only its engine counters say otherwise"

    await _poll(proxy, ticks=4)
    assert EP_CLASS in proxy.svc._state.collapsed_endpoints, "never tripped"
    assert proxy.svc._health.endpoint_healthy(EP_ALIAS) is False

    shed = await proxy.chat("hi", model=EP_ALIAS)
    assert shed.status_code == 503, "a tripped endpoint must fast-fail, not serve"
    body = shed.json()
    blob = str(body)
    assert "goodput" in blob, f"the refusal must name its cause: {body}"
    # The deferrability marker the fleet's clients sniff for — a shed that reads
    # as a hard error turns a retryable deferral into a dropped turn.
    assert "backpressure" in blob, f"refusal must stay deferrable: {body}"


async def test_the_refusal_is_recorded_as_a_backend_fault(proxy):
    """🚨 THE HOLE THAT MADE THE INCIDENT INVISIBLE, closed at the door.

    219 of 226 real timeout events carried `abort_reason` NULL. A goodput refusal
    writes its own row with `abort_reason='goodput_collapse'`, and that reason is
    in `STALL_ABORT_REASONS` — so `GET /v1/timeouts/stalls` can tell a downstream
    consumer "your failure was an upstream substrate failure", which is the one
    question it exists to answer.
    """
    _arm(proxy)
    proxy.svc._state.flags.set_many({FLAG: True})
    _wedged(proxy.controller)
    await _poll(proxy, ticks=4)

    resp = await proxy.chat("hi", model=EP_ALIAS)
    assert resp.status_code == 503

    # ⚠️ The row is written through the SINGLE WRITER THREAD, so it is not on disk
    # when the HTTP response returns. Passed standalone and failed inside the full
    # suite once before this wait was added — the read raced the writer, and under
    # load the writer loses. Reading a queued write back without draining is the
    # shape of flake that gets re-labelled "intermittent" and ignored.
    await asyncio.to_thread(proxy.svc._state.queue_db.flush, 5.0)

    report = await proxy.client.get("/v1/timeouts?hours=1")
    assert report.status_code == 200
    reasons = report.json().get("by_abort_reason") or {}
    assert reasons.get("goodput_collapse", 0) >= 1, (
        f"the refusal left no classifiable row: {reasons}")


async def test_a_sibling_endpoint_keeps_serving(proxy):
    """The breaker is a SET, per endpoint — one wedged backend must not shed the
    fleet. Only `chat` declares thresholds here, so only `chat` can trip."""
    _arm(proxy)
    proxy.svc._state.flags.set_many({FLAG: True})
    _wedged(proxy.controller)
    await _poll(proxy, ticks=4)

    assert proxy.svc._health.endpoint_healthy(EP_ALIAS) is False
    assert proxy.svc._health.endpoint_healthy("tier3") is True
    served = await proxy.chat("hi", model="tier3")
    assert served.status_code == 200, "sibling endpoint must keep serving"
    assert "tier3" not in proxy.svc._state.collapsed_endpoints


async def test_it_recovers_and_serves_again(proxy):
    """Recovery is automatic: the endpoint serves again with no operator action.

    A breaker whose release needs a human is an outage with extra steps.
    """
    _arm(proxy)
    proxy.svc._state.flags.set_many({FLAG: True})
    _wedged(proxy.controller)
    await _poll(proxy, ticks=4)
    assert (await proxy.chat("hi", model=EP_ALIAS)).status_code == 503

    for tick in range(1, 6):
        _working(proxy.controller, tick)
        await _poll(proxy, ticks=1)
        if EP_CLASS not in proxy.svc._state.collapsed_endpoints:
            break
    assert EP_CLASS not in proxy.svc._state.collapsed_endpoints, "never recovered"
    assert proxy.svc._health.endpoint_healthy(EP_ALIAS) is True
    assert (await proxy.chat("hi", model=EP_ALIAS)).status_code == 200


# --------------------------------------------------------------------------- #
# 2 — DARK: flag off, nothing a CALLER can see changes
# --------------------------------------------------------------------------- #

#: Response fields that legitimately differ between any two calls, feature or no
#: feature. Everything else must match across a shadow trip.
_PER_CALL_FIELDS = ("id", "created", "request_id", "system_fingerprint")


def _caller_visible(resp) -> tuple[int, dict, dict]:
    """Everything a caller can observe, minus what differs call to call.

    Status, headers AND body — because "nothing a caller can see" has to include
    the `X-Roadstead-*` enrichment headers, not only the JSON.
    """
    body = {k: v for k, v in resp.json().items() if k not in _PER_CALL_FIELDS}
    headers = {k.lower(): v for k, v in resp.headers.items()
               if k.lower() not in ("date", "content-length")
               and "request-id" not in k.lower()}
    return resp.status_code, headers, body


async def test_flag_off_changes_nothing_the_caller_can_see(proxy):
    """🚨 THE SHIP-DARK GUARANTEE, as an actual COMPARISON.

    Same wedge, same poller passes, a latched shadow trip in between — and the
    caller's status, headers and body are IDENTICAL before and after, modulo the
    per-call identifiers in `_PER_CALL_FIELDS`.

    ⚠️ An earlier cut of this test was named `..._is_byte_identical_...` and
    checked only "two 200s, non-empty content, no 'goodput' substring". It never
    compared `before` to `after` at all, so it would have passed over any change
    it had not thought to name — a test whose NAME claimed more than its
    assertions checked.

    ⚠️ **AND THE NAME IS NARROWER THAN "BYTE-IDENTICAL" ON PURPOSE, BECAUSE THE
    BLANKET CLAIM IS FALSE.** With thresholds armed, shadow mode demonstrably
    changes: one extra `/metrics` GET per poller tick, new `goodput` fields on
    `/v1/status`, eight new `/metrics` series, and a `logger.critical` per trip.
    That IS the observability surface the dark soak exists to produce. What must
    not change is what a CALLER on the inference door receives —
    `test_the_shadow_soak_is_readable_on_status_and_metrics` asserts the other
    half positively, so the two together say where the line falls rather than
    implying there is none.
    """
    _arm(proxy)
    assert proxy.svc._state.flags.get(FLAG) is False, "the default must be OFF"
    _wedged(proxy.controller)

    before = await proxy.chat("hi", model=EP_ALIAS)
    await _poll(proxy, ticks=4)
    after = await proxy.chat("hi", model=EP_ALIAS)

    # The verdict LATCHED — it is an observation, populated in shadow so the soak
    # can read what enforcement WOULD have shed…
    assert EP_CLASS in proxy.svc._state.collapsed_endpoints, (
        "nothing latched, so this test proves nothing about shadow mode")
    assert proxy.svc._health.endpoint_healthy(EP_ALIAS) is True

    # …and the caller could not tell.
    assert _caller_visible(before) == _caller_visible(after), (
        "a shadow trip changed what the caller received — the feature is not dark")
    assert before.status_code == 200
    assert before.json()["choices"][0]["message"]["content"]
    assert "goodput" not in str(before.json())


async def test_a_shadow_recovery_does_not_wake_the_dispatch_loop(proxy):
    """The one shadow-mode side effect that was NOT observability.

    On a CLEAR, `sample_goodput` wakes the dispatcher so the endpoint's deferred
    queue drains. In SHADOW the trip deferred nothing — `endpoint_healthy` never
    returned False — so that wakeup was a bare timing perturbation caused by a
    feature that is supposed to observe and nothing else. Small, and exactly the
    kind of "nearly dark" that makes a soak measure a different system from the
    one the flag flip will arm. It is gated on the flag now.
    """
    _arm(proxy)
    assert proxy.svc._state.flags.get(FLAG) is False
    _wedged(proxy.controller)
    await _poll(proxy, ticks=4)
    assert EP_CLASS in proxy.svc._state.collapsed_endpoints, "never latched"

    proxy.svc._state.dispatch_event.clear()
    for tick in range(1, 6):
        _working(proxy.controller, tick)
        await _poll(proxy, ticks=1)
        if EP_CLASS not in proxy.svc._state.collapsed_endpoints:
            break
    assert EP_CLASS not in proxy.svc._state.collapsed_endpoints, "never cleared"
    assert not proxy.svc._state.dispatch_event.is_set(), (
        "a SHADOW recovery woke the dispatch loop — in shadow there is no "
        "deferred queue to drain, so this is the feature perturbing the system "
        "it is only supposed to be watching")


async def test_an_ARMED_recovery_DOES_wake_the_dispatch_loop(proxy):
    """🚨 THE POSITIVE CONTROL for the test above.

    Without it, that test passes on a code path that never wakes the dispatcher at
    all — and the wakeup is load-bearing when armed: the trip really did defer a
    queue, and nothing else will drain it until the next tick.
    """
    _arm(proxy)
    proxy.svc._state.flags.set_many({FLAG: True})
    _wedged(proxy.controller)
    await _poll(proxy, ticks=4)
    assert EP_CLASS in proxy.svc._state.collapsed_endpoints, "never latched"

    proxy.svc._state.dispatch_event.clear()
    for tick in range(1, 6):
        _working(proxy.controller, tick)
        await _poll(proxy, ticks=1)
        if EP_CLASS not in proxy.svc._state.collapsed_endpoints:
            break
    assert EP_CLASS not in proxy.svc._state.collapsed_endpoints, "never cleared"
    assert proxy.svc._state.dispatch_event.is_set(), (
        "an ARMED recovery must drain the queue its trip deferred")


async def test_the_shadow_soak_is_readable_on_status_and_metrics(proxy):
    """What an operator reads to decide whether to arm it.

    `tripped: true, enforced: false` is the pair that makes the dark ship useful;
    without it the soak reports nothing and the flag flip is a guess.
    """
    _arm(proxy)
    _wedged(proxy.controller)
    await _poll(proxy, ticks=4)

    snap = await _status_ep(proxy)
    gp = snap["goodput"]
    assert gp["verdict"] == "collapsed"
    assert gp["tripped"] is True
    assert gp["enforced"] is False
    assert gp["trips"] >= 1
    assert gp["clauses_held"] == ["occupancy", "iterations", "generation", "prefill"]
    assert gp["running_min"] == 6
    assert gp["prefill_tps"] == 0.0
    # ⚠️ And it must not have been folded into the fields that already exist —
    # `paused` here means `not healthy`, which a shadow trip must not touch.
    assert snap["healthy"] is True
    assert snap["paused"] is False
    assert "admin_paused" not in snap

    text = await _metrics(proxy)
    assert f'roadstead_endpoint_goodput_collapsed{{endpoint="{EP_CLASS}"}} 1' in text
    assert f'roadstead_endpoint_goodput_unknown{{endpoint="{EP_CLASS}"}} 0' in text
    assert f'roadstead_endpoint_goodput_tripped{{endpoint="{EP_CLASS}"}} 1' in text
    assert "roadstead_endpoint_goodput_prefill_tps" in text


async def test_an_unconfigured_endpoint_publishes_nothing_anywhere(proxy):
    """Absent ⇒ off, on every surface.

    No thresholds declared anywhere here, so `/v1/status` must carry no `goodput`
    key for ANY endpoint and `/metrics` no goodput series. 🚨 A zeroed series
    would read as "watched and fine" — the sample-floor rule this repo already
    applies to the structured-empty pair.
    """
    _wedged(proxy.controller)
    await _poll(proxy, ticks=4)

    status = (await proxy.client.get("/v1/status")).json()
    for ep, snap in status["endpoints"].items():
        assert "goodput" not in snap, f"{ep} published a verdict it never earned"
    assert "roadstead_endpoint_goodput" not in await _metrics(proxy)
    assert proxy.svc._state.collapsed_endpoints == set()


# --------------------------------------------------------------------------- #
# 3 — the invariant, end to end
# --------------------------------------------------------------------------- #

async def test_a_blind_endpoint_never_sheds_and_says_so(proxy):
    """🚨 BLINDNESS MUST NEVER READ AS COLLAPSE — through the real front door.

    The fake publishes NO engine work counters (every knob left at None, the
    llama.cpp/absent shape), while the thresholds declare all three progress
    clauses. The conjunction has terms it cannot evaluate, and because it is a
    conjunction, skipping them would make firing EASIER.

    So: unknown, never tripped, the caller served throughout — and the blindness
    published as its own signal, because an endpoint whose instrument is missing
    must not read as verified-healthy either.
    """
    _arm(proxy)
    proxy.svc._state.flags.set_many({FLAG: True})
    proxy.controller.num_requests_running = None
    proxy.controller.iteration_tokens_count = None
    proxy.controller.generation_tokens_total = None
    proxy.controller.prompt_tokens_total = None

    await _poll(proxy, ticks=5)

    assert proxy.svc._state.collapsed_endpoints == set(), (
        "a missing instrument must not be able to trip the breaker")
    assert (await proxy.chat("hi", model=EP_ALIAS)).status_code == 200

    gp = (await _status_ep(proxy))["goodput"]
    assert gp["verdict"] == "unknown"
    assert gp["reason"].startswith("counters_absent:")

    text = await _metrics(proxy)
    assert f'roadstead_endpoint_goodput_unknown{{endpoint="{EP_CLASS}"}} 1' in text
    assert f'roadstead_endpoint_goodput_collapsed{{endpoint="{EP_CLASS}"}}' not in text, (
        "an unevaluable endpoint must publish NO collapsed series — a 0 there "
        "reads as verified-healthy")

    alerts = {a["name"] for a in (await proxy.client.get("/v1/status")).json()["alerts"]}
    assert "endpoint_goodput_blind" in alerts


async def test_a_prefill_heavy_caller_is_never_shed(proxy):
    """🚨 THE FALSE-POSITIVE CONTROL, at the front door — the `hermes` case.

    Measured: 31-37k-token prompts returning 2-3 output tokens, six concurrent,
    reading 0.42 generation tokens/sec/busy-slot — inside the wedge band — while
    perfectly healthy, because the engine is doing 1,209 prompt tokens/sec of real
    prefill. Wedge minutes never exceeded 247.5.

    If this ever fails, the feature is shedding a legitimate caller's traffic, and
    the arming flag should go back to False.
    """
    _arm(proxy)
    proxy.svc._state.flags.set_many({FLAG: True})

    for tick in range(6):
        proxy.controller.num_requests_running = 6
        proxy.controller.iteration_tokens_count = 500_000 + tick * 4
        proxy.controller.generation_tokens_total = 9_000_000 + tick * 25
        proxy.controller.prompt_tokens_total = 80_000_000 + tick * 12_090
        await _poll(proxy, ticks=1)

    gp = (await _status_ep(proxy))["goodput"]
    assert gp["generation_tps_per_request"] == pytest.approx(0.42, abs=0.02), (
        "the control must sit inside the wedge band or it proves nothing")
    assert gp["prefill_tps"] == pytest.approx(1209.0, abs=1.0)
    assert gp["verdict"] == "healthy"
    assert gp["clauses_cleared"] == ["prefill"], (
        "prefill must be the ONLY clause saving it")
    assert proxy.svc._state.collapsed_endpoints == set()
    assert (await proxy.chat("hi", model=EP_ALIAS)).status_code == 200
