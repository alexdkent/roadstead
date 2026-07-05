"""Stall resilience (2026-06-06): detect a backend that is alive-but-not-
generating (the thinker contention episode — /health up, generation throughput
collapsed, requests burning their full timeout on FREE slots) and make it
VISIBLE, without auto-fast-failing a backend that is only partially degraded.
"""

from __future__ import annotations

import time

from originfleet.llmproxy.observability import (
    MetricsSample,
    RollingMetrics,
    check_alerts,
)


def _metrics_with_timeouts(
    endpoint: str, n: int, now: float, *,
    premature: bool = False, best_effort: bool = False,
) -> RollingMetrics:
    m = RollingMetrics(window_s=300.0)
    for _ in range(n):
        m.record(MetricsSample(
            timestamp=now, endpoint=endpoint, agent_id="a", priority="P3_INGESTION",
            queue_wait_ms=0.0, backend_latency_ms=0.0, status="timeout",
            slot_seconds=0.0, premature=premature, best_effort=best_effort))
    return m


def _names(alerts):
    return {a.name for a in alerts}


def _severity(alerts, name):
    return next(a.severity for a in alerts if a.name == name)


def test_endpoint_stalled_fires_on_free_slots_timeout_burst():
    now = time.monotonic()
    m = _metrics_with_timeouts("thinker", 8, now)  # >= threshold (8)
    snaps = {"thinker": {"in_flight": 3, "max_slots": 32, "queued": 0, "paused": False}}
    alerts = check_alerts(
        endpoint_snapshots=snaps, agent_budgets=[], metrics=m,
        cost_model_samples={}, queue_wal_size=0, now=now)
    assert "endpoint_stalled" in _names(alerts)
    # A non-best-effort backend (thinker) stays ERROR severity.
    assert _severity(alerts, "endpoint_stalled") == "ERROR"


def test_no_stall_when_saturated():
    # Slots full + queued ⇒ saturation, not a stall — must NOT fire.
    now = time.monotonic()
    m = _metrics_with_timeouts("thinker", 8, now)
    snaps = {"thinker": {"in_flight": 32, "max_slots": 32, "queued": 5, "paused": False}}
    alerts = check_alerts(
        endpoint_snapshots=snaps, agent_budgets=[], metrics=m,
        cost_model_samples={}, queue_wal_size=0, now=now)
    assert "endpoint_stalled" not in _names(alerts)


def test_no_stall_below_threshold():
    now = time.monotonic()
    m = _metrics_with_timeouts("thinker", 5, now)  # < threshold (8)
    snaps = {"thinker": {"in_flight": 1, "max_slots": 32, "queued": 0, "paused": False}}
    alerts = check_alerts(
        endpoint_snapshots=snaps, agent_budgets=[], metrics=m,
        cost_model_samples={}, queue_wal_size=0, now=now)
    assert "endpoint_stalled" not in _names(alerts)


def test_no_stall_on_premature_timeout_burst():
    # Best-effort gemma paths (greeter advisory ~0.9s, sidekick.extract_obs 3.0s) give
    # up below the gemma 60s floor by design. A burst of those premature
    # timeouts with free slots is a CLIENT give-up, not a backend stall — it must
    # NOT manufacture an endpoint_stalled ERROR (the false-alert churn fixed here).
    now = time.monotonic()
    m = _metrics_with_timeouts("gemma", 8, now, premature=True)
    snaps = {"gemma": {"in_flight": 0, "max_slots": 2, "queued": 0, "paused": False}}
    alerts = check_alerts(
        endpoint_snapshots=snaps, agent_budgets=[], metrics=m,
        cost_model_samples={}, queue_wal_size=0, now=now)
    assert "endpoint_stalled" not in _names(alerts)


def test_genuine_timeouts_still_fire_amid_premature_noise():
    # A real stall (requests waiting PAST the recommended deadline → premature=
    # False, best_effort=False) must still surface even when premature give-ups
    # share the window.
    now = time.monotonic()
    m = _metrics_with_timeouts("gemma", 5, now, premature=True)
    for _ in range(8):  # genuine, non-premature, non-best-effort (>= threshold)
        m.record(MetricsSample(
            timestamp=now, endpoint="gemma", agent_id="a", priority="P3_INGESTION",
            queue_wait_ms=0.0, backend_latency_ms=0.0, status="timeout",
            slot_seconds=0.0, premature=False, best_effort=False))
    snaps = {"gemma": {"in_flight": 0, "max_slots": 2, "queued": 0, "paused": False}}
    alerts = check_alerts(
        endpoint_snapshots=snaps, agent_budgets=[], metrics=m,
        cost_model_samples={}, queue_wal_size=0, now=now)
    assert "endpoint_stalled" in _names(alerts)


def test_best_effort_timeouts_excluded_from_stall():
    # 2026-07-05 gemma stall-burst fix: a burst of best_effort timeouts (a caller
    # applied a deadline far below the recommended time — greeter advisory ~0.9s,
    # sidekick.extract ~3s) is NOT backend-stall evidence even when non-premature (a
    # proxy-initiated fast-fail on such a path forces premature=False). Well above
    # the fire threshold, it must still NOT manufacture endpoint_stalled — this is
    # the amplifier that latched the real-world bursts.
    now = time.monotonic()
    m = _metrics_with_timeouts("gemma", 20, now, best_effort=True)
    snaps = {"gemma": {"in_flight": 0, "max_slots": 2, "queued": 0, "paused": False}}
    alerts = check_alerts(
        endpoint_snapshots=snaps, agent_budgets=[], metrics=m,
        cost_model_samples={}, queue_wal_size=0, now=now)
    assert "endpoint_stalled" not in _names(alerts)


def test_gemma_stall_is_warning_not_error():
    # A genuine gemma stall (fair-time timeouts) still fires, but gemma is a
    # best-effort router/greeter whose stall degrades gracefully → WARNING, not
    # the ERROR that pages / reads as fleet-degraded.
    now = time.monotonic()
    m = _metrics_with_timeouts("gemma", 8, now)  # genuine, >= threshold
    snaps = {"gemma": {"in_flight": 0, "max_slots": 2, "queued": 0, "paused": False}}
    alerts = check_alerts(
        endpoint_snapshots=snaps, agent_budgets=[], metrics=m,
        cost_model_samples={}, queue_wal_size=0, now=now)
    assert "endpoint_stalled" in _names(alerts)
    assert _severity(alerts, "endpoint_stalled") == "WARNING"


def test_timeout_below_recommended_ratio():
    # The best_effort tag: an applied deadline under half the recommended time is
    # a sub-floor give-up; anything else (incl. missing/zero inputs) is genuine.
    from originfleet.llmproxy.lifecycle import _timeout_below_recommended
    assert _timeout_below_recommended(0.9, 60000) is True    # greeter ~0.9s vs 60s
    assert _timeout_below_recommended(3.0, 60000) is True    # sidekick.extract ~3s vs 60s
    assert _timeout_below_recommended(45.0, 60000) is False  # 45s vs 60s — fair time
    assert _timeout_below_recommended(60.0, 60000) is False  # at the recommended
    assert _timeout_below_recommended(0.9, 0) is False       # no recommendation → genuine
    assert _timeout_below_recommended(0.0, 60000) is False    # no applied deadline → genuine
    assert _timeout_below_recommended(None, None) is False    # both missing → genuine


def test_intertoken_gap_constant_present():
    # The mid-stream no-progress watchdog must be wired (extends the
    # first-token-only TTFT watchdog). The streaming path (and its constant)
    # moved to the Lifecycle collaborator in de-monolith Step 3.
    from originfleet.llmproxy import lifecycle
    assert lifecycle._STREAM_INTERTOKEN_GAP_S > 0


# --- Phase 2 hardening: no-consumer streaming dispatch frees its slot --------

import asyncio
import pytest

from originfleet.llmproxy.config import ProxyConfig
from originfleet.llmproxy.scheduler import QueuedRequest
from originfleet.llmproxy.service import ProxyService


@pytest.mark.asyncio
async def test_streaming_dispatch_without_consumer_records_completion():
    """A streaming dispatch whose _pending_streams entry is missing (the
    recovered-after-restart shape) must record a 'cancelled' completion so the
    scheduler slot frees instead of leaking until the next proxy restart."""
    svc = ProxyService(ProxyConfig())
    await svc.startup()
    try:
        req = QueuedRequest.create(
            agent_id="a", endpoint="thinker", priority="P3_INGESTION",
            call_site="t", payload_type="chat_completion",
            payload={"messages": [{"role": "user", "content": "x"}],
                     "stream": True},
            timeout_s=30.0,
        )
        # Enqueue WITHOUT handle_submit → no _pending_streams entry, exactly
        # like a recovered row dispatched by the scheduler loop.
        svc._scheduler.enqueue(req)
        svc._dispatch_event.set()

        for _ in range(200):
            snap = svc._scheduler.endpoint_snapshot("thinker")
            if snap["queued"] == 0 and snap["in_flight"] == 0 \
                    and svc._scheduler.stats()["total_completed"] >= 1:
                break
            await asyncio.sleep(0.01)

        snap = svc._scheduler.endpoint_snapshot("thinker")
        assert snap["in_flight"] == 0, "slot leaked for the consumer-less stream"
        assert snap["queued"] == 0
        assert svc._scheduler.stats()["total_completed"] == 1
    finally:
        await svc.shutdown()
