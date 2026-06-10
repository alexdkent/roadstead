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


def _metrics_with_timeouts(endpoint: str, n: int, now: float) -> RollingMetrics:
    m = RollingMetrics(window_s=300.0)
    for _ in range(n):
        m.record(MetricsSample(
            timestamp=now, endpoint=endpoint, agent_id="a", priority="P3_INGESTION",
            queue_wait_ms=0.0, backend_latency_ms=0.0, status="timeout",
            slot_seconds=0.0))
    return m


def _names(alerts):
    return {a.name for a in alerts}


def test_endpoint_stalled_fires_on_free_slots_timeout_burst():
    now = time.monotonic()
    m = _metrics_with_timeouts("thinker", 4, now)
    snaps = {"thinker": {"in_flight": 3, "max_slots": 32, "queued": 0, "paused": False}}
    alerts = check_alerts(
        endpoint_snapshots=snaps, agent_budgets=[], metrics=m,
        cost_model_samples={}, queue_wal_size=0, now=now)
    assert "endpoint_stalled" in _names(alerts)


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
    m = _metrics_with_timeouts("thinker", 2, now)  # < 3
    snaps = {"thinker": {"in_flight": 1, "max_slots": 32, "queued": 0, "paused": False}}
    alerts = check_alerts(
        endpoint_snapshots=snaps, agent_budgets=[], metrics=m,
        cost_model_samples={}, queue_wal_size=0, now=now)
    assert "endpoint_stalled" not in _names(alerts)


def test_intertoken_gap_constant_present():
    # The mid-stream no-progress watchdog must be wired (extends the
    # first-token-only TTFT watchdog).
    from originfleet.llmproxy import service
    assert service._STREAM_INTERTOKEN_GAP_S > 0


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
