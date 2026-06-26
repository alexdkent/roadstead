"""Metrics collection, request logging, and alerting.

Provides rolling-window metrics, JSONL request logging, and Jain's
fairness index computation.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

logger = logging.getLogger(__name__)

# Request statuses that are an ACTUAL failure worth surfacing in the text log +
# docker logs. Everything else (ok, cancelled=client-disconnect) is silent there
# — the JSONL is the authoritative per-request record. (persistence cleanup)
_PROBLEM_STATUSES = frozenset({"error", "timeout", "truncated"})


# ---------------------------------------------------------------------------
# Request log record
# ---------------------------------------------------------------------------

@dataclass
class RequestLogRecord:
    ts: str
    request_id: str
    agent_id: str
    endpoint: str
    call_site: str
    priority: str
    band: str
    payload_type: str
    input_tokens: int
    output_tokens: int
    estimated_cost_ss: float
    actual_cost_ss: float
    queue_wait_ms: float
    backend_latency_ms: float
    total_latency_ms: float
    occupancy_at_dispatch: int
    status: str
    cache_hit: bool = False
    coalesced: bool = False
    estimated_input_tokens: int = 0
    max_output_tokens: int = 0
    session_id: str | None = None
    turn_id: str | None = None
    caller_id: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "ts": self.ts,
                "request_id": self.request_id,
                "agent_id": self.agent_id,
                "endpoint": self.endpoint,
                "call_site": self.call_site,
                "priority": self.priority,
                "band": self.band,
                "payload_type": self.payload_type,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "estimated_cost_ss": round(self.estimated_cost_ss, 3),
                "actual_cost_ss": round(self.actual_cost_ss, 3),
                "queue_wait_ms": round(self.queue_wait_ms, 1),
                "backend_latency_ms": round(self.backend_latency_ms, 1),
                "total_latency_ms": round(self.total_latency_ms, 1),
                "occupancy_at_dispatch": self.occupancy_at_dispatch,
                "status": self.status,
                "estimated_input_tokens": self.estimated_input_tokens,
                "max_output_tokens": self.max_output_tokens,
                "cache_hit": self.cache_hit,
                "coalesced": self.coalesced,
                "session_id": self.session_id,
                "turn_id": self.turn_id,
                "caller_id": self.caller_id,
            },
            separators=(",", ":"),
        )


class RequestLogger:
    """Writes JSONL request logs to a file (the authoritative per-request
    record). Genuine failures also surface to the text log / docker logs; ok /
    cancelled requests do not (the JSONL is the source of truth)."""

    def __init__(self, log_path: str | None = None) -> None:
        self._file: TextIO | None = None
        if log_path:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "a", buffering=1)

    def log(self, record: RequestLogRecord) -> None:
        line = record.to_json()
        if self._file:
            self._file.write(line + "\n")
        # JSONL is authoritative; only surface genuine failures in the text log /
        # docker logs (was an unconditional INFO that triplicated every request).
        if record.status in _PROBLEM_STATUSES:
            logger.warning("req: %s", line)

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


# ---------------------------------------------------------------------------
# Rolling metrics window
# ---------------------------------------------------------------------------

@dataclass
class MetricsSample:
    timestamp: float
    endpoint: str
    agent_id: str
    priority: str
    queue_wait_ms: float
    backend_latency_ms: float
    status: str
    slot_seconds: float
    # True when a `status="timeout"` sample fired BELOW the proxy's own
    # recommended deadline for that model/size (a client-side give-up, not a
    # backend problem). Best-effort interactive paths (gemma greeter advisory,
    # sidekick.extract_obs) deliberately apply sub-second deadlines well under the
    # gemma 60s floor; their give-ups must NOT be read as a backend stall.
    premature: bool = False


class RollingMetrics:
    """Rolling-window metrics for the last N seconds."""

    def __init__(self, window_s: float = 300.0) -> None:
        self._window_s = window_s
        self._samples: deque[MetricsSample] = deque()

    def record(self, sample: MetricsSample) -> None:
        self._samples.append(sample)
        self._prune(sample.timestamp)

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_s
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()

    def percentile(
        self,
        field: str,
        pct: float,
        *,
        endpoint: str | None = None,
        agent_id: str | None = None,
        now: float | None = None,
    ) -> float:
        if now:
            self._prune(now)
        values = []
        for s in self._samples:
            if endpoint and s.endpoint != endpoint:
                continue
            if agent_id and s.agent_id != agent_id:
                continue
            values.append(getattr(s, field, 0.0))
        if not values:
            return 0.0
        values.sort()
        idx = int(len(values) * pct / 100)
        idx = min(idx, len(values) - 1)
        return values[idx]

    def count(
        self,
        *,
        endpoint: str | None = None,
        agent_id: str | None = None,
        status: str | None = None,
        premature: bool | None = None,
        now: float | None = None,
    ) -> int:
        if now:
            self._prune(now)
        n = 0
        for s in self._samples:
            if endpoint and s.endpoint != endpoint:
                continue
            if agent_id and s.agent_id != agent_id:
                continue
            if status and s.status != status:
                continue
            if premature is not None and s.premature != premature:
                continue
            n += 1
        return n

    def throughput_rps(
        self, endpoint: str | None = None, now: float | None = None,
    ) -> float:
        if now:
            self._prune(now)
        n = self.count(endpoint=endpoint)
        elapsed = self._window_s
        if self._samples:
            elapsed = max(1.0, (self._samples[-1].timestamp - self._samples[0].timestamp))
        return n / elapsed if elapsed > 0 else 0

    def slot_seconds_consumed(
        self, endpoint: str | None = None, now: float | None = None,
    ) -> float:
        if now:
            self._prune(now)
        total = 0.0
        for s in self._samples:
            if endpoint and s.endpoint != endpoint:
                continue
            total += s.slot_seconds
        return total

    def per_agent_consumed(self, now: float | None = None) -> dict[str, float]:
        if now:
            self._prune(now)
        result: dict[str, float] = {}
        for s in self._samples:
            result[s.agent_id] = result.get(s.agent_id, 0.0) + s.slot_seconds
        return result


# ---------------------------------------------------------------------------
# Fairness metrics
# ---------------------------------------------------------------------------

def jains_fairness_index(shares: list[float]) -> float:
    """Jain's fairness index.  1.0 = perfectly fair.

    J(x) = (sum(x))^2 / (n * sum(x^2))
    """
    n = len(shares)
    if n == 0:
        return 1.0
    total = sum(shares)
    sum_sq = sum(x * x for x in shares)
    if sum_sq == 0:
        return 1.0
    return (total * total) / (n * sum_sq)


# ---------------------------------------------------------------------------
# Alert conditions
# ---------------------------------------------------------------------------

@dataclass
class AlertCondition:
    name: str
    severity: str        # "INFO" | "WARNING" | "ERROR" | "CRITICAL"
    triggered: bool
    detail: str = ""


def check_alerts(
    *,
    endpoint_snapshots: dict[str, dict],
    agent_budgets: list[dict],
    metrics: RollingMetrics,
    cost_model_samples: dict[str, int],
    queue_wal_size: int,
    now: float,
) -> list[AlertCondition]:
    """Evaluate alert conditions.  Returns triggered alerts."""
    alerts: list[AlertCondition] = []

    # Admission timeout spike (>5 in 5-min window)
    timeout_count = metrics.count(status="timeout", now=now)
    if timeout_count > 5:
        alerts.append(AlertCondition(
            "admission_timeout_spike", "WARNING", True,
            f"{timeout_count} timeouts in last 5 minutes",
        ))

    # Endpoint stall (alive-but-not-generating): a burst of timeouts on an
    # endpoint that has FREE slots and NOTHING queued ⇒ requests are hanging on
    # a backend that isn't saturated. This is the signature the /health probe
    # CANNOT see (the backend answers /health while generation throughput
    # collapses — the 2026-06-06 thinker contention episode). Surfaced as an
    # ERROR for operator response; deliberately NOT an auto-circuit-trip — the
    # backend is usually only PARTIALLY degraded (most requests still succeed),
    # so fast-failing all of its traffic would be worse than letting the healthy
    # majority through. Distinct from admission_timeout_spike (global, load).
    #
    # Count only GENUINE timeouts (premature=False). A premature timeout fired
    # below the proxy's own recommended deadline — a client-side give-up, not a
    # backend stall. Best-effort interactive gemma paths (greeter advisory ~0.9s,
    # sidekick.extract_obs 3.0s) deliberately apply sub-second deadlines well under
    # the gemma 60s floor and routinely give up early; counting those manufactured
    # false `endpoint_stalled` ERROR bursts (same root cause as the creative-boxa
    # 120/220s-constant episode, fixed there by deriving from the advice). A real
    # stall makes requests wait PAST the recommended deadline → premature=False →
    # still counted, so the genuine signal is preserved, only sharper.
    for ep, snap in endpoint_snapshots.items():
        ep_timeouts = metrics.count(
            endpoint=ep, status="timeout", premature=False, now=now)
        max_slots = snap.get("max_slots", 0) or 0
        if (ep_timeouts >= 3
                and snap.get("queued", 0) == 0
                and (max_slots == 0 or snap.get("in_flight", 0) < max_slots)):
            alerts.append(AlertCondition(
                "endpoint_stalled", "ERROR", True,
                f"endpoint {ep}: {ep_timeouts} non-premature timeouts in 5min "
                f"with free slots (in_flight={snap.get('in_flight', 0)}/"
                f"{max_slots or '?'}, queued=0) — backend likely stalling, "
                f"not saturated",
            ))

    # Agent starvation — REMOVED. The DRR fix (pick_agent on head-of-queue wait)
    # made balance-sign "starving" a lie: a heavy consumer being served
    # continuously has a perpetually-negative balance yet is NOT starved. Real
    # starvation = denied-service wait, which lives in the scheduler, not this
    # budget snapshot. Don't page on the misleading signal; a proper
    # head-of-queue-wait alert is a future refinement.

    # Endpoint paused
    for ep, snap in endpoint_snapshots.items():
        if snap.get("paused"):
            alerts.append(AlertCondition(
                "endpoint_paused", "ERROR", True,
                f"endpoint {ep} backend unreachable",
            ))

    # Queue depth excessive
    for ep, snap in endpoint_snapshots.items():
        if snap.get("queued", 0) > 10:
            alerts.append(AlertCondition(
                "queue_depth_excessive", "WARNING", True,
                f"endpoint {ep} has {snap['queued']} queued requests",
            ))

    # Fairness imbalance — only actionable UNDER CONTENTION. Phase 5C: gate on a
    # real queue somewhere. An imbalanced slot-second share with every endpoint
    # idle (queued=0) is harmless (a heavy consumer being served alone is not
    # starving anyone) and was firing continuously as noise. Only page when at
    # least one endpoint actually has requests waiting.
    any_queued = any(s.get("queued", 0) > 0 for s in endpoint_snapshots.values())
    if any_queued:
        consumed = metrics.per_agent_consumed(now)
        shares = list(consumed.values())
        if len(shares) >= 2:
            ji = jains_fairness_index(shares)
            if ji < 0.7:
                alerts.append(AlertCondition(
                    "drr_imbalance", "WARNING", True,
                    f"Jain's index = {ji:.2f} (with queued work)",
                ))

    # Cost model stale
    for ep, count in cost_model_samples.items():
        if count < 10:
            alerts.append(AlertCondition(
                "cost_model_stale", "INFO", True,
                f"endpoint {ep} has only {count} cost model samples",
            ))

    # WAL growth
    if queue_wal_size > 10 * 1024 * 1024:
        alerts.append(AlertCondition(
            "wal_growth", "WARNING", True,
            f"queue WAL size = {queue_wal_size / 1024 / 1024:.1f} MB",
        ))

    return alerts
