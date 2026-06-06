"""DRR scheduler with priority bands and concurrency-aware admission.

Pure computation — no I/O, no framework imports.  The scheduler
operates on ``QueuedRequest`` objects and produces ``DispatchDecision``
results.  Actual HTTP calls, persistence, and I/O are the caller's
responsibility.

Architecture:
  - Three priority bands: INTERACTIVE > FOREGROUND > BACKGROUND
  - Within each band: deficit-round-robin across agents
  - Per-endpoint concurrency gating with degradation awareness
  - Background floor guarantee (configurable % of each endpoint)
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from .agent_budget import BudgetManager
from .config import (
    EndpointConfig,
    LLMPriority,
    PriorityBand,
    ProxyConfig,
    normalize_endpoint,
    priority_to_band,
)
from .cost_model import CostModel, estimate_input_tokens


# ---------------------------------------------------------------------------
# Request / active-request types
# ---------------------------------------------------------------------------

@dataclass
class QueuedRequest:
    request_id: str
    agent_id: str
    endpoint: str              # normalized endpoint class
    priority: LLMPriority
    band: PriorityBand
    call_site: str
    payload_type: str          # "chat_completion" | "embedding" | "rerank"
    payload: dict
    timeout_deadline: float    # monotonic deadline
    enqueued_at: float         # monotonic timestamp
    estimated_cost_ss: float = 0.0
    est_input_tokens: int = 0  # cached at enqueue (context size) for the live in-flight view
    session_id: str | None = None
    turn_id: str | None = None
    caller_id: str | None = None
    timeout_s: float = 180.0
    stream: bool = False

    @classmethod
    def create(
        cls,
        *,
        agent_id: str,
        endpoint: str,
        priority: LLMPriority | str | int | None,
        call_site: str,
        payload_type: str,
        payload: dict,
        timeout_s: float = 180.0,
        session_id: str | None = None,
        turn_id: str | None = None,
        caller_id: str | None = None,
        request_id: str | None = None,
        now: float | None = None,
    ) -> QueuedRequest:
        # Soft-default a malformed priority to P1 (never raise on the request
        # path — a bad label must not 500 the caller's LLM call).
        pri = LLMPriority.coerce(priority, default=LLMPriority.P1_TURN_SUPPORT)
        ep = normalize_endpoint(endpoint)
        ts = now if now is not None else time.monotonic()
        return cls(
            request_id=request_id or f"req_{uuid.uuid4().hex[:12]}",
            agent_id=agent_id,
            endpoint=ep,
            priority=pri,
            band=priority_to_band(pri),
            call_site=call_site,
            payload_type=payload_type,
            payload=payload,
            timeout_deadline=ts + timeout_s,
            enqueued_at=ts,
            timeout_s=timeout_s,
            session_id=session_id,
            turn_id=turn_id,
            caller_id=caller_id,
            stream=bool(payload.get("stream")),
        )


@dataclass
class ActiveRequest:
    """A request currently being processed by a backend."""
    request: QueuedRequest
    dispatched_at: float       # monotonic timestamp
    estimated_remaining_s: float = 0.0

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def endpoint(self) -> str:
        return self.request.endpoint

    @property
    def agent_id(self) -> str:
        return self.request.agent_id


@dataclass
class DispatchDecision:
    """Result of a scheduler dispatch decision."""
    request: QueuedRequest
    queue_wait_ms: float
    occupancy_at_dispatch: int


@dataclass
class CompletionRecord:
    """Reported by the caller when a dispatched request finishes."""
    request_id: str
    duration_s: float
    input_tokens: int
    output_tokens: int
    success: bool
    occupancy_during: int


# ---------------------------------------------------------------------------
# Per-endpoint queue
# ---------------------------------------------------------------------------

class EndpointQueue:
    """Per-endpoint, per-band, per-agent request queues."""

    def __init__(self) -> None:
        # band → agent_id → FIFO deque of requests
        self._queues: dict[PriorityBand, dict[str, deque[QueuedRequest]]] = {
            band: {} for band in PriorityBand
        }
        # Round-robin order per band
        self._robin: dict[PriorityBand, deque[str]] = {
            band: deque() for band in PriorityBand
        }

    def enqueue(self, req: QueuedRequest) -> None:
        band_q = self._queues[req.band]
        if req.agent_id not in band_q:
            band_q[req.agent_id] = deque()
            self._robin[req.band].append(req.agent_id)
        band_q[req.agent_id].append(req)

    def dequeue(self, band: PriorityBand, agent_id: str) -> QueuedRequest | None:
        band_q = self._queues[band]
        agent_q = band_q.get(agent_id)
        if not agent_q:
            return None
        req = agent_q.popleft()
        if not agent_q:
            del band_q[agent_id]
            try:
                self._robin[band].remove(agent_id)
            except ValueError:
                pass
        return req

    def peek(self, band: PriorityBand, agent_id: str) -> QueuedRequest | None:
        agent_q = self._queues[band].get(agent_id)
        if not agent_q:
            return None
        return agent_q[0]

    def agents_in_band(self, band: PriorityBand) -> list[str]:
        return list(self._robin[band])

    def band_depth(self, band: PriorityBand) -> int:
        return sum(len(q) for q in self._queues[band].values())

    def agent_depth(self, band: PriorityBand, agent_id: str) -> int:
        q = self._queues[band].get(agent_id)
        return len(q) if q else 0

    def total_depth(self) -> int:
        return sum(self.band_depth(b) for b in PriorityBand)

    def rotate_robin(self, band: PriorityBand) -> None:
        """Advance the round-robin pointer for this band."""
        robin = self._robin[band]
        if robin:
            robin.rotate(-1)

    def expire_before(self, deadline: float) -> list[QueuedRequest]:
        """Remove and return all requests whose timeout_deadline has passed."""
        expired: list[QueuedRequest] = []
        for band in PriorityBand:
            for agent_id in list(self._queues[band].keys()):
                q = self._queues[band][agent_id]
                while q and q[0].timeout_deadline <= deadline:
                    expired.append(q.popleft())
                if not q:
                    del self._queues[band][agent_id]
                    try:
                        self._robin[band].remove(agent_id)
                    except ValueError:
                        pass
        return expired


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class Scheduler:
    """Core DRR scheduler with priority bands.

    This is the pure-computation engine.  It does not make HTTP calls
    or touch the filesystem.  The caller (the proxy service) feeds it
    requests via ``enqueue()``, retrieves dispatch decisions via
    ``tick()``, and reports completions via ``complete()``.
    """

    def __init__(
        self,
        config: ProxyConfig,
        cost_model: CostModel,
        budget_manager: BudgetManager,
    ) -> None:
        self._config = config
        self._cost = cost_model
        self._budgets = budget_manager

        # Per-endpoint queues
        self._queues: dict[str, EndpointQueue] = {}
        for ep in config.endpoints:
            self._queues[ep] = EndpointQueue()

        # Active (in-flight) requests per endpoint
        self._active: dict[str, dict[str, ActiveRequest]] = {
            ep: {} for ep in config.endpoints
        }

        # Stats
        self._total_dispatched: int = 0
        self._total_timeouts: int = 0
        self._total_completed: int = 0

        # Callbacks (set by the proxy service)
        self.on_dispatch: Callable[[DispatchDecision], None] | None = None
        self.on_timeout: Callable[[QueuedRequest], None] | None = None
        # Circuit breaker: the service owns per-endpoint health (probe-based);
        # when set and an endpoint is unhealthy, the scheduler stops dispatching
        # to it so its queue defers instead of feeding a dead backend.
        self.is_endpoint_healthy: Callable[[str], bool] | None = None

    # ----- public API -----

    def enqueue(self, req: QueuedRequest) -> None:
        """Add a request to the scheduling queue."""
        ep = req.endpoint
        if ep not in self._queues:
            self._queues[ep] = EndpointQueue()
            self._active[ep] = {}

        # Ensure agent has a budget
        agent_cfg = self._config.agent_config(req.agent_id)
        self._budgets.get_or_create(
            req.agent_id,
            weight=agent_cfg.weight,
            max_balance=agent_cfg.max_balance_ss,
            now=req.enqueued_at,
        )

        # Estimate cost
        input_tokens = estimate_input_tokens(req.payload)
        req.est_input_tokens = input_tokens  # cache for the live in-flight view (context size)
        max_output = req.payload.get("max_tokens", 256)
        occupancy = len(self._active.get(ep, {}))
        req.estimated_cost_ss = self._cost.estimate_cost(
            ep, input_tokens, max_output, req.call_site, occupancy,
        )

        self._queues[ep].enqueue(req)

    def tick(self, now: float) -> list[DispatchDecision]:
        """Run one scheduling pass.  Returns dispatch decisions for all
        requests that should be sent to backends right now.

        Call this on every enqueue and every completion, or on a
        periodic timer (e.g. every 10ms).
        """
        # 1. Replenish DRR budgets
        self._budgets.replenish(now)

        # 2. Expire timed-out queued requests
        for ep, eq in self._queues.items():
            expired = eq.expire_before(now)
            for req in expired:
                self._total_timeouts += 1
                if self.on_timeout:
                    self.on_timeout(req)

        # 3. Dispatch
        decisions: list[DispatchDecision] = []
        for ep_name, ep_cfg in self._config.endpoints.items():
            ep_decisions = self._dispatch_endpoint(ep_name, ep_cfg, now)
            decisions.extend(ep_decisions)

        return decisions

    def complete(self, record: CompletionRecord, now: float) -> None:
        """Report that a dispatched request has finished (success or failure)."""
        # Find and remove from active. Count the completion only on a real hit
        # so a double-complete (e.g. a late backend return for an already-
        # completed request) doesn't inflate the counter or touch a slot.
        for ep, active_map in self._active.items():
            if record.request_id in active_map:
                self._total_completed += 1
                active_req = active_map.pop(record.request_id)

                # Retroactive cost adjustment
                actual_cost = record.duration_s
                estimated = active_req.request.estimated_cost_ss
                self._budgets.retroactive_adjust(
                    active_req.agent_id, estimated, actual_cost,
                )

                # Calibrate cost model
                if record.success:
                    self._cost.record_completion(
                        ep,
                        active_req.request.call_site,
                        record.input_tokens,
                        record.output_tokens,
                        record.duration_s,
                        record.occupancy_during,
                    )
                break

    def cancel(self, request_id: str) -> bool:
        """Remove a request from the queue (not yet dispatched).
        Returns True if found and removed."""
        for ep_q in self._queues.values():
            for band in PriorityBand:
                for agent_id in list(ep_q._queues[band].keys()):
                    q = ep_q._queues[band][agent_id]
                    for i, req in enumerate(q):
                        if req.request_id == request_id:
                            del q[i]
                            if not q:
                                del ep_q._queues[band][agent_id]
                                try:
                                    ep_q._robin[band].remove(agent_id)
                                except ValueError:
                                    pass
                            return True
        return False

    def active_count(self, endpoint: str) -> int:
        return len(self._active.get(endpoint, {}))

    def queue_depth(self, endpoint: str) -> int:
        eq = self._queues.get(endpoint)
        return eq.total_depth() if eq else 0

    def stats(self) -> dict:
        return {
            "total_dispatched": self._total_dispatched,
            "total_timeouts": self._total_timeouts,
            "total_completed": self._total_completed,
        }

    def endpoint_snapshot(self, endpoint: str) -> dict:
        eq = self._queues.get(endpoint)
        ep_cfg = self._config.endpoints.get(endpoint)
        active = self._active.get(endpoint, {})
        return {
            "max_slots": ep_cfg.max_slots if ep_cfg else 0,
            "in_flight": len(active),
            "queued": eq.total_depth() if eq else 0,
            "queue_by_band": {
                b.name.lower(): eq.band_depth(b) if eq else 0
                for b in PriorityBand
            },
        }

    def inflight_snapshot(self, now: float) -> dict:
        """Live snapshot of every in-flight (dispatched, not-yet-completed)
        request plus per-endpoint occupancy/queue state. Pure in-memory read —
        safe to call on a fast cadence (powers GET /v1/inflight + the SSE
        `inflight` frame). `now` is a monotonic timestamp (matches
        ActiveRequest.dispatched_at) for the elapsed computation."""
        requests: list[dict] = []
        for ep, active_map in self._active.items():
            for ar in active_map.values():
                req = ar.request
                requests.append({
                    "request_id": req.request_id,
                    "endpoint": ep,
                    "agent": req.agent_id,
                    "call_site": req.call_site,
                    "priority": req.priority.name,
                    "band": req.band.name.lower(),
                    "input_tokens": req.est_input_tokens,
                    "elapsed_s": round(max(0.0, now - ar.dispatched_at), 2),
                    "estimated_remaining_s": round(ar.estimated_remaining_s, 2),
                })
        requests.sort(key=lambda r: r["elapsed_s"], reverse=True)
        return {
            "requests": requests,
            "per_endpoint": {
                ep: self.endpoint_snapshot(ep) for ep in self._config.endpoints
            },
        }

    # ----- internal dispatch logic -----

    def _dispatch_endpoint(
        self,
        ep_name: str,
        ep_cfg: EndpointConfig,
        now: float,
    ) -> list[DispatchDecision]:
        eq = self._queues.get(ep_name)
        if not eq:
            return []

        # Circuit breaker: skip a backend the service has marked unhealthy so its
        # queued work defers (and expires to a deferrable timeout) rather than
        # dispatching into a black hole. Interactive/foreground are fast-failed
        # at submit time + on the unhealthy transition; this defers background.
        if self.is_endpoint_healthy is not None and not self.is_endpoint_healthy(ep_name):
            return []

        active = self._active.get(ep_name, {})
        in_flight = len(active)
        decisions: list[DispatchDecision] = []

        for band in PriorityBand:
            if eq.band_depth(band) == 0:
                continue

            available = self._available_for_band(
                ep_cfg, band, in_flight + len(decisions), eq,
            )

            dispatched_in_band = 0
            max_attempts = eq.band_depth(band) + len(eq.agents_in_band(band))
            attempts = 0

            while available > 0 and eq.band_depth(band) > 0 and attempts < max_attempts:
                attempts += 1
                candidates = eq.agents_in_band(band)
                if not candidates:
                    break

                # Head-of-queue wait per candidate so the picker can detect
                # genuine denial of service (a light consumer locked behind a
                # heavy producer) rather than keying starvation off the heavy
                # producer's perpetually-negative balance.
                wait_by_agent: dict[str, float] = {}
                for cand in candidates:
                    head = eq.peek(band, cand)
                    if head is not None:
                        wait_by_agent[cand] = now - head.enqueued_at

                agent_id = self._budgets.pick_agent(candidates, now, wait_by_agent)
                if agent_id is None:
                    break

                req = eq.peek(band, agent_id)
                if req is None:
                    eq.rotate_robin(band)
                    continue

                total_occ = in_flight + len(decisions)
                if not self._should_admit(ep_name, ep_cfg, req, total_occ, now):
                    break

                # Dispatch
                req = eq.dequeue(band, agent_id)
                if req is None:
                    continue

                self._budgets.charge(agent_id, req.estimated_cost_ss, now)

                active_req = ActiveRequest(
                    request=req,
                    dispatched_at=now,
                    estimated_remaining_s=req.estimated_cost_ss,
                )
                if ep_name not in self._active:
                    self._active[ep_name] = {}
                self._active[ep_name][req.request_id] = active_req

                decision = DispatchDecision(
                    request=req,
                    queue_wait_ms=(now - req.enqueued_at) * 1000,
                    occupancy_at_dispatch=total_occ,
                )
                decisions.append(decision)
                self._total_dispatched += 1

                if self.on_dispatch:
                    self.on_dispatch(decision)

                dispatched_in_band += 1
                available -= 1
                eq.rotate_robin(band)

        return decisions

    def _available_for_band(
        self,
        ep_cfg: EndpointConfig,
        band: PriorityBand,
        current_in_flight: int,
        eq: EndpointQueue,
    ) -> int:
        """How many slots are available for requests in this band."""
        # effective_max_slots applies any per-endpoint concurrency cap (G1:
        # companion runs at 3 of 4 physical slots for crash-headroom).
        total_free = ep_cfg.effective_max_slots - current_in_flight
        if total_free <= 0:
            return 0

        if band == PriorityBand.BACKGROUND:
            # Cap is background_cap_slots (>= floor): background may burst onto
            # idle slots above its reserved floor. The interactive reservation
            # below still keys off background_floor_slots, so raising the cap
            # never reduces interactive headroom.
            bg_cap = ep_cfg.background_cap_slots
            bg_in_flight = sum(
                1 for ar in self._active.get(ep_cfg.endpoint_class, {}).values()
                if ar.request.band == PriorityBand.BACKGROUND
            )
            return min(total_free, max(0, bg_cap - bg_in_flight))

        # For INTERACTIVE/FOREGROUND: reserve bg floor only if bg has queued work
        bg_queued = eq.band_depth(PriorityBand.BACKGROUND)
        if bg_queued > 0:
            bg_floor = ep_cfg.background_floor_slots
            # Never let the background reservation drop interactive's ceiling
            # below 1 when a slot is free — otherwise queued background work
            # blocks interactive entirely on a 1-slot endpoint (where
            # max_slots - bg_floor == 0). No preemption is introduced: the
            # total_free<=0 early-return above still holds when the endpoint is
            # fully occupied; this only governs handing out a FREE slot, and the
            # dispatch loop processes INTERACTIVE before BACKGROUND.
            ceiling = max(1, ep_cfg.effective_max_slots - bg_floor)
            non_bg_in_flight = current_in_flight - sum(
                1 for ar in self._active.get(ep_cfg.endpoint_class, {}).values()
                if ar.request.band == PriorityBand.BACKGROUND
            )
            return max(0, ceiling - non_bg_in_flight)

        return total_free

    def _should_admit(
        self,
        ep_name: str,
        ep_cfg: EndpointConfig,
        req: QueuedRequest,
        current_occupancy: int,
        now: float,
    ) -> bool:
        """Concurrency-aware admission control.

        Currently slot-count gating only. The decode_tps degradation
        guard was removed — llama.cpp's per-slot KV isolation means
        adding a request doesn't degrade in-flight work the way shared-
        decode would, and the cost model's observed_tps math produces
        unreliable factors on prefill-dominated workloads (inflated
        tps when prefill eats most of the duration).
        """
        if ep_cfg.max_slots <= 0:
            return False
        # effective_max_slots applies any per-endpoint concurrency cap (G1).
        if current_occupancy >= ep_cfg.effective_max_slots:
            return False
        return True
