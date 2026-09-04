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
from enum import Enum
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
from .cost_model import CostModel, context_fit, estimate_input_tokens


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
    #: 🚨 The ``agent_id`` the CALLER asked to be billed as, as sent — kept
    #: beside the resolved ``agent_id`` above rather than replacing it, because
    #: the two differing is exactly what has to be disclosed. A key with no
    #: delegation grant IGNORES a declared name (docs/api.md §1.5 rule 3), which
    #: is right for upgrade compatibility and would be a silencer if it were
    #: also invisible: the caller asks to be `chat-agent`, is billed as the credential,
    #: and gets a 200 either way. Empty means the caller declared nothing.
    declared_agent_id: str = ""
    #: 🚨 The credential that asserted ``agent_id``, set ONLY when a delegation
    #: grant was exercised — empty means the credential IS the ``agent_id``.
    #: Recorded so "which key ran up this caller's bill" has an answer; before
    #: delegation the question could not arise.
    asserted_by_key_id: str = ""
    timeout_s: float = 180.0
    stream: bool = False
    # context_per_slot observed at ADMISSION (0 = not captured, e.g. a
    # WAL-recovered request). The poller mutates the endpoint's live value, so
    # give-up-time reporting must use the denominator the gate actually saw
    # (audit 2026-07-02).
    ctx_per_slot_at_admission: int = 0
    # Correction.apply_json_object_guard removed a bare `response_format`
    # json_object from ``payload`` (whitespace-banned backend — see that method).
    # The payload no longer LOOKS structured, but the caller still expects JSON,
    # so the response-side gates keyed off the payload (truncation integrity,
    # the schema/JSON backstop, the structured-stream validity guard) must keep
    # treating it as such. Without this the strip would silently drop those
    # guarantees and hand a truncated half-object back to a caller that parses it.
    json_object_stripped: bool = False
    # True when ``timeout_deadline`` is a deadline the PROXY chose (the
    # default/adaptive resolve_default_timeout path), False when the CALLER
    # supplied one explicitly (body ``timeout_s`` / ``X-Timeout-S``).
    #
    # The distinction is a contract boundary, not a detail: a caller deadline is
    # a promise we keep to the letter, while a deadline we invented for a caller
    # that expressed no opinion is only ever a BUDGET. The streaming path (and
    # ONLY the streaming path) treats the latter as soft, extending it while the
    # stream demonstrably emits tokens. Everything else that reads
    # ``timeout_deadline`` — admission, the retry budget, the sync dispatch
    # bound — is unchanged and unaware of this flag.
    #
    # Defaults False so every construction site that does not opt in keeps
    # today's hard-wall behaviour. That deliberately includes WAL-recovered
    # requests: a recovered STREAM has no SSE consumer to receive it anyway (it
    # records a 'cancelled' completion immediately), so persisting this across a
    # restart would buy nothing and add a migration for no reader.
    deadline_is_default: bool = False
    # § 9 tier3 failover: the endpoint class this request was ORIGINALLY
    # submitted for, set only when it was rerouted while that endpoint was
    # unhealthy. ``endpoint`` above is always the class actually serving it, so
    # every existing consumer — queueing, occupancy, cost, dispatch, the
    # in-flight view, the served-model reported to the caller — stays correct
    # without knowing this field exists. It carries the ORIGIN, so the drain
    # check can count the degraded cohort and the response can be labelled.
    #
    # Never persisted/recovered from the WAL: a recovered request has already
    # lost its caller, and re-deriving degraded state for it would only
    # complicate the drain.
    degraded_from: str | None = None
    # Workstream D spill: the endpoint class this request was submitted for,
    # set only when it was moved to remote capacity because the local one was
    # FULL. Deliberately a SECOND field beside ``degraded_from`` and not a reuse
    # of it — a request can be both (spilled off a tier that was itself already
    # a failover target), and the two answer different questions for whoever
    # reads the completion row: "was this served by a worse model" and "did this
    # cost money". Collapsing them would make each unanswerable.
    #
    # Never persisted/recovered from the WAL, same as ``degraded_from`` and for
    # the same reason: a recovered request has lost its caller.
    spilled_from: str | None = None
    # --- Workstream C: what the caller asked for, and what we chose ---------
    #: The caller's OWN WORDS — an intent profile (``"reasoning"``), a pinned
    #: endpoint name, or the model string an OpenAI client sent. Echoed back as
    #: ``attribution.requested`` and never used for routing: it is what a caller
    #: needs to recognise its own request in the answer, and an alias it used is
    #: more recognisable to it than the class that alias resolved to.
    requested: str = ""
    #: The endpoint intent resolution CHOSE, before any substitution. ``endpoint``
    #: above always names the class actually serving, so the two differ exactly
    #: when the request was moved by failover or by spill — which is the whole
    #: definition of ``attribution.substituted``.
    #:
    #: 🚨 An explicit field rather than ``degraded_from or spilled_from or
    #: endpoint``. That expression is *correct* today and would stop being so the
    #: first time a third mover is added, and re-deriving it at each of the three
    #: sites that disclose substitution is the same "three answers to one
    #: question" this repo refuses elsewhere.
    routed_to: str = ""
    #: Per-request NARROWING of the two substitution opt-ins. ``None`` = defer to
    #: the agent's config, which is the only thing that can GRANT either.
    #:
    #: 🚨 Narrowing only, never widening, and the enforcement is an ``and``
    #: rather than an ``or`` at both reader sites. `degrade_ok` and `spill_ok`
    #: are the operator's answers to "may this caller be served a worse answer"
    #: and "may we spend money on this caller's behalf"; a request that could
    #: set either to True would let a caller grant itself a permission its
    #: operator withheld, which is the self-asserted `agent_id` bug in a
    #: different costume. A caller declining a permission it *was* granted is
    #: always safe, and is exactly what a caller sending one confidential prompt
    #: on an otherwise spill-happy identity needs.
    allow_degrade: bool | None = None
    allow_spill: bool | None = None

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
        deadline_is_default: bool = False,
        declared_agent_id: str = "",
        asserted_by_key_id: str = "",
        requested: str = "",
        allow_degrade: bool | None = None,
        allow_spill: bool | None = None,
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
            declared_agent_id=declared_agent_id,
            asserted_by_key_id=asserted_by_key_id,
            stream=bool(payload.get("stream")),
            deadline_is_default=deadline_is_default,
            # Falls back to the endpoint the caller named, so a caller that
            # pinned one gets its own spelling back rather than "" — and the
            # OpenAI doors, which have no intent vocabulary, still disclose
            # something truthful.
            requested=requested or str(endpoint),
            # Set HERE, from the endpoint resolution has already settled, so
            # every construction path records it and none can forget.
            routed_to=ep,
            allow_degrade=allow_degrade,
            allow_spill=allow_spill,
        )


def _fits_context(req: "QueuedRequest", ctx_limit: int) -> bool:
    """Whether ``req`` fits in ``ctx_limit`` tokens of context.

    The spill gate's view of ``cost_model.context_fit`` — literally the same
    predicate, denominator and estimator as the admission gate
    (``lifecycle.handle_submit``), the failover gate (``failover.plan``) and
    the recovery tally, because it IS that function. This docstring used to
    claim the sameness and four hand-written copies used to have to keep the
    claim true; one of them had already drifted.

    The spill gate wants only the boolean: a target that cannot fit the request
    cannot serve it, and the answer is a DEFER carrying no message of its own.
    """
    return context_fit(req.payload, req.payload_type, ctx_limit).fits


class Admission(Enum):
    """What to do with one request at the head of its band. THE decision.

    🚨 Three outcomes from ONE call, which is the point (``docs/roadmap.md``,
    "Remote capacity is overflow, not a parallel universe"). Spill is not a
    second scheduler with its own rules that runs when the first one gives up —
    it is a third answer to the same question, decided at the same moment, for
    the same request, by the same code. The alternative shape (a local pass, then
    an overflow pass) is how remote capacity quietly becomes a parallel system
    with its own fairness, its own admission and its own bugs.
    """

    #: A local slot is available now.
    DISPATCH = "dispatch"
    #: Local is full, but this request may be served by paid remote capacity.
    SPILL = "spill"
    #: Neither. Stay queued — which is a perfectly good answer and the one the
    #: local-first design expects most of the time.
    DEFER = "defer"


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
        self._total_spilled: int = 0

        # Callbacks (set by the proxy service)
        self.on_dispatch: Callable[[DispatchDecision], None] | None = None
        self.on_timeout: Callable[[QueuedRequest], None] | None = None
        # Circuit breaker: the service owns per-endpoint health (probe-based);
        # when set and an endpoint is unhealthy, the scheduler stops dispatching
        # to it so its queue defers instead of feeding a dead backend.
        self.is_endpoint_healthy: Callable[[str], bool] | None = None
        # Workstream D: "is this caller inside its spend threshold right now?"
        # The ONLY thing about spill this pure-computation module cannot answer
        # itself — `spill_ok` is config it already holds, capacity it can see,
        # and health it already asks about, but money lives in `spend.py` and is
        # updated by the completion path.
        #
        # 🚨 ``None`` means UNCONSTRAINED, not forbidden, and that asymmetry is
        # deliberate. Authorisation to spend is `spill_ok` — an operator saying
        # yes about a caller. The ledger is a THRESHOLD, and a threshold that is
        # not wired up removes nothing; making its absence a refusal would turn
        # "the accounting is not plumbed in" into "this configured feature
        # silently does nothing", which is the failure mode this codebase keeps
        # writing tests about.
        self.may_spend: Callable[[str], bool] | None = None
        # Fired after a request is moved to remote capacity — the service uses
        # it to log and to record the reroute, exactly as failover does.
        self.on_spill: Callable[[QueuedRequest, str, str], None] | None = None

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

    def queued_requests(
        self, endpoint: str, bands: tuple[PriorityBand, ...],
    ) -> list[QueuedRequest]:
        """Enumerate (read-only) queued requests for an endpoint across the
        given priority bands, in band → agent → FIFO order.

        Public accessor for the health layer's fast-fail on the unhealthy
        transition (release queued INTERACTIVE/FOREGROUND requests without a
        reach-in to per-band/per-agent queue internals). The full list is
        snapshotted before return so the caller may cancel/expire during
        iteration without mutating the structure mid-walk.
        """
        eq = self._queues.get(endpoint)
        if not eq:
            return []
        out: list[QueuedRequest] = []
        for band in bands:
            for agent_id in list(eq._queues.get(band, {}).keys()):
                out.extend(list(eq._queues[band][agent_id]))
        return out

    def active_count(self, endpoint: str) -> int:
        return len(self._active.get(endpoint, {}))

    def degraded_inflight(self, source_endpoint: str) -> int:
        """How many requests originally bound for ``source_endpoint`` are still
        queued or in flight on its failover target (§ 9.7's drain condition).

        Derived by walking the live structures rather than kept as a counter on
        purpose: an increment/decrement pair has to be maintained across every
        completion, timeout, cancel and error path, and the one that gets missed
        leaves the drain permanently non-zero — a recovery that never happens
        and reports nothing wrong.
        """
        n = 0
        for active_map in self._active.values():
            n += sum(1 for ar in active_map.values()
                     if ar.request.degraded_from == source_endpoint)
        for eq in self._queues.values():
            for by_agent in eq._queues.values():
                for dq in by_agent.values():
                    n += sum(1 for r in dq if r.degraded_from == source_endpoint)
        return n

    def queue_depth(self, endpoint: str) -> int:
        eq = self._queues.get(endpoint)
        return eq.total_depth() if eq else 0

    def stats(self) -> dict:
        return {
            "total_dispatched": self._total_dispatched,
            "total_timeouts": self._total_timeouts,
            "total_completed": self._total_completed,
            "total_spilled": self._total_spilled,
        }

    def agent_snapshot(self, agent_id: str) -> dict:
        """Queued and in-flight counts for ONE caller, across the whole fleet.

        A read, not a mutation — the management plane's per-caller view
        (roadmap E) asks "what is this caller doing right now", and the existing
        snapshots are all per-ENDPOINT, which is the same population sliced the
        other way. Walks the queues rather than keeping a counter: a counter
        would be a second piece of scheduler state to keep in step with the
        queues themselves, for a surface an operator reads a few times a day.
        """
        queued = 0
        for eq in self._queues.values():
            for band in PriorityBand:
                queued += eq.agent_depth(band, agent_id)
        in_flight = sum(
            1
            for active in self._active.values()
            for req in active.values()
            if req.agent_id == agent_id
        )
        return {"queued": queued, "in_flight": in_flight}

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
            ep_cfg = self._config.endpoints.get(ep)
            # Where this request actually went. `host:port` for a local
            # engine — the shape the dashboard has always shown — and the base
            # URL for a remote provider, which has no host:port to show. Left
            # None only when we genuinely do not know, which stays distinct
            # from an endpoint that has an address of a shape nobody expected.
            backend = None
            if ep_cfg is not None and (ep_cfg.host or ep_cfg.base_url):
                backend = (f"{ep_cfg.host}:{ep_cfg.port}" if ep_cfg.host
                           else ep_cfg.backend_url)
            served_model = ep_cfg.effective_model_id if ep_cfg else None
            for ar in active_map.values():
                req = ar.request
                requests.append({
                    "request_id": req.request_id,
                    # `endpoint` = the role/endpoint-class (e.g. "thinker").
                    # Surfaced in the UI as "Model". `backend` is the actual
                    # host:port doing the processing (UI "Endpoint").
                    "endpoint": ep,
                    "served_model": served_model,
                    "backend": backend,
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

            # 🚨 The old loop guard was `while available > 0 and ...`, so a full
            # endpoint never entered the body at all. It still does not, unless
            # this endpoint declares somewhere to spill to: with no `spill_to`
            # the two loops are identical instruction for instruction, which is
            # what keeps the shipped catalog's behaviour untouched by all of
            # this. The local-capacity test itself has moved INTO `_admit`,
            # because a request that local capacity cannot take is exactly the
            # request spill has an opinion about — asking "is there a slot" in
            # the loop guard is what would have made spill a second pass.
            spill_target = self._spill_target(ep_cfg)
            if available <= 0 and not spill_target:
                continue

            dispatched_in_band = 0
            max_attempts = eq.band_depth(band) + len(eq.agents_in_band(band))
            attempts = 0

            while eq.band_depth(band) > 0 and attempts < max_attempts:
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
                verdict = self._admit(
                    ep_name, ep_cfg, req, total_occ, available, spill_target,
                )
                if verdict is Admission.DEFER:
                    break
                if verdict is Admission.SPILL:
                    spilled = eq.dequeue(band, agent_id)
                    if spilled is not None:
                        self._spill(spilled, spill_target)
                    # Rotate so the next attempt considers a different agent:
                    # without this one caller with a deep queue spills its whole
                    # backlog before anyone else is looked at, which is the
                    # round-robin failing in the one place it costs money.
                    eq.rotate_robin(band)
                    continue

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

    def _spill_target(self, ep_cfg: EndpointConfig) -> str:
        """The endpoint class ``ep_cfg`` may spill to right now, or ``""``.

        Resolved here rather than read straight off the config so ONE definition
        of "a usable spill target" serves both the loop guard and ``_admit`` —
        the same lesson ``failover.pairs()`` records, where two places deciding
        the same thing separately is how one of them ends up wrong.

        A target that is unhealthy is not a target. Spill exists to answer local
        SCARCITY, and queueing into a second dead backend answers nothing —
        deferring is strictly better, because the local slot the request is
        waiting for will actually open.
        """
        target = ep_cfg.spill_to
        if not target or target not in self._config.endpoints:
            return ""
        if target == ep_cfg.endpoint_class:
            # An endpoint spilling to itself would re-enqueue the request onto
            # the very queue being drained, forever. The catalog parser already
            # refuses to resolve this, but the parser is not the only way an
            # EndpointConfig gets built.
            return ""
        if self.is_endpoint_healthy is not None and not self.is_endpoint_healthy(target):
            return ""
        return target

    def _admit(
        self,
        ep_name: str,
        ep_cfg: EndpointConfig,
        req: QueuedRequest,
        current_occupancy: int,
        band_available: int,
        spill_target: str,
    ) -> Admission:
        """THE admission decision: dispatch locally, spill, or defer.

        Concurrency-aware, and now money-aware — but only in the direction that
        can never cost a caller a local slot. Read the order of the tests below
        as the doctrine it is:

        **Local capacity is tried first, always, for everybody.** A caller over
        its spend threshold, a caller with no `spill_ok`, a caller nobody has
        ever configured — all of them reach the same DISPATCH on a free local
        slot as anyone else. Nothing about money appears above this line, and
        nothing should: admission control is about capacity, and a budget that
        could leave a free slot idle while a request waits serves nobody
        (``spend.py``, ``docs/roadmap.md``).

        **Spill is considered only once local has said no.** That is what makes
        remote capacity overflow rather than a parallel universe: it is never the
        first answer, so the local fleet is never bypassed while it has room, and
        a deployment with no remote provider configured never executes a line of
        it.

        The slot-count gate itself is unchanged. The decode_tps degradation guard
        was removed long ago — llama.cpp's per-slot KV isolation means adding a
        request does not degrade in-flight work the way shared decode would, and
        the cost model's observed_tps math produces unreliable factors on
        prefill-dominated workloads (inflated tps when prefill eats most of the
        duration).
        """
        # --- local, for everyone -------------------------------------------
        if ep_cfg.max_slots <= 0:
            # An endpoint with no slots at all is misconfigured, not busy, and
            # the difference matters here: "busy" is what spill answers, and
            # spilling a config error would quietly convert it into an invoice.
            # Defer, exactly as this returned False before Workstream D.
            return Admission.DEFER
        if band_available > 0:
            # effective_max_slots applies any per-endpoint concurrency cap (G1).
            if current_occupancy < ep_cfg.effective_max_slots:
                return Admission.DISPATCH

        # --- overflow, for callers who opted in and are inside their cap ----
        if not spill_target:
            return Admission.DEFER
        # 🚨 Spill NEVER CHAINS, the same rule failover follows and for a sharper
        # reason. Two endpoints whose configs point at each other would otherwise
        # hand one request back and forth a hop per tick, forever, overwriting
        # `spilled_from` each time so nothing downstream could even see it
        # happening. A single hop is also the honest semantics: the question was
        # "local is full, may somebody else answer", and it has been answered.
        if req.spilled_from:
            return Admission.DEFER
        # Default-deny, and a stronger default-deny than `degrade_ok`: this one
        # spends money and sends the prompt off the machine. See
        # AgentQuotaConfig.spill_ok — AND, since Workstream C, the request's own
        # narrowing of it. 🚨 An `and`, never an `or`: the operator GRANTS the
        # permission and the caller may only decline it. See
        # ``QueuedRequest.allow_spill`` for why the reverse would be the
        # self-asserted-identity bug wearing a different hat.
        if not self._config.agent_config(req.agent_id).spill_ok:
            return Admission.DEFER
        if req.allow_spill is False:
            # The caller opted this ONE request out — a confidential prompt on
            # an identity that is otherwise happy to spill. It is not an error
            # and not a refusal: the request simply waits for local capacity,
            # which is the same DEFER every un-opted-in caller already gets.
            return Admission.DEFER
        if self.may_spend is not None and not self.may_spend(req.agent_id):
            # Over threshold. 🚨 DEFER, never a refusal — the request keeps its
            # place in the local queue and will be served by the local endpoint
            # like everything else. Losing spill is the whole penalty.
            return Admission.DEFER
        # Physics, not policy, and the same predicate failover uses: a target
        # that cannot fit the request cannot serve it, and truncating to make it
        # fit would answer a different question than the one that was asked.
        tgt_cfg = self._config.endpoints[spill_target]
        if not _fits_context(req, tgt_cfg.context_per_slot):
            return Admission.DEFER
        # Do not spill into a remote endpoint that is already at its own
        # config-seeded concurrency cap. That cap is a policy knob we chose
        # rather than a discovered capacity (CLAUDE.md), which makes it the only
        # bound on how fast a full local tier can turn into an invoice.
        if len(self._active.get(spill_target, {})) >= tgt_cfg.effective_max_slots:
            return Admission.DEFER
        return Admission.SPILL

    def _spill(self, req: QueuedRequest, target: str) -> None:
        """Move ``req`` onto ``target``'s queue, in place.

        Modelled on ``failover.apply`` and for the same reason: re-pointing
        ``req.endpoint`` is the whole change, and everything downstream — queue,
        occupancy, cost model, dispatch, the backend lookup, the served model
        reported to the caller — re-resolves against the new class on its own.

        Two things it deliberately does NOT do:

        * **Re-derive the deadline.** It was computed from the SOURCE endpoint's
          floors, and re-deriving would silently move a deadline a caller may
          have supplied explicitly. Same rule as failover.
        * **Charge anything.** DRR is charged at DISPATCH, not at enqueue, so a
          spilled request has not been charged yet and must not be — it is about
          to be charged against the target it actually occupies. Re-enqueueing
          through ``enqueue()`` re-estimates its slot-second cost for the target,
          which matters because a remote endpoint's cost model is its own.
        """
        src = req.endpoint
        req.spilled_from = src
        req.endpoint = target
        req.ctx_per_slot_at_admission = self._config.endpoints[target].context_per_slot
        self._total_spilled += 1
        self.enqueue(req)
        if self.on_spill:
            self.on_spill(req, src, target)
