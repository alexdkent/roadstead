"""Failover — degraded routing from a sick endpoint to its declared fallback.

An endpoint declaring ``failover_to:`` degrades to that endpoint class while it
is unhealthy, for the agents that have opted in. In the example catalog that is
``tier3 -> tier2``: the heavy tier falls back to the mid one rather than to
nothing. Failover never CHAINS — the target may not declare one of its own, or
an outage would walk the fleet.

Three things this module deliberately does NOT do, each because the proxy
already owns it and a second copy would be worse than none:

* **Detection.** ``health.Health.endpoint_healthy()`` is the whole signal —
  consecutive-probe circuit, rate-windowed cooldown, operator drain, and
  recovery hysteresis. Note the consequence: **an operator DRAIN reads
  unhealthy**, so failover fires during planned maintenance too. That is the
  single biggest operational win here (a ~10-minute tier3 cold start becomes a
  ~10-minute degradation, not an outage) and it is intended — the Phase 2
  cutover runbook has to say so.
* **Contention.** Rerouted work is enqueued on the TARGET's queue and sorted by
  the same DRR scheduler as everything else: priority bands, per-agent weight,
  ``max_balance_ss``, the background floor, the shed depth and the
  head-of-queue starvation rescue. Backlog on the target under a tier3 outage
  is the scheduler WORKING, not a failure. 🚨 Do not add a failover-specific
  reservation or admission cap here later — the instinct recurs, and it was
  explicitly refused by the operator (2026-08-19).
* **Retry.** A refusal is a clean labelled 503 with ``Retry-After``, using the
  existing deferrable-error vocabulary the fleet's clients already classify.

What it DOES own: the two admission gates, the degraded-mode state machine
(enter / drain / dwell / leave), and the visibility surface.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import normalize_endpoint
from .cost_model import context_fit

if TYPE_CHECKING:
    from .health import Health
    from .scheduler import QueuedRequest
    from .state import ProxyState

logger = logging.getLogger(__name__)


# Refusal codes. These are the machine-readable taxonomy field on the error
# envelope, alongside the existing `circuit_open` / `draining` / `backpressure`.
CODE_NOT_OPTED_IN = "degraded_not_opted_in"
CODE_CONTEXT_OVERFLOW = "degraded_context_overflow"


@dataclass(frozen=True)
class FailoverPlan:
    """Outcome of considering one request for degraded routing."""
    target: str | None = None        # endpoint class to reroute to
    refusal_code: str | None = None  # set when the request may NOT be rerouted
    refusal_detail: str = ""         # a CLAUSE appended to the existing error

    @property
    def rerouted(self) -> bool:
        return self.target is not None


class Failover:
    """Degraded-routing policy over the shared ProxyState."""

    def __init__(self, state: "ProxyState", health: "Health") -> None:
        self.state = state
        self.health = health

    # ---------------------------------------------------------------- state

    def pairs(self) -> dict[str, str]:
        """source endpoint class → target endpoint class, for every endpoint
        that declares a resolvable, USABLE ``failover_to``.

        🚨 An ON-DEMAND endpoint can never be a failover target, and the reason
        is an ordering one rather than a policy one. ``handle_submit`` runs
        ``on_demand.ensure_loaded()`` for the endpoint the request arrived for,
        and the reroute happens AFTER that — at the circuit-breaker branch — so
        a rerouted request would be dispatched at a model that was never asked
        to load. Nothing is on-demand today (the comment in lifecycle.py naming
        `creative` as an example is itself stale), which is exactly why this is
        worth pinning now: the day someone makes the target on-demand, the
        failure would be a dispatch into an unloaded backend, not an error.
        Pinned by test_failover.py::test_failover_target_is_never_on_demand.
        """
        return {
            ep: cfg.failover_to
            for ep, cfg in self.state.config.endpoints.items()
            if cfg.failover_to
            and cfg.failover_to in self.state.config.endpoints
            and not self.state.on_demand.manages(cfg.failover_to)
        }

    def is_degraded(self, endpoint: str) -> bool:
        return normalize_endpoint(endpoint) in self.state.degraded_endpoints

    def refresh(self, now: float | None = None) -> None:
        """Drive the degraded-mode state machine. Idempotent; safe to call on
        every poller tick and on the submit path.

        ENTER when the source is unhealthy AND the target is healthy. Requiring
        a healthy target is not belt-and-braces: entering degraded mode while
        the target is also down would raise a chip announcing a degradation
        that cannot happen, and every request would 503 anyway.

        LEAVE only when ALL THREE hold (§ 9.7): the source is healthy again per
        the existing hysteresis, the degraded cohort has drained to zero, and
        the minimum dwell has elapsed. The flip then happens at a request
        boundary by construction — nothing in flight is moved, so no single
        conversation spans two models.
        """
        now = time.monotonic() if now is None else now
        for src, tgt in self.pairs().items():
            src_healthy = self.health.endpoint_healthy(src)
            if src in self.state.degraded_endpoints:
                if not src_healthy:
                    continue
                dwell = self.state.config.endpoints[src].failover_dwell_s
                since = self.state.degraded_since.get(src, now)
                inflight = self.state.scheduler.degraded_inflight(src)
                if inflight:
                    continue
                if now - since < dwell:
                    continue
                self.state.degraded_endpoints.discard(src)
                self.state.degraded_since.pop(src, None)
                logger.warning(
                    "LLMPROXY_FAILOVER_LEAVE endpoint=%s target=%s "
                    "degraded_for_s=%.0f rerouted=%d refused=%d — recovered, "
                    "drained and past dwell; next request goes to %s",
                    src, tgt, now - since,
                    self.state.degraded_rerouted.get(src, 0),
                    sum(self.state.degraded_refused.get(src, {}).values()),
                    src,
                )
            else:
                if src_healthy or not self.health.endpoint_healthy(tgt):
                    continue
                self.state.degraded_endpoints.add(src)
                self.state.degraded_since[src] = now
                logger.critical(
                    "LLMPROXY_FAILOVER_ENTER endpoint=%s target=%s — %s is "
                    "unhealthy; opted-in callers will be served by %s until it "
                    "recovers, drains and clears a %.0fs dwell",
                    src, tgt, src, tgt,
                    self.state.config.endpoints[src].failover_dwell_s,
                )

    # ---------------------------------------------------------------- gates

    def plan(self, req: "QueuedRequest") -> FailoverPlan:
        """Decide whether ``req`` may be rerouted. Called ONLY on the path
        where the request would otherwise be refused (source unhealthy), so a
        healthy fleet never executes a line of this.

        Two gates, both of which refuse rather than degrade. Order is
        deliberate: the opt-in answer is the POLICY one and is the honest
        primary reason for the great majority of the fleet.

        🚨 ``refusal_detail`` is a CLAUSE APPENDED to the caller's existing
        error, never a replacement for it. The pre-existing message already
        distinguishes an operator drain ("paused for maintenance") from an
        auto-circuit trip, which is what an operator triages on, and it carries
        the substrings the fleet's clients sniff for deferrability. A first
        draft here overwrote both — it read better and told the operator less.
        ``refusal_code`` likewise travels in its OWN response field rather than
        replacing the `circuit_open` / `draining` taxonomy code that callers
        already classify on: why the endpoint is down and why failover did not
        rescue this request are two different facts.
        """
        src = normalize_endpoint(req.endpoint)
        # Resolved through pairs(), not read straight off the EndpointConfig, so
        # this shares ONE definition of "a usable failover target" with
        # refresh() — including the on-demand exclusion above. Two places
        # deciding the same thing separately is how one of them ends up wrong.
        tgt = self.pairs().get(src, "")
        if not tgt or src not in self.state.degraded_endpoints:
            return FailoverPlan()
        if not self.health.endpoint_healthy(tgt):
            # Target went sick after we entered degraded mode. Refuse rather
            # than queue into a second dead backend.
            return FailoverPlan()

        # Gate 1 — opt-in (§ 9.5). Default-deny. This is the branch most likely
        # to rot into always-true, which is why it has its own test.
        agent_cfg = self.state.config.agent_config(req.agent_id)
        if not agent_cfg.degrade_ok:
            return FailoverPlan(
                refusal_code=CODE_NOT_OPTED_IN,
                refusal_detail=(
                    f"; it is serving from {tgt} for opted-in callers, and "
                    f"agent {req.agent_id!r} is not opted in "
                    f"(`degrade_ok` in the agents config)"),
            )
        # Workstream C: the request's own NARROWING of that opt-in. 🚨 An
        # `and`, never an `or` — the operator grants, the caller may only
        # decline. Refused with the same code, because from the caller's side
        # the outcome is identical (a clean labelled 503 rather than a smaller
        # model's answer) and a second code would ask every existing client to
        # learn a distinction it cannot act on differently.
        if req.allow_degrade is False:
            return FailoverPlan(
                refusal_code=CODE_NOT_OPTED_IN,
                refusal_detail=(
                    f"; it is serving from {tgt} for opted-in callers, and "
                    f"this request declared `substitution.degrade: false`"),
            )

        # Gate 2 — context fit (§ 9.4). Physics, not policy: the target is a
        # much smaller-context model and an oversized request CANNOT be
        # rerouted. 🚨 Never truncate to make it fit — a silently shortened
        # prompt answers a different question than the one that was asked, and
        # would pass every check downstream.
        #
        # ONE predicate, shared with the M1 gate on the normal path
        # (lifecycle.handle_submit), the spill gate and the recovery tally —
        # `cost_model.context_fit`, which is also where the marker substring
        # this message carries is defined. Unlike the M1 gate this is
        # ENFORCE-always and has no shadow mode: there the alternative to
        # refusing is a request that probably works, here it is a guaranteed
        # backend 400. 🚨 That difference lives HERE, in what we do with the
        # answer, not in a second copy of the arithmetic.
        tgt_cfg = self.state.config.endpoints[tgt]
        fit = context_fit(req.payload, req.payload_type, tgt_cfg.context_per_slot)
        if not fit.fits:
            return FailoverPlan(
                refusal_code=CODE_CONTEXT_OVERFLOW,
                refusal_detail=(
                    f"; it is serving from {tgt} for opted-in callers, but "
                    f"this request {fit.overflow_detail(tgt)}"),
            )

        return FailoverPlan(target=tgt)

    def apply(self, req: "QueuedRequest", target: str) -> None:
        """Re-point ``req`` at the failover target, in place.

        🚨 This sets ``req.endpoint`` — the routing key — and records the origin
        in ``req.degraded_from``. It does NOT rewrite any endpoint NAME mapping:
        ``normalize_endpoint()`` is untouched and stays idempotent, and the
        request is not aliased. Everything downstream (the scheduler's queue and
        occupancy, the cost model, the shed depth, the backend lookup at
        dispatch, ``apply_json_object_guard``'s whitespace flag, and the served
        model reported back to the caller) re-resolves against the target
        automatically, which is the whole reason this is one assignment rather
        than a special case threaded through the dispatcher.

        ``timeout_deadline`` is deliberately NOT re-derived. It was computed
        from the SOURCE's floors, and the source here is the slow deep tier — a
        thinker-derived deadline is generous for the target, i.e. wrong in the
        safe direction. Re-deriving it would also silently move a deadline a
        caller may have supplied explicitly, which is a contract.
        """
        src = normalize_endpoint(req.endpoint)
        req.degraded_from = src
        req.endpoint = target
        tgt_cfg = self.state.config.endpoints[target]
        # The M1 gate stamped the SOURCE's per-slot context on the request for
        # give-up reporting; the target's is what actually applies now.
        req.ctx_per_slot_at_admission = tgt_cfg.context_per_slot
        self.state.degraded_rerouted[src] = self.state.degraded_rerouted.get(src, 0) + 1
        logger.info(
            "LLMPROXY_FAILOVER_REROUTE %s -> %s agent=%s call_site=%s request_id=%s",
            src, target, req.agent_id, req.call_site, req.request_id,
        )

    # ------------------------------------------------------------ reporting

    def status(self) -> dict:
        """The /v1/status view. 🚨 A SET, mirroring ``paused_endpoints`` — NOT a
        per-endpoint bool. The per-endpoint ``paused`` bool already there is
        known-unreliable (``paused: False`` has been observed while the endpoint
        was in ``paused_endpoints``); a second bool with the same bug would just
        give the operator two things to disbelieve.
        """
        now = time.monotonic()
        return {
            "degraded_endpoints": sorted(self.state.degraded_endpoints),
            "failover_pairs": self.pairs(),
            "degraded_for_s": {
                ep: round(now - self.state.degraded_since.get(ep, now))
                for ep in sorted(self.state.degraded_endpoints)
            },
            "degraded_inflight": {
                ep: self.state.scheduler.degraded_inflight(ep)
                for ep in sorted(self.state.degraded_endpoints)
            },
            "degraded_rerouted": dict(self.state.degraded_rerouted),
            "degraded_refused": {
                ep: dict(by_code)
                for ep, by_code in self.state.degraded_refused.items()
            },
        }

    def record_refusal(self, req: "QueuedRequest", plan: FailoverPlan) -> None:
        """Count a refusal — called BY THE CALLER, once it has actually refused.

        Deliberately not done inside ``plan()``, which is pure. A BACKGROUND
        request that fails both gates is NOT refused: it falls through and
        queues on the unhealthy endpoint to defer until recovery, exactly as it
        did before failover existed. Counting it in ``plan()`` inflated
        `degraded_refused` with requests nobody turned away — and that counter
        is the operator's signal that the opt-in set is too small, so a number
        that answers a slightly different question is worse than no number.
        """
        if not plan.refusal_code:
            return
        src = req.degraded_from or normalize_endpoint(req.endpoint)
        by_code = self.state.degraded_refused.setdefault(src, {})
        by_code[plan.refusal_code] = by_code.get(plan.refusal_code, 0) + 1
