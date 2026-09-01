"""ProxyService — the main orchestrator that wires scheduler, backend,
queue, coalescing, and observability into a running service.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from .acl import IPIdentityMap
from .agent_budget import BudgetManager
from .backend import BackendClientPool, BackendError, BackendResponse, BackendTimeout, BackendUnavailable
from .coalesce import DeterministicCache
from .grammar import (
    GrammarResult,
    grammar_hash,
    normalize_and_validate,
    recover_structured_object,
    root_object_keys,
    verify_conformance,
)
from . import cache_stats
from .config import (
    CLASS_TO_ROLE,
    LLMPriority,
    PriorityBand,
    ProxyConfig,
    degeneration_guard_enabled,
    degeneration_shadow_only,
    normalize_endpoint,
    shadow_egress_detect_enabled,
    thinking_enabled,
    thinking_reasoning_budget,
)



# Phase 2.1 — bounded in-flight drain on graceful shutdown. The BOUND is the fix
# for the historic SIGTERM hang (uvicorn waiting forever behind a slow request):
# drain up to this long, then force-cancel stragglers. (Imported by __main__.)
_DRAIN_DEADLINE_S = 30.0


from .cost_model import CostModel, estimate_input_tokens
from .flags import RuntimeFlags
from .timeout_model import TimeoutModel
from .on_demand import OnDemandManager, OnDemandUnavailable
from .observability import (
    AlertCondition,
    MetricsSample,
    RequestLogRecord,
    RequestLogger,
    RollingMetrics,
    check_alerts,
)
from .queue import PersistentQueue
from .scheduler import (
    CompletionRecord,
    DispatchDecision,
    QueuedRequest,
    Scheduler,
)
from .constants import _DEFAULT_TIMEOUT_S, _PAYLOAD_KIND
from .correction import (
    Correction,
    # Re-exported so the moved symbols stay importable from `service` (frozen
    # test surface, contract §1) — used via the correction/lifecycle modules at
    # runtime, hence the noqa.
    _ToolCallStreamSanitizer,  # noqa: F401
    _EMPTY_RESCUE_MIN_TOKENS,  # noqa: F401
    _is_degenerate_text,  # noqa: F401
    _top_shingle_reps,  # noqa: F401
)
from .failover import Failover
from .health import Health
from .http_handlers import (
    ProxyHttpHandlers,
    # Re-exported so `service._to_int` / `service._to_float` stay importable
    # (frozen test surface, contract §1).
    _to_float,  # noqa: F401
    _to_int,  # noqa: F401
)
from .lifecycle import Lifecycle, _openai_error as _mk_openai_error
from .sse_hub import DROP_SENTINEL, SSEHub
from .state import ProxyState

logger = logging.getLogger(__name__)




class _StateField:
    """Data-descriptor forwarding ``ProxyService._<name>`` to
    ``self._state.<name>`` (de-monolith contract §2).

    The data now lives on ``ProxyState``; ``ProxyService`` keeps the identical
    ``self._<field>`` surface the ~530 white-box tests reach into. Read+write —
    a few counters are ``+=``'d and ``_shed_depth`` is reassigned by tests —
    and because ``__get__`` returns the *live* object, in-place mutation
    (``.append``, ``[k]=v``, ``.add``, ``|=``) works unchanged with identity
    preserved.
    """

    __slots__ = ("_attr",)

    def __init__(self, attr: str) -> None:
        self._attr = attr

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj._state, self._attr)

    def __set__(self, obj, value) -> None:
        setattr(obj._state, self._attr, value)


class ProxyService:
    """Main proxy service that coordinates all components."""

    # --- Frozen private surface (contract §2) --------------------------------
    # Every ``self._<field>`` name ProxyService has always exposed, forwarded to
    # the identically-named attribute on ``ProxyState`` (leading underscore
    # dropped). Source of truth for the field set is ``ProxyState.__init__``.
    _config = _StateField("config")
    _cost_model = _StateField("cost_model")
    _timeout_model = _StateField("timeout_model")
    _budget_mgr = _StateField("budget_mgr")
    _scheduler = _StateField("scheduler")
    _backend = _StateField("backend")
    _queue_db = _StateField("queue_db")
    _on_demand = _StateField("on_demand")
    _flags = _StateField("flags")
    _cache = _StateField("cache")
    _grammar_cache = _StateField("grammar_cache")
    _grammar_alerted = _StateField("grammar_alerted")
    _metrics = _StateField("metrics")
    _request_logger = _StateField("request_logger")
    _acl = _StateField("acl")
    _identity = _StateField("identity")
    _prices = _StateField("prices")
    _spend = _StateField("spend")
    _sse = _StateField("sse")
    _dispatch_event = _StateField("dispatch_event")
    _draining = _StateField("draining")
    _pending_futures = _StateField("pending_futures")
    _pending_streams = _StateField("pending_streams")
    _thinking_active = _StateField("thinking_active")
    _thinking_requests = _StateField("thinking_requests")
    _thinking_clean = _StateField("thinking_clean")
    _thinking_recovered = _StateField("thinking_recovered")
    _thinking_truncated = _StateField("thinking_truncated")
    _thinking_fallback = _StateField("thinking_fallback")
    _thinking_noop = _StateField("thinking_noop")
    _shadow_drop = _StateField("shadow_drop")
    _degeneration_detected = _StateField("degeneration_detected")
    _degeneration_recovered = _StateField("degeneration_recovered")
    _degeneration_unrecovered = _StateField("degeneration_unrecovered")
    _degeneration_by_call_site = _StateField("degeneration_by_call_site")
    _degen_redispatch_inflight = _StateField("degen_redispatch_inflight")
    _empty_rescue_attempts = _StateField("empty_rescue_attempts")
    _empty_rescue_recovered = _StateField("empty_rescue_recovered")
    _timed_out_ids = _StateField("timed_out_ids")
    _inflight_tasks = _StateField("inflight_tasks")
    _endpoint_health = _StateField("endpoint_health")
    _health_fail_threshold = _StateField("health_fail_threshold")
    _paused_endpoints = _StateField("paused_endpoints")
    _transient_retry_max = _StateField("transient_retry_max")
    _last_cleanup_at = _StateField("last_cleanup_at")
    _last_budget_save_at = _StateField("last_budget_save_at")
    _last_wal_checkpoint_at = _StateField("last_wal_checkpoint_at")
    _last_incr_vacuum_at = _StateField("last_incr_vacuum_at")
    _last_cache_stats_at = _StateField("last_cache_stats_at")
    _shed_depth = _StateField("shed_depth")
    _alerts = _StateField("alerts")
    _alert_logged = _StateField("alert_logged")
    _admin_ips_seen = _StateField("admin_ips_seen")
    _unknown_endpoint_submits = _StateField("unknown_endpoint_submits")
    _context_overflows = _StateField("context_overflows")
    _smart_default_shadow = _StateField("smart_default_shadow")
    _slot_leak_reclaimed = _StateField("slot_leak_reclaimed")
    _drain_straggler_cancelled = _StateField("drain_straggler_cancelled")
    _scheduler_task = _StateField("scheduler_task")
    _poller_task = _StateField("poller_task")
    _inflight_task = _StateField("inflight_task")
    _started_at = _StateField("started_at")

    def __init__(self, config: ProxyConfig) -> None:
        # All mutable data + references to the injected singletons live on
        # ProxyState (de-monolith contract §3). ProxyService exposes the frozen
        # ``self._<field>`` surface via the _StateField descriptors above; the
        # Health / Correction / Lifecycle / http_handlers collaborators receive
        # and mutate this one shared state object.
        self._state = ProxyState(config)
        # Composed collaborators (de-monolith contract §4). Extracted
        # incrementally: Health first (circuit breaker + capacity poller +
        # drain/pause). ProxyService keeps thin delegators to each collaborator's
        # methods so its frozen private surface stays intact.
        self._health = Health(self._state)
        # Failover (§ 9): degraded routing from a sick endpoint to its declared
        # `fallback:`. Built HERE rather than inside ProxyState because it
        # consumes Health's endpoint_healthy() — the detection half already
        # exists and must not be duplicated.
        self._state.failover = Failover(self._state, self._health)
        # Correction: grammar/thinking/degeneration/empty-rescue/egress guards.
        self._correction = Correction(self._state)
        # Lifecycle: admission -> dispatch -> response hot path (uses Correction
        # + Health by reference).
        self._lifecycle = Lifecycle(self._state, self._correction, self._health)
        # HTTP handlers: the Starlette request surface (uses Lifecycle + Health).
        self._http = ProxyHttpHandlers(self._state, self._lifecycle, self._health)

    # ----- lifecycle -----

    async def startup(self) -> None:
        """Initialize cost model, recover queue, start scheduler loop."""
        # Admission timeouts (queued past deadline) were previously a
        # silent drop — wire the callback so they're logged + the caller
        # is released promptly instead of waiting out its own deadline.
        self._scheduler.on_timeout = self._on_admission_timeout
        # Circuit breaker: the scheduler skips an endpoint the poller has marked
        # unhealthy so its queued work defers instead of dispatching into a dead
        # backend (Phase 1.2).
        self._scheduler.is_endpoint_healthy = self._endpoint_healthy
        # Workstream D: the money half of the admission decision. The scheduler
        # owns "is there room" and "did the operator opt this caller in"; this
        # answers "is the caller inside its spend threshold", which is the only
        # part that lives outside a pure-computation module.
        self._scheduler.may_spend = self._state.spend_may_spill
        self._scheduler.on_spill = self._on_spill

        # Register endpoints in cost model
        for ep_name, ep_cfg in self._config.endpoints.items():
            self._cost_model.register_endpoint(
                ep_name, ep_cfg.max_slots,
                prefill_k=0.0004,
            )

        # Set fleet capacity for DRR
        self._budget_mgr.set_total_capacity(self._config.total_fleet_slots)

        # Pre-register known agents
        for agent_id, acfg in self._config.agents.items():
            self._budget_mgr.get_or_create(
                agent_id,
                weight=acfg.weight,
                max_balance=acfg.max_balance_ss,
            )

        # Restore persisted DRR balances (Phase 3.4) so fairness survives a
        # restart instead of resetting to zero. Balances are clamped to the
        # agent's cap and the replenish clock rebased onto this process's monotonic.
        ghost_rows: list[str] = []
        for row in self._queue_db.load_budgets():
            # Phase 5B.4: prefer the agent's CONFIGURED weight + cap over the
            # persisted row. get_or_create only applies weight/max_balance when
            # the agent is NEW, and for a lazily-created agent (not in config)
            # the old call passed no max_balance → it defaulted to 60.0,
            # clamping the restored balance to the wrong bound + letting a stale
            # persisted weight override config intent. Reconcile explicitly.
            acfg = self._config.agents.get(row["agent_id"])
            # Ghost gate (audit 2026-07-02): the only restart-surviving state
            # that matters for fairness is DEBT. A non-configured id with a
            # non-negative balance is an idle one-off (smoke script, probe,
            # ad-hoc tool) — skip it; if it ever returns it's lazily recreated
            # at the identical idle state. Configured agents always restore.
            if acfg is None and (row["balance"] or 0.0) >= 0.0:
                ghost_rows.append(row["agent_id"])
                continue
            weight = acfg.weight if acfg else row["weight"]
            b = self._budget_mgr.get_or_create(row["agent_id"], weight=weight)
            if acfg:
                b.max_balance = acfg.max_balance_ss
                b.weight = acfg.weight
            b.balance = max(-b.max_balance, min(b.max_balance, row["balance"]))
            b.total_consumed = row["total_consumed"]
            b.last_replenish_at = time.monotonic()
        if ghost_rows:
            # One-shot DB cleanup of the skipped ghosts (steady-state eviction
            # is prune_idle + delete_budgets on the daily retention sweep).
            self._queue_db.delete_budgets(ghost_rows)
            logger.info("budget: dropped %d idle ghost row(s) at load: %s%s",
                        len(ghost_rows), ", ".join(sorted(ghost_rows)[:10]),
                        "…" if len(ghost_rows) > 10 else "")

        # Seed the durable shadow/flip-gate evidence (audit 2026-07-02): the
        # context-overflow tally and the cache-drift dedup map used to be
        # in-memory only, resetting on every ship restart — an empty
        # /v1/status shadow read as "safe to flip" while real overflows sat in
        # the (differently-windowed) timeout rows.
        try:
            self._state.context_overflows = self._queue_db.load_context_overflows()
            persisted_drift = self._queue_db.kv_get("cache_drift_alerted", [])
            self._state.cache_drift_alerted = {
                (str(e[0]), str(e[1])): float(e[2])
                for e in persisted_drift
                if isinstance(e, (list, tuple)) and len(e) == 3
            }
        except Exception:  # noqa: BLE001 — seeding is best-effort
            logger.warning("shadow-evidence seed failed", exc_info=True)

        # Recover queued requests from WAL
        recovered = self._queue_db.recover_queued(time.monotonic())
        for req in recovered:
            # Shadow-tally context overflows on the recovery path too — it
            # bypasses handle_submit, so recovered oversized requests were
            # invisible to the flip-gate evidence (audit 2026-07-02). Count
            # only; recovery never rejects (there is no caller to 422).
            try:
                if req.payload_type == "chat_completion":
                    cfg = self._config.endpoints.get(req.endpoint)
                    limit = cfg.context_per_slot if cfg else 0
                    mt = req.payload.get("max_tokens")
                    est_out = mt if isinstance(mt, int) and mt > 0 else 0
                    est_in = req.est_input_tokens or 0
                    if limit > 0 and est_in + est_out > limit:
                        caller = str(req.caller_id or req.agent_id)
                        tally = self._state.context_overflows.setdefault(
                            req.endpoint,
                            {"count": 0, "callers": {}, "max_est_in": 0})
                        tally["count"] += 1
                        tally["callers"][caller] = tally["callers"].get(caller, 0) + 1
                        tally["max_est_in"] = max(tally["max_est_in"], est_in)
                        self._queue_db.record_context_overflow(
                            req.endpoint, caller, est_in)
            except Exception:  # noqa: BLE001 — evidence must not break recovery
                logger.debug("recovery overflow tally failed", exc_info=True)
            # Same reason the overflow tally is repeated here: recovery bypasses
            # handle_submit. A WAL-recovered request still carrying a bare
            # response_format json_object would dispatch to a whitespace-banned
            # backend and come back as literally `{}`. The guard is total and
            # idempotent, so applying it again on this path is free.
            self._correction.apply_json_object_guard(req)
            self._scheduler.enqueue(req)

        # Re-derive operator drains that were open when the process died.
        # _paused_endpoints was in-memory only: a proxy restart mid-drain
        # forgot the pause, the poller probed the intentionally-down backend,
        # tripped the circuit, and endpoint_paused fired as an UNPLANNED
        # outage despite the open maintenance window saying otherwise.
        try:
            now_wall = time.time()
            for w in self._queue_db.maintenance_windows(hours=24.0):
                if w.get("ended_at") is not None or w.get("source") != "drain":
                    continue
                ep = w.get("endpoint") or ""
                if ep == "*" or ep not in self._config.endpoints:
                    continue
                age_h = (now_wall - (w.get("started_at") or now_wall)) / 3600.0
                if age_h > 24.0:
                    # Forgot-to-resume guard: a day-old open drain is almost
                    # certainly stale — don't re-park the endpoint on it.
                    logger.warning(
                        "ignoring stale open drain window for %s (%.1fh old) — "
                        "endpoint NOT re-paused; close it via "
                        "/v1/admin/endpoints/%s/resume", ep, age_h, ep)
                    continue
                self._paused_endpoints.add(ep)
                logger.warning(
                    "endpoint %s re-PAUSED from open drain window (%.1fh old, "
                    "reason: %s) — resume via /v1/admin/endpoints/%s/resume",
                    ep, age_h, w.get("reason") or "-", ep)
        except Exception:  # noqa: BLE001 — never block startup on this
            logger.warning("paused-endpoint rederivation failed", exc_info=True)

        # Bootstrap cost model from recent completion history
        self._bootstrap_cost_model()

        # Bootstrap timeout-advice model from the same history
        self._bootstrap_timeout_model()

        # One-time persistence reclaim (pre-writer, single-threaded): a gated full
        # VACUUM reclaims dead freelist + truncates the WAL AND commits the
        # auto_vacuum=INCREMENTAL conversion so future freed pages return to the
        # OS via the poller's incremental_vacuum. Normally a no-op (freelist below
        # threshold); runs HERE because VACUUM's exclusive lock can never touch
        # the live loop.
        try:
            reclaimed = self._queue_db.vacuum_full_blocking(
                self._config.startup_vacuum_freelist_threshold_bytes)
            if reclaimed:
                logger.info(
                    "llmproxy startup VACUUM reclaimed %.0f MB", reclaimed / 1e6)
        except Exception as exc:  # noqa: BLE001 — never block startup on maintenance
            logger.warning("llmproxy startup VACUUM skipped: %s", exc)

        # Phase 2.2: all startup recovery + bootstrap reads/writes are done
        # synchronously above while single-threaded; from here, route DB writes
        # to a dedicated writer thread so synchronous SQLite I/O never blocks the
        # event loop that schedules the whole fleet.
        self._queue_db.start_async_writer()

        # Phase 2.3: initial retention sweep (then daily in the poller) so the
        # completion corpus + timeout tables don't grow unbounded.
        self._queue_db.cleanup_old_completions(self._config.completions_retention_s)
        self._queue_db.cleanup_old_payloads(self._config.payload_retention_s)
        self._last_cleanup_at = time.monotonic()
        self._last_cache_stats_at = 0.0  # 0 → compute prefix-cache stats on first poll

        # Start background loops. The done-callbacks make an UNEXPECTED loop
        # exit loud (CRITICAL) — the iteration guards inside the loops should
        # make this unreachable, but a dead scheduler loop is a total outage
        # that uvicorn happily serves 503s through, so belt and braces.
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._scheduler_task.add_done_callback(
            lambda t: self._on_loop_task_exit("scheduler", t))
        self._poller_task = asyncio.create_task(self._capacity_poller_loop())
        self._poller_task.add_done_callback(
            lambda t: self._on_loop_task_exit("capacity poller", t))
        self._inflight_task = asyncio.create_task(self._inflight_stream_loop())
        await self._on_demand.start()

        logger.info(
            "llmproxy started: %d endpoints, %d total slots",
            len(self._config.endpoints), self._config.total_fleet_slots,
        )

    def _bootstrap_cost_model(self) -> None:
        """Replay recent successful completions to calibrate the cost model.

        The cost model is in-memory and lost on restart. This method
        seeds it from the proxy_completions table so DRR budget charges
        use observed performance instead of static defaults.
        """
        rows = self._queue_db.completions_for_calibration(hours=24)
        if not rows:
            return
        replayed = 0
        for r in rows:
            self._cost_model.record_completion(
                endpoint=r["endpoint"],
                call_site=r["call_site"],
                input_tokens=r["input_tokens"],
                output_tokens=r["output_tokens"],
                duration_s=r["duration_s"],
                occupancy_during=0,
            )
            replayed += 1
        logger.info(
            "cost model bootstrapped from %d recent completions", replayed,
        )

    def _bootstrap_timeout_model(self) -> None:
        """Seed the timeout-advice model from recent completion history so
        percentiles survive a restart.  End-to-end latency at replay time
        is ``duration_s*1000 + queue_wait_ms`` (the persisted columns)."""
        window_h = self._config.timeout_advice_window_s / 3600.0
        rows = self._queue_db.timeout_samples(hours=window_h)
        if not rows:
            return
        now = time.monotonic()
        for r in rows:
            end_to_end_ms = (r["duration_s"] * 1000.0) + r["queue_wait_ms"]
            self._timeout_model.record(
                endpoint=r["endpoint"],
                priority=r["priority"],
                input_tokens=r["input_tokens"],
                output_tokens=r["output_tokens"],
                end_to_end_ms=end_to_end_ms,
                status="ok",
                now=now,
            )
        logger.info(
            "timeout model bootstrapped from %d recent completions", len(rows),
        )

    async def shutdown(self) -> None:
        # Phase 2.1: drain in-flight dispatches (bounded) before teardown so a
        # graceful (SIGTERM) restart doesn't drop running LLM work. New submits
        # are rejected (deferrable) while draining; the scheduler stops admitting.
        self._draining.set()
        # Signal SSE subscribers to drop so their handlers return promptly
        # instead of holding uvicorn's graceful-shutdown budget open.
        self._sse.close_all()
        if self._scheduler_task:
            self._scheduler_task.cancel()
        tasks = [t for t in self._inflight_tasks.values() if not t.done()]
        if tasks:
            logger.info(
                "draining %d in-flight dispatch(es) (≤%.0fs)", len(tasks), _DRAIN_DEADLINE_S)
            # asyncio.wait (NOT wait_for(gather)) returns (done, pending) WITHOUT
            # cancelling the pending tasks — so we cancel stragglers explicitly +
            # count them. wait_for(gather) would cancel them itself on timeout,
            # robbing us of the count and the explicit ordering below.
            _done, pending = await asyncio.wait(tasks, timeout=_DRAIN_DEADLINE_S)
            if pending:
                logger.warning(
                    "drain deadline hit — cancelling %d straggler(s)", len(pending))
                for t in pending:
                    t.cancel()
                self._drain_straggler_cancelled += len(pending)
                # Phase 5B.2: AWAIT the cancellations so each straggler's
                # CancelledError handler runs (resolves the caller + records the
                # completion → frees the slot) BEFORE _queue_db.close() flushes
                # below — otherwise close() can flush a half-written completion or
                # leave the caller unresolved. Bounded so a wedged cancel can't
                # hang shutdown.
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*pending, return_exceptions=True),
                        timeout=3.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "drain: %d straggler(s) didn't unwind within 3s of cancel",
                        len([t for t in pending if not t.done()]))
        if self._poller_task:
            self._poller_task.cancel()
        if self._inflight_task:
            self._inflight_task.cancel()
        # Release any held on-demand dispatcher lease so the GPU slot frees for
        # other services across the restart (don't wait out the lease TTL).
        try:
            await self._on_demand.close()
        except Exception:
            logger.warning("on_demand close failed during shutdown", exc_info=True)
        # Persist DRR balances so fairness survives the restart (Phase 3.4); the
        # writer flushes this during close().
        try:
            self._queue_db.save_budgets(self._budget_mgr.snapshot())
        except Exception as exc:  # noqa: BLE001
            logger.debug("budget save on shutdown failed: %s", exc)
        await self._backend.close()
        self._queue_db.close()  # flushes the writer queue, then closes connections
        self._request_logger.close()

    # ----- handler: /v1/submit -----

    def _resolve_endpoint(self, body: dict) -> str:
        return self._lifecycle.resolve_endpoint(body)

    async def handle_submit(
        self, body: dict, request: Request, *, openai: bool = False,
    ) -> Response:
        # ``openai=True`` (set only by the /v1/chat/completions front door)
        # varies ONLY the response serialization: the bare OpenAI
        # chat.completion / chat.completion.chunk + [DONE] stream, instead of
        # the internal submit envelope. The enqueue / scheduler / grammar /
        # cache / DRR / telemetry path is identical. Default False keeps every
        # agent's /v1/submit response byte-identical.
        return await self._lifecycle.handle_submit(body, request, openai=openai)

    def _extract_grammar(self, payload: dict) -> tuple[str | None, str | None]:
        return self._correction.extract_grammar(payload)

    def _process_grammar(self, req: QueuedRequest) -> dict | None:
        return self._correction.process_grammar(req)

    def _shadow_egress_detect(self, req: "QueuedRequest", result: dict) -> None:
        return self._correction.shadow_egress_detect(req, result)

    async def _maybe_correct_degenerate(
        self, req: "QueuedRequest", result: dict,
    ) -> None:
        return await self._correction.maybe_correct_degenerate(req, result)

    def _thinking_allowed_keys(self, payload: dict) -> list[str]:
        return self._correction.thinking_allowed_keys(payload)

    def _apply_thinking(self, req: QueuedRequest) -> None:
        return self._correction.apply_thinking(req)

    def _finalize_thinking(self, req: QueuedRequest, result: dict) -> None:
        return self._correction.finalize_thinking(req, result)

    async def _handle_sync_submit(
        self, req: QueuedRequest, cache_key: str | None, *, openai: bool = False,
    ) -> Response:
        return await self._lifecycle.handle_sync_submit(req, cache_key, openai=openai)

    async def _handle_streaming_submit(
        self, req: QueuedRequest, *, openai: bool = False,
    ) -> Response:
        return await self._lifecycle.handle_streaming_submit(req, openai=openai)

    # ----- handler: OpenAI compat -----

    @staticmethod
    def _openai_error(
        message: str, err_type: str, status_code: int, code: str | None = None,
    ) -> JSONResponse:
        return _mk_openai_error(message, err_type, status_code, code)

    async def handle_openai_chat(self, body: dict, request: Request) -> Response:
        return await self._http.handle_openai_chat(body, request)

    async def handle_openai_embeddings(self, body: dict, request: Request) -> Response:
        return await self._http.handle_openai_embeddings(body, request)

    # ----- handler: models -----

    async def handle_models(self, request: Request) -> Response:
        return await self._http.handle_models(request)

    # ----- handler: status -----

    async def handle_prometheus_metrics(self, request: Request) -> Response:
        return await self._http.handle_prometheus_metrics(request)

    async def handle_status(self, request: Request) -> Response:
        return await self._http.handle_status(request)

    # ----- handler: admin endpoint pause/resume (Phase 5F operator drain) -----

    def _audit_admin_ip(self, route: str, remote_ip: str) -> None:
        return self._http.audit_admin_ip(route, remote_ip)

    async def handle_admin_flags(self, request: Request) -> Response:
        return await self._http.handle_admin_flags(request)

    async def handle_admin_endpoint_pause(
        self, endpoint: str, request: Request, *, pause: bool,
    ) -> Response:
        return await self._http.handle_admin_endpoint_pause(endpoint, request, pause=pause)

    # ----- handler: maintenance-window annotation (planned-restart tag) -----

    async def handle_maintenance(self, request: Request) -> Response:
        return await self._http.handle_maintenance(request)

    async def handle_maintenance_list(self, request: Request) -> Response:
        return await self._http.handle_maintenance_list(request)

    # ----- handler: metrics -----

    def _metrics_payload(self, now: float) -> dict:
        return self._state.metrics_payload(now)

    async def handle_metrics(self, request: Request) -> Response:
        return await self._http.handle_metrics(request)

    # ----- handlers: fleet usage (Phase 1: proxy = fleet call-metrics authority) -----

    async def handle_fleet_activity(self, request: Request) -> Response:
        return await self._http.handle_fleet_activity(request)

    async def handle_fleet_savings(self, request: Request) -> Response:
        return await self._http.handle_fleet_savings(request)

    async def handle_top_callers(self, request: Request) -> Response:
        return await self._http.handle_top_callers(request)

    async def handle_fleet_cache_stats(self, request: Request) -> Response:
        return await self._http.handle_fleet_cache_stats(request)

    async def handle_cache_attribution(self, request: Request) -> Response:
        return await self._http.handle_cache_attribution(request)

    async def handle_usage(self, request: Request) -> Response:
        return await self._http.handle_usage(request)

    async def handle_series(self, request: Request) -> Response:
        return await self._http.handle_series(request)

    # ----- handler: calls ingest (non-LLM fleet calls) -----

    async def handle_calls_log(self, request: Request) -> Response:
        return await self._http.handle_calls_log(request)

    # ----- handler: SSE stream -----

    async def handle_stream(self, request: Request) -> Response:
        return await self._http.handle_stream(request)

    async def handle_cost_model(self, request: Request) -> Response:
        return await self._http.handle_cost_model(request)

    # ----- handler: history -----

    async def handle_history(self, request: Request) -> Response:
        return await self._http.handle_history(request)

    # ----- handler: recent requests (for feed) -----

    async def handle_recent(self, request: Request) -> Response:
        return await self._http.handle_recent(request)

    # ----- handler: live in-flight (what's executing right now) -----

    async def handle_inflight(self, request: Request) -> Response:
        return await self._http.handle_inflight(request)

    # ----- handler: timeout advice -----

    async def handle_timeout_advice(self, request: Request) -> Response:
        return await self._http.handle_timeout_advice(request)

    async def handle_timeout_shadow_report(self, request: Request) -> Response:
        return await self._http.handle_timeout_shadow_report(request)

    async def handle_timeouts_report(self, request: Request) -> Response:
        return await self._http.handle_timeouts_report(request)

    async def handle_stall_aborts(self, request: Request) -> Response:
        return await self._http.handle_stall_aborts(request)

    # ----- handler: health -----

    def _scheduler_loop_alive(self) -> bool:
        return self._health.scheduler_loop_alive()

    def _poller_alive(self) -> bool:
        return self._health.poller_alive()

    async def handle_health(self, request: Request) -> Response:
        return await self._http.handle_health(request)

    async def handle_readyz(self, request: Request) -> Response:
        # Liveness is handle_health (fail-open); this is READINESS and fails
        # closed on the conversational endpoint — §8.0 req 4. ProxyHttpHandlers
        # is composed, not inherited, so a handler needs this delegator too:
        # routes.py calls `svc.handle_readyz`, and without it the route 500s
        # with AttributeError rather than failing to register.
        return await self._http.handle_readyz(request)

    # ----- scheduler loop -----

    def _on_loop_task_exit(self, name: str, task: asyncio.Task) -> None:
        return self._health.on_loop_task_exit(name, task)

    async def _scheduler_loop(self) -> None:
        return await self._lifecycle.scheduler_loop()

    async def _execute_dispatch(self, decision: DispatchDecision) -> None:
        return await self._lifecycle.execute_dispatch(decision)

    async def _execute_sync(
        self,
        req: QueuedRequest,
        ep_cfg: EndpointConfig,
        decision: DispatchDecision,
    ) -> None:
        return await self._lifecycle.execute_sync(req, ep_cfg, decision)

    async def _execute_shadow(
        self,
        req: QueuedRequest,
        ep_cfg: EndpointConfig,
        decision: DispatchDecision,
        primary_resp: BackendResponse,
    ) -> None:
        return await self._lifecycle.execute_shadow(req, ep_cfg, decision, primary_resp)

    async def _execute_streaming(
        self,
        req: QueuedRequest,
        ep_cfg: EndpointConfig,
        decision: DispatchDecision,
    ) -> None:
        return await self._lifecycle.execute_streaming(req, ep_cfg, decision)

    # ----- circuit breaker / integrity helpers (Phase 1) -----

    def _endpoint_healthy(self, endpoint: str) -> bool:
        return self._health.endpoint_healthy(endpoint)

    def _retry_after_s(self, endpoint: str) -> int:
        return self._health.retry_after_s(endpoint)

    @staticmethod
    def _is_transient_backend_error(exc: Exception) -> bool:
        return Correction.is_transient_backend_error(exc)

    def _request_is_structured(self, req: QueuedRequest) -> bool:
        return self._correction.request_is_structured(req)

    async def _update_endpoint_health(
        self, ep_name: str, ep_cfg: "EndpointConfig", probe_ok: bool,
    ) -> None:
        return await self._health.update_endpoint_health(ep_name, ep_cfg, probe_ok)

    def _fast_fail_interactive(self, ep_name: str) -> None:
        return self._health.fast_fail_interactive(ep_name)

    def _evaluate_alerts(self, now: float) -> None:
        return self._health.evaluate_alerts(now)

    def _resolve_error(self, req: QueuedRequest, error: str) -> None:
        return self._state.resolve_error(req, error)

    def _record_completion(
        self,
        req: QueuedRequest,
        decision: DispatchDecision,
        duration_s: float,
        input_tokens: int,
        output_tokens: int,
        status: str,
        response_body: dict | None = None,
        finish_reason: str | None = None,
    ) -> None:
        return self._lifecycle.record_completion(req, decision, duration_s, input_tokens, output_tokens, status, response_body, finish_reason)

    def _record_timeout_shadow(
        self,
        req: QueuedRequest,
        now: float,
        duration_s: float,
        output_tokens: int,
        status: str,
    ) -> None:
        return self._lifecycle.record_timeout_shadow(req, now, duration_s, output_tokens, status)

    # ----- timeout events -----

    def _record_timeout_event(
        self,
        req: QueuedRequest,
        *,
        layer: str,
        elapsed_s: float,
        queue_wait_ms: float | None = None,
        emit_metrics_and_log: bool = True,
    ) -> None:
        return self._lifecycle.record_timeout_event(req, layer=layer, elapsed_s=elapsed_s, queue_wait_ms=queue_wait_ms, emit_metrics_and_log=emit_metrics_and_log)

    def _on_admission_timeout(self, req: QueuedRequest) -> None:
        return self._lifecycle.on_admission_timeout(req)

    def _on_spill(self, req: QueuedRequest, src: str, target: str) -> None:
        """One request moved to remote capacity because local was full."""
        self._state.spilled_from[src] = self._state.spilled_from.get(src, 0) + 1
        logger.info(
            "ROADSTEAD_SPILL %s -> %s agent=%s call_site=%s request_id=%s",
            src, target, req.agent_id, req.call_site, req.request_id,
        )

    # ----- live in-flight streamer -----

    async def _inflight_stream_loop(self) -> None:
        return await self._health.inflight_stream_loop()

    # ----- capacity poller -----

    async def _poll_endpoint_once(self, ep_name: str, ep_cfg: EndpointConfig) -> None:
        return await self._health.poll_endpoint_once(ep_name, ep_cfg)

    async def _capacity_poller_loop(self) -> None:
        return await self._health.capacity_poller_loop()

    async def _poller_iteration(self) -> None:
        return await self._health.poller_iteration()

    async def _compute_cache_stats(self) -> None:
        return await self._health.compute_cache_stats()

    def _apply_discovered_props(
        self, ep_name: str, ep_cfg: EndpointConfig, props: dict,
    ) -> None:
        return self._health.apply_discovered_props(ep_name, ep_cfg, props)

    def _apply_discovered_vllm_capacity(
        self, ep_name: str, ep_cfg: EndpointConfig, cap: dict,
    ) -> None:
        return self._health.apply_discovered_vllm_capacity(ep_name, ep_cfg, cap)


# avoid circular import — EndpointConfig used in type hints
from .config import EndpointConfig as EndpointConfig  # noqa: E402
