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
from .health import Health
from .lifecycle import Lifecycle, _openai_error as _mk_openai_error
from .sse_hub import DROP_SENTINEL, SSEHub
from .state import ProxyState

logger = logging.getLogger(__name__)


def _parse_duration(s: str, default: int) -> int:
    """Parse a '24h'/'90m'/'3600s' duration to seconds (ported from the host
    telemetry daemon so the unified frontend's window params are identical)."""
    import re
    m = re.match(r"^(\d+)([smhd])$", (s or "").strip())
    if not m:
        return default
    n, u = int(m.group(1)), m.group(2)
    return n * {"s": 1, "m": 60, "h": 3600, "d": 86400}[u]


def _clamp_window(s: str, default: int, *, cap: int = 7 * 86400) -> int:
    return max(60, min(_parse_duration(s, default), cap))


def _bin_seconds_for(window_s: int) -> int:
    """Bin width that keeps a series in ~60-150 points (matches the daemon)."""
    if window_s <= 3600:
        return 60
    if window_s <= 6 * 3600:
        return 300
    if window_s <= 86400:
        return 600
    return 3600


def _to_int(value: object, default: int) -> int:
    """Parse a query-param / body field to int, falling back to ``default``
    on anything malformed — so a bad ``?bin=abc`` uses the default instead of
    escaping as an unhandled 500 from these read-only/best-effort handlers."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _to_float(value: object, default: float) -> float:
    """Float counterpart to ``_to_int`` — default on malformed input."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


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
        # Correction: grammar/thinking/degeneration/empty-rescue/egress guards.
        self._correction = Correction(self._state)
        # Lifecycle: admission -> dispatch -> response hot path (uses Correction
        # + Health by reference).
        self._lifecycle = Lifecycle(self._state, self._correction, self._health)

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
        for row in self._queue_db.load_budgets():
            # Phase 5B.4: prefer the agent's CONFIGURED weight + cap over the
            # persisted row. get_or_create only applies weight/max_balance when
            # the agent is NEW, and for a lazily-created agent (not in config)
            # the old call passed no max_balance → it defaulted to 60.0,
            # clamping the restored balance to the wrong bound + letting a stale
            # persisted weight override config intent. Reconcile explicitly.
            acfg = self._config.agents.get(row["agent_id"])
            weight = acfg.weight if acfg else row["weight"]
            b = self._budget_mgr.get_or_create(row["agent_id"], weight=weight)
            if acfg:
                b.max_balance = acfg.max_balance_ss
                b.weight = acfg.weight
            b.balance = max(-b.max_balance, min(b.max_balance, row["balance"]))
            b.total_consumed = row["total_consumed"]
            b.last_replenish_at = time.monotonic()

        # Recover queued requests from WAL
        recovered = self._queue_db.recover_queued(time.monotonic())
        for req in recovered:
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
        remote_ip = request.client.host if request.client else "unknown"
        identity = self._acl.identify(remote_ip)
        if not identity:
            # Phase 5D: OpenAI-shaped 403 (was a bare {"error": "<str>"} that a
            # strict OpenAI client crashes on doing resp.error.message).
            return self._openai_error(
                f"access denied for {remote_ip}", "access_denied", 403,
                code="access_denied")
        agent_id, default_priority = identity
        model = body.get("model", "qwen-analyst")
        # Phase 5D: validate the model maps to a known endpoint BEFORE enqueue.
        # Otherwise an unknown model burns a scheduler slot + DRR charge and
        # fails late with a confusing 502; OpenAI clients expect 404/model_not_found.
        if normalize_endpoint(str(model)) not in self._config.endpoints:
            return self._openai_error(
                f"unknown model {model!r}", "model_not_found", 404,
                code="unknown_endpoint")

        # Honor a client-supplied deadline (goose recipes can run long): a
        # ``timeout_s`` body field or an ``X-Timeout-S`` header overrides the
        # 180s default. Popped from the body so it isn't forwarded to the
        # backend (which would reject the unknown field).
        client_timeout = body.pop("timeout_s", None) or request.headers.get("X-Timeout-S")
        try:
            timeout_s = float(client_timeout) if client_timeout else _DEFAULT_TIMEOUT_S
        except (TypeError, ValueError):
            timeout_s = _DEFAULT_TIMEOUT_S

        submit_body = {
            "agent_id": agent_id,
            "endpoint": model,
            "priority": int(default_priority),
            "call_site": f"{agent_id}.openai_compat",
            "caller_id": agent_id,
            "payload_type": "chat_completion",
            "payload": body,
            "timeout_s": timeout_s,
        }
        return await self.handle_submit(submit_body, request, openai=True)

    async def handle_openai_embeddings(self, body: dict, request: Request) -> Response:
        remote_ip = request.client.host if request.client else "unknown"
        identity = self._acl.identify(remote_ip)
        if not identity:
            return self._openai_error(
                f"access denied for {remote_ip}", "access_denied", 403,
                code="access_denied")
        agent_id, default_priority = identity

        submit_body = {
            "agent_id": agent_id,
            "endpoint": "bge-m3-embed",
            "priority": int(default_priority),
            "call_site": f"{agent_id}.openai_compat_embed",
            "payload_type": "embedding",
            "payload": body,
            "timeout_s": 60.0,
        }
        # Phase 5D: openai=True so success returns the bare OpenAI embeddings
        # object ({object:list,data:[...],usage}) and errors are OpenAI-shaped —
        # was leaking the internal {status,response} envelope to OpenAI clients.
        return await self.handle_submit(submit_body, request, openai=True)

    # ----- handler: models -----

    async def handle_models(self, request: Request) -> Response:
        models = []
        for ep_name, ep_cfg in self._config.endpoints.items():
            models.append({
                "id": ep_cfg.role,
                "object": "model",
                "owned_by": "collective",
                "endpoint_class": ep_name,
                "max_slots": ep_cfg.max_slots,
                "context_per_slot": ep_cfg.context_per_slot,
            })
        return JSONResponse({"object": "list", "data": models})

    # ----- handler: status -----

    async def handle_prometheus_metrics(self, request: Request) -> Response:
        """GET /metrics — Prometheus text exposition of CURRENT QoS aggregates
        (timeseries_migration_plan Phase 3.1). In-memory reads only (scheduler +
        the 300s rolling window + since-boot counters), so ~zero scrape cost; the
        proxy_timeouts/_completions event tables stay the forensic system-of-
        record. Layer-split timeout counts + would-timeout% remain on
        /v1/timeouts (SQL-windowed), not here. Never raises — degrades to empty."""
        from originfleet.framework.metrics import Metric, render_prometheus

        now = time.monotonic()
        out: list[Metric] = []
        _Q = {50: "0.5", 95: "0.95"}
        try:
            for ep_name in self._config.endpoints:
                lbl = {"endpoint": ep_name}
                snap = self._scheduler.endpoint_snapshot(ep_name)
                if snap.get("max_slots") is not None:
                    out.append(Metric("llmproxy_endpoint_slots_total",
                                      snap["max_slots"], lbl, "gauge",
                                      help="configured max concurrent slots"))
                if snap.get("in_flight") is not None:
                    out.append(Metric("llmproxy_endpoint_inflight",
                                      snap["in_flight"], lbl, "gauge",
                                      help="requests dispatched and in flight"))
                if snap.get("queued") is not None:
                    out.append(Metric("llmproxy_endpoint_queued", snap["queued"],
                                      lbl, "gauge", help="requests waiting in queue"))
                for band, n in (snap.get("queue_by_band") or {}).items():
                    out.append(Metric("llmproxy_endpoint_queued_by_band", n,
                                      {"endpoint": ep_name, "band": str(band)},
                                      "gauge", help="queued requests by priority band"))
                for q, qlabel in _Q.items():
                    w = self._metrics.percentile("queue_wait_ms", q,
                                                 endpoint=ep_name, now=now)
                    if w is not None:
                        out.append(Metric("llmproxy_queue_wait_ms", w,
                                          {"endpoint": ep_name, "quantile": qlabel},
                                          "gauge", help="queue-wait latency (ms)"))
                    b = self._metrics.percentile("backend_latency_ms", q,
                                                 endpoint=ep_name, now=now)
                    if b is not None:
                        out.append(Metric("llmproxy_backend_latency_ms", b,
                                          {"endpoint": ep_name, "quantile": qlabel},
                                          "gauge", help="backend latency (ms)"))
                out.append(Metric("llmproxy_recent_timeouts_5m",
                                  self._metrics.count(endpoint=ep_name,
                                                      status="timeout", now=now),
                                  lbl, "gauge",
                                  help="timeouts in the last 5 min"))
                out.append(Metric("llmproxy_recent_requests_5m",
                                  self._metrics.count(endpoint=ep_name, now=now),
                                  lbl, "gauge",
                                  help="requests in the last 5 min"))
                ep_cfg = self._config.endpoints[ep_name]
                ss_consumed = self._metrics.slot_seconds_consumed(ep_name, now)
                out.append(Metric("llmproxy_slot_seconds_5m", ss_consumed, lbl,
                                  "gauge", help="slot-seconds consumed in 5 min"))
                if ep_cfg.max_slots > 0:
                    ss_available = ep_cfg.max_slots * 300.0
                    util = (ss_consumed / ss_available * 100) if ss_available > 0 else 0
                    out.append(Metric("llmproxy_endpoint_utilization_pct",
                                      round(util, 1), lbl, "gauge",
                                      help="5-min slot utilization %"))
                h = self._endpoint_health.get(ep_name, {})
                out.append(Metric("llmproxy_endpoint_healthy",
                                  1 if h.get("healthy", True) else 0, lbl, "gauge",
                                  help="1 if the endpoint is healthy (not paused)"))

            stats = self._scheduler.stats()
            for key, name in (("total_dispatched", "llmproxy_dispatched_total"),
                              ("total_completed", "llmproxy_completed_total"),
                              ("total_timeouts", "llmproxy_timeouts_total")):
                if stats.get(key) is not None:
                    out.append(Metric(name, stats[key], {}, "counter",
                                      help="since-boot scheduler counter"))
        except Exception:
            logger.exception("llmproxy /metrics render failed")
            out = []
        return Response(render_prometheus(out),
                        media_type="text/plain; version=0.0.4")

    async def handle_status(self, request: Request) -> Response:
        now = time.monotonic()
        endpoints = {}
        for ep_name in self._config.endpoints:
            snap = self._scheduler.endpoint_snapshot(ep_name)
            snap["p50_wait_ms"] = self._metrics.percentile(
                "queue_wait_ms", 50, endpoint=ep_name, now=now,
            )
            snap["p95_wait_ms"] = self._metrics.percentile(
                "queue_wait_ms", 95, endpoint=ep_name, now=now,
            )
            snap["throughput_rps"] = round(
                self._metrics.throughput_rps(ep_name, now), 2,
            )
            ep_cfg = self._config.endpoints[ep_name]
            if ep_cfg.max_slots > 0:
                ss_consumed = self._metrics.slot_seconds_consumed(ep_name, now)
                ss_available = ep_cfg.max_slots * 300.0  # 5-min window
                snap["utilization_pct"] = round(
                    (ss_consumed / ss_available * 100) if ss_available > 0 else 0, 1,
                )
            h = self._endpoint_health.get(ep_name, {})
            snap["healthy"] = h.get("healthy", True)
            snap["paused"] = not h.get("healthy", True)  # check_alerts (Phase 2.5) keys on this
            # Survivorship fix (2026-06-06): the timeout-advice model + shadow
            # only ingest status==ok, so they reported a misleading "0 would
            # timeout" while requests were actually timing out. Surface the real
            # 5-min timeout count per endpoint so a partial stall is VISIBLE
            # (feeds the endpoint_stalled alert + health-verifier/dashboards).
            snap["recent_timeouts"] = self._metrics.count(
                endpoint=ep_name, status="timeout", now=now)
            endpoints[ep_name] = snap

        agents = {
            b["agent_id"]: b
            for b in self._budget_mgr.snapshot()
        }

        # Loop liveness is computed AT READ TIME: the poller is what evaluates
        # self._alerts, so a dead poller (or scheduler) can never report itself
        # through that path — only through this one.
        alerts = list(self._alerts)
        if not self._scheduler_loop_alive():
            alerts.append({
                "name": "scheduler_loop_dead", "severity": "CRITICAL",
                "detail": "scheduler loop task not running — dispatch is DOWN",
            })
        if not self._poller_alive():
            alerts.append({
                "name": "poller_dead", "severity": "CRITICAL",
                "detail": "capacity poller task not running — health probing, "
                          "alerting, and budget persistence have stopped",
            })

        return JSONResponse({
            "endpoints": endpoints,
            "agents": agents,
            "alerts": alerts,  # Phase 2.5 — health-verifier/log_scan surface
            # Runtime feature flags (flags.py) — read by dashboards + the
            # scheduled shadow→enforce flip checks.
            "flags": self._flags.as_dict(),
            "cache": self._cache.stats(),
            "scheduler": {
                **self._scheduler.stats(),
                "uptime_s": round(time.monotonic() - self._started_at, 0),
            },
            # Phase 5B reliability counters.
            "reliability": {
                "slot_leak_reclaimed": self._slot_leak_reclaimed,
                "drain_straggler_cancelled": self._drain_straggler_cancelled,
                "writer_thread_alive": self._queue_db.writer_alive(),
                "writer_thread_restarts": self._queue_db.writer_restarts(),
                "write_q_dropped": self._queue_db.write_q_dropped(),
                # Critical-loop liveness (read-time; see alerts note above).
                "scheduler_alive": self._scheduler_loop_alive(),
                "poller_alive": self._poller_alive(),
                # Source IPs seen on admin-ish routes since boot — the
                # ACL-tightening go/no-go reads this instead of grepping logs.
                "admin_ips_seen": {
                    route: sorted(ips)
                    for route, ips in self._admin_ips_seen.items()
                },
                # Unknown-endpoint submits since boot (shadow counter for the
                # unknown_endpoint_enforce flip check). Empty = safe to flip.
                "unknown_endpoint_submits": self._unknown_endpoint_submits,
                # Context-gate hits since boot (shadow counter for the
                # context_gate_enforce flip check — compare against actual
                # backend overflow errors before flipping).
                "context_overflows_shadow": self._context_overflows,
                # Thinking option (per-request native reasoning) health. All 0
                # until a caller opts in with thinking:true. Watch
                # thinking_truncated to tune COLLECTIVE_PROXY_THINKING_BUDGET down.
                "thinking_requests": self._thinking_requests,
                "thinking_clean": self._thinking_clean,
                "thinking_recovered": self._thinking_recovered,
                "thinking_truncated": self._thinking_truncated,
                "thinking_fallback": self._thinking_fallback,
                # WS-4 shadow egress detector — silent grammar-drop over ALL
                # grammar-bearing responses (read-only/zero-risk). Per-call_site
                # rate + a flat fleet rate (health-verifier thresholds the scalar).
                "silent_drop_by_call_site": {
                    cs: {**t, "rate": round(t["dropped"] / t["checked"], 4)}
                    for cs, t in self._shadow_drop.items() if t["checked"]
                },
                "silent_drop_rate": round(
                    sum(t["dropped"] for t in self._shadow_drop.values())
                    / max(1, sum(t["checked"] for t in self._shadow_drop.values())),
                    4,
                ),
                # Empty-completion (position-0-EOS) rescue: retries dispatched
                # with min_tokens, and how many produced a real response.
                "empty_rescue_attempts": self._empty_rescue_attempts,
                "empty_rescue_recovered": self._empty_rescue_recovered,
                # Egress degeneration guard — repetition-loop responses detected,
                # and how many an anti-repetition re-dispatch recovered.
                "degeneration_detected": self._degeneration_detected,
                "degeneration_recovered": self._degeneration_recovered,
                "degeneration_unrecovered": self._degeneration_unrecovered,
                "degeneration_by_call_site": {
                    cs: t for cs, t in self._degeneration_by_call_site.items()
                    if t["detected"]
                },
            },
            # Phase 5F — endpoints an operator has drained for maintenance.
            "paused_endpoints": sorted(self._paused_endpoints),
        })

    # ----- handler: admin endpoint pause/resume (Phase 5F operator drain) -----

    def _audit_admin_ip(self, route: str, remote_ip: str) -> None:
        """Track source IPs per admin-ish route (exposed on /v1/status) and log
        the FIRST hit per (route, ip) — the data the ACL-tightening go/no-go
        needs, without per-hit log volume."""
        seen = self._admin_ips_seen.setdefault(route, set())
        if remote_ip not in seen:
            seen.add(remote_ip)
            logger.info("admin-audit: first hit on %s from %s", route, remote_ip)

    async def handle_admin_flags(self, request: Request) -> Response:
        """GET: current runtime flags. POST: update a subset (JSON object of
        flag→bool), persisted across restarts. Internal-only (ACL). This is the
        flip surface for the shadow→enforce switches and kill-switches — flags
        change behaviour immediately, no process restart."""
        remote_ip = request.client.host if request.client else "unknown"
        self._audit_admin_ip("/v1/admin/flags", remote_ip)
        if not self._acl.is_admin(remote_ip):
            return JSONResponse(
                {"error": f"access denied for {remote_ip}"}, status_code=403)
        if request.method == "GET":
            return JSONResponse({"flags": self._flags.as_dict()})
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        try:
            # File write runs off-loop; flag reads elsewhere are plain dict
            # lookups on the loop thread (single mutation source — this handler).
            updated = await asyncio.to_thread(self._flags.set_many, body)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        logger.warning("runtime flags updated by %s: %s", remote_ip, body)
        return JSONResponse({"flags": updated})

    async def handle_admin_endpoint_pause(
        self, endpoint: str, request: Request, *, pause: bool,
    ) -> Response:
        """Pause (drain) or resume an endpoint for maintenance — e.g. a vLLM
        restart to change --max-model-len. Internal-only (ACL).

        PAUSE marks the endpoint unhealthy NOW: background work defers (it
        queues + drains on resume — no loss while deadlines exceed the restart),
        interactive fast-fails with a deferrable error, and the poller stops
        probing it — so no request hits a backend you're about to kill, with no
        ~30s auto-circuit-trip lag. RESUME hands it back to the poller, which
        re-probes, recovers on /health, re-discovers capacity (the new
        max_model_len), and the deferred queue drains."""
        remote_ip = request.client.host if request.client else "unknown"
        self._audit_admin_ip("/v1/admin/endpoints", remote_ip)
        if not self._acl.is_admin(remote_ip):
            return JSONResponse(
                {"error": f"access denied for {remote_ip}"}, status_code=403)
        ep = normalize_endpoint(endpoint)
        if ep not in self._config.endpoints:
            return JSONResponse(
                {"error": f"unknown endpoint {endpoint!r}"}, status_code=404)
        # Optional free-text reason on PAUSE, recorded on the maintenance window
        # so the timeout burst during the restart reads as PLANNED (with the
        # reason) in /v1/timeouts. Body is optional — never fail the drain on it.
        reason = ""
        try:
            body = await request.json()
            if isinstance(body, dict):
                reason = str(body.get("reason") or "")
        except Exception:  # noqa: BLE001 — empty/invalid body is fine
            pass
        if pause:
            self._paused_endpoints.add(ep)
            # Release any already-queued interactive/foreground immediately with
            # a deferrable error (don't make them wait out their deadline).
            self._fast_fail_interactive(ep)
            # Open a maintenance window so timeouts during the restart are tagged
            # planned. Closed on RESUME (below).
            if self._queue_db is not None:
                self._queue_db.maintenance_open(
                    endpoint=ep, reason=reason, operator=remote_ip, source="drain")
            logger.warning(
                "endpoint %s PAUSED by operator (%s) — background defers, "
                "interactive fast-fails; backend safe to restart%s", ep, remote_ip,
                f" (reason: {reason})" if reason else "")
        else:
            self._paused_endpoints.discard(ep)
            self._dispatch_event.set()  # nudge the scheduler to drain deferred work
            if self._queue_db is not None:
                self._queue_db.maintenance_close(endpoint=ep)
            logger.warning(
                "endpoint %s RESUMED by operator (%s) — poller will re-probe / "
                "recover / re-discover capacity; deferred queue draining",
                ep, remote_ip)
        return JSONResponse({
            "endpoint": ep,
            "paused": ep in self._paused_endpoints,
            "healthy": self._endpoint_healthy(ep),
            "paused_endpoints": sorted(self._paused_endpoints),
        })

    # ----- handler: maintenance-window annotation (planned-restart tag) -----

    async def handle_maintenance(self, request: Request) -> Response:
        """Annotate a maintenance window so a timeout burst during a DELIBERATE
        backend restart reads as PLANNED (not an incident) in /v1/timeouts.
        Internal-only (ACL). Use this when you restart a backend WITHOUT the
        pause/resume drain (which records the window automatically).

        Body (JSON):
          endpoint:    str | [str] | "*"   (required) class(es) under maintenance
          reason:      str                 free-text ("thinker restart: 128K")
          duration_s:  float (optional)    backdate a CLOSED window [now-d, now]
          started_at / ended_at: epoch s   explicit bounds (override duration_s)
        With none of duration_s/started_at/ended_at, opens an OPEN window now
        (close it later via the drain resume, or re-POST with ended_at)."""
        remote_ip = request.client.host if request.client else "unknown"
        self._audit_admin_ip("/v1/admin/maintenance", remote_ip)
        if not self._acl.is_admin(remote_ip):
            return JSONResponse(
                {"error": f"access denied for {remote_ip}"}, status_code=403)
        if self._queue_db is None:
            return JSONResponse({"error": "no persistence backend"}, status_code=503)
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        if not isinstance(body, dict):
            body = {}
        raw_eps = body.get("endpoint")
        if not raw_eps:
            return JSONResponse({"error": "endpoint required"}, status_code=400)
        eps = raw_eps if isinstance(raw_eps, list) else [raw_eps]
        norm_eps: list[str] = []
        for e in eps:
            if str(e) == "*":
                norm_eps.append("*")
                continue
            ne = normalize_endpoint(str(e))
            if ne not in self._config.endpoints:
                return JSONResponse(
                    {"error": f"unknown endpoint {e!r}"}, status_code=404)
            norm_eps.append(ne)
        reason = str(body.get("reason") or "")
        started_at = body.get("started_at")
        ended_at = body.get("ended_at")
        duration_s = body.get("duration_s")
        now = time.time()
        recorded: list[dict] = []
        try:
            for ep in norm_eps:
                if started_at is not None or ended_at is not None:
                    s = float(started_at) if started_at is not None else now
                    en = float(ended_at) if ended_at is not None else now
                    self._queue_db.maintenance_record(
                        endpoint=ep, started_at=s, ended_at=en,
                        reason=reason, operator=remote_ip)
                    recorded.append({"endpoint": ep, "started_at": s,
                                     "ended_at": en, "reason": reason})
                elif duration_s is not None:
                    s = now - float(duration_s)
                    self._queue_db.maintenance_record(
                        endpoint=ep, started_at=s, ended_at=now,
                        reason=reason, operator=remote_ip)
                    recorded.append({"endpoint": ep, "started_at": s,
                                     "ended_at": now, "reason": reason})
                else:
                    self._queue_db.maintenance_open(
                        endpoint=ep, reason=reason, operator=remote_ip,
                        source="manual")
                    recorded.append({"endpoint": ep, "started_at": now,
                                     "ended_at": None, "reason": reason})
        except (TypeError, ValueError) as exc:
            return JSONResponse(
                {"error": f"bad window bounds: {exc}"}, status_code=400)
        logger.warning(
            "maintenance window(s) recorded by operator (%s): %s reason=%r",
            remote_ip, [r["endpoint"] for r in recorded], reason)
        return JSONResponse({"recorded": recorded})

    async def handle_maintenance_list(self, request: Request) -> Response:
        """List maintenance windows overlapping the last ``hours`` (default 24).
        Admin surface (was unauthenticated — tightened with the rest)."""
        remote_ip = request.client.host if request.client else "unknown"
        self._audit_admin_ip("/v1/admin/maintenance", remote_ip)
        if not self._acl.is_admin(remote_ip):
            return JSONResponse(
                {"error": f"access denied for {remote_ip}"}, status_code=403)
        if self._queue_db is None:
            return JSONResponse({"hours": 0, "windows": []})
        hours = _to_float(request.query_params.get("hours"), 24)
        hours = min(max(hours, 0.1), 168)
        windows = await asyncio.to_thread(self._queue_db.maintenance_windows, hours)
        return JSONResponse({"hours": hours, "windows": windows})

    # ----- handler: metrics -----

    def _metrics_payload(self, now: float) -> dict:
        return self._state.metrics_payload(now)

    async def handle_metrics(self, request: Request) -> Response:
        return JSONResponse(self._metrics_payload(time.monotonic()))

    # ----- handlers: fleet usage (Phase 1: proxy = fleet call-metrics authority) -----

    async def handle_fleet_activity(self, request: Request) -> Response:
        window_s = _clamp_window(request.query_params.get("window", "24h"), 86400)
        bin_s = _to_int(request.query_params.get("bin"), _bin_seconds_for(window_s))
        # §5c: heavy GROUP-BY over the whole-fleet completions table runs off the
        # event loop so it can't stall fleet LLM scheduling under a hot dashboard.
        data = await asyncio.to_thread(self._queue_db.fleet_activity, window_s, bin_s)
        return JSONResponse(data)

    async def handle_fleet_savings(self, request: Request) -> Response:
        since_q = request.query_params.get("since")
        today_start = float(since_q) if since_q and since_q.isdigit() else None
        data = await asyncio.to_thread(self._queue_db.savings_summary, today_start)
        return JSONResponse(data)

    async def handle_top_callers(self, request: Request) -> Response:
        window_s = _clamp_window(request.query_params.get("window", "1h"), 3600)
        per_endpoint = max(1, min(_to_int(request.query_params.get("per_endpoint"), 5), 20))
        data = await asyncio.to_thread(self._queue_db.top_callers, window_s, per_endpoint)
        return JSONResponse(data)

    async def handle_fleet_cache_stats(self, request: Request) -> Response:
        """Prefix-cache observability: per-model actual hit-rate + per-call_site
        misalignment offenders + rollup + trend + drift. Computed from the
        periodic snapshots in proxy_cache_stats (see _compute_cache_stats)."""
        window_s = _clamp_window(request.query_params.get("window", "7d"), 30 * 86400)
        snaps = await asyncio.to_thread(self._queue_db.cache_stats_snapshots, window_s)
        labels = cache_stats.chat_endpoint_labels()
        engines = {ep: cfg.backend_engine for ep, cfg in self._config.endpoints.items()}
        return JSONResponse(cache_stats.build_fleet_payload(snaps, labels, engines))

    async def handle_usage(self, request: Request) -> Response:
        dimension = request.query_params.get("by", "agent")
        if dimension not in ("agent", "call_site", "endpoint", "provider"):
            dimension = "agent"
        hours = min(_to_float(request.query_params.get("hours"), 24), 168)
        rows = await asyncio.to_thread(self._queue_db.usage_rollup, dimension, hours)
        return JSONResponse({
            "dimension": dimension,
            "hours": hours,
            "rows": rows,
        })

    async def handle_series(self, request: Request) -> Response:
        endpoint = request.query_params.get("endpoint", "")
        if not endpoint:
            return JSONResponse({"error": "endpoint query param required"}, status_code=400)
        window_s = _clamp_window(request.query_params.get("window", "24h"), 86400)
        bin_s = _to_int(request.query_params.get("bin"), _bin_seconds_for(window_s))
        data = await asyncio.to_thread(
            self._queue_db.endpoint_series, endpoint, window_s, bin_s)
        return JSONResponse(data)

    # ----- handler: calls ingest (non-LLM fleet calls) -----

    async def handle_calls_log(self, request: Request) -> Response:
        """Ingest a non-LLM service call (audio/imagegen/ocr/translate) that
        never traversed the scheduler, so the proxy is the single fleet
        call-metrics store. Internal/LAN — gated by the same ACL as admin.
        Best-effort: validates the minimum, records, fans out, returns ok."""
        remote_ip = request.client.host if request.client else "unknown"
        self._audit_admin_ip("/v1/calls/log", remote_ip)
        if not self._acl.is_admin(remote_ip):
            return JSONResponse({"error": f"access denied for {remote_ip}"}, status_code=403)
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse({"error": "invalid json"}, status_code=400)
        endpoint = str(body.get("endpoint") or body.get("provider") or body.get("unit") or "").strip()
        kind = str(body.get("kind") or "").strip() or "external"
        if not endpoint:
            return JSONResponse({"error": "endpoint/provider/unit required"}, status_code=400)
        if kind in _PAYLOAD_KIND.values() or kind == "llm":
            # The proxy is authoritative for LLM traffic via native recording;
            # refuse a pushed LLM call so the two paths can't double-count.
            return JSONResponse(
                {"error": f"kind {kind!r} is proxy-native; do not push LLM calls"},
                status_code=409)
        if normalize_endpoint(endpoint) in self._config.endpoints:
            # Same double-count guard keyed on the ENDPOINT: a push whose
            # endpoint normalizes to a proxy-native LLM class (rerank, embed,
            # chat, ...) duplicates a row the proxy already recorded natively -
            # regardless of the kind label it arrives under (the 2026-06-11
            # rerank double-count arrived as kind='external').
            return JSONResponse(
                {"error": f"endpoint {endpoint!r} is proxy-native "
                          f"({normalize_endpoint(endpoint)}); do not push LLM calls"},
                status_code=409)
        status = str(body.get("status") or ("ok" if body.get("success", True) else "error"))
        request_id = str(body.get("request_id") or f"ext-{uuid.uuid4().hex}")
        in_tok = _to_int(body.get("input_tokens"), 0)
        out_tok = _to_int(body.get("output_tokens"), 0)
        latency_ms = _to_float(body.get("latency_ms"), 0.0)
        duration_s = _to_float(body.get("duration_s"), latency_ms / 1000.0)
        agent_id = str(body.get("agent") or body.get("agent_id") or "unknown")
        call_site = str(body.get("call_site") or kind)
        try:
            self._queue_db.persist_external_call(
                request_id=request_id, agent_id=agent_id, endpoint=endpoint,
                call_site=call_site, kind=kind, input_tokens=in_tok,
                output_tokens=out_tok, duration_s=duration_s, status=status,
                caller_id=body.get("caller_id"),
            )
            self._sse.publish("call.completed", {
                "request_id": request_id, "agent": agent_id,
                "endpoint": normalize_endpoint(endpoint), "call_site": call_site,
                "kind": kind, "priority": "P2_POST_TURN",
                "input_tokens": in_tok, "output_tokens": out_tok,
                "duration_s": round(duration_s, 3), "queue_wait_ms": 0.0,
                "status": status, "ts": time.time(),
            })
        except Exception as exc:  # noqa: BLE001 — never fail the pusher
            logger.warning("calls/log ingest failed: %s", exc)
            return JSONResponse({"error": "ingest failed"}, status_code=500)
        return JSONResponse({"ok": True, "request_id": request_id})

    # ----- handler: SSE stream -----

    async def handle_stream(self, request: Request) -> Response:
        """Server-Sent Events: real-time `call.completed` + periodic `metrics`
        frames. Slow clients are dropped (the browser reconnects + re-syncs via
        the REST endpoints). Mirrors the host-telemetry /stream/v2 shape."""
        q = self._sse.subscribe()

        async def event_gen():
            try:
                yield f"event: hello\ndata: {json.dumps({'ts': time.time()})}\n\n".encode()
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event, data = await asyncio.wait_for(q.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield b": keepalive\n\n"
                        continue
                    if (event, data) == DROP_SENTINEL:
                        break
                    yield f"event: {event}\ndata: {data}\n\n".encode()
            except asyncio.CancelledError:
                pass
            finally:
                self._sse.unsubscribe(q)

        return StreamingResponse(event_gen(), media_type="text/event-stream", headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        })

    async def handle_cost_model(self, request: Request) -> Response:
        return JSONResponse(self._cost_model.snapshot())

    # ----- handler: history -----

    async def handle_history(self, request: Request) -> Response:
        hours = min(_to_float(request.query_params.get("hours"), 4), 168)
        bucket_minutes = max(1, min(_to_int(request.query_params.get("bucket_minutes"), 5), 60))
        buckets = await asyncio.to_thread(
            self._queue_db.history_buckets, hours, bucket_minutes)
        return JSONResponse({"buckets": buckets})

    # ----- handler: recent requests (for feed) -----

    async def handle_recent(self, request: Request) -> Response:
        limit = min(_to_int(request.query_params.get("limit"), 50), 200)
        rows = await asyncio.to_thread(self._queue_db.recent_requests, limit)
        return JSONResponse({"requests": rows})

    # ----- handler: live in-flight (what's executing right now) -----

    async def handle_inflight(self, request: Request) -> Response:
        """Live list of currently-executing requests + per-endpoint occupancy —
        the proxy is the dispatch authority, so this is the authoritative
        real-time "what's flowing through the LLM systems" view. Pure in-memory
        snapshot; also pushed via the SSE `inflight` frame for sub-second feel."""
        snap = self._scheduler.inflight_snapshot(time.monotonic())
        snap["ts"] = time.time()
        return JSONResponse(snap)

    # ----- handler: timeout advice -----

    async def handle_timeout_advice(self, request: Request) -> Response:
        """Recommended timeout (+ min/median/p95) for a model/tier/size,
        derived from measured end-to-end latency."""
        qp = request.query_params
        model = qp.get("model")
        if not model:
            return JSONResponse(
                {"error": "missing required param: model"}, status_code=400,
            )
        endpoint = normalize_endpoint(model)
        if endpoint not in self._config.endpoints:
            return JSONResponse(
                {"error": f"unknown model {model!r}",
                 "known": sorted(self._config.endpoints)},
                status_code=400,
            )
        pri_raw = qp.get("priority") or "P1_TURN_SUPPORT"
        # Accept both numeric strings ("1") and enum names ("P1_TURN_SUPPORT").
        pri_val: str | int = int(pri_raw) if pri_raw.lstrip("-").isdigit() else pri_raw
        try:
            priority = int(LLMPriority.coerce(pri_val))
        except (ValueError, KeyError):
            return JSONResponse(
                {"error": f"unknown priority {qp.get('priority')!r}"},
                status_code=400,
            )
        try:
            est_in = int(qp.get("est_in", "0"))
            est_out = int(qp.get("est_out", "0"))
        except ValueError:
            return JSONResponse(
                {"error": "est_in/est_out must be integers"}, status_code=400,
            )

        advice = self._timeout_model.advise(endpoint, priority, est_in, est_out)
        return JSONResponse({
            "model": endpoint,
            "priority": LLMPriority(priority).name,
            "est_in": est_in,
            "est_out": est_out,
            **advice,
        })

    async def handle_timeout_shadow_report(self, request: Request) -> Response:
        """Per-(model, tier) shadow summary: would-timeout rate and
        headroom reclaimed vs. the timeout actually applied."""
        hours = _to_float(request.query_params.get("hours"), 24)
        hours = min(max(hours, 0.1), 168)
        report = await asyncio.to_thread(self._queue_db.timeout_shadow_report, hours)
        return JSONResponse({"hours": hours, "report": report})

    async def handle_timeouts_report(self, request: Request) -> Response:
        """Calls that hit their timeout instead of finishing, per
        (model, tier, layer), with the load context when they gave up and
        how many fired below the recommended deadline (premature)."""
        hours = _to_float(request.query_params.get("hours"), 24)
        hours = min(max(hours, 0.1), 168)
        report = await asyncio.to_thread(self._queue_db.timeouts_report, hours)
        return JSONResponse({"hours": hours, **report})

    # ----- handler: health -----

    def _scheduler_loop_alive(self) -> bool:
        return self._health.scheduler_loop_alive()

    def _poller_alive(self) -> bool:
        return self._health.poller_alive()

    async def handle_health(self, request: Request) -> Response:
        ok = self._scheduler_loop_alive()
        poller_ok = self._poller_alive()
        unhealthy = [ep for ep, h in self._endpoint_health.items() if not h["healthy"]]
        # The proxy is UP iff its scheduler is alive (200). A dead BACKEND
        # degrades status but must NOT 503 the proxy — that would make a monitor
        # restart a healthy front door over a backend blip (alert-don't-kill).
        # A dead POLLER also only degrades: dispatch still works, but health
        # probing / alerting / budget persistence have stopped — surface it.
        status = "ok" if (ok and poller_ok and not unhealthy) else ("degraded" if ok else "down")
        return JSONResponse(
            {
                "status": status,
                "uptime_s": round(time.monotonic() - self._started_at, 0),
                "total_dispatched": self._scheduler.stats()["total_dispatched"],
                "endpoints": len(self._config.endpoints),
                "total_slots": self._config.total_fleet_slots,
                "unhealthy_endpoints": unhealthy,
                "scheduler_alive": ok,
                "poller_alive": poller_ok,
            },
            status_code=200 if ok else 503,
        )

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
