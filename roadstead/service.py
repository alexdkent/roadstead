"""ProxyService — the main orchestrator that wires scheduler, backend,
queue, coalescing, and observability into a running service.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from .acl import IPIdentityMap
from .agent_budget import BudgetManager
from .backend import BackendClientPool, BackendError, BackendResponse, BackendTimeout, BackendUnavailable
from .coalesce import DeterministicCache
from .grammar import GrammarResult, grammar_hash, normalize_and_validate
from .config import (
    CLASS_TO_ROLE,
    LLMPriority,
    PriorityBand,
    ProxyConfig,
    normalize_endpoint,
)

# Phase 1.3 — bounded in-proxy retry for transient backend failures (defer,
# don't drop). Only retry if at least this much of the caller's deadline remains
# after a short backoff, so a retry never starts work the caller will abandon.
_MIN_RETRY_BUDGET_S = 5.0
_RETRY_BACKOFF_S = 0.5

# Phase 2.1 — bounded in-flight drain on graceful shutdown. The BOUND is the fix
# for the historic SIGTERM hang (uvicorn waiting forever behind a slow request):
# drain up to this long, then force-cancel stragglers.
_DRAIN_DEADLINE_S = 30.0

# Phase 3.1 — single server-side submit-timeout default (was a 180.0 literal in
# three places). Client-side extend-only advice still applies on top per role/
# tier; this is only the floor when a caller supplies no timeout.
_DEFAULT_TIMEOUT_S = 180.0

# Phase 5C — time-to-first-token watchdog for streaming. Data (2026-05-31): the
# companion 80B sometimes produces ZERO tokens on a large-context synth and burns
# the FULL deadline (180s), uselessly holding a scarce slot. If no first token
# arrives within this bound, abort + free the slot + return a deferrable error so
# the caller defers instead of the slot being dead for minutes. Capped to the
# caller's own deadline so a legitimately short request isn't over-waited.
_STREAM_TTFT_DEADLINE_S = 30.0


class _ToolCallStreamSanitizer:
    """Per-stream sanitizer that makes vLLM ``qwen3_xml`` streaming tool-call
    deltas safe for strict OpenAI clients — the Vercel AI SDK /
    ``@ai-sdk/openai-compatible`` provider that opencode uses (Phase 5E, v2).

    THE BUG. When one turn produces MORE THAN ONE tool call (made far more
    likely by MTP speculative decoding, which the thinker runs), vLLM's
    ``qwen3_xml`` parser emits a junk "phantom" tool-call delta between the real
    ones: a fresh ``id``, ``"name": null`` and EMPTY arguments at an in-between
    index that NEVER receives a name — the real next call lands at the following
    index (observed live: real calls at index 0 and 2, phantom at 1). The AI
    SDK opens a tool-call slot for that index, finds ``function.name == null``
    (its check matches null AND undefined), and throws
    ``AI_InvalidResponseDataError: Expected 'function.name' to be a string``,
    aborting the whole turn. Upstream tracking: vLLM #39584 (open, parallel
    tool calls + spec-decode); client side: opencode #24137 / vercel/ai #6687.

    Note the v1 of this fix (strip the null ``name`` key from continuations) was
    aimed at the wrong frame: the AI SDK only validates ``function.name`` when
    OPENING a slot, never on a continuation, so stripping it there was a no-op
    against the real crash.

    The SAME parallel-call bug also corrupts the LAST call's arguments: it
    appends an extra trailing ``}`` (observed live: ``{"path": "/tmp"}}``),
    which is invalid JSON. Once the phantom no longer aborts the turn, the AI
    SDK reaches that argument and fails with a JSON parse error instead. So we
    also TRIM trailing junk: per slot we accumulate the emitted argument text
    and, the moment it parses as a complete JSON value (``raw_decode`` — which
    correctly ignores ``}`` inside string values), we emit exactly up to the end
    of that value and drop anything after. The AI SDK marks the call finished on
    the first valid parse and ignores later deltas, so this matches its model.

    THE FIX — the opencode-maintainer-recommended client behaviour, applied at
    the proxy so we neither patch the (custom) GB10 vLLM nor give up MTP
    throughput: never let a tool-call index reach the client until it has a
    STRING name, and never let its arguments exceed one complete JSON value. Per
    (choice, index) slot we track whether it's been OPENED with a name and the
    argument text emitted so far; we BUFFER the args of a not-yet-named slot and
    emit a proper opener once a name arrives; a slot that closes without ever
    being named (the phantom) is silently dropped. Indices are NOT renumbered —
    the AI SDK keys tool calls by ``id`` and uses the numeric index only as an
    accumulation slot, so the hole left by a dropped phantom is never touched.

    ``feed(data)`` takes one raw backend SSE ``data:`` payload and returns the
    payload to emit. PURE PASS-THROUGH (the ORIGINAL string object, no parse)
    for the >99% of chunks with no ``tool_calls``. Defensive: never raises — on
    any malformed/odd shape it returns the original bytes.
    """

    def __init__(self) -> None:
        # (choice_index, tool_index) -> {opened, id, type, buf, emitted, done}
        self._slots: dict = {}

    @staticmethod
    def _advance(emitted: str, frag: str):
        """Append ``frag`` to the already-emitted args ``emitted`` and return
        ``(delta_to_emit, total_emitted, done)``. If the combined text contains
        a complete JSON value, ``delta`` is only the part of ``frag`` up to the
        end of that value (trailing junk like an extra ``}`` is dropped) and
        ``done`` is True; otherwise the whole ``frag`` passes through."""
        whole = emitted + frag
        try:
            _, end = json.JSONDecoder().raw_decode(whole)
        except ValueError:
            return frag, whole, False          # not a complete value yet
        total = whole[:end]
        return total[len(emitted):], total, True

    def feed(self, data: str) -> str:
        if '"tool_calls"' not in data:
            return data
        try:
            obj = json.loads(data)
            if not isinstance(obj, dict):
                return data
            changed = False
            for ch in (obj.get("choices") or []):
                if not isinstance(ch, dict):
                    continue
                delta = ch.get("delta")
                if not isinstance(delta, dict):
                    continue
                tcs = delta.get("tool_calls")
                if not isinstance(tcs, list):
                    continue
                ci = ch.get("index", 0)
                kept: list = []
                for tc in tcs:
                    if not isinstance(tc, dict):
                        kept.append(tc)
                        continue
                    changed = True  # a tool_calls chunk is always re-serialized
                    idx = tc.get("index")
                    st = self._slots.get((ci, idx))
                    if st is None:
                        st = {"opened": False, "id": None, "type": None,
                              "buf": "", "emitted": "", "done": False}
                        self._slots[(ci, idx)] = st
                    if st["id"] is None and isinstance(tc.get("id"), str):
                        st["id"] = tc["id"]
                    if st["type"] is None and isinstance(tc.get("type"), str):
                        st["type"] = tc["type"]
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    rn = fn.get("name")
                    name = rn if (isinstance(rn, str) and rn != "") else None
                    ra = fn.get("arguments")
                    args = ra if isinstance(ra, str) else ""
                    if st["done"]:
                        # Arguments already a complete JSON value — drop trailing
                        # junk (the AI SDK ignores it too, via hasFinished).
                        continue
                    if st["opened"]:
                        # Continuation — emit only the argument increment, trimmed
                        # at the end of the first complete JSON value.
                        if args:
                            d, st["emitted"], st["done"] = self._advance(st["emitted"], args)
                            if d:
                                kept.append({"index": idx,
                                             "function": {"arguments": d}})
                        continue
                    if name is not None:
                        # Open the slot, merging any buffered pre-name args and
                        # trimming if they already form a complete value.
                        oargs, st["emitted"], st["done"] = self._advance("", st["buf"] + args)
                        kept.append({
                            "index": idx,
                            "id": st["id"] or tc.get("id"),
                            "type": st["type"] or "function",
                            "function": {"name": name, "arguments": oargs},
                        })
                        st["opened"] = True
                        st["buf"] = ""
                    elif args:
                        # Name-less and not yet opened: hold args until a name
                        # arrives (a pure phantom has none → nothing held, and
                        # the slot is dropped when the stream ends).
                        st["buf"] += args
                    # else: name-less, no args → phantom; emit nothing.
                if changed:
                    if kept:
                        delta["tool_calls"] = kept
                    elif "tool_calls" in delta:
                        del delta["tool_calls"]
            return json.dumps(obj) if changed else data
        except Exception:  # noqa: BLE001 — a sanitizer bug must not break the stream
            return data
from .cost_model import CostModel, estimate_input_tokens
from .timeout_model import TimeoutModel
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

logger = logging.getLogger(__name__)


class ProxyService:
    """Main proxy service that coordinates all components."""

    def __init__(self, config: ProxyConfig) -> None:
        self._config = config

        # Core components
        self._cost_model = CostModel()
        self._timeout_model = TimeoutModel(
            margin=config.timeout_advice_margin,
            window_s=config.timeout_advice_window_s,
            min_samples=config.timeout_advice_min_samples,
        )
        self._budget_mgr = BudgetManager(starvation_timeout_s=config.starvation_timeout_s)
        self._scheduler = Scheduler(config, self._cost_model, self._budget_mgr)
        self._backend = BackendClientPool()
        self._queue_db = PersistentQueue(config.queue_db_path or None)

        # Deterministic response cache (temperature=0).
        self._cache = DeterministicCache()

        # Grammar authority: cache of normalize+validate results keyed by
        # grammar hash, + a set of hashes we've already alerted on so each
        # bad grammar logs loudly once (not per request).
        self._grammar_cache: dict[str, "GrammarResult"] = {}
        self._grammar_alerted: set[str] = set()

        # Observability
        self._metrics = RollingMetrics(window_s=300.0)
        self._request_logger = RequestLogger(config.request_log_path or None)
        self._acl = IPIdentityMap.from_env()

        # Async plumbing
        self._dispatch_event = asyncio.Event()
        self._draining = asyncio.Event()  # set during shutdown drain (Phase 2.1)
        self._pending_futures: dict[str, asyncio.Future] = {}
        self._pending_streams: dict[str, asyncio.Queue] = {}
        # Dedupe set so a single request that races across two timeout
        # layers (e.g. admission expiry + client-wait) is logged once.
        self._timed_out_ids: set[str] = set()
        # In-flight dispatch tasks keyed by request_id (Phase 1.5 / 2.1). Lets
        # the drain path await them on shutdown; the slot-leak fix is the
        # deadline-bound backend call in _execute_sync/_execute_streaming.
        self._inflight_tasks: dict[str, asyncio.Task] = {}
        # Per-endpoint circuit-breaker health (Phase 1.2). An endpoint flips
        # unhealthy only after consecutive capacity-probe failures CONFIRMED by a
        # failed /health probe — latency/saturation never flips it
        # (alert-don't-kill). While unhealthy the scheduler defers its queue.
        self._endpoint_health: dict[str, dict] = {
            ep: {"healthy": True, "consecutive_failures": 0, "unhealthy_since": None}
            for ep in config.endpoints
        }
        self._health_fail_threshold = 3
        # Phase 5F — operator drain: endpoints an operator has explicitly PAUSED
        # for maintenance (e.g. a vLLM restart to change --max-model-len). A
        # paused endpoint reads as unhealthy (→ background defers, interactive
        # fast-fails deferrably) so NO request hits the backend while it's down,
        # WITHOUT the ~30s auto-circuit-trip lag. Distinct from the auto-circuit
        # so a planned drain never fires the endpoint_paused ERROR alert. The
        # poller skips paused endpoints; /resume hands them back to the poller.
        self._paused_endpoints: set[str] = set()
        self._transient_retry_max = 1
        # Retention sweep cadence (Phase 2.3) — monotonic ts of the last DB trim.
        self._last_cleanup_at = 0.0
        # DRR-balance persistence cadence (Phase 3.4).
        self._last_budget_save_at = 0.0
        # Load-shed threshold (Phase 2.4): per-(endpoint, band) queue depth at
        # which NON-interactive submits are shed with 429 + Retry-After.
        self._shed_depth = 50
        # Alerting (Phase 2.5) — current triggered alerts (exposed on /v1/status)
        # + the set already logged, so a sustained condition logs once not every
        # poll tick.
        self._alerts: list[dict] = []
        self._alert_logged: set = set()
        # Phase 5B observability counters (exposed on /v1/status).
        self._slot_leak_reclaimed = 0       # streaming dispatches cancelled on
                                            # consumer-disconnect → slot freed
        self._drain_straggler_cancelled = 0  # in-flight tasks cancelled at the
                                            # shutdown drain deadline
        self._scheduler_task: asyncio.Task | None = None
        self._poller_task: asyncio.Task | None = None
        self._started_at = time.monotonic()

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

        # Bootstrap cost model from recent completion history
        self._bootstrap_cost_model()

        # Bootstrap timeout-advice model from the same history
        self._bootstrap_timeout_model()

        # Phase 2.2: all startup recovery + bootstrap reads/writes are done
        # synchronously above while single-threaded; from here, route DB writes
        # to a dedicated writer thread so synchronous SQLite I/O never blocks the
        # event loop that schedules the whole fleet.
        self._queue_db.start_async_writer()

        # Phase 2.3: initial retention sweep (then daily in the poller) so the
        # completion corpus + timeout tables don't grow unbounded.
        self._queue_db.cleanup_old_completions()
        self._last_cleanup_at = time.monotonic()

        # Start background loops
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._poller_task = asyncio.create_task(self._capacity_poller_loop())

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
        """Pick the routing endpoint, honoring the requested ``model`` over
        the caller's client role (OpenAI-consistent). The client role is the
        default when no model is given.

        Only overrides for chat completions when the payload's ``model``
        maps to a *known* endpoint that differs from the submit endpoint —
        so embeddings (no model) and rerank (model='bge', not an endpoint)
        fall through untouched. Logs every reconciliation loudly so a
        mis-wired client (role != requested model) stays visible.
        """
        submit_ep = normalize_endpoint(body.get("endpoint", "chat"))
        if body.get("payload_type", "chat_completion") != "chat_completion":
            return submit_ep
        model = (body.get("payload") or {}).get("model")
        if not model:
            return submit_ep
        model_ep = normalize_endpoint(str(model))
        if model_ep in self._config.endpoints and model_ep != submit_ep:
            logger.warning(
                "route reconcile: client endpoint=%s but model=%s -> routing to %s "
                "(caller=%s call_site=%s)",
                submit_ep, model, model_ep,
                body.get("caller_id"), body.get("call_site"),
            )
            return model_ep
        return submit_ep

    async def handle_submit(
        self, body: dict, request: Request, *, openai: bool = False,
    ) -> Response:
        # ``openai=True`` (set only by the /v1/chat/completions front door)
        # varies ONLY the response serialization: the bare OpenAI
        # chat.completion / chat.completion.chunk + [DONE] stream, instead of
        # the internal submit envelope. The enqueue / scheduler / grammar /
        # cache / DRR / telemetry path is identical. Default False keeps every
        # agent's /v1/submit response byte-identical.
        if self._draining.is_set():
            # Phase 2.1: refuse new work while draining for shutdown so it defers
            # to the (about-to-restart) next instance instead of being dropped.
            # Phase 5C: include the "backpressure" marker so the body is
            # classified deferrable (is_deferrable_llm_error) by BOTH the sync
            # and streaming clients — without it, a streaming turn caught mid-
            # SIGTERM surfaced a hard error instead of deferring cleanly.
            err = "proxy draining for shutdown — backpressure"
            if openai:
                return self._openai_error(err, "backpressure", 503)
            return JSONResponse({"status": "error", "error": err}, status_code=503)

        now = time.monotonic()

        req = QueuedRequest.create(
            agent_id=body.get("agent_id", "unknown"),
            endpoint=self._resolve_endpoint(body),
            priority=body.get("priority"),
            call_site=body.get("call_site", "unknown"),
            payload_type=body.get("payload_type", "chat_completion"),
            payload=body.get("payload", {}),
            timeout_s=float(body.get("timeout_s", _DEFAULT_TIMEOUT_S)),
            session_id=body.get("session_id"),
            turn_id=body.get("turn_id"),
            caller_id=body.get("caller_id"),
            request_id=body.get("request_id"),
            now=now,
        )

        # Grammar authority: validate + safe-normalize any GBNF grammar
        # BEFORE enqueue. Fail loud on an invalid grammar rather than
        # dispatching it (llama-server would silently run unconstrained).
        if req.payload_type == "chat_completion":
            grammar_err = self._process_grammar(req)
            if grammar_err is not None:
                if openai:
                    return self._openai_error(
                        grammar_err.get("detail", "invalid grammar"),
                        "invalid_request_error", 422,
                    )
                return JSONResponse(
                    {"status": "error", "request_id": req.request_id, **grammar_err},
                    status_code=422,
                )

        # Check deterministic cache
        cache_key = self._cache.cache_key(req.endpoint, req.payload)
        if cache_key:
            cached = self._cache.get(cache_key)
            if cached:
                # Phase 4.1: count cache hits in metrics. They bypass dispatch,
                # so without this they're invisible in /v1/metrics and real
                # traffic is undercounted (the cache's own hit_rate aside).
                self._metrics.record(MetricsSample(
                    timestamp=now, endpoint=req.endpoint, agent_id=req.agent_id,
                    priority=req.priority.name, queue_wait_ms=0.0,
                    backend_latency_ms=0.0, status="ok", slot_seconds=0.0))
                # OpenAI consumers get the bare cached completion; internal
                # consumers get the submit envelope (unchanged).
                if openai:
                    return JSONResponse(cached)
                return JSONResponse({
                    "status": "ok",
                    "request_id": req.request_id,
                    "queue_wait_ms": 0,
                    "backend_latency_ms": 0,
                    "estimated_cost_ss": 0,
                    "response": cached,
                    "cache_hit": True,
                })

        # Circuit breaker (Phase 1.2): when the backend is marked unhealthy,
        # fast-fail interactive/foreground submits immediately with a DEFERRABLE
        # error instead of queuing them to wait out their full deadline; let
        # background work queue so it defers until the backend recovers. (Cache
        # hits above are served regardless — they don't need the backend.)
        if not self._endpoint_healthy(req.endpoint) and req.band != PriorityBand.BACKGROUND:
            # Phase 5F: distinguish an operator drain (planned) from an
            # auto-circuit trip (backend unreachable). Both are DEFERRABLE
            # ("circuit open" / "backpressure" are is_deferrable_llm_error
            # markers) so the caller retries; the wording just aids triage.
            if normalize_endpoint(req.endpoint) in self._paused_endpoints:
                err = f"backend {req.endpoint} paused for maintenance (drain) — backpressure"
            else:
                err = f"backend {req.endpoint} unavailable (circuit open)"
            if openai:
                return self._openai_error(err, "backend_unavailable", 503)
            return JSONResponse(
                {"status": "error", "request_id": req.request_id, "error": err},
                status_code=503,
            )

        # Load-shed / backpressure (Phase 2.4): under sustained saturation, shed
        # NON-interactive work with 429 + Retry-After so callers defer instead of
        # all queuing until their deadlines and 504ing together. Interactive is
        # never shed.
        if req.band != PriorityBand.INTERACTIVE:
            snap = self._scheduler.endpoint_snapshot(req.endpoint)
            band_key = req.band.name.lower()
            if snap.get("queue_by_band", {}).get(band_key, 0) >= self._shed_depth:
                err = f"backpressure: {req.endpoint} {band_key} queue saturated"
                retry_after = self._retry_after_s(req.endpoint)
                if openai:
                    resp = self._openai_error(err, "backpressure", 429)
                else:
                    resp = JSONResponse(
                        {"status": "error", "request_id": req.request_id, "error": err},
                        status_code=429)
                resp.headers["Retry-After"] = str(retry_after)
                return resp

        # Streaming vs non-streaming
        if req.stream:
            return await self._handle_streaming_submit(req, openai=openai)
        else:
            return await self._handle_sync_submit(req, cache_key, openai=openai)

    def _extract_grammar(self, payload: dict) -> tuple[str | None, str | None]:
        """Return (grammar_string, location) where location is 'top' or
        'extra_body', or (None, None) if no grammar present."""
        g = payload.get("grammar")
        if isinstance(g, str) and g.strip():
            return g, "top"
        eb = payload.get("extra_body")
        if isinstance(eb, dict):
            g = eb.get("grammar")
            if isinstance(g, str) and g.strip():
                return g, "extra_body"
        return None, None

    def _process_grammar(self, req: QueuedRequest) -> dict | None:
        """Validate + safe-normalize the request's grammar in place.

        Returns None on success (req.payload updated with the normalized
        grammar). Returns an error payload dict on failure — the caller
        must fail loud rather than dispatch. Results are cached by grammar
        hash; each invalid grammar is logged loudly once.
        """
        grammar, location = self._extract_grammar(req.payload)
        if grammar is None:
            return None

        h = grammar_hash(grammar)
        result = self._grammar_cache.get(h)
        if result is None:
            result = normalize_and_validate(grammar)
            self._grammar_cache[h] = result

        if not result.ok:
            if h not in self._grammar_alerted:
                self._grammar_alerted.add(h)
                logger.error(
                    "GRAMMAR INVALID — failing loud (call_site=%s endpoint=%s): %s",
                    req.call_site, req.endpoint, result.error_payload()["detail"],
                )
            return result.error_payload()

        # Write the normalized grammar back where it came from.
        if result.normalized:
            if location == "top":
                req.payload["grammar"] = result.grammar
            else:
                req.payload["extra_body"]["grammar"] = result.grammar
        return None

    async def _handle_sync_submit(
        self, req: QueuedRequest, cache_key: str | None, *, openai: bool = False,
    ) -> Response:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_futures[req.request_id] = future

        self._scheduler.enqueue(req)
        self._queue_db.persist_enqueue(req)
        self._dispatch_event.set()

        try:
            result = await asyncio.wait_for(
                future, timeout=req.timeout_s,
            )
        except asyncio.TimeoutError:
            self._scheduler.cancel(req.request_id)
            self._queue_db.persist_expire(req.request_id)
            self._pending_futures.pop(req.request_id, None)
            # The caller's deadline fired — work may still be in flight.
            self._record_timeout_event(req, layer="client_wait", elapsed_s=req.timeout_s)
            if openai:
                return self._openai_error(
                    f"proxy timeout after {req.timeout_s:.0f}s", "proxy_timeout", 504,
                )
            return JSONResponse(
                {"error": "timeout", "request_id": req.request_id},
                status_code=504,
            )
        finally:
            self._pending_futures.pop(req.request_id, None)

        # Admission timeout (scheduler callback) resolves the future with a
        # timeout result — already logged there; surface the same 504.
        if result.get("status") == "timeout":
            if openai:
                return self._openai_error(
                    f"proxy timeout after {req.timeout_s:.0f}s", "proxy_timeout", 504,
                )
            return JSONResponse(
                {"error": "timeout", "request_id": req.request_id},
                status_code=504,
            )

        # Cache if deterministic
        if cache_key and result.get("status") == "ok":
            self._cache.put(cache_key, result.get("response", {}))

        # OpenAI consumers get the bare chat.completion (or an OpenAI-shaped
        # error); internal consumers get the submit envelope (unchanged).
        if openai:
            if result.get("status") == "ok":
                return JSONResponse(result.get("response", {}))
            return self._openai_error(
                result.get("error", "backend error"), "backend_error", 502,
            )

        status_code = 200 if result.get("status") == "ok" else 502
        return JSONResponse(result, status_code=status_code)

    async def _handle_streaming_submit(
        self, req: QueuedRequest, *, openai: bool = False,
    ) -> Response:
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._pending_streams[req.request_id] = queue

        self._scheduler.enqueue(req)
        self._queue_db.persist_enqueue(req)
        self._dispatch_event.set()

        async def stream_generator():
            # Per-request tool-call stream sanitizer (OpenAI front door only;
            # stateful across this one stream). See _ToolCallStreamSanitizer.
            toolcall_sanitizer = _ToolCallStreamSanitizer()
            # Internal envelope consumers get the queued marker; OpenAI
            # consumers (goose-cli) get ONLY chat.completion.chunk frames, so
            # the queued/admitted markers are dropped — an OpenAI client chokes
            # parsing them.
            if not openai:
                yield f"data: {json.dumps({'type': 'queued', 'request_id': req.request_id})}\n\n"

            try:
                while True:
                    event = await asyncio.wait_for(
                        queue.get(), timeout=req.timeout_s,
                    )
                    if openai:
                        etype = event.get("type")
                        if etype == "chunk":
                            # event["data"] is the backend's raw OpenAI
                            # chat.completion.chunk line. Plain content chunks
                            # pass through byte-identical; streaming tool-call
                            # deltas are sanitized so strict clients (the Vercel
                            # AI SDK that opencode uses) don't choke on vLLM's
                            # qwen3_xml phantom/name-less openers. See
                            # _ToolCallStreamSanitizer.
                            yield f"data: {toolcall_sanitizer.feed(event['data'])}\n\n"
                            continue
                        if etype == "done":
                            yield "data: [DONE]\n\n"
                            break
                        if etype == "error":
                            yield (
                                "data: "
                                + json.dumps({"error": {
                                    "message": event.get("error", "stream error"),
                                    "type": "proxy_error",
                                }})
                                + "\n\n"
                            )
                            break
                        # queued / admitted / anything else → not an OpenAI frame.
                        continue
                    # Internal envelope path (unchanged): re-emit every event.
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") in ("done", "error"):
                        break
            except asyncio.TimeoutError:
                if openai:
                    yield (
                        "data: "
                        + json.dumps({"error": {
                            "message": f"proxy stream timeout after {req.timeout_s:.0f}s",
                            "type": "proxy_timeout",
                        }})
                        + "\n\n"
                    )
                else:
                    yield f"data: {json.dumps({'type': 'error', 'error': 'timeout'})}\n\n"
                self._record_timeout_event(req, layer="stream", elapsed_s=req.timeout_s)
            finally:
                # Phase 5B.1: the SSE consumer is gone (client disconnect, our
                # own timeout, or normal completion). Cancel the producer
                # dispatch task if it's still running — otherwise a producer
                # blocked on a full stream_q.put (maxsize=256, consumer no longer
                # draining) wedges forever, holding the scheduler slot until the
                # proxy restarts. That leak cascaded all 4 companion slots into a
                # full endpoint jam (2026-05-31). The CancelledError branch in
                # _execute_dispatch records the completion → frees the slot.
                self._pending_streams.pop(req.request_id, None)
                producer = self._inflight_tasks.get(req.request_id)
                if producer is not None and not producer.done():
                    producer.cancel()

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ----- handler: OpenAI compat -----

    @staticmethod
    def _openai_error(message: str, err_type: str, status_code: int) -> JSONResponse:
        """OpenAI-shaped error envelope for the /v1/chat/completions front door
        (goose-cli + any OpenAI client expects ``{"error": {...}}``)."""
        return JSONResponse(
            {"error": {"message": str(message), "type": err_type}},
            status_code=status_code,
        )

    async def handle_openai_chat(self, body: dict, request: Request) -> Response:
        remote_ip = request.client.host if request.client else "unknown"
        identity = self._acl.identify(remote_ip)
        if not identity:
            # Phase 5D: OpenAI-shaped 403 (was a bare {"error": "<str>"} that a
            # strict OpenAI client crashes on doing resp.error.message).
            return self._openai_error(
                f"access denied for {remote_ip}", "access_denied", 403)
        agent_id, default_priority = identity
        model = body.get("model", "qwen-analyst")
        # Phase 5D: validate the model maps to a known endpoint BEFORE enqueue.
        # Otherwise an unknown model burns a scheduler slot + DRR charge and
        # fails late with a confusing 502; OpenAI clients expect 404/model_not_found.
        if normalize_endpoint(str(model)) not in self._config.endpoints:
            return self._openai_error(
                f"unknown model {model!r}", "model_not_found", 404)

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
                f"access denied for {remote_ip}", "access_denied", 403)
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
            endpoints[ep_name] = snap

        agents = {
            b["agent_id"]: b
            for b in self._budget_mgr.snapshot()
        }

        return JSONResponse({
            "endpoints": endpoints,
            "agents": agents,
            "alerts": self._alerts,  # Phase 2.5 — health-verifier/log_scan surface
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
            },
            # Phase 5F — endpoints an operator has drained for maintenance.
            "paused_endpoints": sorted(self._paused_endpoints),
        })

    # ----- handler: admin endpoint pause/resume (Phase 5F operator drain) -----

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
        if not self._acl.identify(remote_ip):
            return JSONResponse(
                {"error": f"access denied for {remote_ip}"}, status_code=403)
        ep = normalize_endpoint(endpoint)
        if ep not in self._config.endpoints:
            return JSONResponse(
                {"error": f"unknown endpoint {endpoint!r}"}, status_code=404)
        if pause:
            self._paused_endpoints.add(ep)
            # Release any already-queued interactive/foreground immediately with
            # a deferrable error (don't make them wait out their deadline).
            self._fast_fail_interactive(ep)
            logger.warning(
                "endpoint %s PAUSED by operator (%s) — background defers, "
                "interactive fast-fails; backend safe to restart", ep, remote_ip)
        else:
            self._paused_endpoints.discard(ep)
            self._dispatch_event.set()  # nudge the scheduler to drain deferred work
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

    # ----- handler: metrics -----

    async def handle_metrics(self, request: Request) -> Response:
        now = time.monotonic()
        return JSONResponse({
            "per_endpoint": {
                ep: {
                    "requests": self._metrics.count(endpoint=ep, now=now),
                    "timeouts": self._metrics.count(endpoint=ep, status="timeout", now=now),
                    "p50_wait_ms": self._metrics.percentile("queue_wait_ms", 50, endpoint=ep, now=now),
                    "p95_wait_ms": self._metrics.percentile("queue_wait_ms", 95, endpoint=ep, now=now),
                    "p50_backend_ms": self._metrics.percentile("backend_latency_ms", 50, endpoint=ep, now=now),
                    "p95_backend_ms": self._metrics.percentile("backend_latency_ms", 95, endpoint=ep, now=now),
                    "slot_seconds_consumed": round(self._metrics.slot_seconds_consumed(ep, now), 1),
                }
                for ep in self._config.endpoints
            },
            "per_agent": {
                aid: round(ss, 1)
                for aid, ss in self._metrics.per_agent_consumed(now).items()
            },
        })

    async def handle_cost_model(self, request: Request) -> Response:
        return JSONResponse(self._cost_model.snapshot())

    # ----- handler: history -----

    async def handle_history(self, request: Request) -> Response:
        hours = float(request.query_params.get("hours", "4"))
        bucket_minutes = int(request.query_params.get("bucket_minutes", "5"))
        hours = min(hours, 168)
        bucket_minutes = max(1, min(bucket_minutes, 60))
        buckets = self._queue_db.history_buckets(hours, bucket_minutes)
        return JSONResponse({"buckets": buckets})

    # ----- handler: recent requests (for feed) -----

    async def handle_recent(self, request: Request) -> Response:
        limit = int(request.query_params.get("limit", "50"))
        limit = min(limit, 200)
        rows = self._queue_db.recent_requests(limit)
        return JSONResponse({"requests": rows})

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
        hours = float(request.query_params.get("hours", "24"))
        hours = min(max(hours, 0.1), 168)
        report = self._queue_db.timeout_shadow_report(hours)
        return JSONResponse({"hours": hours, "report": report})

    async def handle_timeouts_report(self, request: Request) -> Response:
        """Calls that hit their timeout instead of finishing, per
        (model, tier, layer), with the load context when they gave up and
        how many fired below the recommended deadline (premature)."""
        hours = float(request.query_params.get("hours", "24"))
        hours = min(max(hours, 0.1), 168)
        report = self._queue_db.timeouts_report(hours)
        return JSONResponse({"hours": hours, **report})

    # ----- handler: health -----

    async def handle_health(self, request: Request) -> Response:
        ok = self._scheduler_task is not None and not self._scheduler_task.done()
        unhealthy = [ep for ep, h in self._endpoint_health.items() if not h["healthy"]]
        # The proxy is UP iff its scheduler is alive (200). A dead BACKEND
        # degrades status but must NOT 503 the proxy — that would make a monitor
        # restart a healthy front door over a backend blip (alert-don't-kill).
        status = "ok" if (ok and not unhealthy) else ("degraded" if ok else "down")
        return JSONResponse(
            {
                "status": status,
                "uptime_s": round(time.monotonic() - self._started_at, 0),
                "total_dispatched": self._scheduler.stats()["total_dispatched"],
                "endpoints": len(self._config.endpoints),
                "total_slots": self._config.total_fleet_slots,
                "unhealthy_endpoints": unhealthy,
            },
            status_code=200 if ok else 503,
        )

    # ----- scheduler loop -----

    async def _scheduler_loop(self) -> None:
        """Main scheduling loop — runs dispatch on every event or interval."""
        while True:
            try:
                await asyncio.wait_for(
                    self._dispatch_event.wait(),
                    timeout=self._config.drr_tick_interval_s * 10,
                )
            except asyncio.TimeoutError:
                pass
            self._dispatch_event.clear()

            now = time.monotonic()
            decisions = self._scheduler.tick(now)

            for decision in decisions:
                rid = decision.request.request_id
                task = asyncio.create_task(self._execute_dispatch(decision))
                self._inflight_tasks[rid] = task
                task.add_done_callback(
                    lambda t, _rid=rid: self._inflight_tasks.pop(_rid, None)
                )

    async def _execute_dispatch(self, decision: DispatchDecision) -> None:
        """Execute a dispatch decision: call the backend and resolve the
        caller's future/stream."""
        req = decision.request
        ep_cfg = self._config.endpoints.get(req.endpoint)
        if not ep_cfg:
            self._resolve_error(req, f"unknown endpoint {req.endpoint}")
            return

        self._queue_db.persist_dispatch(req.request_id)

        t0 = time.monotonic()
        try:
            if req.stream:
                await self._execute_streaming(req, ep_cfg, decision)
            else:
                await self._execute_sync(req, ep_cfg, decision)
        except asyncio.CancelledError:
            # Phase 5B.2: this dispatch task was cancelled. Two callers cancel
            # it: (a) a streaming consumer disconnected and the SSE generator's
            # finally cancels us (Phase 5B.1 — otherwise a producer wedged on a
            # full stream_q.put would hold the scheduler slot forever — the live
            # companion-jam bug), or (b) the shutdown drain deadline fired.
            # Either way we MUST record the completion so the scheduler frees the
            # slot, then re-raise so cancellation propagates and the task ends.
            # CancelledError is a BaseException, so the `except Exception` below
            # would NOT catch it — without this branch the slot leaks.
            duration = time.monotonic() - t0
            self._slot_leak_reclaimed += 1
            logger.info(
                "dispatch %s cancelled (consumer gone / drain) after %.1fs — "
                "reclaiming slot", req.request_id, duration,
            )
            self._resolve_error(
                req, "llm proxy stream cancelled (consumer disconnected)")
            self._record_completion(req, decision, duration, 0, 0, "cancelled")
            raise
        except Exception as exc:
            duration = time.monotonic() - t0
            logger.error(
                "dispatch %s failed: %s", req.request_id, exc,
            )
            self._resolve_error(req, str(exc))
            self._record_completion(req, decision, duration, 0, 0, "error")

    async def _execute_sync(
        self,
        req: QueuedRequest,
        ep_cfg: EndpointConfig,
        decision: DispatchDecision,
    ) -> None:
        attempts = 0
        while True:
            attempts += 1
            # Phase 1.5 slot-leak fix: bound each backend attempt to the
            # caller's ABSOLUTE deadline (timeout_deadline), not a fresh full
            # timeout_s. The client started its clock at enqueue and the backend
            # at dispatch (after queue_wait), so a fresh timeout_s here would let
            # an abandoned call outlive its caller and hold the slot for
            # queue_wait + timeout_s. The remaining-deadline bound frees the slot
            # at the SLA instead.
            remaining = req.timeout_deadline - time.monotonic()
            if remaining <= 0:
                # Phase 5C: the deadline passed in-queue. For BACKGROUND work,
                # surface a DEFERRABLE error ("backpressure") so the caller
                # re-queues it on a later pass instead of dead-lettering work it
                # never got to run — the "stop abandoning work at the deadline"
                # goal. Interactive/foreground still hard-fail (the turn is over).
                is_bg = int(req.priority) >= int(LLMPriority.P3_INGESTION)
                msg = (
                    f"backend {ep_cfg.role} deadline exceeded in queue — backpressure"
                    if is_bg else
                    f"backend {ep_cfg.role} deadline exceeded before dispatch"
                )
                self._resolve_error(req, msg)
                self._record_completion(req, decision, 0.0, 0, 0, "timeout")
                self._record_timeout_event(
                    req, layer="backend", elapsed_s=0.0,
                    queue_wait_ms=decision.queue_wait_ms, emit_metrics_and_log=False,
                )
                return

            t0 = time.monotonic()
            try:
                resp = await self._backend.call(
                    ep_cfg, req.payload, req.payload_type,
                    req.request_id, timeout_s=max(1.0, remaining),
                )
            except BackendTimeout as exc:
                duration = time.monotonic() - t0
                self._resolve_error(req, str(exc))
                self._record_completion(req, decision, duration, 0, 0, "timeout")
                self._record_timeout_event(
                    req, layer="backend", elapsed_s=duration,
                    queue_wait_ms=decision.queue_wait_ms, emit_metrics_and_log=False,
                )
                return
            except (BackendUnavailable, BackendError) as exc:
                duration = time.monotonic() - t0
                # Phase 1.3 defer-don't-drop: a transient infra failure (backend
                # unreachable/503, or an empty completion — a backend hiccup, not
                # a content error) RETRIES within the remaining deadline rather
                # than burning the call. Deterministic 4xx/other 5xx surface.
                if (
                    self._is_transient_backend_error(exc)
                    and attempts <= self._transient_retry_max
                    and (req.timeout_deadline - time.monotonic()) > _MIN_RETRY_BUDGET_S
                    and self._endpoint_healthy(req.endpoint)
                ):
                    logger.warning(
                        "transient backend error on %s (attempt %d) — retrying: %s",
                        ep_cfg.role, attempts, exc,
                    )
                    await asyncio.sleep(_RETRY_BACKOFF_S)
                    continue
                self._resolve_error(req, str(exc))
                self._record_completion(req, decision, duration, 0, 0, "error")
                return

            duration = time.monotonic() - t0

            # Phase 1.1 truncation integrity. finish_reason=length means the
            # backend hit max_tokens mid-output. For a STRUCTURED request
            # (grammar / response_format / structured_outputs) the body is almost
            # certainly broken/unparseable JSON — fail loud with a DEFERRABLE
            # error so the caller re-chunks instead of recording garbage, and
            # never cache it (status != ok). Free-form truncation is benign.
            if resp.finish_reason == "length" and self._request_is_structured(req):
                self._resolve_error(
                    req,
                    f"backend {ep_cfg.role} truncated structured output "
                    f"(finish_reason=length, output_tokens={resp.output_tokens})",
                )
                self._record_completion(
                    req, decision, duration,
                    resp.input_tokens, resp.output_tokens, "truncated",
                    response_body=resp.body if req.payload_type == "chat_completion" else None,
                    finish_reason=resp.finish_reason,
                )
                return

            result = {
                "request_id": req.request_id,
                "queue_wait_ms": round(decision.queue_wait_ms, 1),
                "backend_latency_ms": round(duration * 1000, 1),
                "estimated_cost_ss": round(req.estimated_cost_ss, 3),
                "response": resp.body,
                "status": "ok",
            }

            future = self._pending_futures.get(req.request_id)
            if future and not future.done():
                future.set_result(result)

            capture_response = resp.body if req.payload_type == "chat_completion" else None
            self._record_completion(
                req, decision, duration,
                resp.input_tokens, resp.output_tokens, "ok",
                response_body=capture_response, finish_reason=resp.finish_reason,
            )

            # Shadow backend A/B: fire-and-forget to the shadow if configured
            if ep_cfg.shadow_host and ep_cfg.shadow_port:
                asyncio.create_task(self._execute_shadow(
                    req, ep_cfg, decision, resp,
                ))
            return

    async def _execute_shadow(
        self,
        req: QueuedRequest,
        ep_cfg: EndpointConfig,
        decision: DispatchDecision,
        primary_resp: BackendResponse,
    ) -> None:
        """Send the same request to the shadow backend and record the
        comparison result. Never affects the primary caller."""
        shadow_resp = await self._backend.call_shadow(
            ep_cfg.shadow_host, ep_cfg.shadow_port,
            req.payload, req.payload_type,
            req.request_id, timeout_s=req.timeout_s,
        )
        if shadow_resp is None:
            logger.debug("shadow dispatch %s failed", req.request_id)
            return

        self._queue_db.persist_complete(
            f"shadow-{req.request_id}", req.agent_id, req.endpoint,
            req.call_site, int(req.priority),
            shadow_resp.input_tokens, shadow_resp.output_tokens,
            shadow_resp.duration_s, decision.queue_wait_ms, "ok",
            payload=req.payload, response=shadow_resp.body,
        )

    async def _execute_streaming(
        self,
        req: QueuedRequest,
        ep_cfg: EndpointConfig,
        decision: DispatchDecision,
    ) -> None:
        stream_q = self._pending_streams.get(req.request_id)
        if not stream_q:
            return

        await stream_q.put({
            "type": "admitted",
            "queue_wait_ms": round(decision.queue_wait_ms, 1),
        })

        t0 = time.monotonic()
        input_tokens = 0
        output_tokens = 0
        last_finish_reason: str | None = None
        ttft_ms: float | None = None  # Phase 4.1 — time to first token

        # Phase 1.5: bound the stream to the caller's remaining deadline so an
        # abandoned stream can't hold its slot past the SLA.
        stream_timeout = max(1.0, req.timeout_deadline - time.monotonic())
        # Phase 5C: time-to-first-token watchdog. Start with a SHORT deadline; on
        # the first token, reschedule to the full SLA. A 0-token hang then aborts
        # in ~TTFT seconds (freeing the slot) instead of burning the whole 180s.
        ttft_deadline_s = min(_STREAM_TTFT_DEADLINE_S, stream_timeout)
        loop = asyncio.get_event_loop()
        try:
            async with asyncio.timeout(ttft_deadline_s) as _cm:
                async for event in self._backend.stream(
                    ep_cfg, req.payload, req.payload_type,
                    req.request_id, timeout_s=stream_timeout,
                ):
                    if event.event_type == "chunk":
                        if ttft_ms is None:
                            ttft_ms = (time.monotonic() - t0) * 1000.0
                            # First token — extend the watchdog to the full SLA.
                            _cm.reschedule(loop.time() + max(
                                0.1, stream_timeout - (time.monotonic() - t0)))
                        await stream_q.put({
                            "type": "chunk",
                            "data": event.data,
                        })
                        if event.parsed:
                            usage = event.parsed.get("usage")
                            if usage:
                                input_tokens = usage.get("prompt_tokens", input_tokens)
                                output_tokens = usage.get("completion_tokens", output_tokens)
                            choices = event.parsed.get("choices") or []
                            if choices and isinstance(choices[0], dict):
                                fr = choices[0].get("finish_reason")
                                if fr:
                                    last_finish_reason = fr
                    elif event.event_type == "done":
                        break
        except (asyncio.TimeoutError, BackendTimeout) as exc:
            # ttft watchdog OR overall stream deadline OR backend timeout. The
            # "backpressure" marker makes it deferrable (is_deferrable_llm_error)
            # so the caller defers instead of dead-lettering. A 0-token hang is
            # the common case — name it so log_scan can see the pattern.
            if ttft_ms is None and isinstance(exc, asyncio.TimeoutError):
                err = ("backend produced no output within "
                       f"{ttft_deadline_s:.0f}s (ttft timeout) — backpressure")
            else:
                err = f"stream deadline exceeded — backpressure ({exc})"
            await stream_q.put({"type": "error", "error": err})
            duration = time.monotonic() - t0
            self._record_completion(req, decision, duration, input_tokens, output_tokens, "timeout")
            self._record_timeout_event(
                req, layer="stream", elapsed_s=duration,
                queue_wait_ms=decision.queue_wait_ms, emit_metrics_and_log=False,
            )
            return
        except Exception as exc:
            await stream_q.put({"type": "error", "error": str(exc)})
            duration = time.monotonic() - t0
            self._record_completion(req, decision, duration, input_tokens, output_tokens, "error")
            return

        duration = time.monotonic() - t0
        await stream_q.put({
            "type": "done",
            "queue_wait_ms": round(decision.queue_wait_ms, 1),
            "backend_latency_ms": round(duration * 1000, 1),
            "ttft_ms": round(ttft_ms or 0.0, 1),  # Phase 4.1
            "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
        })
        # Phase 1.1: the chunks already streamed (can't un-send), but record
        # truncation of a structured stream so the storm is visible in metrics.
        status = (
            "truncated"
            if last_finish_reason == "length" and self._request_is_structured(req)
            else "ok"
        )
        self._record_completion(
            req, decision, duration, input_tokens, output_tokens, status,
            finish_reason=last_finish_reason,
        )

    # ----- circuit breaker / integrity helpers (Phase 1) -----

    def _endpoint_healthy(self, endpoint: str) -> bool:
        ep = normalize_endpoint(endpoint)
        # Phase 5F: an operator drain overrides the auto-circuit — reads
        # unhealthy so all the defer/fast-fail machinery applies immediately.
        if ep in self._paused_endpoints:
            return False
        h = self._endpoint_health.get(ep)
        return h["healthy"] if h else True

    def _retry_after_s(self, endpoint: str) -> int:
        """Retry-After for a shed (Phase 2.4), derived from the endpoint's
        recent p95 backend latency (a drained slot frees on ~that cadence),
        clamped to [5, 60]s."""
        p95 = self._metrics.percentile(
            "backend_latency_ms", 95, endpoint=endpoint, now=time.monotonic())
        return max(5, min(60, int((p95 or 10000.0) / 1000.0)))

    @staticmethod
    def _is_transient_backend_error(exc: Exception) -> bool:
        """Infra-transient backend failures that should DEFER (retry within the
        deadline) rather than surface: unreachable/503 and an empty completion
        (a backend hiccup). A real 4xx / other-5xx is deterministic → surface.
        BackendTimeout is handled on its own branch and never reaches here."""
        if isinstance(exc, BackendUnavailable):
            return True
        if isinstance(exc, BackendError):
            return "empty completion" in (exc.detail or "")
        return False

    def _request_is_structured(self, req: QueuedRequest) -> bool:
        """True when the request constrained its output (grammar / JSON schema /
        structured outputs), so a finish_reason=length truncation almost
        certainly produced broken/unparseable output — not a benign capped reply."""
        if req.payload_type != "chat_completion":
            return False
        p = req.payload
        if not isinstance(p, dict):
            return False
        if self._extract_grammar(p)[0]:
            return True
        if p.get("response_format") or p.get("structured_outputs"):
            return True
        eb = p.get("extra_body")
        if isinstance(eb, dict) and any(
            eb.get(k) for k in (
                "response_format", "structured_outputs",
                "guided_grammar", "guided_json", "guided_choice",
            )
        ):
            return True
        return False

    async def _update_endpoint_health(
        self, ep_name: str, ep_cfg: "EndpointConfig", probe_ok: bool,
    ) -> None:
        """Drive the per-endpoint circuit from poller probe results.

        Liveness (/health) is decoupled from capacity-discovery (/props,
        /v1/models): a sustained discovery failure trips the circuit ONLY when a
        /health probe also fails (a saturated backend still answers /health, so
        load never trips it — alert-don't-kill). Phase 5C makes RECOVERY
        symmetric: an already-tripped endpoint recovers as soon as /health is
        back, even if capacity-discovery is still flaky — otherwise a backend
        whose /props stays down (but is alive + serving) would latch open
        forever (the one-way-latch the audit flagged)."""
        h = self._endpoint_health.setdefault(
            ep_name, {"healthy": True, "consecutive_failures": 0, "unhealthy_since": None})
        if probe_ok:
            if not h["healthy"]:
                logger.warning("endpoint %s RECOVERED (discovery) — resuming dispatch", ep_name)
                self._dispatch_event.set()  # drain its deferred queue
            h["healthy"] = True
            h["consecutive_failures"] = 0
            h["unhealthy_since"] = None
            return
        # Discovery failed. If already tripped, recover on LIVENESS alone so a
        # /health-up backend with flaky discovery doesn't latch open forever.
        if not h["healthy"]:
            if await self._backend.probe_health(ep_cfg):
                logger.warning(
                    "endpoint %s RECOVERED (/health up, discovery still flaky) — "
                    "resuming dispatch", ep_name)
                h["healthy"] = True
                h["consecutive_failures"] = 0
                h["unhealthy_since"] = None
                self._dispatch_event.set()
            return
        h["consecutive_failures"] += 1
        if h["healthy"] and h["consecutive_failures"] >= self._health_fail_threshold:
            alive = await self._backend.probe_health(ep_cfg)
            if not alive:
                h["healthy"] = False
                h["unhealthy_since"] = time.monotonic()
                logger.critical(
                    "endpoint %s UNHEALTHY — %d consecutive probe failures + /health "
                    "down; deferring its queue, fast-failing interactive",
                    ep_name, h["consecutive_failures"],
                )
                self._fast_fail_interactive(ep_name)
            else:
                # Backend answers /health → up but discovery is flaky; don't trip.
                h["consecutive_failures"] = 0

    def _fast_fail_interactive(self, ep_name: str) -> None:
        """On the unhealthy transition, release queued INTERACTIVE/FOREGROUND
        requests for this endpoint with a deferrable error so they don't wait out
        their full deadline; BACKGROUND stays queued to defer until recovery."""
        eq = self._scheduler._queues.get(ep_name)
        if not eq:
            return
        for band in (PriorityBand.INTERACTIVE, PriorityBand.FOREGROUND):
            for agent_id in list(eq._queues.get(band, {}).keys()):
                for req in list(eq._queues[band][agent_id]):
                    self._scheduler.cancel(req.request_id)
                    self._queue_db.persist_expire(req.request_id)
                    self._resolve_error(
                        req, f"backend {ep_name} unavailable (circuit open)")

    def _evaluate_alerts(self, now: float) -> None:
        """Evaluate proxy-internal alert conditions (Phase 2.5) and surface them
        to logs (log_scan/health-verifier) + /v1/status. Never restarts a backend — a dead
        backend is alerted, not killed (alert-don't-kill)."""
        snaps: dict[str, dict] = {}
        for ep in self._config.endpoints:
            s = self._scheduler.endpoint_snapshot(ep)
            # Phase 5F: the endpoint_paused ERROR alert is for an UNINTENDED
            # backend-down. Exclude operator drains so a planned maintenance
            # pause doesn't page as an outage — those surface as a separate
            # informational endpoint_drained alert below.
            s["paused"] = (not self._endpoint_healthy(ep)) and ep not in self._paused_endpoints
            snaps[ep] = s
        alerts = check_alerts(
            endpoint_snapshots=snaps,
            agent_budgets=self._budget_mgr.snapshot(),
            metrics=self._metrics,
            cost_model_samples={},  # cost-model-stale is INFO-only; skip for now
            queue_wal_size=self._queue_db.wal_size_bytes(),
            now=now,
        )
        # Phase 5F: operator drains — visible (so it's clear thinker is parked),
        # but WARNING not ERROR (intentional, not an outage).
        for ep in sorted(self._paused_endpoints):
            alerts.append(AlertCondition(
                name="endpoint_drained", severity="WARNING", triggered=True,
                detail=f"endpoint {ep} paused for maintenance (operator drain)",
            ))
        # Phase 5B.3: surface DB-writer-thread death (the single sanctioned bg
        # thread). If it dies, persistence degrades to loud sync fallback — page.
        if not self._queue_db.writer_alive():
            alerts.append(AlertCondition(
                name="writer_thread_dead", severity="CRITICAL", triggered=True,
                detail=f"db writer thread down (restarts={self._queue_db.writer_restarts()})",
            ))
        dropped = self._queue_db.write_q_dropped()
        if dropped:
            alerts.append(AlertCondition(
                name="write_queue_overflow", severity="WARNING", triggered=True,
                detail=f"{dropped} best-effort DB write(s) dropped (queue full)",
            ))
        self._alerts = [
            {"name": a.name, "severity": a.severity, "detail": a.detail}
            for a in alerts
        ]
        current = {(a.name, a.detail) for a in alerts}
        emit = {"CRITICAL": logger.critical, "ERROR": logger.error,
                "WARNING": logger.warning}
        for a in alerts:
            if (a.name, a.detail) not in self._alert_logged:
                emit.get(a.severity, logger.info)(
                    "ALERT [%s] %s: %s", a.severity, a.name, a.detail)
        self._alert_logged = current

    def _resolve_error(self, req: QueuedRequest, error: str) -> None:
        future = self._pending_futures.get(req.request_id)
        if future and not future.done():
            future.set_result({
                "request_id": req.request_id,
                "status": "error",
                "error": error,
            })
        stream_q = self._pending_streams.get(req.request_id)
        if stream_q:
            try:
                stream_q.put_nowait({"type": "error", "error": error})
            except asyncio.QueueFull:
                pass

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
        now = time.monotonic()

        # Report to scheduler
        self._scheduler.complete(
            CompletionRecord(
                request_id=req.request_id,
                duration_s=duration_s,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=(status == "ok"),
                occupancy_during=decision.occupancy_at_dispatch,
            ),
            now,
        )

        # Persist completion (with payload + response for corpus).
        # Skip corpus capture for embedding/rerank — large vectors bloat
        # the DB and aren't useful for replay testing.
        capture = req.payload_type == "chat_completion"
        self._queue_db.persist_complete(
            req.request_id, req.agent_id, req.endpoint,
            req.call_site, int(req.priority),
            input_tokens, output_tokens, duration_s,
            decision.queue_wait_ms, status,
            payload=req.payload if capture else None,
            response=response_body,
            session_id=req.session_id,
            turn_id=req.turn_id,
            caller_id=req.caller_id,
            finish_reason=finish_reason,
        )

        # Log
        self._request_logger.log(RequestLogRecord(
            ts=datetime.now(timezone.utc).isoformat(),
            request_id=req.request_id,
            agent_id=req.agent_id,
            endpoint=req.endpoint,
            call_site=req.call_site,
            priority=req.priority.name,
            band=req.band.name.lower(),
            payload_type=req.payload_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_ss=req.estimated_cost_ss,
            actual_cost_ss=duration_s,
            queue_wait_ms=decision.queue_wait_ms,
            backend_latency_ms=duration_s * 1000,
            total_latency_ms=(now - req.enqueued_at) * 1000,
            occupancy_at_dispatch=decision.occupancy_at_dispatch,
            status=status,
            estimated_input_tokens=estimate_input_tokens(req.payload),
            max_output_tokens=req.payload.get("max_tokens", 0),
            session_id=req.session_id,
            turn_id=req.turn_id,
            caller_id=req.caller_id,
        ))

        # Metrics
        self._metrics.record(MetricsSample(
            timestamp=now,
            endpoint=req.endpoint,
            agent_id=req.agent_id,
            priority=req.priority.name,
            queue_wait_ms=decision.queue_wait_ms,
            backend_latency_ms=duration_s * 1000,
            status=status,
            slot_seconds=duration_s,
        ))

        # Timeout-advice model + shadow log (observational only — guarded
        # so a fault here never disturbs the caller or the scheduler).
        try:
            self._record_timeout_shadow(req, now, duration_s, output_tokens, status)
        except Exception as exc:  # noqa: BLE001
            logger.warning("timeout shadow record failed for %s: %s", req.request_id, exc)

        # Trigger scheduler (a slot freed up)
        self._dispatch_event.set()

    def _record_timeout_shadow(
        self,
        req: QueuedRequest,
        now: float,
        duration_s: float,
        output_tokens: int,
        status: str,
    ) -> None:
        """Feed the timeout model and log the counterfactual: what
        ``recommended`` would have been for this call vs. the actual
        end-to-end latency and the timeout actually applied."""
        end_to_end_ms = (now - req.enqueued_at) * 1000.0
        est_in = estimate_input_tokens(req.payload)
        est_out = int(req.payload.get("max_tokens", 0) or 0)
        priority = int(req.priority)

        # Advice reflects history BEFORE this sample is folded in.
        advice = self._timeout_model.advise(req.endpoint, priority, est_in, est_out)

        self._timeout_model.record(
            endpoint=req.endpoint,
            priority=priority,
            input_tokens=est_in,
            output_tokens=output_tokens,
            end_to_end_ms=end_to_end_ms,
            status=status,
            now=now,
        )

        # Only successful calls give a representative latency to compare.
        if status != "ok":
            return
        recommended_ms = advice["recommended_ms"]
        self._queue_db.persist_timeout_shadow(
            request_id=req.request_id,
            endpoint=normalize_endpoint(req.endpoint),
            priority=priority,
            est_in=est_in,
            est_out=est_out,
            actual_out=output_tokens,
            actual_total_ms=round(end_to_end_ms, 1),
            applied_timeout_s=req.timeout_s,
            recommended_ms=recommended_ms,
            p95_ms=advice["p95_ms"],
            median_ms=advice["median_ms"],
            min_ms=advice["min_ms"],
            source=advice["source"],
            would_timeout=(end_to_end_ms > recommended_ms),
        )

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
        """Record a call that hit its timeout instead of finishing.

        Writes a queryable ``proxy_timeouts`` row with the load context at
        the moment it gave up, emits a WARNING (so log_scan/health-verifier see it),
        and — for the layers that don't otherwise flow through
        ``_record_completion`` (admission/client_wait) — a metrics sample
        and request-log line so the timeout counter and JSONL trail are
        complete. Fully guarded: a fault here never disturbs the caller.

        ``layer``: admission | client_wait | backend | stream.
        """
        rid = req.request_id
        if rid in self._timed_out_ids:
            return  # already counted this request's timeout
        self._timed_out_ids.add(rid)
        if len(self._timed_out_ids) > 8192:
            self._timed_out_ids.clear()  # bounded; rare duplicate after reset is harmless

        try:
            now = time.monotonic()
            snap = self._scheduler.endpoint_snapshot(req.endpoint)
            est_in = estimate_input_tokens(req.payload)
            est_out = int(req.payload.get("max_tokens", 0) or 0)
            priority = int(req.priority)
            ep_cfg = self._config.endpoints.get(normalize_endpoint(req.endpoint))
            context_window = ep_cfg.context_per_slot if ep_cfg else 0
            context_used_pct = (
                round(est_in / context_window * 100.0, 1) if context_window else None
            )
            try:
                recommended_ms = self._timeout_model.advise(
                    req.endpoint, priority, est_in, est_out,
                )["recommended_ms"]
            except Exception:  # noqa: BLE001
                recommended_ms = 0.0
            under = bool(recommended_ms and elapsed_s * 1000.0 <= recommended_ms)

            logger.warning(
                "LLM TIMEOUT layer=%s endpoint=%s tier=%s caller=%s "
                "elapsed=%.1fs applied=%.1fs in_flight=%d queued=%d est_in=%d "
                "est_out=%d ctx_used=%s%% recommended=%.0fms premature=%s",
                layer, req.endpoint, req.priority.name,
                req.caller_id or f"{req.agent_id}/{req.call_site}",
                elapsed_s, req.timeout_s, snap["in_flight"], snap["queued"],
                est_in, est_out, context_used_pct, recommended_ms, under,
            )

            if emit_metrics_and_log:
                # Make the /v1/metrics "timeouts" counter real for the
                # paths that never reach _record_completion.
                self._metrics.record(MetricsSample(
                    timestamp=now,
                    endpoint=req.endpoint,
                    agent_id=req.agent_id,
                    priority=req.priority.name,
                    queue_wait_ms=(queue_wait_ms or 0.0),
                    backend_latency_ms=0.0,
                    status="timeout",
                    slot_seconds=0.0,
                ))
                self._request_logger.log(RequestLogRecord(
                    ts=datetime.now(timezone.utc).isoformat(),
                    request_id=rid,
                    agent_id=req.agent_id,
                    endpoint=req.endpoint,
                    call_site=req.call_site,
                    priority=req.priority.name,
                    band=req.band.name.lower(),
                    payload_type=req.payload_type,
                    input_tokens=0,
                    output_tokens=0,
                    estimated_cost_ss=req.estimated_cost_ss,
                    actual_cost_ss=0.0,
                    queue_wait_ms=(queue_wait_ms or 0.0),
                    backend_latency_ms=0.0,
                    total_latency_ms=elapsed_s * 1000.0,
                    occupancy_at_dispatch=snap["in_flight"],
                    status="timeout",
                    estimated_input_tokens=est_in,
                    max_output_tokens=est_out,
                    session_id=req.session_id,
                    turn_id=req.turn_id,
                    caller_id=req.caller_id,
                ))

            self._queue_db.persist_timeout_event(
                request_id=rid,
                endpoint=normalize_endpoint(req.endpoint),
                priority=priority,
                agent_id=req.agent_id,
                call_site=req.call_site,
                layer=layer,
                elapsed_s=round(elapsed_s, 3),
                applied_timeout_s=req.timeout_s,
                queue_wait_ms=(round(queue_wait_ms, 1) if queue_wait_ms is not None else None),
                in_flight=snap["in_flight"],
                queued=snap["queued"],
                max_slots=snap["max_slots"],
                est_in=est_in,
                est_out=est_out,
                recommended_ms=recommended_ms,
                under_recommended=under,
                session_id=req.session_id,
                turn_id=req.turn_id,
                caller_id=req.caller_id,
                context_window=context_window,
                context_used_pct=context_used_pct,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("timeout event record failed for %s: %s", rid, exc)

    def _on_admission_timeout(self, req: QueuedRequest) -> None:
        """Scheduler callback: a request expired while still queued. Log
        it and release the caller promptly with a timeout result (instead
        of letting it wait out its own — identical — deadline)."""
        elapsed = time.monotonic() - req.enqueued_at
        self._record_timeout_event(req, layer="admission", elapsed_s=elapsed)
        future = self._pending_futures.get(req.request_id)
        if future and not future.done():
            future.set_result({
                "request_id": req.request_id,
                "status": "timeout",
                "error": "timeout",
            })
        stream_q = self._pending_streams.get(req.request_id)
        if stream_q:
            try:
                stream_q.put_nowait({"type": "error", "error": "timeout"})
            except asyncio.QueueFull:
                pass
        self._queue_db.persist_expire(req.request_id)

    # ----- capacity poller -----

    async def _capacity_poller_loop(self) -> None:
        """Periodically probe backends for slot counts and context sizes."""
        while True:
            for ep_name, ep_cfg in self._config.endpoints.items():
                # Phase 5F: an operator-paused endpoint is intentionally down
                # (maintenance) — don't probe it (probes would fail + churn the
                # circuit/logs). /resume removes it from the set; the next poll
                # then re-probes, recovers, and re-discovers capacity (e.g. the
                # new max_model_len after a vLLM restart).
                if ep_name in self._paused_endpoints:
                    continue
                probe_ok = False
                try:
                    # Capacity discovery is engine-specific. llama.cpp reports
                    # slots + context via /props; vLLM has no /props or /slots,
                    # so the per-request context ceiling comes from /v1/models
                    # max_model_len (concurrency/max_slots stays config-driven).
                    if ep_cfg.backend_engine == "vllm":
                        cap = await self._backend.probe_vllm_capacity(ep_cfg)
                        if cap:
                            self._apply_discovered_vllm_capacity(ep_name, ep_cfg, cap)
                            probe_ok = True
                    else:
                        props = await self._backend.probe_props(ep_cfg)
                        if props:
                            self._apply_discovered_props(ep_name, ep_cfg, props)
                            probe_ok = True
                    # Discover the served model id (the name the backend
                    # answers to). vLLM validates it, so the proxy sends
                    # this — not the caller's role/alias — on dispatch.
                    served = await self._backend.probe_models(ep_cfg)
                    if served:
                        probe_ok = True
                        if served != ep_cfg.served_model_id:
                            logger.info(
                                "endpoint %s: served model id = %s (was %s)",
                                ep_name, served, ep_cfg.served_model_id or "<role>",
                            )
                            ep_cfg.served_model_id = served
                except Exception as exc:
                    logger.debug("poller probe %s failed: %s", ep_name, exc)
                # Circuit-breaker health update (Phase 1.2). Guarded so a fault
                # here never stalls discovery.
                try:
                    await self._update_endpoint_health(ep_name, ep_cfg, probe_ok)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("health update %s failed: %s", ep_name, exc)
            mono = time.monotonic()
            # Retention sweep (Phase 2.3): daily DB trim, off-loop via the writer.
            if mono - self._last_cleanup_at > 86400.0:
                self._last_cleanup_at = mono
                self._queue_db.cleanup_old_completions()
            # Periodic DRR-balance persistence (Phase 3.4) so a SIGKILL loses at
            # most ~60s of fairness state.
            if mono - self._last_budget_save_at > 60.0:
                self._last_budget_save_at = mono
                self._queue_db.save_budgets(self._budget_mgr.snapshot())
            # Alerting (Phase 2.5): evaluate conditions → logs + /v1/status so
            # health-verifier/log_scan see proxy-internal health. Never restarts a backend.
            try:
                self._evaluate_alerts(mono)
            except Exception as exc:  # noqa: BLE001
                logger.debug("alert evaluation failed: %s", exc)
            # Age out stale timeout-model samples (cheap; piggybacks the
            # 10s poller instead of a dedicated task).
            self._timeout_model.prune(mono)
            await asyncio.sleep(10.0)

    def _apply_discovered_props(
        self, ep_name: str, ep_cfg: EndpointConfig, props: dict,
    ) -> None:
        """Update endpoint config from discovered /props data."""
        # llama.cpp format
        gen_settings = props.get("default_generation_settings", {})
        n_parallel = gen_settings.get("n_parallel")
        if n_parallel is None:
            n_parallel = props.get("total_slots")
        if n_parallel is None:
            slots = props.get("slots")
            if isinstance(slots, list):
                n_parallel = len(slots)

        if n_parallel and n_parallel != ep_cfg.max_slots:
            old = ep_cfg.max_slots
            ep_cfg.max_slots = n_parallel
            self._cost_model.update_max_slots(ep_name, n_parallel)
            self._budget_mgr.set_total_capacity(self._config.total_fleet_slots)
            if n_parallel < ep_cfg.min_expected_slots:
                logger.critical(
                    "endpoint %s: discovered %d slots (expected >= %d)",
                    ep_name, n_parallel, ep_cfg.min_expected_slots,
                )
            elif old != n_parallel:
                logger.info(
                    "endpoint %s: slots %d → %d (discovered)",
                    ep_name, old, n_parallel,
                )

        # Context size
        n_ctx = gen_settings.get("n_ctx") or props.get("n_ctx")
        if n_ctx and n_parallel:
            ctx_per_slot = n_ctx // n_parallel
            if ctx_per_slot != ep_cfg.context_per_slot:
                ep_cfg.context_per_slot = ctx_per_slot

    def _apply_discovered_vllm_capacity(
        self, ep_name: str, ep_cfg: EndpointConfig, cap: dict,
    ) -> None:
        """Update a vLLM endpoint's context ceiling from /v1/models.

        vLLM's ``max_model_len`` is the per-request context window directly
        (not a fleet n_ctx to divide by slots). max_slots is left as configured
        — vLLM doesn't expose --max-num-seqs over the API and the proxy's
        admission cap is a deliberate policy knob, not a discovered value."""
        mlen = cap.get("max_model_len")
        if mlen and mlen != ep_cfg.context_per_slot:
            old = ep_cfg.context_per_slot
            ep_cfg.context_per_slot = mlen
            logger.info(
                "endpoint %s: context_per_slot %d → %d (vLLM max_model_len)",
                ep_name, old, mlen,
            )


# avoid circular import — EndpointConfig used in type hints
from .config import EndpointConfig as EndpointConfig  # noqa: E402
