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
from .coalesce import DeterministicCache, EmbedCoalescer
from .grammar import GrammarResult, grammar_hash, normalize_and_validate
from .config import (
    CLASS_TO_ROLE,
    LLMPriority,
    ProxyConfig,
    normalize_endpoint,
)
from .cost_model import CostModel, estimate_input_tokens
from .timeout_model import TimeoutModel
from .observability import (
    MetricsSample,
    RequestLogRecord,
    RequestLogger,
    RollingMetrics,
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

        # Caching / coalescing
        self._cache = DeterministicCache()
        self._coalescer = EmbedCoalescer()

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
        self._pending_futures: dict[str, asyncio.Future] = {}
        self._pending_streams: dict[str, asyncio.Queue] = {}
        # Dedupe set so a single request that races across two timeout
        # layers (e.g. admission expiry + client-wait) is logged once.
        self._timed_out_ids: set[str] = set()
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

        # Recover queued requests from WAL
        recovered = self._queue_db.recover_queued(time.monotonic())
        for req in recovered:
            self._scheduler.enqueue(req)

        # Bootstrap cost model from recent completion history
        self._bootstrap_cost_model()

        # Bootstrap timeout-advice model from the same history
        self._bootstrap_timeout_model()

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
        if self._scheduler_task:
            self._scheduler_task.cancel()
        if self._poller_task:
            self._poller_task.cancel()
        await self._backend.close()
        self._queue_db.close()
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

    async def handle_submit(self, body: dict, request: Request) -> Response:
        now = time.monotonic()

        req = QueuedRequest.create(
            agent_id=body.get("agent_id", "unknown"),
            endpoint=self._resolve_endpoint(body),
            priority=body.get("priority"),
            call_site=body.get("call_site", "unknown"),
            payload_type=body.get("payload_type", "chat_completion"),
            payload=body.get("payload", {}),
            timeout_s=float(body.get("timeout_s", 180.0)),
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
                return JSONResponse(
                    {"status": "error", "request_id": req.request_id, **grammar_err},
                    status_code=422,
                )

        # Check deterministic cache
        cache_key = self._cache.cache_key(req.endpoint, req.payload)
        if cache_key:
            cached = self._cache.get(cache_key)
            if cached:
                return JSONResponse({
                    "status": "ok",
                    "request_id": req.request_id,
                    "queue_wait_ms": 0,
                    "backend_latency_ms": 0,
                    "estimated_cost_ss": 0,
                    "response": cached,
                    "cache_hit": True,
                })

        # Streaming vs non-streaming
        if req.stream:
            return await self._handle_streaming_submit(req)
        else:
            return await self._handle_sync_submit(req, cache_key)

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
        self, req: QueuedRequest, cache_key: str | None,
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
            return JSONResponse(
                {"error": "timeout", "request_id": req.request_id},
                status_code=504,
            )
        finally:
            self._pending_futures.pop(req.request_id, None)

        # Admission timeout (scheduler callback) resolves the future with a
        # timeout result — already logged there; surface the same 504.
        if result.get("status") == "timeout":
            return JSONResponse(
                {"error": "timeout", "request_id": req.request_id},
                status_code=504,
            )

        # Cache if deterministic
        if cache_key and result.get("status") == "ok":
            self._cache.put(cache_key, result.get("response", {}))

        status_code = 200 if result.get("status") == "ok" else 502
        return JSONResponse(result, status_code=status_code)

    async def _handle_streaming_submit(self, req: QueuedRequest) -> Response:
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._pending_streams[req.request_id] = queue

        self._scheduler.enqueue(req)
        self._queue_db.persist_enqueue(req)
        self._dispatch_event.set()

        async def stream_generator():
            # Emit queued event
            yield f"data: {json.dumps({'type': 'queued', 'request_id': req.request_id})}\n\n"

            try:
                while True:
                    event = await asyncio.wait_for(
                        queue.get(), timeout=req.timeout_s,
                    )
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") in ("done", "error"):
                        break
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'error', 'error': 'timeout'})}\n\n"
                self._record_timeout_event(req, layer="stream", elapsed_s=req.timeout_s)
            finally:
                self._pending_streams.pop(req.request_id, None)

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ----- handler: OpenAI compat -----

    async def handle_openai_chat(self, body: dict, request: Request) -> Response:
        remote_ip = request.client.host if request.client else "unknown"
        identity = self._acl.identify(remote_ip)
        if not identity:
            return JSONResponse(
                {"error": "access_denied", "your_ip": remote_ip},
                status_code=403,
            )
        agent_id, default_priority = identity
        model = body.get("model", "qwen-analyst")

        submit_body = {
            "agent_id": agent_id,
            "endpoint": model,
            "priority": int(default_priority),
            "call_site": f"{agent_id}.openai_compat",
            "payload_type": "chat_completion",
            "payload": body,
            "timeout_s": 180.0,
        }
        return await self.handle_submit(submit_body, request)

    async def handle_openai_embeddings(self, body: dict, request: Request) -> Response:
        remote_ip = request.client.host if request.client else "unknown"
        identity = self._acl.identify(remote_ip)
        if not identity:
            return JSONResponse(
                {"error": "access_denied", "your_ip": remote_ip},
                status_code=403,
            )
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
        return await self.handle_submit(submit_body, request)

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
            endpoints[ep_name] = snap

        agents = {
            b["agent_id"]: b
            for b in self._budget_mgr.snapshot()
        }

        return JSONResponse({
            "endpoints": endpoints,
            "agents": agents,
            "cache": self._cache.stats(),
            "coalesce": {
                "saved_calls": self._coalescer.saved_calls,
                "active_pending": self._coalescer.active_pending,
            },
            "scheduler": {
                **self._scheduler.stats(),
                "uptime_s": round(time.monotonic() - self._started_at, 0),
            },
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
        return JSONResponse(
            {
                "status": "ok" if ok else "degraded",
                "uptime_s": round(time.monotonic() - self._started_at, 0),
                "total_dispatched": self._scheduler.stats()["total_dispatched"],
                "endpoints": len(self._config.endpoints),
                "total_slots": self._config.total_fleet_slots,
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
                asyncio.create_task(self._execute_dispatch(decision))

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
        t0 = time.monotonic()
        try:
            resp = await self._backend.call(
                ep_cfg, req.payload, req.payload_type,
                req.request_id, timeout_s=req.timeout_s,
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
            self._resolve_error(req, str(exc))
            self._record_completion(req, decision, duration, 0, 0, "error")
            return

        duration = time.monotonic() - t0
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
            response_body=capture_response,
        )

        # Shadow backend A/B: fire-and-forget to the shadow if configured
        if ep_cfg.shadow_host and ep_cfg.shadow_port:
            asyncio.create_task(self._execute_shadow(
                req, ep_cfg, decision, resp,
            ))

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

        try:
            async for event in self._backend.stream(
                ep_cfg, req.payload, req.payload_type,
                req.request_id, timeout_s=req.timeout_s,
            ):
                if event.event_type == "chunk":
                    await stream_q.put({
                        "type": "chunk",
                        "data": event.data,
                    })
                    if event.parsed:
                        usage = event.parsed.get("usage")
                        if usage:
                            input_tokens = usage.get("prompt_tokens", input_tokens)
                            output_tokens = usage.get("completion_tokens", output_tokens)
                elif event.event_type == "done":
                    break
        except BackendTimeout as exc:
            await stream_q.put({"type": "error", "error": str(exc)})
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
            "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
        })
        self._record_completion(req, decision, duration, input_tokens, output_tokens, "ok")

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
                try:
                    # Capacity discovery is engine-specific. llama.cpp reports
                    # slots + context via /props; vLLM has no /props or /slots,
                    # so the per-request context ceiling comes from /v1/models
                    # max_model_len (concurrency/max_slots stays config-driven).
                    if ep_cfg.backend_engine == "vllm":
                        cap = await self._backend.probe_vllm_capacity(ep_cfg)
                        if cap:
                            self._apply_discovered_vllm_capacity(ep_name, ep_cfg, cap)
                    else:
                        props = await self._backend.probe_props(ep_cfg)
                        if props:
                            self._apply_discovered_props(ep_name, ep_cfg, props)
                    # Discover the served model id (the name the backend
                    # answers to). vLLM validates it, so the proxy sends
                    # this — not the caller's role/alias — on dispatch.
                    served = await self._backend.probe_models(ep_cfg)
                    if served and served != ep_cfg.served_model_id:
                        logger.info(
                            "endpoint %s: served model id = %s (was %s)",
                            ep_name, served, ep_cfg.served_model_id or "<role>",
                        )
                        ep_cfg.served_model_id = served
                except Exception as exc:
                    logger.debug("poller probe %s failed: %s", ep_name, exc)
            # Age out stale timeout-model samples (cheap; piggybacks the
            # 10s poller instead of a dedicated task).
            self._timeout_model.prune(time.monotonic())
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
