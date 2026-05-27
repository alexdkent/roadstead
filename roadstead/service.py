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
from .config import (
    CLASS_TO_ROLE,
    LLMPriority,
    ProxyConfig,
    normalize_endpoint,
)
from .cost_model import CostModel, estimate_input_tokens
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
        self._budget_mgr = BudgetManager(starvation_timeout_s=config.starvation_timeout_s)
        self._scheduler = Scheduler(config, self._cost_model, self._budget_mgr)
        self._backend = BackendClientPool()
        self._queue_db = PersistentQueue(config.queue_db_path or None)

        # Caching / coalescing
        self._cache = DeterministicCache()
        self._coalescer = EmbedCoalescer()

        # Observability
        self._metrics = RollingMetrics(window_s=300.0)
        self._request_logger = RequestLogger(config.request_log_path or None)
        self._acl = IPIdentityMap.from_env()

        # Async plumbing
        self._dispatch_event = asyncio.Event()
        self._pending_futures: dict[str, asyncio.Future] = {}
        self._pending_streams: dict[str, asyncio.Queue] = {}
        self._scheduler_task: asyncio.Task | None = None
        self._poller_task: asyncio.Task | None = None
        self._started_at = time.monotonic()

    # ----- lifecycle -----

    async def startup(self) -> None:
        """Initialize cost model, recover queue, start scheduler loop."""
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

        # Start background loops
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._poller_task = asyncio.create_task(self._capacity_poller_loop())

        logger.info(
            "llmproxy started: %d endpoints, %d total slots",
            len(self._config.endpoints), self._config.total_fleet_slots,
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

    async def handle_submit(self, body: dict, request: Request) -> Response:
        now = time.monotonic()

        req = QueuedRequest.create(
            agent_id=body.get("agent_id", "unknown"),
            endpoint=body.get("endpoint", "chat"),
            priority=body.get("priority"),
            call_site=body.get("call_site", "unknown"),
            payload_type=body.get("payload_type", "chat_completion"),
            payload=body.get("payload", {}),
            timeout_s=float(body.get("timeout_s", 180.0)),
            session_id=body.get("session_id"),
            turn_id=body.get("turn_id"),
            request_id=body.get("request_id"),
            now=now,
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
            return JSONResponse(
                {"error": "timeout", "request_id": req.request_id},
                status_code=504,
            )
        finally:
            self._pending_futures.pop(req.request_id, None)

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
        except (BackendTimeout, BackendUnavailable, BackendError) as exc:
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

        self._record_completion(
            req, decision, duration,
            resp.input_tokens, resp.output_tokens, "ok",
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

        # Persist completion
        self._queue_db.persist_complete(
            req.request_id, req.agent_id, req.endpoint,
            req.call_site, int(req.priority),
            input_tokens, output_tokens, duration_s,
            decision.queue_wait_ms, status,
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

        # Trigger scheduler (a slot freed up)
        self._dispatch_event.set()

    # ----- capacity poller -----

    async def _capacity_poller_loop(self) -> None:
        """Periodically probe backends for slot counts and context sizes."""
        while True:
            for ep_name, ep_cfg in self._config.endpoints.items():
                try:
                    props = await self._backend.probe_props(ep_cfg)
                    if props:
                        self._apply_discovered_props(ep_name, ep_cfg, props)
                except Exception as exc:
                    logger.debug("poller probe %s failed: %s", ep_name, exc)
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


# avoid circular import — EndpointConfig used in type hints
from .config import EndpointConfig as EndpointConfig  # noqa: E402
