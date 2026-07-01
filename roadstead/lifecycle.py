"""Lifecycle — admission -> scheduling -> dispatch -> response for the LLMProxy.

The request hot path: endpoint resolution, the `handle_submit` admission
sequence, the sync/stream response paths, the scheduler loop, backend dispatch
(`execute_*`), and completion/timeout recording. A behavior object over the
shared :class:`ProxyState`; it calls the Correction and Health collaborators by
reference. ProxyService keeps identical-signature delegators (contract §2).

`_openai_error` is a module function here (not a method) so both Lifecycle and
the HTTP handlers can build the OpenAI-shaped error envelope without a cycle.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from .backend import (
    BackendError,
    BackendResponse,
    BackendTimeout,
    BackendUnavailable,
)
from .config import (
    LLMPriority,
    PriorityBand,
    normalize_endpoint,
    uniform_correction_enabled,
)
from .constants import (
    _DEFAULT_TIMEOUT_S,
    _MIN_RETRY_BUDGET_S,
    _PAYLOAD_KIND,
    _RETRY_BACKOFF_S,
    _STREAM_INTERTOKEN_GAP_S,
    _STREAM_TTFT_DEADLINE_S,
)
from .correction import _EMPTY_RESCUE_MIN_TOKENS, _ToolCallStreamSanitizer
from .cost_model import estimate_input_tokens
from .observability import MetricsSample, RequestLogRecord
from .on_demand import OnDemandUnavailable
from .scheduler import CompletionRecord, DispatchDecision, QueuedRequest
from .sse_hub import DROP_SENTINEL

if TYPE_CHECKING:
    from .config import EndpointConfig  # noqa: F401
    from .correction import Correction
    from .health import Health
    from .state import ProxyState

logger = logging.getLogger(__name__)


# OpenAI-shaped error envelope — shared by Lifecycle + the HTTP front door.
def _openai_error(
    message: str, err_type: str, status_code: int, code: str | None = None,
) -> JSONResponse:
    """OpenAI-shaped error envelope for the /v1/chat/completions front door
    (goose-cli + any OpenAI client expects ``{"error": {...}}``). ``code``
    is the machine-readable taxonomy field (M2) — additive; the message
    substrings the fleet's deferral classifier sniffs are unchanged."""
    err: dict = {"message": str(message), "type": err_type}
    if code:
        err["code"] = code
    return JSONResponse({"error": err}, status_code=status_code)


class Lifecycle:
    """The request hot path over the shared ProxyState."""

    def __init__(self, state: "ProxyState", correction: "Correction",
                 health: "Health") -> None:
        self.state = state
        self.correction = correction
        self.health = health

    def resolve_endpoint(self, body: dict) -> str:
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
        if model_ep in self.state.config.endpoints and model_ep != submit_ep:
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
        if self.state.draining.is_set():
            # Phase 2.1: refuse new work while draining for shutdown so it defers
            # to the (about-to-restart) next instance instead of being dropped.
            # Phase 5C: include the "backpressure" marker so the body is
            # classified deferrable (is_deferrable_llm_error) by BOTH the sync
            # and streaming clients — without it, a streaming turn caught mid-
            # SIGTERM surfaced a hard error instead of deferring cleanly.
            err = "proxy draining for shutdown — backpressure"
            if openai:
                return _openai_error(err, "backpressure", 503, code="draining")
            return JSONResponse(
                {"status": "error", "error": err, "code": "draining"},
                status_code=503)

        now = time.monotonic()

        # Unknown-endpoint gate. Without it a typo'd role enqueues into a
        # queue the dispatch loop never visits and the caller blocks its FULL
        # timeout_s before a useless 504 (the OpenAI front door already 404s).
        # Shadow (flag off): count + WARN, behaviour unchanged. Enforce: fast
        # 404 whose message deliberately carries NO deferrable marker — a
        # typo is deterministic and must surface, not defer-loop.
        endpoint = self.resolve_endpoint(body)
        if endpoint not in self.state.config.endpoints:
            caller = str(body.get("caller_id") or body.get("agent_id") or "unknown")
            tally = self.state.unknown_endpoint_submits.setdefault(
                endpoint, {"count": 0, "callers": {}})
            tally["count"] += 1
            tally["callers"][caller] = tally["callers"].get(caller, 0) + 1
            if self.state.flags.get("unknown_endpoint_enforce"):
                err = (f"unknown endpoint {endpoint!r} — no such model/role "
                       f"(known: {sorted(self.state.config.endpoints)})")
                if openai:
                    return _openai_error(
                        err, "model_not_found", 404, code="unknown_endpoint")
                return JSONResponse(
                    {"status": "error", "error": err, "code": "unknown_endpoint"},
                    status_code=404)
            logger.warning(
                "unknown endpoint %r submitted by %s (call_site=%s) — SHADOW: "
                "request will wait out its full deadline; flip "
                "unknown_endpoint_enforce for a fast 404",
                endpoint, caller, body.get("call_site"))

        # Robust timeout_s: a malformed value must default, not 500 the request.
        # (priority is soft-defaulted inside QueuedRequest.create; payload/
        # endpoint/call_site already use safe .get defaults.)
        try:
            timeout_s = float(body.get("timeout_s", _DEFAULT_TIMEOUT_S))
            if not (timeout_s > 0) or timeout_s != timeout_s:  # non-positive / NaN
                raise ValueError("timeout_s must be a positive number")
        except (TypeError, ValueError) as exc:
            logger.warning("submit: bad timeout_s %r (%s); using default %.0fs",
                           body.get("timeout_s"), exc, _DEFAULT_TIMEOUT_S)
            timeout_s = _DEFAULT_TIMEOUT_S

        # On-demand endpoints cold-load for minutes — a caller's short timeout
        # (or the 180s default) would expire mid-load and never see a token.
        # Floor the deadline at the endpoint's timeout floor so the request can
        # wait out the load regardless of what the caller sent. (Extend-only:
        # a caller asking for MORE than the floor keeps their value.)
        if self.state.on_demand.manages(endpoint):
            floor_s = self.state.timeout_model.floor_ms(endpoint) / 1000.0
            if timeout_s < floor_s:
                timeout_s = floor_s

        req = QueuedRequest.create(
            agent_id=body.get("agent_id", "unknown"),
            endpoint=endpoint,
            priority=body.get("priority"),
            call_site=body.get("call_site", "unknown"),
            payload_type=body.get("payload_type", "chat_completion"),
            payload=body.get("payload", {}),
            timeout_s=timeout_s,
            session_id=body.get("session_id"),
            turn_id=body.get("turn_id"),
            caller_id=body.get("caller_id"),
            request_id=body.get("request_id"),
            now=now,
        )

        # Payload-shape gate (north-face hardening). A chat payload whose
        # ``messages`` is not a list of objects is unambiguously malformed: the
        # backend would 400 on it, and worse, ``estimate_input_tokens`` (context
        # gate below) and ``backend._normalize_chat_payload`` both do
        # ``msg.get(...)`` on each element → ``AttributeError`` → a confusing
        # generic 500 instead of a clean rejection. Reject here as a typed 400.
        # No shadow phase: unlike the unknown-endpoint / context heuristics a
        # non-dict message is never legitimate, so there is no false-positive
        # risk to soak. (Guards the Phase-T fuzz repro: messages=["hi","there"].)
        if req.payload_type == "chat_completion":
            messages = req.payload.get("messages")
            if messages is not None and (
                not isinstance(messages, list)
                or any(not isinstance(m, dict) for m in messages)
            ):
                err = ("invalid request: 'messages' must be a list of "
                       "{role, content} objects")
                if openai:
                    return _openai_error(
                        err, "invalid_request_error", 400,
                        code="invalid_messages")
                return JSONResponse(
                    {"status": "error", "request_id": req.request_id,
                     "error": err, "code": "invalid_messages"},
                    status_code=400)

        # Grammar authority: validate + safe-normalize any GBNF grammar
        # BEFORE enqueue. Fail loud on an invalid grammar rather than
        # dispatching it (llama-server would silently run unconstrained).
        if req.payload_type == "chat_completion":
            grammar_err = self.correction.process_grammar(req)
            if grammar_err is not None:
                if openai:
                    return _openai_error(
                        grammar_err.get("detail", "invalid grammar"),
                        "invalid_request_error", 422, code="invalid_grammar",
                    )
                return JSONResponse(
                    {"status": "error", "request_id": req.request_id,
                     "code": "invalid_grammar", **grammar_err},
                    status_code=422,
                )

        # Context-window pre-admission gate (M1). The proxy KNOWS the live
        # per-slot context (poller-discovered for thinker, config-seeded
        # elsewhere) and estimates input tokens anyway — an oversized prompt
        # should fail fast with an actionable message, not queue, dispatch,
        # and die as a confusing backend 400. Shadow (default): WARN + counter
        # only. Enforce (runtime flag, flipped after the shadow window shows
        # no false positives): 422 whose message embeds the canonical
        # context-overflow marker ("exceeds the available context size") so
        # chunking callers' re-chunk handling engages exactly as it does for
        # the backend's own overflow error. Known undercounts are all in the
        # SAFE direction (false-negative): the struct-policy/thinking
        # max_tokens bumps apply after this check, and image blocks contribute
        # ~0 chars to the estimate.
        if req.payload_type == "chat_completion":
            gate_cfg = self.state.config.endpoints.get(req.endpoint)
            ctx_limit = gate_cfg.context_per_slot if gate_cfg else 0
            if ctx_limit > 0:
                est_in = estimate_input_tokens(req.payload)
                mt = req.payload.get("max_tokens")
                est_out = mt if isinstance(mt, int) and mt > 0 else 0
                if est_in + est_out > ctx_limit:
                    caller = str(body.get("caller_id") or req.agent_id)
                    tally = self.state.context_overflows.setdefault(
                        req.endpoint, {"count": 0, "callers": {}, "max_est_in": 0})
                    tally["count"] += 1
                    tally["callers"][caller] = tally["callers"].get(caller, 0) + 1
                    tally["max_est_in"] = max(tally["max_est_in"], est_in)
                    err = (
                        f"request (est {est_in} input tokens + max_tokens "
                        f"{est_out}) exceeds the available context size "
                        f"({ctx_limit}/slot on {req.endpoint}) — chunk the "
                        f"input or route to a larger-context endpoint")
                    if self.state.flags.get("context_gate_enforce"):
                        if openai:
                            return _openai_error(
                                err, "invalid_request_error", 422,
                                code="context_overflow")
                        return JSONResponse(
                            {"status": "error", "request_id": req.request_id,
                             "error": err, "code": "context_overflow"},
                            status_code=422)
                    logger.warning(
                        "context gate SHADOW: %s (caller=%s call_site=%s) — "
                        "request admitted; flip context_gate_enforce for a "
                        "fast 422", err, caller, req.call_site)

        # Check deterministic cache
        cache_key = self.state.cache.cache_key(req.endpoint, req.payload)
        if cache_key:
            cached = self.state.cache.get(cache_key)
            if cached:
                # Phase 4.1: count cache hits in metrics. They bypass dispatch,
                # so without this they're invisible in /v1/metrics and real
                # traffic is undercounted (the cache's own hit_rate aside).
                self.state.metrics.record(MetricsSample(
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

        # On-demand backends (e.g. creative, Gemma-4-31B abliterated): acquire the anvil
        # GPU-slot dispatcher lease so the model is resident before dispatch.
        # Blocks (FIFO) behind any other on-demand service (imagegen/diarize/…)
        # holding the slot, and may cold-load for minutes — the 900s timeout
        # floor covers it. A dispatcher failure surfaces as a DEFERRABLE error
        # ("backpressure") so the caller retries instead of dispatching into a
        # dead backend. After the cache check so a cached reply never needlessly
        # wakes the model; before the circuit breaker (which _endpoint_healthy
        # short-circuits to True for on-demand endpoints).
        if self.state.on_demand.manages(req.endpoint):
            try:
                await self.state.on_demand.ensure_loaded(req.endpoint)
            except OnDemandUnavailable as exc:
                err = (f"on-demand backend {req.endpoint} could not be loaded "
                       f"— backpressure: {exc}")
                logger.warning("on_demand ensure_loaded failed: %s", err)
                if openai:
                    return _openai_error(
                        err, "backend_unavailable", 503, code="on_demand_unavailable")
                return JSONResponse(
                    {"status": "error", "request_id": req.request_id,
                     "error": err, "code": "on_demand_unavailable"},
                    status_code=503)

        # Circuit breaker (Phase 1.2): when the backend is marked unhealthy,
        # fast-fail interactive/foreground submits immediately with a DEFERRABLE
        # error instead of queuing them to wait out their full deadline; let
        # background work queue so it defers until the backend recovers. (Cache
        # hits above are served regardless — they don't need the backend.)
        if not self.health.endpoint_healthy(req.endpoint) and req.band != PriorityBand.BACKGROUND:
            # Phase 5F: distinguish an operator drain (planned) from an
            # auto-circuit trip (backend unreachable). Both are DEFERRABLE
            # ("circuit open" / "backpressure" are is_deferrable_llm_error
            # markers) so the caller retries; the wording just aids triage.
            if normalize_endpoint(req.endpoint) in self.state.paused_endpoints:
                err = f"backend {req.endpoint} paused for maintenance (drain) — backpressure"
                code = "draining"
            else:
                err = f"backend {req.endpoint} unavailable (circuit open)"
                code = "circuit_open"
            if openai:
                return _openai_error(err, "backend_unavailable", 503, code=code)
            return JSONResponse(
                {"status": "error", "request_id": req.request_id, "error": err,
                 "code": code},
                status_code=503,
            )

        # Load-shed / backpressure (Phase 2.4): under sustained saturation, shed
        # NON-interactive work with 429 + Retry-After so callers defer instead of
        # all queuing until their deadlines and 504ing together. Interactive is
        # never shed.
        if req.band != PriorityBand.INTERACTIVE:
            snap = self.state.scheduler.endpoint_snapshot(req.endpoint)
            band_key = req.band.name.lower()
            if snap.get("queue_by_band", {}).get(band_key, 0) >= self.state.shed_depth:
                err = f"backpressure: {req.endpoint} {band_key} queue saturated"
                retry_after = self.health.retry_after_s(req.endpoint)
                if openai:
                    resp = _openai_error(err, "backpressure", 429, code="backpressure")
                else:
                    resp = JSONResponse(
                        {"status": "error", "request_id": req.request_id, "error": err,
                         "code": "backpressure"},
                        status_code=429)
                resp.headers["Retry-After"] = str(retry_after)
                return resp

        # Streaming vs non-streaming
        if req.stream:
            return await self.handle_streaming_submit(req, openai=openai)
        else:
            return await self.handle_sync_submit(req, cache_key, openai=openai)
    async def handle_sync_submit(
        self, req: QueuedRequest, cache_key: str | None, *, openai: bool = False,
    ) -> Response:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self.state.pending_futures[req.request_id] = future

        # Thinking option: honor a per-request `thinking:true` opt-in (enable
        # native <think> on vLLM + generous budget bump). No-op otherwise.
        self.correction.apply_thinking(req)

        self.state.scheduler.enqueue(req)
        self.state.queue_db.persist_enqueue(req)
        self.state.dispatch_event.set()

        try:
            result = await asyncio.wait_for(
                future, timeout=req.timeout_s,
            )
        except asyncio.TimeoutError:
            self.state.scheduler.cancel(req.request_id)
            self.state.queue_db.persist_expire(req.request_id)
            self.state.pending_futures.pop(req.request_id, None)
            self.state.thinking_active.pop(req.request_id, None)
            # The caller's deadline fired — work may still be in flight.
            self.record_timeout_event(req, layer="client_wait", elapsed_s=req.timeout_s)
            if openai:
                return _openai_error(
                    f"proxy timeout after {req.timeout_s:.0f}s", "proxy_timeout", 504,
                    code="proxy_timeout",
                )
            return JSONResponse(
                {"error": "timeout", "request_id": req.request_id,
                 "code": "proxy_timeout"},
                status_code=504,
            )
        finally:
            self.state.pending_futures.pop(req.request_id, None)

        # Admission timeout (scheduler callback) resolves the future with a
        # timeout result — already logged there; surface the same 504.
        if result.get("status") == "timeout":
            self.state.thinking_active.pop(req.request_id, None)
            if openai:
                return _openai_error(
                    f"proxy timeout after {req.timeout_s:.0f}s", "proxy_timeout", 504,
                    code="proxy_timeout",
                )
            return JSONResponse(
                {"error": "timeout", "request_id": req.request_id,
                 "code": "proxy_timeout"},
                status_code=504,
            )

        # Uniform non-streaming correction (Step 4a): thinking-finalize →
        # degeneration-correct → shadow-egress-detect, in that load-bearing order
        # (finalizers first so the guard/detector/cache see corrected content;
        # degeneration before the detector + cache so a never-cache-degenerate flag
        # is set first). Byte-identical to the prior inline sequence — see
        # Correction.apply.
        await self.correction.apply(req, result)

        # Cache if deterministic — but NEVER cache an unrecovered degenerate
        # response (don't serve the same garbage for the cache TTL).
        degen_unrecovered = result.pop("_degenerate_unrecovered", False)
        if cache_key and result.get("status") == "ok" and not degen_unrecovered:
            self.state.cache.put(cache_key, result.get("response", {}))

        # OpenAI consumers get the bare chat.completion (or an OpenAI-shaped
        # error); internal consumers get the submit envelope (unchanged).
        if openai:
            if result.get("status") == "ok":
                return JSONResponse(result.get("response", {}))
            return _openai_error(
                result.get("error", "backend error"), "backend_error", 502,
                code="backend_error",
            )

        if result.get("status") == "ok":
            return JSONResponse(result, status_code=200)
        result.setdefault("code", "backend_error")
        return JSONResponse(result, status_code=502)
    async def handle_streaming_submit(
        self, req: QueuedRequest, *, openai: bool = False,
    ) -> Response:
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self.state.pending_streams[req.request_id] = queue

        self.state.scheduler.enqueue(req)
        self.state.queue_db.persist_enqueue(req)
        self.state.dispatch_event.set()

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
                    # Internal envelope path: re-emit every event. Under uniform
                    # correction (Step 4a), route tool-call CHUNK frames through the
                    # SAME sanitizer the OpenAI door uses so internal /v1/submit
                    # consumers (agents via ProxyLLMClient) get the qwen3_xml
                    # phantom/truncated-arg fix too — not just the OpenAI door.
                    # Default OFF == byte-identical (internal streams emit raw).
                    if (uniform_correction_enabled()
                            and event.get("type") == "chunk" and "data" in event):
                        event = {**event, "data": toolcall_sanitizer.feed(event["data"])}
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
                self.record_timeout_event(req, layer="stream", elapsed_s=req.timeout_s)
            finally:
                # Phase 5B.1: the SSE consumer is gone (client disconnect, our
                # own timeout, or normal completion). Cancel the producer
                # dispatch task if it's still running — otherwise a producer
                # blocked on a full stream_q.put (maxsize=256, consumer no longer
                # draining) wedges forever, holding the scheduler slot until the
                # proxy restarts. That leak cascaded all 4 companion slots into a
                # full endpoint jam (2026-05-31). The CancelledError branch in
                # _execute_dispatch records the completion → frees the slot.
                self.state.pending_streams.pop(req.request_id, None)
                producer = self.state.inflight_tasks.get(req.request_id)
                if producer is not None and not producer.done():
                    producer.cancel()

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    async def scheduler_loop(self) -> None:
        """Main scheduling loop — runs dispatch on every event or interval.

        The iteration body is guarded: one poisoned tick (a scheduler bug, a
        corrupt request) must not kill dispatching for the WHOLE fleet. On an
        escaped exception we log CRITICAL and back off 1s; CancelledError
        (shutdown) propagates."""
        while True:
            try:
                await asyncio.wait_for(
                    self.state.dispatch_event.wait(),
                    timeout=self.state.config.drr_tick_interval_s * 10,
                )
            except asyncio.TimeoutError:
                pass
            self.state.dispatch_event.clear()

            try:
                now = time.monotonic()
                decisions = self.state.scheduler.tick(now)

                for decision in decisions:
                    rid = decision.request.request_id
                    task = asyncio.create_task(self.execute_dispatch(decision))
                    self.state.inflight_tasks[rid] = task
                    task.add_done_callback(
                        lambda t, _rid=rid: self.state.inflight_tasks.pop(_rid, None)
                    )
            except Exception:  # noqa: BLE001 — keep the fleet dispatching
                logger.critical(
                    "scheduler loop iteration failed — dispatch continues after "
                    "1s backoff", exc_info=True,
                )
                await asyncio.sleep(1.0)
    async def execute_dispatch(self, decision: DispatchDecision) -> None:
        """Execute a dispatch decision: call the backend and resolve the
        caller's future/stream."""
        req = decision.request
        ep_cfg = self.state.config.endpoints.get(req.endpoint)
        if not ep_cfg:
            self.state.resolve_error(req, f"unknown endpoint {req.endpoint}")
            return

        if (
            ep_cfg.slot_affinity
            and req.payload_type == "chat_completion"
            and req.session_id
            and req.band == PriorityBand.INTERACTIVE
        ):
            slot_n = ep_cfg.dispatch_concurrency_cap or ep_cfg.max_slots or 1
            slot_id = (
                int.from_bytes(
                    hashlib.md5(req.session_id.encode()).digest()[:4], "little"
                ) % slot_n
            )
            req.payload = {**req.payload, "id_slot": slot_id}
            logger.debug(
                "slot_affinity: session=%s → id_slot=%d (n=%d) on %s",
                req.session_id, slot_id, slot_n, ep_cfg.role,
            )

        self.state.queue_db.persist_dispatch(req.request_id)

        # Real-time fan-out — a request just started executing. Lets /v1/stream
        # subscribers render the live in-flight board the instant work begins
        # (the `call.completed` event later removes it). Synchronous + no-op when
        # no clients; guarded so a fault never disturbs the dispatch path.
        try:
            self.state.sse.publish("call.dispatched", {
                "request_id": req.request_id,
                "agent": req.agent_id,
                "endpoint": req.endpoint,
                "call_site": req.call_site,
                "priority": req.priority.name,
                "band": req.band.name.lower(),
                "input_tokens": req.est_input_tokens,
                "estimated_remaining_s": round(req.estimated_cost_ss, 2),
                "queue_wait_ms": round(decision.queue_wait_ms, 1),
                "ts": time.time(),
            })
        except Exception as exc:  # noqa: BLE001
            logger.debug("sse call.dispatched publish failed for %s: %s", req.request_id, exc)

        t0 = time.monotonic()
        try:
            if req.stream:
                await self.execute_streaming(req, ep_cfg, decision)
            else:
                await self.execute_sync(req, ep_cfg, decision)
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
            self.state.slot_leak_reclaimed += 1
            logger.info(
                "dispatch %s cancelled (consumer gone / drain) after %.1fs — "
                "reclaiming slot", req.request_id, duration,
            )
            self.state.resolve_error(
                req, "llm proxy stream cancelled (consumer disconnected)")
            self.record_completion(req, decision, duration, 0, 0, "cancelled")
            raise
        except Exception as exc:
            duration = time.monotonic() - t0
            logger.error(
                "dispatch %s failed: %s", req.request_id, exc,
            )
            self.state.resolve_error(req, str(exc))
            self.record_completion(req, decision, duration, 0, 0, "error")
    async def execute_sync(
        self,
        req: QueuedRequest,
        ep_cfg: EndpointConfig,
        decision: DispatchDecision,
    ) -> None:
        attempts = 0
        # Local dispatch payload: the empty-completion rescue swaps in a COPY
        # with min_tokens on the retry — req.payload itself is corpus-persisted
        # and must stay the caller's bytes.
        dispatch_payload = req.payload
        rescue_armed = False
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
                self.state.resolve_error(req, msg)
                self.record_completion(req, decision, 0.0, 0, 0, "timeout")
                self.record_timeout_event(
                    req, layer="backend", elapsed_s=0.0,
                    queue_wait_ms=decision.queue_wait_ms, emit_metrics_and_log=False,
                )
                return

            t0 = time.monotonic()
            try:
                resp = await self.state.backend.call(
                    ep_cfg, dispatch_payload, req.payload_type,
                    req.request_id, timeout_s=max(1.0, remaining),
                )
            except BackendTimeout as exc:
                duration = time.monotonic() - t0
                self.state.resolve_error(req, str(exc))
                self.record_completion(req, decision, duration, 0, 0, "timeout")
                self.record_timeout_event(
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
                    self.correction.is_transient_backend_error(exc)
                    and attempts <= self.state.transient_retry_max
                    and (req.timeout_deadline - time.monotonic()) > _MIN_RETRY_BUDGET_S
                    and self.health.endpoint_healthy(req.endpoint)
                ):
                    # Empty-completion rescue: a position-0-EOS degeneration is
                    # DETERMINISTIC for its prompt — re-dispatching the same
                    # bytes can't recover it. Mask EOS for the first N tokens
                    # on the retry (vLLM min_tokens; llama.cpp ignores it).
                    if (
                        req.payload_type == "chat_completion"
                        and "empty completion" in (exc.detail or "")
                    ):
                        dispatch_payload = {
                            **req.payload,
                            "min_tokens": _EMPTY_RESCUE_MIN_TOKENS,
                        }
                        rescue_armed = True
                        self.state.empty_rescue_attempts += 1
                        logger.warning(
                            "empty completion on %s (attempt %d) — retrying "
                            "with min_tokens=%d (EOS-degeneration rescue)",
                            ep_cfg.role, attempts, _EMPTY_RESCUE_MIN_TOKENS,
                        )
                    else:
                        logger.warning(
                            "transient backend error on %s (attempt %d) — retrying: %s",
                            ep_cfg.role, attempts, exc,
                        )
                    await asyncio.sleep(_RETRY_BACKOFF_S)
                    continue
                self.state.resolve_error(req, str(exc))
                self.record_completion(req, decision, duration, 0, 0, "error")
                return

            duration = time.monotonic() - t0

            if rescue_armed:
                # The min_tokens re-dispatch produced a real response where the
                # plain dispatch got 1-token EOS — the rescue worked.
                self.state.empty_rescue_recovered += 1
                logger.info(
                    "empty-completion rescue RECOVERED on %s (call_site=%s, "
                    "output_tokens=%d)", ep_cfg.role, req.call_site,
                    resp.output_tokens,
                )
                rescue_armed = False

            # Phase 1.1 truncation integrity. finish_reason=length means the
            # backend hit max_tokens mid-output. For a STRUCTURED request
            # (grammar / response_format / structured_outputs) the body is almost
            # certainly broken/unparseable JSON — fail loud with a DEFERRABLE
            # error so the caller re-chunks instead of recording garbage, and
            # never cache it (status != ok). Free-form truncation is benign.
            if resp.finish_reason == "length" and self.correction.request_is_structured(req):
                self.state.resolve_error(
                    req,
                    f"backend {ep_cfg.role} truncated structured output "
                    f"(finish_reason=length, output_tokens={resp.output_tokens})",
                )
                self.record_completion(
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

            future = self.state.pending_futures.get(req.request_id)
            if future and not future.done():
                future.set_result(result)

            capture_response = resp.body if req.payload_type == "chat_completion" else None
            self.record_completion(
                req, decision, duration,
                resp.input_tokens, resp.output_tokens, "ok",
                response_body=capture_response, finish_reason=resp.finish_reason,
            )

            # Shadow backend A/B: fire-and-forget to the shadow if configured
            if ep_cfg.shadow_host and ep_cfg.shadow_port:
                asyncio.create_task(self.execute_shadow(
                    req, ep_cfg, decision, resp,
                ))
            return
    async def execute_shadow(
        self,
        req: QueuedRequest,
        ep_cfg: EndpointConfig,
        decision: DispatchDecision,
        primary_resp: BackendResponse,
    ) -> None:
        """Send the same request to the shadow backend and record the
        comparison result. Never affects the primary caller."""
        shadow_resp = await self.state.backend.call_shadow(
            ep_cfg.shadow_host, ep_cfg.shadow_port,
            req.payload, req.payload_type,
            req.request_id, timeout_s=req.timeout_s,
        )
        if shadow_resp is None:
            logger.debug("shadow dispatch %s failed", req.request_id)
            return

        self.state.queue_db.persist_complete(
            f"shadow-{req.request_id}", req.agent_id, req.endpoint,
            req.call_site, int(req.priority),
            shadow_resp.input_tokens, shadow_resp.output_tokens,
            shadow_resp.duration_s, decision.queue_wait_ms, "ok",
            payload=req.payload, response=shadow_resp.body,
        )
    async def execute_streaming(
        self,
        req: QueuedRequest,
        ep_cfg: EndpointConfig,
        decision: DispatchDecision,
    ) -> None:
        stream_q = self.state.pending_streams.get(req.request_id)
        if not stream_q:
            # No consumer for this stream (it died with a previous process, or
            # any future no-consumer path). The dispatch already claimed a
            # scheduler slot — record a completion so it's FREED instead of
            # leaking until the next proxy restart (recovery now drops queued
            # stream rows, so this is the belt-and-braces layer).
            logger.warning(
                "streaming dispatch %s has no consumer — reclaiming slot",
                req.request_id)
            self.record_completion(req, decision, 0.0, 0, 0, "cancelled")
            return

        await stream_q.put({
            "type": "admitted",
            "queue_wait_ms": round(decision.queue_wait_ms, 1),
        })

        # Streaming usage accounting (2026-06-10): without
        # stream_options.include_usage most backends emit NO usage in the
        # stream, so streaming completions recorded 0 tokens (live: 495
        # zero-token orchestrator rows/day) — undercounting every usage/savings
        # rollup and starving the cost model. Inject it into the BACKEND
        # payload on a LOCAL COPY (req.payload is corpus-persisted and must
        # stay the caller's bytes). If the CALLER didn't ask for usage, the
        # usage-only frame (usage present, empty choices) is captured and
        # DROPPED below so strict OpenAI clients see a byte-identical stream.
        # Kill-switch: runtime flag inject_stream_usage.
        payload = req.payload
        so = payload.get("stream_options") if isinstance(payload, dict) else None
        client_wants_usage = bool(isinstance(so, dict) and so.get("include_usage"))
        inject_usage = (
            req.payload_type == "chat_completion"
            and not client_wants_usage
            and self.state.flags.get("inject_stream_usage")
        )
        if inject_usage:
            so = dict(so) if isinstance(so, dict) else {}
            so["include_usage"] = True
            payload = {**payload, "stream_options": so}

        t0 = time.monotonic()
        input_tokens = 0
        output_tokens = 0
        last_finish_reason: str | None = None
        ttft_ms: float | None = None  # Phase 4.1 — time to first token
        # Step 4a: accumulate assistant content across chunks for end-of-stream
        # DETECTION (degeneration loop / silent grammar-drop) — gated on the flag
        # so the flag-OFF path adds zero per-chunk work and stays byte-identical.
        uniform_on = uniform_correction_enabled()
        accumulated_content = ""

        # Phase 1.5: bound the stream to the caller's remaining deadline so an
        # abandoned stream can't hold its slot past the SLA.
        stream_timeout = max(1.0, req.timeout_deadline - time.monotonic())
        # Phase 5C: time-to-first-token watchdog. Start with a SHORT deadline; on
        # the first token, reschedule to the full SLA. A 0-token hang then aborts
        # in ~TTFT seconds (freeing the slot) instead of burning the whole 180s.
        ttft_deadline_s = min(_STREAM_TTFT_DEADLINE_S, stream_timeout)
        gap_deadline_s = min(_STREAM_INTERTOKEN_GAP_S, stream_timeout)
        loop = asyncio.get_event_loop()
        last_chunk_at = t0
        try:
            async with asyncio.timeout(ttft_deadline_s) as _cm:
                async for event in self.state.backend.stream(
                    ep_cfg, payload, req.payload_type,
                    req.request_id, timeout_s=stream_timeout,
                ):
                    if event.event_type == "chunk":
                        now_m = time.monotonic()
                        if ttft_ms is None:
                            ttft_ms = (now_m - t0) * 1000.0
                        last_chunk_at = now_m
                        # Per-token no-progress watchdog: each token resets a gap
                        # deadline bounded by the caller's remaining SLA. A
                        # 0-token hang aborts in ~TTFT; a MID-STREAM stall aborts
                        # in ~gap — both free the slot instead of burning 180s.
                        remaining = stream_timeout - (now_m - t0)
                        if remaining <= 0:
                            raise asyncio.TimeoutError
                        _cm.reschedule(loop.time() + min(gap_deadline_s, remaining))
                        usage_only = False
                        if event.parsed:
                            usage = event.parsed.get("usage")
                            choices = event.parsed.get("choices") or []
                            if usage:
                                input_tokens = usage.get("prompt_tokens", input_tokens)
                                output_tokens = usage.get("completion_tokens", output_tokens)
                                # The synthetic usage frame (usage, no choices)
                                # exists because WE injected include_usage —
                                # capture it but never relay it to a client
                                # that didn't ask. Usage riding on a normal
                                # content/finish chunk passes through.
                                usage_only = inject_usage and not choices
                            if choices and isinstance(choices[0], dict):
                                fr = choices[0].get("finish_reason")
                                if fr:
                                    last_finish_reason = fr
                                if uniform_on:
                                    delta = choices[0].get("delta")
                                    piece = delta.get("content") if isinstance(delta, dict) else None
                                    if isinstance(piece, str):
                                        accumulated_content += piece
                        if not usage_only:
                            await stream_q.put({
                                "type": "chunk",
                                "data": event.data,
                            })
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
            elif (isinstance(exc, asyncio.TimeoutError)
                  and (time.monotonic() - last_chunk_at) >= gap_deadline_s - 0.5):
                err = ("backend stalled mid-stream (no token for "
                       f"{gap_deadline_s:.0f}s) — backpressure")
            else:
                err = f"stream deadline exceeded — backpressure ({exc})"
            await stream_q.put({"type": "error", "error": err})
            duration = time.monotonic() - t0
            self.record_completion(req, decision, duration, input_tokens, output_tokens, "timeout")
            self.record_timeout_event(
                req, layer="stream", elapsed_s=duration,
                queue_wait_ms=decision.queue_wait_ms, emit_metrics_and_log=False,
            )
            return
        except Exception as exc:
            await stream_q.put({"type": "error", "error": str(exc)})
            duration = time.monotonic() - t0
            self.record_completion(req, decision, duration, input_tokens, output_tokens, "error")
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
            if last_finish_reason == "length" and self.correction.request_is_structured(req)
            else "ok"
        )
        self.record_completion(
            req, decision, duration, input_tokens, output_tokens, status,
            finish_reason=last_finish_reason,
        )
        # Step 4a: uniform streaming detection over the reassembled content
        # (degeneration loop / silent grammar-drop). Detect-only + fail-open +
        # self-gated on the flag; no-op when uniform correction is off.
        if uniform_on:
            self.correction.finalize_stream(req, accumulated_content, last_finish_reason)
    def on_admission_timeout(self, req: QueuedRequest) -> None:
        """Scheduler callback: a request expired while still queued. Log
        it and release the caller promptly with a timeout result (instead
        of letting it wait out its own — identical — deadline)."""
        elapsed = time.monotonic() - req.enqueued_at
        self.record_timeout_event(req, layer="admission", elapsed_s=elapsed)
        future = self.state.pending_futures.get(req.request_id)
        if future and not future.done():
            future.set_result({
                "request_id": req.request_id,
                "status": "timeout",
                "error": "timeout",
            })
        stream_q = self.state.pending_streams.get(req.request_id)
        if stream_q:
            try:
                stream_q.put_nowait({"type": "error", "error": "timeout"})
            except asyncio.QueueFull:
                pass
        self.state.queue_db.persist_expire(req.request_id)
    def record_completion(
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

        # On-demand: a request to this endpoint reached a terminal outcome
        # (ok/error/timeout/cancel) — release its in-flight hold so the idle
        # watchdog can eventually drop the dispatcher lease. No-op for always-on
        # endpoints. This is also the natural attach point for a future explicit
        # "unload when done" agent signal (release immediately on that request).
        self.state.on_demand.request_done(req.endpoint)

        # Report to scheduler
        self.state.scheduler.complete(
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
        kind = _PAYLOAD_KIND.get(req.payload_type, "llm")
        self.state.queue_db.persist_complete(
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
            kind=kind,
        )

        # Log
        self.state.request_logger.log(RequestLogRecord(
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
            estimated_input_tokens=req.est_input_tokens or estimate_input_tokens(req.payload),
            max_output_tokens=req.payload.get("max_tokens", 0),
            session_id=req.session_id,
            turn_id=req.turn_id,
            caller_id=req.caller_id,
        ))

        # Metrics
        self.state.metrics.record(MetricsSample(
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
            self.record_timeout_shadow(req, now, duration_s, output_tokens, status)
        except Exception as exc:  # noqa: BLE001
            logger.warning("timeout shadow record failed for %s: %s", req.request_id, exc)

        # Real-time fan-out — emit the completed call to /v1/stream subscribers.
        # Synchronous + no-op when no clients; guarded so a fault never disturbs
        # the caller or the scheduler.
        try:
            self.state.sse.publish("call.completed", {
                "request_id": req.request_id,
                "agent": req.agent_id,
                "endpoint": req.endpoint,
                "call_site": req.call_site,
                "kind": kind,
                "priority": req.priority.name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "duration_s": round(duration_s, 3),
                "queue_wait_ms": round(decision.queue_wait_ms, 1),
                "status": status,
                "ts": time.time(),
            })
        except Exception as exc:  # noqa: BLE001
            logger.debug("sse call.completed publish failed for %s: %s", req.request_id, exc)

        # Trigger scheduler (a slot freed up)
        self.state.dispatch_event.set()
    def record_timeout_shadow(
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
        est_in = req.est_input_tokens or estimate_input_tokens(req.payload)
        est_out = int(req.payload.get("max_tokens", 0) or 0)
        priority = int(req.priority)

        # Advice reflects history BEFORE this sample is folded in.
        advice = self.state.timeout_model.advise(req.endpoint, priority, est_in, est_out)

        self.state.timeout_model.record(
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
        self.state.queue_db.persist_timeout_shadow(
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
    def record_timeout_event(
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
        if rid in self.state.timed_out_ids:
            return  # already counted this request's timeout
        self.state.timed_out_ids.add(rid)
        if len(self.state.timed_out_ids) > 8192:
            self.state.timed_out_ids.clear()  # bounded; rare duplicate after reset is harmless

        try:
            now = time.monotonic()
            snap = self.state.scheduler.endpoint_snapshot(req.endpoint)
            est_in = req.est_input_tokens or estimate_input_tokens(req.payload)
            est_out = int(req.payload.get("max_tokens", 0) or 0)
            priority = int(req.priority)
            ep_cfg = self.state.config.endpoints.get(normalize_endpoint(req.endpoint))
            context_window = ep_cfg.context_per_slot if ep_cfg else 0
            context_used_pct = (
                round(est_in / context_window * 100.0, 1) if context_window else None
            )
            try:
                recommended_ms = self.state.timeout_model.advise(
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
                self.state.metrics.record(MetricsSample(
                    timestamp=now,
                    endpoint=req.endpoint,
                    agent_id=req.agent_id,
                    priority=req.priority.name,
                    queue_wait_ms=(queue_wait_ms or 0.0),
                    backend_latency_ms=0.0,
                    status="timeout",
                    slot_seconds=0.0,
                    # `under` = fired below the proxy's recommended deadline; a
                    # client-side give-up, excluded from the endpoint_stalled
                    # backend-stall heuristic (best-effort sub-floor callers).
                    premature=under,
                ))
                self.state.request_logger.log(RequestLogRecord(
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

            self.state.queue_db.persist_timeout_event(
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
