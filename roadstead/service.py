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
    inject_reason_field,
    normalize_and_validate,
    recover_structured_object,
    root_object_keys,
    strip_top_field,
    verify_conformance,
)
from .config import (
    CLASS_TO_ROLE,
    LLMPriority,
    PriorityBand,
    ProxyConfig,
    StructKind,
    StructMechanism,
    StructMode,
    normalize_endpoint,
    shadow_egress_detect_enabled,
    struct_policy_for,
    thinking_enabled,
    thinking_reasoning_budget,
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

    THIRD DEFECT (truncation — the residual leak this version closes). When a
    big tool argument (a long ``bash`` command / heredoc) is cut off at
    ``max_tokens`` mid-JSON-string, the arguments NEVER form a complete value AND
    vLLM mislabels ``finish_reason`` as ``tool_calls`` (not ``length``) — so the
    old per-fragment passthrough emitted broken JSON the client's ``JSON.parse``
    then threw on (measured: 6/10 big-arg turns leaked through). No structural
    recovery is possible — the data is genuinely incomplete.

    THE FIX — buffer-then-emit-atomically, applied at the proxy so we neither
    patch the (custom) GB10 vLLM nor give up MTP throughput. Per (choice, index)
    slot we ACCUMULATE the argument fragments and emit NOTHING until the buffer
    forms ONE complete JSON value; then we emit the whole call (string name +
    complete args) in a single delta and ignore any trailing junk. A
    truncated/partial/broken argument therefore never reaches the client. At the
    finish chunk we finalize every still-pending slot for the choice: a named
    slot with empty args is a legitimate no-arg call (emit ``{}``); a named slot
    with INCOMPLETE args is truncated → dropped, and ``finish_reason`` is
    relabeled ``length`` so the client retries instead of dispatching a
    half-command; a name-less slot (the phantom) is dropped silently. Indices are
    NOT renumbered — the AI SDK keys tool calls by ``id`` and uses the numeric
    index only as an accumulation slot, so a dropped phantom's hole is harmless.
    (Emitting a call atomically on completion rather than streaming arg fragments
    is invisible to the AI SDK, which dispatches only once args parse anyway.)

    ``feed(data)`` takes one raw backend SSE ``data:`` payload and returns the
    payload to emit. PURE PASS-THROUGH (the ORIGINAL string object, no parse) for
    the >99% of chunks with no ``tool_calls`` while no call is mid-accumulation.
    Defensive: never raises — on any malformed/odd shape it returns the original
    bytes.
    """

    def __init__(self) -> None:
        # (choice_index, tool_index) -> {id, type, name, buf, done}
        self._slots: dict = {}

    @staticmethod
    def _complete_value(text: str):
        """If ``text`` (leading whitespace allowed) begins with a COMPLETE JSON
        value, return ``(value_str, True)`` — trimming any trailing junk such as
        the extra ``}`` the parallel-call bug appends (``raw_decode`` correctly
        ignores ``}`` inside string values). Else ``(None, False)``. An empty /
        whitespace-only buffer is NOT complete (arguments may still be
        streaming)."""
        if text.strip() == "":
            return None, False
        try:
            _, end = json.JSONDecoder().raw_decode(text)
        except ValueError:
            return None, False
        return text[:end], True

    def _pending(self) -> bool:
        """True while any slot is still accumulating — so the finish chunk and
        intervening content chunks get inspected to finalize. False in the
        steady state, which keeps ``feed`` a pure pass-through."""
        return any(not st["done"] for st in self._slots.values())

    def feed(self, data: str) -> str:
        # Fast path: only engage when this chunk carries tool_calls OR a call is
        # mid-accumulation (then we must watch for its completion + the finish
        # chunk). >99% of chunks short-circuit here untouched (original object).
        if '"tool_calls"' not in data and not self._pending():
            return data
        try:
            obj = json.loads(data)
            if not isinstance(obj, dict):
                return data
            changed = False
            for ch in (obj.get("choices") or []):
                if not isinstance(ch, dict):
                    continue
                ci = ch.get("index", 0)
                delta = ch.get("delta") if isinstance(ch.get("delta"), dict) else None
                emit: list = []  # (idx, slot, args) — calls to emit on THIS chunk

                # 1) Buffer tool-call fragments per slot. Emit NOTHING until a
                #    slot's arguments form a COMPLETE JSON value, then emit the
                #    whole call (name + complete args) atomically. A partial /
                #    truncated / broken-JSON argument therefore NEVER reaches the
                #    client (defect B: vLLM truncates big args mid-string and
                #    mislabels finish_reason as tool_calls).
                if delta is not None and isinstance(delta.get("tool_calls"), list):
                    changed = True  # a tool_calls chunk is always rebuilt
                    for tc in delta["tool_calls"]:
                        if not isinstance(tc, dict):
                            continue
                        idx = tc.get("index")
                        st = self._slots.get((ci, idx))
                        if st is None:
                            st = {"id": None, "type": None, "name": None,
                                  "buf": "", "done": False}
                            self._slots[(ci, idx)] = st
                        if st["id"] is None and isinstance(tc.get("id"), str):
                            st["id"] = tc["id"]
                        if st["type"] is None and isinstance(tc.get("type"), str):
                            st["type"] = tc["type"]
                        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                        rn = fn.get("name")
                        if st["name"] is None and isinstance(rn, str) and rn != "":
                            st["name"] = rn
                        ra = fn.get("arguments")
                        if isinstance(ra, str):
                            st["buf"] += ra
                        if st["done"] or st["name"] is None:
                            continue  # phantom (no name) or already emitted
                        args, ok = self._complete_value(st["buf"])
                        if ok:
                            emit.append((idx, st, args))
                            st["done"] = True

                # 2) On the finish chunk, finalize every still-pending slot for
                #    this choice: a named slot with NO args is a legitimate
                #    no-arg call (emit "{}"); a named slot with INCOMPLETE args is
                #    TRUNCATED — drop it and relabel finish_reason to "length" so
                #    the client retries instead of dispatching a half-command or
                #    throwing on broken JSON; a name-less slot is a phantom (vLLM
                #    #39584) — drop it silently.
                fr = ch.get("finish_reason")
                if fr is not None:
                    truncated = False
                    for (cci, sidx), st in self._slots.items():
                        if cci != ci or st["done"]:
                            continue
                        if st["name"] is not None and st["buf"].strip() == "":
                            emit.append((sidx, st, "{}"))
                        elif st["name"] is not None:
                            truncated = True
                        st["done"] = True
                    if truncated and fr in ("tool_calls", "stop"):
                        ch["finish_reason"] = "length"
                        changed = True

                # 3) Rebuild this choice's tool_calls delta from completed calls.
                if delta is not None and ("tool_calls" in delta or emit):
                    if emit:
                        delta["tool_calls"] = [
                            {"index": eidx, "id": est["id"],
                             "type": est["type"] or "function",
                             "function": {"name": est["name"], "arguments": eargs}}
                            for (eidx, est, eargs) in emit
                        ]
                        changed = True
                    elif "tool_calls" in delta:
                        del delta["tool_calls"]
                        changed = True
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
from .sse_hub import DROP_SENTINEL, SSEHub

logger = logging.getLogger(__name__)

# Map the scheduler's payload_type to the completion `kind` tag (so the unified
# Inference page can group LLM sub-kinds; non-LLM calls pushed via /v1/calls/log
# carry their own kind — audio/imagegen/ocr/translate).
_PAYLOAD_KIND = {
    "chat_completion": "chat",
    "embedding": "embed",
    "rerank": "rerank",
}


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
        # Real-time fan-out (Phase 1: proxy = fleet call-metrics authority).
        # Every completion emits a `call.completed` event; the poller pushes a
        # periodic `metrics` frame. Drives the unified Inference page's usage
        # panels without 5-10s polling. No-op when nobody is subscribed.
        self._sse = SSEHub()

        # Async plumbing
        self._dispatch_event = asyncio.Event()
        self._draining = asyncio.Event()  # set during shutdown drain (Phase 2.1)
        self._pending_futures: dict[str, asyncio.Future] = {}
        self._pending_streams: dict[str, asyncio.Queue] = {}
        # Structured-output policy (reason-then-constrain): per-request injection
        # state keyed by request_id — set on dispatch, consumed (popped) on the
        # response path. {request_id: {"policy","field","grammar","location"}}.
        # Stays EMPTY unless a call_site policy is flipped on (all ship OFF).
        self._struct_inject: dict[str, dict] = {}
        self._struct_egress_failures = 0   # ACTIVE egress double-check failures
        self._struct_shadow_diffs = 0      # SHADOW non-conformances (no caller impact)
        # Thinking option (per-request native reasoning): per-request state keyed
        # by request_id — set on dispatch, consumed on the response path for
        # structured-output recovery. {request_id: {"allowed_keys": [...]}}.
        # Stays EMPTY unless a caller opts in with thinking:true.
        self._thinking_active: dict[str, dict] = {}
        self._thinking_requests = 0    # opted-in thinking requests seen
        self._thinking_clean = 0       # structured output already conformant (no fix)
        self._thinking_recovered = 0   # stray-brace artifact deterministically cleaned
        self._thinking_truncated = 0   # finish=length (raise budget) — failed safe
        self._thinking_fallback = 0    # unrecoverable structured output — failed safe
        # WS-4 shadow egress detector: per-call_site silent grammar-drop tally
        # over ALL grammar-bearing responses (read-only; NEVER mutates a
        # response). {call_site: {"checked": int, "dropped": int}}. Populated by
        # _shadow_egress_detect when COLLECTIVE_PROXY_SHADOW_EGRESS is on (default).
        self._shadow_drop: dict[str, dict] = {}
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
        # queue.db maintenance cadences (persistence cleanup): WAL truncate +
        # incremental freelist return.
        self._last_wal_checkpoint_at = 0.0
        self._last_incr_vacuum_at = 0.0
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
        self._inflight_task: asyncio.Task | None = None
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

        # Start background loops
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._poller_task = asyncio.create_task(self._capacity_poller_loop())
        self._inflight_task = asyncio.create_task(self._inflight_stream_loop())

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

    # ----- structured-output policy (reason-then-constrain) -----

    def _apply_struct_policy(self, req: QueuedRequest) -> None:
        """Inject a leading free-text ``reason`` field into the grammar for an
        ACTIVE/SHADOW JUDGMENT call_site (non-streaming, grammar-bearing), so the
        model reasons before the constrained decision (CRANE-style), bumping
        max_tokens so the added tokens don't truncate the payload. Records the
        injection (keyed by request_id) for the response-side strip. Fully
        transparent when OFF (default), non-judgment, streaming, or no-grammar —
        all early-out. Never dispatches an invalid injected grammar."""
        if req.stream or req.payload_type != "chat_completion":
            return
        policy = struct_policy_for(req.call_site)
        if policy is None or policy.mode == StructMode.OFF:
            return
        if policy.kind != StructKind.JUDGMENT or policy.mechanism != StructMechanism.REASON_FIELD:
            return
        grammar, location = self._extract_grammar(req.payload)
        if grammar is None:
            return
        ep = self._config.endpoints.get(normalize_endpoint(req.endpoint))
        engine = ep.backend_engine if ep is not None else "llama.cpp"
        # b9357-class llama.cpp rejects large bounded {0,N} → emit unbounded there.
        max_chars = policy.reason_max_chars if engine == "vllm" else None
        res = inject_reason_field(grammar, field=policy.reason_field, max_chars=max_chars)
        if not res.injected:
            return
        vr = normalize_and_validate(res.grammar)
        if not vr.ok:
            logger.warning(
                "struct: injected grammar invalid (call_site=%s): %s — skipping injection",
                req.call_site, vr.error_payload()["detail"])
            return
        if location == "top":
            req.payload["grammar"] = vr.grammar
        else:
            req.payload.setdefault("extra_body", {})["grammar"] = vr.grammar
        if policy.max_tokens_bump > 0:
            cur = req.payload.get("max_tokens")
            if isinstance(cur, int) and cur > 0:
                req.payload["max_tokens"] = cur + policy.max_tokens_bump
        self._struct_inject[req.request_id] = {
            "policy": policy, "field": policy.reason_field,
            "grammar": grammar, "location": location}

    def _finalize_struct_output(self, req: QueuedRequest, result: dict) -> None:
        """Egress double-check for an injected response (mutates ``result`` in
        place). Strips the injected reason field, then verifies the stripped
        output conforms to the caller's ORIGINAL grammar. SHADOW → log any
        discrepancy but return the BASELINE (untransformed) response (zero caller
        risk). ACTIVE → on success write the stripped content back; on a
        conformance failure FAIL SAFE: convert to a structured error so the
        caller's existing retry/force-synth path engages, never passing leaky or
        garbled data. No-op when nothing was injected for this request."""
        inj = self._struct_inject.pop(req.request_id, None)
        if inj is None or result.get("status") != "ok":
            return
        response = result.get("response")
        if not isinstance(response, dict):
            return
        try:
            content = response["choices"][0]["message"]["content"]
        except Exception:  # noqa: BLE001 — not a chat.completion shape; leave untouched
            return
        if not isinstance(content, str):
            return
        field = inj["field"]
        stripped, removed = strip_top_field(content, field)
        ok, why = verify_conformance(stripped, inj["grammar"], forbid_field=field)
        if inj["policy"].mode == StructMode.SHADOW:
            if not ok or not removed:
                self._struct_shadow_diffs += 1
                logger.warning(
                    "struct SHADOW non-conformance (call_site=%s removed=%s): %s",
                    req.call_site, removed, why)
            return  # baseline response unchanged — zero caller risk
        # ACTIVE
        if not ok:
            self._struct_egress_failures += 1
            logger.error(
                "struct EGRESS FAIL (call_site=%s): %s — failing safe to caller retry",
                req.call_site, why)
            result["status"] = "error"
            result["error"] = f"structured-output egress check failed: {why}"
            result.pop("response", None)
            return
        # success — write the stripped content back (copy to avoid mutating shared refs)
        try:
            choices = list(response.get("choices") or [])
            ch0 = dict(choices[0]); msg = dict(ch0.get("message") or {})
            msg["content"] = stripped; ch0["message"] = msg; choices[0] = ch0
            new_resp = dict(response); new_resp["choices"] = choices
            result["response"] = new_resp
        except Exception:  # noqa: BLE001 — never break the response on a strip-write error
            logger.exception("struct: strip-write failed (call_site=%s)", req.call_site)

    def _shadow_egress_detect(self, req: "QueuedRequest", result: dict) -> None:
        """WS-4: SHADOW silent-drop detector over ALL grammar-bearing responses.

        Runs ``verify_conformance`` on every grammar-bearing structured response
        whose call_site is NOT already covered by a registry SHADOW/ACTIVE policy
        (those are handled by ``_finalize_struct_output`` — skip to avoid double
        counting). Tallies per-call_site checked/dropped so ``/v1/status`` can
        surface a silent-drop rate to health→health-verifier. This closes the
        no-silent-failure gap fleet-wide: a markdown fence / non-JSON / wrong-keys
        response means llama.cpp dropped the grammar and ran free-form.

        READ-ONLY: it never mutates ``result`` (zero caller risk), and any error
        is swallowed so the detector can never break a real response."""
        try:
            if not shadow_egress_detect_enabled():
                return
            if result.get("status") != "ok":
                return
            if req.stream or req.payload_type != "chat_completion":
                return
            # Registry SHADOW/ACTIVE call_sites already run a conformance check in
            # _finalize_struct_output; OFF/None fall through to this blanket pass.
            pol = struct_policy_for(req.call_site)
            if pol is not None and pol.mode != StructMode.OFF:
                return
            grammar, _loc = self._extract_grammar(req.payload)
            if not grammar:
                return
            resp = result.get("response", {}) or {}
            ch = (resp.get("choices") or [{}])[0]
            content = (ch.get("message", {}) or {}).get("content")
            if not isinstance(content, str) or not content:
                return
            ok, reason = verify_conformance(content, grammar)
            cs = req.call_site or "unknown"
            tally = self._shadow_drop.setdefault(cs, {"checked": 0, "dropped": 0})
            tally["checked"] += 1
            if not ok:
                tally["dropped"] += 1
                logger.warning(
                    "shadow_egress: silent grammar-drop call_site=%s endpoint=%s "
                    "reason=%s", cs, req.endpoint, reason,
                )
        except Exception:  # noqa: BLE001 — detector must never break a response
            logger.debug("shadow_egress_detect failed", exc_info=True)

    def _thinking_allowed_keys(self, payload: dict) -> list[str]:
        """Top-level object keys the structured constraint permits — used to
        anchor response recovery. Covers GBNF (top / extra_body /
        structured_outputs.grammar) and response_format json_schema. Empty list
        means 'no object-root structured constraint' (a plain thinking request —
        nothing to recover; content is already the clean answer)."""
        grammar, _ = self._extract_grammar(payload)
        if grammar is None:
            so = payload.get("structured_outputs")
            if isinstance(so, dict) and isinstance(so.get("grammar"), str):
                grammar = so["grammar"]
        if isinstance(grammar, str) and grammar.strip():
            try:
                return root_object_keys(grammar)
            except Exception:  # noqa: BLE001
                return []
        rf = payload.get("response_format")
        if isinstance(rf, dict) and rf.get("type") == "json_schema":
            sch = (rf.get("json_schema") or {}).get("schema") or {}
            props = sch.get("properties")
            if isinstance(props, dict):
                return list(props.keys())
        return []

    def _apply_thinking(self, req: QueuedRequest) -> None:
        """Request-side: honor a per-request ``thinking: true`` opt-in. On a vLLM
        (reasoning-parser) backend, enable native <think> and add a GENEROUS
        reasoning budget to max_tokens (reasoning is generated output → counts
        against the cap; operator directive is to prefer slowness over cutoffs).
        Records the request for response-side structured-output recovery. Strips
        the ``thinking`` control field (not a backend param) regardless. Fully
        transparent when not requested, feature-disabled, streaming, or non-vLLM."""
        p = req.payload
        if not isinstance(p, dict):
            return
        want = bool(p.get("thinking"))
        eb = p.get("extra_body")
        if isinstance(eb, dict):
            want = want or bool(eb.get("thinking"))
            eb.pop("thinking", None)
        p.pop("thinking", None)  # control field — never forward to the backend
        if not want or req.stream or req.payload_type != "chat_completion":
            return
        if not thinking_enabled():
            return
        ep = self._config.endpoints.get(normalize_endpoint(req.endpoint))
        engine = ep.backend_engine if ep is not None else "llama.cpp"
        if engine != "vllm":   # reasoning parser is vLLM-only; llama.cpp ignores
            return
        ck = p.get("chat_template_kwargs")
        ck = dict(ck) if isinstance(ck, dict) else {}
        ck["enable_thinking"] = True
        p["chat_template_kwargs"] = ck
        budget = thinking_reasoning_budget()
        cur = p.get("max_tokens")
        p["max_tokens"] = (cur if isinstance(cur, int) and cur > 0 else 800) + budget
        self._thinking_active[req.request_id] = {
            "allowed_keys": self._thinking_allowed_keys(p)}

    def _finalize_thinking(self, req: QueuedRequest, result: dict) -> None:
        """Response-side normalization for an opted-in thinking request (mutates
        ``result`` in place). vLLM already splits reasoning into
        ``message.reasoning`` (content stays clean of CoT). This repairs the one
        residual artifact: the bounded stray opening-brace that PR#44142's
        one-step-deferred FSM advance leaves before the constrained object. If the
        content already parses+conforms → no-op. Else deterministically recover
        the object (clean-then-verify, never guess) and write it back. If nothing
        recovers (truncation / genuine garbage) → FAIL SAFE to caller retry rather
        than pass noise. No-op when the caller didn't opt in."""
        info = self._thinking_active.pop(req.request_id, None)
        if info is None or result.get("status") != "ok":
            return
        response = result.get("response")
        if not isinstance(response, dict):
            return
        try:
            ch0 = response["choices"][0]
            content = ch0["message"]["content"]
            finish = ch0.get("finish_reason")
        except Exception:  # noqa: BLE001 — not a chat.completion shape; leave untouched
            return
        if not isinstance(content, str):
            return
        self._thinking_requests += 1
        allowed = info.get("allowed_keys") or []
        if not allowed:
            return  # plain thinking (no object-root constraint) — content is the answer
        # Already clean + conformant?
        try:
            obj = json.loads(content)
            if isinstance(obj, dict) and set(obj.keys()) <= set(allowed):
                self._thinking_clean += 1
                return
        except Exception:  # noqa: BLE001
            pass
        recovered = recover_structured_object(content, allowed_keys=allowed)
        if recovered is not None:
            self._thinking_recovered += 1
            try:
                choices = list(response.get("choices") or [])
                c0 = dict(choices[0]); msg = dict(c0.get("message") or {})
                msg["content"] = recovered; c0["message"] = msg; choices[0] = c0
                new_resp = dict(response); new_resp["choices"] = choices
                result["response"] = new_resp
            except Exception:  # noqa: BLE001 — never break the response on a write error
                logger.exception("thinking: recovery-write failed (call_site=%s)", req.call_site)
            return
        # Unrecoverable → fail safe (caller retry / 2-call), never pass noise.
        if finish == "length":
            self._thinking_truncated += 1
            why = "thinking output truncated (finish=length) — raise COLLECTIVE_PROXY_THINKING_BUDGET"
        else:
            self._thinking_fallback += 1
            why = "thinking structured output unrecoverable"
        logger.warning("thinking egress FAIL (call_site=%s): %s — failing safe", req.call_site, why)
        result["status"] = "error"
        result["error"] = f"thinking structured-output recovery failed: {why}"
        result.pop("response", None)

    async def _handle_sync_submit(
        self, req: QueuedRequest, cache_key: str | None, *, openai: bool = False,
    ) -> Response:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_futures[req.request_id] = future

        # Structured-output policy: reason-inject the grammar for managed
        # judgment call_sites (no-op when OFF). Done here (post cache-key, pre
        # enqueue) so the cache key stays the caller's ORIGINAL request and the
        # response path can strip what we inject.
        self._apply_struct_policy(req)
        # Thinking option: honor a per-request `thinking:true` opt-in (enable
        # native <think> on vLLM + generous budget bump). No-op otherwise.
        self._apply_thinking(req)

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
            self._struct_inject.pop(req.request_id, None)  # no _finalize on timeout
            self._thinking_active.pop(req.request_id, None)
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
            self._struct_inject.pop(req.request_id, None)  # no _finalize on timeout
            self._thinking_active.pop(req.request_id, None)
            if openai:
                return self._openai_error(
                    f"proxy timeout after {req.timeout_s:.0f}s", "proxy_timeout", 504,
                )
            return JSONResponse(
                {"error": "timeout", "request_id": req.request_id},
                status_code=504,
            )

        # Structured-output egress: strip the injected reason + run the
        # conformance double-check (no-op when nothing was injected). On an ACTIVE
        # egress failure this flips result→error (fail-safe to caller retry); so
        # it must run BEFORE the cache so we never cache a leaky/failed response.
        self._finalize_struct_output(req, result)
        # Thinking option: normalize the structured response (deterministic
        # stray-brace recovery; fail-safe if unrecoverable). Before the cache so a
        # repaired (or failed-safe) response is what gets cached, never the noise.
        self._finalize_thinking(req, result)
        # WS-4: SHADOW silent-drop detector over ALL grammar-bearing responses
        # (read-only; never mutates). Runs after the finalizers so it observes the
        # baseline content callers receive, and before the cache so every response
        # is seen exactly once.
        self._shadow_egress_detect(req, result)

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
                # Structured-output policy (reason-then-constrain) egress health.
                # Both stay 0 until a call_site policy is flipped on.
                "struct_egress_failures": self._struct_egress_failures,
                "struct_shadow_diffs": self._struct_shadow_diffs,
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

    def _metrics_payload(self, now: float) -> dict:
        """Rolling 5-min metrics — shared by GET /v1/metrics and the periodic
        SSE `metrics` frame."""
        return {
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
        }

    async def handle_metrics(self, request: Request) -> Response:
        return JSONResponse(self._metrics_payload(time.monotonic()))

    # ----- handlers: fleet usage (Phase 1: proxy = fleet call-metrics authority) -----

    async def handle_fleet_activity(self, request: Request) -> Response:
        window_s = _clamp_window(request.query_params.get("window", "24h"), 86400)
        bin_s = int(request.query_params.get("bin", _bin_seconds_for(window_s)))
        return JSONResponse(self._queue_db.fleet_activity(window_s, bin_s))

    async def handle_fleet_savings(self, request: Request) -> Response:
        since_q = request.query_params.get("since")
        today_start = float(since_q) if since_q and since_q.isdigit() else None
        return JSONResponse(self._queue_db.savings_summary(today_start))

    async def handle_top_callers(self, request: Request) -> Response:
        window_s = _clamp_window(request.query_params.get("window", "1h"), 3600)
        per_endpoint = max(1, min(int(request.query_params.get("per_endpoint", "5")), 20))
        return JSONResponse(self._queue_db.top_callers(window_s, per_endpoint))

    async def handle_usage(self, request: Request) -> Response:
        dimension = request.query_params.get("by", "agent")
        if dimension not in ("agent", "call_site", "endpoint", "provider"):
            dimension = "agent"
        hours = min(float(request.query_params.get("hours", "24")), 168)
        return JSONResponse({
            "dimension": dimension,
            "hours": hours,
            "rows": self._queue_db.usage_rollup(dimension, hours),
        })

    async def handle_series(self, request: Request) -> Response:
        endpoint = request.query_params.get("endpoint", "")
        if not endpoint:
            return JSONResponse({"error": "endpoint query param required"}, status_code=400)
        window_s = _clamp_window(request.query_params.get("window", "24h"), 86400)
        bin_s = int(request.query_params.get("bin", _bin_seconds_for(window_s)))
        return JSONResponse(self._queue_db.endpoint_series(endpoint, window_s, bin_s))

    # ----- handler: calls ingest (non-LLM fleet calls) -----

    async def handle_calls_log(self, request: Request) -> Response:
        """Ingest a non-LLM service call (audio/imagegen/ocr/translate) that
        never traversed the scheduler, so the proxy is the single fleet
        call-metrics store. Internal/LAN — gated by the same ACL as admin.
        Best-effort: validates the minimum, records, fans out, returns ok."""
        remote_ip = request.client.host if request.client else "unknown"
        if not self._acl.identify(remote_ip):
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
        status = str(body.get("status") or ("ok" if body.get("success", True) else "error"))
        request_id = str(body.get("request_id") or f"ext-{uuid.uuid4().hex}")
        in_tok = int(body.get("input_tokens") or 0)
        out_tok = int(body.get("output_tokens") or 0)
        latency_ms = float(body.get("latency_ms") or 0.0)
        duration_s = float(body.get("duration_s") or (latency_ms / 1000.0))
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

        self._queue_db.persist_dispatch(req.request_id)

        # Real-time fan-out — a request just started executing. Lets /v1/stream
        # subscribers render the live in-flight board the instant work begins
        # (the `call.completed` event later removes it). Synchronous + no-op when
        # no clients; guarded so a fault never disturbs the dispatch path.
        try:
            self._sse.publish("call.dispatched", {
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
        kind = _PAYLOAD_KIND.get(req.payload_type, "llm")
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
            kind=kind,
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

        # Real-time fan-out — emit the completed call to /v1/stream subscribers.
        # Synchronous + no-op when no clients; guarded so a fault never disturbs
        # the caller or the scheduler.
        try:
            self._sse.publish("call.completed", {
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

    # ----- live in-flight streamer -----

    async def _inflight_stream_loop(self) -> None:
        """Fast SSE `inflight` frame so /v1/stream subscribers see what's
        executing right now in (near) real time. The instant signals are the
        per-request `call.dispatched`/`call.completed` events; this periodic
        snapshot reconciles missed events and refreshes elapsed/queue/occupancy.
        Cheap + gated on client_count (no work when nobody's watching), so it
        never touches the dispatch hot path."""
        while True:
            try:
                if self._sse.client_count:
                    snap = self._scheduler.inflight_snapshot(time.monotonic())
                    snap["ts"] = time.time()
                    self._sse.publish("inflight", snap)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.debug("sse inflight publish failed: %s", exc)
            await asyncio.sleep(self._config.inflight_stream_interval_s)

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
                self._queue_db.cleanup_old_completions(
                    self._config.completions_retention_s)
                self._queue_db.cleanup_old_payloads(
                    self._config.payload_retention_s)
            # Periodic SSE `metrics` frame so /v1/stream subscribers get an
            # aggregate refresh between individual call.completed events
            # (no-op when nobody is subscribed).
            if self._sse.client_count:
                try:
                    self._sse.publish("metrics", self._metrics_payload(mono))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("sse metrics publish failed: %s", exc)
            # WAL TRUNCATE-checkpoint so the -wal sidecar can't camp at a burst
            # high-water mark (persistence cleanup).
            if (mono - self._last_wal_checkpoint_at
                    > self._config.wal_checkpoint_interval_s):
                self._last_wal_checkpoint_at = mono
                self._queue_db.checkpoint_truncate()
            # Return freed pages to the OS gradually (cheap; no full VACUUM lock).
            if (mono - self._last_incr_vacuum_at
                    > self._config.incremental_vacuum_interval_s):
                self._last_incr_vacuum_at = mono
                self._queue_db.incremental_vacuum(
                    self._config.incremental_vacuum_pages)
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
