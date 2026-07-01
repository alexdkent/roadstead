"""ProxyState — the shared mutable state + injected-singleton references for the
LLMProxy ``ProxyService`` (de-monolith, Phase 1 Step 3).

The composed-collaborators split (``Health`` / ``Correction`` / ``Lifecycle`` /
``ProxyHttpHandlers``) keeps the *behavior* in near-stateless collaborator
objects and the *data* here. Nearly every mutable field is written by one
cluster and read by another (esp. the HTTP ``/v1/status`` handler reads almost
every tally), so a single shared ``ProxyState`` — rather than a disjoint slice
per collaborator — is the minimal behavior-preserving move (decomposition
contract §3).

``ProxyService`` continues to expose every ``self._<field>`` name it always did,
as a data-descriptor that forwards to this object (contract §2), so the ~530
white-box tests that reach into those privates keep biting through the refactor.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from .acl import IPIdentityMap
from .agent_budget import BudgetManager
from .backend import BackendClientPool
from .coalesce import DeterministicCache
from .config import ProxyConfig
from .cost_model import CostModel
from .flags import RuntimeFlags
from .observability import RequestLogger, RollingMetrics
from .on_demand import OnDemandManager
from .queue import PersistentQueue
from .scheduler import Scheduler
from .sse_hub import SSEHub
from .timeout_model import TimeoutModel

if TYPE_CHECKING:
    from .grammar import GrammarResult
    from .scheduler import QueuedRequest


class ProxyState:
    """Shared mutable data + references to the injected singletons.

    Constructed once in ``ProxyService.__init__``. Field names drop the leading
    underscore (``state.backend``, not ``state._backend``); ``ProxyService``
    keeps the underscore-prefixed descriptor names for its frozen surface.
    """

    def __init__(self, config: ProxyConfig) -> None:
        self.config = config

        # Core components
        self.cost_model = CostModel()
        self.timeout_model = TimeoutModel(
            margin=config.timeout_advice_margin,
            window_s=config.timeout_advice_window_s,
            min_samples=config.timeout_advice_min_samples,
        )
        self.budget_mgr = BudgetManager(starvation_timeout_s=config.starvation_timeout_s)
        self.scheduler = Scheduler(config, self.cost_model, self.budget_mgr)
        self.backend = BackendClientPool()
        self.queue_db = PersistentQueue(config.queue_db_path or None)
        # On-demand endpoints (e.g. `creative`, Gemma-4-31B abliterated): the model is loaded
        # lazily under the anvil GPU-slot dispatcher lease and idle-unloaded.
        self.on_demand = OnDemandManager(config.endpoints)

        # Runtime-mutable feature flags (flags.py) — shadow→enforce switches +
        # kill-switches that must be flippable without a process restart (no
        # env gates). Reads are dict lookups; mutation only via /v1/admin/flags.
        self.flags = RuntimeFlags(config.runtime_flags_path or None)

        # Deterministic response cache (temperature=0).
        self.cache = DeterministicCache()

        # Grammar authority: cache of normalize+validate results keyed by
        # grammar hash, + a set of hashes we've already alerted on so each
        # bad grammar logs loudly once (not per request).
        self.grammar_cache: dict[str, "GrammarResult"] = {}
        self.grammar_alerted: set[str] = set()

        # Observability
        self.metrics = RollingMetrics(window_s=300.0)
        self.request_logger = RequestLogger(config.request_log_path or None)
        self.acl = IPIdentityMap.from_env()
        # Real-time fan-out (Phase 1: proxy = fleet call-metrics authority).
        # Every completion emits a `call.completed` event; the poller pushes a
        # periodic `metrics` frame. Drives the unified Inference page's usage
        # panels without 5-10s polling. No-op when nobody is subscribed.
        self.sse = SSEHub()

        # Async plumbing
        self.dispatch_event = asyncio.Event()
        self.draining = asyncio.Event()  # set during shutdown drain (Phase 2.1)
        self.pending_futures: dict[str, asyncio.Future] = {}
        self.pending_streams: dict[str, asyncio.Queue] = {}
        # Thinking option (per-request native reasoning): per-request state keyed
        # by request_id — set on dispatch, consumed on the response path for
        # structured-output recovery. {request_id: {"allowed_keys": [...]}}.
        # Stays EMPTY unless a caller opts in with thinking:true.
        self.thinking_active: dict[str, dict] = {}
        self.thinking_requests = 0    # opted-in thinking requests seen
        self.thinking_clean = 0       # structured output already conformant (no fix)
        self.thinking_recovered = 0   # stray-brace artifact deterministically cleaned
        self.thinking_truncated = 0   # finish=length (raise budget) — failed safe
        self.thinking_fallback = 0    # unrecoverable structured output — failed safe
        # WS-4 shadow egress detector: per-call_site silent grammar-drop tally
        # over ALL grammar-bearing responses (read-only; NEVER mutates a
        # response). {call_site: {"checked": int, "dropped": int}}. Populated by
        # _shadow_egress_detect when COLLECTIVE_PROXY_SHADOW_EGRESS is on (default).
        self.shadow_drop: dict[str, dict] = {}
        # Egress degeneration guard tallies (repetition-loop detect + re-dispatch).
        self.degeneration_detected = 0      # responses flagged as a repetition loop
        self.degeneration_recovered = 0     # …fixed by an anti-repetition re-dispatch
        self.degeneration_unrecovered = 0   # …still degenerate after re-dispatch(es)
        self.degeneration_by_call_site: dict[str, dict] = {}
        # Re-dispatches run OUTSIDE scheduler slot accounting (the original
        # slot was freed when the degenerate 200 completed), so bound their
        # fleet-wide concurrency with a plain counter — single event loop, no
        # lock needed; NOT a semaphore (waiting would queue caller responses).
        self.degen_redispatch_inflight = 0
        # Empty-completion (position-0-EOS) rescue tallies — see
        # _EMPTY_RESCUE_MIN_TOKENS. attempts = retries dispatched with
        # min_tokens; recovered = those that produced a real response.
        self.empty_rescue_attempts = 0
        self.empty_rescue_recovered = 0
        # Dedupe set so a single request that races across two timeout
        # layers (e.g. admission expiry + client-wait) is logged once.
        self.timed_out_ids: set[str] = set()
        # In-flight dispatch tasks keyed by request_id (Phase 1.5 / 2.1). Lets
        # the drain path await them on shutdown; the slot-leak fix is the
        # deadline-bound backend call in _execute_sync/_execute_streaming.
        self.inflight_tasks: dict[str, asyncio.Task] = {}
        # Per-endpoint circuit-breaker health (Phase 1.2). An endpoint flips
        # unhealthy only after consecutive capacity-probe failures CONFIRMED by a
        # failed /health probe — latency/saturation never flips it
        # (alert-don't-kill). While unhealthy the scheduler defers its queue.
        self.endpoint_health: dict[str, dict] = {
            ep: {"healthy": True, "consecutive_failures": 0, "unhealthy_since": None}
            for ep in config.endpoints
        }
        self.health_fail_threshold = 3
        # Phase 5F — operator drain: endpoints an operator has explicitly PAUSED
        # for maintenance (e.g. a vLLM restart to change --max-model-len). A
        # paused endpoint reads as unhealthy (→ background defers, interactive
        # fast-fails deferrably) so NO request hits the backend while it's down,
        # WITHOUT the ~30s auto-circuit-trip lag. Distinct from the auto-circuit
        # so a planned drain never fires the endpoint_paused ERROR alert. The
        # poller skips paused endpoints; /resume hands them back to the poller.
        self.paused_endpoints: set[str] = set()
        # Step 4b — rate-windowed per-endpoint cooldown. Complements the
        # consecutive-fail circuit above: a FLAKY backend (intermittent 5xx that
        # never strings ``health_fail_threshold`` in a row) trips THIS instead.
        # Per endpoint: recent backend-fault dispatch-failure timestamps (sliding
        # window), the monotonic time until which it's cooled (enforce only), and a
        # trip counter for the shadow report. All empty/off by default → the
        # feature is byte-identical until a cooldown flag is set.
        self.endpoint_failure_times: dict[str, list[float]] = {}
        self.endpoint_cooldown_until: dict[str, float] = {}
        self.endpoint_cooldown_trips: dict[str, int] = {}
        self.transient_retry_max = 1
        # Retention sweep cadence (Phase 2.3) — monotonic ts of the last DB trim.
        self.last_cleanup_at = 0.0
        # DRR-balance persistence cadence (Phase 3.4).
        self.last_budget_save_at = 0.0
        # queue.db maintenance cadences (persistence cleanup): WAL truncate +
        # incremental freelist return.
        self.last_wal_checkpoint_at = 0.0
        self.last_incr_vacuum_at = 0.0
        # Prefix-cache stats cadence (seeded to 0 → compute on first poll). Set
        # by Lifecycle at startup, advanced by Health at runtime.
        self.last_cache_stats_at = 0.0
        # Phase 2a — per-endpoint ACTUAL prefix-cache hit rate, refreshed once per
        # cache-stats cycle by Health from the GLOBAL vLLM /metrics prefix-cache
        # counters (the per-request cached_tokens field is NULL on our vLLM builds,
        # so this is the real per-endpoint number the /v1/status.cache_hit_rate +
        # attribution by_endpoint overlay read). Only vLLM endpoints appear;
        # llama.cpp exposes no counter → absent = n/a.
        # {endpoint: {"hit_rate": float, "queries": int, "source": str}}.
        self.endpoint_cache_hit_rate: dict[str, dict] = {}
        # Load-shed threshold (Phase 2.4): per-(endpoint, band) queue depth at
        # which NON-interactive submits are shed with 429 + Retry-After.
        self.shed_depth = 50
        # Alerting (Phase 2.5) — current triggered alerts (exposed on /v1/status)
        # + the set already logged, so a sustained condition logs once not every
        # poll tick.
        self.alerts: list[dict] = []
        self.alert_logged: set = set()
        # Admin-surface audit: source IPs seen per admin-ish route, exposed on
        # /v1/status so the ACL-tightening go/no-go can read the live set
        # instead of grepping logs. First hit per (route, ip) also logs INFO.
        self.admin_ips_seen: dict[str, set[str]] = {}
        # Unknown-endpoint submits (shadow counter): endpoint → {count,
        # callers}. Feeds the scheduled flip-check for unknown_endpoint_enforce
        # via /v1/status; non-empty means some caller submits a role the proxy
        # can't route (today that request rots to its deadline — the bug
        # enforce mode fixes with a fast 404).
        self.unknown_endpoint_submits: dict[str, dict] = {}
        # Context-overflow gate counter (shadow): endpoint → {count, callers,
        # max_est_in}. Feeds the context_gate_enforce flip check — compared
        # against ACTUAL backend overflow errors before enforcement flips.
        self.context_overflows: dict[str, dict] = {}
        # Phase 5B observability counters (exposed on /v1/status).
        self.slot_leak_reclaimed = 0       # streaming dispatches cancelled on
                                           # consumer-disconnect → slot freed
        self.drain_straggler_cancelled = 0  # in-flight tasks cancelled at the
                                            # shutdown drain deadline
        self.scheduler_task: asyncio.Task | None = None
        self.poller_task: asyncio.Task | None = None
        self.inflight_task: asyncio.Task | None = None
        self.started_at = time.monotonic()

    # ----- shared response/report helpers -----
    # Small read/write helpers that span clusters (Lifecycle + Health both call
    # resolve_error; Health poller + HTTP both call metrics_payload). Lifting
    # them here removes the only cross-collaborator method calls (contract §5.1
    # prefers the shared-state form). ProxyService keeps `_resolve_error` /
    # `_metrics_payload` as thin delegators to these.

    def resolve_error(self, req: "QueuedRequest", error: str) -> None:
        """Release a queued/pending request with a deferrable error, resolving
        whichever wait primitive the caller is blocked on (sync future or
        streaming queue)."""
        future = self.pending_futures.get(req.request_id)
        if future and not future.done():
            future.set_result({
                "request_id": req.request_id,
                "status": "error",
                "error": error,
            })
        stream_q = self.pending_streams.get(req.request_id)
        if stream_q:
            try:
                stream_q.put_nowait({"type": "error", "error": error})
            except asyncio.QueueFull:
                pass

    def metrics_payload(self, now: float) -> dict:
        """Rolling 5-min metrics — shared by GET /v1/metrics and the periodic
        SSE `metrics` frame."""
        return {
            "per_endpoint": {
                ep: {
                    "requests": self.metrics.count(endpoint=ep, now=now),
                    "timeouts": self.metrics.count(endpoint=ep, status="timeout", now=now),
                    "p50_wait_ms": self.metrics.percentile("queue_wait_ms", 50, endpoint=ep, now=now),
                    "p95_wait_ms": self.metrics.percentile("queue_wait_ms", 95, endpoint=ep, now=now),
                    "p50_backend_ms": self.metrics.percentile("backend_latency_ms", 50, endpoint=ep, now=now),
                    "p95_backend_ms": self.metrics.percentile("backend_latency_ms", 95, endpoint=ep, now=now),
                    "slot_seconds_consumed": round(self.metrics.slot_seconds_consumed(ep, now), 1),
                }
                for ep in self.config.endpoints
            },
            "per_agent": {
                aid: round(ss, 1)
                for aid, ss in self.metrics.per_agent_consumed(now).items()
            },
        }
