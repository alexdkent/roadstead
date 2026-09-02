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
import math
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from . import hooks
from .acl import IPIdentityMap
from .agent_budget import BudgetManager
from .backend import BackendClientPool
from .coalesce import DeterministicCache
from .config import ProxyConfig, normalize_endpoint
from .cost_model import CostModel
from .flags import RuntimeFlags
from .identity import IdentityResolver, KeyRegistry
from .management import AdminOverlay
from .observability import RequestLogger, RollingMetrics
from .on_demand import OnDemandManager
from .queue import PersistentQueue
from .scheduler import Scheduler
from .spend import (
    PriceBook,
    SpendLedger,
    SpendStanding,
    day_bucket,
    declared_price,
    standing,
)
from .rate import RateLedger, RateStanding
from .rate import standing as rate_standing
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
        # Per-class timeout floors come from models.yaml (`timeout_floor_s`), the
        # declared single source of truth, layered OVER the hardcoded FLOOR_S
        # fallback. Without this the yaml field was inert and a floor bump for a
        # model swap (e.g. classify 45→180 / composer 180→360, 2026-07 nexus
        # loadout) silently did nothing — enforcement kept using the stale
        # hardcoded values. Fallback covers any class absent from the yaml.
        from .model_catalog import (
            build_class_ceilings, build_class_floors,
            build_class_stream_hard_caps,
        )
        from .timeout_model import FLOOR_S
        floors = {**FLOOR_S, **build_class_floors()}
        self.timeout_model = TimeoutModel(
            margin=config.timeout_advice_margin,
            window_s=config.timeout_advice_window_s,
            min_samples=config.timeout_advice_min_samples,
            floors=floors,
        )
        # Per-role timeout-ceiling overrides (models.yaml `timeout_ceiling_s`) —
        # let an inherently long-running class keep a generous ceiling even on an
        # interactive tier, overriding the interactive/background tier band.
        self.timeout_ceilings = build_class_ceilings()
        # Per-class ABSOLUTE streaming hard caps (models.yaml
        # `stream_hard_cap_s`) — the upper bound on extending a PROXY-CHOSEN
        # streaming deadline that is still making token progress. Classes absent
        # here fall back to the band default in constants. Same yaml→catalog→
        # state path as the floors/ceilings above, so the yaml stays the single
        # source of truth instead of the cap being a bare literal in the
        # streaming path.
        self.stream_hard_caps = build_class_stream_hard_caps()
        # Progress-governed streaming: how often a proxy-chosen deadline was
        # actually outlived by a still-progressing stream, and by how much in
        # total. Without these the extension is an INVISIBLE capacity sink — an
        # operator must be able to see how often it fires and for how long.
        # Plain ints on the single event loop (no locks, no threads).
        self.stream_deadline_extended = 0      # streams that ran past their soft budget
        self.stream_extension_s_total = 0.0    # summed seconds granted beyond it
        self.stream_hard_cap_aborts = 0        # ...and how many hit the absolute cap
        # C6: gap deadlines pushed out because the backend's /metrics counters
        # proved it was still working. A stall that is NOT in this number was a
        # genuinely frozen backend; one that is tells you the watchdog saved a
        # turn it used to kill. Without it the fix is unobservable.
        self.stream_progress_extensions = 0
        self.budget_mgr = BudgetManager(starvation_timeout_s=config.starvation_timeout_s)
        # Money (roadmap Workstream D). The price book is seeded from the
        # catalog and then kept current by capacity discovery on any provider
        # whose descriptor says `publishes_token_costs`; the ledger is the LIVE
        # per-caller account a threshold decision reads without going to the DB.
        # 🚨 Both are ordinary single-loop in-memory state — the concurrency
        # invariant in CLAUDE.md covers them exactly as it covers `budget_mgr`.
        self.prices = PriceBook()
        for ep_name, ep_cfg in config.endpoints.items():
            declared = declared_price(ep_name, ep_cfg)
            if declared is not None:
                self.prices.declare(ep_name, declared)
        self.spend = SpendLedger(self.prices)
        # agent_id -> the epoch day its over-cap demotion was last reported, so
        # a caller that stays over its cap all day reports once rather than on
        # every request. A report per call would bury the event it exists to
        # surface.
        self.spend_demotion_noted: dict[str, int] = {}
        # The rate window, and the same once-a-day notice bookkeeping. Separate
        # from the spend one because an operator needs to know WHICH threshold
        # moved a caller — one merged notice would make a rate problem look like
        # a billing one.
        self.rate = RateLedger()
        self.rate_demotion_noted: dict[str, int] = {}
        # source endpoint class -> how many requests it has spilled to remote
        # capacity. The counterpart to `degraded_rerouted`, and separate from it
        # for the same reason the two request fields are separate: a reroute is
        # a worse answer, a spill is a bill.
        self.spilled_from: dict[str, int] = {}
        self.scheduler = Scheduler(config, self.cost_model, self.budget_mgr)
        self.backend = BackendClientPool()
        self.queue_db = PersistentQueue(config.queue_db_path or None)
        # On-demand endpoints: the model is NOT always-resident — it is loaded
        # lazily under a host dispatcher's lease and idle-unloaded when quiet.
        self.on_demand = OnDemandManager(config.endpoints)
        # § 9 tier3 failover. Assigned by ProxyService right after Health is
        # built (Failover consumes endpoint_healthy and must not duplicate it),
        # which is why this is a late-bound attribute rather than constructed
        # here like the collaborators above. Never None in a running proxy.
        self.failover: Any = None

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
        # Caller identity: an API key first, ``self.acl`` as the second factor.
        # The resolver is what the request path asks — the ACL is kept as its
        # own field because it is the thing an operator configures and a test
        # registers into, not because anything reads it directly any more.
        self.identity = IdentityResolver(self.acl, KeyRegistry.from_env())
        # The management plane's overlay (roadmap E). Applied AFTER the registry
        # and the agent configs are loaded, because it is a layer OVER them:
        # runtime enrolments, revocations and quota overrides that must win over
        # the files without rewriting them. `apply` is the only thing that
        # touches either object outside a request.
        self.admin_overlay = AdminOverlay(config.admin_store_path or None)
        self.admin_overlay.apply(self.identity.keys, config)
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
        self.thinking_noop = 0        # thinking applied but response carried NO
                                      # reasoning (backend ignored enable_thinking)
        # WS-4 shadow egress detector: per-call_site silent grammar-drop tally
        # over ALL grammar-bearing responses (read-only; NEVER mutates a
        # response). {call_site: {"checked": int, "dropped": int}}. Populated by
        # _shadow_egress_detect when ROADSTEAD_PROXY_SHADOW_EGRESS is on (default).
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
        # Phase 3 schema-repair backstop tallies (structured/tool responses that
        # parse-but-fail-schema / need json-repair). detected = would-fire;
        # repaired = fixed in-memory by json-repair (no backend); retry_recovered
        # = fixed by the one bounded error-fed-back re-dispatch; unrecoverable =
        # failed loud (deferrable, never cached); stream_invalid = detected on a
        # streaming reassembly (detect-only). by_call_site mirrors the degeneration
        # tally shape. retry_inflight bounds the OUT-of-slot re-dispatch concurrency
        # (same rationale as degen_redispatch_inflight above).
        self.schema_detected = 0
        self.schema_repaired = 0
        self.schema_retry_recovered = 0
        self.schema_unrecoverable = 0
        self.schema_invalid_stream = 0
        self.schema_by_call_site: dict[str, dict] = {}
        self.schema_retry_inflight = 0
        # Empty-completion (position-0-EOS) rescue tallies — see
        # _EMPTY_RESCUE_MIN_TOKENS. attempts = retries dispatched with
        # min_tokens; recovered = those that produced a real response.
        self.empty_rescue_attempts = 0
        self.empty_rescue_recovered = 0
        # Per-endpoint empty-completion events (audit 2026-07-12, C-3): a 2xx
        # backend response with no content/tool_calls (position-0-EOS), counted
        # each time the fail-loud gate in backend.call() trips — the
        # reliability signature for "this endpoint returned an empty
        # completion". Loop-thread-only writes (single-writer invariant), emitted
        # on /metrics as ``llmproxy_empty_completion_total{endpoint}``.
        self.empty_completion_by_endpoint: dict[str, int] = {}
        # Truncation / structured-validity guard tallies (operator mandate
        # 2026-07-11 — truncation must never pass silently; structured responses
        # must be valid JSON or an explicit error). Keyed "endpoint|agent_id" so
        # /v1/status shows WHO is hitting output caps on WHICH model. truncation
        # rows split structured vs freetext (freetext truncation still serves —
        # log + count only). parse failures = structured content that failed
        # json.loads after every repair layer ran (sync 502 / stream error frame).
        self.truncation_total = 0
        self.truncation_by_model_caller: dict[str, dict] = {}
        self.structured_parse_failure_total = 0
        self.structured_parse_failures_by_model_caller: dict[str, int] = {}
        # Empty-structured-response detection (2026-08-01, ledger
        # `tier3-json-object-empty-brace`). A structured request whose content
        # is a WELL-FORMED JSON object carrying no answer ("{}") defeats every
        # emptiness / truncation / degeneration check the proxy has — that is
        # how one endpoint returned nothing for 31 hours with no error, no
        # exception and finish_reason=stop. Detection is telemetry only (the
        # response is never altered); the operator surface is the per-endpoint
        # RATE alarm on /v1/status.alerts, computed off `structured_empty_window`
        # (endpoint -> deque[(monotonic_ts, was_empty, call_site)], pruned to
        # observability.STRUCTURED_EMPTY_WINDOW_S). Loop-thread-only writes,
        # so no lock (single event loop invariant).
        self.structured_empty_total = 0
        self.structured_empty_by_call_site: dict[str, int] = {}
        self.structured_empty_window: dict[str, deque] = {}
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
        # § 9 tier3 failover — endpoints currently serving their traffic from a
        # declared fallback because they are unhealthy.
        #
        # 🚨 A SET, mirroring paused_endpoints above, and deliberately NOT a
        # per-endpoint bool: the `paused` bool on /v1/status is already
        # known-unreliable (observed False for an endpoint that WAS in
        # paused_endpoints), and a second bool with the same bug would only give
        # the operator a second thing to disbelieve.
        #
        # NOT persisted across a restart (unlike paused_endpoints, which is an
        # operator INTENTION and must survive one). Degraded mode is a derived
        # observation: a fresh process re-derives it from health within one
        # poller tick, and re-seeding it would assert an outage that may be over.
        self.degraded_endpoints: set[str] = set()
        self.degraded_since: dict[str, float] = {}
        self.degraded_rerouted: dict[str, int] = {}
        # source endpoint → refusal code → count. The refusals matter more than
        # the reroutes: they are what tells the operator that the opt-in set is
        # too small, or that the failover target is too small for the traffic.
        self.degraded_refused: dict[str, dict[str, int]] = {}
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
        # Per endpoint: timeouts EXCLUDED from the cooldown window because the
        # caller's own applied deadline was a best-effort sub-floor give-up
        # (health.record_dispatch_failure(best_effort=True)). Counted so the
        # exclusion is visible on /v1/status — an invisible guard cannot be told
        # apart from one that never fires.
        self.cooldown_best_effort_skips: dict[str, int] = {}
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
        # Tier-2 step 3 — prefix-cache DRIFT alarm dedup: {(call_site, endpoint):
        # last_fired_wall} so a standing drift re-alerts at most every
        # CACHE_DRIFT_REALERT_S instead of every cache-stats cycle.
        self.cache_drift_alerted: dict = {}
        # Currently-drifting call_sites from the last drift evaluation (audit
        # 2026-07-02): evaluate_alerts turns these into standing AlertConditions
        # so drift reaches /v1/status.alerts + the llmproxy_alerts_active gauge
        # (the one-shot CACHE_DRIFT_ALERT log line + store-less security event
        # were unreachable by any automated consumer).
        self.cache_drift_current: list[dict] = []
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
        # Vision-capability violations (shadow counter): endpoint → {count,
        # callers}. Non-empty means somebody is sending IMAGE content to an
        # endpoint whose models.yaml stanza says `capabilities.vision: false`
        # — i.e. a backend with no mmproj, which answers HTTP 500 "image input
        # is not supported". Added 2026-08-20 after exactly that shipped
        # silently for a day: a shared role constant was re-pointed to a
        # text-only box and carried discord's image describe with it, and the
        # 500 was swallowed into "" by the caller's own error handling, which
        # every vision caller reads as "couldn't read the image". The catalog
        # DECLARED vision:false throughout and nothing read the field.
        # Ledger: `a-role-rename-carried-vision-to-a-text-only-box`.
        self.vision_capability_violations: dict[str, dict] = {}
        # Context-overflow gate counter (shadow): endpoint → {count, callers,
        # max_est_in}. Feeds the context_gate_enforce flip check — compared
        # against ACTUAL backend overflow errors before enforcement flips.
        self.context_overflows: dict[str, dict] = {}
        # Phase 5a — smart-default-timeout shadow tally: endpoint → {count,
        # flat_s, smart_s_min, smart_s_max, smart_s_sum}. Records, for callers
        # that OMIT timeout_s, what the data-driven default WOULD be (smart_s)
        # vs the flat _DEFAULT_TIMEOUT_S — populated whether the
        # smart_default_timeout flag is on or off, so the flip decision has a
        # direct artifact on /v1/status. mean = smart_s_sum / count.
        self.smart_default_shadow: dict[str, dict] = {}
        # Phase 5B observability counters (exposed on /v1/status).
        self.slot_leak_reclaimed = 0       # streaming dispatches cancelled on
                                           # consumer-disconnect → slot freed
        self.drain_straggler_cancelled = 0  # in-flight tasks cancelled at the
                                            # shutdown drain deadline
        self.scheduler_task: asyncio.Task | None = None
        self.poller_task: asyncio.Task | None = None
        self.inflight_task: asyncio.Task | None = None
        self.started_at = time.monotonic()
        # Wall-clock twin of started_at. `started_at` is monotonic and therefore
        # NOT a timestamp — /v1/models needs a real epoch for the OpenAI
        # `created` field. Stamped once per boot so the listing is stable
        # request-to-request (a per-request time.time() would make every poll
        # look like a different model set to a caching client).
        self.boot_time_epoch = int(time.time())

    # ----- shared response/report helpers -----
    # Small read/write helpers that span clusters (Lifecycle + Health both call
    # resolve_error; Health poller + HTTP both call metrics_payload). Lifting
    # them here removes the only cross-collaborator method calls (contract §5.1
    # prefers the shared-state form). ProxyService keeps `_resolve_error` /
    # `_metrics_payload` as thin delegators to these.

    # ----- spend thresholds (Workstream D) -----
    # Here rather than on ProxyService for the same reason: Lifecycle applies
    # the band demotion on the submit path and the Scheduler asks the spill
    # question at dispatch, and both hold the state, not the service.

    def spend_standing(self, agent_id: str) -> SpendStanding:
        """Where ``agent_id`` stands against its daily cap, right now.

        The one place the ledger and the agent config are joined, so the
        dispatch gate and the band demotion cannot end up reading different
        answers to the same question about the same request.
        """
        return standing(
            self.spend, agent_id,
            self.config.agent_config(agent_id).daily_spend_usd,
        )

    def rate_standing(self, agent_id: str) -> "RateStanding":
        """How fast ``agent_id`` is going, against its threshold, right now."""
        return rate_standing(
            self.rate, agent_id,
            self.config.agent_config(agent_id).requests_per_minute,
            now=time.time(),
        )

    def record_request(self, agent_id: str) -> None:
        """Note one request against ``agent_id``'s rate window. On the loop."""
        self.rate.record(agent_id, time.time())

    def prune_rate_state(self) -> int:
        """Drop rate state for callers with nothing left in the window.

        🚨 Both dicts, and both are keyed by ``agent_id`` — which is a
        CALLER-SUPPLIED string on the address path (``docs/api.md`` §1.5: an
        address only fills in an ``agent_id`` the body omitted, so a body may
        declare its own). Without this they grow one entry per distinct name
        ever seen, which is unbounded input under a caller's control. The
        threshold this feeds cannot reject, so the rate ledger is *deliberately*
        not a defence against a runaway caller — that is exactly why its own
        memory must not be one either.

        🚨 On ``time.time()``, NOT the poller's monotonic clock. ``record``
        stamps wall time, so pruning with a monotonic ``now`` would compare a
        process uptime against epoch timestamps: the cutoff lands decades in the
        past, nothing is ever stale, and this returns 0 forever while looking
        wired. Pinned by test_rate_prune.py.

        Forgetting a caller also forgets that we noted its demotion today. That
        is correct rather than convenient: it had no traffic in the window, so
        if it comes back and crosses again the operator should be told again.
        """
        now = time.time()
        dropped = self.rate.prune(now)
        if dropped:
            live = self.rate.windows
            for agent_id in [a for a in self.rate_demotion_noted if a not in live]:
                del self.rate_demotion_noted[agent_id]
        return dropped

    def spend_may_spill(self, agent_id: str) -> bool:
        """Whether ``agent_id`` may currently be served by PAID remote capacity.

        This is the only thing crossing a threshold takes away at dispatch. It
        does not gate local dispatch and it is not reachable from any path that
        could refuse a request: ``Scheduler._admit`` calls it strictly after
        local capacity has already said no.
        """
        # 🚨 BOTH thresholds, and an AND: either one removes paid spill. A
        # caller going far too fast is the last one whose overflow should become
        # an invoice on somebody else's hardware, and a caller over its money
        # cap is the obvious one. Unlike the band demotion below, these do not
        # need a non-stacking rule — "may not spill" has no second step.
        return (self.spend_standing(agent_id).may_spill
                and self.rate_standing(agent_id).may_spill)

    def effective_priority(
        self, agent_id: str, declared: LLMPriority,
    ) -> LLMPriority:
        """The band ``agent_id`` actually gets. **Read-only — no notice.**

        🚨 THE non-stacking rule, and it is non-stacking BY CONSTRUCTION: at
        most one standing is ever consulted, so a caller over two thresholds
        drops one band and a third threshold added later cannot quietly make it
        three. The alternative — compose the two answers — is the shape where
        adding a threshold silently doubles the penalty of the ones already
        there, which is the unbounded-penalty argument in
        ``spend.SpendStanding.effective_priority`` with more force.

        Split from :meth:`spend_demote` because the management plane reports
        this number on a read, and a read that fired a degradation notice would
        make opening a dashboard look like a caller misbehaving.
        """
        spend_st = self.spend_standing(agent_id)
        if spend_st.over:
            return spend_st.effective_priority(declared)
        rate_st = self.rate_standing(agent_id)
        if rate_st.over:
            return rate_st.effective_priority(declared)
        return declared

    def spend_demote(self, agent_id: str, declared: LLMPriority) -> LLMPriority:
        """The band ``agent_id`` actually gets, given what it declared.

        🚨 **One step down, however many thresholds it has crossed.** THE place
        that rule lives, because this is the one place every standing is known.
        Two independent one-band penalties would mean that adding a second
        threshold silently doubled the first one's — and the unbounded-penalty
        argument in ``spend.SpendStanding.effective_priority`` applies with more
        force to two of them than to one. A caller over its money cap AND going
        far too fast is behind everyone behaving themselves, which is all any of
        this is for; putting it two bands down would be a rejection taking its
        time.

        Called on the submit path before the request is queued, because the band
        is fixed at enqueue and re-banding a queued request would mean moving it
        between per-agent deques mid-flight for no gain.

        Each crossing is reported through ``hooks.degradation`` the first time it
        bites in a day, SEPARATELY — an operator needs to know which threshold
        moved a caller, and one merged notice would make a rate problem look like
        a billing one. A caller that suddenly waits longer with nothing logged is
        indistinguishable from a slow backend, and that ambiguity is the whole
        cost of choosing to degrade rather than to reject — worth paying, but
        only if somebody can see it happening.

        The name is Workstream D's and is kept: it is called from the submit
        path and from nothing else, and renaming a seam to describe its second
        reason is churn that reaches ``git blame`` before it reaches anybody.
        """
        spend_st = self.spend_standing(agent_id)
        rate_st = self.rate_standing(agent_id)
        effective = self.effective_priority(agent_id, declared)
        if effective is declared:
            return declared
        today = day_bucket(time.time())
        if spend_st.over and self.spend_demotion_noted.get(agent_id) != today:
            self.spend_demotion_noted[agent_id] = today
            hooks.degradation(
                component="spend",
                reason="caller is over its daily spend cap",
                impact=("its requests drop one priority band and it may not use "
                        "paid remote spill; local capacity is unaffected"),
                agent_id=agent_id,
                spent_today_usd=round(spend_st.spent_today_usd, 6),
                cap_usd=spend_st.cap_usd,
                priority_declared=declared.name,
                priority_effective=effective.name,
            )
        if rate_st.over and self.rate_demotion_noted.get(agent_id) != today:
            self.rate_demotion_noted[agent_id] = today
            hooks.degradation(
                component="rate",
                reason="caller is over its request-rate threshold",
                impact=("its requests drop one priority band and it may not use "
                        "paid remote spill; local capacity is unaffected. This "
                        "is not a rate LIMIT: nothing is refused"),
                agent_id=agent_id,
                observed_per_min=round(rate_st.observed_per_min, 2),
                limit_per_min=rate_st.limit_per_min,
                priority_declared=declared.name,
                priority_effective=effective.name,
            )
        return effective

    def effective_timeout_advice(
        self, endpoint: str, priority: int, est_in: int, est_out: int,
    ) -> dict:
        """The empirical ``advise()`` result UPLIFTED for live load + prompt
        size and bounded by the per-caller-class ceiling.

        This is what callers (the ``/v1/timeout-advice`` endpoint) and the
        server default resolver consume, so the deadline widens when the
        endpoint is busy or the prompt is large — instead of the flat empirical
        p99×margin failing a merely-slow call. The RECORDING/counterfactual
        sites keep raw ``advise()`` (load-neutral base for the shadow compare).

        Fully guarded: any fault falls back to the raw advice so timeout
        resolution can never 500 a request. Adds ``surge`` / ``size_stretch`` /
        ``ceiling_s`` to the dict for observability."""
        from .timeout_model import apply_load_and_ceiling, resolve_ceiling_s

        advice = self.timeout_model.advise(endpoint, priority, est_in, est_out)
        try:
            snap = self.scheduler.endpoint_snapshot(normalize_endpoint(endpoint))
            floor_s = self.timeout_model.floor_ms(endpoint) / 1000.0
            ceiling_s = resolve_ceiling_s(
                endpoint,
                interactive=int(priority) <= 2,  # P0/P1/P2 (see LLMPriority)
                role_ceilings=self.timeout_ceilings,
                floor_s=floor_s,
                interactive_s=self.config.timeout_ceiling_interactive_s,
                background_s=self.config.timeout_ceiling_background_s,
            )
            effective_ms, surge, stretch = apply_load_and_ceiling(
                advice["recommended_ms"],
                in_flight=snap.get("in_flight", 0),
                queued=snap.get("queued", 0),
                max_slots=snap.get("max_slots", 0),
                est_in=est_in,
                ceiling_ms=ceiling_s * 1000.0,
                k_load=self.config.timeout_surge_k,
                surge_max=self.config.timeout_surge_max,
                k_size=self.config.timeout_size_k,
                size_max=self.config.timeout_size_max,
            )
            advice = {
                **advice,
                "recommended_ms": round(effective_ms, 1),
                "recommended_timeout_s": math.ceil(effective_ms / 1000.0),
                "surge": round(surge, 3),
                "size_stretch": round(stretch, 3),
                "ceiling_s": round(ceiling_s, 1),
            }
        except Exception:  # noqa: BLE001 — advice uplift must never 500 a request
            pass
        return advice

    def resolve_error(self, req: "QueuedRequest", error: str, *,
                      backend_status: int | None = None) -> None:
        """Release a queued/pending request with an error, resolving whichever
        wait primitive the caller is blocked on (sync future or streaming queue).

        🚨 ``backend_status`` is the status the BACKEND returned, when the
        failure came from one. It is carried because the proxy already knows
        whether a failure is deterministic — `is_transient_backend_error` says
        so in its own docstring, "a real 4xx / other-5xx is deterministic" — and
        used to keep that to itself. The caller was told `backend_error`, which
        every client classifies as retryable, so the proxy gave up on a
        permanent failure and simultaneously advised retrying it. One fact, two
        readers: the retry decision here and the caller's, instead of the same
        judgement made twice and differently.
        """
        payload = {
            "request_id": req.request_id,
            "status": "error",
            "error": error,
        }
        if backend_status is not None:
            payload["backend_status"] = int(backend_status)
        future = self.pending_futures.get(req.request_id)
        if future and not future.done():
            future.set_result(dict(payload))
        stream_q = self.pending_streams.get(req.request_id)
        if stream_q:
            frame = {"type": "error", "error": error}
            if backend_status is not None:
                frame["backend_status"] = int(backend_status)
            try:
                stream_q.put_nowait(frame)
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
