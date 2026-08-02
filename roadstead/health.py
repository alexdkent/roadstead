"""Health — circuit breaker + capacity poller + drain/pause for the LLMProxy.

Extracted from the ``service.py`` monolith (de-monolith Phase 1, Step 3). A
near-stateless behavior object: it receives the shared :class:`ProxyState` and
mutates ``state.*``. ``ProxyService`` keeps thin delegators to every method here
so its frozen private surface (the ~530 white-box tests) is unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

from . import cache_stats
from .backend import BackendError
from .config import (
    PriorityBand,
    cache_drift_alarm_enabled,
    cooldown_allowed_fails,
    cooldown_duration_s,
    cooldown_window_s,
    endpoint_cooldown_enabled,
    endpoint_cooldown_shadow,
    max_slots_reconcile_enabled,
    normalize_endpoint,
)
from .observability import (
    AlertCondition,
    check_alerts,
    structured_empty_alerts,
)
from originfleet.framework.prompt_security import record_security_event

if TYPE_CHECKING:
    from .config import EndpointConfig
    from .state import ProxyState

logger = logging.getLogger(__name__)


class Health:
    """Per-endpoint circuit breaker, capacity poller, and drain/pause logic."""

    def __init__(self, state: "ProxyState") -> None:
        self.state = state

    def scheduler_loop_alive(self) -> bool:
        return self.state.scheduler_task is not None and not self.state.scheduler_task.done()
    def poller_alive(self) -> bool:
        return self.state.poller_task is not None and not self.state.poller_task.done()
    def on_loop_task_exit(self, name: str, task: asyncio.Task) -> None:
        """A critical background loop ended. Normal only during shutdown
        (cancellation while draining); anything else is a silent total/partial
        outage — uvicorn keeps serving, run_agent.sh sees no exit — so scream."""
        if task.cancelled():
            return  # shutdown path
        exc = task.exception()
        logger.critical(
            "llmproxy %s loop EXITED unexpectedly%s — proxy degraded until restart",
            name, f": {exc!r}" if exc else " (clean return — should be impossible)",
        )
    def endpoint_healthy(self, endpoint: str) -> bool:
        ep = normalize_endpoint(endpoint)
        # Phase 5F: an operator drain overrides the auto-circuit — reads
        # unhealthy so all the defer/fast-fail machinery applies immediately.
        if ep in self.state.paused_endpoints:
            return False
        # Step 4b: rate-windowed cooldown (ENFORCE only). A flaky backend that
        # tripped the cooldown reads unhealthy until it expires, so the scheduler
        # defers its traffic (interactive was fast-failed at trip time); the next
        # dispatch tick after expiry re-admits it (auto-recovery).
        if endpoint_cooldown_enabled():
            until = self.state.endpoint_cooldown_until.get(ep, 0.0)
            if until and time.monotonic() < until:
                return False
        # On-demand endpoints are intentionally unloaded when idle; availability
        # is gated by OnDemandManager.ensure_loaded (loads on demand or raises),
        # not the always-on poller probe. Don't let a probe of an unloaded
        # backend trip the circuit and reject creative requests before they get
        # a chance to load it.
        if self.state.on_demand.manages(ep):
            return True
        h = self.state.endpoint_health.get(ep)
        return h["healthy"] if h else True
    def retry_after_s(self, endpoint: str) -> int:
        """Retry-After for a shed (Phase 2.4), derived from the endpoint's
        recent p95 backend latency (a drained slot frees on ~that cadence),
        clamped to [5, 60]s."""
        p95 = self.state.metrics.percentile(
            "backend_latency_ms", 95, endpoint=endpoint, now=time.monotonic())
        return max(5, min(60, int((p95 or 10000.0) / 1000.0)))
    async def update_endpoint_health(
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
        h = self.state.endpoint_health.setdefault(
            ep_name, {"healthy": True, "consecutive_failures": 0, "unhealthy_since": None})
        if probe_ok:
            if not h["healthy"]:
                logger.warning("endpoint %s RECOVERED (discovery) — resuming dispatch", ep_name)
                self.state.dispatch_event.set()  # drain its deferred queue
            h["healthy"] = True
            h["consecutive_failures"] = 0
            h["unhealthy_since"] = None
            return
        # Discovery failed. If already tripped, recover on LIVENESS alone so a
        # /health-up backend with flaky discovery doesn't latch open forever.
        if not h["healthy"]:
            if await self.state.backend.probe_health(ep_cfg):
                logger.warning(
                    "endpoint %s RECOVERED (/health up, discovery still flaky) — "
                    "resuming dispatch", ep_name)
                h["healthy"] = True
                h["consecutive_failures"] = 0
                h["unhealthy_since"] = None
                self.state.dispatch_event.set()
            return
        h["consecutive_failures"] += 1
        if h["healthy"] and h["consecutive_failures"] >= self.state.health_fail_threshold:
            alive = await self.state.backend.probe_health(ep_cfg)
            if not alive:
                h["healthy"] = False
                h["unhealthy_since"] = time.monotonic()
                logger.critical(
                    "endpoint %s UNHEALTHY — %d consecutive probe failures + /health "
                    "down; deferring its queue, fast-failing interactive",
                    ep_name, h["consecutive_failures"],
                )
                self.fast_fail_interactive(ep_name)
            else:
                # Backend answers /health → up but discovery is flaky; don't trip.
                h["consecutive_failures"] = 0
    def fast_fail_interactive(self, ep_name: str) -> None:
        """On the unhealthy transition, release queued INTERACTIVE/FOREGROUND
        requests for this endpoint with a deferrable error so they don't wait out
        their full deadline; BACKGROUND stays queued to defer until recovery."""
        queued = self.state.scheduler.queued_requests(
            ep_name, (PriorityBand.INTERACTIVE, PriorityBand.FOREGROUND))
        for req in queued:
            self.state.scheduler.cancel(req.request_id)
            self.state.queue_db.persist_expire(req.request_id)
            self.state.resolve_error(
                req, f"backend {ep_name} unavailable (circuit open)")
    def record_dispatch_failure(self, endpoint: str, exc: Exception) -> None:
        """Step 4b: feed the rate-windowed cooldown from the dispatch error paths.

        Counts ONLY a BACKEND-FAULT failure (5xx / timeout=504 / unavailable=503 —
        ``status_code >= 500``); a 4xx is the CALLER's fault and never cools the
        backend. After ``cooldown_allowed_fails`` within ``cooldown_window_s``,
        cool the endpoint (enforce → briefly unhealthy + fast-fail its queued
        interactive, exactly like the circuit) or log a would-cool (shadow). No-op
        — and zero cost — when both cooldown flags are off (byte-identical)."""
        shadow = endpoint_cooldown_shadow()
        enforce = endpoint_cooldown_enabled()
        if not (shadow or enforce):
            return
        if not (isinstance(exc, BackendError) and exc.status_code >= 500):
            return
        ep = normalize_endpoint(endpoint)
        now = time.monotonic()
        window = cooldown_window_s()
        times = self.state.endpoint_failure_times.setdefault(ep, [])
        times.append(now)
        cutoff = now - window
        while times and times[0] < cutoff:
            times.pop(0)
        if len(times) < cooldown_allowed_fails():
            return
        # Tripped — reset the window and record the trip (surfaced on /v1/status
        # even in shadow mode, so the operator can review before enforcing).
        times.clear()
        self.state.endpoint_cooldown_trips[ep] = (
            self.state.endpoint_cooldown_trips.get(ep, 0) + 1)
        dur = cooldown_duration_s()
        if enforce:
            self.state.endpoint_cooldown_until[ep] = now + dur
            logger.warning(
                "endpoint %s COOLED %.0fs — %d backend-fault failures within %.0fs "
                "(flaky backend); deferring its queue, fast-failing interactive",
                ep, dur, cooldown_allowed_fails(), window)
            self.fast_fail_interactive(ep)
        else:
            logger.warning(
                "endpoint %s WOULD cool %.0fs (SHADOW) — %d backend-fault failures "
                "within %.0fs", ep, dur, cooldown_allowed_fails(), window)
    def evaluate_alerts(self, now: float) -> None:
        """Evaluate proxy-internal alert conditions (Phase 2.5) and surface them
        to logs (log_scan/health-verifier) + /v1/status. Never restarts a backend — a dead
        backend is alerted, not killed (alert-don't-kill)."""
        snaps: dict[str, dict] = {}
        for ep in self.state.config.endpoints:
            s = self.state.scheduler.endpoint_snapshot(ep)
            # Phase 5F: the endpoint_paused ERROR alert is for an UNINTENDED
            # backend-down. Exclude operator drains so a planned maintenance
            # pause doesn't page as an outage — those surface as a separate
            # informational endpoint_drained alert below.
            s["paused"] = (not self.endpoint_healthy(ep)) and ep not in self.state.paused_endpoints
            snaps[ep] = s
        alerts = check_alerts(
            endpoint_snapshots=snaps,
            agent_budgets=self.state.budget_mgr.snapshot(),
            metrics=self.state.metrics,
            cost_model_samples={},  # cost-model-stale is INFO-only; skip for now
            queue_wal_size=self.state.queue_db.wal_size_bytes(),
            now=now,
        )
        # Phase 5F: operator drains — visible (so it's clear thinker is parked),
        # but WARNING not ERROR (intentional, not an outage).
        for ep in sorted(self.state.paused_endpoints):
            alerts.append(AlertCondition(
                name="endpoint_drained", severity="WARNING", triggered=True,
                detail=f"endpoint {ep} paused for maintenance (operator drain)",
            ))
        # Phase 5B.3: surface DB-writer-thread death (the single sanctioned bg
        # thread). If it dies, persistence degrades to loud sync fallback — page.
        if not self.state.queue_db.writer_alive():
            alerts.append(AlertCondition(
                name="writer_thread_dead", severity="CRITICAL", triggered=True,
                detail=f"db writer thread down (restarts={self.state.queue_db.writer_restarts()})",
            ))
        dropped = self.state.queue_db.write_q_dropped()
        if dropped:
            alerts.append(AlertCondition(
                name="write_queue_overflow", severity="WARNING", triggered=True,
                detail=f"{dropped} best-effort DB write(s) dropped (queue full)",
            ))
        # Step 4c: SHADOW max_slots-drift reconciler. vLLM's --max-num-seqs isn't
        # API-discoverable, so max_slots stays config-seeded and can silently
        # diverge from the backend's real launch cap (the reasoner 32-vs-20 class,
        # Step 2b). Warn — never auto-change admission — when a vLLM endpoint's
        # admitted max_slots differs from its documented launch value.
        if max_slots_reconcile_enabled():
            for ep_name, ep_cfg in self.state.config.endpoints.items():
                doc = getattr(ep_cfg, "documented_max_num_seqs", 0)
                if doc and ep_cfg.max_slots != doc:
                    alerts.append(AlertCondition(
                        name="max_slots_drift", severity="WARNING", triggered=True,
                        detail=(f"endpoint {ep_name} admits max_slots={ep_cfg.max_slots} "
                                f"but documented --max-num-seqs={doc} — reconcile "
                                f"models.yaml `slots` with the serve script"),
                    ))
        # Standing STRUCTURED-EMPTY rate alarm (2026-08-01, ledger
        # `tier3-json-object-empty-brace`). The 31-hour silent outage produced
        # NO other signal — `{}` is well-formed, so every existing emptiness /
        # truncation / degeneration check passed it. A per-endpoint rate is the
        # only thing that separates "one caller answered nothing" from "this
        # endpoint stopped answering", and putting it on the alerts channel is
        # what makes it reach /v1/status.alerts + the llmproxy_alerts_active
        # gauge + the health-verifier chip instead of dying as a log line nobody greps.
        # Window / threshold / sample-floor rationale: llmproxy/observability.py.
        alerts.extend(structured_empty_alerts(
            self.state.structured_empty_window, now))
        # Standing cache-drift conditions (audit 2026-07-02): mirror the current
        # drift set into the alerts channel so it reaches /v1/status.alerts +
        # the llmproxy_alerts_active gauge + the health-verifier chip — the CACHE_DRIFT
        # log line alone had no automated consumer. Bucketed detail (10-pt) so
        # a jittering LCP% doesn't re-log every tick via the dedup key.
        for d in (self.state.cache_drift_current or []):
            alerts.append(AlertCondition(
                name="cache_drift", severity="WARNING", triggered=True,
                detail=(f"{d.get('call_site')} on {d.get('endpoint')}: prefix "
                        f"LCP ~{round(float(d.get('from', 0)) / 10) * 10}%→"
                        f"~{round(float(d.get('to', 0)) / 10) * 10}% vs baseline"),
            ))

        self.state.alerts = [
            {"name": a.name, "severity": a.severity, "detail": a.detail}
            for a in alerts
        ]
        current = {(a.name, a.detail) for a in alerts}
        emit = {"CRITICAL": logger.critical, "ERROR": logger.error,
                "WARNING": logger.warning}
        for a in alerts:
            if (a.name, a.detail) not in self.state.alert_logged:
                emit.get(a.severity, logger.info)(
                    "ALERT [%s] %s: %s", a.severity, a.name, a.detail)
        self.state.alert_logged = current
    async def inflight_stream_loop(self) -> None:
        """Fast SSE `inflight` frame so /v1/stream subscribers see what's
        executing right now in (near) real time. The instant signals are the
        per-request `call.dispatched`/`call.completed` events; this periodic
        snapshot reconciles missed events and refreshes elapsed/queue/occupancy.
        Cheap + gated on client_count (no work when nobody's watching), so it
        never touches the dispatch hot path."""
        while True:
            try:
                if self.state.sse.client_count:
                    snap = self.state.scheduler.inflight_snapshot(time.monotonic())
                    snap["ts"] = time.time()
                    self.state.sse.publish("inflight", snap)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.debug("sse inflight publish failed: %s", exc)
            await asyncio.sleep(self.state.config.inflight_stream_interval_s)
    async def poll_endpoint_once(self, ep_name: str, ep_cfg: EndpointConfig) -> None:
        """One poller pass for one endpoint: capacity discovery (or a plain
        health probe for skip_discovery shims) + the circuit-breaker update."""
        # On-demand endpoints are intentionally unloaded when idle — probing a
        # stopped backend would 404/timeout every cycle (log spam) and falsely
        # mark it unhealthy. Skip until a lease is held (model resident); then
        # discover served-model/context normally.
        if getattr(ep_cfg, "on_demand", False) and not self.state.on_demand.is_loaded(ep_name):
            return
        probe_ok = False
        try:
            if ep_cfg.skip_discovery:
                # Non-OpenAI FastAPI shim (embed/rerank): no /props or
                # /v1/models to discover — a /health probe IS the signal.
                # Without this branch the discovery probes 404 every cycle
                # forever (log spam + a meaningless failure counter).
                probe_ok = await self.state.backend.probe_health(ep_cfg)
            elif ep_cfg.backend_engine == "vllm":
                # Capacity discovery is engine-specific. llama.cpp reports
                # slots + context via /props; vLLM has no /props or /slots,
                # so the per-request context ceiling comes from /v1/models
                # max_model_len (concurrency/max_slots stays config-driven).
                cap = await self.state.backend.probe_vllm_capacity(ep_cfg)
                if cap:
                    self.apply_discovered_vllm_capacity(ep_name, ep_cfg, cap)
                    probe_ok = True
            else:
                props = await self.state.backend.probe_props(ep_cfg)
                if props:
                    self.apply_discovered_props(ep_name, ep_cfg, props)
                    probe_ok = True
            if not ep_cfg.skip_discovery:
                # Discover the served model id (the name the backend
                # answers to). vLLM validates it, so the proxy sends
                # this — not the caller's role/alias — on dispatch.
                served = await self.state.backend.probe_models(ep_cfg)
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
            await self.update_endpoint_health(ep_name, ep_cfg, probe_ok)
        except Exception as exc:  # noqa: BLE001
            logger.debug("health update %s failed: %s", ep_name, exc)
    async def capacity_poller_loop(self) -> None:
        """Periodically probe backends for slot counts and context sizes.

        The iteration body is guarded: the poller also drives alerting, budget
        persistence, retention, and WAL maintenance — one escaped exception
        must not silently kill all of that until the next restart. The sleep
        sits OUTSIDE the guard so a failing iteration can't hot-spin, and
        CancelledError (shutdown) propagates from either."""
        while True:
            try:
                await self.poller_iteration()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — keep probing/alerting alive
                logger.critical(
                    "capacity poller iteration failed — continuing next cycle",
                    exc_info=True,
                )
            await asyncio.sleep(self.state.config.poller_interval_s)
    async def poller_iteration(self) -> None:
        """One full poller pass: probe every endpoint, then the periodic chores."""
        for ep_name, ep_cfg in self.state.config.endpoints.items():
            # Phase 5F: an operator-paused endpoint is intentionally down
            # (maintenance) — don't probe it (probes would fail + churn the
            # circuit/logs). /resume removes it from the set; the next poll
            # then re-probes, recovers, and re-discovers capacity (e.g. the
            # new max_model_len after a vLLM restart).
            if ep_name in self.state.paused_endpoints:
                continue
            await self.poll_endpoint_once(ep_name, ep_cfg)
        mono = time.monotonic()
        # Retention sweep (Phase 2.3): daily DB trim, off-loop via the writer.
        if mono - self.state.last_cleanup_at > 86400.0:
            self.state.last_cleanup_at = mono
            self.state.queue_db.cleanup_old_completions(
                self.state.config.completions_retention_s)
            self.state.queue_db.cleanup_old_payloads(
                self.state.config.payload_retention_s)
            # Evict one-off agent_ids (ad-hoc tools / smoke scripts /
            # probe-*) idle >1h so the DRR budget map can't grow unbounded;
            # a returning agent is recreated at the identical idle state.
            pruned = self.state.budget_mgr.prune_idle(mono, idle_ttl_s=3600.0)
            if pruned:
                # Also delete the persisted rows, or load_budgets resurrects
                # every ghost on the next boot (audit 2026-07-02).
                self.state.queue_db.delete_budgets(pruned)
                logger.info("budget: pruned %d idle agent(s): %s",
                            len(pruned), ", ".join(sorted(pruned)[:20]))
        # Periodic SSE `metrics` frame so /v1/stream subscribers get an
        # aggregate refresh between individual call.completed events
        # (no-op when nobody is subscribed).
        if self.state.sse.client_count:
            try:
                self.state.sse.publish("metrics", self.state.metrics_payload(mono))
            except Exception as exc:  # noqa: BLE001
                logger.debug("sse metrics publish failed: %s", exc)
        # WAL TRUNCATE-checkpoint so the -wal sidecar can't camp at a burst
        # high-water mark (persistence cleanup).
        if (mono - self.state.last_wal_checkpoint_at
                > self.state.config.wal_checkpoint_interval_s):
            self.state.last_wal_checkpoint_at = mono
            self.state.queue_db.checkpoint_truncate()
        # Return freed pages to the OS gradually (cheap; no full VACUUM lock).
        if (mono - self.state.last_incr_vacuum_at
                > self.state.config.incremental_vacuum_interval_s):
            self.state.last_incr_vacuum_at = mono
            self.state.queue_db.incremental_vacuum(
                self.state.config.incremental_vacuum_pages)
        # Periodic DRR-balance persistence (Phase 3.4) so a SIGKILL loses at
        # most ~60s of fairness state.
        if mono - self.state.last_budget_save_at > 60.0:
            self.state.last_budget_save_at = mono
            self.state.queue_db.save_budgets(self.state.budget_mgr.snapshot())
        # Prefix-cache observability (continuous): scrape each chat backend's
        # /metrics + run the cache-ability screen, persist one snapshot/endpoint.
        # Piggybacks the poller cadence but gated to its own interval.
        if mono - self.state.last_cache_stats_at > self.state.config.cache_stats_interval_s:
            self.state.last_cache_stats_at = mono
            try:
                await self.compute_cache_stats()
            except Exception as exc:  # noqa: BLE001 — best-effort observability
                logger.debug("cache-stats iteration failed: %s", exc)
        # Alerting (Phase 2.5): evaluate conditions → logs + /v1/status so
        # health-verifier/log_scan see proxy-internal health. Never restarts a backend.
        try:
            self.evaluate_alerts(mono)
        except Exception as exc:  # noqa: BLE001
            logger.debug("alert evaluation failed: %s", exc)
        # Age out stale timeout-model samples (cheap; piggybacks the
        # poller cadence instead of a dedicated task).
        self.state.timeout_model.prune(mono)
    async def compute_cache_stats(self) -> None:
        """One prefix-cache snapshot pass over the chat endpoints. For each:
        scrape vLLM /metrics prefix-cache counters (None for llama.cpp), run the
        cache-ability screen over recent stored prompts (off-loop), persist a row.
        Cheap + bounded; runs at ``cache_stats_interval_s``."""
        chat_classes = set(cache_stats.chat_endpoint_labels())
        snap_at = time.time()
        rates: dict[str, dict] = {}
        for ep_name, ep_cfg in self.state.config.endpoints.items():
            if ep_name not in chat_classes or ep_name in self.state.paused_endpoints:
                continue
            cum_hits = cum_queries = None
            if ep_cfg.backend_engine == "vllm":
                try:
                    pc = await self.state.backend.probe_prefix_cache(ep_cfg)
                    if pc:
                        cum_hits, cum_queries = pc["hits"], pc["queries"]
                except Exception:  # noqa: BLE001
                    pass
            # Phase 2a — ACTUAL per-endpoint prefix-cache hit rate. vLLM per-request
            # usage.prompt_tokens_details.cached_tokens is NULL on our builds (a
            # vLLM build gap, not a flag — verified 2026-07-01), so the per-caller
            # cached_tokens rollup reads n/a. The GLOBAL vLLM /metrics prefix-cache
            # counters DO work, so surface the real per-ENDPOINT lifetime rate from
            # those here (llama.cpp exposes no counter → stays absent = n/a). This is
            # what /v1/status.cache_hit_rate + the attribution by_endpoint overlay
            # read. Per-CALLER actual stays backend-gated (Tier-1 LCP screen predicts
            # it); this is the real ENDPOINT number the operator asked to surface.
            if cum_queries and cum_queries > 0 and cum_hits is not None:
                rates[ep_name] = {
                    "hit_rate": round(cum_hits / cum_queries, 4),
                    "queries": int(cum_queries),
                    "source": "backend_prefix_cache_metrics",
                }
            try:
                rows = await asyncio.to_thread(
                    self.state.queue_db.cache_screen_payloads, ep_name, 168.0, 2000)
                screen = await asyncio.to_thread(cache_stats.screen, rows)
            except Exception:  # noqa: BLE001
                screen = []
            self.state.queue_db.persist_cache_snapshot(
                snapshot_at=snap_at, endpoint=ep_name,
                cum_hits=cum_hits, cum_queries=cum_queries,
                screen_json=json.dumps(screen),
            )
        self.state.queue_db.prune_cache_stats()
        self.state.endpoint_cache_hit_rate = rates
        # Tier-2 step 3 — prefix-cache DRIFT alarm (observability only; never
        # touches routing/output). Best-effort: a failure here must not break the
        # cache-stats pass.
        if cache_drift_alarm_enabled():
            try:
                await self._evaluate_cache_drift(snap_at)
            except Exception as exc:  # noqa: BLE001
                logger.debug("cache-drift alarm failed: %s", exc)

    async def _evaluate_cache_drift(self, now: float) -> None:
        """Read the recent cache snapshots, detect call_sites whose front-loaded
        prefix collapsed vs baseline, and raise a ``CACHE_DRIFT_ALERT`` log marker
        + a store-less ``llmproxy_cache_drift`` security event for each NEW drift
        (dedup'd via ``state.cache_drift_alerted``). The heavy snapshot read +
        median math run off the event loop."""
        snaps = await asyncio.to_thread(
            self.state.queue_db.cache_stats_snapshots, cache_stats.CACHE_DRIFT_WINDOW_S)
        drift = cache_stats.detect_drift(snaps)
        # Standing-alert feed: evaluate_alerts converts the CURRENT drift set
        # into AlertConditions each tick (reachability, audit 2026-07-02).
        self.state.cache_drift_current = drift
        to_fire, self.state.cache_drift_alerted = cache_stats.drift_alarms_to_fire(
            drift, self.state.cache_drift_alerted, now)
        # Persist the dedup map (audit 2026-07-02): without this every proxy
        # boot re-alerted the full standing-drift set 1s after startup.
        # Wall-clock timestamps, so they survive across processes correctly.
        try:
            self.state.queue_db.kv_set(
                "cache_drift_alerted",
                [[cs_, ep_, ts_] for (cs_, ep_), ts_ in
                 self.state.cache_drift_alerted.items()])
        except Exception:  # noqa: BLE001 — dedup persistence is best-effort
            logger.debug("cache-drift dedup persist failed", exc_info=True)
        for d in to_fire:
            logger.warning(
                "CACHE_DRIFT_ALERT call_site=%s endpoint=%s lcp_pct=%.1f->%.1f — "
                "prefix cacheability collapsed vs baseline; a prompt edit likely "
                "broke the front-loaded block (see cache_audit / the Inference "
                "cacheability card)",
                d["call_site"], d["endpoint"], d["from"], d["to"])
            record_security_event(
                None, logger,
                event_type="llmproxy_cache_drift", severity="warning",
                source=d["endpoint"], action=d["call_site"],
                reason=f"prefix LCP% {d['from']}->{d['to']}",
                metadata={"call_site": d["call_site"], "endpoint": d["endpoint"],
                          "lcp_from": d["from"], "lcp_to": d["to"]})
    def apply_discovered_props(
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
            self.state.cost_model.update_max_slots(ep_name, n_parallel)
            self.state.budget_mgr.set_total_capacity(self.state.config.total_fleet_slots)
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

        # Context size. default_generation_settings.n_ctx is ALREADY per-slot
        # in current llama.cpp builds (confirmed live: --ctx-size 131072
        # --parallel 4 reports n_ctx=32768 there, not 131072) — dividing it by
        # n_parallel again silently quartered every multi-slot llama.cpp
        # endpoint's discovered context_per_slot (e.g. 32768 -> 8192 for a
        # 4-slot unit), which feeds the context-gate admission check and could
        # wrongly reject requests that actually fit. Only the top-level
        # `props["n_ctx"]` fallback (older/different builds, unconfirmed
        # whether it's ever populated as an aggregate) still gets divided.
        gen_n_ctx = gen_settings.get("n_ctx")
        if gen_n_ctx:
            if gen_n_ctx != ep_cfg.context_per_slot:
                ep_cfg.context_per_slot = gen_n_ctx
        else:
            top_n_ctx = props.get("n_ctx")
            if top_n_ctx and n_parallel:
                ctx_per_slot = top_n_ctx // n_parallel
                if ctx_per_slot != ep_cfg.context_per_slot:
                    ep_cfg.context_per_slot = ctx_per_slot
    def apply_discovered_vllm_capacity(
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
