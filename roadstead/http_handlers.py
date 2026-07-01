"""ProxyHttpHandlers — the Starlette HTTP surface for the LLMProxy.

The ~26 `handle_*` request handlers (OpenAI front door, admin, metrics, status,
SSE, fleet usage) + the admin-IP audit. Thin translators: parse the request,
read/mutate the shared :class:`ProxyState` or call the Lifecycle/Health
collaborators, serialize a response. ProxyService keeps identical-signature
`handle_*` delegators so `routes.make_routes(svc)` binds unchanged (contract §1).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from . import cache_stats
from .config import LLMPriority, normalize_endpoint
from .constants import _DEFAULT_TIMEOUT_S, _PAYLOAD_KIND
from .lifecycle import _openai_error
from .sse_hub import DROP_SENTINEL

if TYPE_CHECKING:
    from .health import Health
    from .lifecycle import Lifecycle
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


class ProxyHttpHandlers:
    """The Starlette HTTP surface over the shared ProxyState."""

    def __init__(self, state: "ProxyState", lifecycle: "Lifecycle",
                 health: "Health") -> None:
        self.state = state
        self.lifecycle = lifecycle
        self.health = health

    async def handle_openai_chat(self, body: dict, request: Request) -> Response:
        remote_ip = request.client.host if request.client else "unknown"
        identity = self.state.acl.identify(remote_ip)
        if not identity:
            # Phase 5D: OpenAI-shaped 403 (was a bare {"error": "<str>"} that a
            # strict OpenAI client crashes on doing resp.error.message).
            return _openai_error(
                f"access denied for {remote_ip}", "access_denied", 403,
                code="access_denied")
        agent_id, default_priority = identity
        model = body.get("model", "qwen-analyst")
        # Phase 5D: validate the model maps to a known endpoint BEFORE enqueue.
        # Otherwise an unknown model burns a scheduler slot + DRR charge and
        # fails late with a confusing 502; OpenAI clients expect 404/model_not_found.
        if normalize_endpoint(str(model)) not in self.state.config.endpoints:
            return _openai_error(
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
        return await self.lifecycle.handle_submit(submit_body, request, openai=True)
    async def handle_openai_embeddings(self, body: dict, request: Request) -> Response:
        remote_ip = request.client.host if request.client else "unknown"
        identity = self.state.acl.identify(remote_ip)
        if not identity:
            return _openai_error(
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
        return await self.lifecycle.handle_submit(submit_body, request, openai=True)
    async def handle_models(self, request: Request) -> Response:
        models = []
        for ep_name, ep_cfg in self.state.config.endpoints.items():
            models.append({
                "id": ep_cfg.role,
                "object": "model",
                "owned_by": "collective",
                "endpoint_class": ep_name,
                "max_slots": ep_cfg.max_slots,
                "context_per_slot": ep_cfg.context_per_slot,
            })
        return JSONResponse({"object": "list", "data": models})
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
            for ep_name in self.state.config.endpoints:
                lbl = {"endpoint": ep_name}
                snap = self.state.scheduler.endpoint_snapshot(ep_name)
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
                    w = self.state.metrics.percentile("queue_wait_ms", q,
                                                 endpoint=ep_name, now=now)
                    if w is not None:
                        out.append(Metric("llmproxy_queue_wait_ms", w,
                                          {"endpoint": ep_name, "quantile": qlabel},
                                          "gauge", help="queue-wait latency (ms)"))
                    b = self.state.metrics.percentile("backend_latency_ms", q,
                                                 endpoint=ep_name, now=now)
                    if b is not None:
                        out.append(Metric("llmproxy_backend_latency_ms", b,
                                          {"endpoint": ep_name, "quantile": qlabel},
                                          "gauge", help="backend latency (ms)"))
                out.append(Metric("llmproxy_recent_timeouts_5m",
                                  self.state.metrics.count(endpoint=ep_name,
                                                      status="timeout", now=now),
                                  lbl, "gauge",
                                  help="timeouts in the last 5 min"))
                out.append(Metric("llmproxy_recent_requests_5m",
                                  self.state.metrics.count(endpoint=ep_name, now=now),
                                  lbl, "gauge",
                                  help="requests in the last 5 min"))
                ep_cfg = self.state.config.endpoints[ep_name]
                ss_consumed = self.state.metrics.slot_seconds_consumed(ep_name, now)
                out.append(Metric("llmproxy_slot_seconds_5m", ss_consumed, lbl,
                                  "gauge", help="slot-seconds consumed in 5 min"))
                if ep_cfg.max_slots > 0:
                    ss_available = ep_cfg.max_slots * 300.0
                    util = (ss_consumed / ss_available * 100) if ss_available > 0 else 0
                    out.append(Metric("llmproxy_endpoint_utilization_pct",
                                      round(util, 1), lbl, "gauge",
                                      help="5-min slot utilization %"))
                h = self.state.endpoint_health.get(ep_name, {})
                out.append(Metric("llmproxy_endpoint_healthy",
                                  1 if h.get("healthy", True) else 0, lbl, "gauge",
                                  help="1 if the endpoint is healthy (not paused)"))

            stats = self.state.scheduler.stats()
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
        for ep_name in self.state.config.endpoints:
            snap = self.state.scheduler.endpoint_snapshot(ep_name)
            snap["p50_wait_ms"] = self.state.metrics.percentile(
                "queue_wait_ms", 50, endpoint=ep_name, now=now,
            )
            snap["p95_wait_ms"] = self.state.metrics.percentile(
                "queue_wait_ms", 95, endpoint=ep_name, now=now,
            )
            snap["throughput_rps"] = round(
                self.state.metrics.throughput_rps(ep_name, now), 2,
            )
            ep_cfg = self.state.config.endpoints[ep_name]
            if ep_cfg.max_slots > 0:
                ss_consumed = self.state.metrics.slot_seconds_consumed(ep_name, now)
                ss_available = ep_cfg.max_slots * 300.0  # 5-min window
                snap["utilization_pct"] = round(
                    (ss_consumed / ss_available * 100) if ss_available > 0 else 0, 1,
                )
            h = self.state.endpoint_health.get(ep_name, {})
            snap["healthy"] = h.get("healthy", True)
            snap["paused"] = not h.get("healthy", True)  # check_alerts (Phase 2.5) keys on this
            # Survivorship fix (2026-06-06): the timeout-advice model + shadow
            # only ingest status==ok, so they reported a misleading "0 would
            # timeout" while requests were actually timing out. Surface the real
            # 5-min timeout count per endpoint so a partial stall is VISIBLE
            # (feeds the endpoint_stalled alert + health-verifier/dashboards).
            snap["recent_timeouts"] = self.state.metrics.count(
                endpoint=ep_name, status="timeout", now=now)
            endpoints[ep_name] = snap

        agents = {
            b["agent_id"]: b
            for b in self.state.budget_mgr.snapshot()
        }

        # Loop liveness is computed AT READ TIME: the poller is what evaluates
        # self.state.alerts, so a dead poller (or scheduler) can never report itself
        # through that path — only through this one.
        alerts = list(self.state.alerts)
        if not self.health.scheduler_loop_alive():
            alerts.append({
                "name": "scheduler_loop_dead", "severity": "CRITICAL",
                "detail": "scheduler loop task not running — dispatch is DOWN",
            })
        if not self.health.poller_alive():
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
            "flags": self.state.flags.as_dict(),
            "cache": self.state.cache.stats(),
            "scheduler": {
                **self.state.scheduler.stats(),
                "uptime_s": round(time.monotonic() - self.state.started_at, 0),
            },
            # Phase 5B reliability counters.
            "reliability": {
                "slot_leak_reclaimed": self.state.slot_leak_reclaimed,
                "drain_straggler_cancelled": self.state.drain_straggler_cancelled,
                "writer_thread_alive": self.state.queue_db.writer_alive(),
                "writer_thread_restarts": self.state.queue_db.writer_restarts(),
                "write_q_dropped": self.state.queue_db.write_q_dropped(),
                # Critical-loop liveness (read-time; see alerts note above).
                "scheduler_alive": self.health.scheduler_loop_alive(),
                "poller_alive": self.health.poller_alive(),
                # Source IPs seen on admin-ish routes since boot — the
                # ACL-tightening go/no-go reads this instead of grepping logs.
                "admin_ips_seen": {
                    route: sorted(ips)
                    for route, ips in self.state.admin_ips_seen.items()
                },
                # Unknown-endpoint submits since boot (shadow counter for the
                # unknown_endpoint_enforce flip check). Empty = safe to flip.
                "unknown_endpoint_submits": self.state.unknown_endpoint_submits,
                # Context-gate hits since boot (shadow counter for the
                # context_gate_enforce flip check — compare against actual
                # backend overflow errors before flipping).
                "context_overflows_shadow": self.state.context_overflows,
                # Thinking option (per-request native reasoning) health. All 0
                # until a caller opts in with thinking:true. Watch
                # thinking_truncated to tune COLLECTIVE_PROXY_THINKING_BUDGET down.
                "thinking_requests": self.state.thinking_requests,
                "thinking_clean": self.state.thinking_clean,
                "thinking_recovered": self.state.thinking_recovered,
                "thinking_truncated": self.state.thinking_truncated,
                "thinking_fallback": self.state.thinking_fallback,
                # WS-4 shadow egress detector — silent grammar-drop over ALL
                # grammar-bearing responses (read-only/zero-risk). Per-call_site
                # rate + a flat fleet rate (health-verifier thresholds the scalar).
                "silent_drop_by_call_site": {
                    cs: {**t, "rate": round(t["dropped"] / t["checked"], 4)}
                    for cs, t in self.state.shadow_drop.items() if t["checked"]
                },
                "silent_drop_rate": round(
                    sum(t["dropped"] for t in self.state.shadow_drop.values())
                    / max(1, sum(t["checked"] for t in self.state.shadow_drop.values())),
                    4,
                ),
                # Empty-completion (position-0-EOS) rescue: retries dispatched
                # with min_tokens, and how many produced a real response.
                "empty_rescue_attempts": self.state.empty_rescue_attempts,
                "empty_rescue_recovered": self.state.empty_rescue_recovered,
                # Egress degeneration guard — repetition-loop responses detected,
                # and how many an anti-repetition re-dispatch recovered.
                "degeneration_detected": self.state.degeneration_detected,
                "degeneration_recovered": self.state.degeneration_recovered,
                "degeneration_unrecovered": self.state.degeneration_unrecovered,
                "degeneration_by_call_site": {
                    cs: t for cs, t in self.state.degeneration_by_call_site.items()
                    if t["detected"]
                },
            },
            # Phase 5F — endpoints an operator has drained for maintenance.
            "paused_endpoints": sorted(self.state.paused_endpoints),
        })
    async def handle_admin_flags(self, request: Request) -> Response:
        """GET: current runtime flags. POST: update a subset (JSON object of
        flag→bool), persisted across restarts. Internal-only (ACL). This is the
        flip surface for the shadow→enforce switches and kill-switches — flags
        change behaviour immediately, no process restart."""
        remote_ip = request.client.host if request.client else "unknown"
        self.audit_admin_ip("/v1/admin/flags", remote_ip)
        if not self.state.acl.is_admin(remote_ip):
            return JSONResponse(
                {"error": f"access denied for {remote_ip}"}, status_code=403)
        if request.method == "GET":
            return JSONResponse({"flags": self.state.flags.as_dict()})
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        try:
            # File write runs off-loop; flag reads elsewhere are plain dict
            # lookups on the loop thread (single mutation source — this handler).
            updated = await asyncio.to_thread(self.state.flags.set_many, body)
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
        self.audit_admin_ip("/v1/admin/endpoints", remote_ip)
        if not self.state.acl.is_admin(remote_ip):
            return JSONResponse(
                {"error": f"access denied for {remote_ip}"}, status_code=403)
        ep = normalize_endpoint(endpoint)
        if ep not in self.state.config.endpoints:
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
            self.state.paused_endpoints.add(ep)
            # Release any already-queued interactive/foreground immediately with
            # a deferrable error (don't make them wait out their deadline).
            self.health.fast_fail_interactive(ep)
            # Open a maintenance window so timeouts during the restart are tagged
            # planned. Closed on RESUME (below).
            if self.state.queue_db is not None:
                self.state.queue_db.maintenance_open(
                    endpoint=ep, reason=reason, operator=remote_ip, source="drain")
            logger.warning(
                "endpoint %s PAUSED by operator (%s) — background defers, "
                "interactive fast-fails; backend safe to restart%s", ep, remote_ip,
                f" (reason: {reason})" if reason else "")
        else:
            self.state.paused_endpoints.discard(ep)
            self.state.dispatch_event.set()  # nudge the scheduler to drain deferred work
            if self.state.queue_db is not None:
                self.state.queue_db.maintenance_close(endpoint=ep)
            logger.warning(
                "endpoint %s RESUMED by operator (%s) — poller will re-probe / "
                "recover / re-discover capacity; deferred queue draining",
                ep, remote_ip)
        return JSONResponse({
            "endpoint": ep,
            "paused": ep in self.state.paused_endpoints,
            "healthy": self.health.endpoint_healthy(ep),
            "paused_endpoints": sorted(self.state.paused_endpoints),
        })
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
        self.audit_admin_ip("/v1/admin/maintenance", remote_ip)
        if not self.state.acl.is_admin(remote_ip):
            return JSONResponse(
                {"error": f"access denied for {remote_ip}"}, status_code=403)
        if self.state.queue_db is None:
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
            if ne not in self.state.config.endpoints:
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
                    self.state.queue_db.maintenance_record(
                        endpoint=ep, started_at=s, ended_at=en,
                        reason=reason, operator=remote_ip)
                    recorded.append({"endpoint": ep, "started_at": s,
                                     "ended_at": en, "reason": reason})
                elif duration_s is not None:
                    s = now - float(duration_s)
                    self.state.queue_db.maintenance_record(
                        endpoint=ep, started_at=s, ended_at=now,
                        reason=reason, operator=remote_ip)
                    recorded.append({"endpoint": ep, "started_at": s,
                                     "ended_at": now, "reason": reason})
                else:
                    self.state.queue_db.maintenance_open(
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
        self.audit_admin_ip("/v1/admin/maintenance", remote_ip)
        if not self.state.acl.is_admin(remote_ip):
            return JSONResponse(
                {"error": f"access denied for {remote_ip}"}, status_code=403)
        if self.state.queue_db is None:
            return JSONResponse({"hours": 0, "windows": []})
        hours = _to_float(request.query_params.get("hours"), 24)
        hours = min(max(hours, 0.1), 168)
        windows = await asyncio.to_thread(self.state.queue_db.maintenance_windows, hours)
        return JSONResponse({"hours": hours, "windows": windows})
    async def handle_metrics(self, request: Request) -> Response:
        return JSONResponse(self.state.metrics_payload(time.monotonic()))
    async def handle_fleet_activity(self, request: Request) -> Response:
        window_s = _clamp_window(request.query_params.get("window", "24h"), 86400)
        bin_s = _to_int(request.query_params.get("bin"), _bin_seconds_for(window_s))
        # §5c: heavy GROUP-BY over the whole-fleet completions table runs off the
        # event loop so it can't stall fleet LLM scheduling under a hot dashboard.
        data = await asyncio.to_thread(self.state.queue_db.fleet_activity, window_s, bin_s)
        return JSONResponse(data)
    async def handle_fleet_savings(self, request: Request) -> Response:
        since_q = request.query_params.get("since")
        today_start = float(since_q) if since_q and since_q.isdigit() else None
        data = await asyncio.to_thread(self.state.queue_db.savings_summary, today_start)
        return JSONResponse(data)
    async def handle_top_callers(self, request: Request) -> Response:
        window_s = _clamp_window(request.query_params.get("window", "1h"), 3600)
        per_endpoint = max(1, min(_to_int(request.query_params.get("per_endpoint"), 5), 20))
        data = await asyncio.to_thread(self.state.queue_db.top_callers, window_s, per_endpoint)
        return JSONResponse(data)
    async def handle_fleet_cache_stats(self, request: Request) -> Response:
        """Prefix-cache observability: per-model actual hit-rate + per-call_site
        misalignment offenders + rollup + trend + drift. Computed from the
        periodic snapshots in proxy_cache_stats (see _compute_cache_stats)."""
        window_s = _clamp_window(request.query_params.get("window", "7d"), 30 * 86400)
        snaps = await asyncio.to_thread(self.state.queue_db.cache_stats_snapshots, window_s)
        labels = cache_stats.chat_endpoint_labels()
        engines = {ep: cfg.backend_engine for ep, cfg in self.state.config.endpoints.items()}
        return JSONResponse(cache_stats.build_fleet_payload(snaps, labels, engines))
    async def handle_usage(self, request: Request) -> Response:
        dimension = request.query_params.get("by", "agent")
        if dimension not in ("agent", "call_site", "endpoint", "provider"):
            dimension = "agent"
        hours = min(_to_float(request.query_params.get("hours"), 24), 168)
        rows = await asyncio.to_thread(self.state.queue_db.usage_rollup, dimension, hours)
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
            self.state.queue_db.endpoint_series, endpoint, window_s, bin_s)
        return JSONResponse(data)
    async def handle_calls_log(self, request: Request) -> Response:
        """Ingest a non-LLM service call (audio/imagegen/ocr/translate) that
        never traversed the scheduler, so the proxy is the single fleet
        call-metrics store. Internal/LAN — gated by the same ACL as admin.
        Best-effort: validates the minimum, records, fans out, returns ok."""
        remote_ip = request.client.host if request.client else "unknown"
        self.audit_admin_ip("/v1/calls/log", remote_ip)
        if not self.state.acl.is_admin(remote_ip):
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
        if normalize_endpoint(endpoint) in self.state.config.endpoints:
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
            self.state.queue_db.persist_external_call(
                request_id=request_id, agent_id=agent_id, endpoint=endpoint,
                call_site=call_site, kind=kind, input_tokens=in_tok,
                output_tokens=out_tok, duration_s=duration_s, status=status,
                caller_id=body.get("caller_id"),
            )
            self.state.sse.publish("call.completed", {
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
    async def handle_stream(self, request: Request) -> Response:
        """Server-Sent Events: real-time `call.completed` + periodic `metrics`
        frames. Slow clients are dropped (the browser reconnects + re-syncs via
        the REST endpoints). Mirrors the host-telemetry /stream/v2 shape."""
        q = self.state.sse.subscribe()

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
                self.state.sse.unsubscribe(q)

        return StreamingResponse(event_gen(), media_type="text/event-stream", headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        })
    async def handle_cost_model(self, request: Request) -> Response:
        return JSONResponse(self.state.cost_model.snapshot())
    async def handle_history(self, request: Request) -> Response:
        hours = min(_to_float(request.query_params.get("hours"), 4), 168)
        bucket_minutes = max(1, min(_to_int(request.query_params.get("bucket_minutes"), 5), 60))
        buckets = await asyncio.to_thread(
            self.state.queue_db.history_buckets, hours, bucket_minutes)
        return JSONResponse({"buckets": buckets})
    async def handle_recent(self, request: Request) -> Response:
        limit = min(_to_int(request.query_params.get("limit"), 50), 200)
        rows = await asyncio.to_thread(self.state.queue_db.recent_requests, limit)
        return JSONResponse({"requests": rows})
    async def handle_inflight(self, request: Request) -> Response:
        """Live list of currently-executing requests + per-endpoint occupancy —
        the proxy is the dispatch authority, so this is the authoritative
        real-time "what's flowing through the LLM systems" view. Pure in-memory
        snapshot; also pushed via the SSE `inflight` frame for sub-second feel."""
        snap = self.state.scheduler.inflight_snapshot(time.monotonic())
        snap["ts"] = time.time()
        return JSONResponse(snap)
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
        if endpoint not in self.state.config.endpoints:
            return JSONResponse(
                {"error": f"unknown model {model!r}",
                 "known": sorted(self.state.config.endpoints)},
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

        advice = self.state.timeout_model.advise(endpoint, priority, est_in, est_out)
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
        report = await asyncio.to_thread(self.state.queue_db.timeout_shadow_report, hours)
        return JSONResponse({"hours": hours, "report": report})
    async def handle_timeouts_report(self, request: Request) -> Response:
        """Calls that hit their timeout instead of finishing, per
        (model, tier, layer), with the load context when they gave up and
        how many fired below the recommended deadline (premature)."""
        hours = _to_float(request.query_params.get("hours"), 24)
        hours = min(max(hours, 0.1), 168)
        report = await asyncio.to_thread(self.state.queue_db.timeouts_report, hours)
        return JSONResponse({"hours": hours, **report})
    async def handle_health(self, request: Request) -> Response:
        ok = self.health.scheduler_loop_alive()
        poller_ok = self.health.poller_alive()
        unhealthy = [ep for ep, h in self.state.endpoint_health.items() if not h["healthy"]]
        # The proxy is UP iff its scheduler is alive (200). A dead BACKEND
        # degrades status but must NOT 503 the proxy — that would make a monitor
        # restart a healthy front door over a backend blip (alert-don't-kill).
        # A dead POLLER also only degrades: dispatch still works, but health
        # probing / alerting / budget persistence have stopped — surface it.
        status = "ok" if (ok and poller_ok and not unhealthy) else ("degraded" if ok else "down")
        return JSONResponse(
            {
                "status": status,
                "uptime_s": round(time.monotonic() - self.state.started_at, 0),
                "total_dispatched": self.state.scheduler.stats()["total_dispatched"],
                "endpoints": len(self.state.config.endpoints),
                "total_slots": self.state.config.total_fleet_slots,
                "unhealthy_endpoints": unhealthy,
                "scheduler_alive": ok,
                "poller_alive": poller_ok,
            },
            status_code=200 if ok else 503,
        )
    def audit_admin_ip(self, route: str, remote_ip: str) -> None:
        """Track source IPs per admin-ish route (exposed on /v1/status) and log
        the FIRST hit per (route, ip) — the data the ACL-tightening go/no-go
        needs, without per-hit log volume."""
        seen = self.state.admin_ips_seen.setdefault(route, set())
        if remote_ip not in seen:
            seen.add(remote_ip)
            logger.info("admin-audit: first hit on %s from %s", route, remote_ip)
