"""Starlette route handlers for the LLM proxy.

Two north faces, and the split is deliberate (``docs/api.md`` §1):

  - ``/v1/*``  — OpenAI-compatible, strictly. A drop-in target for existing
                 clients; enrichment rides in ``X-Roadstead-*`` headers and
                 never in the body.
  - ``/rs/v1/*`` — the enriched Roadstead API (``enriched.py``), with its own
                 version because the other one is versioned by OpenAI.

Provides:
  - GET  /rs/v1/models     — enriched catalogue: capabilities, live state, price
  - POST /rs/v1/plan       — resolve an intent and price it, without dispatching
  - POST /rs/v1/chat       — the enriched call
  - POST /v1/chat/completions — OpenAI-compatible (LAN consumers)
  - POST /v1/embeddings    — OpenAI-compatible embeddings
  - GET  /v1/models        — list available endpoints
  - GET  /v1/status        — live scheduler status
  - GET  /v1/metrics       — rolling metrics
  - GET  /v1/metrics/cost-model — cost model state
  - GET  /v1/timeout-advice — recommended timeout for a model/tier/size
  - GET  /v1/timeout-advice/shadow-report — shadow-mode impact summary
  - GET  /v1/timeouts      — calls that hit their timeout (per model/tier/layer)
  - GET  /v1/timeouts/stalls — backend-stall aborts for one caller + window
  - GET/POST /v1/admin/maintenance — annotate/list planned-restart windows
  - GET  /rs/v1/admin/config    — configuration sources + what is NOT in force
  - GET/POST /rs/v1/admin/keys  — the key registry (redacted) / enrol a key
  - DELETE /rs/v1/admin/keys/{key_id} — revoke a key
  - POST /rs/v1/admin/keys/{key_id}/rotate — issue a successor, retire this one
  - GET  /rs/v1/admin/callers   — per-caller identity, quota, DRR, spend
  - PATCH /rs/v1/admin/callers/{agent_id} — edit one caller's quota
  - GET  /rs/v1/admin/providers — providers + endpoints: declared vs in force
  - POST /rs/v1/admin/providers/{provider}/credential — supply an api_key_env value
  - POST /rs/v1/admin/endpoints/{endpoint}/status — promote/demote (active|planned)
  - PUT/PATCH/DELETE /rs/v1/admin/providers/{provider} — create/edit/delete (J2)
  - PUT/PATCH/DELETE /rs/v1/admin/endpoints/{endpoint} — create/edit/delete (J2)
  - GET  /rs/v1/admin/audit     — who changed what, and when
  - GET  /rs/v1/admin/ui        — the operator UI (only when ROADSTEAD_ADMIN_UI)
  - GET  /rs/v1/admin/stream    — the SSE stream, aliased for the UI's EventSource
  - GET  /health           — health check
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .enriched import PREFIX
from .management import PREFIX as ADMIN_PREFIX
from .management import admin_ui_enabled

if TYPE_CHECKING:
    from .service import ProxyService

logger = logging.getLogger(__name__)


def make_routes(svc: "ProxyService") -> list[Route]:
    """Build the Starlette route list for the proxy."""

    async def handle_rs_models(request: Request) -> Response:
        return await svc.handle_rs_models(request)

    async def handle_rs_plan(request: Request) -> Response:
        body = await request.json()
        return await svc.handle_rs_plan(body, request)

    async def handle_rs_chat(request: Request) -> Response:
        body = await request.json()
        return await svc.handle_rs_chat(body, request)

    async def handle_chat_completions(request: Request) -> Response:
        body = await request.json()
        return await svc.handle_openai_chat(body, request)

    async def handle_embeddings(request: Request) -> Response:
        body = await request.json()
        return await svc.handle_openai_embeddings(body, request)

    async def handle_models(request: Request) -> Response:
        return await svc.handle_models(request)

    async def handle_status(request: Request) -> Response:
        return await svc.handle_status(request)

    async def handle_metrics(request: Request) -> Response:
        return await svc.handle_metrics(request)

    async def handle_prometheus_metrics(request: Request) -> Response:
        return await svc.handle_prometheus_metrics(request)

    async def handle_cost_model(request: Request) -> Response:
        return await svc.handle_cost_model(request)

    async def handle_health(request: Request) -> Response:
        return await svc.handle_health(request)

    async def handle_readyz(request: Request) -> Response:
        return await svc.handle_readyz(request)

    async def handle_history(request: Request) -> Response:
        return await svc.handle_history(request)

    async def handle_recent(request: Request) -> Response:
        return await svc.handle_recent(request)

    async def handle_timeout_advice(request: Request) -> Response:
        return await svc.handle_timeout_advice(request)

    async def handle_timeout_shadow_report(request: Request) -> Response:
        return await svc.handle_timeout_shadow_report(request)

    async def handle_timeouts_report(request: Request) -> Response:
        return await svc.handle_timeouts_report(request)

    async def handle_stall_aborts(request: Request) -> Response:
        return await svc.handle_stall_aborts(request)

    async def handle_admin_pause(request: Request) -> Response:
        return await svc.handle_admin_endpoint_pause(
            request.path_params["endpoint"], request, pause=True)

    async def handle_admin_resume(request: Request) -> Response:
        return await svc.handle_admin_endpoint_pause(
            request.path_params["endpoint"], request, pause=False)

    async def handle_maintenance(request: Request) -> Response:
        if request.method == "GET":
            return await svc.handle_maintenance_list(request)
        return await svc.handle_maintenance(request)

    async def handle_admin_flags(request: Request) -> Response:
        return await svc.handle_admin_flags(request)

    # Phase 1 — proxy as fleet call-metrics authority.
    async def handle_stream(request: Request) -> Response:
        return await svc.handle_stream(request)

    async def handle_calls_log(request: Request) -> Response:
        return await svc.handle_calls_log(request)

    async def handle_fleet_activity(request: Request) -> Response:
        return await svc.handle_fleet_activity(request)

    async def handle_fleet_savings(request: Request) -> Response:
        return await svc.handle_fleet_savings(request)

    async def handle_top_callers(request: Request) -> Response:
        return await svc.handle_top_callers(request)

    async def handle_fleet_cache_stats(request: Request) -> Response:
        return await svc.handle_fleet_cache_stats(request)

    async def handle_cache_attribution(request: Request) -> Response:
        return await svc.handle_cache_attribution(request)

    async def handle_usage(request: Request) -> Response:
        return await svc.handle_usage(request)

    async def handle_series(request: Request) -> Response:
        return await svc.handle_series(request)

    async def handle_inflight(request: Request) -> Response:
        return await svc.handle_inflight(request)

    async def handle_admin_config(request: Request) -> Response:
        return await svc.handle_admin_config(request)

    async def handle_admin_keys(request: Request) -> Response:
        return await svc.handle_admin_keys(request)

    async def handle_admin_key(request: Request) -> Response:
        return await svc.handle_admin_key(request)

    async def handle_admin_callers(request: Request) -> Response:
        return await svc.handle_admin_callers(request)

    async def handle_admin_caller(request: Request) -> Response:
        return await svc.handle_admin_caller(request)

    async def handle_admin_providers(request: Request) -> Response:
        return await svc.handle_admin_providers(request)

    async def handle_admin_provider_credential(request: Request) -> Response:
        return await svc.handle_admin_provider_credential(request)

    async def handle_admin_endpoint_status(request: Request) -> Response:
        return await svc.handle_admin_endpoint_status(request)

    async def handle_admin_catalog_entry(request: Request) -> Response:
        return await svc.handle_admin_catalog_entry(request)

    async def handle_admin_key_rotate(request: Request) -> Response:
        return await svc.handle_admin_key_rotate(request)

    async def handle_admin_audit(request: Request) -> Response:
        return await svc.handle_admin_audit(request)

    async def handle_admin_ui(request: Request) -> Response:
        return await svc.handle_admin_ui(request)

    ui_routes: list[Route] = []
    if admin_ui_enabled():
        # 🚨 Registered only when ROADSTEAD_ADMIN_UI is set, so OFF means the
        # route does not EXIST rather than that it refuses. A capability that
        # widens what is reachable is the operator's decision — same posture as
        # ROADSTEAD_TRUSTED_PROXIES and ROADSTEAD_REQUIRE_API_KEY.
        ui_routes = [
            Route(f"{ADMIN_PREFIX}/ui", handle_admin_ui, methods=["GET"]),
            # 🚨 The stream, aliased under the admin prefix — and this alias is
            # load-bearing rather than tidy. `EventSource` cannot set a request
            # header AT ALL, so a browser page can only reach an authenticated
            # stream via credentials the browser itself attaches; a browser
            # attaches cached Basic credentials by directory, and this is the
            # one directory the UI page was challenged in. Same handler, same
            # gate — the pattern the four control routes already established.
            Route(f"{ADMIN_PREFIX}/stream", handle_stream, methods=["GET"]),
        ]

    return ui_routes + [
        # --- the enriched north face (roadmap Workstream C) -----------------
        # 🚨 This REPLACES the `/v1/submit` envelope, which was removed rather
        # than deprecated: it had no intent vocabulary, no attribution and no
        # timing, and every one of those would have had to be bolted onto a
        # shape that was never designed to carry them. See CHANGELOG.md.
        Route(f"{PREFIX}/models", handle_rs_models, methods=["GET"]),
        Route(f"{PREFIX}/plan", handle_rs_plan, methods=["POST"]),
        Route(f"{PREFIX}/chat", handle_rs_chat, methods=["POST"]),
        Route("/v1/chat/completions", handle_chat_completions, methods=["POST"]),
        Route("/v1/embeddings", handle_embeddings, methods=["POST"]),
        Route("/v1/models", handle_models, methods=["GET"]),
        Route("/v1/status", handle_status, methods=["GET"]),
        Route("/v1/metrics", handle_metrics, methods=["GET"]),
        Route("/v1/metrics/cost-model", handle_cost_model, methods=["GET"]),
        Route("/v1/history", handle_history, methods=["GET"]),
        Route("/v1/recent", handle_recent, methods=["GET"]),
        Route("/v1/inflight", handle_inflight, methods=["GET"]),
        Route("/v1/timeout-advice", handle_timeout_advice, methods=["GET"]),
        Route("/v1/timeout-advice/shadow-report", handle_timeout_shadow_report, methods=["GET"]),
        Route("/v1/timeouts", handle_timeouts_report, methods=["GET"]),
        # Per-caller, per-window backend-stall lookup. /v1/timeouts aggregates;
        # this answers "did the backend stall under THIS run?" — the evidence a
        # consumer needs to tell our substrate failing from its own hang.
        Route("/v1/timeouts/stalls", handle_stall_aborts, methods=["GET"]),
        # Phase 1 — fleet call-metrics authority: real-time stream, non-LLM
        # ingest, and the usage/savings rollups ported from the host daemon.
        Route("/v1/stream", handle_stream, methods=["GET"]),
        Route("/v1/calls/log", handle_calls_log, methods=["POST"]),
        Route("/v1/fleet/activity", handle_fleet_activity, methods=["GET"]),
        Route("/v1/fleet/savings", handle_fleet_savings, methods=["GET"]),
        Route("/v1/fleet/top-callers", handle_top_callers, methods=["GET"]),
        Route("/v1/fleet/cache-stats", handle_fleet_cache_stats, methods=["GET"]),
        Route("/v1/fleet/cache-attribution", handle_cache_attribution, methods=["GET"]),
        Route("/v1/usage", handle_usage, methods=["GET"]),
        Route("/v1/series", handle_series, methods=["GET"]),
        # Phase 5F — operator drain for backend maintenance (internal-only/ACL).
        Route("/v1/admin/endpoints/{endpoint}/pause", handle_admin_pause, methods=["POST"]),
        Route("/v1/admin/endpoints/{endpoint}/resume", handle_admin_resume, methods=["POST"]),
        # Maintenance-window annotation: tag a deliberate restart as PLANNED so
        # its timeout burst doesn't read as an incident in /v1/timeouts.
        # POST records a window; GET lists recent windows. (internal-only/ACL)
        Route("/v1/admin/maintenance", handle_maintenance, methods=["GET", "POST"]),
        # Runtime feature flags: the shadow→enforce flip surface (internal/ACL).
        Route("/v1/admin/flags", handle_admin_flags, methods=["GET", "POST"]),
        # --- the management plane (roadmap Workstream E) ---------------------
        # 🚨 On `/rs/v1/admin/*` rather than `/v1/admin/*`: `/v1` is versioned by
        # OpenAI, and management is the surface most likely to need its own
        # second version. See `management.py`.
        Route(f"{ADMIN_PREFIX}/config", handle_admin_config, methods=["GET"]),
        Route(f"{ADMIN_PREFIX}/keys", handle_admin_keys, methods=["GET", "POST"]),
        Route(f"{ADMIN_PREFIX}/keys/{{key_id}}", handle_admin_key, methods=["DELETE"]),
        # 🚨 One action, because doing it by hand is two calls in an order that
        # matters and both orders are wrong. See ManagementApi.
        Route(f"{ADMIN_PREFIX}/keys/{{key_id}}/rotate", handle_admin_key_rotate,
              methods=["POST"]),
        Route(f"{ADMIN_PREFIX}/callers", handle_admin_callers, methods=["GET"]),
        Route(f"{ADMIN_PREFIX}/callers/{{agent_id}}", handle_admin_caller,
              methods=["PATCH"]),
        Route(f"{ADMIN_PREFIX}/providers", handle_admin_providers, methods=["GET"]),
        # J1 — the two writes on this plane. Both mutate, so both inherit the
        # read/write split from the HTTP METHOD in the shared gate; there is no
        # list of write routes to fall behind.
        Route(f"{ADMIN_PREFIX}/providers/{{provider}}/credential",
              handle_admin_provider_credential, methods=["POST"]),
        Route(f"{ADMIN_PREFIX}/endpoints/{{endpoint}}/status",
              handle_admin_endpoint_status, methods=["POST"]),
        # J2 — the catalog is writable. One handler, six methods: PUT replaces a
        # stanza, PATCH merges into it, DELETE tombstones the name. All three
        # validate by building the catalog they would install.
        Route(f"{ADMIN_PREFIX}/providers/{{provider}}",
              handle_admin_catalog_entry, methods=["PUT", "PATCH", "DELETE"]),
        Route(f"{ADMIN_PREFIX}/endpoints/{{endpoint}}",
              handle_admin_catalog_entry, methods=["PUT", "PATCH", "DELETE"]),
        # Every mutating admin route records here. A READ, so a read-only admin
        # scope reaches it — which is the point: the operator who cannot change
        # anything is often exactly the one auditing what changed.
        Route(f"{ADMIN_PREFIX}/audit", handle_admin_audit, methods=["GET"]),
        # The four control routes that predate the management plane, served at
        # the new prefix TOO. They keep their `/v1/admin/*` spelling because §3
        # published it and external consumers read it; the alias exists so an
        # operator has one prefix rather than two. Same handler, same gate — the
        # pair is pinned by `tests/test_management_plane.py`, because an alias
        # that silently stopped aliasing is a control surface that works on one
        # spelling and 404s on the other.
        Route(f"{ADMIN_PREFIX}/endpoints/{{endpoint}}/pause", handle_admin_pause,
              methods=["POST"]),
        Route(f"{ADMIN_PREFIX}/endpoints/{{endpoint}}/resume", handle_admin_resume,
              methods=["POST"]),
        Route(f"{ADMIN_PREFIX}/maintenance", handle_maintenance,
              methods=["GET", "POST"]),
        Route(f"{ADMIN_PREFIX}/flags", handle_admin_flags, methods=["GET", "POST"]),
        Route("/health", handle_health, methods=["GET"]),
        # LIVENESS is /health (fail-open, alert-don't-kill). READINESS is here
        # and fails CLOSED on the conversational endpoint — §8.0 req 4. The two
        # answer different questions; do not collapse them.
        Route("/readyz", handle_readyz, methods=["GET"]),
        # Prometheus text exposition of current QoS aggregates (scraped by VM).
        Route("/metrics", handle_prometheus_metrics, methods=["GET"]),
    ]
