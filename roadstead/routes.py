"""Starlette route handlers for the LLM proxy.

Provides:
  - POST /v1/submit        — internal submission API
  - POST /v1/chat/completions — OpenAI-compatible (LAN consumers)
  - POST /v1/embeddings    — OpenAI-compatible embeddings
  - GET  /v1/models        — list available endpoints
  - GET  /v1/status        — live scheduler status
  - GET  /v1/metrics       — rolling metrics
  - GET  /v1/metrics/cost-model — cost model state
  - GET  /v1/timeout-advice — recommended timeout for a model/tier/size
  - GET  /v1/timeout-advice/shadow-report — shadow-mode impact summary
  - GET  /v1/timeouts      — calls that hit their timeout (per model/tier/layer)
  - GET/POST /v1/admin/maintenance — annotate/list planned-restart windows
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

if TYPE_CHECKING:
    from .service import ProxyService

logger = logging.getLogger(__name__)


def make_routes(svc: "ProxyService") -> list[Route]:
    """Build the Starlette route list for the proxy."""

    async def handle_submit(request: Request) -> Response:
        body = await request.json()
        return await svc.handle_submit(body, request)

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

    return [
        Route("/v1/submit", handle_submit, methods=["POST"]),
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
        Route("/health", handle_health, methods=["GET"]),
        # Prometheus text exposition of current QoS aggregates (scraped by VM).
        Route("/metrics", handle_prometheus_metrics, methods=["GET"]),
    ]
