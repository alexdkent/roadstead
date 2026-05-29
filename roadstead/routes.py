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
        Route("/v1/timeout-advice", handle_timeout_advice, methods=["GET"]),
        Route("/v1/timeout-advice/shadow-report", handle_timeout_shadow_report, methods=["GET"]),
        Route("/health", handle_health, methods=["GET"]),
    ]
