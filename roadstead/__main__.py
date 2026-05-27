"""Entry point for the LLM proxy service.

    python -m originfleet.llmproxy [--port 42161]

Runs as a standalone Starlette/uvicorn process.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys

import uvicorn
from starlette.applications import Starlette

from .config import ProxyConfig
from .routes import make_routes
from .service import ProxyService

logger = logging.getLogger("originfleet.llmproxy")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Centralized LLM scheduler proxy")
    p.add_argument("--port", type=int, default=int(os.environ.get("LLM_PROXY_PORT", "42161")))
    p.add_argument("--host", default=os.environ.get("LLM_PROXY_HOST", "0.0.0.0"))
    p.add_argument("--log-level", default=os.environ.get("LLM_PROXY_LOG_LEVEL", "info"))
    return p.parse_args()


def build_app(config: ProxyConfig | None = None) -> Starlette:
    """Build the Starlette app.  Usable from tests without running uvicorn."""
    if config is None:
        data_dir = os.environ.get(
            "LLM_PROXY_DATA_DIR",
            os.path.join(os.environ.get("COLLECTIVE_HOT_ROOT", "/tmp"), "agents", "llmproxy"),
        )
        os.makedirs(data_dir, exist_ok=True)
        log_dir = os.path.join(os.environ.get("COLLECTIVE_HOT_ROOT", "/tmp"), "logs")
        os.makedirs(log_dir, exist_ok=True)

        config = ProxyConfig(
            queue_db_path=os.environ.get(
                "LLM_PROXY_QUEUE_DB",
                os.path.join(data_dir, "queue.db"),
            ),
            request_log_path=os.environ.get(
                "LLM_PROXY_REQUEST_LOG",
                os.path.join(log_dir, "llmproxy_requests.jsonl"),
            ),
        )

    svc = ProxyService(config)
    routes = make_routes(svc)
    app = Starlette(
        routes=routes,
        on_startup=[svc.startup],
        on_shutdown=[svc.shutdown],
    )
    app.state.proxy_service = svc
    return app


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    app = build_app()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=False,
    )


if __name__ == "__main__":
    main()
