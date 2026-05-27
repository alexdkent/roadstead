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

from .config import ProxyConfig, load_agent_configs
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
            # Per-agent DRR quota overrides (weight, max_balance_ss,
            # default_priority). Reads originfleet/llmproxy/agents.yaml
            # by default, or the path in LLM_PROXY_AGENTS_CONFIG.
            # Missing file → empty dict → proxy lazy-creates agent
            # configs at AgentQuotaConfig dataclass defaults.
            agents=load_agent_configs(),
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

    # Emit a ship_version line on startup so the ship harness'
    # restart-phase health-poll can confirm the new build is running.
    # Same shape as originfleet.tools.llm_qos and every agent — the
    # harness greps for "[ship_version] agent=<name>" in the log.
    from originfleet.framework.ship_version import log_ship_version
    log_ship_version("llmproxy")

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
