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

os.environ.setdefault("COLLECTIVE_AGENT_NAME", "llmproxy")

import uvicorn
from starlette.applications import Starlette

from .config import ProxyConfig, load_agent_configs
from .routes import make_routes
from .service import _DRAIN_DEADLINE_S, ProxyService

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
            # queue.db maintenance knobs (persistence cleanup) — env overrides,
            # falling back to the ProxyConfig dataclass defaults.
            payload_retention_s=float(os.environ.get(
                "LLM_PROXY_PAYLOAD_RETENTION_S", 48 * 3600.0)),
            completions_retention_s=float(os.environ.get(
                "LLM_PROXY_COMPLETIONS_RETENTION_S", 30 * 86400.0)),
            wal_checkpoint_interval_s=float(os.environ.get(
                "LLM_PROXY_WAL_CHECKPOINT_INTERVAL_S", 300.0)),
            incremental_vacuum_interval_s=float(os.environ.get(
                "LLM_PROXY_INCR_VACUUM_INTERVAL_S", 600.0)),
            incremental_vacuum_pages=int(os.environ.get(
                "LLM_PROXY_INCR_VACUUM_PAGES", 4000)),
            startup_vacuum_freelist_threshold_bytes=int(os.environ.get(
                "LLM_PROXY_STARTUP_VACUUM_FREELIST_BYTES", 200 * 1024 * 1024)),
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
        # Phase 2.1/5C: bound the graceful (SIGTERM) shutdown so it can't hang
        # forever behind a slow in-flight request. svc.shutdown() drains within
        # _DRAIN_DEADLINE_S, then the queue DB close() does flush(5s)+join(5s).
        # uvicorn's budget MUST exceed drain + close-tail (+margin) or it
        # hard-kills the process mid-flush and drops queued writes (budgets +
        # completions). Derive it so the two can't drift apart.
        timeout_graceful_shutdown=int(_DRAIN_DEADLINE_S) + 18,  # 30 + 18 = 48s
    )


def _test_cli() -> None:
    """Entry point for `python -m originfleet.llmproxy test ...`."""
    p = argparse.ArgumentParser(prog="originfleet.llmproxy test")
    sub = p.add_subparsers(dest="mode", required=True)

    sim_p = sub.add_parser("sim", help="Discrete-event simulation")
    sim_p.add_argument("scenario", help="Scenario name or 'all'")

    replay_p = sub.add_parser("replay", help="Replay recorded traffic through proxy")
    replay_p.add_argument("--hours", type=float, default=4.0)
    replay_p.add_argument("--endpoint", default=None)
    replay_p.add_argument("--call-site", default=None)
    replay_p.add_argument("--compress", type=float, default=1.0)
    replay_p.add_argument("--multiply", type=int, default=1)
    replay_p.add_argument("--proxy-url", default="http://127.0.0.1:42161")
    replay_p.add_argument("--db", default=None, help="Path to proxy queue.db")

    ab_p = sub.add_parser("ab", help="A/B backend comparison")
    ab_p.add_argument("--hours", type=float, default=4.0)
    ab_p.add_argument("--endpoint", default=None)
    ab_p.add_argument("--call-site", default=None)
    ab_p.add_argument("--shadow-host", required=True)
    ab_p.add_argument("--shadow-port", type=int, required=True)
    ab_p.add_argument("--concurrency", type=int, default=4)
    ab_p.add_argument("--proxy-url", default="http://127.0.0.1:42161")
    ab_p.add_argument("--db", default=None, help="Path to proxy queue.db")

    args = p.parse_args(sys.argv[2:])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    from .test_harness import ProxyTestHarness, ShapingConfig

    db_path = args.db or os.path.join(
        os.environ.get("COLLECTIVE_HOT_ROOT", "/tmp"),
        "agents", "llmproxy", "queue.db",
    )

    if args.mode == "sim":
        harness = ProxyTestHarness()
        if args.scenario == "all":
            from .simulation import BUILTIN_SCENARIOS as SCENARIOS
            for name in sorted(SCENARIOS):
                result = harness.run_sim(name)
                print(result.report())
                status = "PASS" if result.passed else "FAIL"
                print(f"  → {status}")
        else:
            result = harness.run_sim(args.scenario)
            print(result.report())
            status = "PASS" if result.passed else "FAIL"
            print(f"  → {status}")

    elif args.mode == "replay":
        harness = ProxyTestHarness(
            proxy_url=args.proxy_url, db_path=db_path,
        )
        shaping = ShapingConfig(
            compress=args.compress,
            multiply=args.multiply,
            endpoint_filter=args.endpoint,
            call_site_filter=args.call_site,
        )
        report = asyncio.run(harness.run_replay(
            hours=args.hours, shaping=shaping,
        ))
        print(report.summary())

    elif args.mode == "ab":
        harness = ProxyTestHarness(
            proxy_url=args.proxy_url, db_path=db_path,
        )
        report = asyncio.run(harness.run_ab(
            hours=args.hours,
            endpoint=args.endpoint,
            call_site=args.call_site,
            shadow_host=args.shadow_host,
            shadow_port=args.shadow_port,
            concurrency=args.concurrency,
        ))
        print(report.summary())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _test_cli()
    else:
        main()
