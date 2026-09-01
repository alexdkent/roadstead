"""Entry point for the LLM proxy service.

    python -m roadstead [--port 42161]

Runs as a standalone Starlette/uvicorn process.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

os.environ.setdefault("ROADSTEAD_AGENT_NAME", "llmproxy")

import json

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import ProxyConfig, load_agent_configs
from .hooks import set_degradation_sink
from .routes import make_routes
from .service import _DRAIN_DEADLINE_S, ProxyService

logger = logging.getLogger("roadstead")

# Server-side idle keepalive close (seconds). Raised from uvicorn's DEFAULT 5s
# (2026-07-06 — regression_ledger: llmproxy-keepalive-disconnect): 5s coincided
# with httpx's default client keepalive_expiry, so under a concurrent burst either
# side could close an idle socket first and a POST reusing a server-closed one
# raised RemoteProtocolError("Server disconnected without sending a response.").
# MUST stay ABOVE the client's keepalive (llm_proxy_client._CLIENT_KEEPALIVE_EXPIRY_S,
# 4.5s) with margin, so the CLIENT always retires idle connections first — the proxy
# never yanks a socket an agent is about to reuse. test_proxy_keepalive_invariant pins it.
PROXY_SERVER_KEEPALIVE_S = int(os.environ.get("ROADSTEAD_PROXY_SERVER_KEEPALIVE_S", "30"))


async def _on_invalid_json(request: Request, exc: Exception) -> JSONResponse:
    """Malformed request body → clean 400 (not an unhandled ASGI 500)."""
    return JSONResponse({"status": "error", "error": "invalid JSON body"},
                        status_code=400)


async def _on_unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Backstop for ANY uncaught route exception: log it and return a clean
    JSON 500 instead of a raw ASGI 500. This is the front door for all fleet
    LLM traffic — a malformed field from any caller must never surface as an
    unhandled `Exception in ASGI application`."""
    logger.exception("unhandled proxy error in %s %s",
                     request.method, request.url.path)
    return JSONResponse({"status": "error", "error": "internal proxy error"},
                        status_code=500)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Centralized LLM scheduler proxy")
    p.add_argument("--port", type=int, default=int(os.environ.get("LLM_PROXY_PORT", "42161")))
    p.add_argument("--host", default=os.environ.get("LLM_PROXY_HOST", "0.0.0.0"))
    p.add_argument("--log-level", default=os.environ.get("LLM_PROXY_LOG_LEVEL", "info"))
    return p.parse_args()


def warn_on_retired_env_vars() -> list[str]:
    """Say something when a `COLLECTIVE_*` variable is still set.

    🚨 Every environment variable was renamed `COLLECTIVE_* -> ROADSTEAD_*` on
    2026-08-31 (CHANGELOG: a recorded break). The dangerous half of that rename
    is not the flags that stop working — it is that they stop working IN
    SILENCE: an operator who had turned a correction layer OFF gets it back ON,
    a tuned keepalive reverts to the default, and nothing anywhere says so. The
    old names are NOT honoured, deliberately — two spellings for one switch is
    how they end up disagreeing — but an unread one is worth a loud line.

    Returns the retired names found, so a test can prove this fires.
    """
    stale = sorted(k for k in os.environ if k.startswith("COLLECTIVE_"))
    for name in stale:
        logging.getLogger(__name__).warning(
            "IGNORED: %s is a retired variable name and has no effect. Rename it "
            "to %s.", name, "ROADSTEAD_" + name[len("COLLECTIVE_"):])
    return stale


def build_app(config: ProxyConfig | None = None) -> Starlette:
    """Build the Starlette app.  Usable from tests without running uvicorn."""
    warn_on_retired_env_vars()
    if config is None:
        data_dir = os.environ.get(
            "LLM_PROXY_DATA_DIR",
            os.path.join(os.environ.get("ROADSTEAD_HOT_ROOT", "/tmp"), "agents", "llmproxy"),
        )
        os.makedirs(data_dir, exist_ok=True)
        log_dir = os.path.join(os.environ.get("ROADSTEAD_HOT_ROOT", "/tmp"), "logs")
        os.makedirs(log_dir, exist_ok=True)

        config = ProxyConfig(
            queue_db_path=os.environ.get(
                "LLM_PROXY_QUEUE_DB",
                os.path.join(data_dir, "queue.db"),
            ),
            # Runtime-mutable feature flags (flags.py) — persisted next to the
            # queue DB, flipped via POST /v1/admin/flags (no env gates).
            runtime_flags_path=os.environ.get(
                "LLM_PROXY_RUNTIME_FLAGS",
                os.path.join(data_dir, "runtime_flags.json"),
            ),
            # The management plane's overlay (management.AdminOverlay) — runtime
            # key enrolments, revocations and quota overrides. Beside the flags
            # file for the same reason: it is state the API writes, never
            # something an operator hand-edits, and it must not be mixed in with
            # config that is.
            admin_store_path=os.environ.get(
                "ROADSTEAD_ADMIN_STORE",
                os.path.join(data_dir, "admin_overlay.json"),
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

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        """Startup/shutdown wiring, as an ASGI lifespan context manager.

        Replaced the ``on_startup=``/``on_shutdown=`` constructor arguments on
        2026-08-31: Starlette 1.0 removed them, and the upper bound that kept
        this working was a ceiling on a core dependency. Semantics are
        unchanged — ``svc.startup()`` runs once before the first request,
        ``svc.shutdown()`` runs the bounded drain on ``lifespan.shutdown``,
        which is what uvicorn sends on SIGTERM.

        The ``finally`` is deliberate and is NOT what ``on_shutdown`` did: if
        the lifespan task is cancelled rather than shut down cleanly,
        ``on_shutdown`` handlers never ran at all, so the drain was skipped
        outright. Here it at least starts — ``_draining`` is set and the
        scheduler stops admitting before the first ``await`` can re-raise the
        cancellation. A startup failure still propagates without running
        shutdown, exactly as before, because the ``try`` is entered after
        ``svc.startup()`` returns.
        """
        await svc.startup()
        try:
            yield
        finally:
            await svc.shutdown()

    app = Starlette(
        routes=routes,
        lifespan=lifespan,
        # Robustness backstop: malformed JSON → 400; any other uncaught route
        # exception → logged clean 500 (never a raw ASGI 500). Per-field
        # coercions (priority/timeout_s/query-params) are handled at the source;
        # this catches anything they miss.
        exception_handlers={
            # A non-UTF8 request body raises UnicodeDecodeError inside
            # request.json() BEFORE json parsing — it is a ValueError but NOT a
            # JSONDecodeError, so without this entry it fell through to the
            # generic 500 backstop. Both malformed-encoding and malformed-JSON
            # bodies are the same "unparseable body" class → clean 400.
            UnicodeDecodeError: _on_invalid_json,
            json.JSONDecodeError: _on_invalid_json,
            Exception: _on_unhandled,
        },
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
    # httpx logs every backend request at INFO ("HTTP Request: POST ..."), an
    # order-of-magnitude log inflation the proxy's own per-request INFO line
    # was already removed for. Failures still surface via our own handlers.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # ---- Host-application integration (OPTIONAL) --------------------------
    # This entrypoint is the ONLY module in the package that knows originfleet
    # exists, and every import below is soft: the proxy runs fully standalone
    # without them, falling back to the built-in reporting in ``hooks.py``. A
    # standalone deployment replaces this file wholesale and drops the block.
    # Keep it that way — an unguarded import here re-couples the package.
    try:
        from originfleet.framework.observability import degradation
        set_degradation_sink(degradation)
    except ImportError:
        logging.getLogger(__name__).debug(
            "originfleet observability unavailable — degradations will be "
            "reported via the built-in logging sink")

    # Emit a ship_version line on startup so the ship harness'
    # restart-phase health-poll can confirm the new build is running.
    # Same shape as originfleet.tools.llm_qos and every agent — the
    # harness greps for "[ship_version] agent=<name>" in the log.
    try:
        from originfleet.framework.ship_version import log_ship_version
        log_ship_version("llmproxy")
    except ImportError:
        logging.getLogger(__name__).debug(
            "originfleet ship_version unavailable — no [ship_version] marker")

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
        # Idle keepalive close — raised above the client's keepalive_expiry so the
        # client retires idle sockets first (no stale-reuse RemoteProtocolError).
        timeout_keep_alive=PROXY_SERVER_KEEPALIVE_S,
    )


def _test_cli() -> None:
    """Entry point for `python -m roadstead test ...`."""
    p = argparse.ArgumentParser(prog="roadstead test")
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
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from .test_harness import ProxyTestHarness, ShapingConfig

    # `sim` defines no --db (pure in-process); replay/ab do. getattr keeps the
    # shared path computation from AttributeError-ing the sim mode.
    db_path = getattr(args, "db", None) or os.path.join(
        os.environ.get("ROADSTEAD_HOT_ROOT", "/tmp"),
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
