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
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import (
    ProxyConfig,
    env_with_legacy_prefix as _env,
    load_agent_configs,
    max_request_bytes_from_env,
)
from .observability import (
    DEFAULT_REQUEST_LOG_BACKUPS,
    DEFAULT_REQUEST_LOG_MAX_BYTES,
)
from .routes import make_routes
from .service import (
    RECOMMENDED_STOP_GRACE_S,
    SHUTDOWN_DEADLINE_S,
    UVICORN_GRACEFUL_S,
    ProxyService,
)

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


#: The envelope a straggler's caller gets when uvicorn cancels its handler at
#: the graceful-shutdown mark.
#:
#: 🚨 `draining` + 503 + the literal ``backpressure`` marker, which is the same
#: triple `lifecycle` already emits for work REFUSED while draining — and it has
#: to be, because the caller's situation is identical: this instance is going
#: away and the next one can serve you. §2.2 classifies deferrable on the marker
#: substring, so dropping the word would make a shutdown look like a hard
#: failure to every client that has not migrated to classifying on `code`.
_SHUTDOWN_CANCELLED_BODY = {
    "status": "error",
    "error": "proxy shutting down — request cancelled before completion, "
             "backpressure",
    "code": "draining",
}


class ShutdownEnvelopeMiddleware:
    """Turn uvicorn's shutdown cancellation into the proxy's own envelope.

    🚨 **The bug this closes** (`docs/ledger.md`): at
    ``timeout_graceful_shutdown`` uvicorn calls ``task.cancel()`` on every
    in-flight handler. ``asyncio.CancelledError`` derives from
    ``BaseException``, not ``Exception``, so it sails past BOTH
    ``exception_handlers`` below and Starlette's own ``ServerErrorMiddleware``,
    and the caller of a straggler received a raw
    ``500 Internal Server Error`` — plain text, no ``code``, no marker — for a
    condition that is retryable and entirely our doing. Measured at 48.77s by
    ``tools/sigterm_drain_probe.py`` scenario S3.

    It is a middleware rather than another entry in ``exception_handlers``
    because a handler there would never be reached: Starlette dispatches
    handlers from inside an ``except Exception`` block. User middleware sits
    *inside* ``ServerErrorMiddleware`` and *outside* the router, so it is the
    outermost place a ``BaseException`` from a route is still catchable.

    🚨 **The cancellation is always re-raised.** Swallowing it would break the
    asyncio contract and leave uvicorn waiting on a task that has decided not to
    die — turning a bounded shutdown into the unbounded one the whole shutdown
    budget exists to prevent. This adds a response; it does not decline to stop.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        started = False
        is_sse = False

        async def _send(message) -> None:
            nonlocal started, is_sse
            if message["type"] == "http.response.start":
                started = True
                for key, value in message.get("headers") or ():
                    if key.lower() == b"content-type":
                        is_sse = value.lower().startswith(b"text/event-stream")
            await send(message)

        try:
            await self.app(scope, receive, _send)
        except asyncio.CancelledError:
            # Best effort, and never at the cost of the cancellation: a failure
            # to deliver this must not mask why we are unwinding.
            try:
                await self._explain(send, started=started, is_sse=is_sse)
            except BaseException:      # noqa: BLE001 — see the comment above
                logger.debug("could not deliver the shutdown envelope",
                             exc_info=True)
            raise

    @staticmethod
    async def _explain(send, *, started: bool, is_sse: bool) -> None:
        body = json.dumps(_SHUTDOWN_CANCELLED_BODY).encode()
        if not started:
            await send({
                "type": "http.response.start", "status": 503,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode())],
            })
            await send({"type": "http.response.body", "body": body})
            return

        if is_sse:
            # 🚨 An error frame and deliberately NO `[DONE]`. `[DONE]` is the
            # backend asserting completeness, and this stream is truncated — the
            # proxy's own `finish_reason` repair exists precisely because those
            # two must never be collapsed. Emitting it here would tell the
            # caller a cancelled answer was a finished one.
            await send({"type": "http.response.body",
                        "body": b"data: " + body + b"\n\n",
                        "more_body": False})
            return

        # Headers are already on the wire and it is not a stream: there is no
        # frame to say this in. Close the body rather than leave the caller
        # waiting on a response that will never continue.
        await send({"type": "http.response.body", "body": b"",
                    "more_body": False})


#: The error envelope a caller sees when a request body exceeds
#: `ROADSTEAD_MAX_REQUEST_BYTES` (`config.max_request_bytes_from_env`).
#: `code` is `invalid_request_error` — §2.1 mints no new code for this (the
#: existing one is already classified non-deferrable, and the same body is
#: the same size on retry either way), it just gains a new status, `413`.
def _request_too_large_body(max_bytes: int) -> bytes:
    return json.dumps({
        "status": "error",
        "error": (f"request body exceeds the {max_bytes}-byte cap "
                  f"(ROADSTEAD_MAX_REQUEST_BYTES) — refused before it was "
                  f"parsed"),
        "code": "invalid_request_error",
    }).encode()


class RequestSizeLimitMiddleware:
    """Refuse an oversized request body BEFORE any handler calls
    ``request.json()`` — the ASGI layer, so nothing downstream even sees the
    bytes.

    🚨 **The gap this closes.** Every door on this proxy — the OpenAI-shaped
    doors, the enriched `/rs/v1/*` doors and the admin plane alike — calls
    ``await request.json()`` with nothing upstream bounding how large that
    body may be. `Content-Length` is honoured when a caller declares it
    honestly; a caller that lies short and then keeps streaming is caught by
    counting bytes as they actually arrive, which is also what covers a
    chunked-encoding body that declares no length at all.

    Sized at :data:`config.DEFAULT_MAX_REQUEST_BYTES` (16 MiB): a vision chat
    payload inlines its images as base64, and a caller sending a handful of
    them in one turn is the legitimate case this has to clear.

    Modelled on :class:`ShutdownEnvelopeMiddleware` just above: wrap ``send``
    so the refusal can be delivered exactly once even though the decision is
    made partway through the app's own read of the body, and swallow
    whatever the app tries to send afterwards rather than risk a second
    ``http.response.start`` on the same connection.
    """

    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        declared: int | None = None
        for key, value in scope.get("headers") or ():
            if key == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = None
                break
        if declared is not None and declared > self.max_bytes:
            await self._refuse(send)
            return

        refused = False

        async def _send(message) -> None:
            if refused:
                return  # the refusal already went out; nothing else may follow
            await send(message)

        seen = 0

        async def _receive():
            nonlocal seen, refused
            message = await receive()
            if message.get("type") == "http.request":
                seen += len(message.get("body") or b"")
                if seen > self.max_bytes and not refused:
                    refused = True
                    await self._refuse(send)
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        await self.app(scope, _receive, _send)

    async def _refuse(self, send) -> None:
        body = _request_too_large_body(self.max_bytes)
        await send({
            "type": "http.response.start", "status": 413,
            "headers": [(b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode())],
        })
        await send({"type": "http.response.body", "body": body})


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        # `prog` is pinned because there are two ways in — the console script
        # and `python -m roadstead` — and argparse would otherwise derive
        # `__main__.py` from argv[0] for the second, printing a usage line
        # nobody can type.
        prog="roadstead",
        # Kept word-for-word in step with `pyproject.toml`'s `description`, so
        # `--help` and the package index say the same thing about what this is.
        # The old text ("Centralized LLM scheduler proxy") described the origin
        # deployment's role, not the project.
        description="Capacity-aware admission control for self-hosted LLM "
                    "inference fleets (llama.cpp + vLLM), without Kubernetes.",
        epilog="These four flags are the whole command line; everything else "
               "this proxy does is configured by environment variable. The "
               "full reference — every ROADSTEAD_* variable, its default, and "
               "whether it is contract or internal — is docs/configuration.md: "
               "https://github.com/alexdkent/roadstead/blob/main/docs/"
               "configuration.md",
    )
    p.add_argument("--port", type=int, default=int(_env("PORT", "42161")))
    p.add_argument("--host", default=_env("HOST", "0.0.0.0"))
    p.add_argument("--log-level", default=_env("LOG_LEVEL", "info"))
    p.add_argument(
        "--data-dir", default=None,
        help="Directory for durable state — queue.db, runtime flags, the admin "
             "overlay and the request log all live under it. Overrides "
             f"ROADSTEAD_DATA_DIR; default {default_data_dir()}.")
    return p.parse_args()


#: Variables that were REMOVED rather than renamed, and what to do instead.
#: A rename leaves a name that means something else somewhere; a removal leaves
#: a name that means nothing at all, which is the quieter of the two failures —
#: nothing is wrong, the setting simply does not happen.
_REMOVED_VARS = {
    "ROADSTEAD_HOT_ROOT": (
        "it only composed the default data directory, which is now the XDG "
        "state directory. Set ROADSTEAD_DATA_DIR to the full path instead"),
}


def warn_on_retired_env_vars() -> list[str]:
    """Say something when a variable that no longer works is still set.

    🚨 Every environment variable was renamed `COLLECTIVE_* -> ROADSTEAD_*` on
    2026-08-31 (CHANGELOG: a recorded break). The dangerous half of that rename
    is not the flags that stop working — it is that they stop working IN
    SILENCE: an operator who had turned a correction layer OFF gets it back ON,
    a tuned keepalive reverts to the default, and nothing anywhere says so. The
    old names are NOT honoured, deliberately — two spellings for one switch is
    how they end up disagreeing — but an unread one is worth a loud line.

    The same argument covers a variable that was **removed** outright, and it is
    the reason `_REMOVED_VARS` exists rather than the name just disappearing
    from the source: a deployment carrying `ROADSTEAD_HOT_ROOT=/srv/state`
    across the 2026-09-05 upgrade would otherwise find its durable record in a
    new place with nothing to read that explains it.

    Returns the retired names found, so a test can prove this fires.
    """
    log = logging.getLogger(__name__)
    stale = sorted(k for k in os.environ if k.startswith("COLLECTIVE_"))
    for name in stale:
        log.warning(
            "IGNORED: %s is a retired variable name and has no effect. Rename it "
            "to %s.", name, "ROADSTEAD_" + name[len("COLLECTIVE_"):])

    removed = sorted(k for k in _REMOVED_VARS if k in os.environ)
    for name in removed:
        log.warning("IGNORED: %s was removed and has no effect — %s.",
                    name, _REMOVED_VARS[name])
    return stale + removed


#: Storage that a container recreate or a reboot throws away. Since 2026-09-05
#: no DEFAULT lands here — `default_data_dir` resolves to the XDG state dir — so
#: this fires for a path somebody chose, or for the no-home fallback.
#: `/private/tmp` is macOS's real `/tmp` (the latter is a symlink to it), and it
#: is kept because `TMPDIR`-style paths and hand-typed ones both land there.
_EPHEMERAL_PREFIXES = ("/tmp/", "/private/tmp/", "/var/tmp/",
                       "/private/var/tmp/", "/dev/shm/")


def _warn_if_durable_state_is_ephemeral(queue_db_path: str) -> bool:
    """🚨 The durable record on storage that does not survive a restart.

    This used to be the DEFAULT's disclosure, and the default moved on
    2026-09-05 (`default_data_dir`). The check stays, and it is not vestigial:
    the two remaining ways to land here are an operator who set
    `ROADSTEAD_DATA_DIR`/`ROADSTEAD_QUEUE_DB`/`--data-dir` to a temporary path
    on purpose, and the no-home fallback. Both are cases where the path was
    chosen rather than inherited, which is when the warning is most worth
    reading — and neither would be caught by anything else.

    What makes it worth an alarm is the gap between the care taken on one side
    and the storage on the other: SIGTERM runs a bounded drain specifically to
    persist DRR budgets and completion rows, the shutdown budget is computed and
    published so a container stop-grace of 108s does not truncate that flush,
    and `startup` replays the day's rows so a caller's spend survives a deploy.
    Measured on a real container 2026-09-02: `queue.db` sat on the ephemeral
    writable layer while the mounted volume held only the admin overlay, so
    every rebuild silently reset the DRR balances and the day's spend that the
    drain had carefully written.
    """
    if not queue_db_path.startswith(_EPHEMERAL_PREFIXES):
        return False
    logging.getLogger(__name__).warning(
        "durable state is on EPHEMERAL storage: queue.db is at %s. DRR budgets, "
        "today's spend and endpoint drain state are written here and are LOST on "
        "a reboot or a container recreate — which is exactly what the shutdown "
        "drain exists to preserve. Set ROADSTEAD_DATA_DIR (or ROADSTEAD_QUEUE_DB) "
        "to a persistent path; in a container, one that is a mounted volume.",
        queue_db_path)
    return True


def default_data_dir() -> str:
    """Where durable state goes when nobody says otherwise: the XDG state dir.

    ``$XDG_STATE_HOME/roadstead``, or ``~/.local/state/roadstead`` when that is
    unset — the XDG basedir spec's home for *state that should persist between
    restarts and is not a cache and not config*, which is this data verbatim.

    🚨 **This replaced ``/tmp/agents/llmproxy`` on 2026-09-05, and the move is
    the point.** The old default put the durable record on storage a reboot
    throws away, and the previous position was to DISCLOSE that rather than fix
    it, on the argument that moving the default would relocate an existing
    deployment's state on upgrade. That argument was answered rather than
    ignored: the shipped image sets ``ROADSTEAD_DATA_DIR`` explicitly and is
    unaffected, and for anyone who was relying on the old default, silently
    keeping a durable record somewhere a reboot deletes is not a state worth
    preserving. It is a recorded break (``CHANGELOG.md``), which is what
    ``docs/compatibility.md`` asks of one.

    Three things the old arrangement did, and where each one went:

    * *it was relocatable without naming a full path*, via
      ``ROADSTEAD_HOT_ROOT``. That variable is DELETED — it composed this path
      and did nothing else, and ``ROADSTEAD_DATA_DIR`` says the same thing
      directly.
    * *it disclosed itself as ephemeral at boot.* Unchanged:
      ``_warn_if_durable_state_is_ephemeral`` still fires, now only for an
      operator who points at ``/tmp`` deliberately, which is the case where the
      warning is worth reading.
    * *it was absolute without depending on the environment.* Preserved below,
      and it took explicit work — see the fallback.

    Every caller resolves the default HERE: the entry point, ``--data-dir``'s
    help text, and ``roadstead test``, which used to carry its own copy of the
    literal. That copy was not merely duplication — ``roadstead test replay``
    and ``ab`` read ``queue.db`` out of it, so a deployment that had moved its
    data dir (the documented, supported thing to do) had its own replay tooling
    look somewhere else and find an empty database, with nothing saying why.
    """
    xdg = os.environ.get("XDG_STATE_HOME", "").strip()
    if xdg:
        return os.path.join(xdg, "roadstead")

    home_state = os.path.expanduser("~/.local/state")
    if os.path.isabs(home_state):
        return os.path.join(home_state, "roadstead")

    # 🚨 ``expanduser`` returns its argument UNCHANGED when it cannot resolve
    # ``~`` — no HOME and no passwd entry for the uid, which is exactly the
    # distroless/``runAsUser: 10001`` container. Left alone, the result is a
    # RELATIVE path and the durable record lands under the working directory,
    # in a place nobody would look and nothing would say. Fall back to an
    # absolute path that the ephemeral check below will then complain about, so
    # the operator gets two loud lines instead of a silent wrong location.
    logging.getLogger(__name__).warning(
        "cannot resolve a home directory (no HOME, no passwd entry), so the "
        "XDG state directory is unavailable and durable state falls back to "
        "/tmp/roadstead. Set ROADSTEAD_DATA_DIR or pass --data-dir.")
    return "/tmp/roadstead"


def build_app(config: ProxyConfig | None = None) -> Starlette:
    """Build the Starlette app.  Usable from tests without running uvicorn."""
    warn_on_retired_env_vars()
    if config is None:
        data_dir = _env("DATA_DIR", default_data_dir())
        os.makedirs(data_dir, exist_ok=True)
        # 🚨 UNDER the data dir, not beside it in a separately-rooted log dir.
        # Changed 2026-09-02, and the reason is the deployment contract: the
        # operator is asked to provide ONE persistent path and the application
        # undertakes to keep everything it owns inside it. With the log rooted
        # somewhere else, `ROADSTEAD_DATA_DIR=/var/lib/roadstead` moved the
        # queue DB onto the mount and quietly left the request log on the
        # container's ephemeral layer — a half-kept promise, which is the
        # version of this that gets discovered late.
        log_dir = _env("LOG_DIR", os.path.join(data_dir, "logs"))
        os.makedirs(log_dir, exist_ok=True)

        config = ProxyConfig(
            queue_db_path=_env(
                "QUEUE_DB",
                os.path.join(data_dir, "queue.db"),
            ),
            # Runtime-mutable feature flags (flags.py) — persisted next to the
            # queue DB, flipped via POST /v1/admin/flags (no env gates).
            runtime_flags_path=_env(
                "RUNTIME_FLAGS",
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
            request_log_path=_env(
                "REQUEST_LOG",
                os.path.join(log_dir, "llmproxy_requests.jsonl"),
            ),
            # Size bound on that log. It grew forever until 2026-09-02 while
            # describing itself as the per-request record; rotation lives in the
            # application because the file is the application's, and because an
            # external rotation of a handle held open in append mode silently
            # writes to the unlinked inode.
            request_log_max_bytes=int(_env(
                "REQUEST_LOG_MAX_BYTES", DEFAULT_REQUEST_LOG_MAX_BYTES)),
            request_log_backups=int(_env(
                "REQUEST_LOG_BACKUPS", DEFAULT_REQUEST_LOG_BACKUPS)),
            # Per-agent DRR quota overrides (weight, max_balance_ss,
            # default_priority). Reads the shipped `roadstead/agents.yaml` by
            # default, or the path in ROADSTEAD_AGENTS_CONFIG.
            # Missing file → empty dict → proxy lazy-creates agent
            # configs at AgentQuotaConfig dataclass defaults.
            agents=load_agent_configs(),
            # queue.db maintenance knobs (persistence cleanup) — env overrides,
            # falling back to the ProxyConfig dataclass defaults.
            payload_retention_s=float(_env(
                "PAYLOAD_RETENTION_S", 48 * 3600.0)),
            completions_retention_s=float(_env(
                "COMPLETIONS_RETENTION_S", 30 * 86400.0)),
            wal_checkpoint_interval_s=float(_env(
                "WAL_CHECKPOINT_INTERVAL_S", 300.0)),
            incremental_vacuum_interval_s=float(_env(
                "INCR_VACUUM_INTERVAL_S", 600.0)),
            incremental_vacuum_pages=int(_env(
                "INCR_VACUUM_PAGES", 4000)),
            startup_vacuum_freelist_threshold_bytes=int(_env(
                "STARTUP_VACUUM_FREELIST_BYTES", 200 * 1024 * 1024)),
        )

    # Checked for BOTH branches — a caller that builds its own ProxyConfig can
    # put the durable log on /tmp just as easily, and it is the same loss.
    _warn_if_durable_state_is_ephemeral(config.queue_db_path)

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
        # 🚨 Outside the router, inside `ServerErrorMiddleware` — the only place
        # a `BaseException` from a route is still catchable. See the class.
        # Outermost first: an oversized body is refused before anything else
        # runs, including the shutdown-envelope rewrite below (which only
        # ever engages on a cancellation, so the ordering is not otherwise
        # load-bearing — this is just where "refuse before you do more work"
        # belongs).
        middleware=[
            Middleware(RequestSizeLimitMiddleware,
                      max_bytes=max_request_bytes_from_env()),
            Middleware(ShutdownEnvelopeMiddleware),
        ],
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

    # 🚨 Published into the environment rather than passed down, and the reason
    # is that the data dir is a ROOT: the queue DB, the runtime flags, the admin
    # overlay and the request log are each derived from it by their OWN
    # `ROADSTEAD_*` variable further down, and each of those must still be able
    # to override the flag individually. Setting the variable puts the flag
    # exactly where the existing precedence already works — flag beats env,
    # env beats default — instead of adding a fourth rule.
    if args.data_dir is not None:
        os.environ["ROADSTEAD_DATA_DIR"] = args.data_dir

    app = build_app()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=False,
        # 🚨 This bounds the in-flight HTTP CONNECTIONS and nothing else. It does
        # NOT bound `ProxyService.shutdown` — uvicorn hands over to the lifespan
        # shutdown once this expires and never bounds that at all, so the worst
        # case is the SUM of the two (docs/ledger.md). The old value was derived
        # as `_DRAIN_DEADLINE_S + 18` under the opposite assumption; it is its
        # own knob now, for its own job, and `service.py` owns the arithmetic
        # that turns both into the operator's stop-grace.
        timeout_graceful_shutdown=int(UVICORN_GRACEFUL_S),
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
    #
    # ROADSTEAD_DATA_DIR is honoured here, which it was not before: this line
    # rebuilt the default path itself and so ignored the one variable an
    # operator sets to move the database it is trying to read.
    db_path = getattr(args, "db", None) or os.path.join(
        _env("DATA_DIR", default_data_dir()), "queue.db")

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
