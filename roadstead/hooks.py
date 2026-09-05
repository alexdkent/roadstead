"""Integration seam — the one place the proxy reports OUT to a host application.

Every module in this package emits degradations and security events through the
callables here, and each one has a working stdlib-only default. A host
application (originfleet) may replace a sink at startup via ``set_*_sink``; a
standalone deployment simply doesn't, and the defaults keep the proxy fully
functional with no loss of information — only of integration.

Why this file exists: before 2026-08-31 four modules imported
``originfleet.framework.*`` directly, which meant the package could not be run
or published without the whole monorepo. Those imports now live here and in
``__main__.py`` (the entrypoint, which a standalone deployment replaces
wholesale), so the other 29 modules import nothing but the standard library and
this package's declared third-party dependencies.

🚨 Keep this module dependency-free. An import of anything outside the stdlib
re-creates the coupling it exists to remove.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Degradation reporting
# ---------------------------------------------------------------------------

class DegradationSink(Protocol):
    """The shape ``originfleet.framework.observability.degradation`` already
    has, so the host can register that function directly."""

    def __call__(
        self, *, component: str, reason: str, impact: str, **fields: Any,
    ) -> None: ...


_degradation_sink: DegradationSink | None = None


def set_degradation_sink(sink: DegradationSink | None) -> None:
    """Register the host's degradation reporter (``None`` restores the default).

    Called once at startup from ``__main__``. Not thread-safe by design — the
    proxy wires this before the event loop starts and never mutates it again.
    """
    global _degradation_sink
    _degradation_sink = sink


def degradation(
    *, component: str, reason: str, impact: str, **fields: Any,
) -> None:
    """Record a degradation: the system is doing the right thing under the
    circumstances, but the caller's experience is worse than nominal.

    Falls back to a WARNING log in the same grep-able shape the framework sink
    uses, so an operator sees the same line either way. Never raises — a
    reporting seam must not be able to break a response.
    """
    if _degradation_sink is not None:
        try:
            _degradation_sink(
                component=component, reason=reason, impact=impact, **fields)
            return
        except Exception:  # noqa: BLE001 — the seam must never break a response
            logger.debug("degradation sink failed; falling back to log",
                         exc_info=True)
    logger.warning(
        "DEGRADATION component=%s reason=%s impact=%s fields=%r",
        component, reason, impact, fields,
    )


# ---------------------------------------------------------------------------
# Configuration notices — "you wrote this and it has no effect"
# ---------------------------------------------------------------------------
#
# Three loaders in this package accept a fixed set of keys and drop everything
# else: ``model_catalog._POLICY_PASSTHROUGH``, ``config.load_agent_configs`` and
# ``identity.KeyRegistry._FILE_FIELDS``. Each of them has silently ignored a
# real knob at least once, and the reason that failure is so expensive is that
# from outside it is INDISTINGUISHABLE from the policy decision the operator was
# trying to make: a dropped ``spill_ok`` looks exactly like a caller who never
# opted in.
#
# A WARNING at startup is not enough, because the operator reading it is usually
# not the operator who wrote the file, and the line has scrolled by the time
# anybody asks why the knob does nothing. So a notice is RETAINED as well as
# logged, and the management plane reads it back (``docs/api.md`` §3.4) — which
# is what makes "what did you write that is not in force?" an answerable
# question rather than a log-grep.
#
# 🚨 Emitted at LOAD time, from the thread doing the loading — startup, or an
# operator edit on the loop thread. Nothing on the hot path appends here, which
# is why a plain deque with no lock is the right amount of machinery.

class ConfigNoticeSink(Protocol):
    """A host application's sink for configuration notices."""

    def __call__(
        self, *, source: str, subject: str, problem: str, detail: str,
        **fields: Any,
    ) -> None: ...


#: Bounded so a pathological config (or a test suite loading a catalog a few
#: thousand times) cannot grow this without limit. Oldest notices are dropped
#: first: a config problem that is still true is re-reported on the next load,
#: so the recent end is the useful one.
_MAX_CONFIG_NOTICES = 256

_config_notices: "deque[dict[str, Any]]" = deque(maxlen=_MAX_CONFIG_NOTICES)
_config_notice_sink: ConfigNoticeSink | None = None


def set_config_notice_sink(sink: ConfigNoticeSink | None) -> None:
    """Register the host's config-notice reporter (``None`` restores default)."""
    global _config_notice_sink
    _config_notice_sink = sink


def config_notice(
    *, source: str, subject: str, problem: str, detail: str, **fields: Any,
) -> None:
    """Record something an operator WROTE that is not in force.

    ``source`` is the file or environment variable it was written in, ``subject``
    the stanza within it, ``problem`` a stable slug (``unknown_key``,
    ``unparseable``, ``duplicate``…) and ``detail`` the human sentence. Never
    raises: a reporting seam must not be able to stop a config from loading.
    """
    record = {
        "source": source,
        "subject": subject,
        "problem": problem,
        "detail": detail,
        **fields,
    }
    _config_notices.append(record)
    if _config_notice_sink is not None:
        try:
            _config_notice_sink(
                source=source, subject=subject, problem=problem,
                detail=detail, **fields)
        except Exception:  # noqa: BLE001 — the seam must never break a load
            logger.debug("config notice sink failed; falling back to log",
                         exc_info=True)
    logger.warning("CONFIG NOTICE source=%s subject=%s problem=%s detail=%s",
                   source, subject, problem, detail)


def config_notices() -> list[dict[str, Any]]:
    """Every retained notice, oldest first. Read by the management plane."""
    return list(_config_notices)


def clear_config_notices() -> None:
    """Drop the retained notices. For tests, and for a deliberate reload."""
    _config_notices.clear()


# ---------------------------------------------------------------------------
# Security events
# ---------------------------------------------------------------------------
#
# Vendored from ``originfleet/framework/prompt_security.py`` (2026-08-31). The
# proxy's only call site passed ``store=None``, so the persistence half of the
# framework helper was already dead here and the log line below is byte-for-byte
# what it emitted. The event name ``roadstead_cache_drift`` is documented in
# ``config.py`` (the origin's prefix-cache observability note is not part of
# this repository) — preserve the shape if you touch this.

def compact_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """JSON-safe, length-capped copy of ``metadata``.

    Strings clipped at 200 chars, lists at 20 elements (each string in the list
    clipped at 120 chars), nested dicts recursed — so a log line stays a line.
    """
    out: dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, str):
            out[key] = value[:200]
        elif isinstance(value, (int, float, bool)):
            out[key] = value
        elif isinstance(value, (list, tuple)):
            out[key] = [
                item[:120] if isinstance(item, str) else item
                for item in value[:20]
            ]
        elif isinstance(value, dict):
            out[key] = compact_metadata(value)
        else:
            out[key] = str(value)[:200]
    return out


def record_security_event(
    log: logging.Logger,
    *,
    event_type: str,
    severity: str = "info",
    provider: str | None = None,
    message_id: str | None = None,
    source: str | None = None,
    action: str | None = None,
    blocked: bool = False,
    reason: str | None = None,
    metadata: dict[str, Any] | None = None,
    log_level: int | None = None,
) -> None:
    """Emit a structured security-event log line. Never raises.

    ``severity`` is one of ``info|warning|high``; warning/high default to
    WARNING log level.
    """
    safe_metadata = compact_metadata(metadata)
    if log_level is None:
        log_level = logging.WARNING if severity in {"warning", "high"} else logging.INFO
    log.log(
        log_level,
        "%s severity=%s provider=%s msg=%s source=%s action=%s blocked=%s "
        "reason=%s metadata=%s",
        event_type, severity, provider, (message_id or "")[:40], source,
        action, blocked, reason, json.dumps(safe_metadata, sort_keys=True),
    )
