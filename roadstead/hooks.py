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
# Security events
# ---------------------------------------------------------------------------
#
# Vendored from ``originfleet/framework/prompt_security.py`` (2026-08-31). The
# proxy's only call site passed ``store=None``, so the persistence half of the
# framework helper was already dead here and the log line below is byte-for-byte
# what it emitted. The event name ``llmproxy_cache_drift`` is documented in
# ``config.py`` and ``docs/llmproxy_prefix_cache_observability.md`` — preserve
# the shape if you touch this.

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
