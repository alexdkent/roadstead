"""In-process SSE fan-out hub for the LLM proxy.

The proxy is the single front door for all LLM traffic, so it is the
authoritative real-time source of call/usage telemetry. Producers (the
completion-recording hot path, the periodic poller) call
``hub.publish(event, data)`` — a SYNCHRONOUS fan-out (no awaits inside)
so the dispatch hot path emits without scheduling a task. Each connected
``GET /v1/stream`` handler holds its own bounded ``asyncio.Queue``
subscribed to the hub; a slow consumer is dropped (queue full → a
``__drop__`` sentinel) so a stuck client can never stall producers. The
browser reconnects on ``EventSource.onerror`` and re-syncs via the REST
snapshot endpoints.

This mirrors the proven host-telemetry ``telemetry_core.sse_hub`` pattern
(same drop-slow-client semantics) but lives in the proxy package — the
two run in different deployment trees (container vs inference hosts) and
must not import across that boundary.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

DROP_SENTINEL: tuple[str, str] = ("__drop__", "")


class SSEHub:
    """Bounded-queue SSE fan-out. All methods are safe to call from the
    event-loop thread; ``publish`` is synchronous so the completion hot
    path emits inline."""

    def __init__(self, *, queue_maxsize: int = 256) -> None:
        self._clients: set[asyncio.Queue] = set()
        self._maxsize = queue_maxsize

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        self._clients.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._clients.discard(q)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def publish(self, event: str, data: Any) -> None:
        """Fan ``(event, data)`` out to every subscriber. No-op when no
        clients are connected (the common case — costs one set check on
        the hot path). Drops a slow client rather than blocking."""
        if not self._clients:
            return
        try:
            payload = (event, json.dumps(data, default=_json_default, separators=(",", ":")))
        except (TypeError, ValueError) as exc:
            logger.warning("sse_hub: payload serialize failed for %s: %s", event, exc)
            return
        dead: list[asyncio.Queue] = []
        for q in self._clients:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            # Make room for the sentinel by dropping the oldest queued event;
            # the handler is blocked on q.get() and will receive the sentinel
            # next, close its response, and the browser re-syncs + reconnects.
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                q.put_nowait(DROP_SENTINEL)
            except asyncio.QueueFull:
                pass

    def close_all(self) -> None:
        """Signal every subscriber to drop (graceful shutdown)."""
        for q in list(self._clients):
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                q.put_nowait(DROP_SENTINEL)
            except asyncio.QueueFull:
                pass


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not json-serializable: {type(obj).__name__}")
