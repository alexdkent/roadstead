"""Embedding coalescing and deterministic response caching.

Embedding coalescing: identical embed requests arriving concurrently
share one backend call.

Deterministic cache: for temperature=0 requests with identical payloads,
cache the response for a short TTL.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding coalescer
# ---------------------------------------------------------------------------

class EmbedCoalescer:
    """Deduplicates concurrent identical embedding requests.

    If request B arrives while request A (with the same text) is
    in-flight, B awaits A's result instead of making a second backend
    call.
    """

    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Future] = {}
        self._saved_calls: int = 0

    @staticmethod
    def _key(endpoint: str, text: str) -> str:
        return hashlib.sha256(f"{endpoint}:{text}".encode()).hexdigest()

    def check(self, endpoint: str, text: str) -> asyncio.Future | None:
        """If an identical request is in-flight, return its future.
        The caller should await it instead of making a new backend call."""
        key = self._key(endpoint, text)
        fut = self._pending.get(key)
        if fut is not None and not fut.done():
            self._saved_calls += 1
            return fut
        return None

    def register(self, endpoint: str, text: str) -> tuple[str, asyncio.Future]:
        """Register a new in-flight embedding request.  Returns (key, future).
        The caller must set_result or set_exception on the future when done."""
        key = self._key(endpoint, text)
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[key] = fut
        return key, fut

    def resolve(self, key: str, result: dict) -> None:
        fut = self._pending.pop(key, None)
        if fut and not fut.done():
            fut.set_result(result)

    def reject(self, key: str, exc: Exception) -> None:
        fut = self._pending.pop(key, None)
        if fut and not fut.done():
            fut.set_exception(exc)

    @property
    def saved_calls(self) -> int:
        return self._saved_calls

    @property
    def active_pending(self) -> int:
        return sum(1 for f in self._pending.values() if not f.done())


# ---------------------------------------------------------------------------
# Deterministic response cache
# ---------------------------------------------------------------------------

class DeterministicCache:
    """LRU cache for deterministic LLM responses (temperature=0)."""

    def __init__(self, max_size: int = 256, ttl_s: float = 300.0) -> None:
        self._cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._max_size = max_size
        self._ttl_s = ttl_s
        self._hits: int = 0
        self._misses: int = 0

    def cache_key(self, endpoint: str, payload: dict) -> str | None:
        """Return a cache key if this request is cacheable, else None."""
        temp = payload.get("temperature")
        if temp is None or temp != 0:
            return None
        if payload.get("stream"):
            return None

        # NOTE: cache_key runs on the pre-normalization payload (before
        # backend._normalize_chat_payload inlines `system` into messages),
        # so a top-level `system` field is NOT reflected in `messages`
        # here. It MUST be in the key independently — otherwise two
        # temperature=0 requests with identical messages but different
        # system prompts (e.g. different extraction schemas) collide and
        # the second gets the first's response. That's a correctness/
        # cache-poisoning bug for structured extraction.
        canonical = json.dumps(
            {
                "endpoint": endpoint,
                "system": payload.get("system"),
                "messages": payload.get("messages"),
                "grammar": payload.get("grammar") or payload.get("extra_body", {}).get("grammar"),
                "max_tokens": payload.get("max_tokens"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        expires_at, response = entry
        if time.monotonic() > expires_at:
            del self._cache[key]
            self._misses += 1
            return None
        self._cache.move_to_end(key)
        self._hits += 1
        return response

    def put(self, key: str, response: dict) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (time.monotonic() + self._ttl_s, response)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    @property
    def hit_rate_pct(self) -> float:
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return (self._hits / total) * 100

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def evictions_approx(self) -> int:
        return max(0, self._hits + self._misses - self._max_size)

    def stats(self) -> dict:
        return {
            "hit_rate_pct": round(self.hit_rate_pct, 1),
            "entries": self.size,
            "hits": self._hits,
            "misses": self._misses,
        }
