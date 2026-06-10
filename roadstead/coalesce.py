"""Deterministic response caching.

For temperature=0 requests with identical payloads, cache the response for a
short TTL.

(An ``EmbedCoalescer`` lived here to dedupe concurrent identical embed requests,
but it was never wired into the dispatch path — its check/register/resolve were
never called and it only reported an always-zero ``saved_calls`` on /v1/status.
Removed in Phase 3 rather than wiring a new single-flight concurrency surface
(hung-waiter risk) for a throughput optimization the deterministic cache already
covers after the first call.)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)


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
        """Return a cache key if this request is cacheable, else None.

        The key hashes the ENTIRE canonical payload, not an enumerated field
        subset. The old enumeration (system/messages/grammar/max_tokens) had
        poisoning blind spots: two temperature=0 requests with identical
        messages but different ``response_format`` / ``structured_outputs`` /
        ``tools`` (or any future field) collided, and the second caller got
        the first's response. "Identical payload" is the docstring promise —
        hashing the whole thing makes it structurally true and can only
        REDUCE spurious hits, never add them."""
        temp = payload.get("temperature")
        if temp is None or temp != 0:
            return None
        if payload.get("stream"):
            return None
        try:
            canonical = json.dumps(
                {"endpoint": endpoint, "payload": payload},
                sort_keys=True,
                separators=(",", ":"),
                default=str,  # payload arrived as JSON, so this never fires in practice
            )
        except (TypeError, ValueError):
            return None  # uncacheable beats a wrong key
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

    def stats(self) -> dict:
        return {
            "hit_rate_pct": round(self.hit_rate_pct, 1),
            "entries": self.size,
            "hits": self._hits,
            "misses": self._misses,
        }
