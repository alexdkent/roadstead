"""Reasoning replay — re-attaching a prior turn's own reasoning to history.

THE DEFECT THIS EXISTS FOR (measured on a live fleet, 2026-09-25). A hybrid
"always-thinking" reasoner's chat template renders a past assistant turn one
way when the message carries its own reasoning (``<think>{reasoning}</think>
{content}``) and a DIFFERENT way when it does not (``<think></think>
{content}``). The engine's prefix cache holds the tokens it actually
generated — WITH the reasoning — so a caller that replays history with plain
``{"role": "assistant", "content": ...}`` (which is what ``message.content``
alone gives you; most chat loops rebuild history from exactly that) diverges
from the cached prefix at the FIRST assistant turn, and every later turn in
that conversation is re-prefilled from scratch. Measured turn-2 prefix hit:
95.9% with plain content vs 99.9% with the reasoning re-attached (as either
``reasoning_content`` or ``reasoning`` — the engine accepts both on input).

Roadstead already RETURNS the reasoning to a caller (``message.reasoning`` /
``delta.reasoning``), so the information exists — it is just usually thrown
away one hop later, by a caller that keeps only ``content``. This module lets
an opted-in endpoint (``policy.replay_reasoning_history``, see
``model_catalog._POLICY_PASSTHROUGH``) remember what it generated and put it
back if a caller's next call replays the same turn without it, closing the
gap without asking every caller in the fleet to change its history-rebuilding
code.

THE KEY, and why it is not just a hash of the assistant turn. Two identical
short replies ("OK") in two different conversations must never collide — the
reasoning that produced one is not necessarily right for the other, and
serving it would be silently wrong rather than absent. So the key covers the
FULL conversation PREFIX that led to the turn (every message strictly before
it), not just the turn itself; see :func:`replay_key` and
``tests/test_reasoning_replay.py::test_cross_conversation_isolation``.

Every message in that prefix is NORMALIZED first — ``reasoning`` /
``reasoning_content`` stripped off each one — so a request this proxy has
already enriched (via :func:`restore`, on a LATER turn of the same
conversation) hashes identically to the client's own un-enriched bytes. Without
that, restoring turn 1 would change the key turn 2 is stored under, and the
store would never see two requests agree with each other again.

THE STORE is bounded three ways at once (entry count, total bytes, TTL) and
evicts LRU, matching the shape of :class:`roadstead.coalesce.DeterministicCache`
next door. It lives in a single :class:`ProxyState` instance, on the single
event loop, with no lock — the same concurrency invariant every other
in-memory scheduler/cache object in this repo relies on
(``docs/internals.md`` "The concurrency invariant"). A Roadstead restart
simply empties it: the worst case is a cold cache (today's behaviour,
everywhere), never a wrong answer, so nothing here is persisted across a
restart on purpose.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# --- bounds -----------------------------------------------------------------
# These are MEASUREMENTS-SHAPED defaults, not measurements — unlike the
# fleet-hardware numbers in docs/internals.md § 2 (goodput thresholds, timeout
# floors), there is no backend to profile here: the cost is proxy-process
# heap, not GPU capacity, and it scales with HOW MANY conversations are live
# at once, which is a property of the deployment, not the model. So these are
# sized generously (round numbers, not tuned) and are trivially overridable
# per deployment via the ``ReasoningReplayStore`` constructor — a deployment
# that opts many high-traffic endpoints in should size them from its own
# concurrent-conversation count rather than trust the default.
#
# 50k entries * ~avg a few KB of reasoning each is comfortably under the byte
# cap below, so in practice the byte cap is the one that bites first on a
# verbose reasoner; the entry cap exists for a workload of many SHORT
# reasoning turns (a chat lane) where byte pressure never gets there.
DEFAULT_MAX_ENTRIES = 50_000
DEFAULT_MAX_BYTES = 256 * 1024 * 1024  # 256 MiB of reasoning text
# 24h covers "picked the conversation back up tomorrow" without keeping a
# reasoning blob alive indefinitely for a conversation nobody is coming back
# to — a stale hit is never WRONG (the key still requires the same prefix),
# just wasted memory, so the TTL is a housekeeping bound, not a correctness one.
DEFAULT_TTL_S = 24 * 3600.0


def _normalize_message(m: Any) -> Any:
    """Strip ``reasoning``/``reasoning_content`` from one message.

    This is what makes a request THIS PROXY already enriched (via
    :func:`restore`) hash identically to the client's own original bytes —
    without it, restoring turn 1 of a conversation would change the key turn
    2 is stored under, and a store/restore pair could never agree with
    itself past the first turn."""
    if not isinstance(m, dict):
        return m
    return {k: v for k, v in m.items() if k not in ("reasoning", "reasoning_content")}


def _canonical(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        # Same "uncacheable beats a wrong key" reasoning as
        # DeterministicCache.cache_key — repr() at least varies with content
        # instead of colliding every unhashable payload onto one key.
        return repr(obj)


def replay_key(messages_before: Any, system: Any, content: Any,
                tool_calls: Any) -> str:
    """The stable key for one assistant turn: its full conversation PREFIX
    (normalized, see :func:`_normalize_message`) plus a separate ``system``
    field when the payload carries one (Anthropic-shaped callers), plus a
    deterministic serialization of what the assistant actually produced
    (``content`` and ``tool_calls``).

    The prefix is what gives two identical short replies in two different
    conversations two different keys — see the module docstring."""
    norm_messages = (
        [_normalize_message(m) for m in messages_before]
        if isinstance(messages_before, list) else [])
    keyed: dict[str, Any] = {
        "messages": norm_messages,
        "content": content if isinstance(content, str) else "",
        "tool_calls": tool_calls or None,
    }
    if system is not None:
        keyed["system"] = system
    return hashlib.sha256(_canonical(keyed).encode()).hexdigest()


def accumulate_tool_call_deltas(acc: dict[int, dict], deltas: Any) -> None:
    """Fold one streaming chunk's ``delta.tool_calls`` fragments into ``acc``,
    keyed by the OpenAI streaming ``index`` field. ``name``/``arguments`` are
    strings split across many deltas and must be CONCATENATED, not replaced —
    the same reassembly every OpenAI-compatible streaming client does on its
    own side; done here so a store/restore pair can key streaming and
    non-streaming turns the same way."""
    if not isinstance(deltas, list):
        return
    for d in deltas:
        if not isinstance(d, dict):
            continue
        idx = d.get("index", 0)
        if not isinstance(idx, int):
            idx = 0
        entry = acc.setdefault(idx, {
            "id": "", "type": "function",
            "function": {"name": "", "arguments": ""},
        })
        if d.get("id"):
            entry["id"] = d["id"]
        if d.get("type"):
            entry["type"] = d["type"]
        fn = d.get("function")
        if isinstance(fn, dict):
            if isinstance(fn.get("name"), str):
                entry["function"]["name"] += fn["name"]
            if isinstance(fn.get("arguments"), str):
                entry["function"]["arguments"] += fn["arguments"]


@dataclass
class _Entry:
    reasoning: str
    expires_at: float
    size: int


class ReasoningReplayStore:
    """Bounded, TTL'd LRU cache: replay key -> the reasoning that produced it.

    Single-event-loop object, no lock — see the module docstring's
    concurrency note. Shape mirrors
    :class:`roadstead.coalesce.DeterministicCache`: an ``OrderedDict`` for
    LRU, eviction from the front. The one addition is a running BYTE total,
    because unlike a response cache (bounded implicitly by the caller's own
    ``max_tokens``) reasoning length varies enormously by endpoint, and a
    count-only bound lets one verbose reasoner's entries dominate memory."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES,
                 max_bytes: int = DEFAULT_MAX_BYTES,
                 ttl_s: float = DEFAULT_TTL_S) -> None:
        self._cache: OrderedDict[str, _Entry] = OrderedDict()
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._ttl_s = ttl_s
        self._total_bytes = 0

    def put(self, key: str, reasoning: str) -> None:
        if not reasoning:
            return
        size = len(reasoning.encode("utf-8", "replace"))
        if size > self._max_bytes:
            # One entry that alone exceeds the whole budget can never be
            # stored without starving everything else — same "uncacheable
            # beats a wrong entry" call DeterministicCache makes for a
            # payload it can't hash; here it's a payload too big to keep.
            logger.debug(
                "reasoning_replay: entry (%d bytes) exceeds max_bytes (%d) — "
                "not stored", size, self._max_bytes)
            return
        existing = self._cache.pop(key, None)
        if existing is not None:
            self._total_bytes -= existing.size
        self._cache[key] = _Entry(reasoning, time.monotonic() + self._ttl_s, size)
        self._total_bytes += size
        self._evict()

    def get(self, key: str) -> str | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        if time.monotonic() > entry.expires_at:
            del self._cache[key]
            self._total_bytes -= entry.size
            return None
        self._cache.move_to_end(key)
        return entry.reasoning

    def _evict(self) -> None:
        while self._cache and (
            len(self._cache) > self._max_entries
            or self._total_bytes > self._max_bytes
        ):
            _, victim = self._cache.popitem(last=False)
            self._total_bytes -= victim.size

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def stats(self) -> dict:
        return {"entries": self.size, "bytes": self.total_bytes}
