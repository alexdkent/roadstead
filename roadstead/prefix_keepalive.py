"""Prefix keep-alive — touching a call site's declared prefix before the
engine's prefix-cache LRU evicts it.

THE DEFECT THIS EXISTS FOR (measured 2026-09-26 on the tier3 endpoint: vLLM,
hybrid GLM-5.3-Flash). An interactive agent (call_site ``hermes.
openai_compat``) sends a ~36-40K-token prefix (tools + system prompt) on
EVERY turn. The engine's prefix cache is a small LRU pool: after ~300K
tokens of OTHER uncached prefill the prefix is evicted, and the next new
session re-prefills it cold (~20-25s vs ~1s warm). Real fleet traffic runs
190K-1.37M uncached prefill tokens/hour, so between real turns the prefix
naturally survives only 13-50 minutes.

WHY A NEW-SESSION-SHAPED TOUCH, AND WHY THE NONCE IS THE LOAD-BEARING
DETAIL. Touching the prefix with a byte-IDENTICAL request every ~150K tokens
was tried first: every such touch was a full cache hit, but a genuinely NEW
session (a different user message) still missed after 600K tokens of filler.
The hybrid model's recurrent state is checkpointed at the point a request
DIVERGES from what came before, and an identical touch only ever refreshes
the checkpoint at the END of ITS OWN prompt — never the system/user JUNCTION
checkpoint a new session actually resumes from. Touching with a UNIQUE user
suffix each time (``keepalive <nonce>``) fixed it: a new session dispatched
after 606K tokens of filler hit 39,542/39,550 cached tokens, 1.5s. The first
such touch, fired at ~150K, missed once — which is why a deployment should
size its own ``prefix_keepalive_trigger_tokens`` to fire earlier than
whatever it measures its own eviction point to be (~100K was the number that
worked on the fleet that measured this), and the whole design tolerates an
occasional miss rather than trying to prevent one: a miss just re-seeds the
prefix from whatever request happens to trigger it next.

🚨 **NOT ARMED ON THE SHIPPED EXAMPLE CATALOG, DELIBERATELY.** Unlike most
``policy.*`` knobs in ``models.yaml``, this one is not given a live worked
example on ``tier3`` — every one of this endpoint's four keys is a
per-deployment MEASUREMENT of a real workload's own call_site names and its
own eviction cadence, and arming it here would make the whole test suite's
background poller start dispatching real synthetic touches against the fake
backend on every run that happens to declare a matching call_site, which is
a behaviour change no test in this repository asked for. Absent (the shipped
default) is the same "byte-identical to before this module existed" contract
every other undeclared endpoint gets.

WHAT THIS MODULE OWNS, AND WHAT IT DOES NOT. This module holds the STATE and
every PURE decision — deriving a prefix skeleton from a real request's
payload, keying it, accounting uncached tokens against every tracked
prefix's countdown, and deciding which prefixes are due. It has NO I/O of
its own, matching the shape of ``reasoning_replay.py`` next door (bounded
in-memory store, single event loop, no lock — see docs/internals.md "The
concurrency invariant"). The network call that actually touches a backend,
and the health/capacity gating around firing one, live in ``health.py``
beside the thinking canary this is modelled on
(``Health._schedule_thinking_canary`` / ``_maybe_run_thinking_canary``) —
see there for why a proxy-internal generation is dispatched OUTSIDE the
DRR/admission path entirely rather than as a BACKGROUND-band
``QueuedRequest``.

Absent ``policy.prefix_keepalive_call_sites`` (empty tuple, the default) this
is completely inert: no capture, no accounting, no touches, zero cost —
byte-identical to before this module existed. See ``EndpointConfig``'s
``prefix_keepalive_*`` fields in ``config.py`` for the full policy shape and
``model_catalog._POLICY_PASSTHROUGH`` for how it reaches an endpoint from
``models.yaml``.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import secrets
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: ``max_tokens`` on a touch. Small on purpose — the touch's only job is to
#: walk the backend through prefill on a byte-identical prefix (refreshing
#: the KV/recurrent-state checkpoint a later real session resumes from); it
#: is never meant to be READ. Not 1: a forced-reasoning or always-thinking
#: endpoint spends its first output tokens on the CoT, and a 1-token cap
#: would return an empty completion every single time — fine for the
#: prefill this exists to cause, but it leaves nothing to compute a hit/miss
#: verdict from. A handful of tokens gives a thinking-off endpoint enough
#: room for one real (discarded) word, and a thinking-on endpoint a chance
#: of a few reasoning tokens landing before the cap.
TOUCH_MAX_TOKENS = 4

#: A touch counts as a HIT when at least this fraction of its OWN prompt
#: tokens came back cached. Deliberately not 100%: the touch's trailing user
#: turn (``keepalive <nonce>``) is unique every time — see the module
#: docstring for why — so it is NEVER itself in the cache, and a handful of
#: tokens of genuine miss on top of a perfect prefix hit is expected, not a
#: sign the touch failed.
HIT_RATIO = 0.90

#: Prefix for the synthetic trailing user turn every touch sends. The nonce
#: is what makes it unique; this text is only ever discarded on read.
TOUCH_USER_PREFIX = "keepalive"


def _leading_system_messages(messages: Any) -> list:
    """Every message from the start of ``messages`` up to (not including) the
    first non-``system`` one — the "leading system prefix" this whole module
    exists to keep warm. A ``tool``/``user``/``assistant`` message ANYWHERE
    before the run ends it; only a contiguous run from index 0 counts."""
    out: list = []
    if not isinstance(messages, list):
        return out
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            out.append(m)
        else:
            break
    return out


def derive_skeleton(payload: dict) -> dict | None:
    """The prefix SKELETON a real request rendered: its ``tools``, its
    leading ``system`` message(s), and the chat-template variables that
    change the rendered prefix bytes (``model``, ``chat_template_kwargs`` —
    which by capture time carries whatever ``reasoning_effort``/thinking
    switch ``Correction`` folded in for this call).

    🚨 CAPTURED FROM ``req.payload`` AFTER CORRECTION, BEFORE ENGINE
    NORMALIZATION — and that is what makes a touch built from it render the
    SAME prefix bytes without redoing any of Correction's own work.
    ``Lifecycle.handle_submit`` runs the full correction chain
    (``apply_forced_reasoning_budget`` / ``apply_json_object_guard`` /
    ``apply_thinking`` / ``apply_reasoning_effort_map``) against ``req.
    payload`` IN PLACE, before dispatch; by the time a request completes (the
    capture point — see ``PrefixKeepaliveTracker.observe_completion``) that
    payload holds exactly what those corrections produced. ``backend.call()``
    then hands it to ``Provider.prepare_chat_payload`` for the PER-ENGINE
    normalization (the served ``model`` id, a launch-flag-gated reasoning
    budget) — and that function never mutates its input (see
    ``providers/llamacpp.py``: ``p = dict(payload)``), so ``req.payload``
    itself never carries the engine-level additions. A touch is therefore
    built from precisely the state a real request's payload was in the ONE
    time it was ever handed to ``prepare_chat_payload``, and passing it
    through that SAME function again (see ``backend.probe_prefix_touch``)
    applies the identical one-time normalization a fresh real request would
    also get. Returns None when there is nothing distinctive to keep warm."""
    if not isinstance(payload, dict):
        return None
    system_messages = _leading_system_messages(payload.get("messages"))
    tools = payload.get("tools")
    ck = payload.get("chat_template_kwargs")
    if not system_messages and not (isinstance(tools, list) and tools):
        # A bare user turn with no system prompt and no tools has no prefix
        # beyond the model's own template boilerplate, which the engine
        # never evicts on its own — nothing here is worth keeping warm.
        return None
    return {
        "system_messages": system_messages,
        "tools": tools if isinstance(tools, list) and tools else None,
        "model": payload.get("model") or None,
        "chat_template_kwargs": ck if isinstance(ck, dict) and ck else None,
    }


def _canonical(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        # Same "uncacheable beats a wrong key" reasoning as
        # DeterministicCache.cache_key / reasoning_replay._canonical.
        return repr(obj)


def skeleton_key(skeleton: dict) -> str:
    """Stable identity for a prefix skeleton. Two requests that render the
    SAME prefix bytes (same tools, same leading system messages, same model,
    same chat-template variables) must key identically regardless of what
    their own trailing turn said; two that differ in any of those must key
    differently, or a touch would refresh the wrong prefix."""
    return hashlib.sha256(_canonical(skeleton).encode()).hexdigest()


@dataclass
class TrackedPrefix:
    key: str
    skeleton: dict
    call_site: str
    last_real_seen: float
    approx_tokens: int = 0
    tokens_since_touch: float = 0.0
    touch_in_flight: bool = False


@dataclass
class _EndpointStats:
    sent: int = 0
    hit: int = 0
    missed: int = 0
    skipped: int = 0


class PrefixKeepaliveTracker:
    """Per-endpoint prefix tracking + accounting. No I/O — see the module
    docstring for why the network half lives in ``health.py``."""

    def __init__(self) -> None:
        self._by_endpoint: dict[str, "OrderedDict[str, TrackedPrefix]"] = {}
        self._stats: dict[str, _EndpointStats] = {}

    # --- policy reads -----------------------------------------------------

    @staticmethod
    def enabled(ep_cfg: Any) -> bool:
        return (bool(getattr(ep_cfg, "prefix_keepalive_call_sites", ()))
                and int(getattr(ep_cfg, "prefix_keepalive_trigger_tokens", 0)
                        or 0) > 0)

    @staticmethod
    def matches(ep_cfg: Any, call_site: str) -> bool:
        patterns = getattr(ep_cfg, "prefix_keepalive_call_sites", ())
        return any(fnmatch.fnmatch(call_site or "", pat) for pat in patterns)

    # --- capture + accounting ----------------------------------------------

    def observe_completion(
        self, ep_name: str, ep_cfg: Any, *, call_site: str, payload: dict,
        status: str, input_tokens: int | None, cached_tokens: int | None,
        now: float,
    ) -> None:
        """Called for EVERY completed chat_completion request on ``ep_name``.

        A real, successful, call_site-matching completion refreshes (or
        creates) the tracked prefix its skeleton belongs to — resetting that
        prefix's countdown, because it just did the job a touch exists to do.
        EVERY completion, matching or not, then adds its own uncached prompt
        tokens (``input_tokens - cached_tokens``, when both are known — never
        coerced from an absent value, which would silently mean "0 other
        traffic" instead of "cannot tell") to every OTHER tracked prefix's
        countdown, per the module's whole thesis: the countdown is OTHER
        uncached prefill on this endpoint, not calls to any one call_site."""
        if not self.enabled(ep_cfg):
            return
        bucket = self._by_endpoint.setdefault(ep_name, OrderedDict())
        matched_key: str | None = None
        if status == "ok" and self.matches(ep_cfg, call_site):
            skeleton = derive_skeleton(payload)
            if skeleton is not None:
                matched_key = skeleton_key(skeleton)
                entry = bucket.get(matched_key)
                if entry is None:
                    entry = TrackedPrefix(
                        key=matched_key, skeleton=skeleton, call_site=call_site,
                        last_real_seen=now)
                    bucket[matched_key] = entry
                    self._evict(bucket, int(
                        getattr(ep_cfg, "prefix_keepalive_max_prefixes", 0) or 0))
                entry.last_real_seen = now
                entry.tokens_since_touch = 0.0
                if input_tokens:
                    entry.approx_tokens = int(input_tokens)
                bucket.move_to_end(matched_key)
        if input_tokens is None or cached_tokens is None or not bucket:
            return
        uncached = input_tokens - cached_tokens
        if uncached <= 0:
            return
        for key, entry in bucket.items():
            if key == matched_key:
                continue  # already reset above — it just refreshed itself
            entry.tokens_since_touch += uncached

    @staticmethod
    def _evict(bucket: "OrderedDict[str, TrackedPrefix]", max_prefixes: int) -> None:
        """LRU by last REAL use (insertion/move-to-end tracks that, never a
        touch — see ``EndpointConfig.prefix_keepalive_max_prefixes``)."""
        if max_prefixes <= 0:
            return
        while len(bucket) > max_prefixes:
            bucket.popitem(last=False)

    # --- touch scheduling ---------------------------------------------------

    def due_for_touch(self, ep_name: str, ep_cfg: Any, now: float) -> list[TrackedPrefix]:
        bucket = self._by_endpoint.get(ep_name)
        if not bucket:
            return []
        trigger = int(getattr(ep_cfg, "prefix_keepalive_trigger_tokens", 0) or 0)
        if trigger <= 0:
            return []
        idle_s = int(getattr(ep_cfg, "prefix_keepalive_idle_s", 0) or 0)
        due = []
        for entry in bucket.values():
            if entry.touch_in_flight:
                continue
            if entry.tokens_since_touch < trigger:
                continue
            if idle_s > 0 and (now - entry.last_real_seen) > idle_s:
                # Nobody is coming back to this conversation; do not spend
                # decode keeping it warm forever.
                continue
            due.append(entry)
        return due

    def build_touch_payload(self, entry: TrackedPrefix) -> dict:
        """Skeleton + ONE unique trailing user turn. The nonce is what makes
        this a NEW-SESSION-shaped touch rather than a repeat of the last one
        — see the module docstring for why that distinction is load-bearing."""
        nonce = secrets.token_hex(8)
        messages = list(entry.skeleton.get("system_messages") or [])
        messages.append({"role": "user", "content": f"{TOUCH_USER_PREFIX} {nonce}"})
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": TOUCH_MAX_TOKENS,
            "temperature": 0.0,
        }
        if entry.skeleton.get("tools"):
            payload["tools"] = entry.skeleton["tools"]
        if entry.skeleton.get("model"):
            payload["model"] = entry.skeleton["model"]
        if entry.skeleton.get("chat_template_kwargs"):
            payload["chat_template_kwargs"] = dict(entry.skeleton["chat_template_kwargs"])
        return payload

    def record_dispatch(self, entry: TrackedPrefix) -> None:
        """At most one touch in flight per prefix; reset the counter HERE —
        on dispatch, not on a verdict — so a touch that is itself slow does
        not also let its own prefix cross the trigger a second time before it
        returns."""
        entry.touch_in_flight = True
        entry.tokens_since_touch = 0.0

    def record_result(self, ep_name: str, entry: TrackedPrefix | None, result: str) -> None:
        """``result`` is one of ``"hit"``/``"missed"``/``"skipped"``. A
        ``"skipped"`` touch (endpoint paused/unhealthy/draining/full) may be
        recorded with no entry at all — see ``Health._schedule_prefix_keepalive_
        touches``, which can skip before it ever picks one."""
        if entry is not None:
            entry.touch_in_flight = False
        stats = self._stats.setdefault(ep_name, _EndpointStats())
        if result == "hit":
            stats.sent += 1
            stats.hit += 1
        elif result == "missed":
            stats.sent += 1
            stats.missed += 1
        elif result == "skipped":
            stats.skipped += 1

    # --- observability -------------------------------------------------------

    def status_snapshot(self, ep_name: str, now: float) -> dict | None:
        """``/v1/status`` per-endpoint ``prefix_keepalive`` block. Omitted
        entirely (returns None) for an endpoint that has never tracked or
        touched anything, so the payload stays quiet on every endpoint that
        does not use this feature — same convention as ``goodput``/
        ``thinking`` next door in ``http_handlers.py``."""
        bucket = self._by_endpoint.get(ep_name)
        stats = self._stats.get(ep_name)
        if not bucket and stats is None:
            return None
        return {
            "tracked": [
                {
                    "key": entry.key[:12],
                    "call_site": entry.call_site,
                    "approx_tokens": entry.approx_tokens,
                    "last_real_seen_age_s": round(now - entry.last_real_seen, 1),
                    "tokens_since_touch": int(entry.tokens_since_touch),
                    "touch_in_flight": entry.touch_in_flight,
                }
                for entry in (bucket or {}).values()
            ],
            "touches_sent": stats.sent if stats else 0,
            "touches_hit": stats.hit if stats else 0,
            "touches_missed": stats.missed if stats else 0,
            "touches_skipped": stats.skipped if stats else 0,
        }
