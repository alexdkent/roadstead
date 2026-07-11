"""Empirical timeout-advice model — realistic per-call timeout values
derived from the proxy's own measured end-to-end latency.

Pure computation — no I/O, no framework imports.  State lives in
``TimeoutModel``; the ``ProxyService`` feeds it from every completion
and bootstraps it from ``proxy_completions`` on startup (mirrors the
``cost_model`` lifecycle).

"End-to-end latency" is what a caller actually waits on: admission queue
wait + backend inference.  Because the priority tier only moves the
queue-wait portion, conditioning the distribution on ``(endpoint, tier,
token-size)`` lets the tier input change the answer empirically without
any separate queue model.

``advise()`` returns four numbers for a ``(model, tier, est_in, est_out)``
query:
  - ``min`` / ``median`` / ``p95`` — raw percentiles, for callers that
    want a tighter fail-fast bound (e.g. an interactive chat turn).
  - ``recommended`` — ``max(p99 * margin, floor)``: the conservative
    default most callers should use.  The margin absorbs tail/load drift;
    the floor keeps a thin sample from returning a dangerously low value.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import deque
from typing import Deque

from .config import normalize_endpoint

# ---------------------------------------------------------------------------
# Per-endpoint-class floors (seconds).
#
# Proxy-path analog of framework ``nexus_translate._PER_ROLE_TIMEOUT_S``
# (which only governs the dead nexus-direct path).  ``recommended`` is
# never allowed below the floor for the class.  KEEP IN SYNC with the
# classes in ``config.DEFAULT_ENDPOINTS`` — ``test_timeout_model`` asserts
# every endpoint class has a floor here.
# ---------------------------------------------------------------------------

#
# THIS TABLE IS A SYNCED MIRROR OF models.yaml `timeout_floor_s`, not a second
# source of truth. models.yaml is authoritative: it seeds the SERVER's
# TimeoutModel floors (state.py, via model_catalog.build_class_floors), layered
# over these as the fallback for any class absent from the yaml. These values
# ALSO back the CLIENT-side floor_for() (framework/timeout_advice) — which has a
# one-way dep on this module and can't read the catalog — so a value here that
# disagrees with the yaml makes the client's sub-floor-honor decision drift from
# what the server enforces. Keep them equal; test_timeout_floor_yaml_sync pins it.
FLOOR_S: dict[str, float] = {
    # classify (Qwen3.6-35B + vision): raised 45→180 in the 2026-07-04 nexus
    # loadout (commit 35df6545). kv-unified allows big single requests; a cold
    # ~37K prefill was hitting the old 45s floor. The vision half (image encode +
    # extraction) also runs 15-30s/call. Mirrors models.yaml classify.timeout_floor_s.
    "classify": 180.0,
    # companion = the `composer` role, now Qwen3.5-122B (2026-07-03 cutover from
    # the 80B). Raised 180→360 (commit 35df6545): the 122B is materially slower,
    # and the old 180s floor was truncating turns + streams under load. Mirrors
    # models.yaml composer.timeout_floor_s.
    "companion": 360.0,
    "thinker": 180.0,
    # creative (Gemma-4-31B abliterated) is dense (~25 tok/s single-stream, 16 slots on the Arc Pro boxa);
    # long-form creative generation needs a high floor so cold-start (no history) doesn't
    # cap requests at the 60s default. recommended = max(p99*margin, floor) once warm.
    # 900s (15min) so an ON-DEMAND cold model LOAD (several minutes) + generation can WAIT
    # in-queue rather than time out — the song-compose author stages tolerate the wait.
    "creative": 900.0,
    # companion-lite (Gemma-4-26B-A4B abliterated MoE ~4B-active, boxa co-tenant, cutover 2026-07-11):
    # the DEDICATED chat-loop model (Sidekick's chat turn) — 26B-smart but fast (~4B active @64.7 t/s)
    # off a warm persona+catalog prefix. 60s floor mirrors models.yaml companion-lite.timeout_floor_s
    # (bump BOTH in lockstep). Superseded the dense Ministral-3-8B; replaces the retired `mellum` slot.
    "companion-lite": 60.0,
    "gemma": 60.0,
    # 2026-06-08: the "gemma-hot" endpoint class was removed from DEFAULT_ENDPOINTS
    # (E2B :9090 decommissioned; gemma-greeter consolidated onto the "gemma"/E4B
    # backend). The "gemma-hot" legacy-label entry that used to live here was
    # dropped 2026-07-03: "gemma-hot" is a real catalog alias (of gemma/E4B),
    # so normalize_endpoint() now resolves it to "gemma" before this dict is
    # ever consulted, making a separate "gemma-hot" key permanently
    # unreachable dead code. Tests needing an isolated small-floor target use
    # "rerank" instead.
    "rerank": 10.0,
    "embed": 15.0,
    # ("vision9b": 45.0 dropped 2026-07-03 — the dedicated 8B analyst-vision was
    # consolidated into `classify`; no role/alias resolves to "vision9b", so the
    # key was unreachable. Vision now uses the classify floor above.)
}

# Fallback floor for an unknown endpoint class.
_DEFAULT_FLOOR_S = 60.0

# ---------------------------------------------------------------------------
# Per-caller-class timeout CEILINGS (seconds).
#
# The upper bound on the adaptive (surge × size) recommendation, split by
# caller class per the operator decision (2026-07-05): interactive turn/app
# traffic gets a tighter ceiling; background/ingestion + long-form author
# roles get a generous one.  Resolution order (see ``resolve_ceiling_s``):
#   1. a per-role ``timeout_ceiling_s`` override from models.yaml (for roles
#      that are inherently long-running regardless of tier — e.g. creative
#      song-compose, the generation roles), else
#   2. the interactive band if the request is P0/P1/P2, else the background
#      band (P3_INGESTION / P4_HYGIENE).
# The resolved ceiling is ALWAYS lifted to at least the class floor, so a
# ceiling can never strangle a call below the deadline the model already
# guarantees (that would re-introduce the sub-floor-cliff regression).
# ---------------------------------------------------------------------------
_INTERACTIVE_CEILING_S = 600.0
_BACKGROUND_CEILING_S = 1800.0


def resolve_ceiling_s(
    endpoint: str,
    *,
    interactive: bool,
    role_ceilings: dict[str, float] | None = None,
    floor_s: float,
    interactive_s: float = _INTERACTIVE_CEILING_S,
    background_s: float = _BACKGROUND_CEILING_S,
) -> float:
    """Upper bound (seconds) for the adaptive recommendation of this call.

    ``interactive`` is ``priority <= P2_POST_TURN`` (computed at the call site
    so this stays enum-agnostic and pure).  A per-role override in
    ``role_ceilings`` (from ``models.yaml timeout_ceiling_s``) wins over the
    tier band.  The result is never below ``floor_s``."""
    ep = normalize_endpoint(endpoint)
    override = (role_ceilings or {}).get(ep)
    ceiling = float(override) if override and override > 0 else (
        interactive_s if interactive else background_s
    )
    return max(ceiling, float(floor_s))


def surge_factor(
    in_flight: int,
    queued: int,
    max_slots: int,
    *,
    k_load: float = 0.5,
    surge_max: float = 3.0,
) -> float:
    """Live-contention multiplier ≥ 1.0.

    At or under capacity the factor is 1.0 (no change).  Backlog beyond
    capacity — the number of requests that must clear before this one starts —
    stretches the deadline roughly with expected queue wait: ``over`` is the
    backlog measured in units of endpoint capacity, clamped at ``surge_max``.
    ``max_slots <= 0`` (unknown capacity) yields 1.0."""
    if max_slots <= 0:
        return 1.0
    over = max(0.0, (int(in_flight) + int(queued) - int(max_slots)) / float(max_slots))
    return 1.0 + k_load * min(over, surge_max)


def size_stretch(
    est_in: int,
    *,
    k_size: float = 0.5,
    size_max: float = 4.0,
) -> float:
    """Continuous multiplier ≥ 1.0 for prompts past the top input bucket.

    The coarse ``_IN_EDGES`` buckets top out at 16K, so a 60K-token prompt gets
    the same empirical cell as a 17K one.  This smooths that cliff: ``over`` is
    how many multiples of the top edge the prompt exceeds, clamped at
    ``size_max``.  At or below the top edge the factor is 1.0."""
    top = _IN_EDGES[-1]
    over = max(0.0, (int(est_in) - top) / float(top))
    return 1.0 + k_size * min(over, size_max)


def apply_load_and_ceiling(
    recommended_ms: float,
    *,
    in_flight: int,
    queued: int,
    max_slots: int,
    est_in: int,
    ceiling_ms: float,
    k_load: float = 0.5,
    surge_max: float = 3.0,
    k_size: float = 0.5,
    size_max: float = 4.0,
) -> tuple[float, float, float]:
    """Apply the live-load surge and size-stretch to a base recommendation,
    bounded by ``ceiling_ms``.  Returns ``(effective_ms, surge, stretch)`` so
    the caller can surface the factors for observability.  Pure."""
    surge = surge_factor(in_flight, queued, max_slots, k_load=k_load, surge_max=surge_max)
    stretch = size_stretch(est_in, k_size=k_size, size_max=size_max)
    effective = min(recommended_ms * surge * stretch, ceiling_ms)
    return effective, surge, stretch

# Token-size bucket edges.  Output dominates decode time, so it is the
# finer dimension; input (prefill) is coarse.
_OUT_EDGES = [128, 512, 2048, 8192]   # → 5 buckets (indices 0..4)
_IN_EDGES = [1024, 4096, 16384]       # → 4 buckets (indices 0..3)


def _bucket(n: int, edges: list[int]) -> int:
    """Stable bucket index for a token count given ascending edges."""
    return bisect_left(edges, max(0, int(n)))


def percentile(sorted_values: list[float], pct: float) -> float:
    """Percentile by the same index method the proxy uses elsewhere
    (``observability.RollingMetrics.percentile``): nearest-rank on a
    pre-sorted list.  ``sorted_values`` must be sorted ascending."""
    if not sorted_values:
        return 0.0
    idx = int(len(sorted_values) * pct / 100)
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


class TimeoutModel:
    """Conditioned end-to-end latency distributions for timeout advice.

    Cells are keyed by ``(endpoint_class, priority, in_bucket, out_bucket)``.
    Each cell holds a bounded, age-pruned reservoir of end-to-end
    latencies (ms).  ``advise()`` walks a coarsening fallback hierarchy
    until it finds a level with enough samples, so sparse/cold cells
    degrade gracefully instead of erroring.
    """

    def __init__(
        self,
        *,
        margin: float = 1.5,
        window_s: float = 7 * 86400.0,
        min_samples: int = 30,
        max_samples_per_cell: int = 2000,
        floors: dict[str, float] | None = None,
    ) -> None:
        self._margin = margin
        self._window_s = window_s
        self._min_samples = min_samples
        self._max_per_cell = max_samples_per_cell
        self._floors = dict(FLOOR_S if floors is None else floors)
        # key -> deque of (timestamp_s, latency_ms)
        self._cells: dict[tuple[str, int, int, int], Deque[tuple[float, float]]] = {}

    # ----- feed -----

    def record(
        self,
        endpoint: str,
        priority: int,
        input_tokens: int,
        output_tokens: int,
        end_to_end_ms: float,
        status: str,
        now: float,
    ) -> None:
        """Add one completion to the distribution.

        Only ``status == "ok"`` samples enter — a timeout/error duration
        is not a representative latency (mirrors the ``status='ok'``
        filter the cost-model bootstrap uses).
        """
        if status != "ok" or end_to_end_ms <= 0:
            return
        ep = normalize_endpoint(endpoint)
        key = (
            ep,
            int(priority),
            _bucket(input_tokens, _IN_EDGES),
            _bucket(output_tokens, _OUT_EDGES),
        )
        cell = self._cells.get(key)
        if cell is None:
            cell = deque(maxlen=self._max_per_cell)
            self._cells[key] = cell
        cell.append((now, float(end_to_end_ms)))

    def prune(self, now: float) -> None:
        """Drop samples older than the window and any cells left empty."""
        cutoff = now - self._window_s
        empty: list[tuple] = []
        for key, cell in self._cells.items():
            while cell and cell[0][0] < cutoff:
                cell.popleft()
            if not cell:
                empty.append(key)
        for key in empty:
            del self._cells[key]

    # ----- query -----

    def _collect(
        self,
        ep: str,
        *,
        priority: int | None = None,
        in_b: int | None = None,
        out_b: int | None = None,
    ) -> list[float]:
        """Merge latencies from every cell matching the given filter.

        ``None`` on a dimension means "aggregate over it" — this is how
        the fallback hierarchy coarsens."""
        out: list[float] = []
        for (k_ep, k_pri, k_in, k_out), cell in self._cells.items():
            if k_ep != ep:
                continue
            if priority is not None and k_pri != priority:
                continue
            if in_b is not None and k_in != in_b:
                continue
            if out_b is not None and k_out != out_b:
                continue
            out.extend(v for _, v in cell)
        return out

    def floor_ms(self, endpoint: str) -> float:
        return self._floors.get(normalize_endpoint(endpoint), _DEFAULT_FLOOR_S) * 1000.0

    def advise(
        self,
        endpoint: str,
        priority: int,
        est_in: int,
        est_out: int,
    ) -> dict:
        """Return ``{min_ms, median_ms, p95_ms, recommended_ms,
        recommended_timeout_s, sample_count, source}`` for the query.

        ``source`` names the fallback level that produced the numbers, so
        callers (and the shadow report) can tell apart a well-supported
        cell from a floor-only cold start.
        """
        ep = normalize_endpoint(endpoint)
        pri = int(priority)
        in_b = _bucket(est_in, _IN_EDGES)
        out_b = _bucket(est_out, _OUT_EDGES)
        floor_ms = self.floor_ms(ep)

        levels = (
            ("cell", dict(priority=pri, in_b=in_b, out_b=out_b)),
            ("tier_out", dict(priority=pri, out_b=out_b)),
            ("tier", dict(priority=pri)),
            ("endpoint", dict()),
        )
        samples: list[float] = []
        source = "floor"
        for name, filt in levels:
            collected = self._collect(ep, **filt)
            if len(collected) >= self._min_samples:
                samples = collected
                source = name
                break

        if samples:
            samples.sort()
            mn = samples[0]
            med = percentile(samples, 50)
            p95 = percentile(samples, 95)
            p99 = percentile(samples, 99)
            recommended = max(p99 * self._margin, floor_ms)
            n = len(samples)
        else:
            mn = med = p95 = floor_ms
            recommended = floor_ms
            n = 0

        return {
            "min_ms": round(mn, 1),
            "median_ms": round(med, 1),
            "p95_ms": round(p95, 1),
            "recommended_ms": round(recommended, 1),
            "recommended_timeout_s": math.ceil(recommended / 1000.0),
            "sample_count": n,
            "source": source,
        }

    def snapshot(self) -> dict:
        """Debug view: sample count per endpoint class."""
        per_ep: dict[str, int] = {}
        for (ep, _pri, _in, _out), cell in self._cells.items():
            per_ep[ep] = per_ep.get(ep, 0) + len(cell)
        return {
            "margin": self._margin,
            "window_s": self._window_s,
            "min_samples": self._min_samples,
            "cells": len(self._cells),
            "samples_per_endpoint": per_ep,
        }
