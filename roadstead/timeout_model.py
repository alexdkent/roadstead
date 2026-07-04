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
