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

FLOOR_S: dict[str, float] = {
    "chat": 30.0,
    "companion": 180.0,
    "thinker": 180.0,
    "gemma": 60.0,
    # 2026-06-08: the "gemma-hot" endpoint class was removed from DEFAULT_ENDPOINTS
    # (E2B :9090 decommissioned; gemma-greeter consolidated onto the "gemma"/E4B
    # backend). This floor is RETAINED as a harmless legacy label — the forward-only
    # doctrine (test_every_endpoint_class_has_a_floor) doesn't require it, and several
    # timeout tests still exercise the model mechanics with a "gemma-hot" label.
    "gemma-hot": 8.0,
    "rerank": 10.0,
    "embed": 15.0,
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
