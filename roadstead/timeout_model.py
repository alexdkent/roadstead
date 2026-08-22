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
    # (The former "classify" FLOOR_S entry is removed 2026-07-11 — classify/analyst/vision re-homed
    # onto the boxa `creative` endpoint as ALIASES, so normalize_endpoint() resolves "classify" to
    # "creative" before this lookup and the "creative" floor below applies. On the dedicated boxa the
    # cold prefill that motivated the old 180s floor is ~20s at 1807 t/s pp, and vision extract
    # 15-30s — all well inside the 120s creative floor, with adaptive surge×size widening for big docs.)
    # companion = the `composer` role, now Qwen3.5-122B (2026-07-03 cutover from
    # the 80B). Raised 180→360 (commit 35df6545): the 122B is materially slower,
    # and the old 180s floor was truncating turns + streams under load. Mirrors
    # models.yaml composer.timeout_floor_s.
    # HISTORICAL ONLY as of 2026-08-02 — `companion` is no longer an endpoint
    # class (the 122B backup left the proxy; its class name collided with the
    # `companion` alias, see models.yaml). normalize_endpoint() now resolves
    # "companion" to "thinker" in ONE step, so this key is unreachable for new
    # traffic and the thinker floor below applies. Kept as a tombstone: deleting
    # it invites someone to re-add `companion` as a class and reintroduce the
    # collision. Ledger: `endpoint-class-alias-collision`.
    "companion": 360.0,
    "thinker": 180.0,
    # creative (Qwen3.6-35B-A3B abliterated MoE, ~3B active, 6 slots on the Arc Pro boxa). CONSOLIDATED
    # 2026-07-11: this ONE endpoint now backs THREE roles — `creative`, `companion-lite` (chat loop),
    # AND the `classify`/`analyst`/vision family (re-homed off anvil) — all pure ALIASES that
    # normalize to "creative". 120s is a compromise floor across long song-compose, latency-sensitive
    # chat, and classify text/vision extraction: a stuck turn fails in ~120s instead of 900s while
    # adaptive surge×size still widens up to the 1800s ceiling for a big song-compose or a big document
    # vision extract. Mirrors models.yaml creative.timeout_floor_s (bump BOTH in lockstep). (The former
    # companion-lite AND classify FLOOR_S entries are removed — both resolve via the alias.)
    "creative": 120.0,
    # tier2-chat (Qwen3.6-35B-A3B abliterated, 20 slots on jetty's R9700 under
    # llama.cpp/Vulkan). NEW CLASS 2026-08-19 at the Phase 3 split: `chat`,
    # `nexus-chat` and `companion-lite` left the `creative` class for this one, so
    # normalize_endpoint() resolves all three to "tier2-chat" and THIS floor is what
    # the orchestrator inner loop, the glasses lane and Discord chat now get.
    # Deliberately EQUAL to creative's 120s at the cutover: the split moves aliases,
    # not callers, and changing the floor in the same commit would confound any
    # post-cutover latency reading. This lane is purely interactive (long-form
    # authoring goes to tier3) and decode here is ~1.5x the boxa, so 120 is loose —
    # right-size it at Phase 5 against the proved workload, together with the yaml.
    # Mirrors models.yaml tier2-chat.timeout_floor_s (bump BOTH in lockstep).
    "tier2-chat": 120.0,
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


# ---------------------------------------------------------------------------
# Size stretch (D1, 2026-08-03).
#
# The stretch used to measure ``over`` in LINEAR multiples of ``_IN_EDGES[-1]``
# (16,384) and clamp it at ``size_max`` (4.0 from config), so the multiplier
# capped at 3.0x and stopped growing at 5 x 16,384 = 81,920 input tokens. tier3
# now serves 700K context — the constant was tuned when the ceiling was 128K,
# and models.yaml has carried a warning comment about it. A 700K prompt was
# advised exactly what an 82K one was.
#
# Two things change, and neither of them is "raise the clamp" (config.py owns
# `timeout_size_k` / `timeout_size_max` and is not ours to edit — the fix has to
# be right with k_size=0.5, size_max=4.0 arriving unchanged):
#
#   1. `over` is now measured in SIZE UNITS — a fixed multiplicative step in
#      prompt length — rather than in linear multiples of the reference. The
#      step is chosen so that the default clamp of 4.0 units is reached at the
#      largest context the fleet serves (700K) instead of at 82K.
#   2. The term is SUPERLINEAR in those units, because prefill is superlinear in
#      length: measured tier3 prefill falls from ~1,426 tok/s at 213K to ~729
#      tok/s at 578K, so doubling the prompt more than doubles the prefill. A
#      linear term with a bigger clamp would still under-serve the top end.
#
# The reference is pinned to its own constant rather than to ``_IN_EDGES[-1]``,
# so widening the empirical buckets (D2) does not silently move the point where
# the stretch starts — those are independent decisions and coupling them cost a
# regression while this was being written.
#
# The result is the MAX of the legacy linear term and the new superlinear one,
# so the well-behaved 16K..82K band keeps exactly the multiplier it has today
# (this fix only ever widens a deadline) and the superlinear term takes over
# past ~107K, where the legacy one had already flatlined. That leaves a short
# plateau at 82K..107K: intentional, and it is where the empirical cells that
# D2 adds are best populated.
#
# With the config defaults (k_size=0.5, size_max=4.0) the curve reads:
#     32K -> 1.50x   64K -> 2.50x   82K -> 3.00x  (all unchanged)
#    128K -> 3.46x  262K -> 5.37x  700K -> 9.00x  (was a flat 3.00x)
# 9.0x on the 180s thinker floor is 1620s, just inside the 1800s background
# ceiling and comfortably past the ~1085s of pure prefill a 700K prompt implies.
# ---------------------------------------------------------------------------

#: Prompt size at which the stretch starts (below it the factor is exactly 1.0).
_SIZE_STRETCH_REF_TOKENS = 16_384
#: The context at which the stretch clamps. ⚠️ 2026-08-22: tier3 now serves
#: 1,048,576, so this is NO LONGER 'the largest context the fleet serves' —
#: it is a CALIBRATION POINT, left at 700K DELIBERATELY. Moving it changes
#: _SIZE_STRETCH_UNIT_RATIO (2.556 -> 2.828) and therefore retunes the timeout
#: curve for EVERY endpoint, which is not a cutover-window change.
#: Leaving it is safe, and that was checked rather than assumed: prompts above
#: 700K simply clamp to the same 9.0x, giving 9.0 x 180s = 1620s against a
#: MEASURED 974s of prefill for a 994,120-token call (2026-08-22) — inside the
#: 1800s background ceiling with room. Revisit only with a deliberate retune.
_MAX_SERVED_CONTEXT_TOKENS = 700_000
#: The default ``size_max``. Used ONLY to calibrate the unit step; the live value
#: still arrives as an argument (config.timeout_size_max).
_SIZE_STRETCH_DEFAULT_MAX_UNITS = 4.0
#: One size unit = this multiplicative step in prompt length (~2.556x).
_SIZE_STRETCH_UNIT_RATIO = (
    _MAX_SERVED_CONTEXT_TOKENS / _SIZE_STRETCH_REF_TOKENS
) ** (1.0 / _SIZE_STRETCH_DEFAULT_MAX_UNITS)
#: Prefill is superlinear in prompt length, so the stretch is too.
_SIZE_STRETCH_EXPONENT = 2.0


def size_stretch(
    est_in: int,
    *,
    k_size: float = 0.5,
    size_max: float = 4.0,
    ref_tokens: int = _SIZE_STRETCH_REF_TOKENS,
    unit_ratio: float = _SIZE_STRETCH_UNIT_RATIO,
    exponent: float = _SIZE_STRETCH_EXPONENT,
) -> float:
    """Continuous, monotonic multiplier ≥ 1.0 for prompts past ``ref_tokens``.

    The empirical cells resolve a prompt's size only as far as the top input
    bucket; past that, this is what makes a bigger prompt get a longer deadline.
    ``size_max`` clamps ``over`` as it always did — but ``over`` is now counted
    in ``unit_ratio``-fold size units rather than linear multiples of the
    reference, so the default clamp of 4.0 is reached at 700K rather than 82K.
    See the block comment above for the calibration and the resulting curve."""
    n = int(est_in)
    if n <= ref_tokens:
        return 1.0
    # legacy linear term — unchanged below its clamp, so nothing in the
    # 16K..82K band ever gets a shorter deadline than it does today.
    linear = min((n - ref_tokens) / float(ref_tokens), size_max)
    units = math.log(n / float(ref_tokens)) / math.log(unit_ratio)
    superlinear = min(units, size_max) ** exponent
    return 1.0 + k_size * max(linear, superlinear)


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
# D2 (2026-08-03): the top edge used to be 16384, so EVERY prompt from 17K to
# 700K shared one empirical cell. That cell's p99 is dominated by ~20K-token
# calls, so the class floor always won and the model could never learn
# long-context latency — the live timeout events logged `recommended_ms` of
# exactly 180000.0 (the thinker floor) for 123K-token prompts.
#
# The new edges follow the shape of the measured `thinker` traffic (14 days,
# status=ok): <16K n=61,319 · 16-32K n=272 · 32-64K n=216 · 64-128K n=165 ·
# >128K n=192. Every one of those clears `min_samples=30` in a 7-day window, so
# the finer cells resolve empirically from the day they deploy rather than after
# a warm-up. The 262144 edge separates the 128-256K population (avg input in the
# >128K bucket is 177K) from the genuinely huge tier3 prompts; if that top cell
# is thin it falls back up the ladder and the monotonicity guard gives it the
# 128-256K cell's answer, which is the honest floor for it.
_IN_EDGES = [1024, 4096, 16384, 32768, 65536, 131072, 262144]  # → 8 buckets

#: D3: percentile of the OBSERVED output-size distribution used when a caller
#: declared no output budget (``est_out <= 0``).  High, not median — an absent
#: ``max_tokens`` is an absent CAP, and this picks the CELL whose p99 x margin
#: then sizes the deadline, so it is a tail bound on a tail bound.
#:
#: Calibrated against the live `thinker` distribution (7 days, status=ok,
#: n=32,858 — of which 62% genuinely omit ``max_tokens``, confirming this is the
#: common path and not an edge case):
#:     bucket 0 (<=128)   60.6%      p50 =    70 tokens
#:     bucket 1 (<=512)   34.4%      p90 =   404 tokens  -> bucket 1
#:     bucket 2 (<=2048)   4.9%      p95 -> bucket 2
#:     bucket 3 (<=8192)   0.2%      p99 = 1,090 tokens
#: 95 rather than 90: p90 lands in bucket 1 and leaves the ~5% of unknown-budget
#: calls that run past 512 tokens sized off a distribution they will overrun —
#: which is the exact failure this defect is about. 95 covers 99.4% of them.
#: Raising this only ever lengthens a deadline; the per-caller ceiling still binds.
_UNKNOWN_OUT_PCT = 95.0
#: Used only when the endpoint has no samples at all; such a model answers from
#: the floor regardless of bucket, so this is inert in practice.
_UNKNOWN_OUT_FALLBACK_BUCKET = len(_OUT_EDGES) // 2


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
        if priority is not None and in_b is not None and out_b is not None:
            # Fully-specified: one dict lookup instead of a full scan. The
            # monotonicity guard walks the whole (in x out) lattice, so this is
            # the hot path now.
            cell = self._cells.get((ep, priority, in_b, out_b))
            return [v for _, v in cell] if cell else []
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

    def _resolve_unknown_out_bucket(self, ep: str, pri: int) -> int:
        """Out-bucket to use when the caller declared NO output budget (D3).

        ``est_out <= 0`` does not mean "this call will produce ~no output" — it
        means the caller omitted ``max_tokens``, which EVERY stock OpenAI client
        does (the ``pool`` CLI sends ``{messages, model, tools, stream,
        stream_options}``). Bucketing that as 0 put it in the SHORTEST-output
        distribution, and 0 is the lowest bucket the monotonicity guard can lift
        from, so the guard structurally could not rescue it.

        Policy: resolve it EMPIRICALLY, to the ``_UNKNOWN_OUT_PCT`` percentile of
        the output sizes this endpoint/priority is actually observed to produce.
        Rationale for a high percentile rather than the median: an absent
        ``max_tokens`` is an absent CAP, so the call may run to the model's own
        limit, and the cost of over-estimating is a slot held slightly too long
        while the cost of under-estimating is the call being killed mid-answer.
        Rationale for empirical rather than a constant: an endpoint whose outputs
        really are tiny (``embed`` / ``rerank`` record 0 output tokens) resolves
        straight back to bucket 0 and is not inflated at all.

        Falls back across priorities when the tier is thin, then to a fixed
        mid bucket — inert in practice, because a model with no samples for the
        endpoint answers from the floor whatever bucket it picks."""
        counts: dict[int, int] = {}
        for (k_ep, k_pri, _k_in, k_out), cell in self._cells.items():
            if k_ep != ep or k_pri != pri:
                continue
            counts[k_out] = counts.get(k_out, 0) + len(cell)
        if not counts:
            for (k_ep, _k_pri, _k_in, k_out), cell in self._cells.items():
                if k_ep != ep:
                    continue
                counts[k_out] = counts.get(k_out, 0) + len(cell)
        if not counts:
            return _UNKNOWN_OUT_FALLBACK_BUCKET
        target = sum(counts.values()) * _UNKNOWN_OUT_PCT / 100.0
        cum = 0
        for b in sorted(counts):
            cum += counts[b]
            if cum >= target:
                return b
        return max(counts)

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
        # D3: est_out <= 0 is UNKNOWN, not zero. See _resolve_unknown_out_bucket.
        out_unknown = int(est_out) <= 0
        out_b = (
            self._resolve_unknown_out_bucket(ep, pri) if out_unknown
            else _bucket(est_out, _OUT_EDGES)
        )
        floor_ms = self.floor_ms(ep)

        # One memo per advice call: the guard below re-walks the ladder for every
        # lower cell, and the coarse levels (tier / tier_out / endpoint) are
        # identical across all of them.
        cache: dict[tuple, list[float]] = {}
        mn, med, p95, recommended, n, source = self._advise_at(
            ep, pri, in_b, out_b, floor_ms, cache)

        # ── MONOTONICITY GUARD (regression ledger `llm-output-budget-starvation`) ──
        # The advice MUST NOT shrink as the caller asks for more output. It could,
        # and it did: the fallback ladder picks the FIRST level with >= min_samples,
        # so a well-populated `tier_out` bucket answers for a mid-sized request
        # while a sparse HIGH out-bucket falls through to `tier` — which aggregates
        # over every out-bucket and is therefore dominated by small, fast calls.
        #
        # Measured live 2026-07-29 on `thinker` @ P3_INGESTION:
        #     est_out 8192 -> 459s   (source=tier_out, n=107)
        #     est_out 9000 -> 227s   (source=tier,     n=8682)   <-- HALVED
        #
        # That inversion is what made kv4's truncation-retry actively destructive.
        # `resilient_json_call` correctly re-derives its deadline when it bumps
        # max_tokens (6000 -> 9000), and the re-derived deadline came back SMALLER
        # than the one the 6000-token attempt already needed 276s of. Result: 95%
        # of those retries (136/143) were killed at the deadline having produced
        # ZERO output tokens, burning ~12.5 hours of thinker time in three days for
        # nothing at all.
        #
        # And it was self-perpetuating: `record()` only admits status=="ok"
        # samples, so a bucket whose calls always time out can never accumulate the
        # samples that would give it an honest recommendation. The sparse bucket
        # stays sparse forever. A guard is the only way out of that loop.
        #
        # The guard: a request can never be advised LESS than a request for FEWER
        # output tokens on the same endpoint/priority. Strictly safe — it can only
        # ever RAISE a deadline, and the caller-side `cap_s` ceiling still bounds
        # the result.
        #
        # ── EXTENDED TO THE INPUT AXIS (2026-08-03, with D2) ──
        # Finer `_IN_EDGES` made input a real axis, and it starves exactly the
        # same way: the 262K+ cell is thin, falls through to `tier`, and `tier` is
        # dominated by small fast calls — so asking for MORE input would be
        # advised LESS time. Reproduced in test_timeout_sizing.py the moment the
        # new edges landed and before this loop was widened: 120K -> 360000ms,
        # 400K -> 180000ms. Same self-perpetuating trap, too, since `record()`
        # admits only status=="ok" samples and a cell whose calls always time out
        # can never gather the evidence that would correct it.
        #
        # The guarantee is over the whole lattice, not the two axes separately: a
        # caller can raise both at once (a truncation retry on a long prompt), and
        # max-over-each-axis alone does not compose into monotonicity in both.
        # Cost is (in_b+1)*(out_b+1) <= 40 ladder walks, but with `cache` the only
        # uncached work is a dict lookup per cell — the expensive coarse levels
        # are computed once.
        lifted_from = None
        lifted_from_in = None
        for lo_in in range(in_b + 1):
            for lo_out in range(out_b + 1):
                if lo_in == in_b and lo_out == out_b:
                    continue
                _mn, _med, _p95, cand, _n, _src = self._advise_at(
                    ep, pri, lo_in, lo_out, floor_ms, cache)
                if cand > recommended:
                    recommended = cand
                    lifted_from = lo_out if lo_out < out_b else None
                    lifted_from_in = lo_in if lo_in < in_b else None

        out = {
            "min_ms": round(mn, 1),
            "median_ms": round(med, 1),
            "p95_ms": round(p95, 1),
            "recommended_ms": round(recommended, 1),
            "recommended_timeout_s": math.ceil(recommended / 1000.0),
            "sample_count": n,
            "source": source,
        }
        if out_unknown:
            # Observable: a shadow report must be able to tell an absent
            # max_tokens from a caller that genuinely asked for ~nothing.
            out["out_bucket_unknown_resolved_to"] = out_b
        if lifted_from is not None:
            # Observable, so a dashboard/shadow report can see that this endpoint's
            # high out-buckets are starved of samples rather than genuinely fast.
            out["monotonic_lift_from_out_bucket"] = lifted_from
        if lifted_from_in is not None:
            out["monotonic_lift_from_in_bucket"] = lifted_from_in
        return out

    def _advise_at(
        self, ep: str, pri: int, in_b: int, out_b: int, floor_ms: float,
        cache: dict[tuple, list[float]] | None = None,
    ) -> tuple[float, float, float, float, int, str]:
        """The raw fallback-ladder lookup for one (in_bucket, out_bucket) cell.

        Returns ``(min, median, p95, recommended, n, source)`` in ms. Split out of
        ``advise`` so the monotonicity guard can re-query lower cells without
        duplicating the ladder.  ``cache`` memoises ``_collect`` results for the
        duration of ONE ``advise()`` call — the coarse ladder levels do not depend
        on the cell being probed, so the guard's lattice walk recomputes nothing."""
        levels = (
            ("cell", dict(priority=pri, in_b=in_b, out_b=out_b)),
            ("tier_out", dict(priority=pri, out_b=out_b)),
            ("tier", dict(priority=pri)),
            ("endpoint", dict()),
        )
        samples: list[float] = []
        source = "floor"
        for name, filt in levels:
            if cache is None:
                collected = self._collect(ep, **filt)
            else:
                ck = (filt.get("priority"), filt.get("in_b"), filt.get("out_b"))
                collected = cache.get(ck)
                if collected is None:
                    collected = self._collect(ep, **filt)
                    cache[ck] = collected
            if len(collected) >= self._min_samples:
                samples = collected
                source = name
                break

        if samples:
            samples.sort()
            return (samples[0], percentile(samples, 50), percentile(samples, 95),
                    max(percentile(samples, 99) * self._margin, floor_ms),
                    len(samples), source)
        return (floor_ms, floor_ms, floor_ms, floor_ms, 0, source)

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
