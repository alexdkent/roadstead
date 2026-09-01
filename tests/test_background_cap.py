"""Background-band concurrency cap (fast_path_reserve_slots).

~95% of tier3 load is background; the band should be able to use all but a
small fast-path reserve, scaling with max_slots (not a hard-coded number).
Intra-band fairness is the DRR scheduler's job, not this cap. The interactive
reservation still keys off background_floor_slots, so raising the background
cap doesn't shrink interactive headroom.
"""
from __future__ import annotations

from types import SimpleNamespace

from roadstead.config import DEFAULT_ENDPOINTS, EndpointConfig, PriorityBand
from roadstead.scheduler import Scheduler


def _ep(max_slots: int, reserve: int, cap: int = 0) -> EndpointConfig:
    return EndpointConfig(
        endpoint_class="tier3", role="tier3",
        max_slots=max_slots, fast_path_reserve_slots=reserve,
        dispatch_concurrency_cap=cap,
    )


# --- config: cap = max_slots - reserve, scales, never below floor ---

def test_thinker_default_cap_is_max_slots_minus_one():
    t = DEFAULT_ENDPOINTS["tier3"]
    assert t.fast_path_reserve_slots == 1
    assert t.background_floor_slots == 1          # pinned via background_floor_pct=0.0
    assert t.background_cap_slots == t.max_slots - 1  # 31 of 32 today


def test_no_endpoint_floor_starves_interactive():
    """Doctrine guard against the max_slots/background_floor_pct drift that bit
    the tier3 (6→32 slots made the 0.20 floor reserve 6 of 32). For EVERY
    endpoint the background floor must leave at least one slot for an
    interactive/fast-path call, and a reserve-configured endpoint's background
    cap must equal effective_max_slots - reserve. The cap is computed against
    effective_max_slots (NOT max_slots) so a G1 dispatch_concurrency_cap shrinks
    the background ceiling too, preserving the interactive reserve under the cap
    (tier3: effective 3, reserve 1 -> background cap 2). Fails loud if a
    future topology bump changes a slot count without its dependents."""
    for name, ep in DEFAULT_ENDPOINTS.items():
        assert ep.effective_max_slots <= ep.max_slots, (
            f"{name}: effective_max_slots {ep.effective_max_slots} exceeds "
            f"physical max_slots {ep.max_slots}"
        )
        assert ep.background_floor_slots <= max(1, ep.effective_max_slots - 1), (
            f"{name}: floor {ep.background_floor_slots} would starve interactive "
            f"on {ep.effective_max_slots} effective slots"
        )
        if ep.fast_path_reserve_slots:
            assert ep.background_cap_slots == ep.effective_max_slots - ep.fast_path_reserve_slots, (
                f"{name}: cap {ep.background_cap_slots} != effective_max_slots-reserve "
                f"{ep.effective_max_slots - ep.fast_path_reserve_slots}"
            )


def test_companion_dispatch_concurrency_cap():
    """G1 (2026-06-01): the tier3 (then Qwen3-Next-80B) was
    concurrency-fragile (a llama.cpp KV-seq-removal assertion aborted it
    under full pressure), so a 3-wide ceiling was kept explicit via
    dispatch_concurrency_cap.

    2026-06-05 (1ca82e2f): tier3 was downsized from 4×98304 to --parallel 3
    (3×32768) to free nexus memory for Chatterbox Turbo TTS.

    2026-07-03: tier3 swapped to Qwen3.5-122B-A10B (a different model —
    the 80B's specific G1 crash history doesn't carry over) on a newer
    llama.cpp build (b9849). Re-validated via llama-batched-bench at 1/2/4/8
    parallel with no instability found through 8 — see
    infra/nexus/bench_results/composer_122b_optimization_20260703.md.

    2026-07-17 prefill campaign: --parallel raised 4->8 (llama-batched-bench on
    the new env: aggregate decode 28.5->62.8 t/s at 1->8-way, monotonic, no
    cliff; prefill holds ~430-460 t/s aggregate through 8-way) — see
    infra/nexus/bench_results/composer_prefill_campaign_20260717.md.
    cap == max_slots == 8; the interactive reserve is preserved
    (background cap = 8 - 1 = 7).

    2026-08-02: `tier3` is no longer an endpoint class — the 122B stanza left
    the proxy because its class name collided with the `tier3` ALIAS of tier3
    (ledger `endpoint-class-alias-collision`). The history above is kept because
    it is the reasoning behind the cap MECHANISM, which is unchanged; the
    assertions now run against a synthetic endpoint with the 122B's final shape
    rather than a live class, plus the real uncapped endpoint.
    """
    comp = _ep(8, 1, cap=8)                  # the 122B's final shape (2026-07-17)
    assert comp.max_slots == 8               # physical (prefill campaign, 2026-07-17)
    assert comp.dispatch_concurrency_cap == 8
    assert comp.effective_max_slots == 8     # dispatch ceiling (== physical now)
    assert comp.background_cap_slots == 7    # leaves 1 for interactive
    # An uncapped LIVE endpoint is unaffected: effective == physical.
    tier3 = DEFAULT_ENDPOINTS["tier3"]
    assert tier3.dispatch_concurrency_cap == 0
    assert tier3.effective_max_slots == tier3.max_slots


def test_effective_max_slots_clamps_to_min():
    assert _ep(4, 1, cap=3).effective_max_slots == 3
    assert _ep(4, 1, cap=10).effective_max_slots == 4   # cap above physical -> physical
    assert _ep(4, 1, cap=0).effective_max_slots == 4    # 0 -> no cap
    assert _ep(8, 1).effective_max_slots == 8           # default helper, uncapped


def test_cap_scales_with_max_slots():
    # The whole point of not hard-coding: bump slots, cap follows.
    assert _ep(6, 1).background_cap_slots == 5
    assert _ep(8, 1).background_cap_slots == 7
    assert _ep(10, 1).background_cap_slots == 9
    assert _ep(10, 2).background_cap_slots == 8


def test_legacy_cap_defaults_to_floor():
    # reserve=0 → unchanged behavior (floor doubles as cap) for other endpoints.
    ep = _ep(6, 0)
    assert ep.background_cap_slots == ep.background_floor_slots


def test_cap_never_below_floor():
    # Degenerate reserve >= max_slots can't drive the cap under the floor.
    assert _ep(6, 9).background_cap_slots == _ep(6, 9).background_floor_slots


# --- scheduler: background uses up to the cap; fast-path slot preserved ---

def _bg_active(n: int) -> dict:
    return {"tier3": {
        f"r{i}": SimpleNamespace(request=SimpleNamespace(band=PriorityBand.BACKGROUND))
        for i in range(n)
    }}


def test_background_can_dispatch_up_to_cap():
    ep = _ep(6, 1)
    fake = SimpleNamespace(_active={})           # nothing in flight
    avail = Scheduler._available_for_band(fake, ep, PriorityBand.BACKGROUND, 0, None)
    assert avail == 5                            # was 1 under floor-as-cap


def test_background_stops_at_cap_leaving_fast_path_slot():
    ep = _ep(6, 1)
    fake = SimpleNamespace(_active=_bg_active(5))  # 5 background already running
    avail = Scheduler._available_for_band(fake, ep, PriorityBand.BACKGROUND, 5, None)
    assert avail == 0                            # 6th slot held free for fast path


def test_interactive_reservation_unchanged():
    # With background queued, interactive ceiling still = max_slots - floor.
    ep = _ep(6, 1)                               # floor = 1
    eq = SimpleNamespace(band_depth=lambda b: 1)  # background has queued work
    fake = SimpleNamespace(_active={})
    avail = Scheduler._available_for_band(fake, ep, PriorityBand.INTERACTIVE, 0, eq)
    assert avail == 5                            # max_slots - background_floor_slots
