"""Background-band concurrency cap (fast_path_reserve_slots).

~95% of thinker load is background; the band should be able to use all but a
small fast-path reserve, scaling with max_slots (not a hard-coded number).
Intra-band fairness is the DRR scheduler's job, not this cap. The interactive
reservation still keys off background_floor_slots, so raising the background
cap doesn't shrink interactive headroom.
"""
from __future__ import annotations

from types import SimpleNamespace

from originfleet.llmproxy.config import DEFAULT_ENDPOINTS, EndpointConfig, PriorityBand
from originfleet.llmproxy.scheduler import Scheduler


def _ep(max_slots: int, reserve: int) -> EndpointConfig:
    return EndpointConfig(
        endpoint_class="thinker", role="llama-thinker",
        max_slots=max_slots, fast_path_reserve_slots=reserve,
    )


# --- config: cap = max_slots - reserve, scales, never below floor ---

def test_thinker_default_cap_is_max_slots_minus_one():
    t = DEFAULT_ENDPOINTS["thinker"]
    assert t.fast_path_reserve_slots == 1
    assert t.background_floor_slots == 1          # reservation unchanged
    assert t.background_cap_slots == t.max_slots - 1  # 5 of 6 today


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
    return {"thinker": {
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
