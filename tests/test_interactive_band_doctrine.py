"""The interactive band's invariants.

Two defects this pins, both live on the `beacon` registration between Phase 0
(2026-08-23) and the cutover (2026-08-24):

1. **A user-facing chat caller was in the BACKGROUND band.** Beacon was
   registered `P3_INGESTION` while it was still framed as a third evaluation
   surface. After it became the chat brain behind the SPA and SidekickApp, that put
   every turn a human waits on behind ingestion/hygiene batch work and excluded
   it from `fast_path_reserve_slots`. Measured at the time: `thinker` p95
   background wait 583,017 ms, one trivial call 254.9 s.

2. **A deadline floor ABOVE its own ceiling.** The floor was
   `_SMART_DEFAULT_CAP_S` (1800 s) — the BACKGROUND cap — which is 3x the
   interactive ceiling (600 s). Correct while the caller was background;
   incoherent the moment it moved. The generic guard below catches this shape
   for ANY interactive registration, not just Beacon.
"""
from __future__ import annotations

import pytest

from originfleet.llmproxy.acl import IPIdentityMap
from originfleet.llmproxy.config import LLMPriority, ProxyConfig
from originfleet.llmproxy.constants import (
    _INTERACTIVE_CEILING_S,
    _SMART_DEFAULT_CAP_S,
)

_INTERACTIVE = (LLMPriority.P0_REALTIME, LLMPriority.P1_TURN_SUPPORT)

# Source IPs whose traffic a human is directly waiting on. Beacon is the chat
# brain as of the 2026-08-24 cutover (GATEWAY_ORCHESTRATOR_URL -> beacon_bridge
# -> Beacon). If Beacon is ever retired, delete the row rather than demoting it.
_USER_FACING_CALLERS = {"10.0.0.23": "beacon"}


def _acl() -> IPIdentityMap:
    return IPIdentityMap.from_env()


@pytest.mark.parametrize("ip,expected_id", sorted(_USER_FACING_CALLERS.items()))
def test_user_facing_callers_are_in_the_interactive_band(ip, expected_id):
    agent_id, priority = _acl().identify(ip)
    assert agent_id == expected_id, (
        f"{ip} resolves to {agent_id!r}, not {expected_id!r} — the ACL row moved "
        "or the address was reused. A stale source-IP identity is the .12/.18/.19/.86 "
        "mis-claim class; fix the registration, don't relax this test."
    )
    assert priority in _INTERACTIVE, (
        f"{expected_id} ({ip}) is registered {priority.name}, which is NOT the "
        "INTERACTIVE band. A human waits on every one of this caller's requests, "
        "so scheduling it with batch work is a latency regression in the thing "
        "the user actually feels."
    )


def test_no_interactive_caller_has_a_floor_above_the_interactive_ceiling():
    """A `min_timeout_s` floor above the band's ceiling is incoherent.

    Generic on purpose: this is the shape that bit `beacon`, and it will bite
    the next caller promoted from BACKGROUND to INTERACTIVE whose floor is left
    at `_SMART_DEFAULT_CAP_S`.
    """
    ceiling = ProxyConfig().timeout_ceiling_interactive_s
    acl = _acl()
    offenders = []
    for ip in _USER_FACING_CALLERS:
        _, priority = acl.identify(ip)
        if priority not in _INTERACTIVE:
            continue
        floor = acl.min_timeout_s(ip)
        if floor is not None and floor > ceiling:
            offenders.append(f"{ip} floor={floor}s > interactive ceiling={ceiling}s")
    assert not offenders, (
        "interactive caller(s) floored above their own ceiling: "
        + "; ".join(offenders)
        + f". _SMART_DEFAULT_CAP_S ({_SMART_DEFAULT_CAP_S}s) is the BACKGROUND cap "
        f"— interactive callers floor at _INTERACTIVE_CEILING_S ({_INTERACTIVE_CEILING_S}s)."
    )


def test_the_interactive_ceiling_has_one_source_of_truth():
    """`config` must derive its default from `constants`, not restate the number.

    Two literals drift silently; the floor and the ceiling then disagree and
    nothing fails until a request is scheduled.
    """
    assert ProxyConfig().timeout_ceiling_interactive_s == _INTERACTIVE_CEILING_S


def test_batch_harnesses_stay_in_the_background_band():
    """The counterweight: promoting Beacon must not drag the batch fleet with it.

    goose/dsh are agentic harnesses doing long unattended work. They belong
    behind interactive traffic, and `fast_path_reserve_slots` only means anything
    while something is actually reserved *from*.
    """
    acl = _acl()
    for ip, expected in (("10.0.0.14", "recipe-runner"), ("10.0.0.25", "cli-read"),
                         ("10.0.0.41", "cli-write")):
        agent_id, priority = acl.identify(ip)
        assert agent_id == expected
        assert priority not in _INTERACTIVE, (
            f"{expected} ({ip}) is in the INTERACTIVE band; it is unattended batch "
            "work and would compete with user-facing chat."
        )
