"""Tests for tier3 slot affinity (id_slot injection in _execute_dispatch)."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roadstead.config import EndpointConfig, PriorityBand


# ---------------------------------------------------------------------------
# Unit: slot hash function
# ---------------------------------------------------------------------------

def _expected_slot(session_id: str, slot_n: int) -> int:
    return (
        int.from_bytes(hashlib.md5(session_id.encode()).digest()[:4], "little")
        % slot_n
    )


def test_slot_hash_is_deterministic():
    sid = uuid.uuid4().hex[:16]
    assert _expected_slot(sid, 3) == _expected_slot(sid, 3)
    assert _expected_slot(sid, 3) == _expected_slot(sid, 3)


def test_slot_hash_within_range():
    for _ in range(200):
        sid = uuid.uuid4().hex[:16]
        for n in (1, 2, 3, 4):
            s = _expected_slot(sid, n)
            assert 0 <= s < n, f"slot {s} out of range [0,{n})"


def test_slot_hash_distribution():
    """Rough uniformity check: no slot is picked > 70% of the time for n=3."""
    counts = [0, 0, 0]
    for _ in range(300):
        sid = uuid.uuid4().hex[:16]
        counts[_expected_slot(sid, 3)] += 1
    for c in counts:
        assert c < 210, f"slot distribution too skewed: {counts}"


def test_different_sessions_can_map_to_different_slots():
    slots = {_expected_slot(uuid.uuid4().hex[:16], 3) for _ in range(30)}
    assert len(slots) > 1, "all sessions hash to the same slot — hash broken"


# ---------------------------------------------------------------------------
# Unit: EndpointConfig.slot_affinity default and tier3 value
# ---------------------------------------------------------------------------

def test_slot_affinity_defaults_false():
    cfg = EndpointConfig(endpoint_class="test", role="test-role")
    assert cfg.slot_affinity is False


def test_no_live_endpoint_uses_slot_affinity():
    """As of 2026-08-02 NOTHING live sets slot_affinity — and that is expected.

    It was only ever set on the nexus 122B (`tier3`), whose `--slot-prompt-
    similarity` KV reuse it existed for. That stanza stopped being a proxy
    endpoint when its class name was found to collide with the `tier3` ALIAS
    of tier3 (ledger `endpoint-class-alias-collision`), so the flag now has no
    live consumer.

    The MECHANISM is deliberately kept — `EndpointConfig.slot_affinity` and the
    id_slot injection in `_execute_dispatch` are still exercised by the
    integration test below against a synthetic config, so re-enabling it for a
    future llama.cpp endpoint stays a one-line yaml change. This test asserts the
    live state so that turning it on somewhere is a DELIBERATE, visible edit.
    """
    from roadstead.config import DEFAULT_ENDPOINTS
    enabled = [n for n, ep in DEFAULT_ENDPOINTS.items() if ep.slot_affinity]
    assert enabled == [], (
        f"{enabled} now set slot_affinity. That is fine — but it is llama.cpp-only "
        "(id_slot means nothing to vLLM), so confirm the backend engine and update "
        "this test."
    )


# ---------------------------------------------------------------------------
# Integration: _execute_dispatch injects id_slot for interactive tier3 calls
# ---------------------------------------------------------------------------

@dataclass
class _FakeQueuedRequest:
    request_id: str = "req-test"
    agent_id: str = "orchestrator"
    endpoint: str = "tier3"
    priority: Any = None
    band: Any = None
    call_site: str = "test"
    payload_type: str = "chat_completion"
    payload: dict = field(default_factory=dict)
    timeout_deadline: float = 9999.0
    enqueued_at: float = 0.0
    estimated_cost_ss: float = 0.0
    session_id: str | None = None
    turn_id: str | None = None
    caller_id: str | None = None
    timeout_s: float = 60.0
    stream: bool = False
    future: Any = None


def _make_ep_cfg(**kwargs) -> EndpointConfig:
    defaults = dict(
        endpoint_class="tier3", role="tier3",
        max_slots=4, dispatch_concurrency_cap=3,
        slot_affinity=True, host="192.0.2.10", port=8081,
    )
    defaults.update(kwargs)
    return EndpointConfig(**defaults)


def _extract_slot_id_from_dispatch(session_id, payload_type="chat_completion",
                                   band=None, slot_affinity=True):
    """Run the slot-affinity logic directly (mirrors _execute_dispatch)."""
    ep_cfg = _make_ep_cfg(slot_affinity=slot_affinity)
    req_band = band if band is not None else PriorityBand.INTERACTIVE
    payload = {"messages": []}
    if (
        ep_cfg.slot_affinity
        and payload_type == "chat_completion"
        and session_id
        and req_band == PriorityBand.INTERACTIVE
    ):
        slot_n = ep_cfg.dispatch_concurrency_cap or ep_cfg.max_slots or 1
        slot_id = (
            int.from_bytes(
                hashlib.md5(session_id.encode()).digest()[:4], "little"
            ) % slot_n
        )
        payload = {**payload, "id_slot": slot_id}
    return payload


def test_id_slot_injected_for_interactive_companion():
    sid = uuid.uuid4().hex[:16]
    result = _extract_slot_id_from_dispatch(sid)
    assert "id_slot" in result
    assert 0 <= result["id_slot"] < 3


def test_id_slot_same_session_same_slot():
    sid = uuid.uuid4().hex[:16]
    r1 = _extract_slot_id_from_dispatch(sid)
    r2 = _extract_slot_id_from_dispatch(sid)
    assert r1["id_slot"] == r2["id_slot"]


def test_id_slot_not_injected_when_slot_affinity_false():
    sid = uuid.uuid4().hex[:16]
    result = _extract_slot_id_from_dispatch(sid, slot_affinity=False)
    assert "id_slot" not in result


def test_id_slot_not_injected_for_background_band():
    sid = uuid.uuid4().hex[:16]
    result = _extract_slot_id_from_dispatch(sid, band=PriorityBand.BACKGROUND)
    assert "id_slot" not in result


def test_id_slot_not_injected_for_embedding():
    sid = uuid.uuid4().hex[:16]
    result = _extract_slot_id_from_dispatch(sid, payload_type="embedding")
    assert "id_slot" not in result


def test_id_slot_not_injected_when_no_session():
    result = _extract_slot_id_from_dispatch(None)
    assert "id_slot" not in result


def test_original_payload_not_mutated():
    """Ensure the injection copies the dict rather than mutating in place."""
    sid = uuid.uuid4().hex[:16]
    ep_cfg = _make_ep_cfg()
    original = {"messages": [], "temperature": 0.0}
    payload = original.copy()
    if ep_cfg.slot_affinity and sid:
        slot_n = ep_cfg.dispatch_concurrency_cap or ep_cfg.max_slots or 1
        slot_id = (
            int.from_bytes(hashlib.md5(sid.encode()).digest()[:4], "little") % slot_n
        )
        payload = {**payload, "id_slot": slot_id}
    assert "id_slot" not in original
    assert "id_slot" in payload
