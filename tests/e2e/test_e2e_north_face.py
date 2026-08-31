"""North-face adversarial E2E: fire every hostile / malformed CALLER request in
the WU3 corpus at the proxy's real front door and assert it's handled CLEANLY.

The two load-bearing invariants (the Phase-T bar), mirrored from
``test_e2e_adversarial.py``:
  1. the proxy never returns the generic unhandled-500 backstop
     ("internal proxy error") — every hostile request is HANDLED, not crashed;
  2. after the call settles, in-flight returns to 0 (no leaked slot).

Cases that pin a specific typed status (``expect_status``) additionally assert
it. The duplicate-storm case fires N identical requests concurrently and asserts
every one is handled with no residual leak.
"""

from __future__ import annotations

import asyncio

import pytest

# Exhaustive ProxyService-spinning adversarial matrix — deselected from the
# per-ship in_container_tollgate via `-m 'not heavy'` (see pyproject `heavy`).
pytestmark = pytest.mark.heavy

from tests.corpus.north_face import NORTH_FACE_CASES, NorthFaceCase


_UNHANDLED_500_MARKER = "internal proxy error"
_JSON_HEADERS = {"content-type": "application/json"}


async def _assert_no_leak(proxy, timeout: float = 2.0) -> None:
    """In-flight must return to zero. Poll briefly — streaming/coerced paths can
    resolve a beat after the response is returned."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if proxy.total_in_flight() == 0:
            return
        await asyncio.sleep(0.02)
    assert proxy.total_in_flight() == 0, "slot leak: in-flight did not return to 0"


def _assert_handled(resp) -> None:
    """Never the generic unhandled-500 backstop — the fault was HANDLED."""
    body_text = resp.text or ""
    assert not (resp.status_code == 500 and _UNHANDLED_500_MARKER in body_text), (
        f"unhandled crash: status={resp.status_code} body={body_text[:200]!r}")


async def _fire(proxy, case: NorthFaceCase):
    """POST one copy of the case at the proxy front door."""
    if case.raw_body is not None:
        return await proxy.client.post(
            case.path, content=case.raw_body, headers=_JSON_HEADERS)
    return await proxy.client.post(case.path, json=case.body)


@pytest.mark.parametrize("case", NORTH_FACE_CASES, ids=[c.name for c in NORTH_FACE_CASES])
async def test_north_face_case_handled_and_no_leak(proxy, case: NorthFaceCase):
    if case.concurrency > 1:
        responses = await asyncio.gather(
            *[_fire(proxy, case) for _ in range(case.concurrency)],
            return_exceptions=True)
        for r in responses:
            assert not isinstance(r, Exception), f"request raised: {r!r}"
            _assert_handled(r)
            if case.expect_status is not None:
                assert r.status_code == case.expect_status, (
                    f"{case.name}: expected {case.expect_status}, "
                    f"got {r.status_code} body={r.text[:200]!r}")
    else:
        resp = await _fire(proxy, case)
        _assert_handled(resp)
        if case.expect_status is not None:
            assert resp.status_code == case.expect_status, (
                f"{case.name}: expected {case.expect_status}, "
                f"got {resp.status_code} body={resp.text[:200]!r}")

    await _assert_no_leak(proxy)


# --- payload-shape gate (Phase-1 north-face hardening) -----------------------
# The flipped fuzz guard (test_north_face_500_gaps_phase1) covers the OpenAI
# door + the non-dict-element branch. These pin the OTHER door (internal
# /v1/submit envelope) and the "messages is not a list" branch, and assert the
# typed taxonomy code — so a regression that only broke one door is caught.

@pytest.mark.parametrize("bad_messages", [
    "not a list",              # messages is a bare string
    123,                       # messages is a scalar
    ["hi", "there"],           # list of non-dict elements
    [{"role": "user", "content": "ok"}, 7],  # one bad element among good
])
async def test_submit_door_rejects_malformed_messages(proxy, bad_messages):
    resp = await proxy.client.post("/v1/submit", json={
        "agent_id": "t", "endpoint": "chat", "priority": "P3_INGESTION",
        "call_site": "north_face", "payload_type": "chat_completion",
        "payload": {"messages": bad_messages, "max_tokens": 8},
    })
    _assert_handled(resp)
    assert resp.status_code == 400, f"got {resp.status_code} body={resp.text[:200]!r}"
    assert resp.json().get("code") == "invalid_messages"
    await _assert_no_leak(proxy)


async def test_submit_door_accepts_valid_and_absent_messages(proxy):
    # Positive control: the gate must NOT over-reject. A well-formed messages
    # list — and a payload with NO messages key at all — pass the gate.
    for payload in ({"messages": [{"role": "user", "content": "hi"}],
                     "max_tokens": 8},
                    {"prompt": "hi", "max_tokens": 8}):
        resp = await proxy.client.post("/v1/submit", json={
            "agent_id": "t", "endpoint": "chat", "priority": "P3_INGESTION",
            "call_site": "north_face", "payload_type": "chat_completion",
            "payload": payload,
        })
        _assert_handled(resp)
        assert resp.status_code != 400, (
            f"over-rejected valid payload: {resp.status_code} "
            f"{resp.text[:200]!r}")
    await _assert_no_leak(proxy)
