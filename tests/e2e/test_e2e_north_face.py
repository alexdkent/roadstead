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

from tests.llmproxy.corpus.north_face import NORTH_FACE_CASES, NorthFaceCase


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
