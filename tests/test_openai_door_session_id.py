"""``POST /v1/chat/completions`` records a caller's own correlation id
(2026-09-17) — the second half of "which model answered, and for which job".

Measured live: the fleet's per-job attribution needs `session_id` recorded on
`proxy_completions`, same as it already is on the enriched door (`enriched.py`)
and the legacy internal door (`legacy.py`). The OpenAI door never read it —
`docs/api.md` §1.1 listed it as travelling to the backend inside the payload,
unread — so every call through this door recorded `session_id = NULL` even
when the caller sent one.

The fix reads `session_id`/`turn_id` off the body and pops them (so they never
reach the backend as unknown fields), same treatment `timeout_s` already gets.
It does NOT widen what a caller can claim: `agent_id`/`caller_id`/`priority`
still come from nowhere but the resolved principal — a correlation id lets a
caller tag its OWN request, never relabel who it is.

Pins:
  * a body-supplied `session_id` is recorded on the completion row;
  * a body-supplied `turn_id` is recorded on the completion row;
  * a body-supplied `agent_id`/`caller_id` does NOT override the principal —
    the security-relevant half, since this door is reachable off-box;
  * a non-string or oversized `session_id`/`turn_id` is dropped, never raised,
    fail-open like the rest of this door's untrusted-input handling;
  * neither ever reaches the backend payload (popped, like `timeout_s`);
  * the enriched door (`/rs/v1/chat`) is unaffected.
"""

from __future__ import annotations

import json

import pytest

from roadstead.backend import BackendResponse
from roadstead.config import ProxyConfig
from roadstead.http_handlers import _MAX_CORRELATION_ID_LEN
from roadstead.service import ProxyService


# A docker-bridge IP → ACL "internal" identity (mirrors test_openai_frontdoor.py).
class _FakeRequest:
    class _Client:
        host = "172.16.0.5"

    client = _Client()
    headers: dict = {}


_COMPLETION = {
    "id": "chatcmpl-test", "object": "chat.completion", "created": 1,
    "model": "tier3",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}


def _openai_body(*, extra: dict | None = None) -> dict:
    body = {
        "model": "tier3",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 16,
    }
    if extra:
        body.update(extra)
    return body


async def _make_service(db_path, *, seen_payloads: list | None = None) -> ProxyService:
    svc = ProxyService(ProxyConfig(queue_db_path=str(db_path)))

    async def fake_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        if seen_payloads is not None:
            seen_payloads.append(payload)
        return BackendResponse(status_code=200, body=_COMPLETION,
                               duration_s=0.01, input_tokens=5, output_tokens=2)

    async def _none(*a, **k):
        return None

    svc._backend.call = fake_call
    svc._backend.probe_vllm_capacity = _none
    svc._backend.probe_props = _none
    svc._backend.probe_models = _none
    await svc.startup()
    return svc


def _completion_row(svc, cols: str):
    svc._queue_db.flush()
    row = svc._queue_db._conn.execute(
        f"SELECT {cols} FROM proxy_completions",
    ).fetchone()
    assert row is not None, "no proxy_completions row was written"
    return row


# --------------------------------------------------------------------------- #
# the correlation fields ARE recorded
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_body_supplied_session_id_is_recorded(tmp_path):
    svc = await _make_service(tmp_path / "q.db")
    try:
        resp = await svc.handle_openai_chat(
            _openai_body(extra={"session_id": "corr-abc-123"}), _FakeRequest())
        assert resp.status_code == 200
        (session_id,) = _completion_row(svc, "session_id")
        assert session_id == "corr-abc-123"
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_body_supplied_turn_id_is_recorded(tmp_path):
    svc = await _make_service(tmp_path / "q.db")
    try:
        resp = await svc.handle_openai_chat(
            _openai_body(extra={"turn_id": "turn-7"}), _FakeRequest())
        assert resp.status_code == 200
        (turn_id,) = _completion_row(svc, "turn_id")
        assert turn_id == "turn-7"
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_correlation_ids_never_reach_the_backend_payload(tmp_path):
    """Popped like `timeout_s` — proxy metadata, not part of the OpenAI
    request, and a strict backend could reject the unknown key."""
    seen: list = []
    svc = await _make_service(tmp_path / "q.db", seen_payloads=seen)
    try:
        await svc.handle_openai_chat(
            _openai_body(extra={"session_id": "corr-abc", "turn_id": "t-1"}),
            _FakeRequest())
        assert len(seen) == 1
        assert "session_id" not in seen[0]
        assert "turn_id" not in seen[0]
    finally:
        await svc.shutdown()


# --------------------------------------------------------------------------- #
# the security-relevant half: identity does NOT widen
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_body_supplied_agent_id_and_caller_id_do_not_override_principal(tmp_path):
    svc = await _make_service(tmp_path / "q.db")
    try:
        resp = await svc.handle_openai_chat(
            _openai_body(extra={
                "agent_id": "attacker", "caller_id": "attacker",
                "session_id": "corr-xyz",
            }),
            _FakeRequest())
        assert resp.status_code == 200
        agent_id, caller_id, session_id = _completion_row(
            svc, "agent_id, caller_id, session_id")
        # "internal" is what the fake docker-bridge address resolves to
        # (test_openai_frontdoor.py's _FakeRequest, same address here).
        assert agent_id == "internal"
        assert caller_id == "internal"
        # The correlation id is still honoured — only identity is locked.
        assert session_id == "corr-xyz"
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_body_supplied_priority_does_not_override_principal(tmp_path):
    """Same claim, the third identity-adjacent field: a body `priority` must
    not become the recorded priority any more than it did before this change."""
    svc = await _make_service(tmp_path / "q.db")
    try:
        resp = await svc.handle_openai_chat(
            _openai_body(extra={"priority": "P0_REALTIME"}), _FakeRequest())
        assert resp.status_code == 200
        (priority,) = _completion_row(svc, "priority")
        # P0_REALTIME would be 0; the default/configured band for an
        # unregistered caller is NOT that, whatever it resolves to.
        from roadstead.config import LLMPriority
        assert priority != int(LLMPriority.P0_REALTIME)
    finally:
        await svc.shutdown()


# --------------------------------------------------------------------------- #
# untrusted input: fail-open
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_non_string_session_id_is_dropped_not_raised(tmp_path):
    svc = await _make_service(tmp_path / "q.db")
    try:
        resp = await svc.handle_openai_chat(
            _openai_body(extra={"session_id": 12345}), _FakeRequest())
        assert resp.status_code == 200
        (session_id,) = _completion_row(svc, "session_id")
        assert session_id is None
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_list_valued_session_id_is_dropped_not_raised(tmp_path):
    """A list is a plausible 'absurd value' (accidentally sending a whole
    conversation's worth of ids) — must not reach sqlite as a bind parameter."""
    svc = await _make_service(tmp_path / "q.db")
    try:
        resp = await svc.handle_openai_chat(
            _openai_body(extra={"session_id": ["a", "b"]}), _FakeRequest())
        assert resp.status_code == 200
        (session_id,) = _completion_row(svc, "session_id")
        assert session_id is None
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_oversized_session_id_is_dropped_not_raised(tmp_path):
    svc = await _make_service(tmp_path / "q.db")
    try:
        too_long = "x" * (_MAX_CORRELATION_ID_LEN + 1)
        resp = await svc.handle_openai_chat(
            _openai_body(extra={"session_id": too_long}), _FakeRequest())
        assert resp.status_code == 200
        (session_id,) = _completion_row(svc, "session_id")
        assert session_id is None
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_a_session_id_exactly_at_the_bound_is_kept(tmp_path):
    svc = await _make_service(tmp_path / "q.db")
    try:
        exactly = "x" * _MAX_CORRELATION_ID_LEN
        resp = await svc.handle_openai_chat(
            _openai_body(extra={"session_id": exactly}), _FakeRequest())
        assert resp.status_code == 200
        (session_id,) = _completion_row(svc, "session_id")
        assert session_id == exactly
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_empty_string_session_id_is_dropped_not_recorded_as_empty(tmp_path):
    svc = await _make_service(tmp_path / "q.db")
    try:
        resp = await svc.handle_openai_chat(
            _openai_body(extra={"session_id": "   "}), _FakeRequest())
        assert resp.status_code == 200
        (session_id,) = _completion_row(svc, "session_id")
        assert session_id is None
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_no_session_id_supplied_is_fine(tmp_path):
    """The overwhelmingly common case: no correlation id at all."""
    svc = await _make_service(tmp_path / "q.db")
    try:
        resp = await svc.handle_openai_chat(_openai_body(), _FakeRequest())
        assert resp.status_code == 200
        session_id, turn_id = _completion_row(svc, "session_id, turn_id")
        assert session_id is None and turn_id is None
    finally:
        await svc.shutdown()


# --------------------------------------------------------------------------- #
# the enriched door is unaffected
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_enriched_door_session_id_recording_is_unchanged(tmp_path):
    """`/rs/v1/chat` already read `session_id` before this change
    (`enriched.py`); this change touches only the OpenAI door's handler."""
    svc = await _make_service(tmp_path / "q.db")
    try:
        submit_body = {
            "agent_id": "a",
            "endpoint": "tier3",
            "priority": "P3_INGESTION",
            "call_site": "test",
            "payload_type": "chat_completion",
            "payload": _openai_body(),
            "session_id": "enriched-corr-1",
            "timeout_s": 10.0,
        }
        resp = await svc.handle_submit(submit_body, _FakeRequest())
        assert resp.status_code == 200
        env = json.loads(resp.body.decode())
        assert env["status"] == "ok"
        (session_id,) = _completion_row(svc, "session_id")
        assert session_id == "enriched-corr-1"
    finally:
        await svc.shutdown()
