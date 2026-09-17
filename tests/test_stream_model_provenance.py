"""Streaming completions must name which model served them (2026-09-17).

``record_completion`` persists ``response_json`` on every completion, but
``execute_streaming``'s terminal calls never passed a ``response_body`` —
only the non-streaming path did (``resp.body``, the backend's own reply).
Measured fleet-wide: 264/264 streaming ``proxy_completions`` rows have
``response_json IS NULL``, vs 0/1332 non-streaming rows. A downstream reader
(the fleet's fix-validator) cannot tell a healthy ``tier3`` answer from a
silent reroute for any streaming caller — which is every ``dsh`` job, since
``dsh`` always sends ``stream: true``.

The fix captures the model each backend chunk itself echoes in its top-level
``model`` field (evidence of what actually served), falling back to the
endpoint's declared ``effective_model_id`` only when no chunk carried one.
``model_source`` distinguishes the two so a reader can tell a proven answer
from a belief.

Pins:
  * A streaming call whose backend echoes ``model`` in a chunk records that
    model with ``model_source: backend_echo``.
  * A streaming call whose backend names no model anywhere records the
    endpoint's configured model with ``model_source: endpoint_config`` —
    never a raise, never a NULL row.
  * A non-streaming call is unchanged: ``response_json`` is still the whole
    backend body, verbatim, no ``model_source`` key added.
  * A stream that fails before any chunk arrives (cancelled/error/timeout)
    still records a sane, config-sourced body rather than leaving the row
    unprovable.
"""

from __future__ import annotations

import json

import pytest

from roadstead.backend import BackendResponse, BackendStreamEvent
from roadstead.config import ProxyConfig
from roadstead.service import ProxyService

# --------------------------------------------------------------------------- #
# harness helpers (mirrors tests/test_truncation_guard.py)
# --------------------------------------------------------------------------- #


class _Req:
    class _Client:
        host = "127.0.0.1"

    client = _Client()
    headers: dict = {}


def _body(payload, timeout_s=15.0):
    return {
        "agent_id": "kv4", "endpoint": "tier3", "priority": "P3_INGESTION",
        "call_site": "kv4.judge", "payload_type": "chat_completion",
        "payload": payload, "timeout_s": timeout_s,
    }


def _payload(content="x", *, stream=False, max_tokens=64):
    p = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens}
    if stream:
        p["stream"] = True
    return p


def _chunk(content=None, finish=None, model=None):
    delta = {"content": content} if content is not None else {}
    obj = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if model is not None:
        obj["model"] = model
    return json.dumps(obj)


def _svc_stream(pieces, finish, *, model=None, db_path):
    """A proxy whose backend streams `pieces` then a finish chunk. The FIRST
    piece carries `model` in its chunk when one is given — mirrors a real
    llama-server/vLLM stream, where every chunk (including the first) already
    carries the top-level `model` field.

    ``db_path`` is required (not the default empty-string config): an unset
    ``queue_db_path`` means ``PersistentQueue`` never opens a connection at
    all, so there would be no ``proxy_completions`` row to read back."""
    svc = ProxyService(ProxyConfig(queue_db_path=str(db_path)))

    async def fake_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        for i, piece in enumerate(pieces):
            text = _chunk(piece, model=model if i == 0 else None)
            yield BackendStreamEvent("chunk", text, json.loads(text))
        final = _chunk(finish=finish)
        yield BackendStreamEvent("chunk", final, json.loads(final))
        yield BackendStreamEvent("done", "[DONE]")

    svc._backend.stream = fake_stream
    return svc


def _svc_sync(content, finish, output_tokens=16, *, db_path):
    svc = ProxyService(ProxyConfig(queue_db_path=str(db_path)))

    async def fake_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"role": "assistant", "content": content},
                               "finish_reason": finish}],
                  "usage": {"prompt_tokens": 5, "completion_tokens": output_tokens}},
            duration_s=0.01, input_tokens=5, output_tokens=output_tokens,
            finish_reason=finish)

    svc._backend.call = fake_call
    return svc


async def _collect_events(resp, timeout: float = 5.0) -> list[dict]:
    import asyncio
    events: list[dict] = []

    async def _drain():
        async for chunk in resp.body_iterator:
            text = chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk
            for part in text.split("\n\n"):
                part = part.strip()
                if part.startswith("data: "):
                    events.append(json.loads(part[len("data: "):]))

    await asyncio.wait_for(_drain(), timeout=timeout)
    return events


def _completion_row(svc):
    """The single completion row this test wrote (one request per test).
    Writes go through the async writer thread (``PersistentQueue._w``), so
    ``flush()`` first or a fast test can read before the write lands. Must
    also run BEFORE ``svc.shutdown()`` — it closes the queue_db connection,
    and the fixture's db file has no other reader."""
    svc._queue_db.flush()
    row = svc._queue_db._conn.execute(
        "SELECT response_json, status FROM proxy_completions",
    ).fetchone()
    assert row is not None, "no proxy_completions row was written"
    return row


async def _drive_stream(svc, payload):
    """Runs the request, reads back its completion row, THEN shuts down —
    ordering matters, see ``_completion_row``."""
    await svc.startup()
    try:
        resp = await svc.handle_submit(_body(payload), _Req())
        events = await _collect_events(resp)
        row = _completion_row(svc)
        return events, row
    finally:
        await svc.shutdown()


async def _drive_sync(svc, payload):
    await svc.startup()
    try:
        import asyncio
        resp = await asyncio.wait_for(
            svc.handle_submit(_body(payload), _Req()), timeout=15.0)
        row = _completion_row(svc)
        return resp, json.loads(resp.body), row
    finally:
        await svc.shutdown()


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_streaming_completion_records_the_backend_echoed_model(tmp_path):
    svc = _svc_stream(["hello"], "stop", model="tier3-instance-42", db_path=tmp_path / "q.db")
    _events, (response_json, status) = await _drive_stream(svc, _payload(stream=True))

    assert status == "ok"
    assert response_json is not None, "response_json is NULL — the defect this test guards"
    body = json.loads(response_json)
    assert body == {"model": "tier3-instance-42", "model_source": "backend_echo"}


@pytest.mark.asyncio
async def test_streaming_completion_falls_back_to_endpoint_config_when_no_chunk_names_a_model(tmp_path):
    svc = _svc_stream(["hello"], "stop", db_path=tmp_path / "q.db")  # no chunk carries `model`
    _events, (response_json, status) = await _drive_stream(svc, _payload(stream=True))

    assert status == "ok"
    assert response_json is not None
    body = json.loads(response_json)
    # The example catalog's tier3 endpoint has no discovered served_model_id,
    # so effective_model_id falls back to its role, "tier3".
    assert body == {"model": "tier3", "model_source": "endpoint_config"}


@pytest.mark.asyncio
async def test_non_streaming_completion_response_body_is_unchanged(tmp_path):
    svc = _svc_sync("hello there", "stop", db_path=tmp_path / "q.db")
    resp, result, (response_json, status) = await _drive_sync(svc, _payload())

    assert status == "ok"
    body = json.loads(response_json)
    # Full backend body, verbatim — no model_source wrapper was introduced.
    assert "model_source" not in body
    assert body["choices"][0]["message"]["content"] == "hello there"


@pytest.mark.asyncio
async def test_streaming_error_before_any_chunk_still_records_a_sane_model(tmp_path):
    """A backend that raises before emitting a single chunk (e.g. connection
    refused) must still record a completion row with a model tag — fail-open,
    config-sourced, never a raise and never silently NULL."""
    svc = ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db")))

    async def fake_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        raise RuntimeError("synthetic backend connection failure")
        yield  # pragma: no cover — makes this an async generator

    svc._backend.stream = fake_stream

    events, (response_json, status) = await _drive_stream(svc, _payload(stream=True))
    assert any(e.get("error") for e in events if isinstance(e, dict))

    assert status == "error"
    assert response_json is not None
    body = json.loads(response_json)
    assert body == {"model": "tier3", "model_source": "endpoint_config"}
