"""Structured content blank-run abort — the SYNC dispatch path
(`Lifecycle.execute_sync` + `BackendClientPool.call_watched`), driven through
`ProxyService.handle_submit` the same way `test_truncation_guard.py` drives
the pre-existing truncation/degeneration guards.

Pins: salvage (grammar-legal trailing whitespace after an already-complete
object), one re-dispatch recovered, one re-dispatch that blank-runs again
(degenerate error, non-"truncated structured output" wording), inert when the
endpoint declares nothing (byte-identical `.call` path), and an eligible
endpoint that still uses `.call` for an ineligible REQUEST (tools present).
"""
from __future__ import annotations

import asyncio
import json
import logging

import pytest

from roadstead.backend import BackendResponse, BackendStreamEvent
from roadstead.config import ProxyConfig
from roadstead.service import ProxyService

_SCHEMA_RF = {"type": "json_schema",
              "json_schema": {"name": "s", "schema": {"type": "object"}}}

_THRESHOLD = 40  # small so the fixtures below stay short and fast


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


def _payload(*, extra=None, max_tokens=64):
    p = {"messages": [{"role": "user", "content": "x"}],
         "max_tokens": max_tokens, "response_format": _SCHEMA_RF}
    if extra:
        p.update(extra)
    return p


def _chunk(content=None, finish=None):
    delta = {"content": content} if content is not None else {}
    parsed = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return BackendStreamEvent("chunk", json.dumps(parsed), parsed)


def _svc(*, armed=True):
    svc = ProxyService(ProxyConfig())
    if armed:
        svc._config.endpoints["tier3"].structured_blank_run_abort_chars = _THRESHOLD
    return svc


async def _drive(svc, payload):
    await svc.startup()
    try:
        resp = await asyncio.wait_for(
            svc.handle_submit(_body(payload), _Req()), timeout=15.0)
        return resp, json.loads(resp.body)
    finally:
        await svc.shutdown()


# --------------------------------------------------------------------------- #
# salvage — grammar-legal trailing whitespace after an already-complete object
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_abort_then_salvage_serves_a_normal_ok_response(caplog):
    """The blank run the detector caught was trailing whitespace AFTER a
    complete object — served as an ordinary success, finish_reason 'stop',
    never an error."""
    svc = _svc()

    async def fake_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        yield _chunk('{"a": 1}')
        yield _chunk(" " * (_THRESHOLD + 5))
        yield _chunk(finish="stop")  # never reached if the abort works
        yield BackendStreamEvent("done", "[DONE]")

    async def fail_call(*a, **kw):
        raise AssertionError("armed+eligible request must use call_watched, not call")

    svc._backend.stream = fake_stream
    svc._backend.call = fail_call
    with caplog.at_level(logging.WARNING):
        resp, result = await _drive(svc, _payload())
    assert resp.status_code == 200
    body = result["response"]
    assert body["choices"][0]["message"]["content"] == '{"a": 1}'
    assert body["choices"][0]["finish_reason"] == "stop"
    assert svc._state.structured_blank_runs_detected == 1
    assert svc._state.structured_blank_runs_salvaged == 1
    assert svc._state.structured_blank_runs_unrecovered == 0
    marker = [r for r in caplog.records
              if "ROADSTEAD_STRUCTURED_BLANK_RUN" in r.getMessage()]
    assert len(marker) == 1
    assert "action=salvaged" in marker[0].getMessage()


# --------------------------------------------------------------------------- #
# retry — recovered
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_abort_then_retry_recovers(caplog):
    """First attempt blank-runs on an UNSALVAGEABLE prefix; the one permitted
    re-dispatch comes back clean — served as an ordinary success, and BOTH
    `retried` and `recovered` are counted."""
    svc = _svc()
    attempts = {"n": 0}

    async def fake_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        attempts["n"] += 1
        if attempts["n"] == 1:
            yield _chunk('{"a": ')  # unparseable once stripped
            yield _chunk(" " * (_THRESHOLD + 5))
        else:
            yield _chunk('{"a": 1}')
            yield _chunk(finish="stop")
            yield BackendStreamEvent("done", "[DONE]")

    svc._backend.stream = fake_stream
    with caplog.at_level(logging.WARNING):
        resp, result = await _drive(svc, _payload())
    assert attempts["n"] == 2
    assert resp.status_code == 200
    assert result["response"]["choices"][0]["message"]["content"] == '{"a": 1}'
    assert svc._state.structured_blank_runs_detected == 1
    assert svc._state.structured_blank_runs_salvaged == 0
    assert svc._state.structured_blank_runs_retried == 1
    assert svc._state.structured_blank_runs_recovered == 1
    assert svc._state.structured_blank_runs_unrecovered == 0
    actions = [r.getMessage().split("action=")[1].split()[0]
               for r in caplog.records if "ROADSTEAD_STRUCTURED_BLANK_RUN" in r.getMessage()]
    assert actions == ["retrying", "recovered"]


# --------------------------------------------------------------------------- #
# retry — blank-runs again -> degenerate error, non-"truncated" wording
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_abort_then_retry_blank_runs_again_is_unrecovered(caplog):
    svc = _svc()
    attempts = {"n": 0}

    async def fake_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        attempts["n"] += 1
        yield _chunk('{"a": ')
        yield _chunk(" " * (_THRESHOLD + 5))

    svc._backend.stream = fake_stream
    with caplog.at_level(logging.WARNING):
        resp, result = await _drive(svc, _payload())
    assert attempts["n"] == 2  # exactly one retry — never more
    assert resp.status_code == 502
    assert "degenerate structured output" in result["error"]
    assert "arm=blank_run" in result["error"]
    # 🚨 THE load-bearing assertion: must not contain the substring a caller's
    # truncation-recovery path matches on to retry with MORE tokens.
    assert "truncated structured output" not in result["error"]
    assert svc._state.structured_blank_runs_detected == 2
    assert svc._state.structured_blank_runs_retried == 1
    assert svc._state.structured_blank_runs_recovered == 0
    assert svc._state.structured_blank_runs_unrecovered == 1
    actions = [r.getMessage().split("action=")[1].split()[0]
               for r in caplog.records if "ROADSTEAD_STRUCTURED_BLANK_RUN" in r.getMessage()]
    assert actions == ["retrying", "unrecovered"]


# --------------------------------------------------------------------------- #
# inert — 0/absent threshold, byte-identical `.call` path
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_inert_when_the_endpoint_declares_nothing():
    svc = _svc(armed=False)
    assert svc._config.endpoints["tier3"].structured_blank_run_abort_chars == 0

    async def fake_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"role": "assistant",
                                           "content": '{"a": 1}' + " " * 500},
                               "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 5, "completion_tokens": 5}},
            duration_s=0.01, input_tokens=5, output_tokens=5,
            finish_reason="stop")

    async def fail_stream(*a, **kw):
        raise AssertionError("an undeclared endpoint must dispatch via .call, never .stream")
        yield  # pragma: no cover — makes this a generator function

    svc._backend.call = fake_call
    svc._backend.stream = fail_stream
    resp, result = await _drive(svc, _payload())
    assert resp.status_code == 200
    assert svc._state.structured_blank_runs_detected == 0


# --------------------------------------------------------------------------- #
# ineligible request (tools present) — armed endpoint, `.call` path anyway
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_ineligible_request_with_tools_uses_call_not_call_watched():
    svc = _svc(armed=True)

    async def fake_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"role": "assistant", "content": '{"a": 1}'},
                               "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
            duration_s=0.01, input_tokens=5, output_tokens=2,
            finish_reason="stop")

    async def fail_stream(*a, **kw):
        raise AssertionError("a request carrying tools must not be watched")
        yield  # pragma: no cover

    svc._backend.call = fake_call
    svc._backend.stream = fail_stream
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    resp, result = await _drive(svc, _payload(extra={"tools": tools}))
    assert resp.status_code == 200
    assert svc._state.structured_blank_runs_detected == 0
