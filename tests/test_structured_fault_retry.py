"""One-shot retry for a structured request that died mid-generation (2026-09-25).

A structured-output request can hit an engine-side desync between the grammar
matcher and speculative decoding: after a draft ROLLBACK the matcher rejects
tokens that were sampled under its own mask, the engine terminates the request
("grammar rejected tokens ... Terminating request") and answers a generic
HTTP 500 InternalServerError. It is a sampling-path fault — a fresh generation
almost always takes a different path — so the proxy retries it ONCE. Pins:

  - structured request + engine 500 → one retry → ok, counter = 1;
  - a second 500 on the retry surfaces (never a loop);
  - an UNSTRUCTURED request with the same 500 is not retried;
  - a structured request with a non-500 BackendError is not retried;
  - the classifier ignores timeouts / unavailable (their own branches).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from roadstead.backend import BackendError, BackendResponse, BackendTimeout, BackendUnavailable
from roadstead.config import ProxyConfig
from roadstead.correction import Correction
from roadstead.service import ProxyService


class _Req:
    class _Client:
        host = "127.0.0.1"

    client = _Client()
    headers: dict = {}


def _body(structured=True, timeout_s=20.0):
    payload = {"messages": [{"role": "user", "content": "x"}], "max_tokens": 400, "temperature": 0.2}
    if structured:
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "t", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}}
    return {"agent_id": "kv4", "endpoint": "tier3", "priority": "P2_POST_TURN",
            "call_site": "kv4.judge", "payload_type": "chat_completion",
            "payload": payload, "timeout_s": timeout_s}


def _ok():
    return BackendResponse(
        status_code=200,
        body={"choices": [{"message": {"content": '{"a": "b"}'}, "finish_reason": "stop"}],
              "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        duration_s=0.05, input_tokens=10, output_tokens=5, finish_reason="stop")


def _engine_500():
    return BackendError(500, 'backend error 500: {"error":{"message":"Internal server error",'
                             '"type":"InternalServerError","param":null,"code":500}}')


async def _run(body, behaviour):
    svc = ProxyService(ProxyConfig())
    calls = {"n": 0}

    async def fake(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        calls["n"] += 1
        return behaviour(calls["n"])

    svc._backend.call = fake
    await svc.startup()
    try:
        resp = await asyncio.wait_for(svc.handle_submit(body, _Req()), timeout=15.0)
        return resp, calls["n"], svc
    finally:
        await svc.shutdown()


def _raise(exc):
    raise exc


@pytest.mark.asyncio
async def test_structured_engine_500_is_retried_once_and_recovers():
    resp, n, svc = await _run(_body(), lambda i: _raise(_engine_500()) if i == 1 else _ok())
    assert resp.status_code == 200 and json.loads(resp.body)["status"] == "ok"
    assert n == 2
    assert svc._structured_fault_retries == 1


@pytest.mark.asyncio
async def test_second_engine_500_surfaces_never_loops():
    resp, n, svc = await _run(_body(), lambda i: _raise(_engine_500()))
    assert resp.status_code >= 500
    assert n == 2, "exactly one retry, then surface"
    assert svc._structured_fault_retries == 1


@pytest.mark.asyncio
async def test_unstructured_request_is_not_retried():
    resp, n, svc = await _run(_body(structured=False), lambda i: _raise(_engine_500()))
    assert resp.status_code >= 500
    assert n == 1
    assert svc._structured_fault_retries == 0


@pytest.mark.asyncio
async def test_structured_non_500_is_not_retried():
    resp, n, svc = await _run(_body(), lambda i: _raise(BackendError(400, "bad schema")))
    assert n == 1
    assert svc._structured_fault_retries == 0


def test_classifier_scope():
    assert Correction.is_structured_generation_fault(_engine_500())
    assert not Correction.is_structured_generation_fault(BackendError(502, "Internal server error"))
    assert not Correction.is_structured_generation_fault(BackendTimeout())
    assert not Correction.is_structured_generation_fault(BackendUnavailable())
    assert not Correction.is_structured_generation_fault(ValueError("Internal server error"))
