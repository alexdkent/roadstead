"""M2 — structured error taxonomy: machine-readable ``code`` on every proxy
error envelope, with the LOAD-BEARING deferrable substrings preserved.

The fleet's deferral classifier (`framework.nexus_errors.is_deferrable_llm_error`)
sniffs error TEXT — these tests use the real classifier as the oracle so any
drift between proxy wording and classifier markers fails here, not in
production. Each error path asserts:
  1. the new ``code`` field (additive, machine-readable),
  2. the classifier's verdict on the message (deferrable vs surface).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from originfleet.framework.nexus_errors import is_deferrable_llm_error
from originfleet.llmproxy.backend import BackendError, BackendResponse
from originfleet.llmproxy.config import ProxyConfig
from originfleet.llmproxy.service import ProxyService


class _Req:
    class _Client:
        host = "127.0.0.1"

    client = _Client()
    headers: dict = {}


def _body(endpoint="llama-thinker", **payload_extra):
    return {
        "agent_id": "a", "endpoint": endpoint, "priority": "P3_INGESTION",
        "call_site": "t", "payload_type": "chat_completion",
        "payload": {"messages": [{"role": "user", "content": "x"}], **payload_extra},
        "timeout_s": 5.0,
    }


def _classifier_says_defer(message: str) -> bool:
    return is_deferrable_llm_error(ConnectionError(message))


@pytest.mark.asyncio
async def test_draining_is_deferrable_with_code():
    svc = ProxyService(ProxyConfig())
    svc._draining.set()
    resp = await svc.handle_submit(_body(), _Req())
    body = json.loads(resp.body)
    assert resp.status_code == 503
    assert body["code"] == "draining"
    assert _classifier_says_defer(body["error"])  # "backpressure" marker


@pytest.mark.asyncio
async def test_circuit_open_and_paused_are_deferrable_with_codes():
    svc = ProxyService(ProxyConfig())
    # Auto-circuit trip.
    svc._endpoint_health["thinker"]["healthy"] = False
    resp = await svc.handle_submit({**_body(), "priority": "P1_TURN_SUPPORT"}, _Req())
    body = json.loads(resp.body)
    assert resp.status_code == 503
    assert body["code"] == "circuit_open"
    assert _classifier_says_defer(body["error"])  # "circuit open" marker

    # Operator drain reads as draining, still deferrable.
    svc._endpoint_health["thinker"]["healthy"] = True
    svc._paused_endpoints.add("thinker")
    resp = await svc.handle_submit({**_body(), "priority": "P1_TURN_SUPPORT"}, _Req())
    body = json.loads(resp.body)
    assert resp.status_code == 503
    assert body["code"] == "draining"
    assert _classifier_says_defer(body["error"])  # "backpressure" marker


@pytest.mark.asyncio
async def test_backpressure_shed_is_deferrable_with_code():
    svc = ProxyService(ProxyConfig())
    svc._shed_depth = 0  # any queued depth sheds
    resp = await svc.handle_submit(_body(), _Req())
    body = json.loads(resp.body)
    assert resp.status_code == 429
    assert body["code"] == "backpressure"
    assert resp.headers.get("retry-after")
    assert _classifier_says_defer(body["error"])


@pytest.mark.asyncio
async def test_invalid_grammar_is_not_deferrable_and_coded():
    svc = ProxyService(ProxyConfig())
    resp = await svc.handle_submit(
        _body(grammar="not a grammar at all"), _Req())
    body = json.loads(resp.body)
    assert resp.status_code == 422
    assert body["code"] == "invalid_grammar"
    # Deterministic request error: the message must NOT trip the deferral
    # classifier (a defer-loop on a static grammar never converges).
    assert not _classifier_says_defer(body.get("detail", "") + body.get("error", ""))


@pytest.mark.asyncio
async def test_unknown_endpoint_enforce_404_not_deferrable():
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"unknown_endpoint_enforce": True})
    resp = await svc.handle_submit(_body(endpoint="qwen-composr-typo"), _Req())
    body = json.loads(resp.body)
    assert resp.status_code == 404
    assert body["code"] == "unknown_endpoint"
    assert not _classifier_says_defer(body["error"])


@pytest.mark.asyncio
async def test_backend_error_envelope_coded_and_deferrable_via_status():
    svc = ProxyService(ProxyConfig())

    async def failing_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        raise BackendError(400, "backend rejected the payload")

    svc._backend.call = failing_call
    await svc.startup()
    try:
        resp = await asyncio.wait_for(svc.handle_submit(_body(), _Req()), timeout=10.0)
        body = json.loads(resp.body)
        assert resp.status_code == 502
        assert body["status"] == "error"
        assert body["code"] == "backend_error"
        # The CLIENT-side mapping makes a 502 envelope deferrable via the
        # "LLM proxy error 502" prefix — pin that contract end to end.
        assert _classifier_says_defer(f"LLM proxy error 502: {json.dumps(body)}")
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_proxy_timeout_coded():
    svc = ProxyService(ProxyConfig())

    async def never_returns(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        # Slow enough to outlive the caller's 0.3s deadline, short enough that
        # shutdown's drain isn't held to its 30s straggler deadline.
        await asyncio.sleep(1.2)
        from originfleet.llmproxy.backend import BackendTimeout
        raise BackendTimeout("late")

    svc._backend.call = never_returns
    await svc.startup()
    try:
        resp = await asyncio.wait_for(
            svc.handle_submit({**_body(), "timeout_s": 0.3}, _Req()), timeout=10.0)
        body = json.loads(resp.body)
        assert resp.status_code == 504
        assert body["code"] == "proxy_timeout"
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_success_envelope_unchanged_no_code_key():
    """The ok envelope is a hard client contract — taxonomy must be additive
    on ERRORS only."""
    svc = ProxyService(ProxyConfig())

    async def ok_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            duration_s=0.01, input_tokens=1, output_tokens=1, finish_reason="stop")

    svc._backend.call = ok_call
    await svc.startup()
    try:
        resp = await asyncio.wait_for(svc.handle_submit(_body(), _Req()), timeout=10.0)
        body = json.loads(resp.body)
        assert resp.status_code == 200
        assert body["status"] == "ok"
        assert "code" not in body
        assert set(body) == {"status", "request_id", "queue_wait_ms",
                             "backend_latency_ms", "estimated_cost_ss", "response"}
    finally:
        await svc.shutdown()
