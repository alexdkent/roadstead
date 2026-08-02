"""The detector is REACHED, and its counters reach /v1/status + /metrics.

CLAUDE.md, on the five ways a green suite lies: "a module can be written,
unit-tested, and never called. Tests prove a unit works; only an end-to-end
journey proves it is REACHED." The unit tests next door bind ``Correction``
methods to a mock ``self``; these drive the REAL ``ProxyService`` with a fake
backend that returns exactly what tier3 returned on 2026-07-31 — `{}` with
finish_reason=stop and no error — and assert the signal comes out of the real
HTTP surfaces.
"""

from __future__ import annotations

import json

import pytest

from originfleet.llmproxy import observability as obs
from originfleet.llmproxy.backend import BackendResponse
from originfleet.llmproxy.config import ProxyConfig
from originfleet.llmproxy.service import ProxyService


class _LoopbackRequest:
    class _Client:
        host = "127.0.0.1"

    client = _Client()
    headers: dict = {}


async def _none(*a, **k):
    return None


def _submit(content, response_format):
    payload = {"messages": [{"role": "user", "content": "critique this draft"}],
               "max_tokens": 400}
    if response_format is not None:
        payload["response_format"] = response_format
    return {
        "agent_id": "sidekick", "endpoint": "llama-thinker",
        "priority": "P3_INGESTION", "call_site": "auto_approve.critic",
        "payload_type": "chat_completion", "payload": payload,
        "timeout_s": 10.0,
    }


def _backend_returning(content):
    async def call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0,
                   **kw):
        return BackendResponse(
            status_code=200,
            body={"id": "x", "object": "chat.completion",
                  "choices": [{"index": 0, "finish_reason": "stop",
                               "message": {"role": "assistant",
                                           "content": content}}],
                  "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                            "total_tokens": 12}},
            duration_s=0.01,
            finish_reason="stop", input_tokens=10, output_tokens=2)
    return call


async def _svc(content):
    svc = ProxyService(ProxyConfig())
    svc._backend.probe_vllm_capacity = _none
    svc._backend.probe_props = _none
    svc._backend.probe_models = _none
    svc._backend.call = _backend_returning(content)
    svc._cache.cache_key = lambda endpoint, payload: None
    await svc.startup()
    return svc


@pytest.mark.asyncio
async def test_a_real_empty_brace_response_is_detected_end_to_end():
    """The 2026-07-31 shape, all the way through the real dispatch path."""
    svc = await _svc("{}")
    try:
        resp = await svc.handle_submit(
            _submit("{}", {"type": "json_object"}), _LoopbackRequest())
        body = json.loads(resp.body)
        # The response is UNTOUCHED — this is telemetry, not a gate.
        assert body["status"] == "ok"
        assert body["response"]["choices"][0]["message"]["content"] == "{}"
        # …and the proxy noticed.
        assert svc._state.structured_empty_total == 1
        assert svc._state.structured_empty_by_call_site == {
            "auto_approve.critic": 1}
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_a_real_healthy_response_is_not_detected_end_to_end():
    svc = await _svc('{"voice_match": true, "overall": 61}')
    try:
        await svc.handle_submit(
            _submit(None, {"type": "json_object"}), _LoopbackRequest())
        assert svc._state.structured_empty_total == 0
        # …but it IS counted as a healthy denominator sample, which is what
        # makes the rate a rate.
        assert svc._state.structured_empty_window["thinker"]
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_the_counters_surface_on_v1_status():
    svc = await _svc("{}")
    try:
        await svc.handle_submit(
            _submit("{}", {"type": "json_object"}), _LoopbackRequest())
        status = json.loads((await svc.handle_status(_LoopbackRequest())).body)
        rel = status["reliability"]
        assert rel["structured_empty_total"] == 1
        assert rel["structured_empty_by_call_site"]["auto_approve.critic"] == 1
        assert rel["structured_empty_rate"]["thinker"]["empty"] == 1
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_the_rate_gauge_is_withheld_until_the_endpoint_is_judgeable():
    """A single sample must not publish a `rate 1.0` gauge that reads as a
    100%-broken endpoint on a dashboard. Below the sample floor there is no
    gauge at all — the same rule the alert obeys."""
    svc = await _svc("{}")
    try:
        await svc.handle_submit(
            _submit("{}", {"type": "json_object"}), _LoopbackRequest())
        text = (await svc.handle_prometheus_metrics(_LoopbackRequest())).body.decode()
        assert "llmproxy_structured_empty_rate" not in text
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_the_rate_gauge_is_published_once_the_floor_is_met():
    """…and it IS published once the endpoint is judgeable, so VictoriaMetrics
    can build the history the standing alert is the page for."""
    svc = await _svc("{}")
    try:
        for _ in range(obs.STRUCTURED_EMPTY_MIN_SAMPLES):
            await svc.handle_submit(
                _submit("{}", {"type": "json_object"}), _LoopbackRequest())
        text = (await svc.handle_prometheus_metrics(_LoopbackRequest())).body.decode()
        # render_prometheus normalizes 1.0 -> "1", so match the label+prefix.
        assert 'llmproxy_structured_empty_rate{endpoint="thinker"} 1' in text
        assert 'llmproxy_structured_samples_30m{endpoint="thinker"} 20' in text
        # …and only for the endpoint that has traffic.
        assert 'llmproxy_structured_empty_rate{endpoint="gemma"}' not in text
    finally:
        await svc.shutdown()
