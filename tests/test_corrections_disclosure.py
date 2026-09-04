"""Correction-layer rewrites are invisible per call — until now (audit P2,
2026-09-04).

The schema backstop (``correction.py``'s ``maybe_repair_schema``) and
``apply_json_object_guard`` silently rewrite a response/payload and only ever
surfaced the fact fleet-wide, on ``/v1/status`` counters — a caller receiving
an in-memory-repaired body had no way to know its content had been altered.
``Correction.corrections_applied`` is the one place that per-call answer is
computed; this file drives it end to end through a real ``ProxyService``:
``X-Roadstead-Corrected`` on the OpenAI door, ``corrections`` in the enriched
envelope, and neither on an untouched response.
"""
from __future__ import annotations

import json
import os

import pytest

from roadstead.backend import BackendResponse
from roadstead.config import ProxyConfig
from roadstead.enriched import ENRICHMENT_HEADERS, WIRE_ENRICHED, WIRE_OPENAI
from roadstead.service import ProxyService

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}},
          "required": ["a"]}


class _Req:
    class _Client:
        host = "127.0.0.1"

    client = _Client()
    headers: dict = {}


def _body(content: str) -> dict:
    return {
        "agent_id": "a", "endpoint": "tier2", "priority": "P1_TURN_SUPPORT",
        "call_site": "t", "payload_type": "chat_completion",
        "payload": {
            "messages": [{"role": "user", "content": "give me an object"}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"schema": SCHEMA}},
        },
        "timeout_s": 5.0,
    }


def _backend_returning(content: str):
    async def call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"content": content},
                               "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 5, "completion_tokens": 3}},
            duration_s=0.01, input_tokens=5, output_tokens=3, finish_reason="stop")
    return call


@pytest.fixture(autouse=True)
def _schema_backstop_enforced(monkeypatch):
    monkeypatch.setenv("ROADSTEAD_PROXY_SCHEMA_BACKSTOP", "1")
    monkeypatch.delenv("ROADSTEAD_PROXY_SCHEMA_BACKSTOP_SHADOW", raising=False)


@pytest.mark.asyncio
async def test_a_repaired_response_carries_the_header_on_the_openai_door():
    svc = ProxyService(ProxyConfig())
    svc._backend.call = _backend_returning('here you go: {"a": 5} — done')
    await svc.startup()
    try:
        resp = await svc.handle_submit(_body("unused"), _Req(), wire=WIRE_OPENAI)
    finally:
        await svc.shutdown()
    assert resp.status_code == 200
    assert resp.headers[ENRICHMENT_HEADERS["corrected"]] == "schema_repaired"


@pytest.mark.asyncio
async def test_a_repaired_response_carries_the_field_on_the_enriched_door():
    svc = ProxyService(ProxyConfig())
    svc._backend.call = _backend_returning('here you go: {"a": 5} — done')
    await svc.startup()
    try:
        resp = await svc.handle_submit(_body("unused"), _Req(), wire=WIRE_ENRICHED)
    finally:
        await svc.shutdown()
    assert resp.status_code == 200
    body = json.loads(resp.body)
    assert body["corrections"] == ["schema_repaired"]
    # The repaired content actually reached the caller — the disclosure names
    # a real rewrite, not a phantom one.
    assert body["response"]["choices"][0]["message"]["content"] == '{"a": 5}'


@pytest.mark.asyncio
async def test_a_clean_response_carries_neither():
    svc = ProxyService(ProxyConfig())
    svc._backend.call = _backend_returning('{"a": 5}')
    await svc.startup()
    try:
        openai_resp = await svc.handle_submit(_body("unused"), _Req(), wire=WIRE_OPENAI)
        assert ENRICHMENT_HEADERS["corrected"] not in openai_resp.headers

        enriched_resp = await svc.handle_submit(_body("unused"), _Req(), wire=WIRE_ENRICHED)
    finally:
        await svc.shutdown()
    body = json.loads(enriched_resp.body)
    assert body["corrections"] == []
