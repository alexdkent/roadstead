"""Structured content blank-run abort — the STREAMING dispatch path
(`Lifecycle.execute_streaming`). Chunks are already relayed to the caller by
the time the guard fires, so there is no salvage and no re-dispatch here —
only an abort, reusing the reasoning-loop-break pattern (set a flag, raise to
the except block, name the abort_reason).
"""
from __future__ import annotations

import asyncio
import json

import pytest

from roadstead.backend import BackendStreamEvent
from roadstead.config import ProxyConfig
from roadstead.enriched import WIRE_ENRICHED
from roadstead.service import ProxyService

_SCHEMA_RF = {"type": "json_schema",
              "json_schema": {"name": "s", "schema": {"type": "object"}}}

_THRESHOLD = 40


class _FakeRequest:
    class _Client:
        host = "172.16.0.5"

    client = _Client()
    headers: dict = {}


async def _none(*a, **k):
    return None


def _stub_probes(svc: ProxyService) -> None:
    svc._backend.probe_vllm_capacity = _none
    svc._backend.probe_props = _none
    svc._backend.probe_models = _none


def _body(*, extra=None):
    p = {"messages": [{"role": "user", "content": "x"}],
         "response_format": _SCHEMA_RF, "stream": True}
    if extra:
        p.update(extra)
    return {
        "agent_id": "a", "endpoint": "tier3", "priority": "P3_INGESTION",
        "call_site": "kv4.judge", "payload_type": "chat_completion",
        "payload": p,
    }


def _chunk(content=None, finish=None):
    delta = {"content": content} if content is not None else {}
    return json.dumps({"choices": [{"index": 0, "delta": delta,
                                    "finish_reason": finish}]})


def _capture_timeout_events(svc: ProxyService) -> list[dict]:
    """Spy on `record_timeout_event` — the client-visible SSE frame carries
    the error MESSAGE, never the machine-readable `abort_reason`, which is
    this call's own kwarg."""
    captured: list[dict] = []
    orig = svc._lifecycle.record_timeout_event

    def tap(req, **kw):
        captured.append(kw)
        return orig(req, **kw)

    svc._lifecycle.record_timeout_event = tap
    return captured


async def _drain(resp, timeout_s: float = 10.0) -> list[dict]:
    events: list[dict] = []

    async def go():
        async for ch in resp.body_iterator:
            text = ch.decode() if isinstance(ch, (bytes, bytearray)) else ch
            for part in text.split("\n\n"):
                part = part.strip()
                if part.startswith("data: "):
                    events.append(json.loads(part[len("data: "):]))
    await asyncio.wait_for(go(), timeout=timeout_s)
    return events


def _svc(*, armed=True):
    svc = ProxyService(ProxyConfig())
    if armed:
        svc._config.endpoints["tier3"].structured_blank_run_abort_chars = _THRESHOLD
    _stub_probes(svc)
    return svc


@pytest.mark.asyncio
async def test_streaming_aborts_with_structured_blank_run_reason():
    svc = _svc()
    calls = {"n": 0}

    async def fake_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        calls["n"] += 1
        yield BackendStreamEvent("chunk", _chunk('{"a": '),
                                 json.loads(_chunk('{"a": ')))
        blank = " " * (_THRESHOLD + 5)
        yield BackendStreamEvent("chunk", _chunk(blank), json.loads(_chunk(blank)))
        # Unreachable if the abort works — a real backend would keep going.
        yield BackendStreamEvent("chunk", _chunk(finish="stop"),
                                 json.loads(_chunk(finish="stop")))
        yield BackendStreamEvent("done", "[DONE]")

    svc._backend.stream = fake_stream
    events = _capture_timeout_events(svc)
    await svc.startup()
    try:
        resp = await svc.handle_submit(_body(), _FakeRequest())
        frames = await _drain(resp)
    finally:
        await svc.shutdown()

    # Never a clean 'done' — the last frame is the error.
    assert frames[-1]["type"] == "error"
    err = frames[-1]["error"]
    assert "blank" in err and "whitespace run" in err
    # Non-deferrable wording (§2.2): a streaming caller has already received
    # every relayed chunk, so it must not read as "retry me".
    assert "backpressure" not in err
    assert "truncated structured output" not in err

    assert calls["n"] == 1, "streaming must never re-dispatch"
    assert svc._state.structured_blank_runs_detected == 1
    assert svc._state.structured_blank_runs_unrecovered == 1
    assert svc._state.structured_blank_runs_salvaged == 0
    assert svc._state.structured_blank_runs_retried == 0

    stream_events = [c for c in events if c.get("layer") == "stream"]
    assert len(stream_events) == 1
    assert stream_events[0]["abort_reason"] == "structured_blank_run"
    assert stream_events[0]["proxy_initiated"] is True


@pytest.mark.asyncio
async def test_streaming_is_inert_when_the_endpoint_declares_nothing():
    svc = _svc(armed=False)

    async def fake_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        yield BackendStreamEvent("chunk", _chunk('{"a": 1}' + " " * 500),
                                 json.loads(_chunk('{"a": 1}' + " " * 500)))
        yield BackendStreamEvent("chunk", _chunk(finish="stop"),
                                 json.loads(_chunk(finish="stop")))
        yield BackendStreamEvent("done", "[DONE]")

    svc._backend.stream = fake_stream
    await svc.startup()
    try:
        resp = await svc.handle_submit(_body(), _FakeRequest())
        frames = await _drain(resp)
    finally:
        await svc.shutdown()

    assert frames[-1]["type"] == "done"
    assert svc._state.structured_blank_runs_detected == 0


@pytest.mark.asyncio
async def test_streaming_ineligible_request_with_tools_is_never_watched():
    svc = _svc(armed=True)
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]

    async def fake_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        yield BackendStreamEvent("chunk", _chunk(" " * 500), json.loads(_chunk(" " * 500)))
        yield BackendStreamEvent("chunk", _chunk(finish="stop"),
                                 json.loads(_chunk(finish="stop")))
        yield BackendStreamEvent("done", "[DONE]")

    svc._backend.stream = fake_stream
    await svc.startup()
    try:
        resp = await svc.handle_submit(_body(extra={"tools": tools}), _FakeRequest())
        frames = await _drain(resp)
    finally:
        await svc.shutdown()

    assert frames[-1]["type"] == "done"
    assert svc._state.structured_blank_runs_detected == 0
