"""OpenAI-SSE front door on /v1/chat/completions — Phase 0.5.

`/v1/chat/completions` must speak real OpenAI (bare chat.completion for
non-streaming; chat.completion.chunk frames + exactly one [DONE] for
streaming) so goose-cli can come through the proxy front door — while the
internal /v1/submit envelope path stays byte-identical for every agent.

These tests drive the FULL path (enqueue → scheduler → dispatch) with a faked
backend so only the response serialization differs; the byte-identical
regression tests assert the envelope path is unchanged.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from originfleet.llmproxy.backend import (
    BackendError,
    BackendResponse,
    BackendStreamEvent,
)
from originfleet.llmproxy.config import ProxyConfig
from originfleet.llmproxy.service import ProxyService


# A docker-bridge IP → ACL "internal" identity (so handle_openai_chat's
# _acl.identify() resolves rather than 403ing).
class _FakeRequest:
    class _Client:
        host = "172.16.0.5"

    client = _Client()
    headers: dict = {}


_COMPLETION = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": "llama-thinker",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi there"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}

_CHUNKS = [
    '{"id":"chatcmpl-test","object":"chat.completion.chunk",'
    '"choices":[{"index":0,"delta":{"content":"hi"}}]}',
    '{"id":"chatcmpl-test","object":"chat.completion.chunk",'
    '"choices":[{"index":0,"delta":{"content":" there"},"finish_reason":"stop"}]}',
]


async def _make_started_service(*, call=None, stream=None) -> ProxyService:
    svc = ProxyService(ProxyConfig())

    async def default_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            status_code=200, body=_COMPLETION,
            duration_s=0.01, input_tokens=5, output_tokens=2,
        )

    async def default_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        for c in _CHUNKS:
            yield BackendStreamEvent(event_type="chunk", data=c, parsed=json.loads(c))
        yield BackendStreamEvent(event_type="done", data="[DONE]")

    async def _none(*a, **k):
        return None

    svc._backend.call = call or default_call
    svc._backend.stream = stream or default_stream
    # Neuter capacity discovery so the poller never touches the network.
    svc._backend.probe_vllm_capacity = _none
    svc._backend.probe_props = _none
    svc._backend.probe_models = _none
    await svc.startup()
    return svc


async def _collect_stream(resp, timeout: float = 5.0) -> list[str]:
    """Drain a StreamingResponse into the list of its `data:` payloads."""
    frames: list[str] = []

    async def _drain():
        async for chunk in resp.body_iterator:
            text = chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk
            for part in text.split("\n\n"):
                part = part.strip()
                if part.startswith("data: "):
                    frames.append(part[len("data: "):])

    await asyncio.wait_for(_drain(), timeout=timeout)
    return frames


def _openai_body(*, stream: bool = False, model: str = "llama-thinker") -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 16,
        "stream": stream,
    }


# --- OpenAI contract ---------------------------------------------------------

@pytest.mark.asyncio
async def test_nonstreaming_returns_bare_completion():
    svc = await _make_started_service()
    try:
        resp = await svc.handle_openai_chat(_openai_body(), _FakeRequest())
        assert resp.status_code == 200
        body = json.loads(resp.body.decode())
        # bare chat.completion — NOT the submit envelope
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "hi there"
        for envelope_key in ("status", "request_id", "queue_wait_ms",
                             "backend_latency_ms", "cache_hit"):
            assert envelope_key not in body, envelope_key
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_streaming_emits_only_chunks_and_one_done():
    svc = await _make_started_service()
    try:
        resp = await svc.handle_openai_chat(_openai_body(stream=True), _FakeRequest())
        assert resp.media_type == "text/event-stream"
        frames = await _collect_stream(resp)

        assert frames.count("[DONE]") == 1
        assert frames[-1] == "[DONE]"
        chunk_frames = [f for f in frames if f != "[DONE]"]
        assert len(chunk_frames) == 2
        for f in chunk_frames:
            obj = json.loads(f)
            assert obj["object"] == "chat.completion.chunk"
            # NO {"type":"queued"/"admitted"/...} envelope leakage
            assert "type" not in obj
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_backend_error_is_openai_shaped_502():
    async def boom(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        raise BackendError(502, "backend boom")

    svc = await _make_started_service(call=boom)
    try:
        resp = await svc.handle_openai_chat(_openai_body(), _FakeRequest())
        assert resp.status_code == 502
        body = json.loads(resp.body.decode())
        assert isinstance(body.get("error"), dict)
        assert body["error"]["type"] == "backend_error"
        assert "type" not in body  # no envelope; OpenAI error shape only
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_timeout_is_openai_shaped_504_and_honors_client_timeout():
    async def slow(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        await asyncio.sleep(0.5)
        return BackendResponse(200, _COMPLETION, 0.5, 5, 2)

    svc = await _make_started_service(call=slow)
    try:
        body = _openai_body()
        body["timeout_s"] = 0.2  # client-supplied short deadline (Phase 0.5 feature)
        resp = await svc.handle_openai_chat(body, _FakeRequest())
        assert resp.status_code == 504
        payload = json.loads(resp.body.decode())
        assert payload["error"]["type"] == "proxy_timeout"
        # let the abandoned backend dispatch finish + free the slot cleanly
        await asyncio.sleep(0.6)
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_cache_hit_returns_bare_completion():
    svc = await _make_started_service()
    try:
        svc._cache.put("k", _COMPLETION)
        svc._cache.cache_key = lambda endpoint, payload: "k"
        resp = await svc.handle_openai_chat(_openai_body(), _FakeRequest())
        assert resp.status_code == 200
        body = json.loads(resp.body.decode())
        assert body["object"] == "chat.completion"
        assert "cache_hit" not in body and "status" not in body
    finally:
        await svc.shutdown()


# --- regression: internal /v1/submit envelope unchanged ----------------------

def _submit_body(*, stream: bool = False) -> dict:
    return {
        "agent_id": "a",
        "endpoint": "llama-thinker",
        "priority": "P3_INGESTION",
        "call_site": "test",
        "payload_type": "chat_completion",
        "payload": _openai_body(stream=stream),
        "timeout_s": 10.0,
    }


@pytest.mark.asyncio
async def test_internal_submit_envelope_unchanged_nonstreaming():
    svc = await _make_started_service()
    try:
        resp = await svc.handle_submit(_submit_body(), _FakeRequest())  # openai=False
        assert resp.status_code == 200
        env = json.loads(resp.body.decode())
        assert env["status"] == "ok"
        assert "request_id" in env and "queue_wait_ms" in env and "estimated_cost_ss" in env
        # backend completion nested under "response" (NOT bare)
        assert env["response"] == _COMPLETION
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_internal_submit_envelope_unchanged_streaming():
    svc = await _make_started_service()
    try:
        resp = await svc.handle_submit(_submit_body(stream=True), _FakeRequest())
        frames = await _collect_stream(resp)
        objs = [json.loads(f) for f in frames if f != "[DONE]"]
        types = [o.get("type") for o in objs]
        assert "queued" in types          # internal queued marker still emitted
        assert "chunk" in types           # type-tagged chunk frames
        assert types[-1] == "done"        # envelope done event (NOT bare [DONE])
        assert "[DONE]" not in frames     # internal path never emits OpenAI [DONE]
    finally:
        await svc.shutdown()
