"""Phase 4 — observability: cache-hit metric counting + stream TTFT."""

from __future__ import annotations

import asyncio
import json

import pytest

from originfleet.llmproxy.backend import BackendResponse, BackendStreamEvent
from originfleet.llmproxy.config import ProxyConfig
from originfleet.llmproxy.service import ProxyService


class _FakeRequest:
    class _Client:
        host = "172.16.0.5"

    client = _Client()
    headers: dict = {}


async def _none(*a, **k):
    return None


def _body(stream=False):
    return {
        "agent_id": "a", "endpoint": "llama-thinker", "priority": "P3_INGESTION",
        "call_site": "t", "payload_type": "chat_completion",
        "payload": {"messages": [{"role": "user", "content": "x"}], "stream": stream},
        "timeout_s": 10.0,
    }


@pytest.mark.asyncio
async def test_cache_hit_counted_in_metrics():
    svc = ProxyService(ProxyConfig())
    svc._backend.probe_vllm_capacity = _none
    svc._backend.probe_props = _none
    svc._backend.probe_models = _none
    await svc.startup()
    try:
        svc._cache.put("k", {"id": "x", "object": "chat.completion",
                             "choices": [{"message": {"content": "c"}}]})
        svc._cache.cache_key = lambda endpoint, payload: "k"
        before = svc._metrics.count(endpoint="thinker")
        await svc.handle_submit(_body(), _FakeRequest())
        assert svc._metrics.count(endpoint="thinker") == before + 1
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_stream_done_event_has_ttft():
    svc = ProxyService(ProxyConfig())

    async def fake_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        await asyncio.sleep(0.02)  # measurable time-to-first-token
        yield BackendStreamEvent(
            "chunk", '{"choices":[{"delta":{"content":"hi"}}]}',
            {"choices": [{"delta": {"content": "hi"}}]})
        yield BackendStreamEvent("done", "[DONE]")

    svc._backend.stream = fake_stream
    svc._backend.probe_vllm_capacity = _none
    svc._backend.probe_props = _none
    svc._backend.probe_models = _none
    await svc.startup()
    try:
        resp = await svc.handle_submit(_body(stream=True), _FakeRequest())  # internal envelope
        frames = []

        async def drain():
            async for ch in resp.body_iterator:
                t = ch.decode() if isinstance(ch, (bytes, bytearray)) else ch
                for p in t.split("\n\n"):
                    p = p.strip()
                    if p.startswith("data: "):
                        frames.append(p[len("data: "):])

        await asyncio.wait_for(drain(), timeout=5.0)
        done = [json.loads(f) for f in frames
                if f.startswith("{") and json.loads(f).get("type") == "done"]
        assert done, frames
        assert "ttft_ms" in done[0] and done[0]["ttft_ms"] >= 0
    finally:
        await svc.shutdown()
