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


@pytest.mark.asyncio
async def test_inflight_handler_shape():
    """GET /v1/inflight returns the live-board shape (requests + per-endpoint)
    even when nothing is executing. No startup needed — it's a pure in-memory
    read off the scheduler."""
    svc = ProxyService(ProxyConfig())
    resp = await svc.handle_inflight(_FakeRequest())
    body = json.loads(resp.body)
    assert body["requests"] == []
    assert isinstance(body["per_endpoint"], dict) and body["per_endpoint"]
    # every configured endpoint reports its slot/queue shape
    for ep, snap in body["per_endpoint"].items():
        assert set(snap) >= {"max_slots", "in_flight", "queued", "queue_by_band"}
    assert "ts" in body


@pytest.mark.asyncio
async def test_status_admin_paused_endpoint_reads_unhealthy():
    """Regression (2026-07-09): /v1/status must NOT report an administratively
    paused endpoint (e.g. classify evicted for a media-gen job) as healthy. The
    poller stops probing a paused endpoint, so endpoint_health.healthy freezes
    True — /status derives `healthy` from it and used to show the evicted
    endpoint healthy+serving the whole eviction window. `healthy` now folds in
    the admin-pause set; the deliberate pause is surfaced via `admin_paused`.
    `paused` stays PROBE-only ("unexpectedly down" — check_alerts keys on it),
    so an operator/coordinator pause reads paused:False by design."""
    svc = ProxyService(ProxyConfig())
    ep = next(iter(svc._config.endpoints))
    # Simulate a live, probe-healthy endpoint that then gets admin-paused.
    svc._endpoint_health[ep] = {"healthy": True, "consecutive_failures": 0,
                                "unhealthy_since": None}
    svc._paused_endpoints.add(ep)
    resp = await svc.handle_status(_FakeRequest())
    snap = json.loads(resp.body)["endpoints"][ep]
    assert snap["healthy"] is False        # not serving while paused
    assert snap["admin_paused"] is True    # deliberate pause is VISIBLE
    assert snap["paused"] is False         # probe-only; unexpected-down semantics preserved

    # Resuming clears the admin pause → healthy tracks probe state again.
    svc._paused_endpoints.discard(ep)
    resp = await svc.handle_status(_FakeRequest())
    snap = json.loads(resp.body)["endpoints"][ep]
    assert snap["healthy"] is True
    assert "admin_paused" not in snap
