"""Phase 4 hardening — streaming usage accounting.

Without ``stream_options.include_usage`` most backends emit no usage in the
stream, so streaming completions recorded 0 tokens (live: 495 zero-token
orchestrator rows/day). The proxy now injects the flag into the BACKEND
payload (local copy) and, when the CLIENT didn't ask for usage, captures and
DROPS the synthetic usage-only frame so strict OpenAI clients (opencode's
Vercel AI SDK) see a byte-identical stream. Pins:

  - injection happens (backend sees include_usage) on a COPY — the caller's
    ``req.payload`` is never mutated (it's corpus-persisted);
  - tokens are recorded in the completion + the internal done event;
  - OpenAI relay: usage-only frame dropped when not requested, forwarded
    byte-identical when requested; [DONE] ordering unchanged;
  - usage riding on a normal (choices-bearing) chunk passes through;
  - kill-switch: runtime flag ``inject_stream_usage`` off → no injection.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from roadstead.backend import BackendStreamEvent
from roadstead.config import ProxyConfig
from roadstead.service import ProxyService


class _Req:
    class _Client:
        host = "127.0.0.1"

    client = _Client()
    headers: dict = {}


async def _collect_stream(resp, timeout: float = 5.0) -> list[str]:
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


def _chunk(content=None, finish=None):
    delta = {"content": content} if content is not None else {}
    return json.dumps({"choices": [{"index": 0, "delta": delta,
                                    "finish_reason": finish}]})


_USAGE_FRAME = json.dumps(
    {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 7}})


def _streaming_backend(seen_payloads: list):
    async def fake_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        seen_payloads.append(payload)
        yield BackendStreamEvent("chunk", _chunk("hel"), json.loads(_chunk("hel")))
        yield BackendStreamEvent("chunk", _chunk("lo", finish="stop"),
                                 json.loads(_chunk("lo", finish="stop")))
        if (payload.get("stream_options") or {}).get("include_usage"):
            yield BackendStreamEvent("chunk", _USAGE_FRAME, json.loads(_USAGE_FRAME))
        yield BackendStreamEvent("done", "[DONE]")
    return fake_stream


def _submit_body(stream_options=None):
    payload = {"messages": [{"role": "user", "content": "x"}], "stream": True}
    if stream_options is not None:
        payload["stream_options"] = stream_options
    return {
        "agent_id": "a", "endpoint": "tier3", "priority": "P1_TURN_SUPPORT",
        "call_site": "t", "payload_type": "chat_completion",
        "payload": payload, "timeout_s": 10.0,
    }


@pytest.mark.asyncio
async def test_injects_usage_and_records_tokens_internal_envelope():
    svc = ProxyService(ProxyConfig())
    seen: list = []
    svc._backend.stream = _streaming_backend(seen)
    captured = {}
    orig_persist = svc._queue_db.persist_complete

    def spy_persist(request_id, agent_id, endpoint, call_site, priority,
                    input_tokens, output_tokens, *a, **k):
        captured.update(input_tokens=input_tokens, output_tokens=output_tokens)
        return orig_persist(request_id, agent_id, endpoint, call_site, priority,
                            input_tokens, output_tokens, *a, **k)

    svc._queue_db.persist_complete = spy_persist
    await svc.startup()
    try:
        body = _submit_body()
        resp = await svc.handle_submit(body, _Req())
        frames = await _collect_stream(resp)

        # Backend got the injected flag; the caller's payload object didn't.
        assert seen[0]["stream_options"] == {"include_usage": True}
        assert "stream_options" not in body["payload"]

        events = [json.loads(f) for f in frames]
        done = [e for e in events if e.get("type") == "done"][0]
        assert done["usage"] == {"prompt_tokens": 11, "completion_tokens": 7}
        # The synthetic usage-only frame is NOT relayed as a chunk.
        chunk_data = [e["data"] for e in events if e.get("type") == "chunk"]
        assert not any('"usage"' in d and '"choices": []' in d for d in chunk_data)
        # Tokens landed in the persisted completion (the live 0-token bug).
        assert captured == {"input_tokens": 11, "output_tokens": 7}
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_openai_relay_drops_usage_frame_and_keeps_done_ordering():
    svc = ProxyService(ProxyConfig())
    seen: list = []
    svc._backend.stream = _streaming_backend(seen)
    await svc.startup()
    try:
        resp = await svc.handle_submit(_submit_body(), _Req(), openai=True)
        frames = await _collect_stream(resp)
        assert frames[-1] == "[DONE]"
        # No synthetic usage frame leaked to a client that never asked for
        # usage — the point of this test, and unchanged.
        payload_frames = frames[:-1]
        assert all("usage" not in f for f in payload_frames)
        # THREE payload frames, not two: this fixture's second chunk carries
        # content AND finish_reason, and the proxy now splits that into
        # content + an empty-delta terminal chunk (test_coalesced_finish_split.py).
        assert len(payload_frames) == 3
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_openai_relay_forwards_usage_when_client_asked():
    svc = ProxyService(ProxyConfig())
    seen: list = []
    svc._backend.stream = _streaming_backend(seen)
    await svc.startup()
    try:
        resp = await svc.handle_submit(
            _submit_body(stream_options={"include_usage": True}), _Req(), openai=True)
        frames = await _collect_stream(resp)
        assert frames[-1] == "[DONE]"
        # The client asked → the usage frame is forwarded byte-identical.
        assert frames[-2] == _USAGE_FRAME
        # No double-injection: the caller's own stream_options pass through.
        assert seen[0]["stream_options"] == {"include_usage": True}
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_usage_on_choices_bearing_chunk_passes_through():
    svc = ProxyService(ProxyConfig())

    async def fake_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        # llama.cpp-style: usage rides ON the finish chunk (choices non-empty).
        final = json.dumps({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        })
        yield BackendStreamEvent("chunk", _chunk("hi"), json.loads(_chunk("hi")))
        yield BackendStreamEvent("chunk", final, json.loads(final))
        yield BackendStreamEvent("done", "[DONE]")

    svc._backend.stream = fake_stream
    await svc.startup()
    try:
        resp = await svc.handle_submit(_submit_body(), _Req(), openai=True)
        frames = await _collect_stream(resp)
        # The finish chunk (with its usage) is forwarded — only the synthetic
        # EMPTY-choices frame is ever dropped.
        assert any('"usage"' in f for f in frames[:-1])
        assert frames[-1] == "[DONE]"
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_kill_switch_disables_injection():
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"inject_stream_usage": False})
    seen: list = []
    svc._backend.stream = _streaming_backend(seen)
    await svc.startup()
    try:
        resp = await svc.handle_submit(_submit_body(), _Req(), openai=True)
        await _collect_stream(resp)
        assert "stream_options" not in seen[0]
    finally:
        await svc.shutdown()
