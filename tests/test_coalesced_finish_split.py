"""A `finish_reason` must never reach a client riding on a content chunk.

## Why the proxy normalises this

vLLM stamps `finish_reason` onto whatever delta the same engine iteration
produced, so when its producer outruns the SSE consumer the last content and
the finish are MERGED into one chunk (`RequestOutputCollector.add(...,
aggregate=True)` — the docstring says so). Measured on this fleet 2026-08-24:
**30 of 34** responses arrived coalesced.

That is schema-legal (OpenAI's OpenAPI spec makes `finish_reason` a required,
nullable field on EVERY choice and never ties it to `delta` emptiness), but it
is NOT what OpenAI's own service emits — and a client written against the
reference implementation may not handle it. Ours has one that doesn't: the
Beacon runtime skips its own `finish_reason` capture when a content chunk's
text trips an SSE-lookalike heuristic (`continue` jumps past the capture), then
reads the finished turn as a mid-stream drop and burns a SECOND model call
"continuing" a complete answer. Root-caused by instrumenting the container;
upstream issue #91373 is open with a measurably wrong diagnosis, so there is no
release to upgrade to.

Failure needs TWO conditions — coalescing AND an SSE-looking final token. This
removes the one we control, from our side, for every caller.

## The lines these tests hold

`test_a_plain_content_chunk_is_relayed_byte_identical` is the important one:
>99% of chunks must pass through as the ORIGINAL bytes, never reserialised.
Reserialising every chunk would be a silent, permanent wire change (key order,
separators, float formatting) for a problem that affects the last chunk only.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from originfleet.llmproxy.backend import BackendStreamEvent
from originfleet.llmproxy.config import ProxyConfig
from originfleet.llmproxy.lifecycle import _split_coalesced_finish_chunk as split
from originfleet.llmproxy.service import ProxyService


# ------------------------------------------------------- the pure splitter

def _coalesced(content=" it.", finish="stop"):
    return {"id": "x", "object": "chat.completion.chunk", "created": 1,
            "model": "llama-thinker",
            "choices": [{"index": 0, "delta": {"content": content},
                         "logprobs": None, "finish_reason": finish,
                         "stop_reason": None}]}


def test_a_coalesced_chunk_becomes_content_then_empty_delta_terminal():
    head, tail = split(_coalesced())
    assert head["choices"][0]["delta"] == {"content": " it."}
    assert head["choices"][0]["finish_reason"] is None
    # The terminal half mirrors what vLLM itself emits for a finish with no
    # text: an empty delta OBJECT, not a dropped key.
    assert tail["choices"][0]["delta"] == {}
    assert tail["choices"][0]["finish_reason"] == "stop"


def test_no_content_is_lost_or_duplicated_by_the_split():
    head, tail = split(_coalesced(content="the final words"))
    assert head["choices"][0]["delta"]["content"] == "the final words"
    assert "content" not in tail["choices"][0]["delta"]


@pytest.mark.parametrize("chunk,why", [
    ({"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]},
     "an ordinary content chunk"),
    ({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
     "already the canonical shape"),
    ({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0}]},
                   "finish_reason": "tool_calls"}]},
     "a TOOL-CALL finish — no content, and the sanitizer's finalize-on-finish "
     "path must see it unchanged"),
    ({"choices": [], "usage": {"prompt_tokens": 1}}, "the usage-only frame"),
    ({"choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}]},
     "empty-string content is not content"),
    ({}, "not a chunk at all"),
])
def test_everything_else_is_left_alone(chunk, why):
    assert split(chunk) is None, why


def test_usage_never_rides_on_the_synthetic_terminal_chunk():
    """`choices == []` is the documented test for the usage chunk, and this
    one HAS choices — but keeping `usage` off it removes any doubt, and
    matches where vLLM puts it."""
    c = _coalesced()
    c["usage"] = {"prompt_tokens": 11, "completion_tokens": 3}
    head, tail = split(c)
    assert head["usage"] == {"prompt_tokens": 11, "completion_tokens": 3}
    assert "usage" not in tail


def test_multiple_choices_are_split_together():
    c = {"choices": [
        {"index": 0, "delta": {"content": "a"}, "finish_reason": "stop"},
        {"index": 1, "delta": {"content": "b"}, "finish_reason": "stop"},
    ]}
    head, tail = split(c)
    assert [ch["finish_reason"] for ch in head["choices"]] == [None, None]
    assert [ch["delta"] for ch in tail["choices"]] == [{}, {}]
    assert [ch["finish_reason"] for ch in tail["choices"]] == ["stop", "stop"]


def test_the_input_chunk_is_not_mutated():
    c = _coalesced()
    before = json.dumps(c, sort_keys=True)
    split(c)
    assert json.dumps(c, sort_keys=True) == before


# ------------------------------------------------------------ on the wire

class _Req:
    class _Client:
        host = "127.0.0.1"
    client = _Client()
    headers: dict = {}


async def _collect(resp, timeout: float = 5.0) -> list[str]:
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


def _backend(final_chunk: dict, plain: str):
    async def fake_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        yield BackendStreamEvent("chunk", plain, json.loads(plain))
        d = json.dumps(final_chunk)
        yield BackendStreamEvent("chunk", d, json.loads(d))
        yield BackendStreamEvent("done", "[DONE]")
    return fake_stream


def _body():
    return {"agent_id": "beacon", "endpoint": "llama-thinker",
            "priority": "P1_TURN_SUPPORT", "call_site": "t",
            "payload_type": "chat_completion",
            "payload": {"messages": [{"role": "user", "content": "x"}], "stream": True},
            "timeout_s": 10.0}


@pytest.mark.asyncio
async def test_no_client_ever_receives_content_and_finish_on_one_chunk():
    """The property the mitigation exists for, asserted on the real wire."""
    svc = ProxyService(ProxyConfig())
    plain = json.dumps({"choices": [{"index": 0, "delta": {"content": "Good backups are"},
                                     "finish_reason": None}]})
    svc._backend.stream = _backend(_coalesced(content=" data"), plain)
    await svc.startup()
    try:
        resp = await svc.handle_submit(_body(), _Req(), openai=True)
        frames = [f for f in await _collect(resp) if f != "[DONE]"]
        for f in frames:
            ch = json.loads(f)["choices"][0]
            content = (ch.get("delta") or {}).get("content")
            assert not (content and ch.get("finish_reason")), f
        # And the answer text is still whole, in order.
        text = "".join((json.loads(f)["choices"][0].get("delta") or {}).get("content") or ""
                       for f in frames)
        assert text == "Good backups are data"
        assert sum(1 for f in frames if json.loads(f)["choices"][0].get("finish_reason")) == 1
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_a_plain_content_chunk_is_relayed_byte_identical():
    """>99% of chunks. Reserialising them all would silently change the wire
    (key order, separators, float formatting) forever, to fix a last-chunk
    problem. The splitter returns None so the ORIGINAL bytes are relayed."""
    svc = ProxyService(ProxyConfig())
    plain = '{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}'
    svc._backend.stream = _backend({"choices": [{"index": 0, "delta": {},
                                                 "finish_reason": "stop"}]}, plain)
    await svc.startup()
    try:
        resp = await svc.handle_submit(_body(), _Req(), openai=True)
        frames = await _collect(resp)
        assert plain in frames, frames
    finally:
        await svc.shutdown()
