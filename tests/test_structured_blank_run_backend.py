"""BackendClientPool.call_watched — reassembly fidelity + the abort-closes-
the-stream guarantee.

Driven directly against the pool (no ProxyService), with `.stream` swapped
for a fake async generator — the same technique `test_truncation_guard.py`
uses for `.call`/`.stream`.
"""
from __future__ import annotations

import json

import pytest

from roadstead.backend import (
    BackendClientPool,
    BackendStreamEvent,
    StructuredBlankRunAborted,
)
from roadstead.config import EndpointConfig
from roadstead.correction import StructuredBlankRunDetector


def _ep() -> EndpointConfig:
    return EndpointConfig(endpoint_class="tier3", role="tier3", served_model_id="glm")


def _chunk(**choice_fields) -> dict:
    base = {"id": "chatcmpl-real", "object": "chat.completion.chunk",
            "created": 1234, "model": "glm"}
    choice = {"index": 0, "delta": {}, "finish_reason": None,
              "logprobs": None, "stop_reason": None}
    choice.update(choice_fields)
    base["choices"] = [choice]
    return base


def _usage_chunk(prompt_tokens=12, completion_tokens=4, cached_tokens=None):
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
              "total_tokens": prompt_tokens + completion_tokens}
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return {"id": "chatcmpl-real", "object": "chat.completion.chunk",
            "choices": [], "usage": usage}


def _event(parsed: dict) -> BackendStreamEvent:
    return BackendStreamEvent("chunk", json.dumps(parsed), parsed)


# --------------------------------------------------------------------------- #
# clean (non-aborted) reassembly
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_reassembled_body_matches_the_native_non_stream_shape():
    """Pinned comparison: the fields lifecycle.py's downstream logic reads —
    id, object, created, model, choices[0].message{role,content}, finish_reason,
    stop_reason, usage incl. prompt_tokens_details.cached_tokens — must match
    what a native non-stream vLLM body carries for the same completion."""
    pool = BackendClientPool()

    async def fake_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        assert payload["stream"] is True
        assert payload["stream_options"]["include_usage"] is True
        yield _event(_chunk(delta={"content": '{"a": 1}'}))
        yield _event(_chunk(delta={}, finish_reason="stop", stop_reason=None))
        yield _event(_usage_chunk(prompt_tokens=20, completion_tokens=5, cached_tokens=8))
        yield BackendStreamEvent("done", "[DONE]")

    pool.stream = fake_stream
    resp = await pool.call_watched(
        _ep(), {"messages": []}, "chat_completion", "req-1",
        StructuredBlankRunDetector(threshold=512),
    )
    native_shaped = {
        "id": "chatcmpl-real", "object": "chat.completion", "created": 1234,
        "model": "glm",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": '{"a": 1}'},
            "logprobs": None, "finish_reason": "stop", "stop_reason": None,
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25,
                  "prompt_tokens_details": {"cached_tokens": 8}},
    }
    assert resp.body == native_shaped
    assert resp.status_code == 200
    assert resp.finish_reason == "stop"
    assert resp.input_tokens == 20
    assert resp.output_tokens == 5
    assert resp.cached_tokens == 8


@pytest.mark.asyncio
async def test_caller_payload_is_never_mutated():
    """`call_watched` must inject stream/stream_options on a COPY."""
    pool = BackendClientPool()

    async def fake_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        yield _event(_chunk(delta={"content": "ok"}, finish_reason="stop"))
        yield BackendStreamEvent("done", "[DONE]")

    pool.stream = fake_stream
    caller_payload = {"messages": []}
    await pool.call_watched(
        _ep(), caller_payload, "chat_completion", "req-2",
        StructuredBlankRunDetector(threshold=512),
    )
    assert caller_payload == {"messages": []}


@pytest.mark.asyncio
async def test_reasoning_is_carried_through_under_whichever_spelling_arrived():
    pool = BackendClientPool()

    async def fake_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        yield _event(_chunk(delta={"reasoning": "thinking..."}))
        yield _event(_chunk(delta={"content": "answer"}, finish_reason="stop"))
        yield BackendStreamEvent("done", "[DONE]")

    pool.stream = fake_stream
    resp = await pool.call_watched(
        _ep(), {"messages": []}, "chat_completion", "req-3",
        StructuredBlankRunDetector(threshold=512),
    )
    msg = resp.body["choices"][0]["message"]
    assert msg["reasoning"] == "thinking..."
    assert "reasoning_content" not in msg


# --------------------------------------------------------------------------- #
# abort — the stream is CLOSED, not merely abandoned
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_detection_raises_and_actually_closes_the_stream():
    pool = BackendClientPool()
    closed = {"value": False}

    async def fake_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        try:
            yield _event(_chunk(delta={"content": '{"a": '}))
            yield _event(_chunk(delta={"content": " " * 600}))
            # Never reached if the abort/close works — a real backend would
            # keep streaming for many more seconds/tokens here.
            yield _event(_chunk(delta={}, finish_reason="stop"))
            yield BackendStreamEvent("done", "[DONE]")
        finally:
            # GeneratorExit reaches here ONLY if something calls
            # `aclose()` on this generator — a bare `break` in the consumer
            # would leave it suspended and this would never run.
            closed["value"] = True

    pool.stream = fake_stream
    with pytest.raises(StructuredBlankRunAborted) as exc_info:
        await pool.call_watched(
            _ep(), {"messages": []}, "chat_completion", "req-4",
            StructuredBlankRunDetector(threshold=512),
        )
    assert closed["value"] is True
    exc = exc_info.value
    assert exc.verdict["run_chars"] >= 512
    assert exc.content.startswith('{"a": ')


@pytest.mark.asyncio
async def test_aborted_exception_carries_body_so_far_for_salvage():
    pool = BackendClientPool()

    async def fake_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        yield _event(_chunk(delta={"content": '{"a": 1}'}))
        yield _event(_chunk(delta={"content": " " * 600}))

    pool.stream = fake_stream
    with pytest.raises(StructuredBlankRunAborted) as exc_info:
        await pool.call_watched(
            _ep(), {"messages": []}, "chat_completion", "req-5",
            StructuredBlankRunDetector(threshold=512),
        )
    exc = exc_info.value
    assert exc.content == '{"a": 1}' + " " * 600
    assert exc.body_so_far["choices"][0]["message"]["content"] == exc.content
    # finish_reason was never seen before the abort — must not be invented.
    assert exc.body_so_far["choices"][0]["finish_reason"] is None
