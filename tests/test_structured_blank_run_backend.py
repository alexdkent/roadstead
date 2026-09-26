"""BackendClientPool.call_watched — reassembly fidelity + the abort-closes-
the-stream guarantee.

Driven directly against the pool (no ProxyService), with `.stream` swapped
for a fake async generator — the same technique `test_truncation_guard.py`
uses for `.call`/`.stream`.
"""
from __future__ import annotations

import asyncio
import json
import logging

import pytest

from roadstead.backend import (
    BackendClientPool,
    BackendStreamEvent,
    BackendTimeout,
    StructuredBlankRunAborted,
)
from roadstead.config import EndpointConfig
from roadstead.correction import StructuredBlankRunDetector
from roadstead.cost_model import estimate_tokens_from_chars


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


# --------------------------------------------------------------------------- #
# the TOTAL deadline — a slow-but-alive stream must not outlive timeout_s
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_a_trickling_stream_times_out_at_the_total_deadline_not_the_gap():
    """🚨 THE REGRESSION THIS TEST PINS. `stream()`'s own `httpx.Timeout
    (read=timeout_s)` bounds the GAP between two reads, not the call's total
    elapsed time — exactly right for `execute_streaming`'s progress-governed
    deadlines, and exactly wrong for a caller standing in for `call()`, whose
    `asyncio.wait_for(timeout_s)` bounds the WHOLE attempt (the Phase-1.5
    slot-leak fix). A stream emitting one token every 50ms — each gap far
    under any reasonable per-read timeout — run for 10 iterations is 500ms of
    real elapsed time; at ``timeout_s=0.2`` that must time out at ~0.2s, not
    run to completion holding the slot past the caller's own deadline.
    Without the total-deadline wrapper this test fails by NOT raising at all
    (the fake stream has no real httpx read to bound it, so nothing stops
    it)."""
    pool = BackendClientPool()
    closed = {"value": False}

    async def fake_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        try:
            for i in range(10):
                await asyncio.sleep(0.05)
                yield _event(_chunk(delta={"content": f"x{i} "}))
            yield _event(_chunk(delta={}, finish_reason="stop"))
            yield BackendStreamEvent("done", "[DONE]")
        finally:
            closed["value"] = True

    pool.stream = fake_stream
    t0 = asyncio.get_event_loop().time()
    with pytest.raises(BackendTimeout) as exc_info:
        await pool.call_watched(
            _ep(), {"messages": []}, "chat_completion", "req-6",
            StructuredBlankRunDetector(threshold=512),
            timeout_s=0.2,
        )
    elapsed = asyncio.get_event_loop().time() - t0
    assert elapsed < 0.45, f"ran past the total deadline ({elapsed:.2f}s)"
    assert "0.2" in str(exc_info.value)
    assert closed["value"] is True, "the stream must still be closed on this path too"


# --------------------------------------------------------------------------- #
# output_tokens on abort — a CHARS estimate, never a chunk count
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_abort_output_tokens_is_a_chars_estimate_not_a_chunk_count():
    """🚨 THE REGRESSION THIS TEST PINS. A chunk is not a token (MTP emits
    several tokens per chunk) — feeding the same content in many tiny
    single-character chunks must NOT change the reported estimate, which a
    chunk-count proxy would."""
    pool = BackendClientPool()
    prefix = '{"a": '
    blank = " " * 600
    full_content = prefix + blank

    async def fake_stream_many_tiny_chunks(ep_cfg, payload, payload_type,
                                           request_id, timeout_s=180.0):
        for ch in full_content:
            yield _event(_chunk(delta={"content": ch}))

    pool.stream = fake_stream_many_tiny_chunks
    with pytest.raises(StructuredBlankRunAborted) as exc_info:
        await pool.call_watched(
            _ep(), {"messages": []}, "chat_completion", "req-7",
            StructuredBlankRunDetector(threshold=512),
        )
    exc = exc_info.value
    # The detector fires the INSTANT the run reaches threshold, so `exc.content`
    # is truncated at that point, not the whole fed body — the estimate must
    # match what was ACTUALLY accumulated, not the full input.
    assert exc.output_tokens == estimate_tokens_from_chars(len(exc.content))
    # One chunk per CHARACTER (len(exc.content) chunks) is wildly different
    # from a chars/4 estimate — if the old chunk-count bug were still there,
    # this would read len(exc.content), not len(exc.content)//4.
    assert exc.output_tokens != len(exc.content)


# --------------------------------------------------------------------------- #
# usage passthrough — the backend's own block, not a rebuild
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_usage_block_is_passed_through_verbatim_including_unknown_fields():
    """Mirrors a live tier3 non-stream body: `usage.completion_tokens_details.
    reasoning_tokens` must survive into the reassembled body. A field-by-field
    rebuild (prompt_tokens/completion_tokens/total_tokens only) would silently
    drop it."""
    pool = BackendClientPool()

    async def fake_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        yield _event(_chunk(delta={"content": '{"a": 1}'}, finish_reason="stop"))
        yield _event({
            "id": "chatcmpl-real", "object": "chat.completion.chunk",
            "choices": [],
            "usage": {
                "prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30,
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
        })
        yield BackendStreamEvent("done", "[DONE]")

    pool.stream = fake_stream
    resp = await pool.call_watched(
        _ep(), {"messages": []}, "chat_completion", "req-8",
        StructuredBlankRunDetector(threshold=512),
    )
    usage = resp.body["usage"]
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 7
    assert usage["prompt_tokens"] == 10 and usage["completion_tokens"] == 20
    assert resp.output_tokens == 20  # the typed field still parses correctly


# --------------------------------------------------------------------------- #
# no usage frame at all on a CLEAN completion — estimate + loud warning
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_clean_completion_with_no_usage_frame_estimates_and_warns(caplog):
    """🚨 THE REGRESSION THIS TEST PINS. A backend that answers without ever
    emitting a usage frame (no `stream_options.include_usage` support, or the
    frame got lost) must not silently report `output_tokens=0` on an
    evidently non-empty answer — that is indistinguishable from a real zero
    to every accounting consumer downstream."""
    pool = BackendClientPool()
    content = '{"a": 1, "b": "hello world, this has real content in it"}'
    expected = estimate_tokens_from_chars(len(content))

    async def fake_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        yield _event(_chunk(delta={"content": content}, finish_reason="stop"))
        yield BackendStreamEvent("done", "[DONE]")

    pool.stream = fake_stream
    with caplog.at_level(logging.WARNING):
        resp = await pool.call_watched(
            _ep(), {"messages": []}, "chat_completion", "req-9",
            StructuredBlankRunDetector(threshold=512),
        )
    assert resp.output_tokens == expected
    assert resp.output_tokens != 0
    marker = [r for r in caplog.records
              if "ROADSTEAD_CALL_WATCHED_NO_USAGE" in r.getMessage()]
    assert len(marker) == 1
    assert f"estimated_output_tokens={expected}" in marker[0].getMessage()
