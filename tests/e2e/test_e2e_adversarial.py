"""South-face adversarial E2E: drive the fake backend to emit each pathological
output and assert the proxy handles it *cleanly* — a typed error or a served
response, **never** an unhandled 500, and **never** a leaked slot.

The two load-bearing invariants for every case (the Phase-T bar):
  1. the proxy never returns the generic unhandled-500 backstop
     ("internal proxy error") — every fault is HANDLED, not crashed;
  2. after the call settles, in-flight returns to baseline (no slot leak).

Exact per-fault terminal states (repair vs typed-fail) get tightened in the
feature phases; Phase T pins the *never crashes / never leaks* floor and the
happy cases that must keep working.
"""

from __future__ import annotations

import asyncio
import json

import pytest

# Exhaustive ProxyService-spinning adversarial matrix — deselected from the
# per-ship in_container_tollgate via `-m 'not heavy'` (see pyproject `heavy`).
pytestmark = pytest.mark.heavy

from tests.llmproxy.fake_backend import (
    FAULT_CAPACITY_DESYNC,
    FAULT_DEGENERATE_LOOP,
    FAULT_EMPTY_COMPLETION,
    FAULT_FINISH_LENGTH,
    FAULT_HTTP_400,
    FAULT_HTTP_500,
    FAULT_HTTP_503,
    FAULT_INTERLEAVED_SSE,
    FAULT_INTERTOKEN_STALL,
    FAULT_MID_STREAM_RESET,
    FAULT_NO_DONE,
    FAULT_NO_USAGE,
    FAULT_PARTIAL_SSE,
    FAULT_PHANTOM_TOOL_CALLS,
    FAULT_TIMEOUT,
    FAULT_TRUNCATED_JSON,
    FAULT_TTFT_STALL,
)


_UNHANDLED_500_MARKER = "internal proxy error"


async def _assert_no_leak(proxy, timeout: float = 2.0) -> None:
    """In-flight must return to zero. Streaming completions resolve when the
    client finishes draining, so poll briefly."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if proxy.total_in_flight() == 0:
            return
        await asyncio.sleep(0.02)
    assert proxy.total_in_flight() == 0, "slot leak: in-flight did not return to 0"


def _assert_handled(resp) -> None:
    """Never the generic unhandled-500 backstop — the fault was HANDLED."""
    body_text = resp.text or ""
    assert not (resp.status_code == 500 and _UNHANDLED_500_MARKER in body_text), (
        f"unhandled crash: status={resp.status_code} body={body_text[:200]!r}")


# --------------------------------------------------------------------------- #
# Non-streaming south-face pathologies
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fault", [
    FAULT_HTTP_400, FAULT_HTTP_500, FAULT_HTTP_503,
    FAULT_TRUNCATED_JSON, FAULT_EMPTY_COMPLETION,
])
async def test_sync_error_faults_fail_cleanly(proxy, fault):
    resp = await proxy.chat("hello", fault=fault)
    _assert_handled(resp)
    # a real error surfaces as a non-2xx (or an OpenAI error body), never a crash
    assert resp.status_code >= 400 or "error" in resp.text.lower()
    await _assert_no_leak(proxy)


async def test_sync_no_usage_is_served(proxy):
    # usage is optional; a 200 with content but no usage block must still serve.
    resp = await proxy.chat("hello", fault=FAULT_NO_USAGE)
    _assert_handled(resp)
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"].startswith("echo:")
    await _assert_no_leak(proxy)


async def test_sync_finish_length_nonstructured_is_served(proxy):
    # finish_reason=length on a NON-structured request is ordinary max_tokens
    # truncation → a served 200 (truncation-integrity only fails structured).
    resp = await proxy.chat("hello", fault=FAULT_FINISH_LENGTH)
    _assert_handled(resp)
    assert resp.status_code == 200
    await _assert_no_leak(proxy)


async def test_sync_phantom_tool_calls_bypass_empty_gate(proxy):
    # content empty BUT tool_calls present → the empty-completion gate is
    # correctly bypassed; response is handled without a crash.
    resp = await proxy.chat("hello", fault=FAULT_PHANTOM_TOOL_CALLS)
    _assert_handled(resp)
    await _assert_no_leak(proxy)


async def test_sync_backend_timeout_fails_cleanly(proxy):
    # fake sleeps 2s; caller deadline 0.5s → clean BackendTimeout, no hang.
    resp = await proxy.chat("hello", fault=FAULT_TIMEOUT, fault_arg=2.0, timeout_s=0.5)
    _assert_handled(resp)
    assert resp.status_code >= 400 or "error" in resp.text.lower()
    await _assert_no_leak(proxy)


async def test_degeneration_transient_recovers(proxy):
    # A transient degenerate loop (fires once, then the backend recovers) must
    # not crash or leak; the proxy either re-dispatches to the recovered output
    # or returns the (handled) response.
    proxy.controller.set_fault(FAULT_DEGENERATE_LOOP, max_hits=1)
    resp = await proxy.chat("sing me a song")
    _assert_handled(resp)
    await _assert_no_leak(proxy)


# --------------------------------------------------------------------------- #
# Streaming south-face pathologies
# --------------------------------------------------------------------------- #

async def test_stream_mid_reset_no_leak(proxy):
    # backend aborts mid-stream → real httpx RemoteProtocolError → proxy maps to
    # a clean BackendUnavailable; the client sees a terminated stream, no leak.
    frames = await proxy.stream_frames("a b c", fault=FAULT_MID_STREAM_RESET)
    # we don't assert on frame content (a reset can truncate mid-flight); the
    # invariant is that we didn't crash the proxy and the slot came back.
    assert isinstance(frames, list)
    await _assert_no_leak(proxy)


async def test_stream_partial_frame_no_crash(proxy):
    frames = await proxy.stream_frames("a b c", fault=FAULT_PARTIAL_SSE)
    assert isinstance(frames, list)
    await _assert_no_leak(proxy)


async def test_stream_interleaved_comments_skipped(proxy):
    # SSE comment / keepalive lines must be skipped by the proxy parser; the
    # real content still arrives.
    frames = await proxy.stream_frames("a b c", fault=FAULT_INTERLEAVED_SSE)
    content = ""
    for f in frames:
        if f == "[DONE]":
            continue
        try:
            obj = json.loads(f)
        except json.JSONDecodeError:
            continue
        ch = obj.get("choices") or []
        if ch:
            content += (ch[0].get("delta") or {}).get("content") or ""
    assert "echo: a b c" in content
    await _assert_no_leak(proxy)


async def test_stream_no_done_no_leak(proxy):
    frames = await proxy.stream_frames("a b c", fault=FAULT_NO_DONE)
    assert isinstance(frames, list)
    await _assert_no_leak(proxy)


@pytest.mark.parametrize("fault", [FAULT_TTFT_STALL, FAULT_INTERTOKEN_STALL])
async def test_stream_stall_aborts_cleanly(proxy, fault):
    # fake stalls 2s; caller deadline 0.5s → the stream is cut at the adaptive
    # deadline (min(watchdog, stream_timeout)), not left hanging; no leak.
    frames = await proxy.stream_frames("a b c", fault=fault, fault_arg=2.0,
                                       timeout_s=0.5)
    assert isinstance(frames, list)
    await _assert_no_leak(proxy)


# --------------------------------------------------------------------------- #
# Capacity desync
# --------------------------------------------------------------------------- #

async def test_capacity_desync_concurrent(proxy):
    # /props advertises 4 slots but the backend accepts only 1 concurrent → the
    # extras 503. Every request must resolve (served or deferrable error), no
    # crash, no leak.
    proxy.controller.accept_limit = 1
    proxy.controller.set_fault(FAULT_CAPACITY_DESYNC, arg=0.2)
    results = await asyncio.gather(
        *[proxy.chat(f"c{i}") for i in range(6)], return_exceptions=True)
    for r in results:
        assert not isinstance(r, Exception), f"request raised: {r!r}"
        _assert_handled(r)
    await _assert_no_leak(proxy)
