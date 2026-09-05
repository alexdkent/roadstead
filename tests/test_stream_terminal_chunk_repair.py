"""A streaming response must never reach a client without a `finish_reason`.

## The defect this pins

In an OpenAI SSE stream `finish_reason` rides ALONE on a final chunk whose
`delta` is `{}` — it carries no content, so losing it is invisible if you only
look at the text. When a backend ends a stream without ever emitting that
chunk, the proxy used to relay the content and then `[DONE]`, and the client
saw text with no finish_reason.

Measured live on 2026-08-24 against Beacon (CTnnn): **47 turns**. Beacon's
`_text_only_dropped_no_finish` guard treats finish_reason-less text as a
mid-stream drop, stamps the turn `length`, and injects
`"[System: The previous response was cut off by a network error mid-stream.
Continue exactly where you left off.]"` — spending a SECOND full model call to
continue an answer that was already complete. Three independent facts said it
was neither a network error nor a token budget: the answers end in complete
sentences, they sit far below any cap (largest 4,493 chars vs 22,482 for a
normal completion), and Beacon has DEDICATED continuation prompts for the
output-limit and dropped-tools causes which it used **0** and **0** times
against 47 for the network-stub variant.

## The line these tests hold

The repair is deliberately NOT "always emit a finish_reason". It fires only
when the backend sent its own `[DONE]` — i.e. asserted the response is
complete and merely failed to label it, which makes the repair a
normalisation, not an invention. A stream that ends with NEITHER `[DONE]` nor
a finish_reason is a REAL truncation; there the client's drop handling is
correct and we must not paper over it. `test_a_genuinely_unterminated_stream_
is_not_repaired` is the one that keeps those two apart, and it is the test to
run first if anyone ever "simplifies" this.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from roadstead.backend import BackendStreamEvent
from roadstead.config import ProxyConfig
from roadstead.service import ProxyService
from roadstead.enriched import WIRE_OPENAI


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


def _chunk(content=None, finish=None):
    delta = {"content": content} if content is not None else {}
    return json.dumps({"choices": [{"index": 0, "delta": delta,
                                    "finish_reason": finish}]})


def _backend(*, emit_finish: bool, emit_done: bool):
    """The exact shape observed live: content chunks, then optionally the
    lone empty-delta finish chunk, then optionally the backend's `[DONE]`."""
    async def fake_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        for piece in ("Today is ", "Monday."):
            yield BackendStreamEvent("chunk", _chunk(piece), json.loads(_chunk(piece)))
        if emit_finish:
            d = _chunk(finish="stop")
            yield BackendStreamEvent("chunk", d, json.loads(d))
        if emit_done:
            yield BackendStreamEvent("done", "[DONE]")
    return fake_stream


def _body():
    return {
        "agent_id": "beacon", "endpoint": "tier3",
        "priority": "P1_TURN_SUPPORT", "call_site": "t",
        "payload_type": "chat_completion",
        "payload": {"messages": [{"role": "user", "content": "x"}], "stream": True},
        "timeout_s": 10.0,
    }


def _finish_reasons(frames: list[str]) -> list[str]:
    out = []
    for f in frames:
        if f == "[DONE]":
            continue
        try:
            obj = json.loads(f)
        except ValueError:
            continue
        if obj.get("type") == "chunk":
            obj = json.loads(obj["data"])
        for ch in (obj.get("choices") or []):
            if ch.get("finish_reason"):
                out.append(ch["finish_reason"])
    return out


@pytest.mark.asyncio
async def test_missing_finish_reason_is_repaired_when_the_backend_said_done():
    """The live Beacon failure, reproduced: content + [DONE], no finish chunk.
    The client must still receive a finish_reason."""
    svc = ProxyService(ProxyConfig())
    svc._backend.stream = _backend(emit_finish=False, emit_done=True)
    await svc.startup()
    try:
        resp = await svc.handle_submit(_body(), _Req(), wire=WIRE_OPENAI)
        frames = await _collect(resp)
        assert frames[-1] == "[DONE]", "the [DONE] sentinel must stay last"
        assert _finish_reasons(frames) == ["stop"]
        # The repair chunk is labelled, so an operator reading a capture can
        # tell a synthesized terminal chunk from a backend-sent one.
        repaired = [f for f in frames if "proxy_synthesized_finish" in f]
        assert len(repaired) == 1
        assert json.loads(repaired[0])["choices"][0]["delta"] == {}
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_a_genuinely_unterminated_stream_is_not_repaired():
    """🚨 The line. No [DONE] and no finish_reason means the stream really did
    stop early — the client's own drop handling is CORRECT there. Repairing
    this case would convert a visible failure into a silent one: the client
    would accept a truncated answer as complete."""
    svc = ProxyService(ProxyConfig())
    svc._backend.stream = _backend(emit_finish=False, emit_done=False)
    await svc.startup()
    try:
        resp = await svc.handle_submit(_body(), _Req(), wire=WIRE_OPENAI)
        frames = await _collect(resp)
        assert _finish_reasons(frames) == []
        assert not any("proxy_synthesized_finish" in f for f in frames)
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_a_healthy_stream_is_untouched():
    """>99% of streams. The backend's own finish chunk passes through and we
    add nothing — a repair that fires on healthy traffic would double the
    terminal chunk and break strict clients."""
    svc = ProxyService(ProxyConfig())
    svc._backend.stream = _backend(emit_finish=True, emit_done=True)
    await svc.startup()
    try:
        resp = await svc.handle_submit(_body(), _Req(), wire=WIRE_OPENAI)
        frames = await _collect(resp)
        assert _finish_reasons(frames) == ["stop"], "exactly one finish_reason"
        assert not any("proxy_synthesized_finish" in f for f in frames)
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_a_content_free_stream_is_not_repaired():
    """Nothing was delivered, so there is no complete answer to vouch for.
    Guards against the repair masking a backend that produced no output."""
    async def empty(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        yield BackendStreamEvent("done", "[DONE]")

    svc = ProxyService(ProxyConfig())
    svc._backend.stream = empty
    await svc.startup()
    try:
        resp = await svc.handle_submit(_body(), _Req(), wire=WIRE_OPENAI)
        frames = await _collect(resp)
        assert not any("proxy_synthesized_finish" in f for f in frames)
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_the_internal_envelope_door_is_repaired_too():
    """`/v1/submit` callers reassemble the same chunks; the repair belongs to
    the stream, not to one front door."""
    svc = ProxyService(ProxyConfig())
    svc._backend.stream = _backend(emit_finish=False, emit_done=True)
    await svc.startup()
    try:
        resp = await svc.handle_submit(_body(), _Req())
        frames = await _collect(resp)
        assert _finish_reasons(frames) == ["stop"]
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_every_stream_logs_one_accountable_line(caplog):
    """Observability, and why it is unconditional: before this there were
    FIVE log lines covering 653 beacon requests, which is exactly why the
    defect above could not be attributed to a layer for as long as it
    existed. A stream that only logs on failure cannot tell you what normal
    looked like."""
    svc = ProxyService(ProxyConfig())
    svc._backend.stream = _backend(emit_finish=True, emit_done=True)
    await svc.startup()
    try:
        with caplog.at_level("INFO", logger="roadstead.lifecycle"):
            resp = await svc.handle_submit(_body(), _Req(), wire=WIRE_OPENAI)
            await _collect(resp)
        lines = [r.getMessage() for r in caplog.records
                 if "ROADSTEAD_STREAM_DONE" in r.getMessage()]
        assert len(lines) == 1, lines
        line = lines[0]
        for field in ("caller=beacon", "finish_reason=stop", "backend_done=True",
                      "chunks=", "ttft_ms=", "duration_ms="):
            assert field in line, (field, line)
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_an_absent_finish_reason_is_named_in_the_log(caplog):
    """`finish_reason=ABSENT` is the searchable signal. Logging `None` or
    omitting the field would make the very event we spent a session chasing
    invisible to a grep."""
    svc = ProxyService(ProxyConfig())
    svc._backend.stream = _backend(emit_finish=False, emit_done=False)
    await svc.startup()
    try:
        with caplog.at_level("INFO", logger="roadstead.lifecycle"):
            resp = await svc.handle_submit(_body(), _Req(), wire=WIRE_OPENAI)
            await _collect(resp)
        msgs = [r.getMessage() for r in caplog.records]
        assert any("finish_reason=ABSENT" in m for m in msgs), msgs
        assert any("ROADSTEAD_STREAM_UNTERMINATED" in m for m in msgs), msgs
    finally:
        await svc.shutdown()
