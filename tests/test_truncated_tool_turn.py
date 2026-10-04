"""A tool turn that ran out of max_tokens is an ERROR, not a 200 — `truncated_tool_turn`.

🚨 THE DEFECT THIS ENDS. A chat completion that declares `tools` and ends
`finish_reason: "length"` with no complete tool call used to come back `200`.
An agent harness reads that as "the model finished" and exits 0 having done
nothing — measured with the `pi` coding agent (a silent no-op). Roadstead had
already logged `ROADSTEAD_TRUNCATION` for it, so the condition was detected;
only the caller was never told.

What is pinned, per the operator's decision:

  * tools + length + NO tool call            -> 502 `truncated_tool_turn`
  * tools + length + PARTIAL tool call       -> the same (a cut-off call is unusable)
  * tools + length + one COMPLETE call       -> still 200 (there is something to run)
  * tools + `tool_calls` / `stop` finishes   -> 200, unchanged
  * NO tools + length                        -> 200, unchanged (the dj agent makes
                                                ~943 of these per 14 days, on purpose)
  * an empty `tools` array declares nothing  -> 200, unchanged
  * STREAMING: the chunks are already on the wire, so the error is the stream's
    error frame (OpenAI door: `data: {"error": {...}}`, no `[DONE]`) — never a
    clean finish.
  * THE CAP-REACHED CASE, measured live on tier3 (vLLM + GLM tool parser) 2026-10-04,
    stream AND sync, max_tokens=400, a write_file whose content is far longer:
    finish_reason "tool_calls", completion_tokens == 400, and `arguments` that
    PARSE (the backend closes the cut string itself: `…which guarded the"}`).
    Neither the `length` label nor "does it parse" can see it; the backend's own
    token count against the cap can. Fails as `truncated_tool_turn`.
  * a STREAM whose call arguments never form a JSON value at a non-length finish
    (the sanitizer would drop the call and relabel `length`) ends in an error
    frame too — `toolcall_truncated`, the sync mirror's code, because under the
    cap it is the model's malformed JSON, not a budget problem.
  * no proxy-side retry: exactly ONE backend dispatch.
  * counted: `truncation_total` ONCE (the existing choke point — not doubled)
    and the new `truncated_tool_turns_*` tally, on /metrics and /v1/status.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from roadstead.backend import BackendResponse, BackendStreamEvent
from roadstead.config import ProxyConfig
from roadstead.correction import (
    TRUNCATED_TOOL_TURN_CODE,
    request_declares_tools,
    tool_call_is_complete,
)
from roadstead.service import ProxyService
from tests.admin_key import ADMIN_HEADERS, enrol_admin


class _Req:
    class _Client:
        host = "172.16.0.5"  # docker-bridge IP -> ACL "internal" identity

    client = _Client()
    headers: dict = {}


@pytest.fixture(autouse=True)
def _isolate_layers(monkeypatch):
    # The repair layers are flag-gated and ON in the container env; pin them off
    # so only the guard under test decides the outcome (same as the sibling
    # truncation tests).
    monkeypatch.setenv("ROADSTEAD_PROXY_SCHEMA_BACKSTOP", "0")
    monkeypatch.delenv("ROADSTEAD_PROXY_STRUCTURED_VALIDITY", raising=False)


_TOOLS = [{"type": "function", "function": {
    "name": "write_file", "parameters": {"type": "object", "properties": {}}}}]

_COMPLETE_CALL = {"id": "c1", "type": "function", "function": {
    "name": "write_file", "arguments": '{"path": "/tmp/a", "text": "hi"}'}}
_PARTIAL_CALL = {"id": "c1", "type": "function", "function": {
    "name": "write_file", "arguments": '{"path": "/tmp/a", "text": "hi th'}}


def _payload(*, tools=_TOOLS, max_tokens=5000, stream=False):
    p = {"model": "tier3", "max_tokens": max_tokens,
         "messages": [{"role": "user", "content": "write the file"}]}
    if tools is not None:
        p["tools"] = tools
    if stream:
        p["stream"] = True
    return p


def _completion(finish, *, content=None, tool_calls=None, out=5000):
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return BackendResponse(
        status_code=200,
        body={"choices": [{"index": 0, "message": msg, "finish_reason": finish}],
              "usage": {"prompt_tokens": 40, "completion_tokens": out}},
        duration_s=0.01, input_tokens=40, output_tokens=out, finish_reason=finish)


async def _svc(*, call=None, stream=None) -> tuple[ProxyService, list]:
    """A started proxy whose backend is faked; `dispatches` counts every call."""
    svc = ProxyService(ProxyConfig())
    dispatches: list = []

    async def _call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        dispatches.append(request_id)
        return await call(ep_cfg, payload, payload_type, request_id, timeout_s)

    async def _stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        dispatches.append(request_id)
        async for ev in stream(ep_cfg, payload, payload_type, request_id, timeout_s):
            yield ev

    async def _none(*a, **k):
        return None

    if call:
        svc._backend.call = _call
    if stream:
        svc._backend.stream = _stream
    svc._backend.probe_vllm_capacity = _none
    svc._backend.probe_props = _none
    svc._backend.probe_models = _none
    await svc.startup()
    return svc, dispatches


def _sync_backend(resp):
    async def call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return resp
    return call


def _frame(delta=None, finish=None, usage=None):
    obj = {"id": "x", "object": "chat.completion.chunk",
           "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage:
        obj["usage"] = usage
    return json.dumps(obj)


def _stream_backend(frames):
    """A backend that streams these raw chunk payloads then `[DONE]`."""
    async def stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        for f in frames:
            yield BackendStreamEvent("chunk", f, json.loads(f))
        yield BackendStreamEvent("done", "[DONE]")
    return stream


def _tool_frames(*, finish, fragments=(), content=None, out=5000):
    """A stream: optional content, optional tool-call fragments, then the
    finish chunk and a usage chunk."""
    frames = []
    if content is not None:
        frames.append(_frame({"content": content}))
    for i, frag in enumerate(fragments):
        frames.append(_frame({"tool_calls": [frag]}))
    frames.append(_frame({}, finish=finish))
    frames.append(_frame(usage={"prompt_tokens": 40, "completion_tokens": out}))
    return frames


def _opener(name="write_file", args=""):
    return {"index": 0, "id": "c1", "type": "function",
            "function": {"name": name, "arguments": args}}


def _cont(args):
    return {"index": 0, "function": {"arguments": args}}


async def _drain(resp, timeout=5.0) -> list[str]:
    frames: list[str] = []

    async def _go():
        async for chunk in resp.body_iterator:
            text = chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk
            for part in text.split("\n\n"):
                part = part.strip()
                if part.startswith("data: "):
                    frames.append(part[len("data: "):])

    await asyncio.wait_for(_go(), timeout=timeout)
    return frames


async def _openai_sync(svc, payload):
    resp = await asyncio.wait_for(
        svc.handle_openai_chat(dict(payload), _Req()), timeout=15.0)
    return resp, json.loads(resp.body)


async def _openai_stream(svc, payload):
    resp = await svc.handle_openai_chat(dict(payload, stream=True), _Req())
    assert resp.media_type == "text/event-stream"
    return await _drain(resp)


def _internal_body(payload, timeout_s=15.0):
    return {"agent_id": "pi", "endpoint": "tier3", "priority": "P3_INGESTION",
            "call_site": "pi.agent", "payload_type": "chat_completion",
            "payload": payload, "timeout_s": timeout_s}


def _st(svc):
    return svc._correction.state


# --------------------------------------------------------------------------- #
# the pure predicates
# --------------------------------------------------------------------------- #

def test_a_complete_tool_call_needs_a_name_and_object_arguments():
    assert tool_call_is_complete(_COMPLETE_CALL)
    # a finished no-arg call renders `{}`
    assert tool_call_is_complete({"function": {"name": "ls", "arguments": "{}"}})
    # cut mid-JSON, empty (started, never filled in), no name, not an object
    assert not tool_call_is_complete(_PARTIAL_CALL)
    assert not tool_call_is_complete({"function": {"name": "ls", "arguments": ""}})
    assert not tool_call_is_complete({"function": {"name": "", "arguments": "{}"}})
    assert not tool_call_is_complete({"function": {"name": "ls", "arguments": "[1]"}})
    assert not tool_call_is_complete("nonsense")
    assert not tool_call_is_complete(None)


def test_an_empty_tools_array_declares_nothing():
    assert request_declares_tools({"tools": _TOOLS})
    assert not request_declares_tools({"tools": []})
    assert not request_declares_tools({})
    assert not request_declares_tools({"tools": "nope"})


# --------------------------------------------------------------------------- #
# NON-STREAMING — the OpenAI door (pi's door)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_tools_length_no_tool_call_is_a_502_with_its_own_code(caplog):
    svc, dispatches = await _svc(call=_sync_backend(
        _completion("length", content="Let me think about this carefully…")))
    try:
        with caplog.at_level(logging.WARNING):
            resp, body = await _openai_sync(svc, _payload())
        assert resp.status_code == 502
        err = body["error"]
        assert err["code"] == TRUNCATED_TOOL_TURN_CODE == "truncated_tool_turn"
        msg = err["message"]
        # the remedy and BOTH numbers
        assert "raise max_tokens" in msg.lower() or "Raise max_tokens" in msg
        assert "max_tokens=5000" in msg and "output_tokens=5000" in msg
        # the marker + fragment prose-matching callers classify on (§2.2)
        assert "truncated structured output" in msg
        # counted: the existing choke point ONCE (not doubled) + the new tally
        assert _st(svc).truncation_total == 1
        assert _st(svc).truncated_tool_turns_total == 1
        assert _st(svc).truncated_tool_turns_by_endpoint == {"tier3": 1}
        assert any("ROADSTEAD_TRUNCATED_TOOL_TURN" in r.getMessage()
                   for r in caplog.records)
        # no proxy-side retry: a retry at the same max_tokens truncates again
        assert len(dispatches) == 1
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_tools_length_with_a_partial_tool_call_is_the_same_error():
    svc, _ = await _svc(call=_sync_backend(
        _completion("length", tool_calls=[_PARTIAL_CALL])))
    try:
        resp, body = await _openai_sync(svc, _payload())
        assert resp.status_code == 502
        assert body["error"]["code"] == "truncated_tool_turn"
        assert _st(svc).truncated_tool_turns_total == 1
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_tools_length_but_one_complete_call_is_still_a_200():
    """The spec's condition is "no COMPLETE tool call". One complete call is
    something the harness can run; failing it would discard real work."""
    svc, _ = await _svc(call=_sync_backend(
        _completion("length", tool_calls=[_COMPLETE_CALL])))
    try:
        resp, body = await _openai_sync(svc, _payload())
        assert resp.status_code == 200
        assert body["choices"][0]["message"]["tool_calls"] == [_COMPLETE_CALL]
        assert _st(svc).truncated_tool_turns_total == 0
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_a_normal_tool_call_finish_is_unchanged():
    svc, _ = await _svc(call=_sync_backend(
        _completion("tool_calls", tool_calls=[_COMPLETE_CALL], out=120)))
    try:
        resp, body = await _openai_sync(svc, _payload())
        assert resp.status_code == 200
        assert body["choices"][0]["finish_reason"] == "tool_calls"
        assert _st(svc).truncated_tool_turns_total == 0
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_tools_with_a_stop_finish_and_content_is_unchanged():
    svc, _ = await _svc(call=_sync_backend(
        _completion("stop", content="All done, nothing to run.", out=30)))
    try:
        resp, body = await _openai_sync(svc, _payload())
        assert resp.status_code == 200
        assert body["choices"][0]["message"]["content"] == "All done, nothing to run."
        assert _st(svc).truncated_tool_turns_total == 0
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_no_tools_with_a_length_finish_is_still_a_200():
    """The dj case: ~943 legitimate `length` completions per 14 days, none of
    which declares tools. Must stay a 200 — and must still be COUNTED as the
    plain truncation it is, but not as a truncated tool turn."""
    svc, _ = await _svc(call=_sync_backend(
        _completion("length", content="a capped but wanted reply", out=300)))
    try:
        resp, body = await _openai_sync(svc, _payload(tools=None, max_tokens=300))
        assert resp.status_code == 200
        assert body["choices"][0]["finish_reason"] == "length"
        assert _st(svc).truncation_total == 1            # existing counter, as before
        assert _st(svc).truncated_tool_turns_total == 0  # not this code's business
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_an_empty_tools_array_with_a_length_finish_is_still_a_200():
    svc, _ = await _svc(call=_sync_backend(_completion("length", content="cut")))
    try:
        resp, _body = await _openai_sync(svc, _payload(tools=[]))
        assert resp.status_code == 200
        assert _st(svc).truncated_tool_turns_total == 0
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_the_enriched_and_internal_doors_carry_the_same_code():
    """One result, three doors: the code must not be an OpenAI-door special."""
    svc, _ = await _svc(call=_sync_backend(_completion("length", content="cut")))
    try:
        resp = await asyncio.wait_for(
            svc.handle_submit(_internal_body(_payload()), _Req()), timeout=15.0)
        body = json.loads(resp.body)
        assert resp.status_code == 502
        assert body["status"] == "error" and body["code"] == "truncated_tool_turn"
        assert "output_tokens=5000" in body["error"]
    finally:
        await svc.shutdown()


# --------------------------------------------------------------------------- #
# STREAMING — the error frame, in place of a clean finish
# --------------------------------------------------------------------------- #

def _error_frame(frames):
    errs = [json.loads(f) for f in frames if f != "[DONE]" and '"error"' in f]
    assert len(errs) == 1, f"expected exactly one error frame, got {frames}"
    return errs[0]["error"]


@pytest.mark.asyncio
async def test_streaming_tools_length_no_tool_call_ends_in_an_error_frame(caplog):
    svc, dispatches = await _svc(stream=_stream_backend(
        _tool_frames(finish="length", content="Let me think…")))
    try:
        with caplog.at_level(logging.WARNING):
            frames = await _openai_stream(svc, _payload())
        err = _error_frame(frames)
        assert err["code"] == "truncated_tool_turn"
        assert err["type"] == "proxy_error"
        assert "max_tokens=5000" in err["message"]
        assert "output_tokens=5000" in err["message"]
        # the error REPLACES the clean ending: nothing follows it, no [DONE]
        assert "[DONE]" not in frames
        assert '"error"' in frames[-1]
        # the finish chunk went out first — it was already on the wire
        assert any('"finish_reason": "length"' in f for f in frames[:-1])
        assert _st(svc).truncation_total == 1
        assert _st(svc).truncated_tool_turns_total == 1
        assert _st(svc).truncated_tool_turns_by_endpoint == {"tier3": 1}
        assert len(dispatches) == 1
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_streaming_tools_length_with_a_partial_tool_call_is_an_error():
    svc, _ = await _svc(stream=_stream_backend(_tool_frames(
        finish="length",
        fragments=[_opener(args='{"path": "/tmp/a", '), _cont('"text": "hi th')])))
    try:
        frames = await _openai_stream(svc, _payload())
        assert _error_frame(frames)["code"] == "truncated_tool_turn"
        assert "[DONE]" not in frames
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_streaming_a_complete_call_reassembled_from_fragments_is_a_normal_finish():
    """Fragments are CONCATENATED before judging — a call whose arguments only
    parse once reassembled must not be mistaken for a partial one."""
    svc, _ = await _svc(stream=_stream_backend(_tool_frames(
        finish="tool_calls", out=90,
        fragments=[_opener(args='{"path": '), _cont('"/tmp/a"}')])))
    try:
        frames = await _openai_stream(svc, _payload())
        assert frames[-1] == "[DONE]"
        assert not any('"error"' in f for f in frames)
        assert _st(svc).truncated_tool_turns_total == 0
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_streaming_tools_length_but_one_complete_call_is_a_normal_finish():
    svc, _ = await _svc(stream=_stream_backend(_tool_frames(
        finish="length", fragments=[_opener(args='{"path": "/tmp/a"}')])))
    try:
        frames = await _openai_stream(svc, _payload())
        assert frames[-1] == "[DONE]"
        assert not any('"error"' in f for f in frames)
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_streaming_tools_with_a_stop_finish_is_a_normal_finish():
    svc, _ = await _svc(stream=_stream_backend(
        _tool_frames(finish="stop", content="Nothing to run.", out=20)))
    try:
        frames = await _openai_stream(svc, _payload())
        assert frames[-1] == "[DONE]"
        assert not any('"error"' in f for f in frames)
        assert _st(svc).truncated_tool_turns_total == 0
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_streaming_without_tools_and_a_length_finish_is_a_normal_finish():
    svc, _ = await _svc(stream=_stream_backend(
        _tool_frames(finish="length", content="a capped but wanted reply", out=300)))
    try:
        frames = await _openai_stream(svc, _payload(tools=None, max_tokens=300))
        assert frames[-1] == "[DONE]"
        assert not any('"error"' in f for f in frames)
        assert _st(svc).truncation_total == 1
        assert _st(svc).truncated_tool_turns_total == 0
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_streaming_on_the_enriched_wire_carries_the_code_on_its_error_frame():
    svc, _ = await _svc(stream=_stream_backend(
        _tool_frames(finish="length", content="cut")))
    try:
        resp = await svc.handle_submit(
            _internal_body(_payload(stream=True)), _Req())
        frames = [json.loads(f) for f in await _drain(resp)]
        errs = [f for f in frames if f.get("type") == "error"]
        assert len(errs) == 1
        assert errs[0]["code"] == "truncated_tool_turn"
        assert frames[-1] is errs[0] or frames[-1] == errs[0]
        assert not any(f.get("type") == "done" for f in frames)
    finally:
        await svc.shutdown()


# --------------------------------------------------------------------------- #
# the metric + status surfaces
# --------------------------------------------------------------------------- #

class _AdminReq:
    """127.0.0.1 + the shared admin key: /metrics is an admin-plane read."""

    class _Client:
        host = "127.0.0.1"

    client = _Client()
    headers: dict = dict(ADMIN_HEADERS)


@pytest.mark.asyncio
async def test_the_tally_is_visible_on_status_and_metrics():
    svc, _ = await _svc(call=_sync_backend(_completion("length", content="cut")))
    try:
        await _openai_sync(svc, _payload())
        await _openai_sync(svc, _payload())
        enrol_admin(svc)

        status = json.loads((await svc.handle_status(_AdminReq())).body)
        rel = status["reliability"]
        assert rel["truncated_tool_turns_total"] == 2
        assert rel["truncated_tool_turns_by_endpoint"] == {"tier3": 2}
        # the plain truncation counter moved with it — a subset, never a swap
        assert rel["truncation_total"] == 2

        text = (await svc.handle_prometheus_metrics(_AdminReq())).body.decode()
        assert 'roadstead_truncated_tool_turns_total{endpoint="tier3"} 2' in text
    finally:
        await svc.shutdown()


# --------------------------------------------------------------------------- #
# THE CAP-REACHED CASE — finish says `tool_calls`, arguments PARSE, tokens == cap
# (the shape captured live from tier3 on 2026-10-04; see the module docstring)
# --------------------------------------------------------------------------- #

_CUT_BUT_CLOSED_ARGS = (
    '{"path": "/tmp/essay.txt", "content": "The Pharos was built in three '
    'tiers: a square base, an octagonal middle section, and a cylindrical top '
    'crowned with a statue, which guarded the"}')
_CUT_BUT_CLOSED_CALL = {"id": "c1", "type": "function", "function": {
    "name": "write_file", "arguments": _CUT_BUT_CLOSED_ARGS}}


def test_the_live_capture_really_parses():
    """The premise of this whole section: what tier3 returned PARSES, so
    nothing that inspects the arguments can tell it was cut."""
    assert tool_call_is_complete(_CUT_BUT_CLOSED_CALL)


@pytest.mark.asyncio
async def test_sync_cap_reached_with_a_tool_calls_finish_is_an_error():
    svc, dispatches = await _svc(call=_sync_backend(_completion(
        "tool_calls", tool_calls=[_CUT_BUT_CLOSED_CALL], out=400)))
    try:
        resp, body = await _openai_sync(svc, _payload(max_tokens=400))
        assert resp.status_code == 502
        err = body["error"]
        assert err["code"] == "truncated_tool_turn"
        assert "finish_reason=tool_calls" in err["message"]
        assert "max_tokens=400" in err["message"] and "output_tokens=400" in err["message"]
        assert "truncated structured output" in err["message"]
        assert "CUT OFF" in err["message"]
        # the finish says tool_calls, so record_completion's `length` choke point
        # never counted it — the gate must, exactly once
        assert _st(svc).truncation_total == 1
        assert _st(svc).truncated_tool_turns_total == 1
        assert len(dispatches) == 1
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_sync_a_short_call_well_under_the_cap_is_unchanged():
    """The live control: `hello world` write_file, 22 tokens against a cap of 400."""
    svc, _ = await _svc(call=_sync_backend(_completion(
        "tool_calls", tool_calls=[_COMPLETE_CALL], out=22)))
    try:
        resp, _body = await _openai_sync(svc, _payload(max_tokens=400))
        assert resp.status_code == 200
        assert _st(svc).truncated_tool_turns_total == 0
        assert _st(svc).truncation_total == 0
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_sync_the_cap_arm_needs_a_tool_call_and_a_known_cap():
    # tokens at the cap but NO tool call and a `stop` finish: an answer that
    # happened to fill the budget, not a cut action
    svc, _ = await _svc(call=_sync_backend(_completion("stop", content="done", out=400)))
    try:
        resp, _ = await _openai_sync(svc, _payload(max_tokens=400))
        assert resp.status_code == 200
    finally:
        await svc.shutdown()
    # a cut-looking call but the caller set no max_tokens: nothing to compare to
    svc, _ = await _svc(call=_sync_backend(_completion(
        "tool_calls", tool_calls=[_CUT_BUT_CLOSED_CALL], out=400)))
    try:
        p = _payload()
        del p["max_tokens"]
        resp, _ = await _openai_sync(svc, p)
        assert resp.status_code == 200
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_stream_cap_reached_with_a_tool_calls_finish_is_an_error_frame():
    """The captured stream: one parseable tool call, finish `tool_calls`, usage
    `completion_tokens` == max_tokens. Without the cap arm this relays as a
    complete write_file followed by `[DONE]`."""
    svc, dispatches = await _svc(stream=_stream_backend(_tool_frames(
        finish="tool_calls", out=400,
        fragments=[_opener(args=_CUT_BUT_CLOSED_ARGS)])))
    try:
        frames = await _openai_stream(svc, _payload(max_tokens=400))
        err = _error_frame(frames)
        assert err["code"] == "truncated_tool_turn"
        assert "CUT OFF" in err["message"] and "output_tokens=400" in err["message"]
        assert "[DONE]" not in frames and '"error"' in frames[-1]
        assert _st(svc).truncation_total == 1
        assert _st(svc).truncated_tool_turns_by_endpoint == {"tier3": 1}
        assert len(dispatches) == 1
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_stream_a_short_call_well_under_the_cap_is_a_normal_finish():
    svc, _ = await _svc(stream=_stream_backend(_tool_frames(
        finish="tool_calls", out=22,
        fragments=[_opener(args='{"content": "hello world", "path": "/tmp/hello.txt"}')])))
    try:
        frames = await _openai_stream(svc, _payload(max_tokens=400))
        assert frames[-1] == "[DONE]" and not any('"error"' in f for f in frames)
        assert _st(svc).truncated_tool_turns_total == 0
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_stream_trailing_brace_repair_must_not_launder_a_cut_call():
    """The sanitizer trims a trailing `}` (vLLM #39584) and emits the call as
    complete. That is right for a call that really finished — and wrong for one
    that was cut, whose repaired arguments still PARSE. Same arguments, same
    stray brace: under the cap it is the normal finish, at the cap it is an
    error. The token count, not the JSON, decides."""
    junk_args = '{"path": "/tmp/a", "content": "cut off here"}}'
    svc, _ = await _svc(stream=_stream_backend(_tool_frames(
        finish="tool_calls", out=90, fragments=[_opener(args=junk_args)])))
    try:
        frames = await _openai_stream(svc, _payload(max_tokens=400))
        assert frames[-1] == "[DONE]" and not any('"error"' in f for f in frames)
    finally:
        await svc.shutdown()
    svc, _ = await _svc(stream=_stream_backend(_tool_frames(
        finish="tool_calls", out=400, fragments=[_opener(args=junk_args)])))
    try:
        frames = await _openai_stream(svc, _payload(max_tokens=400))
        assert _error_frame(frames)["code"] == "truncated_tool_turn"
    finally:
        await svc.shutdown()


# --------------------------------------------------------------------------- #
# the stream a sanitizer would drop-and-relabel `length`
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_stream_a_call_with_unfinished_arguments_at_a_tool_calls_finish_is_an_error():
    """Under the cap, so not a budget problem: the model's own malformed JSON.
    The sanitizer would drop the call and relabel `length`, leaving the client a
    `length` finish with no call and `[DONE]`. Must end in an error frame, with
    the sync mirror's code."""
    svc, _ = await _svc(stream=_stream_backend(_tool_frames(
        finish="tool_calls", out=120,
        fragments=[_opener(args='{"path": "/tmp/a", "content": "oops'),])))
    try:
        frames = await _openai_stream(svc, _payload(max_tokens=400))
        err = _error_frame(frames)
        assert err["code"] == "toolcall_truncated"
        assert "truncated structured output" in err["message"]
        assert "output_tokens=120" in err["message"]
        assert "[DONE]" not in frames
        assert _st(svc).truncation_total == 1
        # the sync mirror's code, not this module's: that counter is not touched
        assert _st(svc).truncated_tool_turns_total == 0
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_stream_a_phantom_opener_and_trailing_junk_are_not_cut_calls():
    """What the sanitizer repairs/drops silently must not become an error: a
    name-less phantom delta (vLLM #39584) beside a real, complete call."""
    phantom = {"index": 1, "id": "ph", "type": "function",
               "function": {"name": None, "arguments": ""}}
    real = _opener(args='{"path": "/tmp/a"}')
    svc, _ = await _svc(stream=_stream_backend(_tool_frames(
        finish="tool_calls", out=40, fragments=[real, phantom])))
    try:
        frames = await _openai_stream(svc, _payload(max_tokens=400))
        assert frames[-1] == "[DONE]" and not any('"error"' in f for f in frames)
    finally:
        await svc.shutdown()
