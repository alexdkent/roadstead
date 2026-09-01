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

from roadstead.backend import (
    BackendError,
    BackendResponse,
    BackendStreamEvent,
)
from roadstead.config import ProxyConfig, normalize_endpoint
from roadstead.service import ProxyService, _ToolCallStreamSanitizer


# A docker-bridge IP → ACL "internal" identity (so handle_openai_chat's
# _acl.identify() resolves rather than 403ing).
class _FakeRequest:
    class _Client:
        host = "172.16.0.5"

    client = _Client()
    headers: dict = {}


# A public IP the ACL won't recognize → 403.
class _DeniedRequest:
    class _Client:
        host = "8.8.8.8"

    client = _Client()
    headers: dict = {}


_COMPLETION = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": "tier3",
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


def _openai_body(*, stream: bool = False, model: str = "tier3") -> dict:
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
        # THREE, not two: this fixture's second chunk carries content AND
        # finish_reason (the shape vLLM emits when its producer outruns the
        # consumer), and the proxy now splits that into content + an
        # empty-delta terminal chunk — see
        # test_coalesced_finish_split.py for why. The property this test
        # actually guards is unchanged: only chat.completion.chunk objects,
        # exactly one [DONE], last.
        assert len(chunk_frames) == 3
        for f in chunk_frames:
            obj = json.loads(f)
            assert obj["object"] == "chat.completion.chunk"
            # NO {"type":"queued"/"admitted"/...} envelope leakage
            assert "type" not in obj
            # and no chunk carries content AND a finish_reason together
            ch = obj["choices"][0]
            assert not ((ch.get("delta") or {}).get("content") and ch.get("finish_reason"))
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


# --- Phase 5D: front-door correctness ---------------------------------------

@pytest.mark.asyncio
async def test_acl_denied_is_openai_shaped_403_chat():
    svc = await _make_started_service()
    try:
        resp = await svc.handle_openai_chat(_openai_body(), _DeniedRequest())
        assert resp.status_code == 403
        body = json.loads(resp.body.decode())
        # OpenAI error object, NOT the bare {"error": "<str>", "your_ip": ...}.
        assert isinstance(body.get("error"), dict)
        assert body["error"]["type"] == "access_denied"
        assert "your_ip" not in body
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_acl_denied_is_openai_shaped_403_embeddings():
    svc = await _make_started_service()
    try:
        resp = await svc.handle_openai_embeddings(
            {"model": "bge-m3", "input": "x"}, _DeniedRequest())
        assert resp.status_code == 403
        body = json.loads(resp.body.decode())
        assert isinstance(body.get("error"), dict)
        assert body["error"]["type"] == "access_denied"
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_unknown_model_is_404_model_not_found_no_slot_burned():
    svc = await _make_started_service()
    try:
        before = svc._scheduler.stats()["total_dispatched"]
        resp = await svc.handle_openai_chat(
            _openai_body(model="gpt-4-turbo"), _FakeRequest())
        assert resp.status_code == 404
        body = json.loads(resp.body.decode())
        assert body["error"]["type"] == "model_not_found"
        # Rejected pre-enqueue → no scheduler slot consumed.
        assert svc._scheduler.stats()["total_dispatched"] == before
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_embeddings_returns_bare_openai_object_not_envelope():
    _EMB = {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
        "model": "bge-m3", "usage": {"prompt_tokens": 2, "total_tokens": 2},
    }

    async def embed_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(status_code=200, body=_EMB,
                               duration_s=0.01, input_tokens=2, output_tokens=0)

    svc = await _make_started_service(call=embed_call)
    try:
        resp = await svc.handle_openai_embeddings(
            {"model": "bge-m3", "input": "hello"}, _FakeRequest())
        assert resp.status_code == 200
        body = json.loads(resp.body.decode())
        # Bare OpenAI embeddings object — NOT the internal {status,response} envelope.
        assert body["object"] == "list"
        assert body["data"][0]["object"] == "embedding"
        for envelope_key in ("status", "request_id", "queue_wait_ms", "response"):
            assert envelope_key not in body, envelope_key
    finally:
        await svc.shutdown()


# --- Phase 5E v2: streaming tool-call sanitizer -----------------------------
#
# Guards vLLM qwen3_xml's phantom/name-less streaming tool-call openers, which
# made strict clients (the Vercel AI SDK opencode uses) throw
# "Expected 'function.name' to be a string" mid-turn. See
# _ToolCallStreamSanitizer for the bug + upstream refs (vLLM #39584).

def _chunk(*tool_calls, choice_index=0):
    """Build one raw SSE chat.completion.chunk payload with these tool_calls."""
    return json.dumps({"id": "x", "object": "chat.completion.chunk", "choices": [
        {"index": choice_index, "delta": {"tool_calls": list(tool_calls)},
         "finish_reason": None}]})


def _ai_sdk_replay(frames):
    """Replay the @ai-sdk/openai-compatible streaming tool-call parser over a
    list of sanitized SSE payloads. Mirrors the bundled SDK logic: a delta that
    OPENS a new (index) slot must carry a string function.name, else it throws
    'Expected function.name to be a string'. Returns {index: {"name","args"}}.
    Raises AssertionError on the exact condition the real client aborts on."""
    slots: dict = {}
    for f in frames:
        obj = json.loads(f)
        for ch in obj.get("choices") or []:
            for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                idx = tc.get("index")
                fn = tc.get("function") or {}
                if idx not in slots:                       # opening a NEW slot
                    name = fn.get("name")
                    assert isinstance(name, str) and name, \
                        f"AI SDK would throw: opener idx={idx} name={name!r}"
                    slots[idx] = {"name": name, "args": fn.get("arguments") or ""}
                else:                                       # continuation
                    slots[idx]["args"] += fn.get("arguments") or ""
    return slots


def test_sanitizer_passthrough_non_toolcall_is_identity():
    # The >99% case: a plain content chunk → the ORIGINAL object, byte-identical.
    s = _ToolCallStreamSanitizer()
    data = ('{"id":"x","object":"chat.completion.chunk",'
            '"choices":[{"index":0,"delta":{"content":"hi"}}]}')
    assert s.feed(data) is data
    assert s.feed("[DONE]") == "[DONE]"
    assert s.feed(": keepalive") == ": keepalive"


def test_sanitizer_single_call_opens_with_name():
    s = _ToolCallStreamSanitizer()
    out = [
        s.feed(_chunk({"index": 0, "id": "c1", "type": "function",
                       "function": {"name": "get_weather", "arguments": ""}})),
        s.feed(_chunk({"index": 0, "function": {"name": None, "arguments": '{"city":'}})),
        s.feed(_chunk({"index": 0, "function": {"name": None, "arguments": '"Paris"}'}})),
    ]
    slots = _ai_sdk_replay(out)
    assert slots[0]["name"] == "get_weather"
    assert slots[0]["args"] == '{"city":"Paris"}'


def test_sanitizer_drops_phantom_opener_in_multi_call():
    # The live failure shape: real calls at index 0 and 2, a phantom (fresh id,
    # null name, empty args) at index 1 that never gets a name.
    s = _ToolCallStreamSanitizer()
    seq = [
        _chunk({"index": 0, "id": "t0", "type": "function",
                "function": {"name": "read_file", "arguments": ""}}),
        _chunk({"index": 0, "function": {"name": None, "arguments": '{"path":"/a"}'}}),
        _chunk({"index": 1, "id": "PHANTOM", "type": "function",
                "function": {"name": None, "arguments": ""}}),
        _chunk({"index": 2, "id": "t2", "type": "function",
                "function": {"name": "list_dir", "arguments": ""}}),
        _chunk({"index": 2, "function": {"name": None, "arguments": '{"path":"/b"}'}}),
    ]
    out = [s.feed(x) for x in seq]
    # No emitted frame carries the phantom id (index 1 dropped entirely).
    assert all("PHANTOM" not in f for f in out)
    # The AI SDK replay must NOT throw and must see exactly the two real calls.
    slots = _ai_sdk_replay(out)
    assert slots[0]["name"] == "read_file" and slots[0]["args"] == '{"path":"/a"}'
    assert slots[2]["name"] == "list_dir" and slots[2]["args"] == '{"path":"/b"}'
    assert 1 not in slots


def test_sanitizer_buffers_args_until_name_arrives():
    # Defensive against the other documented shape (opencode #24137): args begin
    # streaming for a new index BEFORE its name. The opener must carry the name
    # and the full buffered args; nothing may open the slot name-less first.
    s = _ToolCallStreamSanitizer()
    out = [
        s.feed(_chunk({"index": 0, "id": "c1", "type": "function",
                       "function": {"name": None, "arguments": '{"ci'}})),
        s.feed(_chunk({"index": 0, "function": {"name": None, "arguments": 'ty":'}})),
        s.feed(_chunk({"index": 0, "id": "c1", "type": "function",
                       "function": {"name": "geocode", "arguments": '"NYC"}'}})),
    ]
    slots = _ai_sdk_replay(out)
    assert slots[0]["name"] == "geocode"
    assert slots[0]["args"] == '{"city":"NYC"}'


def test_sanitizer_trims_extra_trailing_brace():
    # The other half of the vLLM parallel-call bug (correlated with the phantom):
    # the last call's args stream ends with an extra '}' -> '{"path": "/tmp"}}',
    # which is invalid JSON. The sanitizer must trim it so the AI SDK can parse.
    s = _ToolCallStreamSanitizer()
    out = [
        s.feed(_chunk({"index": 0, "id": "t0", "type": "function",
                       "function": {"name": "read_file", "arguments": ""}})),
        s.feed(_chunk({"index": 0, "function": {"arguments": '{"path":"/a"}'}})),
        s.feed(_chunk({"index": 1, "id": "PH", "type": "function",
                       "function": {"name": None, "arguments": ""}})),
        s.feed(_chunk({"index": 2, "id": "t2", "type": "function",
                       "function": {"name": "list_dir", "arguments": ""}})),
        s.feed(_chunk({"index": 2, "function": {"arguments": '{"path": "'}})),
        s.feed(_chunk({"index": 2, "function": {"arguments": '/tmp'}})),
        s.feed(_chunk({"index": 2, "function": {"arguments": '"'}})),
        s.feed(_chunk({"index": 2, "function": {"arguments": '}}'}})),  # extra brace
    ]
    slots = _ai_sdk_replay(out)               # must not throw
    assert slots[0]["name"] == "read_file"
    assert json.loads(slots[0]["args"]) == {"path": "/a"}
    assert slots[2]["name"] == "list_dir"
    assert json.loads(slots[2]["args"]) == {"path": "/tmp"}   # extra '}' trimmed
    assert 1 not in slots                                     # phantom dropped


def test_sanitizer_brace_inside_string_value_not_mistrimmed():
    # raw_decode (not brace-counting) must keep a literal '}' inside a string.
    s = _ToolCallStreamSanitizer()
    out = [
        s.feed(_chunk({"index": 0, "id": "c1", "type": "function",
                       "function": {"name": "run", "arguments": ""}})),
        s.feed(_chunk({"index": 0, "function": {"arguments": '{"cmd":"echo }"'}})),
        s.feed(_chunk({"index": 0, "function": {"arguments": '}'}})),
    ]
    slots = _ai_sdk_replay(out)
    assert json.loads(slots[0]["args"]) == {"cmd": "echo }"}


def test_sanitizer_pure_phantom_emits_no_toolcall():
    # A name-less, args-less opener that never gets a name → no tool_call ever
    # reaches the client (the frame may still carry other delta fields).
    s = _ToolCallStreamSanitizer()
    out = s.feed(json.dumps({"choices": [{"index": 0, "delta": {
        "content": None,
        "tool_calls": [{"index": 1, "id": "p", "type": "function",
                        "function": {"name": None, "arguments": ""}}]}}]}))
    obj = json.loads(out)
    assert "tool_calls" not in obj["choices"][0]["delta"]   # dropped
    assert _ai_sdk_replay([out]) == {}                       # nothing opened


def test_sanitizer_defensive_shapes_never_raise():
    s = _ToolCallStreamSanitizer()
    # Malformed JSON / non-tool-call odd shapes → returned unchanged, never raise.
    for unchanged in ('{"tool_calls": not json',
                      '{"tool_calls":1,"choices":"nope"}'):
        assert s.feed(unchanged) == unchanged
    # Valid-but-degenerate tool_call (function isn't an object) → treated as a
    # phantom and dropped; must not raise and must stay parseable JSON.
    out = s.feed('{"choices":[{"delta":{"tool_calls":[{"function":42}]}}]}')
    assert "tool_calls" not in json.loads(out)["choices"][0]["delta"]


@pytest.mark.asyncio
async def test_streaming_toolcall_continuations_drop_null_name_end_to_end():
    _TC_CHUNKS = [
        '{"id":"c","object":"chat.completion.chunk","choices":[{"index":0,'
        '"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function",'
        '"function":{"name":"get_weather","arguments":""}}]}}]}',
        '{"id":"c","object":"chat.completion.chunk","choices":[{"index":0,'
        '"delta":{"tool_calls":[{"index":0,"function":{"name":null,'
        '"arguments":"{\\"city\\": "}}]}}]}',
        '{"id":"c","object":"chat.completion.chunk","choices":[{"index":0,'
        '"delta":{"tool_calls":[{"index":0,"function":{"name":null,'
        '"arguments":"\\"Paris\\"}"}}]}}]}',
    ]

    async def tc_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        for c in _TC_CHUNKS:
            yield BackendStreamEvent(event_type="chunk", data=c, parsed=json.loads(c))
        yield BackendStreamEvent(event_type="done", data="[DONE]")

    svc = await _make_started_service(stream=tc_stream)
    try:
        resp = await svc.handle_openai_chat(_openai_body(stream=True), _FakeRequest())
        frames = await _collect_stream(resp)
        chunk_frames = [json.loads(f) for f in frames if f != "[DONE]"]
        funcs = [tc["function"]
                 for fr in chunk_frames
                 for ch in fr.get("choices", [])
                 for tc in (ch.get("delta", {}).get("tool_calls") or [])]
        assert funcs, "expected tool_call deltas in the stream"
        # First delta keeps the string name; continuations carry NO name key.
        assert funcs[0]["name"] == "get_weather"
        for fn in funcs[1:]:
            assert "name" not in fn, fn
        # Arguments reassemble to the full tool call (unchanged end-to-end).
        assembled = "".join(fn.get("arguments", "") for fn in funcs)
        assert assembled == '{"city": "Paris"}'
        assert frames.count("[DONE]") == 1
    finally:
        await svc.shutdown()


# --- the ENRICHED wire, which is the other half of the same door -------------
# `handle_submit` is no longer a route (`/v1/submit` was removed with Workstream
# C) but it is still the one shared hot path, and `wire=` is the only thing that
# differs between the two north faces. These pin that the OpenAI serialization
# above and the enriched one below really are the same request.

def _submit_body(*, stream: bool = False) -> dict:
    return {
        "agent_id": "a",
        "endpoint": "tier3",
        "priority": "P3_INGESTION",
        "call_site": "test",
        "payload_type": "chat_completion",
        "payload": _openai_body(stream=stream),
        "timeout_s": 10.0,
    }


@pytest.mark.asyncio
async def test_enriched_envelope_nonstreaming():
    svc = await _make_started_service()
    try:
        resp = await svc.handle_submit(_submit_body(), _FakeRequest())  # wire=WIRE_ENRICHED
        assert resp.status_code == 200
        env = json.loads(resp.body.decode())
        assert env["status"] == "ok"
        assert "request_id" in env
        # The four blocks, and the backend completion still nested under
        # "response" rather than merged into the envelope — a caller must never
        # have to tell our fields from the model's.
        assert env["response"] == _COMPLETION
        assert env["timing"]["queue_wait_ms"] >= 0
        assert env["timing"]["deadline_source"] == "caller"  # timeout_s supplied
        assert env["usage"]["slot_seconds"] >= 0
        assert env["attribution"]["endpoint"] == "tier3"
        assert env["attribution"]["substituted"] is False
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_enriched_envelope_streaming():
    svc = await _make_started_service()
    try:
        resp = await svc.handle_submit(_submit_body(stream=True), _FakeRequest())
        frames = await _collect_stream(resp)
        objs = [json.loads(f) for f in frames if f != "[DONE]"]
        types = [o.get("type") for o in objs]
        # The opening frame names what will serve BEFORE the first token — the
        # one thing a streaming caller cannot learn from a header, because the
        # headers are already on the wire.
        assert types[0] == "accepted"
        assert objs[0]["attribution"]["endpoint"] == "tier3"
        assert "chunk" in types           # type-tagged chunk frames
        assert types[-1] == "done"        # envelope done event (NOT bare [DONE])
        assert "[DONE]" not in frames     # the enriched wire never emits it
        done = objs[-1]
        # 🚨 Attribution is repeated on `done` and THAT one is authoritative:
        # failover and spill both move a request after `accepted` is sent.
        assert done["attribution"]["endpoint"] == "tier3"
        assert done["usage"]["output_tokens"] >= 0
    finally:
        await svc.shutdown()


# ---------------------------------------------------------------------------
# GET /v1/models — the surface external clients copy a model name FROM
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_v1_models_advertises_the_endpoint_class_not_the_role():
    """🚨 This route had NO test in this repo and was broken by a refactor of
    the catalog without a single failure — the class it called had been renamed
    out from under it. It is also the one north-face surface whose output
    becomes someone else's config: whatever `id` appears here gets pinned by a
    client and has to keep resolving.

    The endpoint CLASS is that name. `role` labels the model behind the class
    for one deployment and is expected to change; advertising it would mint
    external callers on a name we intend to rename.
    """
    svc = ProxyService(ProxyConfig())
    resp = await svc.handle_models(_FakeRequest())
    body = json.loads(resp.body)
    assert resp.status_code == 200
    assert body["object"] == "list"
    ids = {row["id"] for row in body["data"]}
    classes = set(svc._config.endpoints)
    assert ids == classes, (
        f"/v1/models advertises {ids - classes} that are not endpoint classes, "
        f"and omits {classes - ids}")
    for row in body["data"]:
        ep = svc._config.endpoints[row["id"]]
        assert row["max_model_len"] == ep.context_per_slot
        assert row["object"] == "model"
    # And every advertised name must resolve back to itself — a client pinning
    # one has to keep hitting the same endpoint.
    for name in ids:
        assert normalize_endpoint(name) == name, (
            f"{name!r} is advertised but does not normalize to itself")
