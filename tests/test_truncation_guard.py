"""Truncation / structured-validity guard — operator mandate 2026-07-11.

The central response guard: output truncation can never pass silently to ANY
caller, and structured responses are always valid JSON or an explicit error.

Pins (both response modes, internal door):

  * TRUNCATION VISIBILITY (always on, observability-only): every
    finish_reason=length completion — structured or free-text, sync or stream —
    emits the stable, grep-able ``LLMPROXY_TRUNCATION`` ERROR marker carrying
    model / agent / call_site / priority / max_tokens / output_tokens /
    structured, and bumps the per-(model, caller) tally on /v1/status.
  * SYNC STRUCTURED TRUNCATION → the pre-existing 502 ("truncated structured
    output", Phase 1.1) still fires — the guard extends it, never duplicates.
  * SYNC STRUCTURED PARSE FAILURE → a 200 whose content fails json.loads on a
    JSON-implying structured request flips to the established 502 error shape
    (code ``structured_invalid_json``) + ``LLMPROXY_STRUCTURED_INVALID`` marker.
  * STREAMING: a structured stream that truncated or whose reassembled content
    fails json.loads terminates with the established error frame, never a clean
    'done'. Free-text streaming truncation still gets its 'done' (log + counter
    only).
  * FREE-TEXT truncation NEVER fails the request (may be a legitimate cap).
  * guided_choice / bare-token grammars are truncation-gated but never
    parse-gated (their legitimate output is not JSON).
  * Kill-switch COLLECTIVE_PROXY_STRUCTURED_VALIDITY=0 restores the legacy
    caller-visible behavior (markers/counters stay).
  * Counters appear under /v1/status "reliability".
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from originfleet.llmproxy.backend import BackendResponse, BackendStreamEvent
from originfleet.llmproxy.config import ProxyConfig
from originfleet.llmproxy.scheduler import QueuedRequest
from originfleet.llmproxy.service import ProxyService


class _Req:
    class _Client:
        host = "127.0.0.1"

    client = _Client()
    headers: dict = {}


@pytest.fixture(autouse=True)
def _isolate_layers(monkeypatch):
    """Pin the flag-gated repair layers OFF so these tests exercise the
    ALWAYS-ON floor in isolation (the backstop is ON in the container env and
    would repair/502 before the floor runs); leave the floor itself at its
    DEFAULT (on) — that default is part of what this file pins."""
    monkeypatch.setenv("COLLECTIVE_PROXY_SCHEMA_BACKSTOP", "0")
    monkeypatch.delenv("COLLECTIVE_PROXY_STRUCTURED_VALIDITY", raising=False)


# --------------------------------------------------------------------------- #
# harness helpers
# --------------------------------------------------------------------------- #

_SCHEMA_RF = {"type": "json_schema",
              "json_schema": {"name": "s", "schema": {"type": "object"}}}


def _body(payload, timeout_s=15.0):
    return {
        "agent_id": "kv4", "endpoint": "llama-thinker", "priority": "P3_INGESTION",
        "call_site": "kv4.judge", "payload_type": "chat_completion",
        "payload": payload, "timeout_s": timeout_s,
    }


def _payload(content="x", *, structured=False, stream=False, max_tokens=64, extra=None):
    p = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens}
    if structured:
        p["response_format"] = _SCHEMA_RF
    if stream:
        p["stream"] = True
    if extra:
        p.update(extra)
    return p


def _resp(content, finish, output_tokens=16):
    return BackendResponse(
        status_code=200,
        body={"choices": [{"message": {"role": "assistant", "content": content},
                           "finish_reason": finish}],
              "usage": {"prompt_tokens": 5, "completion_tokens": output_tokens}},
        duration_s=0.01, input_tokens=5, output_tokens=output_tokens,
        finish_reason=finish)


def _svc_sync(content, finish, output_tokens=16):
    svc = ProxyService(ProxyConfig())

    async def fake_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return _resp(content, finish, output_tokens)

    svc._backend.call = fake_call
    return svc


def _chunk(content=None, finish=None):
    delta = {"content": content} if content is not None else {}
    return json.dumps({"choices": [{"index": 0, "delta": delta,
                                    "finish_reason": finish}]})


def _svc_stream(pieces, finish):
    """A proxy whose backend streams `pieces` then a finish chunk."""
    svc = ProxyService(ProxyConfig())

    async def fake_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        for piece in pieces:
            yield BackendStreamEvent("chunk", _chunk(piece), json.loads(_chunk(piece)))
        final = _chunk(finish=finish)
        yield BackendStreamEvent("chunk", final, json.loads(final))
        yield BackendStreamEvent("done", "[DONE]")

    svc._backend.stream = fake_stream
    return svc


async def _collect_events(resp, timeout: float = 5.0) -> list[dict]:
    events: list[dict] = []

    async def _drain():
        async for chunk in resp.body_iterator:
            text = chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk
            for part in text.split("\n\n"):
                part = part.strip()
                if part.startswith("data: "):
                    events.append(json.loads(part[len("data: "):]))

    await asyncio.wait_for(_drain(), timeout=timeout)
    return events


async def _drive_sync(svc, payload):
    await svc.startup()
    try:
        resp = await asyncio.wait_for(
            svc.handle_submit(_body(payload), _Req()), timeout=15.0)
        return resp, json.loads(resp.body)
    finally:
        await svc.shutdown()


async def _drive_stream(svc, payload):
    await svc.startup()
    try:
        resp = await svc.handle_submit(_body(payload), _Req())
        return await _collect_events(resp)
    finally:
        await svc.shutdown()


def _tally(svc):
    return svc._correction.state.truncation_by_model_caller.get("thinker|kv4")


# --------------------------------------------------------------------------- #
# sync — truncation
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_sync_structured_truncation_502_and_marker(caplog):
    svc = _svc_sync('{"a":', "length")
    with caplog.at_level(logging.ERROR):
        resp, result = await _drive_sync(svc, _payload(structured=True))
    assert resp.status_code == 502
    assert "truncated structured output" in result["error"]  # pre-existing shape
    marker = [r for r in caplog.records if "LLMPROXY_TRUNCATION" in r.getMessage()]
    assert len(marker) == 1 and marker[0].levelno == logging.ERROR
    msg = marker[0].getMessage()
    assert ("model=thinker" in msg and "agent=kv4" in msg
            and "call_site=kv4.judge" in msg and "priority=P3_INGESTION" in msg
            and "max_tokens=64" in msg and "output_tokens=16" in msg
            and "structured=True" in msg)
    assert _tally(svc) == {"count": 1, "structured": 1, "freetext": 0}
    assert svc._correction.state.truncation_total == 1


@pytest.mark.asyncio
async def test_sync_freetext_truncation_served_but_loud(caplog):
    svc = _svc_sync("a long capped reply", "length")
    with caplog.at_level(logging.ERROR):
        resp, result = await _drive_sync(svc, _payload())
    # free-text truncation must NOT fail the request — response delivered…
    assert resp.status_code == 200 and result["status"] == "ok"
    assert result["response"]["choices"][0]["message"]["content"] == "a long capped reply"
    # …but never silently: marker + freetext tally.
    msg = next(r.getMessage() for r in caplog.records
               if "LLMPROXY_TRUNCATION" in r.getMessage())
    assert "structured=False" in msg
    assert _tally(svc) == {"count": 1, "structured": 0, "freetext": 1}


@pytest.mark.asyncio
async def test_sync_freetext_warmer_probe_exempt_from_marker(caplog):
    """A prefix-cache warmer deliberately asks for a 1-token (gemma: 16) freetext
    completion to keep the composer prefix hot; finish_reason=length on such a
    probe is EXPECTED, not a caller-visible cut-off. It must NOT log a marker or
    tally — else the warmers (fired every ~90s) bury the real truncation signal.
    Regression guard for the 2026-07-12 warmer exemption."""
    svc = _svc_sync("x", "length", output_tokens=1)
    with caplog.at_level(logging.ERROR):
        resp, result = await _drive_sync(svc, _payload(max_tokens=1))
    # Response still delivered (freetext truncation is never a hard failure)…
    assert resp.status_code == 200 and result["status"] == "ok"
    # …but the marker + tally are suppressed for the tiny probe.
    markers = [r for r in caplog.records if "LLMPROXY_TRUNCATION" in r.getMessage()]
    assert markers == []
    assert _tally(svc) is None
    assert svc._correction.state.truncation_total == 0


@pytest.mark.asyncio
async def test_sync_freetext_16token_probe_exempt(caplog):
    """The gemma warmer uses max_tokens=16 — the exemption ceiling is inclusive."""
    svc = _svc_sync("x", "length", output_tokens=16)
    with caplog.at_level(logging.ERROR):
        resp, result = await _drive_sync(svc, _payload(max_tokens=16))
    assert resp.status_code == 200 and result["status"] == "ok"
    assert [r for r in caplog.records if "LLMPROXY_TRUNCATION" in r.getMessage()] == []
    assert svc._correction.state.truncation_total == 0


@pytest.mark.asyncio
async def test_sync_freetext_truncation_never_cached():
    """A temperature=0 free-text truncation is served but must NOT enter the
    deterministic cache — otherwise one capped call re-serves the cut-off text
    for the whole TTL to every identical caller."""
    calls = {"n": 0}
    svc = ProxyService(ProxyConfig())

    async def fake_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        calls["n"] += 1
        return _resp("cut off rep", "length")

    svc._backend.call = fake_call
    await svc.startup()
    try:
        payload = _payload(extra={"temperature": 0})
        for _ in range(2):
            resp = await asyncio.wait_for(
                svc.handle_submit(_body(payload), _Req()), timeout=15.0)
            assert resp.status_code == 200
    finally:
        await svc.shutdown()
    assert calls["n"] == 2, "truncated body was served from cache"


def _toolcall_resp(args, finish="tool_calls"):
    return BackendResponse(
        status_code=200,
        body={"choices": [{"message": {
                  "role": "assistant", "content": "",
                  "tool_calls": [{"id": "t1", "type": "function",
                                  "function": {"name": "run", "arguments": args}}]},
              "finish_reason": finish}],
              "usage": {"prompt_tokens": 5, "completion_tokens": 20}},
        duration_s=0.01, input_tokens=5, output_tokens=20, finish_reason=finish)


def _svc_toolcall(args):
    svc = ProxyService(ProxyConfig())

    async def fake_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return _toolcall_resp(args)

    svc._backend.call = fake_call
    return svc


@pytest.mark.asyncio
async def test_sync_vllm_truncated_toolcall_args_fail_loud(caplog):
    """vLLM mislabels a mid-tool-call truncation as finish_reason="tool_calls":
    cut-off arguments on a vLLM sync 200 = truncation → the pinned deferrable
    502, LLMPROXY_TRUNCATION marker + tally — never a json-repaired
    valid-but-fabricated argument. (thinker is the vLLM endpoint.)"""
    svc = _svc_toolcall('{"cmd": "rm -rf /tmp/x')  # cut mid-string
    with caplog.at_level(logging.ERROR):
        resp, result = await _drive_sync(
            svc, _payload(extra={"tools": [{"type": "function",
                                            "function": {"name": "run"}}]}))
    assert resp.status_code == 502
    assert "truncated structured output" in result["error"]  # pinned marker
    assert result["code"] == "toolcall_truncated"
    assert "response" not in result
    assert any("LLMPROXY_TRUNCATION" in r.getMessage() for r in caplog.records)
    assert _tally(svc)["count"] == 1


@pytest.mark.asyncio
async def test_sync_vllm_truncated_toolcall_wins_over_live_backstop(monkeypatch):
    """ORDER IS LOAD-BEARING: with the schema backstop in LIVE enforce mode
    (COLLECTIVE_PROXY_SCHEMA_BACKSTOP=1 in the container), a truncated vLLM
    tool_call argument must be classified TRUNCATION before the backstop runs —
    json-repair would otherwise close the cut-off string into valid-but-
    FABRICATED JSON and return it as a silent 200. One backend call, no
    backstop engagement."""
    monkeypatch.setenv("COLLECTIVE_PROXY_SCHEMA_BACKSTOP", "1")
    calls = {"n": 0}
    svc = ProxyService(ProxyConfig())

    async def fake_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        calls["n"] += 1
        return _toolcall_resp('{"cmd": "rm -rf /tmp/x')

    svc._backend.call = fake_call
    resp, result = await _drive_sync(
        svc, _payload(extra={"tools": [{"type": "function",
                                        "function": {"name": "run"}}]}))
    assert resp.status_code == 502
    assert result["code"] == "toolcall_truncated"
    assert calls["n"] == 1, "backstop must not retry a truncated tool call"
    st = svc._correction.state
    assert st.schema_detected == 0, "backstop must not engage after the rule fired"


@pytest.mark.asyncio
async def test_sync_vllm_valid_toolcall_args_unaffected():
    svc = _svc_toolcall('{"cmd": "ls"}')
    resp, result = await _drive_sync(
        svc, _payload(extra={"tools": [{"type": "function",
                                        "function": {"name": "run"}}]}))
    assert resp.status_code == 200 and result["status"] == "ok"
    tc = result["response"]["choices"][0]["message"]["tool_calls"][0]
    assert tc["function"]["arguments"] == '{"cmd": "ls"}'
    assert svc._correction.state.truncation_total == 0


# --------------------------------------------------------------------------- #
# sync — structured JSON validity
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_sync_structured_parse_failure_502_and_marker(caplog):
    svc = _svc_sync("garbage not json", "stop")
    with caplog.at_level(logging.ERROR):
        resp, result = await _drive_sync(svc, _payload(structured=True))
    assert resp.status_code == 502
    assert result["status"] == "error"
    assert result["code"] == "structured_invalid_json"
    assert "invalid JSON for a structured request" in result["error"]
    assert "response" not in result  # never a body alongside the error
    marker = [r for r in caplog.records
              if "LLMPROXY_STRUCTURED_INVALID" in r.getMessage()]
    assert len(marker) == 1 and marker[0].levelno == logging.ERROR
    assert "model=thinker" in marker[0].getMessage()
    assert svc._correction.state.structured_parse_failure_total == 1
    assert svc._correction.state.structured_parse_failures_by_model_caller == {
        "thinker|kv4": 1}


@pytest.mark.asyncio
async def test_sync_structured_happy_path_unaffected():
    svc = _svc_sync('{"a": 1}', "stop")
    resp, result = await _drive_sync(svc, _payload(structured=True))
    assert resp.status_code == 200 and result["status"] == "ok"
    assert result["response"]["choices"][0]["message"]["content"] == '{"a": 1}'
    st = svc._correction.state
    assert st.structured_parse_failure_total == 0
    assert st.truncation_total == 0


@pytest.mark.asyncio
async def test_sync_guided_choice_not_parse_gated():
    # guided_choice output is a bare token — legitimate non-JSON. Structured for
    # truncation purposes, but never parse-gated.
    svc = _svc_sync("approve", "stop")
    resp, result = await _drive_sync(
        svc, _payload(extra={"extra_body": {"guided_choice": ["approve", "deny"]}}))
    assert resp.status_code == 200 and result["status"] == "ok"
    assert svc._correction.state.structured_parse_failure_total == 0


@pytest.mark.asyncio
async def test_sync_kill_switch_restores_legacy_200(monkeypatch, caplog):
    monkeypatch.setenv("COLLECTIVE_PROXY_STRUCTURED_VALIDITY", "0")
    svc = _svc_sync("garbage not json", "stop")
    resp, result = await _drive_sync(svc, _payload(structured=True))
    assert resp.status_code == 200 and result["status"] == "ok"  # legacy pass-through
    assert svc._correction.state.structured_parse_failure_total == 0


# --------------------------------------------------------------------------- #
# streaming
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_stream_structured_truncation_error_frame(caplog):
    svc = _svc_stream(['{"a":'], "length")
    with caplog.at_level(logging.ERROR):
        events = await _drive_stream(svc, _payload(structured=True, stream=True))
    kinds = [e.get("type") for e in events]
    assert "done" not in kinds, "a truncated structured stream must not end clean"
    err = next(e for e in events if e.get("type") == "error")
    assert "truncated structured output" in err["error"]
    assert any("LLMPROXY_TRUNCATION" in r.getMessage() for r in caplog.records)
    assert _tally(svc)["structured"] == 1


@pytest.mark.asyncio
async def test_stream_structured_parse_failure_error_frame(caplog):
    svc = _svc_stream(["garbage ", "not json"], "stop")
    with caplog.at_level(logging.ERROR):
        events = await _drive_stream(svc, _payload(structured=True, stream=True))
    kinds = [e.get("type") for e in events]
    assert "done" not in kinds
    err = next(e for e in events if e.get("type") == "error")
    assert "invalid JSON for a structured request" in err["error"]
    assert any("LLMPROXY_STRUCTURED_INVALID" in r.getMessage()
               for r in caplog.records)
    assert svc._correction.state.structured_parse_failures_by_model_caller == {
        "thinker|kv4": 1}


@pytest.mark.asyncio
async def test_stream_freetext_truncation_done_delivered(caplog):
    svc = _svc_stream(["partial reply"], "length")
    with caplog.at_level(logging.ERROR):
        events = await _drive_stream(svc, _payload(stream=True))
    # free-text: chunks + clean done still delivered…
    assert any(e.get("type") == "done" for e in events)
    assert not any(e.get("type") == "error" for e in events)
    # …but the truncation is loud.
    msg = next(r.getMessage() for r in caplog.records
               if "LLMPROXY_TRUNCATION" in r.getMessage())
    assert "structured=False" in msg and "stream=True" in msg
    assert _tally(svc) == {"count": 1, "structured": 0, "freetext": 1}


@pytest.mark.asyncio
async def test_stream_structured_happy_path_done():
    svc = _svc_stream(['{"a":', ' 1}'], "stop")
    events = await _drive_stream(svc, _payload(structured=True, stream=True))
    assert any(e.get("type") == "done" for e in events)
    assert not any(e.get("type") == "error" for e in events)
    st = svc._correction.state
    assert st.structured_parse_failure_total == 0 and st.truncation_total == 0


@pytest.mark.asyncio
async def test_stream_kill_switch_restores_legacy_done(monkeypatch):
    monkeypatch.setenv("COLLECTIVE_PROXY_STRUCTURED_VALIDITY", "0")
    svc = _svc_stream(['{"a":'], "length")
    events = await _drive_stream(svc, _payload(structured=True, stream=True))
    assert any(e.get("type") == "done" for e in events)  # legacy behavior
    # observability is NOT killed by the switch: the tally still counted.
    assert _tally(svc)["structured"] == 1


# --------------------------------------------------------------------------- #
# /v1/status exposure
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_counters_exposed_on_status():
    svc = _svc_sync("garbage not json", "stop")
    await _drive_sync(svc, _payload(structured=True))
    svc2 = _svc_sync("capped", "length")
    await _drive_sync(svc2, _payload())

    r1 = json.loads((await svc.handle_status(_Req())).body)["reliability"]
    assert r1["structured_parse_failure_total"] == 1
    assert r1["structured_parse_failures_by_model_caller"] == {"thinker|kv4": 1}
    r2 = json.loads((await svc2.handle_status(_Req())).body)["reliability"]
    assert r2["truncation_total"] == 1
    assert r2["truncation_by_model_caller"] == {
        "thinker|kv4": {"count": 1, "structured": 0, "freetext": 1}}


# --------------------------------------------------------------------------- #
# request_expects_json predicate (unit)
# --------------------------------------------------------------------------- #

def _req_for(payload):
    return QueuedRequest.create(
        agent_id="a", endpoint="thinker", priority="P3_INGESTION",
        call_site="t", payload_type="chat_completion", payload=payload,
        timeout_s=10.0)


def _predicate(payload) -> bool:
    svc = ProxyService(ProxyConfig())
    return svc._correction.request_expects_json(_req_for(payload))


def test_expects_json_response_format_and_guided_json():
    assert _predicate({"messages": [], "response_format": _SCHEMA_RF})
    assert _predicate({"messages": [], "response_format": {"type": "json_object"}})
    assert _predicate({"messages": [],
                       "extra_body": {"guided_json": {"type": "object"}}})


def test_expects_json_object_rooted_grammar_only():
    obj_grammar = ('root ::= "{" ws "\\"x\\"" ws ":" ws str "}"\n'
                   'str ::= "\\"" [^"]* "\\""\nws ::= [ \\t\\n]*')
    bare_grammar = 'root ::= "yes" | "no"'
    assert _predicate({"messages": [], "extra_body": {"grammar": obj_grammar}})
    assert not _predicate({"messages": [], "extra_body": {"grammar": bare_grammar}})
    assert not _predicate({"messages": [],
                           "extra_body": {"guided_choice": ["a", "b"]}})
    assert not _predicate({"messages": []})
