"""The reasoning-bleed battery: a chain of thought must never be read as an answer.

WHY THIS FILE EXISTS. Measured 2026-09-29: a fleet's reasoning tier (a vLLM server
running GLM-5.3-Flash with ``--reasoning-parser glm45``) returned its whole chain of
thought glued in front of the answer, in ``content``, with no tag to split on —
``content='7391573915'`` for an answer of ``73915``. Cause: that template has no
thinking-off switch, but the PARSER reads ``enable_thinking``/``thinking`` and stands
down when either is false. About 1,560 sampled calls in four days sent one. The
fleet's existing deploy gate did not notice because it asked LEXICAL questions — does
the reply start with "Okay" or "Let me", is a tag present — and this leak has neither.

So every assertion here is SEMANTIC: the caller-visible ``content`` must EQUAL the
answer the fake backend intended, and the reasoning it intended must be somewhere else.
No tag check, no phrase check.

WHAT IT COVERS
  * CANONICAL FORM — on an endpoint that declares its switch key, a caller's other
    spelling is carried onto the declared one (an engine that reads only
    ``enable_thinking`` 400s a ``thinking`` it parses as a different type).
  * PREVENTION — on an endpoint that declares no thinking switch, no switch spelling
    reaches the backend. Every door (``/v1/chat/completions``, ``/v1/submit``,
    ``/rs/v1/chat``) x streaming on/off x every spelling, with tools and with a
    structured-output schema, plus the answer-now re-ask.
  * REPAIR — a trace the engine left in ``content`` WITH a marker is moved to the
    reasoning field: ``<think>..</think>answer``, ``..</think>answer`` (no opening
    tag), a tag split across stream chunks, a truncation inside the trace, a tool
    turn, a structured reply.
  * CONTROLS, so a pass cannot mean the harness is blind: a declared switch is left
    alone, an UNDECLARED endpoint is left alone, unmarked content is byte-identical,
    and the fake GLM backend really does leak when the kwarg reaches it.

RUNNING IT (the fleet's deploy gate runs this file by path, from a source tree):

    cd <tree> && PYTHONPATH=<tree> python -m pytest tests/test_think_bleed.py -q \\
        -p no:cacheprovider --timeout=120

``PYTHONPATH=<tree>`` is what makes ``import roadstead`` resolve INSIDE the tree under
test rather than an editable install pointing somewhere else;
``test_the_battery_is_running_against_the_tree_it_sits_in`` fails loudly if it does
not. Needs pytest, pytest-asyncio, pytest-timeout (the ``asyncio_mode = "auto"`` in
pyproject.toml is what lets the async tests run without a marker) and the package's
own dependencies (httpx, starlette, uvicorn, PyYAML, jsonschema, json-repair).
Self-contained: it imports only ``roadstead`` and the standard library, builds its own
catalog, and needs no other file under ``tests/``.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio

import roadstead
from roadstead import model_catalog
from roadstead.__main__ import build_app
from roadstead.backend import BackendClientPool
from roadstead.config import EndpointConfig, ProxyConfig
from roadstead.testing import (
    THINK_CLEAN, THINK_GLM_VLLM, THINK_SWITCHABLE, THINK_TAG_CLOSE_ONLY,
    THINK_TAG_OPEN, FakeBackend, FakeBackendServer, ThinkScript,
)

# The helpers under test may not exist yet. That is the point of running this
# battery against an UNFIXED tree to see it go red: the end-to-end cases below
# drive nothing but HTTP and must fail there for the RIGHT reason (a leaked trace),
# so an ImportError here must not take the whole file down with it. The unit tests
# that need a helper call `_need()`, which FAILS — never skips — when it is missing.
def _maybe(module: str, name: str):
    """One name at a time: a tree that has some of the helpers must not lose the rest
    to a single missing one."""
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):              # pragma: no cover
        return None


StreamRepair = _maybe("roadstead.think_bleed", "StreamRepair")
ThinkBleedStats = _maybe("roadstead.think_bleed", "ThinkBleedStats")
PREFIX_HOLD_CHARS = _maybe("roadstead.think_bleed", "PREFIX_HOLD_CHARS")
rename_thinking_switch = _maybe("roadstead.think_bleed", "rename_thinking_switch")
repair_message = _maybe("roadstead.think_bleed", "repair_message")
strip_thinking_switch = _maybe("roadstead.think_bleed", "strip_thinking_switch")
_answer_now_switch_off = _maybe("roadstead.lifecycle", "_answer_now_switch_off")
_answer_now_template_kwargs = _maybe("roadstead.lifecycle", "_answer_now_template_kwargs")
_lowest_declared_effort = _maybe("roadstead.lifecycle", "_lowest_declared_effort")
_answer_now_reasoning_budget = _maybe("roadstead.lifecycle", "_answer_now_reasoning_budget")


def _need(obj, name: str):
    if obj is None:
        pytest.fail(f"{name} does not exist in this tree — the fix is not present")
    return obj

#: Captured at import, before any conftest stubs it for network isolation: the
#: touch test below points it at the loopback fake and needs the real method.
_REAL_PROBE_PREFIX_TOUCH = BackendClientPool.probe_prefix_touch

#: What the model INTENDED. Every assertion is against these, never against a shape.
REASONING = "Let me work this out. First 7, then 39, then 15."
ANSWER = "73915"

TOOL_CALL = {"id": "call_0", "type": "function",
             "function": {"name": "get_time", "arguments": '{"tz": "UTC"}'}}
TOOLS = [{"type": "function", "function": {
    "name": "get_time", "description": "Current time",
    "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}}}}]
JSON_ANSWER = '{"total": 73915}'
SCHEMA_FORMAT = {"type": "json_schema", "json_schema": {
    "name": "sum", "strict": True,
    "schema": {"type": "object", "properties": {"total": {"type": "integer"}},
               "required": ["total"], "additionalProperties": False}}}

# --------------------------------------------------------------------------- #
# The catalog: three endpoints, three declarations
# --------------------------------------------------------------------------- #
#
#   tier3  GLM-shaped  — vLLM, always reasons, `thinking_kwargs: []` (DECLARED no switch)
#   tier2  Qwen-shaped — a template with a WORKING switch (`[enable_thinking]`); does
#                        not declare `reasoning` at all, so repair must not act here
#   tier1  llama.cpp   — reasoning-capable, says NOTHING about a switch (UNDECLARED),
#                        which is the case that must never be stripped
#   flash  llama.cpp   — a FORCED reasoner that declares its switch AND an effort
#                        default, so `apply_forced_reasoning_budget` writes the declared
#                        key for a caller that said nothing (tier2-flash's shape)
_CATALOG = """
providers:
  glm-box: {engine: vllm, host: 127.0.0.1, port: 1}
  qwen-box: {engine: vllm, host: 127.0.0.1, port: 1}
  cpp-box: {engine: llama.cpp, host: 127.0.0.1, port: 1}
endpoints:
  tier3:
    provider: glm-box
    kind: chat
    slots: 4
    context_per_slot: 8192
    capabilities: {reasoning: true, streaming: true, tool_calling: true, structured_output: true}
    policy:
      thinking_kwargs: []
      thinking_effort: low
  tier2:
    provider: qwen-box
    kind: chat
    slots: 4
    context_per_slot: 8192
    capabilities: {streaming: true, tool_calling: true, structured_output: true}
    policy:
      thinking_kwargs: [enable_thinking]
  tier1:
    provider: cpp-box
    kind: chat
    slots: 4
    context_per_slot: 8192
    capabilities: {reasoning: true, streaming: true, tool_calling: true, structured_output: true}
  flash:
    provider: cpp-box
    kind: chat
    slots: 4
    context_per_slot: 8192
    capabilities: {reasoning: true, streaming: true, tool_calling: true, structured_output: true}
    policy:
      thinking_kwargs: [enable_thinking]
      reasoning_effort: low
"""


def _endpoints(tmp: Path, host: str, port: int) -> dict[str, EndpointConfig]:
    """Built through the REAL catalog path (`load_catalog` -> `build_endpoint_kwargs`),
    so `thinking_kwargs: []` reaching `EndpointConfig` is part of what is tested."""
    path = tmp / "models.yaml"
    path.write_text(_CATALOG)
    cat = model_catalog.load_catalog(path, force=True)
    out = {}
    for cls, kw in model_catalog.build_endpoint_kwargs(cat).items():
        out[cls] = dataclasses.replace(
            EndpointConfig(**kw), host=host, port=port, max_slots=4,
            context_per_slot=8192)
    return out


@dataclass
class Rig:
    app: Any
    svc: Any
    client: httpx.AsyncClient
    server: FakeBackendServer

    @property
    def fake(self) -> FakeBackend:
        return self.server.controller

    def chat_bodies(self) -> list[dict]:
        return [r.body for r in self.fake.requests
                if r.path == "/v1/chat/completions" and r.body]

    def last_wire_body(self) -> dict:
        return self.chat_bodies()[-1]

    async def status(self) -> dict:
        return (await self.client.get("/v1/status")).json()["reliability"]

    async def counter(self, key: str) -> int:
        """A tally, 0 when the tree has no such counter. Only the CONTROLS read it
        this way: they assert something did NOT happen, which must be checkable on an
        unfixed tree too. The tests that a counter exists index `status()` directly."""
        return (await self.status()).get(key, 0)


@pytest_asyncio.fixture
async def rig(tmp_path, monkeypatch):
    # The flag is read at build_app time: it decides whether the route EXISTS.
    monkeypatch.setenv("ROADSTEAD_LEGACY_SUBMIT", "1")
    server = FakeBackendServer(FakeBackend(engine="vllm")).start()
    config = ProxyConfig(queue_db_path=str(tmp_path / "queue.db"))
    config.endpoints = _endpoints(tmp_path, server.host, server.port)
    config.poller_interval_s = 0.05
    app = build_app(config)
    svc = app.state.proxy_service

    async def _healthy(ep_cfg):
        return True

    # Timing-independent: the circuit breaker must not be able to trip mid-test.
    svc._backend.probe_health = _healthy
    await svc.startup()
    transport = httpx.ASGITransport(
        app=app, raise_app_exceptions=False, client=("127.0.0.1", 41999))
    client = httpx.AsyncClient(transport=transport, base_url="http://proxy",
                               timeout=30.0)
    try:
        yield Rig(app, svc, client, server)
    finally:
        await client.aclose()
        await svc.shutdown()
        server.stop()


# --------------------------------------------------------------------------- #
# Doors — every entry path, sync and streaming, read back the same way
# --------------------------------------------------------------------------- #

@dataclass
class Reply:
    status: int
    content: str = ""
    reasoning: str = ""
    finish: str | None = None
    tool_calls: list = field(default_factory=list)
    text: str = ""            # every byte the caller received


def _fold_chunk(reply: Reply, chunk: dict) -> None:
    for ch in chunk.get("choices") or []:
        d = ch.get("delta") or {}
        reply.content += d.get("content") or ""
        reply.reasoning += d.get("reasoning") or d.get("reasoning_content") or ""
        for tc in d.get("tool_calls") or []:
            reply.tool_calls.append(tc)
        if ch.get("finish_reason"):
            reply.finish = ch["finish_reason"]


def _from_completion(status: int, body: dict, text: str) -> Reply:
    ch = (body.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    return Reply(
        status=status, content=msg.get("content") or "",
        reasoning=msg.get("reasoning") or msg.get("reasoning_content") or "",
        finish=ch.get("finish_reason"), tool_calls=msg.get("tool_calls") or [],
        text=text)


async def _openai(rig: Rig, endpoint: str, payload: dict, stream: bool) -> Reply:
    body = {"model": endpoint, **payload, "stream": stream, "timeout_s": 30}
    if not stream:
        resp = await rig.client.post("/v1/chat/completions", json=body)
        data = resp.json()
        if resp.status_code != 200:
            return Reply(status=resp.status_code, text=resp.text)
        return _from_completion(200, data, resp.text)
    reply = Reply(status=0)
    async with rig.client.stream("POST", "/v1/chat/completions", json=body) as resp:
        reply.status = resp.status_code
        async for line in resp.aiter_lines():
            reply.text += line + "\n"
            if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                _fold_chunk(reply, json.loads(line[6:]))
    return reply


async def _wrapped(rig: Rig, route: str, envelope: dict, payload: dict,
                   stream: bool) -> Reply:
    """`/v1/submit` and `/rs/v1/chat`: the model payload rides in `payload`, and a
    stream frames each backend chunk as `{"type":"chunk","data":"<json>"}`."""
    body = {**envelope, "payload": {**payload, "stream": stream}}
    if not stream:
        resp = await rig.client.post(route, json=body)
        data = resp.json()
        if resp.status_code != 200 or data.get("status") != "ok":
            return Reply(status=resp.status_code, text=resp.text)
        return _from_completion(200, data["response"], resp.text)
    reply = Reply(status=0)
    async with rig.client.stream("POST", route, json=body) as resp:
        reply.status = resp.status_code
        async for line in resp.aiter_lines():
            reply.text += line + "\n"
            if not line.startswith("data: "):
                continue
            frame = json.loads(line[6:])
            if frame.get("type") == "chunk":
                _fold_chunk(reply, json.loads(frame["data"]))
            elif frame.get("type") == "error":
                reply.status = 502
    return reply


async def _legacy(rig, endpoint, payload, stream):
    return await _wrapped(
        rig, "/v1/submit",
        {"agent_id": "bleed-battery", "endpoint": endpoint,
         "payload_type": "chat_completion", "call_site": "bleed.test",
         "timeout_s": 30}, payload, stream)


async def _enriched(rig, endpoint, payload, stream):
    return await _wrapped(
        rig, "/rs/v1/chat", {"model": endpoint, "deadline_s": 30}, payload, stream)


DOORS = {"openai": _openai, "submit": _legacy, "rs_chat": _enriched}


def _payload(**extra) -> dict:
    return {"messages": [{"role": "user", "content": "add 7, 39 and 15 as digits"}],
            "max_tokens": 64, **extra}


# --------------------------------------------------------------------------- #
# 0. The instrument is pointed at the right tree, and can see the defect
# --------------------------------------------------------------------------- #

def test_the_battery_is_running_against_the_tree_it_sits_in():
    """A green run against the WRONG tree is worthless. With an editable install
    pointing elsewhere, `import roadstead` resolves there unless PYTHONPATH says
    otherwise — so this fails loudly instead of letting the gate certify a tree it
    never imported."""
    tree = Path(__file__).resolve().parent.parent
    imported = Path(roadstead.__file__).resolve()
    assert tree in imported.parents, (
        f"`import roadstead` resolved to {imported}, outside the tree this battery "
        f"lives in ({tree}). Run with PYTHONPATH={tree}.")


async def test_sanity_the_fake_glm_backend_really_leaks_when_the_kwarg_reaches_it(rig):
    """The harness's own control. If the fake never leaked, every prevention test
    below would pass on ANY code — including the unfixed tree."""
    rig.fake.think = ThinkScript(THINK_GLM_VLLM, REASONING, ANSWER)
    async with httpx.AsyncClient(base_url=rig.server.url, timeout=10) as direct:
        for spelling in ({"chat_template_kwargs": {"enable_thinking": False}},
                         {"chat_template_kwargs": {"thinking": False}}):
            body = {"messages": [{"role": "user", "content": "x"}], **spelling}
            msg = (await direct.post("/v1/chat/completions", json=body)
                   ).json()["choices"][0]["message"]
            # No tag, no reasoning field: the trace is glued in front of the answer.
            assert msg["content"] == REASONING + ANSWER
            assert "reasoning" not in msg and "reasoning_content" not in msg

        clean = (await direct.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "x"}]})
        ).json()["choices"][0]["message"]
        assert clean["content"] == ANSWER and clean["reasoning"] == REASONING

        streamed = ""
        async with direct.stream("POST", "/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "x"}], "stream": True,
                "chat_template_kwargs": {"enable_thinking": False}}) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and "[DONE]" not in line:
                    for ch in json.loads(line[6:]).get("choices") or []:
                        streamed += (ch.get("delta") or {}).get("content") or ""
        assert streamed == REASONING + ANSWER


def test_declared_empty_is_distinguishable_from_not_declared(tmp_path):
    """`thinking_kwargs: []` (measured: no switch) and an absent key (unmeasured)
    are the same empty tuple. Only the first may license stripping a caller's
    switch, so it needs its own flag — through the real catalog loader."""
    eps = _endpoints(tmp_path, "127.0.0.1", 1)
    assert eps["tier3"].no_thinking_switch is True
    assert eps["tier3"].thinking_kwargs == ()
    assert eps["tier1"].no_thinking_switch is False       # says nothing
    assert eps["tier1"].thinking_kwargs == ()
    assert eps["tier2"].no_thinking_switch is False       # names a switch
    assert eps["tier2"].thinking_kwargs == ("enable_thinking",)


# --------------------------------------------------------------------------- #
# 1. PREVENTION — the switch never reaches an endpoint that has none
# --------------------------------------------------------------------------- #

SPELLINGS = {
    "thinking_false": {"chat_template_kwargs": {"thinking": False}},
    "enable_thinking_false": {"chat_template_kwargs": {"enable_thinking": False}},
    "both_false": {"chat_template_kwargs": {"thinking": False,
                                            "enable_thinking": False}},
    "extra_body_enable_thinking_false":
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    "extra_body_thinking_false":
        {"extra_body": {"chat_template_kwargs": {"thinking": False}}},
    "reasoning_effort_none": {"reasoning_effort": "none"},
    "reasoning_object_effort_none": {"reasoning": {"effort": "none"}},
}

_SWITCH_NAMES = ("thinking", "enable_thinking")


def _assert_no_switch_on_the_wire(body: dict) -> None:
    """The MECHANISM, asserted beside (never instead of) the semantic check."""
    ck = body.get("chat_template_kwargs") or {}
    assert not any(k in ck for k in _SWITCH_NAMES), body
    assert (body.get("reasoning_effort") or "").lower() != "none", body
    assert (body.get("reasoning") or {}).get("effort") != "none", body
    assert "extra_body" not in body or not (
        (body["extra_body"].get("chat_template_kwargs") or {}).keys()
        & set(_SWITCH_NAMES)), body


def _assert_the_answer_and_only_the_answer(reply: Reply, answer: str) -> None:
    assert reply.status == 200, reply.text[:300]
    assert reply.content == answer, (
        f"caller-visible content is {reply.content!r}, the model intended {answer!r}")
    assert REASONING not in reply.content
    assert reply.reasoning.strip() == REASONING, (
        "the reasoning must arrive in the reasoning field, not vanish")


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
@pytest.mark.parametrize("door", DOORS)
async def test_no_switch_reaches_a_no_switch_endpoint(rig, door, stream, spelling):
    rig.fake.think = ThinkScript(THINK_GLM_VLLM, REASONING, ANSWER)
    reply = await DOORS[door](rig, "tier3", _payload(**SPELLINGS[spelling]), stream)
    _assert_the_answer_and_only_the_answer(reply, ANSWER)
    _assert_no_switch_on_the_wire(rig.last_wire_body())


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
async def test_a_tool_turn_keeps_its_call_and_gets_no_trace(rig, stream, spelling):
    rig.fake.think = ThinkScript(THINK_GLM_VLLM, REASONING, "", tool_calls=[TOOL_CALL])
    reply = await _openai(
        rig, "tier3", _payload(tools=TOOLS, **SPELLINGS[spelling]), stream)
    assert reply.status == 200, reply.text[:300]
    assert reply.content == "", f"a tool turn has no prose; got {reply.content!r}"
    assert reply.reasoning.strip() == REASONING
    assert [tc["function"]["name"] for tc in reply.tool_calls] == ["get_time"]
    assert reply.finish == "tool_calls"
    _assert_no_switch_on_the_wire(rig.last_wire_body())


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
async def test_a_structured_reply_is_only_the_json(rig, stream, spelling):
    rig.fake.think = ThinkScript(THINK_GLM_VLLM, REASONING, JSON_ANSWER)
    reply = await _openai(
        rig, "tier3",
        _payload(response_format=SCHEMA_FORMAT, **SPELLINGS[spelling]), stream)
    _assert_the_answer_and_only_the_answer(reply, JSON_ANSWER)
    assert json.loads(reply.content) == {"total": 73915}
    _assert_no_switch_on_the_wire(rig.last_wire_body())


# --- a declared switch, spelled differently by the caller ------------------- #
#
# The other half of "put the switch in the form the endpoint reads". `tier2` declares
# `[enable_thinking]` and its fake engine 400s a `thinking` (measured on a
# Qwen-family engine that parses it as an Anthropic-style object), which the proxy
# surfaces as a 502. The caller's INTENT must arrive, under the key the template reads.

FOREIGN = {
    "thinking_false": ({"chat_template_kwargs": {"thinking": False}},
                       {"enable_thinking": False}),
    "thinking_true": ({"chat_template_kwargs": {"thinking": True}},
                      {"enable_thinking": True}),
    "extra_body_thinking_false":
        ({"extra_body": {"chat_template_kwargs": {"thinking": False}}},
         {"enable_thinking": False}),
    # The caller sent BOTH: its own declared key is the one it meant.
    "both_declared_wins": ({"chat_template_kwargs": {"thinking": True,
                                                     "enable_thinking": False}},
                           {"enable_thinking": False}),
}


def _wire_ck(rig: Rig) -> dict:
    wire = rig.last_wire_body()
    return wire.get("chat_template_kwargs") or {}


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
@pytest.mark.parametrize("door", DOORS)
async def test_a_foreign_switch_spelling_is_carried_onto_the_declared_key(
        rig, door, stream):
    rig.fake.reject_chat_template_kwargs = ("thinking",)
    rig.fake.think = ThinkScript(THINK_SWITCHABLE, REASONING, ANSWER)
    reply = await DOORS[door](
        rig, "tier2", _payload(**FOREIGN["thinking_false"][0]), stream)
    assert reply.status == 200, reply.text[:300]
    assert reply.content == ANSWER
    assert reply.reasoning == "", "the caller said thinking OFF; the value must survive"
    assert _wire_ck(rig) == {"enable_thinking": False}


@pytest.mark.parametrize("case", FOREIGN)
@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
async def test_every_foreign_spelling_keeps_the_callers_value(rig, case, stream):
    sent, wire_ck = FOREIGN[case]
    rig.fake.reject_chat_template_kwargs = ("thinking",)
    rig.fake.think = ThinkScript(THINK_SWITCHABLE, REASONING, ANSWER)
    reply = await _openai(rig, "tier2", _payload(**sent), stream)
    assert reply.status == 200, reply.text[:300]
    assert reply.content == ANSWER
    assert _wire_ck(rig) == wire_ck
    # The value did its job: ON -> the engine reasoned into the reasoning field,
    # OFF -> it did not reason at all.
    assert (reply.reasoning.strip() == REASONING) is wire_ck["enable_thinking"]


async def test_a_rename_is_counted_and_logged(rig, caplog):
    rig.fake.reject_chat_template_kwargs = ("thinking",)
    rig.fake.think = ThinkScript(THINK_SWITCHABLE, REASONING, ANSWER)
    with caplog.at_level(logging.WARNING, logger="roadstead.think_bleed"):
        await _openai(rig, "tier2", _payload(**FOREIGN["thinking_false"][0]), False)
    rel = await rig.status()
    assert rel["think_switch_renamed"] == 1
    assert rel["think_switch_renamed_by_endpoint"]["tier2"]["renames"] == {
        "chat_template_kwargs.thinking->enable_thinking=false": 1}
    assert "ROADSTEAD_THINK_SWITCH_RENAMED" in caplog.text
    assert rel["think_switch_stripped"] == 0     # a rename is not a strip


async def test_the_declared_spelling_is_forwarded_untouched(rig):
    rig.fake.reject_chat_template_kwargs = ("thinking",)
    rig.fake.think = ThinkScript(THINK_SWITCHABLE, REASONING, ANSWER)
    reply = await _openai(
        rig, "tier2", _payload(chat_template_kwargs={"enable_thinking": True}), False)
    assert reply.status == 200 and reply.content == ANSWER
    assert _wire_ck(rig) == {"enable_thinking": True}
    assert await rig.counter("think_switch_renamed") == 0


async def test_an_endpoint_declaring_both_spellings_renames_nothing(rig):
    """A template that reads either name (DeepSeek-style) declares both, and then
    NEITHER spelling is foreign."""
    rig.svc._config.endpoints["tier1"].thinking_kwargs = ("thinking", "enable_thinking")
    rig.fake.think = ThinkScript(THINK_CLEAN, REASONING, ANSWER,
                                 reasoning_key="reasoning_content")
    reply = await _openai(
        rig, "tier1", _payload(chat_template_kwargs={"thinking": False}), False)
    assert reply.status == 200 and reply.content == ANSWER
    assert _wire_ck(rig) == {"thinking": False}
    assert await rig.counter("think_switch_renamed") == 0


async def test_an_undeclared_endpoint_is_never_renamed(rig):
    """`tier1` declares nothing — we do not know which key its template reads, so
    the caller's spelling goes through as sent."""
    rig.fake.think = ThinkScript(THINK_CLEAN, REASONING, ANSWER,
                                 reasoning_key="reasoning_content")
    reply = await _openai(
        rig, "tier1", _payload(chat_template_kwargs={"thinking": False}), False)
    assert reply.status == 200
    assert _wire_ck(rig) == {"thinking": False}
    assert await rig.counter("think_switch_renamed") == 0


def test_rename_carries_the_value_and_never_mutates():
    rename = _need(rename_thinking_switch, "think_bleed.rename_thinking_switch")
    declared = ("enable_thinking",)
    original = {"messages": [], "chat_template_kwargs": {"thinking": False, "keep": 1},
                "extra_body": {"chat_template_kwargs": {"thinking": True}}}
    snapshot = json.loads(json.dumps(original))
    out, found = rename(original, declared)
    assert original == snapshot
    assert out["chat_template_kwargs"] == {"keep": 1, "enable_thinking": False}
    assert out["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert len(found) == 2


def test_rename_drops_a_disagreeing_pair_rather_than_pick_one():
    rename = _need(rename_thinking_switch, "think_bleed.rename_thinking_switch")
    out, found = rename(
        {"chat_template_kwargs": {"thinking": True, "enable_thinking": False}},
        ("some_other_key",))
    assert out["chat_template_kwargs"] == {}
    assert found and "dropped" in found[0]


@pytest.mark.parametrize("value,carried", [
    ({"type": "enabled", "budget_tokens": 1024}, True),
    ({"type": "disabled"}, False),
    ({"type": "adaptive"}, True),
    ("yes", None), (1, None), ({"budget_tokens": 5}, None)])
def test_an_anthropic_style_object_is_read_for_what_it_says_or_dropped(value, carried):
    """The very reading that makes an engine 400 a bare `thinking` is also the one a
    caller may have MEANT. Its `type` is carried; a value with no boolean meaning is
    dropped, because renaming an object onto a boolean key only moves the 400."""
    rename = _need(rename_thinking_switch, "think_bleed.rename_thinking_switch")
    out, _ = rename({"chat_template_kwargs": {"thinking": value}}, ("enable_thinking",))
    expected = {} if carried is None else {"enable_thinking": carried}
    assert out["chat_template_kwargs"] == expected


def test_rename_is_a_no_op_when_nothing_is_foreign():
    rename = _need(rename_thinking_switch, "think_bleed.rename_thinking_switch")
    for declared, ck in [(("enable_thinking",), {"enable_thinking": False}),
                         (("thinking", "enable_thinking"), {"thinking": False}),
                         (("enable_thinking",), {"reasoning_effort": "low"}),
                         ((), {"thinking": False})]:
        p = {"messages": [], "chat_template_kwargs": ck}
        out, found = rename(p, declared)
        assert out is p and found == []


# --- the caller's own switch beats the endpoint DEFAULT ------------------------ #
#
# Measured live on 139494c: tier2-flash declares `[enable_thinking]` and forces
# reasoning. A caller's `chat_template_kwargs.thinking: false` was DROPPED (counter
# `thinking->dropped`) and the reply carried reasoning; `enable_thinking: false` gave
# none. Cause: `apply_forced_reasoning_budget` wrote the declared key (the endpoint
# default, True) BEFORE the last-hop rename ran, so the rename read a proxy-injected
# value as the caller's own and let it win. The caller's spelling is now carried onto
# the declared key at INTAKE, before any default is written.

FORCED_OFF_SHAPES = {
    "ck": {"chat_template_kwargs": {"thinking": False}},
    "extra_body": {"extra_body": {"chat_template_kwargs": {"thinking": False}}},
}


def _forced_switchable(rig: Rig) -> None:
    rig.fake.reject_chat_template_kwargs = ("thinking",)
    rig.fake.think = ThinkScript(THINK_SWITCHABLE, REASONING, ANSWER)


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
@pytest.mark.parametrize("door", DOORS)
@pytest.mark.parametrize("shape", FORCED_OFF_SHAPES)
async def test_a_foreign_off_beats_the_endpoint_default_on_a_forced_reasoner(
        rig, shape, door, stream):
    _forced_switchable(rig)
    reply = await DOORS[door](rig, "flash", _payload(**FORCED_OFF_SHAPES[shape]), stream)
    assert reply.status == 200, reply.text[:300]
    assert reply.content == ANSWER
    assert reply.reasoning == "", (
        "the caller said thinking OFF; the endpoint default must not turn it back on")
    ck = _wire_ck(rig)
    assert ck.get("enable_thinking") is False, ck
    assert "thinking" not in ck, ck
    prefix = "extra_body." if shape == "extra_body" else ""
    rel = await rig.status()
    assert rel["think_switch_renamed_by_endpoint"]["flash"]["renames"] == {
        f"{prefix}chat_template_kwargs.thinking->enable_thinking=false": 1}, (
        "counted once, and never as `dropped`")


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
async def test_a_foreign_on_is_carried_and_the_declared_effort_still_applies(rig, stream):
    _forced_switchable(rig)
    reply = await _openai(
        rig, "flash", _payload(chat_template_kwargs={"thinking": True}), stream)
    assert reply.status == 200 and reply.content == ANSWER
    assert reply.reasoning.strip() == REASONING
    assert _wire_ck(rig) == {"enable_thinking": True, "reasoning_effort": "low"}


@pytest.mark.parametrize("sent,wire_switch", [
    ({"thinking": False, "enable_thinking": True}, True),
    ({"thinking": True, "enable_thinking": False}, False)],
    ids=["declared_on_foreign_off", "declared_off_foreign_on"])
@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
async def test_both_spellings_disagreeing_the_declared_key_wins(
        rig, sent, wire_switch, stream):
    """The documented rule: a caller who sent the DECLARED key meant it, and the
    foreign spelling is dropped. Only a caller-sent key qualifies — the proxy's own
    default never did (the test above)."""
    _forced_switchable(rig)
    reply = await _openai(rig, "flash", _payload(chat_template_kwargs=sent), stream)
    assert reply.status == 200 and reply.content == ANSWER
    ck = _wire_ck(rig)
    assert ck.get("enable_thinking") is wire_switch and "thinking" not in ck, ck
    assert (reply.reasoning.strip() == REASONING) is wire_switch


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
async def test_a_caller_that_says_nothing_gets_the_endpoint_default(rig, stream):
    """CONTROL: the default must still apply, or the fix above is just "never
    inject"."""
    _forced_switchable(rig)
    reply = await _openai(rig, "flash", _payload(), stream)
    assert reply.status == 200 and reply.content == ANSWER
    assert reply.reasoning.strip() == REASONING
    assert _wire_ck(rig) == {"enable_thinking": True, "reasoning_effort": "low"}
    assert await rig.counter("think_switch_renamed") == 0


# --- composition with the effort policy -------------------------------------- #

async def test_a_callers_real_effort_still_works_on_a_no_switch_endpoint(rig):
    """"none" cannot mean off where there is no switch — but "high" is a real lever
    and must still arrive."""
    rig.fake.think = ThinkScript(THINK_GLM_VLLM, REASONING, ANSWER)
    reply = await _openai(rig, "tier3", _payload(reasoning_effort="high"), False)
    _assert_the_answer_and_only_the_answer(reply, ANSWER)
    wire = rig.last_wire_body()
    assert (wire.get("reasoning_effort")
            or (wire.get("chat_template_kwargs") or {}).get("reasoning_effort")) == "high"


async def test_effort_none_alone_is_dropped_so_the_endpoint_default_applies(rig):
    rig.fake.think = ThinkScript(THINK_GLM_VLLM, REASONING, ANSWER)
    reply = await _openai(rig, "tier3", _payload(reasoning_effort="none"), False)
    _assert_the_answer_and_only_the_answer(reply, ANSWER)
    wire = rig.last_wire_body()
    assert "reasoning_effort" not in wire
    assert "reasoning_effort" not in (wire.get("chat_template_kwargs") or {})


async def test_thinking_true_with_effort_none_takes_the_declared_effort(rig):
    """The caller asked to think AND said "none". Thinking outranks it (the rule
    `fold_caller_effort` already applies), so the endpoint's declared
    `thinking_effort` is what goes out — and still no switch."""
    rig.fake.think = ThinkScript(THINK_GLM_VLLM, REASONING, ANSWER)
    reply = await _openai(
        rig, "tier3", _payload(thinking=True, reasoning_effort="none"), False)
    _assert_the_answer_and_only_the_answer(reply, ANSWER)
    wire = rig.last_wire_body()
    assert (wire.get("chat_template_kwargs") or {}).get("reasoning_effort") == "low"
    _assert_no_switch_on_the_wire(wire)


# --- the counter and the log line -------------------------------------------- #

async def test_a_strip_is_counted_and_logged(rig, caplog):
    rig.fake.think = ThinkScript(THINK_GLM_VLLM, REASONING, ANSWER)
    with caplog.at_level(logging.WARNING, logger="roadstead.think_bleed"):
        await _openai(
            rig, "tier3", _payload(chat_template_kwargs={"enable_thinking": False}),
            False)
    rel = await rig.status()
    assert rel["think_switch_stripped"] == 1
    row = rel["think_switch_stripped_by_endpoint"]["tier3"]
    assert row["spellings"] == {"chat_template_kwargs.enable_thinking=false": 1}
    assert "ROADSTEAD_THINK_SWITCH_STRIPPED" in caplog.text


async def test_the_status_row_says_the_endpoint_declared_no_switch(rig):
    body = (await rig.client.get("/v1/status")).json()
    eps = body.get("endpoints") or {}
    row = eps.get("tier3", {})
    assert row.get("thinking") == {"kwargs": [], "no_switch": True}, row


# --- the answer-now re-ask ---------------------------------------------------- #

@pytest.mark.parametrize("declared", ["no_switch", "switch", "undeclared"])
def test_answer_now_builds_its_switch_from_the_declaration(declared):
    _need(_answer_now_switch_off, "lifecycle._answer_now_switch_off")
    ep = dataclasses.make_dataclass("Ep", ["thinking_kwargs", "no_thinking_switch"])
    if declared == "no_switch":
        assert _answer_now_switch_off(ep((), True)) == {}
    elif declared == "switch":
        assert _answer_now_switch_off(ep(("enable_thinking",), False)) == {
            "enable_thinking": False}
    else:
        assert _answer_now_switch_off(ep((), False)) == {
            "thinking": False, "enable_thinking": False}


def _loop_text() -> str:
    return "Need to check the table again, same cell, same value. " * 80


def _arm_answer_now(rig: Rig, monkeypatch, endpoint: str = "tier3") -> None:
    monkeypatch.setenv("ROADSTEAD_PROXY_REASONING_LOOP_SHADOW", "0")
    monkeypatch.setenv("ROADSTEAD_PROXY_REASONING_LOOP_ANSWER_NOW", "1")
    ep = rig.svc._config.endpoints[endpoint]
    ep.reasoning_loop_window_chars = 800
    ep.reasoning_loop_min_chars = 800
    ep.reasoning_loop_max_distinct_ratio = 0.5
    ep.reasoning_loop_check_every_chars = 200


def _looping_then(reask: ThinkScript):
    """First call loops in its REASONING channel; the re-ask (recognisable by the
    `<notes>` block the proxy adds) gets `reask`."""
    def script(body: dict) -> ThinkScript:
        if any("<notes>" in str(m.get("content")) for m in body.get("messages", [])):
            return reask
        return ThinkScript(THINK_GLM_VLLM, _loop_text(), "never reached",
                           chunk_chars=40)
    return script


async def test_the_answer_now_reask_on_a_no_switch_endpoint_returns_only_the_answer(
        rig, monkeypatch):
    """The rescue call was itself a leak site: it hardcoded both switch spellings,
    so on a no-switch endpoint the re-ask's answer came back with the trace glued in
    front of it. Driven through the real stream: the first call loops in its
    REASONING channel, the detector breaks it, and the re-ask must deliver the answer
    alone."""
    _arm_answer_now(rig, monkeypatch)
    reask_reasoning = "Notes say the total is what was asked."
    rig.fake.think = _looping_then(
        ThinkScript(THINK_GLM_VLLM, reask_reasoning, ANSWER, chunk_chars=40))
    reply = await _openai(rig, "tier3", _payload(max_tokens=256), True)
    assert reply.status == 200, reply.text[-400:]
    assert len(rig.chat_bodies()) == 2, "the re-ask never happened — the test is blind"
    assert reply.content == ANSWER, f"caller-visible content is {reply.content!r}"
    assert reask_reasoning not in reply.content
    _assert_no_switch_on_the_wire(rig.chat_bodies()[1])
    assert (await rig.status())["reasoning_loops_answered"] == 1


async def test_a_reask_that_spends_its_budget_on_reasoning_is_not_counted_as_answered(
        rig, monkeypatch):
    """A no-switch endpoint cannot be told not to reason, so the re-ask can run out
    of budget inside its own trace: reasoning chunks, no content, `length`. That is a
    FAILED rescue. It must fall back to the loop-break error, and
    `reasoning_loops_answered` must not claim an answer nobody received."""
    _arm_answer_now(rig, monkeypatch)
    rig.fake.think = _looping_then(ThinkScript(
        THINK_GLM_VLLM, "still working it out " * 20, "", chunk_chars=40,
        truncate_in_reasoning=True))
    reply = await _openai(rig, "tier3", _payload(max_tokens=256), True)
    assert len(rig.chat_bodies()) == 2, "the re-ask never happened — the test is blind"
    assert reply.content == ""
    assert "the answer-now re-ask returned nothing" in reply.text
    assert (await rig.status())["reasoning_loops_answered"] == 0


async def test_the_reask_keeps_the_callers_other_template_kwargs(rig, monkeypatch):
    """A template variable describes the PROMPT, not the mode: dropping it changed
    what the re-ask rendered. Only the switch goes — and, on a no-switch endpoint
    that declares effort rungs, the effort is the endpoint's LOWEST (below)."""
    _arm_answer_now(rig, monkeypatch)
    rig.fake.think = _looping_then(
        ThinkScript(THINK_GLM_VLLM, "notes", ANSWER, chunk_chars=40))
    reply = await _openai(rig, "tier3", _payload(
        max_tokens=256,
        chat_template_kwargs={"enable_thinking": False, "my_var": 7}), True)
    assert reply.content == ANSWER
    assert rig.chat_bodies()[1]["chat_template_kwargs"] == {
        "reasoning_effort": "low", "my_var": 7}


@pytest.mark.parametrize("declared,expected", [
    ("no_switch", {"reasoning_effort": "high"}),
    ("switch", {"reasoning_effort": "high", "enable_thinking": False}),
    ("undeclared", {"reasoning_effort": "high", "thinking": False,
                    "enable_thinking": False}),
])
def test_answer_now_template_kwargs_per_declaration(declared, expected):
    build = _need(_answer_now_template_kwargs, "lifecycle._answer_now_template_kwargs")
    ep = dataclasses.make_dataclass("Ep", ["thinking_kwargs", "no_thinking_switch"])
    eps = {"no_switch": ep((), True), "switch": ep(("enable_thinking",), False),
           "undeclared": ep((), False)}
    payload = {"chat_template_kwargs": {"reasoning_effort": "high", "thinking": True,
                                        "enable_thinking": True}}
    assert build(payload, eps[declared]) == expected


# --- the re-ask on a no-switch endpoint is BOUNDED ---------------------------- #
#
# A no-switch endpoint cannot be told not to reason, so the re-ask reasons and could
# spend its whole `answer_max` on it (a failed rescue). Two bounds, both built from
# the endpoint's own declarations: its LOWEST declared effort rung, and (vLLM only, and
# only where the endpoint declares the launch flag that makes it legal) a
# `thinking_token_budget` with the answer reserve added on top of `max_tokens`.

#: tier3-shaped rungs: `none`/`minimal`/`low` all land on `low`; `ultra` is off the
#: ladder the code knows and must never be picked.
_RUNGS = {"none": "low", "minimal": "low", "low": "low", "medium": "high",
          "high": "high", "xhigh": "high", "max": "max", "ultra": "max"}


def _declare_bound(ep, *, effort_map=None, thinking_effort="", ratio=0.6):
    ep.reasoning_effort_map = dict(effort_map or {})
    ep.thinking_effort = thinking_effort
    ep.thinking_budget_ratio = ratio


async def _reask_body(rig, monkeypatch, endpoint="tier3", **payload) -> dict:
    _arm_answer_now(rig, monkeypatch, endpoint)
    rig.fake.think = _looping_then(
        ThinkScript(THINK_GLM_VLLM, "notes", ANSWER, chunk_chars=40))
    await _openai(rig, endpoint, _payload(**payload), True)
    assert len(rig.chat_bodies()) == 2, "the re-ask never happened — the test is blind"
    return rig.chat_bodies()[1]


async def test_the_reask_on_a_no_switch_vllm_endpoint_asks_for_the_lowest_effort_and_a_budget(
        rig, monkeypatch):
    _declare_bound(rig.svc._config.endpoints["tier3"], effort_map=_RUNGS,
                   thinking_effort="high")
    body = await _reask_body(
        rig, monkeypatch, max_tokens=256, reasoning_effort="high",
        chat_template_kwargs={"reasoning_effort": "max", "my_var": 7})
    assert body["chat_template_kwargs"] == {"reasoning_effort": "low", "my_var": 7}
    assert "reasoning_effort" not in body, (
        "a top-level effort beside the injected one is a 'conflicting reasoning_effort' 400")
    budget = body["thinking_token_budget"]
    assert budget == _need(_answer_now_reasoning_budget, "lifecycle._answer_now_reasoning_budget")(
        rig.svc._config.endpoints["tier3"], 256)
    assert 0 < budget
    assert body["max_tokens"] == 256 + budget, (
        "the answer reserve must sit ON TOP of the reasoning budget, not inside it")
    _assert_no_switch_on_the_wire(body)


async def test_the_reask_effort_also_replaces_one_nested_in_extra_body(rig, monkeypatch):
    """`extra_body` is merged over the top level by the provider, so an effort left in
    there would win over the lowest one."""
    _declare_bound(rig.svc._config.endpoints["tier3"], effort_map=_RUNGS)
    body = await _reask_body(
        rig, monkeypatch, max_tokens=256,
        extra_body={"chat_template_kwargs": {"reasoning_effort": "max", "my_var": 7}})
    assert body["chat_template_kwargs"] == {"reasoning_effort": "low", "my_var": 7}


async def test_a_reask_bound_is_absent_where_the_endpoint_declares_no_budget_support(
        rig, monkeypatch):
    """vLLM 400s the WHOLE request when `thinking_token_budget` arrives at a server
    launched without `--reasoning-config`; `thinking_budget_ratio` is the mirror of
    that flag. No declaration, no field — the effort still applies."""
    _declare_bound(rig.svc._config.endpoints["tier3"], effort_map=_RUNGS, ratio=0.0)
    body = await _reask_body(rig, monkeypatch, max_tokens=256)
    assert "thinking_token_budget" not in body
    assert body["max_tokens"] == 256
    assert body["chat_template_kwargs"] == {"reasoning_effort": "low"}


async def test_a_reask_on_an_endpoint_with_no_declared_rungs_invents_no_effort(
        rig, monkeypatch):
    """No map, no `thinking_effort`, no `reasoning_effort`: nothing to pick from, so
    nothing is injected and the caller's own effort is left as it was. The budget does
    not depend on the rungs."""
    _declare_bound(rig.svc._config.endpoints["tier3"])
    body = await _reask_body(
        rig, monkeypatch, max_tokens=256,
        chat_template_kwargs={"reasoning_effort": "high"})
    assert body["chat_template_kwargs"] == {"reasoning_effort": "high"}
    assert body["thinking_token_budget"] > 0


async def test_a_reask_on_a_no_switch_llamacpp_endpoint_gets_no_budget_field(
        rig, monkeypatch):
    """llama.cpp names its cap `reasoning_budget_tokens` and the bound here is vLLM's
    mechanism only: no field of either name, `max_tokens` untouched."""
    ep = rig.svc._config.endpoints["tier1"]
    ep.no_thinking_switch = True
    _declare_bound(ep, effort_map=_RUNGS)
    body = await _reask_body(rig, monkeypatch, endpoint="tier1", max_tokens=256)
    assert "thinking_token_budget" not in body
    assert "reasoning_budget_tokens" not in body
    # tier1 is a forced reasoner, so the first call's cap was padded; the re-ask
    # inherits it and adds nothing of its own.
    assert body["max_tokens"] == rig.chat_bodies()[0]["max_tokens"]


async def test_a_reask_on_a_switch_endpoint_is_unchanged(rig, monkeypatch):
    """The switch turns thinking off there; nothing about effort or a budget is
    added, whatever rungs and ratio the endpoint declares."""
    ep = rig.svc._config.endpoints["tier2"]
    ep.engine = "vllm"
    _declare_bound(ep, effort_map=_RUNGS, thinking_effort="low")
    body = await _reask_body(
        rig, monkeypatch, endpoint="tier2", max_tokens=256,
        chat_template_kwargs={"reasoning_effort": "high", "enable_thinking": True})
    assert body["chat_template_kwargs"] == {"reasoning_effort": "high",
                                            "enable_thinking": False}
    assert "thinking_token_budget" not in body
    assert body["max_tokens"] == 256


@pytest.mark.parametrize("answer_max,expected", [
    (64, 512),          # the floor: a tiny answer must not starve its own reasoning
    (256, 512),
    (2048, 1024),       # half the answer reserve
    (4000, 2000),
    (8000, 2000),       # the ceiling: the bound must actually bind
])
def test_the_reask_reasoning_budget_is_a_bounded_fraction_of_the_answer_reserve(
        tmp_path, answer_max, expected):
    budget = _need(_answer_now_reasoning_budget, "lifecycle._answer_now_reasoning_budget")
    ep = _endpoints(tmp_path, "127.0.0.1", 1)["tier3"]
    ep.thinking_budget_ratio = 0.6
    assert budget(ep, answer_max) == expected


def test_the_reask_reasoning_budget_is_zero_off_its_mechanism(tmp_path):
    budget = _need(_answer_now_reasoning_budget, "lifecycle._answer_now_reasoning_budget")
    eps = _endpoints(tmp_path, "127.0.0.1", 1)
    eps["tier3"].thinking_budget_ratio = 0.6
    assert budget(eps["tier3"], 256) > 0
    eps["tier3"].thinking_budget_ratio = 0.0
    assert budget(eps["tier3"], 256) == 0                # no launch flag declared
    eps["tier3"].reasoning_budget_tokens = 700
    assert budget(eps["tier3"], 256) > 0                 # an absolute cap declares it too
    for name in ("tier1", "tier2"):
        eps[name].thinking_budget_ratio = 0.6
    assert budget(eps["tier2"], 256) == 0                # has a switch
    eps["tier1"].no_thinking_switch = True
    assert budget(eps["tier1"], 256) == 0                # llama.cpp names it differently


@pytest.mark.parametrize("declared,expected", [
    ({"effort_map": _RUNGS}, "low"),
    ({"effort_map": {"medium": "high", "max": "max"}}, "high"),
    ({"effort_map": {"max": "max"}, "thinking_effort": "high"}, "high"),
    ({"thinking_effort": "LOW "}, "LOW"),                # verbatim, trimmed
    ({"effort_map": {"none": "none", "x": "ultra", "y": "bogus"}}, ""),
    ({"effort_map": {"a": "xhigh", "b": "max"}}, "xhigh"),
    ({}, "")])
def test_the_lowest_declared_effort_is_picked_from_declared_words_only(
        tmp_path, declared, expected):
    lowest = _need(_lowest_declared_effort, "lifecycle._lowest_declared_effort")
    ep = _endpoints(tmp_path, "127.0.0.1", 1)["tier3"]
    ep.reasoning_effort_map = dict(declared.get("effort_map", {}))
    ep.thinking_effort = declared.get("thinking_effort", "")
    assert lowest(ep) == expected


# --- unit level: the strip itself -------------------------------------------- #

def test_strip_removes_every_spelling_and_copies_rather_than_mutates():
    original = {
        "messages": [], "reasoning_effort": "none", "reasoning": {"effort": "none"},
        "chat_template_kwargs": {"thinking": False, "enable_thinking": True,
                                 "reasoning_effort": "none", "keep": 1},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False},
                       "reasoning_effort": "none", "other": 2},
    }
    snapshot = json.loads(json.dumps(original))
    out, found = _need(strip_thinking_switch, "think_bleed.strip_thinking_switch")(original)
    assert original == snapshot, "the caller's dict must never be mutated"
    assert out["chat_template_kwargs"] == {"keep": 1}
    assert "reasoning_effort" not in out and "reasoning" not in out
    assert out["extra_body"] == {"other": 2}
    assert len(found) == 7, found


def test_strip_leaves_other_efforts_and_untouched_payloads_alone():
    p = {"messages": [], "reasoning_effort": "high",
         "chat_template_kwargs": {"reasoning_effort": "low"}}
    out, found = _need(strip_thinking_switch, "think_bleed.strip_thinking_switch")(p)
    assert out is p and found == []


async def test_the_backend_strips_for_every_caller_of_the_pool(rig):
    """The choke point, driven directly: `call`, `stream` and the keep-alive touch
    all go through the pool, so a path that never sees `handle_submit` (the
    answer-now re-ask, a touch) is covered by the same strip."""
    rig.fake.think = ThinkScript(THINK_GLM_VLLM, REASONING, ANSWER)
    ep = rig.svc._config.endpoints["tier3"]
    pool: BackendClientPool = rig.svc._backend
    payload = _payload(chat_template_kwargs={"enable_thinking": False})

    resp = await pool.call(ep, payload, "chat_completion", "r-call")
    assert resp.body["choices"][0]["message"]["content"] == ANSWER
    content = ""
    async for ev in pool.stream(ep, {**payload, "stream": True},
                                "chat_completion", "r-stream"):
        for ch in ((ev.parsed or {}).get("choices") or []):
            content += (ch.get("delta") or {}).get("content") or ""
    assert content == ANSWER
    touch = _REAL_PROBE_PREFIX_TOUCH.__get__(pool, BackendClientPool)
    assert await touch(ep, payload, 5.0) is not None
    assert len(rig.chat_bodies()) == 3
    for body in rig.chat_bodies():
        _assert_no_switch_on_the_wire(body)


# --------------------------------------------------------------------------- #
# 2. REPAIR — a trace that carries a marker
# --------------------------------------------------------------------------- #

def _tagged(mode, *, answer=ANSWER, **kw) -> ThinkScript:
    return ThinkScript(mode, REASONING, answer, **kw)


REPAIR_MODES = {"open_tag": THINK_TAG_OPEN, "close_only": THINK_TAG_CLOSE_ONLY}


def _arm(rig: Rig, shape: str, stream: bool, endpoint: str = "tier1") -> None:
    """Streaming close-only repair holds content until the tag shows up, so it is an
    operator opt-in (`policy.repair_close_only_reasoning`). Everything else — the
    open-tag shape, and close-only when not streaming on a forced reasoner — needs
    no declaration, and is exercised WITHOUT one."""
    if shape == "close_only" and stream:
        rig.svc._config.endpoints[endpoint].repair_close_only_reasoning = True


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
@pytest.mark.parametrize("door", DOORS)
@pytest.mark.parametrize("shape", REPAIR_MODES)
async def test_a_tagged_trace_is_moved_to_the_reasoning_field(rig, shape, door, stream):
    _arm(rig, shape, stream)
    rig.fake.think = _tagged(REPAIR_MODES[shape])
    reply = await DOORS[door](rig, "tier1", _payload(), stream)
    _assert_the_answer_and_only_the_answer(reply, ANSWER)


@pytest.mark.parametrize("width", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("shape", REPAIR_MODES)
async def test_a_tag_split_across_stream_chunks_is_still_found(rig, shape, width):
    _arm(rig, shape, True)
    rig.fake.think = _tagged(REPAIR_MODES[shape], chunk_chars=width)
    reply = await _openai(rig, "tier1", _payload(), True)
    _assert_the_answer_and_only_the_answer(reply, ANSWER)


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
@pytest.mark.parametrize("shape", REPAIR_MODES)
async def test_a_tool_turn_with_a_tagged_trace_keeps_call_and_drops_trace(
        rig, shape, stream):
    _arm(rig, shape, stream)
    rig.fake.think = _tagged(REPAIR_MODES[shape], answer="", tool_calls=[TOOL_CALL])
    reply = await _openai(rig, "tier1", _payload(tools=TOOLS), stream)
    assert reply.status == 200, reply.text[:300]
    assert reply.content == ""
    assert reply.reasoning.strip() == REASONING
    assert [tc["function"]["name"] for tc in reply.tool_calls] == ["get_time"]


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
@pytest.mark.parametrize("shape", REPAIR_MODES)
async def test_a_structured_reply_behind_a_tagged_trace_is_only_the_json(
        rig, shape, stream):
    _arm(rig, shape, stream)
    rig.fake.think = _tagged(REPAIR_MODES[shape], answer=JSON_ANSWER)
    reply = await _openai(
        rig, "tier1", _payload(response_format=SCHEMA_FORMAT), stream)
    _assert_the_answer_and_only_the_answer(reply, JSON_ANSWER)
    assert json.loads(reply.content) == {"total": 73915}


async def test_a_truncation_inside_the_trace_never_becomes_content_streaming(rig):
    """`<think>…` with no close and `finish_reason=length`: there is NO answer.
    Content stays empty, the trace is reasoning, and the finish stays `length`."""
    rig.fake.think = _tagged(THINK_TAG_OPEN, truncate_in_reasoning=True)
    reply = await _openai(rig, "tier1", _payload(), True)
    assert reply.content == ""
    assert reply.reasoning.strip() == REASONING
    assert reply.finish == "length"


async def test_a_truncation_inside_the_trace_is_the_budget_error_not_a_200(rig):
    """Non-streaming, the repaired body is reasoning + empty content + `length`,
    which is EXACTLY the shape the empty-completion gate already turns into "spent
    its entire budget on REASONING" for a parser-split response. The repair runs
    before that gate so both shapes get the same, accurate error — and the trace is
    in no byte of what the caller receives."""
    rig.fake.think = _tagged(THINK_TAG_OPEN, truncate_in_reasoning=True)
    reply = await _openai(rig, "tier1", _payload(), False)
    assert reply.status == 502
    assert "REASONING" in reply.text and "finish_reason=length" in reply.text
    assert REASONING not in reply.text


def test_repair_message_keeps_length_and_puts_the_trace_in_the_reasoning_field():
    """The message-level truth behind the two tests above, without the gate."""
    msg = {"role": "assistant", "content": "<think>\n" + REASONING}
    out = _need(repair_message, "think_bleed.repair_message")(
        msg, close_only=False, default_key="reasoning_content")
    assert out == ("unclosed", len(REASONING), 0)
    assert msg["content"] == "" and msg["reasoning_content"] == REASONING


@pytest.mark.parametrize("engine_endpoint,key", [("tier3", "reasoning"),
                                                 ("tier1", "reasoning_content")])
async def test_the_reasoning_lands_under_the_engines_own_field_name(
        rig, engine_endpoint, key):
    """No third spelling: vLLM says `reasoning`, llama.cpp `reasoning_content`."""
    rig.fake.think = _tagged(THINK_TAG_OPEN)
    resp = await rig.client.post("/v1/chat/completions", json={
        "model": engine_endpoint, **_payload(), "stream": False})
    msg = resp.json()["choices"][0]["message"]
    assert msg[key].strip() == REASONING
    assert msg["content"] == ANSWER
    assert {"reasoning", "reasoning_content"} - {key} & msg.keys() == set()


async def test_a_repair_is_counted_and_logged(rig, caplog):
    rig.fake.think = _tagged(THINK_TAG_OPEN)
    with caplog.at_level(logging.WARNING, logger="roadstead.think_bleed"):
        await _openai(rig, "tier1", _payload(), False)
        await _openai(rig, "tier1", _payload(), True)
    rel = await rig.status()
    assert rel["think_bleed_repaired"] == 2
    row = rel["think_bleed_repaired_by_endpoint"]["tier1"]
    assert row["sync"] == 1 and row["stream"] == 1
    assert row["shapes"] == {"open_tag": 2}
    assert "ROADSTEAD_THINK_BLEED_REPAIRED" in caplog.text


# --------------------------------------------------------------------------- #
# 3. CONTROLS — the harness can see the things that must NOT change
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
async def test_a_declared_switch_is_forwarded_untouched(rig, stream):
    """A Qwen-style endpoint declares `[enable_thinking]`; its switch WORKS, so the
    caller's kwarg must reach it exactly as sent, and the answer comes back clean."""
    rig.fake.think = ThinkScript(THINK_SWITCHABLE, REASONING, ANSWER)
    reply = await _openai(
        rig, "tier2", _payload(chat_template_kwargs={"enable_thinking": False}), stream)
    assert reply.status == 200, reply.text[:300]
    assert reply.content == ANSWER and reply.reasoning == ""
    wire = rig.last_wire_body()
    assert wire["chat_template_kwargs"] == {"enable_thinking": False}
    assert await rig.counter("think_switch_stripped") == 0


@pytest.mark.parametrize("spelling", ["enable_thinking_false", "thinking_false",
                                      "extra_body_enable_thinking_false"])
async def test_an_undeclared_endpoint_keeps_the_callers_kwargs(rig, spelling):
    """`tier1` says NOTHING about a switch. Not declared is not declared-empty:
    the proxy does not know this template, so the caller's payload is left alone."""
    rig.fake.think = ThinkScript(THINK_CLEAN, REASONING, ANSWER,
                                 reasoning_key="reasoning_content")
    reply = await _openai(rig, "tier1", _payload(**SPELLINGS[spelling]), False)
    assert reply.status == 200 and reply.content == ANSWER
    wire = rig.last_wire_body()
    sent = SPELLINGS[spelling]
    ck = (sent.get("extra_body") or sent)["chat_template_kwargs"]
    got = wire.get("chat_template_kwargs") or {}
    assert all(got.get(k) is v for k, v in ck.items()), wire
    assert await rig.counter("think_switch_stripped") == 0


UNMARKED = ["73915", "  padded \n\n", "<b>bold</b> and <i>not a think tag</i>",
            "a < b and b > c", "<thin", "", "\n\nline one\n  indented line two"]


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
@pytest.mark.parametrize("text", [t for t in UNMARKED if t])
@pytest.mark.parametrize("endpoint", ["tier1", "tier3"])
async def test_content_with_no_marker_is_byte_identical(rig, endpoint, text, stream):
    """On the two endpoints that DO declare reasoning, so repair is armed and holding
    — and still must not alter a byte of a reply that carries no marker."""
    rig.fake.think = ThinkScript(THINK_CLEAN, "", text, chunk_chars=2)
    reply = await _openai(rig, endpoint, _payload(), stream)
    assert reply.status == 200, reply.text[:300]
    assert reply.content == text
    assert await rig.counter("think_bleed_repaired") == 0


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
async def test_a_tag_in_the_answer_is_left_alone_when_the_engine_already_split(
        rig, stream):
    """The reasoning field is populated, so the engine did its job and `</think>` in
    content is the answer TALKING ABOUT the tag. Rewriting it would corrupt it."""
    answer = "Close the block with </think> before the answer."
    rig.fake.think = ThinkScript(THINK_CLEAN, REASONING, answer, chunk_chars=3)
    reply = await _openai(rig, "tier1", _payload(), stream)
    assert reply.content == answer
    assert reply.reasoning.strip() == REASONING


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
async def test_repair_does_not_act_on_an_endpoint_that_declares_no_reasoning(
        rig, stream):
    """`tier2` declares no `reasoning` capability: whatever its content holds is
    delivered as the backend sent it, tag and all."""
    rig.fake.think = _tagged(THINK_TAG_OPEN)
    reply = await _openai(rig, "tier2", _payload(), stream)
    assert reply.content == "<think>\n" + REASONING + "\n</think>\n" + ANSWER
    assert await rig.counter("think_bleed_repaired") == 0


# --------------------------------------------------------------------------- #
# 4. The stream repairer on its own — every chunking of every shape
# --------------------------------------------------------------------------- #

def _repairer(**kw):
    return _need(StreamRepair, "think_bleed.StreamRepair")(**kw)


def _frames(*deltas: dict, finish: str = "stop") -> list[tuple[dict, str]]:
    out = []
    for d in deltas:
        p = {"id": "c", "object": "chat.completion.chunk",
             "choices": [{"index": 0, "delta": d, "finish_reason": None}]}
        out.append((p, json.dumps(p)))
    if finish:
        p = {"id": "c", "object": "chat.completion.chunk",
             "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
        out.append((p, json.dumps(p)))
    return out


def _run(repair: StreamRepair, frames) -> tuple[str, str, str | None]:
    content = reasoning = ""
    finish = None
    emitted = []
    for parsed, data in frames:
        emitted += repair.feed(parsed, data)
    emitted += repair.finish()
    for parsed, _ in emitted:
        for ch in parsed.get("choices") or []:
            d = ch.get("delta") or {}
            content += d.get("content") or ""
            reasoning += d.get("reasoning") or d.get("reasoning_content") or ""
            finish = ch.get("finish_reason") or finish
    return content, reasoning, finish


def _slice(text: str, n: int) -> list[dict]:
    return [{"content": text[i:i + n]} for i in range(0, len(text), n)]


@pytest.mark.parametrize("width", range(1, 14))
@pytest.mark.parametrize("raw,reasoning,answer", [
    ("<think>\nR R R\n</think>\nthe answer", "R R R", "the answer"),
    ("  <think>R</think>A", "R", "A"),
    ("R R R\n</think>\nthe answer", "R R R", "the answer"),
    ("R</think>A", "R", "A"),
    ("<think>R</think>", "R", ""),
    # ONE newline is the template's, and only that is removed: everything the
    # ANSWER starts with (a blank line, indentation) is the answer's own.
    ("<think>R</think>\n\n\n   A  B", "R", "\n\n   A  B"),
    ("<think>R</think>\r\n    code()", "R", "    code()"),
    ("R</think>\n    indented", "R", "    indented"),
], ids=["open", "open_lead_ws", "close_only", "close_only_tiny", "no_answer",
        "only_one_newline", "crlf", "indentation_survives"])
def test_every_chunking_of_every_marked_shape_splits_the_same(
        width, raw, reasoning, answer):
    r = _repairer(hold_for_close_tag=True, default_key="reasoning")
    content, got_reasoning, finish = _run(r, _frames(*_slice(raw, width)))
    assert content == answer
    assert got_reasoning.strip() == reasoning
    assert finish == "stop" and r.repaired


@pytest.mark.parametrize("width", range(1, 10))
def test_an_unclosed_trace_is_reasoning_and_the_finish_survives(width):
    r = _repairer(hold_for_close_tag=False, default_key="reasoning")
    raw = "<think>\nstill thinking when the budget ran out"
    content, reasoning, finish = _run(r, _frames(*_slice(raw, width), finish="length"))
    assert content == ""
    assert reasoning.strip() == "still thinking when the budget ran out"
    assert finish == "length" and r.shape == "unclosed"


@pytest.mark.parametrize("width", range(1, 10))
@pytest.mark.parametrize("raw", ["plain answer", "a </think> b", "<b>hi</b>",
                                 "<thinking about it>", " \n\nspaced"])
def test_unmarked_content_is_released_byte_identical(width, raw):
    """Without an opening tag and NOT expected to reason, nothing is ever held past
    the first chunk that rules out `<think>` — and what comes out is what went in."""
    r = _repairer(hold_for_close_tag=False, default_key="reasoning")
    content, reasoning, _ = _run(r, _frames(*_slice(raw, width)))
    assert content == raw and reasoning == "" and not r.repaired


@pytest.mark.parametrize("width", range(1, 10))
def test_an_expected_reasoner_holds_unmarked_content_then_releases_it_whole(width):
    r = _repairer(hold_for_close_tag=True, default_key="reasoning")
    raw = "just an answer with no marker anywhere"
    content, reasoning, _ = _run(r, _frames(*_slice(raw, width)))
    assert content == raw and reasoning == "" and not r.repaired


def test_a_reasoning_field_on_the_wire_means_the_engine_split_it():
    r = _repairer(hold_for_close_tag=True, default_key="reasoning")
    frames = _frames({"reasoning": "thinking"}, {"content": "a </think> b"})
    content, reasoning, _ = _run(r, frames)
    assert content == "a </think> b" and reasoning == "thinking" and not r.repaired


def test_the_hold_is_bounded():
    r = _repairer(hold_for_close_tag=True, default_key="reasoning", hold_limit=20)
    raw = "x" * 50 + "</think>after"
    content, reasoning, _ = _run(r, _frames(*_slice(raw, 5)))
    assert content == raw and reasoning == ""


def test_untouched_frames_keep_their_original_bytes():
    r = _repairer(hold_for_close_tag=False, default_key="reasoning")
    frames = _frames({"role": "assistant"}, {"content": "hello"}, {"content": " there"})
    out = []
    for parsed, data in frames:
        out += r.feed(parsed, data)
    assert [d for _, d in out] == [d for _, d in frames]


def test_a_usage_only_frame_and_a_second_choice_pass_through():
    r = _repairer(hold_for_close_tag=True, default_key="reasoning")
    usage = {"id": "c", "choices": [], "usage": {"completion_tokens": 3}}
    other = {"id": "c", "choices": [{"index": 1, "delta": {"content": "</think>x"}}]}
    assert r.feed(usage, json.dumps(usage)) == [(usage, json.dumps(usage))]
    assert r.feed(other, json.dumps(other)) == [(other, json.dumps(other))]


def test_tool_calls_arriving_while_holding_release_the_held_text_first():
    r = _repairer(hold_for_close_tag=True, default_key="reasoning")
    tool = {"tool_calls": [{"index": 0, "id": "c0", "type": "function",
                            "function": {"name": "f", "arguments": "{}"}}]}
    frames = _frames({"content": "some prose"}, tool, finish="tool_calls")
    out = []
    for parsed, data in frames:
        out += r.feed(parsed, data)
    kinds = [("content" if (p["choices"][0]["delta"].get("content")) else
              "tool" if p["choices"][0]["delta"].get("tool_calls") else "finish")
             for p, _ in out]
    assert kinds == ["content", "tool", "finish"]


# --------------------------------------------------------------------------- #
# 5. What a healthy stream must NOT pay, and what an answer must NOT lose
# --------------------------------------------------------------------------- #
#
# The first cut of the repair held content back on any endpoint "expected to reason":
# every no-reasoning stream on tier3 (~30% of short calls, measured) and every
# switched-off stream on a forced reasoner. Time-to-first-token became total
# generation time. These pin that a healthy stream flows, and that a `</think>` in an
# answer is the answer's own text.

SLOW_S = 0.2
SENTENCE = "one sentence about the sea"


def _flash_shaped(rig: Rig) -> None:
    """A forced reasoner (llama.cpp + a reasoning model) with a declared switch."""
    ep = rig.svc._config.endpoints["tier1"]
    ep.thinking_kwargs = ("enable_thinking",)
    assert ep.forces_reasoning


async def _first_content_and_total(rig: Rig, endpoint: str, payload: dict):
    """Seconds to the first CONTENT event out of the backend pool, and to the end.
    Measured at `BackendClientPool.stream` because httpx's ASGI transport buffers a
    whole response body and would hide exactly the delay under test."""
    pool: BackendClientPool = rig.svc._backend
    ep = rig.svc._config.endpoints[endpoint]
    t0 = time.monotonic()
    first = None
    async for ev in pool.stream(ep, {**payload, "stream": True},
                                "chat_completion", "latency"):
        for ch in (ev.parsed or {}).get("choices") or []:
            if (ch.get("delta") or {}).get("content") and first is None:
                first = time.monotonic() - t0
    return first, time.monotonic() - t0


def _assert_flowing(first, total):
    assert first is not None
    assert total > SLOW_S * 4, "the fake was not slow — the test measures nothing"
    assert first < SLOW_S * 3 and first < total / 2, (
        f"first content chunk after {first:.2f}s of a {total:.2f}s stream — the "
        f"stream was held back")


async def test_a_no_reasoning_stream_on_a_no_switch_endpoint_is_not_held(rig):
    """tier3-shaped: parser on, the model answered without reasoning."""
    rig.fake.think = ThinkScript(THINK_CLEAN, "", SENTENCE, chunk_chars=5,
                                 token_delay_s=SLOW_S)
    _assert_flowing(*await _first_content_and_total(rig, "tier3", _payload()))


async def test_a_switched_off_stream_on_a_forced_reasoner_is_not_held(rig):
    """tier2-flash-shaped: forces_reasoning, `enable_thinking: false` streams
    content only. An explicit off means "not expected to reason"."""
    _flash_shaped(rig)
    rig.fake.think = ThinkScript(THINK_SWITCHABLE, REASONING, SENTENCE,
                                 chunk_chars=5, token_delay_s=SLOW_S)
    _assert_flowing(*await _first_content_and_total(
        rig, "tier1", _payload(chat_template_kwargs={"enable_thinking": False})))


async def test_a_forced_reasoner_with_no_reasoning_field_is_not_held_either(rig):
    """No kwargs at all, content only. Without an opt-in there is no reason to wait
    for a tag that has given no sign of coming."""
    rig.fake.think = ThinkScript(THINK_CLEAN, "", SENTENCE, chunk_chars=5,
                                 token_delay_s=SLOW_S)
    _assert_flowing(*await _first_content_and_total(rig, "tier1", _payload()))


FALSE_POSITIVE = "Close the block with </think> and then answer."
INDENTED = "    def f():\n        return 1"


@pytest.mark.parametrize("thinking", [False, True], ids=["plain", "thinking_on"])
@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
async def test_a_tag_in_the_answer_survives_on_a_tier3_shaped_endpoint(
        rig, stream, thinking):
    rig.fake.think = ThinkScript(THINK_CLEAN, "", FALSE_POSITIVE, chunk_chars=3)
    extra = {"thinking": True} if thinking else {}
    reply = await _openai(rig, "tier3", _payload(**extra), stream)
    assert reply.status == 200, reply.text[:300]
    assert reply.content == FALSE_POSITIVE
    assert await rig.counter("think_bleed_repaired") == 0


@pytest.mark.parametrize("opted_in", [False, True], ids=["default", "opted_in"])
@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
async def test_a_tag_in_the_answer_survives_on_a_switched_off_forced_reasoner(
        rig, stream, opted_in):
    """Even an endpoint that opted in to the close-only repair does not apply it to
    a call whose switch is explicitly off: that call is not reasoning."""
    _flash_shaped(rig)
    rig.svc._config.endpoints["tier1"].repair_close_only_reasoning = opted_in
    rig.fake.think = ThinkScript(THINK_SWITCHABLE, REASONING, FALSE_POSITIVE,
                                 chunk_chars=3)
    reply = await _openai(
        rig, "tier1", _payload(chat_template_kwargs={"enable_thinking": False}), stream)
    assert reply.status == 200, reply.text[:300]
    assert reply.content == FALSE_POSITIVE
    assert await rig.counter("think_bleed_repaired") == 0


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
@pytest.mark.parametrize("shape", REPAIR_MODES)
async def test_an_answer_that_starts_with_indentation_keeps_it(rig, shape, stream):
    """Only the template's ONE newline after the tag is removed."""
    _arm(rig, shape, stream)
    rig.fake.think = _tagged(REPAIR_MODES[shape], answer=INDENTED)
    reply = await _openai(rig, "tier1", _payload(), stream)
    assert reply.content == INDENTED


def test_reasoning_under_any_key_means_the_engine_split_it():
    repair = _need(repair_message, "think_bleed.repair_message")
    for msg in ({"content": "a</think>b", "reasoning": None, "reasoning_content": "x"},
                {"content": "a</think>b", "reasoning": "x", "reasoning_content": None},
                {"content": "<think>t</think>b", "reasoning_content": "x"}):
        before = dict(msg)
        assert repair(msg, close_only=True, default_key="reasoning") is None
        assert msg == before


# --- the close-only repair is narrow ------------------------------------------ #

async def test_close_only_is_not_repaired_on_an_endpoint_that_neither_opted_in_nor_forces(
        rig):
    """tier3 is not a forced reasoner and has not opted in: `R</think>A` is passed
    through as sent, because it cannot be told from an answer that mentions the tag."""
    rig.fake.think = _tagged(THINK_TAG_CLOSE_ONLY)
    reply = await _openai(rig, "tier3", _payload(), False)
    assert reply.content == REASONING + "\n</think>\n" + ANSWER


async def test_close_only_is_repaired_on_an_opted_in_endpoint_that_does_not_force(rig):
    rig.svc._config.endpoints["tier3"].repair_close_only_reasoning = True
    rig.fake.think = _tagged(THINK_TAG_CLOSE_ONLY)
    for stream in (False, True):
        reply = await _openai(rig, "tier3", _payload(), stream)
        _assert_the_answer_and_only_the_answer(reply, ANSWER)


async def test_streaming_close_only_without_the_opt_in_is_passed_through_unheld(rig):
    """The documented trade: a forced reasoner that has NOT opted in streams the
    close-only shape as it came (and, per the latency tests, promptly)."""
    rig.fake.think = _tagged(THINK_TAG_CLOSE_ONLY)
    reply = await _openai(rig, "tier1", _payload(), True)
    assert reply.content == REASONING + "\n</think>\n" + ANSWER


def test_the_close_only_opt_in_reaches_endpoint_config(tmp_path):
    """A `policy:` key missing from `_POLICY_PASSTHROUGH` is dropped in silence."""
    for policy, expected in (("repair_close_only_reasoning: true", True), ("{}", False)):
        text = _CATALOG.replace(
            "  tier1:\n    provider: cpp-box\n    kind: chat\n",
            "  tier1:\n    provider: cpp-box\n    kind: chat\n    policy: "
            + ("{" + policy + "}" if policy != "{}" else "{}") + "\n")
        path = tmp_path / f"models-{expected}.yaml"
        path.write_text(text)
        cat = model_catalog.load_catalog(path, force=True)
        kw = model_catalog.build_endpoint_kwargs(cat)["tier1"]
        assert EndpointConfig(**kw).repair_close_only_reasoning is expected


# --- the stream repairer: prefix hold, errors --------------------------------- #

def test_without_the_opt_in_the_prefix_hold_is_a_handful_of_characters():
    keep = _need(PREFIX_HOLD_CHARS, "think_bleed.PREFIX_HOLD_CHARS")
    # Text that is plainly not a tag is released by the very first chunk.
    r = _repairer(hold_for_close_tag=False, default_key="reasoning")
    frames = _frames(*_slice("hello there", 1), finish="")
    assert r.feed(*frames[0]) == [frames[0]]
    # A `<` could still become the tag, so it is held; `<b` proves it is not.
    r = _repairer(hold_for_close_tag=False, default_key="reasoning")
    frames = _frames(*_slice("<b>x", 1), finish="")
    assert r.feed(*frames[0]) == []
    released = r.feed(*frames[1])
    assert [f["choices"][0]["delta"]["content"] for f, _ in released] == ["<b"]
    # Whitespace cannot be held forever waiting for a tag.
    r = _repairer(hold_for_close_tag=False, default_key="reasoning")
    got = ""
    for parsed, data in _frames(*_slice(" " * (keep + 4), 1), finish=""):
        for f, _ in r.feed(parsed, data):
            got += f["choices"][0]["delta"].get("content") or ""
    assert got and len(got) <= keep + 4


def test_held_text_is_released_before_an_error_frame():
    r = _repairer(hold_for_close_tag=True, default_key="reasoning")
    held = _frames({"content": "the partial answer"}, finish="")
    assert r.feed(*held[0]) == []
    err = {"error": {"message": "boom", "type": "proxy_error"}}
    out = r.feed(err, json.dumps(err))
    assert out[0][0]["choices"][0]["delta"]["content"] == "the partial answer"
    assert out[-1] == (err, json.dumps(err))


# --- counter cardinality ------------------------------------------------------- #

def test_counter_keys_do_not_grow_with_caller_supplied_values():
    """A counter keyed on raw caller input is a caller-controlled allocation."""
    strip = _need(strip_thinking_switch, "think_bleed.strip_thinking_switch")
    rename = _need(rename_thinking_switch, "think_bleed.rename_thinking_switch")
    stats = _need(ThinkBleedStats, "think_bleed.ThinkBleedStats")()
    for i in range(1000):
        _, found = strip({"chat_template_kwargs": {"enable_thinking": f"value-{i}"}})
        stats.note_stripped("tier3", found, f"r{i}")
        _, found = rename(
            {"chat_template_kwargs": {"thinking": {"type": f"kind-{i}"}}},
            ("enable_thinking",))
        stats.note_renamed("tier2", found, f"r{i}")
        _, found = rename(
            {"chat_template_kwargs": {"thinking": f"value-{i}"}}, ("enable_thinking",))
        stats.note_renamed("tier2", found, f"r{i}")
    stripped = stats.switch_stripped_by_endpoint["tier3"]
    assert stripped["count"] == 1000
    assert set(stripped["spellings"]) == {"chat_template_kwargs.enable_thinking=str"}
    assert len(stats.switch_renamed_by_endpoint["tier2"]["renames"]) <= 3
    assert "value-" not in json.dumps(stats.switch_renamed_by_endpoint)


def test_a_bool_switch_keeps_its_value_in_the_label_because_false_is_the_leak():
    strip = _need(strip_thinking_switch, "think_bleed.strip_thinking_switch")
    labels = {}
    for v in (False, True, None, 3, "x", [1]):
        _, found = strip({"chat_template_kwargs": {"thinking": v}})
        labels[repr(v)] = found[0]
    assert labels["False"].endswith("=false") and labels["True"].endswith("=true")
    assert labels["None"].endswith("=null")
    assert labels["3"].endswith("=int") and labels["'x'"].endswith("=str")
    assert labels["[1]"].endswith("=list")
