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
        -p no:cacheprovider --timeout=120 --rootdir=<tree> -c <tree>/pyproject.toml

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
import json
import logging
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
try:
    from roadstead.think_bleed import (
        StreamRepair, repair_message, strip_thinking_switch,
    )
except ImportError:                                    # pragma: no cover
    StreamRepair = repair_message = strip_thinking_switch = None
try:
    from roadstead.lifecycle import _answer_now_switch_off
except ImportError:                                    # pragma: no cover
    _answer_now_switch_off = None


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


async def test_the_answer_now_reask_on_a_no_switch_endpoint_returns_only_the_answer(
        rig, monkeypatch):
    """The rescue call was itself a leak site: it hardcoded both switch spellings,
    so on a no-switch endpoint the re-ask's answer came back with the trace glued in
    front of it. Driven through the real stream: the first call loops in its
    REASONING channel, the detector breaks it, and the re-ask must deliver the answer
    alone."""
    monkeypatch.setenv("ROADSTEAD_PROXY_REASONING_LOOP_SHADOW", "0")
    monkeypatch.setenv("ROADSTEAD_PROXY_REASONING_LOOP_ANSWER_NOW", "1")
    ep = rig.svc._config.endpoints["tier3"]
    ep.reasoning_loop_window_chars = 800
    ep.reasoning_loop_min_chars = 800
    ep.reasoning_loop_max_distinct_ratio = 0.5
    ep.reasoning_loop_check_every_chars = 200

    reask_reasoning = "Notes say the total is what was asked."

    def script(body: dict) -> ThinkScript:
        if any("<notes>" in str(m.get("content")) for m in body.get("messages", [])):
            return ThinkScript(THINK_GLM_VLLM, reask_reasoning, ANSWER, chunk_chars=40)
        return ThinkScript(THINK_GLM_VLLM, _loop_text(), "never reached",
                           chunk_chars=40)

    rig.fake.think = script
    reply = await _openai(rig, "tier3", _payload(max_tokens=256), True)
    assert reply.status == 200, reply.text[-400:]
    assert len(rig.chat_bodies()) == 2, "the re-ask never happened — the test is blind"
    assert reply.content == ANSWER, f"caller-visible content is {reply.content!r}"
    assert reask_reasoning not in reply.content
    _assert_no_switch_on_the_wire(rig.chat_bodies()[1])


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


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
@pytest.mark.parametrize("door", DOORS)
@pytest.mark.parametrize("shape", REPAIR_MODES)
async def test_a_tagged_trace_is_moved_to_the_reasoning_field(rig, shape, door, stream):
    rig.fake.think = _tagged(REPAIR_MODES[shape])
    reply = await DOORS[door](rig, "tier1", _payload(), stream)
    _assert_the_answer_and_only_the_answer(reply, ANSWER)


@pytest.mark.parametrize("width", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("shape", REPAIR_MODES)
async def test_a_tag_split_across_stream_chunks_is_still_found(rig, shape, width):
    rig.fake.think = _tagged(REPAIR_MODES[shape], chunk_chars=width)
    reply = await _openai(rig, "tier1", _payload(), True)
    _assert_the_answer_and_only_the_answer(reply, ANSWER)


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
@pytest.mark.parametrize("shape", REPAIR_MODES)
async def test_a_tool_turn_with_a_tagged_trace_keeps_call_and_drops_trace(
        rig, shape, stream):
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
        msg, expects_reasoning=False, default_key="reasoning_content")
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
    assert reply.content == "<think>\n" + REASONING + "\n</think>\n\n" + ANSWER
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
    ("<think>\nR R R\n</think>\n\nthe answer", "R R R", "the answer"),
    ("  <think>R</think>A", "R", "A"),
    ("R R R\n</think>\n\nthe answer", "R R R", "the answer"),
    ("R</think>A", "R", "A"),
    ("<think>R</think>", "R", ""),
    ("<think>R</think>\n\n\n   A  B", "R", "A  B"),
], ids=["open", "open_lead_ws", "close_only", "close_only_tiny", "no_answer",
        "ws_after_tag"])
def test_every_chunking_of_every_marked_shape_splits_the_same(
        width, raw, reasoning, answer):
    r = _repairer(expects_reasoning=True, default_key="reasoning")
    content, got_reasoning, finish = _run(r, _frames(*_slice(raw, width)))
    assert content == answer
    assert got_reasoning.strip() == reasoning
    assert finish == "stop" and r.repaired


@pytest.mark.parametrize("width", range(1, 10))
def test_an_unclosed_trace_is_reasoning_and_the_finish_survives(width):
    r = _repairer(expects_reasoning=False, default_key="reasoning")
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
    r = _repairer(expects_reasoning=False, default_key="reasoning")
    content, reasoning, _ = _run(r, _frames(*_slice(raw, width)))
    assert content == raw and reasoning == "" and not r.repaired


@pytest.mark.parametrize("width", range(1, 10))
def test_an_expected_reasoner_holds_unmarked_content_then_releases_it_whole(width):
    r = _repairer(expects_reasoning=True, default_key="reasoning")
    raw = "just an answer with no marker anywhere"
    content, reasoning, _ = _run(r, _frames(*_slice(raw, width)))
    assert content == raw and reasoning == "" and not r.repaired


def test_a_reasoning_field_on_the_wire_means_the_engine_split_it():
    r = _repairer(expects_reasoning=True, default_key="reasoning")
    frames = _frames({"reasoning": "thinking"}, {"content": "a </think> b"})
    content, reasoning, _ = _run(r, frames)
    assert content == "a </think> b" and reasoning == "thinking" and not r.repaired


def test_the_hold_is_bounded():
    r = _repairer(expects_reasoning=True, default_key="reasoning", hold_limit=20)
    raw = "x" * 50 + "</think>after"
    content, reasoning, _ = _run(r, _frames(*_slice(raw, 5)))
    assert content == raw and reasoning == ""


def test_untouched_frames_keep_their_original_bytes():
    r = _repairer(expects_reasoning=False, default_key="reasoning")
    frames = _frames({"role": "assistant"}, {"content": "hello"}, {"content": " there"})
    out = []
    for parsed, data in frames:
        out += r.feed(parsed, data)
    assert [d for _, d in out] == [d for _, d in frames]


def test_a_usage_only_frame_and_a_second_choice_pass_through():
    r = _repairer(expects_reasoning=True, default_key="reasoning")
    usage = {"id": "c", "choices": [], "usage": {"completion_tokens": 3}}
    other = {"id": "c", "choices": [{"index": 1, "delta": {"content": "</think>x"}}]}
    assert r.feed(usage, json.dumps(usage)) == [(usage, json.dumps(usage))]
    assert r.feed(other, json.dumps(other)) == [(other, json.dumps(other))]


def test_tool_calls_arriving_while_holding_release_the_held_text_first():
    r = _repairer(expects_reasoning=True, default_key="reasoning")
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
