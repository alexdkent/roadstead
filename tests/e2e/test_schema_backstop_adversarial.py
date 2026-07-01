"""Phase 3 schema-backstop — INDEPENDENT ADVERSARIAL both-seam matrix + fuzz.

Hostile track (did NOT write the implementation — treat it as guilty until
proven robust). Drives the real in-process ``ProxyService`` end-to-end
(admission → dispatch → correction → response) against the programmable fake
backend, on BOTH doors (internal ``/v1/submit`` + OpenAI
``/v1/chat/completions``), asserting the contract §5 terminal-state matrix with
the flag ON (enforce), in shadow, and OFF (byte-identical parity).

South-face pathologies are emitted by the fake either via existing named faults
(``FAULT_SCHEMA_INVALID`` = trailing-prose, ``FAULT_SCHEMA_VALID_WRONG`` =
parseable-but-schema-violating, ``FAULT_PHANTOM_TOOL_CALLS`` = truncated
tool-call args) or the two controller knobs added by this track
(``structured_content`` / ``structured_tool_args``) — no new named fault, so the
ALL_FAULTS meta-coverage stays intact.

See docs/llmproxy_phase3_schema_backstop_contract.md §5.
"""
from __future__ import annotations

import json
import random

import pytest

from originfleet.llmproxy import correction as correction_mod
from originfleet.llmproxy.backend import BackendUnavailable
from tests.llmproxy.fake_backend import (
    FAULT_FINISH_LENGTH,
    FAULT_PHANTOM_TOOL_CALLS,
    FAULT_SCHEMA_INVALID,
    FAULT_SCHEMA_VALID_WRONG,
)

FLAG = "COLLECTIVE_PROXY_SCHEMA_BACKSTOP"
SHADOW = "COLLECTIVE_PROXY_SCHEMA_BACKSTOP_SHADOW"

# A strict schema: repair CANNOT synthesize a missing required field, so a
# parseable-but-schema-violating body forces the bounded retry (not an in-memory
# repair). additionalProperties:false makes {"unexpected":...} a hard miss.
STRICT = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
# Permissive: any JSON object validates → non-JSON is the only failure mode.
PERMISSIVE = {"type": "object"}
GOOD_JSON = '{"answer": "yes"}'
TOOLS = [{"type": "function", "function": {"name": "do_thing",
                                           "parameters": {"type": "object"}}}]

DOORS = ["internal", "openai"]


def _rf(schema):
    return {"type": "json_schema", "json_schema": {"name": "s", "schema": schema}}


async def drive(proxy, door, *, schema=None, tools=None, tool_choice=None,
                content="question", timeout_s=None):
    """POST a structured/tool chat on the chosen door; return the raw response."""
    payload = {"model": "chat",
               "messages": [{"role": "user", "content": content}],
               "max_tokens": 16}
    if schema is not None:
        payload["response_format"] = _rf(schema)
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if timeout_s is not None:
        payload["timeout_s"] = timeout_s
    if door == "openai":
        return await proxy.client.post("/v1/chat/completions", json=payload)
    body = {"agent_id": "adv", "endpoint": "chat", "priority": "P3_INGESTION",
            "call_site": "schema_adv", "payload_type": "chat_completion",
            "payload": payload}
    if timeout_s is not None:
        body["timeout_s"] = timeout_s
    return await proxy.client.post("/v1/submit", json=body)


def _body(resp, door):
    j = resp.json()
    return j["response"] if door == "internal" else j


def ok_content(resp, door):
    return _body(resp, door)["choices"][0]["message"]["content"]


def ok_tool_args(resp, door):
    msg = _body(resp, door)["choices"][0]["message"]
    return msg["tool_calls"][0]["function"]["arguments"]


def is_fail_loud(resp, door):
    """A schema-backstop deferrable fail-loud (502, typed), NOT a 500/other."""
    if resp.status_code != 502:
        return False
    j = resp.json()
    if door == "internal":
        return j.get("code") == "schema_invalid" and "schema-invalid" in j.get("error", "")
    err = j.get("error")
    msg = err.get("message", "") if isinstance(err, dict) else str(err)
    return "schema-invalid" in msg


def _n_backend_chats(proxy):
    return sum(1 for r in proxy.controller.requests
               if r.path.endswith("/v1/chat/completions"))


# =========================================================================== #
# 1) SOUTH-FACE matrix — enforce path
# =========================================================================== #

@pytest.mark.parametrize("door", DOORS)
async def test_valid_json_untouched(proxy, monkeypatch, door):
    """valid JSON, schema-valid → no-op, original returned, 0 backend retries."""
    monkeypatch.setenv(FLAG, "1")
    proxy.controller.structured_content = GOOD_JSON
    st = proxy.svc._correction.state
    before = st.schema_detected
    resp = await drive(proxy, door, schema=STRICT)
    assert resp.status_code == 200
    assert json.loads(ok_content(resp, door)) == {"answer": "yes"}
    assert st.schema_detected == before, "clean body must not be flagged"
    assert _n_backend_chats(proxy) == 1, "no retry on a clean body"
    assert proxy.total_in_flight() == 0


@pytest.mark.parametrize("door", DOORS)
async def test_trailing_prose_repaired(proxy, monkeypatch, door):
    """valid JSON + trailing prose → repaired in-memory, corrected returned."""
    monkeypatch.setenv(FLAG, "1")
    proxy.controller.set_fault(FAULT_SCHEMA_INVALID)  # '{"answer":"yes"} — prose'
    st = proxy.svc._correction.state
    resp = await drive(proxy, door, schema=STRICT)
    assert resp.status_code == 200
    # the prose tail is gone; content strict-parses + validates
    obj = json.loads(ok_content(resp, door))
    assert obj == {"answer": "yes"}
    assert st.schema_repaired >= 1
    assert _n_backend_chats(proxy) == 1, "in-memory repair must not hit the backend"
    assert proxy.total_in_flight() == 0


async def test_markdown_fenced_repaired(proxy, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    proxy.controller.structured_content = "```json\n" + GOOD_JSON + "\n```"
    st = proxy.svc._correction.state
    resp = await drive(proxy, "internal", schema=STRICT)
    assert resp.status_code == 200
    assert json.loads(ok_content(resp, "internal")) == {"answer": "yes"}
    assert st.schema_repaired >= 1
    assert proxy.total_in_flight() == 0


async def test_missing_brace_repaired(proxy, monkeypatch):
    """Truncated JSON (missing closing brace), finish=stop → repaired."""
    monkeypatch.setenv(FLAG, "1")
    proxy.controller.structured_content = '{"answer": "yes"'
    st = proxy.svc._correction.state
    resp = await drive(proxy, "internal", schema=STRICT)
    assert resp.status_code == 200
    assert json.loads(ok_content(resp, "internal")) == {"answer": "yes"}
    assert st.schema_repaired >= 1


@pytest.mark.parametrize("door", DOORS)
async def test_schema_valid_but_semantically_wrong_no_schema_untouched(proxy, monkeypatch, door):
    """schema-VALID JSON, semantically wrong, PERMISSIVE schema → no-op (we do
    not judge semantics)."""
    monkeypatch.setenv(FLAG, "1")
    proxy.controller.set_fault(FAULT_SCHEMA_VALID_WRONG)  # {"unexpected":...}
    st = proxy.svc._correction.state
    before = st.schema_detected
    resp = await drive(proxy, door, schema=PERMISSIVE)
    assert resp.status_code == 200
    assert json.loads(ok_content(resp, door)) == {"unexpected": "shape", "n": 1}
    assert st.schema_detected == before, "semantically-wrong-but-valid must not flag"
    assert _n_backend_chats(proxy) == 1
    assert proxy.total_in_flight() == 0


async def test_schema_invalid_retry_recovers(proxy, monkeypatch):
    """schema-INVALID parseable JSON, repair insufficient → bounded retry fires
    (error fed back); the retry returns a conformant body → swap-in, recovered."""
    monkeypatch.setenv(FLAG, "1")
    proxy.controller.structured_content = GOOD_JSON        # what the RETRY returns
    proxy.controller.set_fault(FAULT_SCHEMA_VALID_WRONG, max_hits=1)  # first call
    st = proxy.svc._correction.state
    r_before = st.schema_retry_recovered
    resp = await drive(proxy, "internal", schema=STRICT)
    assert resp.status_code == 200
    assert json.loads(ok_content(resp, "internal")) == {"answer": "yes"}
    assert st.schema_retry_recovered == r_before + 1
    assert _n_backend_chats(proxy) == 2, "exactly one retry re-dispatch"
    # the retry fed the error back as an appended user turn
    retry_req = proxy.controller.requests[-1]
    last_msg = retry_req.body["messages"][-1]
    assert last_msg["role"] == "user"
    assert "rejected" in last_msg["content"].lower()
    assert st.schema_retry_inflight == 0, "retry slot leaked"
    assert proxy.total_in_flight() == 0


async def test_schema_invalid_retry_recovers_via_repair(proxy, monkeypatch):
    """The RETRY body is itself imperfect (fenced) — _schema_retry must repair it
    in-memory (status 'repaired') and still count as recovered."""
    monkeypatch.setenv(FLAG, "1")
    proxy.controller.structured_content = "```json\n" + GOOD_JSON + "\n```"
    proxy.controller.set_fault(FAULT_SCHEMA_VALID_WRONG, max_hits=1)
    st = proxy.svc._correction.state
    r_before = st.schema_retry_recovered
    resp = await drive(proxy, "internal", schema=STRICT)
    assert resp.status_code == 200
    assert json.loads(ok_content(resp, "internal")) == {"answer": "yes"}
    assert st.schema_retry_recovered == r_before + 1
    assert _n_backend_chats(proxy) == 2
    assert st.schema_retry_inflight == 0
    assert proxy.total_in_flight() == 0


async def test_bare_grammar_no_schema_repaired_parse_only(proxy, monkeypatch):
    """A grammar-bearing request declares NO JSON Schema → validity == 'parses as
    JSON'. A fenced-but-parseable body is repaired (parse-only), no schema judged."""
    monkeypatch.setenv(FLAG, "1")
    proxy.controller.structured_content = "```json\n" + GOOD_JSON + "\n```"
    grammar = ('root ::= "{" ws "\\"answer\\":" ws str ws "}"\n'
               'str ::= "\\"" [^"]* "\\""\n'
               'ws ::= [ \\t\\n]*\n')
    payload = {"model": "chat", "messages": [{"role": "user", "content": "q"}],
               "max_tokens": 16, "extra_body": {"grammar": grammar}}
    body = {"agent_id": "adv", "endpoint": "chat", "priority": "P3_INGESTION",
            "call_site": "schema_adv", "payload_type": "chat_completion",
            "payload": payload}
    st = proxy.svc._correction.state
    resp = await proxy.client.post("/v1/submit", json=body)
    assert resp.status_code == 200
    assert json.loads(ok_content(resp, "internal")) == {"answer": "yes"}
    assert st.schema_repaired >= 1
    assert proxy.total_in_flight() == 0


@pytest.mark.parametrize("door", DOORS)
async def test_schema_invalid_persistent_fail_loud(proxy, monkeypatch, door):
    """schema-INVALID, repair insufficient, retry ALSO invalid → fail-loud
    deferrable (502, typed), never a malformed 200, never a 500."""
    monkeypatch.setenv(FLAG, "1")
    proxy.controller.set_fault(FAULT_SCHEMA_VALID_WRONG)  # persistent
    st = proxy.svc._correction.state
    u_before = st.schema_unrecoverable
    resp = await drive(proxy, door, schema=STRICT)
    assert is_fail_loud(resp, door), f"expected typed fail-loud, got {resp.status_code}: {resp.text[:200]}"
    assert st.schema_unrecoverable == u_before + 1
    assert _n_backend_chats(proxy) == 2, "one retry then give up"
    assert st.schema_retry_inflight == 0
    assert proxy.total_in_flight() == 0


async def test_unrepairable_content_fail_loud(proxy, monkeypatch):
    """Non-JSON prose content (json_repair can't recover) → retry → fail-loud."""
    monkeypatch.setenv(FLAG, "1")
    proxy.controller.structured_content = "sorry, here is a plain-English answer with no json"
    st = proxy.svc._correction.state
    resp = await drive(proxy, "internal", schema=STRICT)
    assert is_fail_loud(resp, "internal")
    assert st.schema_unrecoverable >= 1
    assert proxy.total_in_flight() == 0


@pytest.mark.parametrize("door", DOORS)
async def test_phantom_tool_args_repaired(proxy, monkeypatch, door):
    """Truncated tool_calls.arguments ('{"x": 1') → repaired to valid JSON."""
    monkeypatch.setenv(FLAG, "1")
    proxy.controller.set_fault(FAULT_PHANTOM_TOOL_CALLS)
    st = proxy.svc._correction.state
    resp = await drive(proxy, door, tools=TOOLS, tool_choice="auto")
    assert resp.status_code == 200
    args = ok_tool_args(resp, door)
    assert json.loads(args) == {"x": 1}, "tool-call args not repaired"
    assert st.schema_repaired >= 1
    assert proxy.total_in_flight() == 0


async def test_unrepairable_tool_args_fail_loud(proxy, monkeypatch):
    """tool_calls.arguments that json_repair cannot recover → retry → fail-loud."""
    monkeypatch.setenv(FLAG, "1")
    proxy.controller.structured_tool_args = "<<< not json at all >>>"
    st = proxy.svc._correction.state
    resp = await drive(proxy, "internal", tools=TOOLS, tool_choice="auto")
    assert is_fail_loud(resp, "internal")
    assert st.schema_unrecoverable >= 1
    assert proxy.total_in_flight() == 0


async def test_finish_length_structured_truncation_integrity_wins(proxy, monkeypatch):
    """finish_reason=length + structured → the pre-existing truncation integrity
    fails-loud FIRST (status flipped to error before apply); the backstop gates on
    status=='ok' so it must NOT engage, retry, or bump its counters."""
    monkeypatch.setenv(FLAG, "1")
    proxy.controller.set_fault(FAULT_FINISH_LENGTH)  # finish=length, non-JSON body
    st = proxy.svc._correction.state
    d_before, u_before = st.schema_detected, st.schema_unrecoverable
    resp = await drive(proxy, "internal", schema=STRICT)
    assert resp.status_code == 502, "truncated structured output must defer"
    assert "truncated" in resp.text
    assert st.schema_detected == d_before, "backstop must not run on a length-truncated body"
    assert st.schema_unrecoverable == u_before
    assert _n_backend_chats(proxy) == 1, "backstop must not retry a truncation defer"
    assert proxy.total_in_flight() == 0


# =========================================================================== #
# 2) RETRY is bounded — concurrency cap + deadline both skip straight to fail-loud
# =========================================================================== #

async def test_retry_skipped_when_concurrency_saturated(proxy, monkeypatch):
    """schema_retry_inflight >= 2 → NO retry re-dispatch, straight to fail-loud;
    the forced counter is untouched by the guard (no phantom decrement)."""
    monkeypatch.setenv(FLAG, "1")
    proxy.controller.set_fault(FAULT_SCHEMA_VALID_WRONG)
    st = proxy.svc._correction.state
    st.schema_retry_inflight = 2
    try:
        resp = await drive(proxy, "internal", schema=STRICT)
        assert is_fail_loud(resp, "internal")
        assert _n_backend_chats(proxy) == 1, "must NOT retry when saturated"
        assert st.schema_retry_inflight == 2, "guard must not touch a saturated counter"
    finally:
        st.schema_retry_inflight = 0
    assert proxy.total_in_flight() == 0


async def test_retry_skipped_past_deadline(proxy, monkeypatch):
    """remaining budget < _MIN_RETRY_BUDGET_S → NO retry, straight to fail-loud."""
    monkeypatch.setenv(FLAG, "1")
    proxy.controller.set_fault(FAULT_SCHEMA_VALID_WRONG)
    st = proxy.svc._correction.state
    # timeout_s=4 < 5s min-retry-budget → the deadline bound trips before dispatch
    resp = await drive(proxy, "internal", schema=STRICT, timeout_s=4.0)
    assert is_fail_loud(resp, "internal")
    assert _n_backend_chats(proxy) == 1, "must NOT retry with < min budget left"
    assert st.schema_retry_inflight == 0
    assert proxy.total_in_flight() == 0


async def test_retry_backend_exception_fails_loud_no_leak(proxy, monkeypatch):
    """The retry re-dispatch itself throws (backend down) → fail-loud, and the
    schema_retry_inflight slot is released by the finally (no leak)."""
    monkeypatch.setenv(FLAG, "1")
    proxy.controller.set_fault(FAULT_SCHEMA_VALID_WRONG)
    st = proxy.svc._correction.state
    real_call = proxy.svc._backend.call

    async def _call(ep_cfg, payload, ptype, request_id, *a, **k):
        if str(request_id).endswith("-schema"):
            raise BackendUnavailable("injected retry outage")
        return await real_call(ep_cfg, payload, ptype, request_id, *a, **k)

    monkeypatch.setattr(proxy.svc._backend, "call", _call)
    resp = await drive(proxy, "internal", schema=STRICT)
    assert is_fail_loud(resp, "internal")
    assert st.schema_retry_inflight == 0, "retry slot leaked on backend exception"
    assert proxy.total_in_flight() == 0


# =========================================================================== #
# 3) NORTH-FACE — hostile caller input degrades cleanly, never 500 / crash / leak
# =========================================================================== #

async def test_unknown_type_schema_parse_only_no_crash(proxy, monkeypatch):
    """A declared schema jsonschema can't compile ({"type":"definitely-not-a-type"})
    must NOT make the guard throw — degrade to parse-only. Valid JSON then → 200."""
    monkeypatch.setenv(FLAG, "1")
    proxy.controller.structured_content = GOOD_JSON
    resp = await drive(proxy, "internal", schema={"type": "definitely-not-a-type"})
    assert resp.status_code == 200, resp.text[:200]
    assert json.loads(ok_content(resp, "internal")) == {"answer": "yes"}
    assert proxy.total_in_flight() == 0


async def test_non_object_declared_schema_ignored(proxy, monkeypatch):
    """response_format.json_schema.schema that isn't an object (a string) →
    _extract_declared_schema returns None → parse-only, no crash."""
    monkeypatch.setenv(FLAG, "1")
    proxy.controller.structured_content = GOOD_JSON
    payload = {"model": "chat", "messages": [{"role": "user", "content": "q"}],
               "max_tokens": 16,
               "response_format": {"type": "json_schema",
                                   "json_schema": {"name": "s", "schema": "not-an-object"}}}
    body = {"agent_id": "adv", "endpoint": "chat", "priority": "P3_INGESTION",
            "call_site": "schema_adv", "payload_type": "chat_completion",
            "payload": payload}
    resp = await proxy.client.post("/v1/submit", json=body)
    assert resp.status_code == 200, resp.text[:200]
    assert proxy.total_in_flight() == 0


async def test_contradictory_response_format_and_grammar(proxy, monkeypatch):
    """A caller sending BOTH a response_format schema AND a grammar → the guard
    must not crash; a clean typed response (200 or a typed 422/502), slot freed."""
    monkeypatch.setenv(FLAG, "1")
    proxy.controller.structured_content = GOOD_JSON
    grammar = ('root ::= "{" ws "\\"answer\\":" ws str ws "}"\n'
               'str ::= "\\"" [^"]* "\\""\n'
               'ws ::= [ \\t\\n]*\n')
    payload = {"model": "chat", "messages": [{"role": "user", "content": "q"}],
               "max_tokens": 16, "response_format": _rf(STRICT),
               "extra_body": {"grammar": grammar}}
    body = {"agent_id": "adv", "endpoint": "chat", "priority": "P3_INGESTION",
            "call_site": "schema_adv", "payload_type": "chat_completion",
            "payload": payload}
    resp = await proxy.client.post("/v1/submit", json=body)
    assert resp.status_code in (200, 422, 502), resp.text[:200]
    assert resp.status_code != 500
    assert proxy.total_in_flight() == 0


@pytest.mark.parametrize("tool_choice", ["not_a_valid_choice",
                                         {"type": "function"},  # missing function.name
                                         12345])
async def test_invalid_tool_choice_no_crash(proxy, monkeypatch, tool_choice):
    """A malformed tool_choice must not crash the backstop/constrained path."""
    monkeypatch.setenv(FLAG, "1")
    proxy.controller.structured_content = GOOD_JSON
    resp = await drive(proxy, "internal", tools=TOOLS, tool_choice=tool_choice)
    assert resp.status_code in (200, 422, 502), resp.text[:200]
    assert resp.status_code != 500
    assert proxy.total_in_flight() == 0


# =========================================================================== #
# 4) FLAG OFF — byte-identical parity (every pathology passes through UNCHANGED)
# =========================================================================== #

@pytest.mark.parametrize("door", DOORS)
async def test_off_trailing_prose_untouched(proxy, monkeypatch, door):
    monkeypatch.delenv(FLAG, raising=False)
    proxy.controller.set_fault(FAULT_SCHEMA_INVALID)
    st = proxy.svc._correction.state
    before = st.schema_detected
    resp = await drive(proxy, door, schema=STRICT)
    assert resp.status_code == 200
    # the raw prose tail is still present (no repair)
    assert "prose" in ok_content(resp, door)
    assert st.schema_detected == before, "OFF must observe nothing"
    assert proxy.total_in_flight() == 0


async def test_off_schema_invalid_passes_through(proxy, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    proxy.controller.set_fault(FAULT_SCHEMA_VALID_WRONG)
    resp = await drive(proxy, "internal", schema=STRICT)
    assert resp.status_code == 200, "OFF must not fail-loud"
    assert json.loads(ok_content(resp, "internal")) == {"unexpected": "shape", "n": 1}
    assert _n_backend_chats(proxy) == 1, "OFF must not retry"
    assert proxy.total_in_flight() == 0


async def test_off_phantom_tool_args_leak(proxy, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    proxy.controller.set_fault(FAULT_PHANTOM_TOOL_CALLS)
    resp = await drive(proxy, "internal", tools=TOOLS, tool_choice="auto")
    assert resp.status_code == 200
    assert ok_tool_args(resp, "internal") == '{"x": 1', "OFF must leave args raw"
    assert proxy.total_in_flight() == 0


# =========================================================================== #
# 5) SHADOW — detect + count, return ORIGINAL untouched
# =========================================================================== #

async def test_shadow_trailing_prose_detected_not_mutated(proxy, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(SHADOW, "1")
    proxy.controller.set_fault(FAULT_SCHEMA_INVALID)
    st = proxy.svc._correction.state
    d_before, r_before = st.schema_detected, st.schema_repaired
    resp = await drive(proxy, "internal", schema=STRICT)
    assert resp.status_code == 200
    assert "prose" in ok_content(resp, "internal"), "shadow must NOT mutate the body"
    assert st.schema_detected == d_before + 1, "shadow must still detect+count"
    assert st.schema_repaired == r_before, "shadow must not record a real repair"
    assert proxy.total_in_flight() == 0


async def test_shadow_schema_invalid_no_retry_no_failloud(proxy, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(SHADOW, "1")
    proxy.controller.set_fault(FAULT_SCHEMA_VALID_WRONG)
    st = proxy.svc._correction.state
    d_before = st.schema_detected
    resp = await drive(proxy, "internal", schema=STRICT)
    assert resp.status_code == 200, "shadow must never fail-loud"
    assert json.loads(ok_content(resp, "internal")) == {"unexpected": "shape", "n": 1}
    assert st.schema_detected == d_before + 1
    assert _n_backend_chats(proxy) == 1, "shadow must not retry-dispatch"
    assert proxy.total_in_flight() == 0


# =========================================================================== #
# 6) GUARD-BITE — revert the guard → the schema-invalid assertion goes red
# =========================================================================== #

async def test_guard_bite_maybe_repair_schema_noop_leaks(proxy, monkeypatch):
    """Revert the guard (stub Correction.maybe_repair_schema to a no-op): the SAME
    schema-invalid input that fails-loud with the guard present now passes STRAIGHT
    THROUGH as a 200 with the invalid body — proving the enforce assertion depends
    on the guard, not on the fault never firing."""
    monkeypatch.setenv(FLAG, "1")

    async def _noop(self, req, result):
        return None

    monkeypatch.setattr(correction_mod.Correction, "maybe_repair_schema", _noop)
    proxy.controller.set_fault(FAULT_SCHEMA_VALID_WRONG)
    resp = await drive(proxy, "internal", schema=STRICT)
    # with the guard gone, the invalid body is handed straight to the caller
    assert resp.status_code == 200, "reverted guard should NOT fail-loud"
    assert json.loads(ok_content(resp, "internal")) == {"unexpected": "shape", "n": 1}
    assert _n_backend_chats(proxy) == 1, "reverted guard should not retry"
    assert proxy.total_in_flight() == 0


async def test_guard_bite_conform_body_stub_leaks_prose(proxy, monkeypatch):
    """Stub _conform_body to always claim ("ok", body): the trailing-prose repair
    assertion must go red (prose leaks through)."""
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(correction_mod, "_conform_body",
                        lambda body, schema: ("ok", body))
    proxy.controller.set_fault(FAULT_SCHEMA_INVALID)
    resp = await drive(proxy, "internal", schema=STRICT)
    assert resp.status_code == 200
    # the repair never ran → the prose tail is STILL present
    assert "prose" in ok_content(resp, "internal"), \
        "with _conform_body stubbed, the prose must leak (guard-bite proof)"
    assert proxy.total_in_flight() == 0


# =========================================================================== #
# 7) FUZZ — random malformed/oversized/truncated/fenced/binary structured bodies
#    (flag ON): never 500, never crash, never leak a slot; serve or fail cleanly.
# =========================================================================== #

def _fuzz_bodies(seed: int, n: int):
    rnd = random.Random(seed)
    frags = ['{"answer":', '"yes"', '}', '```json', '```', ' — prose tail',
             '\x00\x01\xff', "🎵", '[1,2,3', 'null', '{"answer": 12345}',
             ',,,', "'single'", '\\n\\t', '{"answer":"' + "x" * 500 + '"}']
    out = []
    for _ in range(n):
        k = rnd.randint(0, 6)
        s = "".join(rnd.choice(frags) for _ in range(k))
        if rnd.random() < 0.1:
            s = s * rnd.randint(50, 200)  # oversized
        out.append(s)
    out += ["", "   ", "{", "}", "[]", GOOD_JSON, "```\n" + GOOD_JSON + "\n```"]
    return out


@pytest.mark.parametrize("schema", [PERMISSIVE, STRICT, None])
async def test_fuzz_structured_bodies_never_500_or_leak(proxy, monkeypatch, schema):
    monkeypatch.setenv(FLAG, "1")
    st = proxy.svc._correction.state
    # 12 seeded random bodies (+7 fixed edge cases) per schema-mode. The frag set
    # already spans every pathology class (fences/binary/truncated/oversized/
    # unicode/quotes); trimmed from 24 to keep the at-capacity tollgate under its
    # 150s wall (shrink corpus, never raise the budget) while still proving the
    # never-500/never-leak invariant across all three schema modes.
    for junk in _fuzz_bodies(1337, 12):
        proxy.controller.structured_content = junk
        # a bare grammar request (schema None but structured) still enters the guard
        extra = {"schema": schema} if schema is not None else {
            "tools": TOOLS, "tool_choice": "auto"}
        resp = await drive(proxy, "internal", timeout_s=30.0, **extra)
        assert resp.status_code in (200, 502), \
            f"non-typed status {resp.status_code} for {junk[:40]!r}"
        assert resp.status_code != 500
        assert proxy.total_in_flight() == 0, f"slot leak on {junk[:40]!r}"
        assert st.schema_retry_inflight == 0, f"retry slot leak on {junk[:40]!r}"
