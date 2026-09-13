"""A backend with no constrained decoding REFUSES `response_format`, and the
same schema as a forced tool call works.

A candidate backend evaluated 2026-09-13 answers both
``{"type":"json_object"}`` and a strict ``{"type":"json_schema", …}`` with an
immediate HTTP 400: the build has no constrained decoding, so a schema could not
be enforced and the reply would be prose. That refusal is the right behaviour and
it is still fatal to every caller that declares a schema — which, on a fleet that
took the "always declare a schema" rule seriously, is most of them.

The capability is present on that build; it is reached through the tool-argument
path. Same model, same schema handed over as a forced tool call (schema as the
function's ``parameters``, ``tool_choice`` naming the function): 29/29 fully
schema-valid on the hardest schema available — nested objects, ``["string","null"]``
unions, ``maxLength`` rails, objects inside an array's ``items``,
``additionalProperties:false`` closed objects and ``minItems == maxItems == n``
exact counts. On an incumbent backend that supports both, the forced tool call was
also the faster path (4.7s vs 7.5s median, 8/8 schema-valid either way).

``Correction.apply_forced_tool_schema`` performs that translation, and
``finalize_forced_tool_schema`` translates the ANSWER back.

🚨 **The gate is the BACKEND'S OWN `/health`, and the first cut of it was
wrong in a way that would have cost real latency on a live security path.**
That build publishes `"not_implemented": ["response_format", "text.format"]`;
the poller reads it into `EndpointConfig.not_implemented` and the translation
fires only on a POSITIVE statement there. An earlier version read the CATALOG
instead — `tool_calling` declared, `structured_output` absent — and against a
live fleet catalog that conjunction matched exactly one endpoint: a production
classifier on the prompt-injection/PII path whose stanza merely omits
`structured_output` while the backend implements it fine. Measured on it (4
spans, 8 trials per path, temperature 0, cache-busted): native 8/8 valid at 4.1s
median, forced tool call 8/8 valid at 9.1s — 2.2x for zero correctness gain.
An omission in a catalog is indistinguishable from an incapacity; a backend
saying "I do not implement this" is not. Which path is FASTER is also a property
of the backend, not the mechanism: faster on a mid tier, 2.2x slower here.

🚨 The second half is the whole point, and this file pins it hardest. A caller
that sent ``response_format`` parses ``choices[0].message.content``. Rewriting
only the request hands every such call site ``content: null`` beside a
``tool_calls`` array it does not read — a loud 400 converted into a silent empty
parse, which is strictly worse than the incompatibility. The end-to-end half of
this file drives a real ``ProxyService`` so the two halves are pinned together
rather than as two units that happen to agree.

Self-contained in the house style of test_json_object_guard.py for the unit half
(the real ``Correction`` methods bound to a lightweight mock ``self``) and of
test_corrections_disclosure.py for the end-to-end half.
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

correction = importlib.import_module("roadstead.correction")
C = correction.Correction

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "maxLength": 40},
        "score": {"type": "integer"},
        "note": {"type": ["string", "null"]},
    },
    "required": ["verdict", "score"],
}
SCHEMA_RF = {"type": "json_schema",
             "json_schema": {"name": "triage", "schema": SCHEMA}}
ANSWER = '{"verdict": "keep", "score": 3, "note": null}'


def _req(payload, *, endpoint="tier1", ptype="chat_completion", stream=False):
    r = types.SimpleNamespace()
    r.forced_tool_schema = None
    r.forced_tool_name = ""
    r.json_object_stripped = False
    r.payload = payload
    r.payload_type = ptype
    r.endpoint = endpoint
    r.stream = stream
    r.request_id = "r1"
    r.agent_id = "ops"
    r.call_site = "triage.classify"
    r.priority = types.SimpleNamespace(name="P2_POST_TURN")
    return r


def _mock_self(*, not_implemented=("response_format", "text.format")):
    state = types.SimpleNamespace()
    ep = types.SimpleNamespace(backend_engine="llama.cpp",
                               capabilities=frozenset({"tool_calling", "streaming"}),
                               not_implemented=frozenset(not_implemented))
    state.config = types.SimpleNamespace(endpoints={"tier1": ep})
    state.truncation_by_model_caller = {}
    state.truncation_total = 0
    m = types.SimpleNamespace(state=state)
    for name in ("apply_forced_tool_schema", "finalize_forced_tool_schema",
                 "request_is_structured",
                 "request_expects_json", "extract_grammar",
                 "record_truncation_event", "corrections_applied"):
        setattr(m, name, getattr(C, name).__get__(m, C))
    return m


def _tool_reply(name="triage", arguments=ANSWER, finish="tool_calls",
                content=None, extra_calls=()):
    calls = [{"id": "call_0", "type": "function",
              "function": {"name": name, "arguments": arguments}}]
    calls.extend(extra_calls)
    return {"status": "ok", "response": {
        "choices": [{"message": {"role": "assistant", "content": content,
                                 "tool_calls": calls},
                     "finish_reason": finish}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 18}}}


# ---------------------------------------------------------------------------
# Direction 1 — the request rewrite
# ---------------------------------------------------------------------------

def test_json_schema_becomes_a_forced_tool_call():
    """The bug. A backend with no constrained decoding 400s the field; the same
    schema as the forced function's `parameters` is enforced instead."""
    m = _mock_self()
    p = {"messages": [{"role": "user", "content": "triage this"}],
         "max_tokens": 400, "response_format": dict(SCHEMA_RF)}
    req = _req(p)
    m.apply_forced_tool_schema(req)

    assert "response_format" not in p, (
        "the field the backend refuses must be GONE — leaving it is the 400 "
        "this correction exists to avoid")
    assert p["tools"] == [{"type": "function", "function": {
        "name": "triage", "parameters": SCHEMA}}]
    assert p["tool_choice"] == {"type": "function",
                                "function": {"name": "triage"}}
    # A translation, not a rewrite of the caller's turn.
    assert p["max_tokens"] == 400 and len(p["messages"]) == 1


def test_the_schema_object_is_carried_not_copied_shallow():
    """`parameters` must be the caller's schema itself — a truncated or
    re-serialized copy would enforce something the caller did not declare."""
    m = _mock_self()
    p = {"messages": [], "response_format": dict(SCHEMA_RF)}
    m.apply_forced_tool_schema(_req(p))
    assert p["tools"][0]["function"]["parameters"] == SCHEMA
    assert p["tools"][0]["function"]["parameters"]["properties"]["note"][
        "type"] == ["string", "null"]


def test_the_declared_description_rides_along():
    m = _mock_self()
    rf = {"type": "json_schema", "json_schema": {
        "name": "triage", "description": "how to file this item", "schema": SCHEMA}}
    p = {"messages": [], "response_format": rf}
    m.apply_forced_tool_schema(_req(p))
    assert p["tools"][0]["function"]["description"] == "how to file this item"


def test_nothing_unmeasured_is_added_to_the_wire_shape():
    """The measured-good shape is name + parameters + tool_choice. `strict`,
    `response_format` remnants and vendor extras are not invented here: an
    unmeasured field on the backend we are adapting TO is how an adapter starts
    failing in a way nobody can attribute."""
    m = _mock_self()
    p = {"messages": [], "response_format": dict(SCHEMA_RF)}
    m.apply_forced_tool_schema(_req(p))
    assert set(p["tools"][0]["function"]) == {"name", "parameters"}
    assert set(p["tools"][0]) == {"type", "function"}
    assert "response_format" not in p


def test_a_hostile_schema_name_is_sanitized_to_a_legal_function_name():
    m = _mock_self()
    for raw, expected in (("triage report!", "triage_report"),
                          ("", "structured_response"),
                          (None, "structured_response"),
                          ("___", "structured_response"),
                          ("a" * 200, "a" * 64)):
        p = {"messages": [], "response_format": {
            "type": "json_schema", "json_schema": {"name": raw, "schema": SCHEMA}}}
        req = _req(p)
        m.apply_forced_tool_schema(req)
        assert p["tool_choice"]["function"]["name"] == expected
        assert p["tools"][0]["function"]["name"] == expected
        assert req.forced_tool_name == expected


def test_a_schema_under_extra_body_is_translated_too():
    m = _mock_self()
    p = {"messages": [], "extra_body": {"response_format": dict(SCHEMA_RF)}}
    m.apply_forced_tool_schema(_req(p))
    assert "response_format" not in p["extra_body"]
    # tools/tool_choice are top-level fields on the chat-completions wire.
    assert p["tool_choice"]["function"]["name"] == "triage"


# ---------------------------------------------------------------------------
# Pass-through: the cases that must be left exactly alone
# ---------------------------------------------------------------------------

def test_untouched_when_the_backend_disclaims_nothing():
    """Every incumbent engine publishes no `not_implemented` at all, so this is
    the case that covers llama.cpp and vLLM — and the live classifier endpoint
    whose catalog stanza merely omits `structured_output`."""
    m = _mock_self(not_implemented=())
    p = {"messages": [], "response_format": dict(SCHEMA_RF)}
    req = _req(p)
    m.apply_forced_tool_schema(req)
    assert p["response_format"] == SCHEMA_RF
    assert "tools" not in p and "tool_choice" not in p
    assert req.forced_tool_schema is None


def test_untouched_when_the_backend_disclaims_something_else():
    """Membership, not truthiness. A build that disclaims some other field has
    said nothing about `response_format`."""
    m = _mock_self(not_implemented=("text.format", "logit_bias"))
    p = {"messages": [], "response_format": dict(SCHEMA_RF)}
    req = _req(p)
    m.apply_forced_tool_schema(req)
    assert p["response_format"] == SCHEMA_RF
    assert req.forced_tool_schema is None


def test_the_catalog_capabilities_block_is_not_the_gate():
    """🚨 The regression this file exists to prevent. An endpoint whose stanza
    declares `tool_calling` and omits `structured_output` — the exact shape of a
    live classifier on the prompt-injection path — must NOT be translated on
    that basis. Only the backend's own `/health` decides."""
    m = _mock_self(not_implemented=())
    m.state.config.endpoints["tier1"].capabilities = frozenset({"tool_calling",
                                                                "streaming"})
    p = {"messages": [], "response_format": dict(SCHEMA_RF)}
    req = _req(p)
    m.apply_forced_tool_schema(req)
    assert p["response_format"] == SCHEMA_RF, (
        "gating on the catalog cost this endpoint 2.2x latency (9.1s vs 4.1s "
        "median) for zero correctness gain")
    assert req.forced_tool_schema is None


def test_untouched_when_the_reading_could_not_be_taken():
    """An unreachable /health, a non-200, a body that is not JSON and a
    malformed value all arrive here as an EMPTY set (backend.probe_not_
    implemented returns None and the poller stores frozenset()). "I cannot
    tell" must never license a payload rewrite."""
    for absent in (frozenset(), None):
        m = _mock_self()
        m.state.config.endpoints["tier1"].not_implemented = absent
        p = {"messages": [], "response_format": dict(SCHEMA_RF)}
        req = _req(p)
        m.apply_forced_tool_schema(req)
        assert p["response_format"] == SCHEMA_RF, absent
        assert req.forced_tool_schema is None, absent


def test_a_request_with_its_own_tools_is_never_clobbered():
    """Forcing the schema function would SUPPRESS the caller's tool call: they
    would get a schema-shaped answer and none of the side effects they asked
    for. Leave it untranslated and let the backend refuse, loudly."""
    m = _mock_self()
    tools = [{"type": "function", "function": {"name": "send_email",
                                               "parameters": {"type": "object"}}}]
    p = {"messages": [], "tools": tools, "response_format": dict(SCHEMA_RF)}
    req = _req(p)
    m.apply_forced_tool_schema(req)
    assert p["tools"] == tools
    assert "tool_choice" not in p
    assert p["response_format"] == SCHEMA_RF
    assert req.forced_tool_schema is None


def test_a_request_with_its_own_tool_choice_is_never_clobbered():
    m = _mock_self()
    p = {"messages": [], "tool_choice": "auto", "response_format": dict(SCHEMA_RF)}
    req = _req(p)
    m.apply_forced_tool_schema(req)
    assert p["tool_choice"] == "auto"
    assert "tools" not in p
    assert req.forced_tool_schema is None


def test_another_structured_constraint_means_hands_off():
    """Same rule as apply_json_object_guard: a caller who pinned a grammar has
    pinned something this correction has no translation for."""
    m = _mock_self()
    for key, value in (("grammar", 'root ::= "yes"'),
                       ("guided_json", {"type": "object"}),
                       ("structured_outputs", {"grammar": 'root ::= "1"'}),
                       ("guided_choice", ["a", "b"])):
        p = {"messages": [], key: value, "response_format": dict(SCHEMA_RF)}
        req = _req(p)
        m.apply_forced_tool_schema(req)
        assert p["response_format"] == SCHEMA_RF, key
        assert "tools" not in p, key
        assert req.forced_tool_schema is None, key


def test_a_bare_json_object_is_not_this_correction_s_business():
    """`json_object` carries no schema, so there is nothing to put in
    `parameters`. apply_json_object_guard owns that shape."""
    m = _mock_self()
    p = {"messages": [], "response_format": {"type": "json_object"}}
    req = _req(p)
    m.apply_forced_tool_schema(req)
    assert p["response_format"] == {"type": "json_object"}
    assert req.forced_tool_schema is None


def test_a_non_object_root_schema_is_left_alone():
    """A tool's `parameters` is an argument OBJECT. Wrapping an array-rooted
    schema would change the document the caller parses."""
    m = _mock_self()
    for schema in ({"type": "array", "items": {"type": "string"}},
                   {"type": "string"},
                   {}):
        p = {"messages": [], "response_format": {
            "type": "json_schema", "json_schema": {"name": "x", "schema": schema}}}
        req = _req(p)
        m.apply_forced_tool_schema(req)
        assert "response_format" in p, schema
        assert req.forced_tool_schema is None, schema


def test_streaming_is_declined_outright():
    """The response half rewrites a completed body. A stream is already on the
    wire by the time the tool call is whole, and the caller subscribed to
    `content` deltas it would never receive — half-supporting it would turn a
    400 into a stream that ends with nothing in it."""
    m = _mock_self()
    p = {"messages": [], "stream": True, "response_format": dict(SCHEMA_RF)}
    req = _req(p, stream=True)
    m.apply_forced_tool_schema(req)
    assert p["response_format"] == SCHEMA_RF
    assert "tools" not in p
    assert req.forced_tool_schema is None


def test_non_chat_payloads_and_unknown_endpoints_are_noops():
    m = _mock_self()
    p = {"messages": [], "response_format": dict(SCHEMA_RF)}
    m.apply_forced_tool_schema(_req(p, ptype="embedding"))
    assert "response_format" in p
    m.apply_forced_tool_schema(_req(p, endpoint="nope-not-a-role"))
    assert "response_format" in p


def test_never_raises_on_a_hostile_payload():
    """Fail-open. A malformed payload passes through, it does not 500 a caller."""
    m = _mock_self()
    for payload in (None, [], "not-a-dict",
                    {"messages": [], "response_format": "json_schema"},
                    {"messages": [], "response_format": None},
                    {"messages": [], "response_format": {"type": "json_schema"}},
                    {"messages": [], "response_format": {
                        "type": "json_schema", "json_schema": {"schema": "nope"}}},
                    {"messages": [], "extra_body": "nope",
                     "response_format": dict(SCHEMA_RF)}):
        m.apply_forced_tool_schema(_req(payload))
    # …and the last shape is still a valid translation target.
    p = {"messages": [], "extra_body": "nope", "response_format": dict(SCHEMA_RF)}
    m.apply_forced_tool_schema(_req(p))
    assert "response_format" not in p and p["tool_choice"]["function"]["name"] == "triage"


# ---------------------------------------------------------------------------
# Direction 2 — the response translation. THE HALF THAT MATTERS.
# ---------------------------------------------------------------------------

def _translated_req():
    m = _mock_self()
    p = {"messages": [], "response_format": dict(SCHEMA_RF)}
    req = _req(p)
    m.apply_forced_tool_schema(req)
    return m, req


def test_the_tool_call_becomes_content():
    m, req = _translated_req()
    result = _tool_reply()
    m.finalize_forced_tool_schema(req, result)
    msg = result["response"]["choices"][0]["message"]
    assert msg["content"] == ANSWER, (
        "the caller sent response_format and parses message.content — leaving "
        "the answer in tool_calls is a silent empty parse at 55 call sites")
    assert json.loads(msg["content"])["score"] == 3


def test_the_synthesized_tool_call_does_not_leak_to_the_caller():
    m, req = _translated_req()
    result = _tool_reply()
    m.finalize_forced_tool_schema(req, result)
    msg = result["response"]["choices"][0]["message"]
    assert "tool_calls" not in msg, (
        "the caller never declared a tool; a tool_calls array in its response is "
        "a proxy artefact it has no way to interpret")


def test_finish_reason_becomes_what_a_content_response_carries():
    m, req = _translated_req()
    result = _tool_reply(finish="tool_calls")
    m.finalize_forced_tool_schema(req, result)
    assert result["response"]["choices"][0]["finish_reason"] == "stop"


def test_a_length_finish_is_preserved_not_overwritten():
    """A truncation must stay visible to the gate that reads finish_reason —
    stamping `stop` on a cut answer hides it."""
    m, req = _translated_req()
    result = _tool_reply(finish="length")
    m.finalize_forced_tool_schema(req, result)
    assert result["response"]["choices"][0]["finish_reason"] == "length"
    assert result["response"]["choices"][0]["message"]["content"] == ANSWER


def test_the_translation_is_disclosed_to_the_caller():
    m, req = _translated_req()
    result = _tool_reply()
    m.finalize_forced_tool_schema(req, result)
    assert m.corrections_applied(req, result) == ["forced_tool_schema"]


def test_arguments_that_arrived_as_an_object_are_normalized_to_text():
    """Engines disagree about the type of `arguments`; the caller is about to
    json.loads() whatever lands in content."""
    m, req = _translated_req()
    result = _tool_reply(arguments={"verdict": "keep", "score": 3})
    m.finalize_forced_tool_schema(req, result)
    assert json.loads(
        result["response"]["choices"][0]["message"]["content"])["score"] == 3


def test_an_untranslated_request_is_never_touched_on_the_way_back():
    """The response half must be scoped to requests the request half rewrote —
    a genuine tool-calling caller keeps its tool call."""
    m = _mock_self(not_implemented=())
    req = _req({"messages": [], "tools": [{"type": "function", "function": {
        "name": "triage", "parameters": SCHEMA}}]})
    result = _tool_reply()
    m.finalize_forced_tool_schema(req, result)
    msg = result["response"]["choices"][0]["message"]
    assert msg["tool_calls"][0]["function"]["arguments"] == ANSWER
    assert msg["content"] is None
    assert result["response"]["choices"][0]["finish_reason"] == "tool_calls"
    assert m.corrections_applied(req, result) == []


# --- the failure paths: fail where the caller can see it, never fabricate ---

@pytest.mark.parametrize("arguments", [
    '{"verdict": "keep", "sco',        # cut mid-JSON
    "",                                # emitted the call and stopped
    "   ",
    None,                              # no arguments key at all
    '"just a string"',                 # JSON, but not an argument object
    "not json at all",
])
def test_unusable_arguments_fail_loud_and_deferrable(arguments):
    m, req = _translated_req()
    result = _tool_reply(arguments=arguments)
    m.finalize_forced_tool_schema(req, result)
    assert result["status"] == "error"
    assert result["code"] == "toolcall_truncated"
    assert "truncated structured output" in result["error"], (
        "the §2.2 marker substring is the contract clients classify on")
    assert "response" not in result, (
        "a body with no usable answer must not be served alongside the error")
    assert m.state.truncation_total == 1


def test_unusable_arguments_are_never_repaired_into_an_answer():
    """json-repair would close a cut argument object into valid-but-fabricated
    JSON. On a schema turn that is a confidently wrong answer."""
    m, req = _translated_req()
    result = _tool_reply(arguments='{"verdict": "kee')
    m.finalize_forced_tool_schema(req, result)
    assert result.get("status") == "error"
    assert "verdict" not in json.dumps(result.get("response", {}))


def test_no_forced_tool_call_at_all_is_passed_through_not_fabricated():
    """The backend ignored tool_choice and answered in the content channel.
    Nothing here invents an answer: the body is left exactly as it came and the
    always-on structured-validity floor judges it (JSON reaches the caller,
    non-JSON becomes the established structured_invalid_json 502)."""
    m, req = _translated_req()
    result = {"status": "ok", "response": {
        "choices": [{"message": {"role": "assistant",
                                 "content": "I think you should keep it."},
                     "finish_reason": "stop"}],
        "usage": {"completion_tokens": 7}}}
    m.finalize_forced_tool_schema(req, result)
    assert result["status"] == "ok"
    assert result["response"]["choices"][0]["message"]["content"] == (
        "I think you should keep it.")
    assert m.corrections_applied(req, result) == []
    assert m.request_expects_json(req) is True, (
        "…and the floor that judges it is armed: without this the pass-through "
        "would hand prose to a caller that parses JSON")


def test_a_tool_call_under_another_name_is_not_our_answer():
    m, req = _translated_req()
    result = _tool_reply(name="something_else")
    m.finalize_forced_tool_schema(req, result)
    assert result["status"] == "ok"
    assert result["response"]["choices"][0]["message"]["content"] is None


def test_finalize_never_raises_on_a_hostile_body():
    m, req = _translated_req()
    for result in ({}, {"status": "ok"}, {"status": "ok", "response": None},
                   {"status": "ok", "response": {}},
                   {"status": "ok", "response": {"choices": []}},
                   {"status": "ok", "response": {"choices": [None]}},
                   {"status": "ok", "response": {"choices": [{"message": None}]}},
                   {"status": "error", "error": "boom"}):
        m.finalize_forced_tool_schema(req, result)


# ---------------------------------------------------------------------------
# What the removed response_format was ALSO buying (enumerate + re-assert)
# ---------------------------------------------------------------------------

def test_a_translated_request_still_counts_as_structured():
    m, req = _translated_req()
    assert m.request_is_structured(req) is True, (
        "truncation integrity would lapse: a finish_reason=length reply would "
        "read as a benign capped free-form answer instead of failing loud")


def test_a_translated_request_still_expects_json():
    m, req = _translated_req()
    assert m.request_expects_json(req) is True, (
        "the JSON parse floor would lapse — content: null, status 200")


def test_the_declared_schema_survives_the_move_out_of_response_format():
    """Every response-side schema check reads the schema from the payload. The
    translation moves it into a tool's `parameters`, so without this seam they
    all degrade silently to parses-only — a guarantee dropped by the replacement
    rather than re-asserted."""
    m, req = _translated_req()
    assert correction._declared_schema(req) == SCHEMA


def test_declared_schema_falls_back_to_the_payload_for_everyone_else():
    m = _mock_self(not_implemented=())
    req = _req({"messages": [], "response_format": dict(SCHEMA_RF)})
    assert correction._declared_schema(req) == SCHEMA
    assert correction._declared_schema(_req({"messages": []})) is None


def test_an_untranslated_request_does_not_claim_the_markers():
    m = _mock_self(not_implemented=())
    req = _req({"messages": []})
    m.apply_forced_tool_schema(req)
    assert req.forced_tool_schema is None
    assert m.request_is_structured(req) is False
    assert m.request_expects_json(req) is False


def test_queued_request_carries_the_fields():
    """Real dataclass fields, not attributes the correction invents — the
    WAL-recovery path rebuilds QueuedRequest from the DB."""
    scheduler = importlib.import_module("roadstead.scheduler")
    fields = {f.name for f in scheduler.dataclasses.fields(scheduler.QueuedRequest)} \
        if hasattr(scheduler, "dataclasses") else None
    if fields is None:
        import dataclasses
        fields = {f.name for f in dataclasses.fields(scheduler.QueuedRequest)}
    assert {"forced_tool_schema", "forced_tool_name"} <= fields


# ---------------------------------------------------------------------------
# End to end — the two halves are only ever correct TOGETHER
# ---------------------------------------------------------------------------
# The unit halves above can both pass while nothing calls either of them, and a
# request rewrite that ships without its response translation is strictly worse
# than the incompatibility it fixes. These drive a real ProxyService: what the
# BACKEND received, and what the CALLER got back.

from roadstead.backend import BackendClientPool, BackendResponse  # noqa: E402
from roadstead.config import ProxyConfig               # noqa: E402
from roadstead.enriched import ENRICHMENT_HEADERS, WIRE_OPENAI  # noqa: E402
from roadstead.service import ProxyService             # noqa: E402


class _HTTPReq:
    class _Client:
        host = "127.0.0.1"

    client = _Client()
    headers: dict = {}


def _submit_body(endpoint: str, *, timeout_s: float = 5.0) -> dict:
    return {
        "agent_id": "ops", "endpoint": endpoint, "priority": "P1_TURN_SUPPORT",
        "call_site": "triage.classify", "payload_type": "chat_completion",
        "payload": {"messages": [{"role": "user", "content": "triage this"}],
                    "response_format": dict(SCHEMA_RF)},
        "timeout_s": timeout_s,
    }


def _recording_backend(seen: list, *, body: dict):
    async def call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        seen.append(payload)
        return BackendResponse(
            status_code=200, body=body, duration_s=0.01,
            input_tokens=20, output_tokens=18,
            finish_reason=(body.get("choices") or [{}])[0].get("finish_reason"))
    return call


def _assistant_tool_call(arguments=ANSWER, name="triage"):
    return {"choices": [{"message": {"role": "assistant", "content": None,
                                     "tool_calls": [{"id": "call_0",
                                                     "type": "function",
                                                     "function": {"name": name,
                                                                  "arguments": arguments}}]},
                         "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 18}}


async def _round_trip(endpoint: str, backend_body: dict,
                      disclaims=("response_format", "text.format")):
    """A real ProxyService round trip. `disclaims` is what the POLLER would have
    read off that backend's /health — the gate this correction actually reads.
    Default: a build with no constrained decoding. Pass () for an incumbent."""
    seen: list = []
    svc = ProxyService(ProxyConfig())
    svc._backend.call = _recording_backend(seen, body=backend_body)
    await svc.startup()
    svc._config.endpoints[endpoint].not_implemented = frozenset(disclaims)
    try:
        resp = await svc.handle_submit(_submit_body(endpoint), _HTTPReq(),
                                       wire=WIRE_OPENAI)
    finally:
        await svc.shutdown()
    return seen, resp


@pytest.mark.asyncio
async def test_end_to_end_the_backend_sees_a_forced_tool_call():
    seen, resp = await _round_trip("tier1", _assistant_tool_call())
    assert resp.status_code == 200
    assert len(seen) == 1
    wire = seen[0]
    assert "response_format" not in wire, (
        "the field this endpoint's build refuses reached the backend anyway")
    assert wire["tool_choice"] == {"type": "function",
                                   "function": {"name": "triage"}}
    assert wire["tools"][0]["function"]["parameters"] == SCHEMA


@pytest.mark.asyncio
async def test_end_to_end_the_caller_gets_content_not_a_tool_call():
    """The whole point, through the real pipeline: a caller that sent
    `response_format` reads `choices[0].message.content`."""
    _seen, resp = await _round_trip("tier1", _assistant_tool_call())
    body = json.loads(bytes(resp.body))
    message = body["choices"][0]["message"]
    assert json.loads(message["content"]) == json.loads(ANSWER)
    assert "tool_calls" not in message
    assert body["choices"][0]["finish_reason"] == "stop"
    assert resp.headers[ENRICHMENT_HEADERS["corrected"]] == "forced_tool_schema"


@pytest.mark.asyncio
async def test_end_to_end_an_incumbent_backend_is_untouched():
    """An incumbent engine publishes no `not_implemented` at all, so its native
    constrained-decoding path is untouched — which is the outcome the live
    classifier measurement (9.1s translated vs 4.1s native) demands."""
    seen, resp = await _round_trip("tier2", disclaims=(), backend_body={
        "choices": [{"message": {"role": "assistant", "content": ANSWER},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 18}})
    assert seen[0]["response_format"] == SCHEMA_RF
    assert "tools" not in seen[0]
    body = json.loads(bytes(resp.body))
    assert body["choices"][0]["message"]["content"] == ANSWER
    assert ENRICHMENT_HEADERS["corrected"] not in resp.headers


@pytest.mark.asyncio
async def test_end_to_end_a_cut_tool_call_fails_the_request():
    """No silent empty parse, and no fabricated answer: the caller is told."""
    _seen, resp = await _round_trip(
        "tier1", _assistant_tool_call(arguments='{"verdict": "kee'))
    assert resp.status_code == 502
    assert b"truncated structured output" in bytes(resp.body)


@pytest.mark.asyncio
async def test_end_to_end_the_schema_backstop_retry_can_still_recover(monkeypatch):
    """The backstop re-dispatches the TRANSLATED payload, so its reply is a tool
    call too. Without translating that reply the retry can only ever fail — and a
    retry that cannot succeed is worse than no retry, because it is billed."""
    monkeypatch.setenv("ROADSTEAD_PROXY_SCHEMA_BACKSTOP", "1")
    monkeypatch.delenv("ROADSTEAD_PROXY_SCHEMA_BACKSTOP_SHADOW", raising=False)
    bodies = [
        _assistant_tool_call(arguments='{"verdict": "keep"}'),   # missing `score`
        _assistant_tool_call(arguments=ANSWER),
    ]
    seen: list = []

    async def call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        seen.append(payload)
        body = bodies[min(len(seen) - 1, len(bodies) - 1)]
        return BackendResponse(status_code=200, body=body, duration_s=0.01,
                               input_tokens=20, output_tokens=18,
                               finish_reason="tool_calls")

    svc = ProxyService(ProxyConfig())
    svc._backend.call = call
    await svc.startup()
    svc._config.endpoints["tier1"].not_implemented = frozenset({"response_format"})
    try:
        # The retry needs `_MIN_RETRY_BUDGET_S` (5s) left on the caller's
        # deadline — a 5s request has spent some of it by the time the first
        # answer lands, and the backstop would skip the retry silently.
        resp = await svc.handle_submit(_submit_body("tier1", timeout_s=60.0),
                                       _HTTPReq(), wire=WIRE_OPENAI)
    finally:
        await svc.shutdown()

    assert len(seen) == 2, "the schema miss should have bought exactly one retry"
    assert "tools" in seen[1], "the retry re-dispatches the translated payload"
    assert resp.status_code == 200
    body = json.loads(bytes(resp.body))
    assert json.loads(body["choices"][0]["message"]["content"]) == json.loads(ANSWER)
    corrected = resp.headers[ENRICHMENT_HEADERS["corrected"]]
    assert "forced_tool_schema" in corrected and "schema_retried" in corrected


@pytest.mark.asyncio
async def test_end_to_end_the_declared_schema_still_gates_a_translated_reply(
        monkeypatch):
    """The schema left `response_format` for a tool's `parameters`; every
    response-side check must still find it. If it degraded to parses-only, this
    unrecoverable miss would reach the caller as a 200."""
    monkeypatch.setenv("ROADSTEAD_PROXY_SCHEMA_BACKSTOP", "1")
    monkeypatch.delenv("ROADSTEAD_PROXY_SCHEMA_BACKSTOP_SHADOW", raising=False)
    _seen, resp = await _round_trip(
        "tier1", _assistant_tool_call(arguments='{"verdict": "keep"}'))
    assert resp.status_code == 502
    assert b"schema-invalid" in bytes(resp.body)


@pytest.mark.asyncio
async def test_end_to_end_an_unpolled_endpoint_is_untouched():
    """🚨 The gate is CLOSED until the backend says otherwise. Nothing polled
    here, so `not_implemented` is empty — which is also what an unreachable
    /health, a non-200 or a malformed body leaves behind. A proxy that has not
    yet been told must behave exactly like one told "I implement it"."""
    seen, resp = await _round_trip("tier1", _assistant_tool_call(), disclaims=())
    assert seen[0]["response_format"] == SCHEMA_RF
    assert "tools" not in seen[0] and "tool_choice" not in seen[0]
    assert ENRICHMENT_HEADERS["corrected"] not in resp.headers


# ---------------------------------------------------------------------------
# The gate's evidence: the probe, and the poller chore that keeps it fresh
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code=200, payload=None, raises=False):
        self.status_code = status_code
        self._payload = payload
        self._raises = raises

    def json(self):
        if self._raises:
            raise ValueError("not json")
        return self._payload


#: The autouse no-network fixture stubs BOTH of these on the CLASS (a probe
#: reachable from the poller must be, or the unit suite dials real hosts — see
#: tests/conftest.py STUBBED_PROBES). Captured here at import time, before the
#: fixture runs, and restored on the INSTANCE below so these tests exercise the
#: real parsing rather than the stub. The pattern is test_provider_openrouter's.
_REAL_PROBE_JSON = BackendClientPool.probe_json
_REAL_PROBE_NOT_IMPLEMENTED = BackendClientPool.probe_not_implemented


def _backend_with_health(svc, response):
    class _Client:
        async def get(self, path, headers=None):
            assert path == "/health", f"the disclaimer is read from /health, not {path}"
            if isinstance(response, Exception):
                raise response
            return response

    svc._backend._client_for = lambda _url: _Client()
    svc._backend.probe_json = types.MethodType(_REAL_PROBE_JSON, svc._backend)
    svc._backend.probe_not_implemented = types.MethodType(
        _REAL_PROBE_NOT_IMPLEMENTED, svc._backend)


@pytest.mark.parametrize("response, expected", [
    (_FakeResponse(payload={"status": "ok",
                            "not_implemented": ["response_format", "text.format"]}),
     frozenset({"response_format", "text.format"})),
    # A build that publishes the key and disclaims nothing.
    (_FakeResponse(payload={"status": "ok", "not_implemented": []}), frozenset()),
    # Every incumbent engine: no such key.
    (_FakeResponse(payload={"status": "ok"}), None),
    # …and every way of not getting an answer.
    (_FakeResponse(status_code=503, payload={}), None),
    (_FakeResponse(raises=True), None),
    (_FakeResponse(payload={"not_implemented": "response_format"}), None),
    (_FakeResponse(payload={"not_implemented": {"response_format": True}}), None),
    (OSError("connection refused"), None),
])
@pytest.mark.asyncio
async def test_probe_not_implemented_reads_only_a_positive_statement(
        response, expected):
    """None means "cannot tell" and MUST be distinguishable from an empty
    list only in the log — both leave the gate shut. A non-string member is
    dropped rather than coerced: stringifying `{"response_format": true}` would
    invent a declaration the backend never made."""
    svc = ProxyService(ProxyConfig())
    _backend_with_health(svc, response)
    got = await svc._backend.probe_not_implemented(svc._config.endpoints["tier1"])
    assert got == expected


@pytest.mark.asyncio
async def test_the_poller_stores_and_then_CLEARS_the_reading(monkeypatch):
    """Staleness runs in the dangerous direction here: `not_implemented`
    describes a BUILD, so a reading survives a restart into a different one. A
    failed read must therefore CLEAR it, not leave the last one standing — the
    caller then gets the backend's own 400, loud and recoverable next poll,
    instead of the proxy rewriting payloads against a reading that is no longer
    true."""
    svc = ProxyService(ProxyConfig())
    ep = svc._config.endpoints["tier1"]
    readings = [frozenset({"response_format"}), None]

    async def fake_probe(ep_cfg):
        return readings.pop(0)

    async def noop(*_a, **_kw):
        return None

    monkeypatch.setattr(svc._backend, "probe_not_implemented", fake_probe)
    monkeypatch.setattr(svc._backend, "probe_props", noop)
    monkeypatch.setattr(svc._backend, "probe_models", noop)
    monkeypatch.setattr(svc._backend, "probe_model_fingerprint", noop)
    monkeypatch.setattr(svc._backend, "probe_vllm_capacity", noop)
    monkeypatch.setattr(svc._backend, "probe_health", noop)

    await svc._poll_endpoint_once("tier1", ep)
    assert ep.not_implemented == frozenset({"response_format"})

    await svc._poll_endpoint_once("tier1", ep)
    assert ep.not_implemented == frozenset(), (
        "a failed read left the previous reading armed — the one outcome that "
        "keeps the correction firing on evidence that may no longer be true")


@pytest.mark.asyncio
async def test_a_RAISING_probe_also_clears_the_reading(monkeypatch):
    """The OTHER way a read fails, and it was not covered.

    `probe_not_implemented` returns None on every failure it anticipates, so the
    test above exercises the None path. A RAISE takes the `except` branch
    instead, which was a bare log — so the previous reading survived it, which is
    the one outcome the poller's own comment forbids. Both failure modes must
    land on the same value or the invariant is only half-held."""
    svc = ProxyService(ProxyConfig())
    ep = svc._config.endpoints["tier1"]
    calls = {"n": 0}

    async def fake_probe(ep_cfg):
        calls["n"] += 1
        if calls["n"] == 1:
            return frozenset({"response_format"})
        raise RuntimeError("connection reset mid-probe")

    async def noop(*_a, **_kw):
        return None

    for name in ("probe_props", "probe_models", "probe_model_fingerprint",
                 "probe_vllm_capacity", "probe_health"):
        monkeypatch.setattr(svc._backend, name, noop)
    monkeypatch.setattr(svc._backend, "probe_not_implemented", fake_probe)

    await svc._poll_endpoint_once("tier1", ep)
    assert ep.not_implemented == frozenset({"response_format"}), "setup failed"

    await svc._poll_endpoint_once("tier1", ep)
    assert ep.not_implemented == frozenset(), (
        "a RAISING probe left the previous reading armed — the correction stays "
        "armed on evidence that may no longer be true")


@pytest.mark.asyncio
async def test_the_poller_does_not_ask_a_non_chat_endpoint(monkeypatch):
    """An embed/rerank shim can never be sent `response_format`, so asking it
    spends a poll slot on something no code path can use."""
    svc = ProxyService(ProxyConfig())
    asked: list = []

    async def fake_probe(ep_cfg):
        asked.append(ep_cfg.endpoint_class)
        return frozenset({"response_format"})

    async def fake_health(ep_cfg):
        return True

    monkeypatch.setattr(svc._backend, "probe_not_implemented", fake_probe)
    monkeypatch.setattr(svc._backend, "probe_health", fake_health)
    ep = svc._config.endpoints["embed"]
    assert ep.kind != "chat"
    await svc._poll_endpoint_once("embed", ep)
    assert asked == []
    assert ep.not_implemented == frozenset()
