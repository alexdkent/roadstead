"""Builder happy-path + parity tests for the Phase 3 schema-repair backstop
(`Correction.maybe_repair_schema` + the pure `_conform_body`/helpers).

Contract: docs/llmproxy_phase3_schema_backstop_contract.md. On a structured/tool
SYNC response whose JSON is wrong (trailing prose / fenced / malformed /
schema-invalid / bad tool_calls.arguments) the backstop runs
json-repair → schema-validate → one bounded retry (error fed back) → fail-loud
deferrable. Default OFF == byte-identical.

Invariants pinned here (builder scope — happy path + parity; the independent
adversarial track owns the hostile both-seam fake-backend matrix + guard-bite):
  - the pure classifier repairs / passes / fails the right way per pathology;
  - flag OFF → total no-op (byte-identical, backend never called);
  - valid response → fast-path no-op (backend never called);
  - in-memory repair (trailing prose/fence) swaps the corrected body, no backend;
  - a schema-type miss repair-can't-fix → one retry; a valid retry recovers;
  - retry-still-invalid → fail-loud (status=error, code=schema_invalid, no body,
    _schema_unrecoverable set, never a raise);
  - shadow → detect-only (counts, returns original, no backend, no fail-loud);
  - retry is bounded (concurrency cap + caller deadline);
  - non-structured request → skipped; fail-open on an internal error.

Self-contained; runs under pytest OR as a plain script.
"""
import asyncio
import importlib
import os
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(REPO))

correction = importlib.import_module("roadstead.correction")
config = importlib.import_module("roadstead.config")
C = correction.Correction

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}


# --------------------------------------------------------------------------- #
# Pure classifier (`_conform_body`) — the core repair/validate logic.
# --------------------------------------------------------------------------- #
def _body(content=None, tool_args=None):
    msg = {}
    if content is not None:
        msg["content"] = content
    if tool_args is not None:
        msg["tool_calls"] = [{"function": {"name": "f", "arguments": tool_args}}]
    return {"choices": [{"message": msg}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}


def test_conform_valid_is_ok():
    assert correction._conform_body(_body('{"a": 1}'), SCHEMA)[0] == "ok"


def test_conform_trailing_prose_repaired():
    st, nb = correction._conform_body(_body('sure thing: {"a": 1} — done'), SCHEMA)
    assert st == "repaired"
    assert nb["choices"][0]["message"]["content"] == '{"a": 1}'


def test_conform_markdown_fenced_repaired():
    st, nb = correction._conform_body(_body('```json\n{"a": 1}\n```'), SCHEMA)
    assert st == "repaired" and '"a": 1' in nb["choices"][0]["message"]["content"]


def test_conform_truncated_repaired():
    assert correction._conform_body(_body('{"a": 1'), SCHEMA)[0] == "repaired"


def test_conform_schema_type_miss_failed():
    # valid JSON, but a=str violates a:integer and repair can't change the type.
    assert correction._conform_body(_body('{"a": "x"}'), SCHEMA)[0] == "failed"


def test_conform_no_schema_valid_json_ok():
    assert correction._conform_body(_body('{"anything": true}'), None)[0] == "ok"


def test_conform_no_schema_garbage_failed():
    assert correction._conform_body(_body('utterly not json <<<'), None)[0] == "failed"


def test_conform_malformed_declared_schema_degrades_to_parse_only():
    # North-face: a caller declares a broken schema → validate parse-only, never
    # fail the response over the caller's bug.
    assert correction._conform_body(_body('{"a": 1}'), {"type": "not-a-real-type"})[0] == "ok"


def test_conform_empty_content_no_tools_ok():
    assert correction._conform_body(_body(''), SCHEMA)[0] == "ok"


def test_conform_bad_tool_args_repaired():
    st, nb = correction._conform_body(_body(tool_args='{k: 1,}'), None)
    assert st == "repaired"
    import json
    assert json.loads(nb["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]) == {"k": 1}


def test_conform_unrepairable_tool_args_failed():
    assert correction._conform_body(_body(tool_args='@@@@@'), None)[0] == "failed"


def test_extract_declared_schema_shapes():
    rf = {"response_format": {"type": "json_schema", "json_schema": {"schema": SCHEMA}}}
    assert correction._extract_declared_schema(rf) == SCHEMA
    assert correction._extract_declared_schema({"guided_json": SCHEMA}) == SCHEMA
    assert correction._extract_declared_schema({"extra_body": {"guided_json": SCHEMA}}) == SCHEMA
    assert correction._extract_declared_schema({"grammar": "root ::= object"}) is None
    assert correction._extract_declared_schema({}) is None


# --------------------------------------------------------------------------- #
# maybe_repair_schema (async, over a real Correction + mock state).
# --------------------------------------------------------------------------- #
def _req(deadline_offset=120.0):
    r = types.SimpleNamespace()
    r.payload_type = "chat_completion"
    r.payload = {
        "messages": [{"role": "user", "content": "give me a"}],
        "response_format": {"type": "json_schema", "json_schema": {"schema": SCHEMA}},
    }
    r.endpoint = "reasoner"
    r.request_id = "rid-schema-1"
    r.call_site = "knowledge.extract"
    r.timeout_deadline = time.monotonic() + deadline_offset
    r.agent_id = "knowledge"
    r.priority = config.LLMPriority.P1_TURN_SUPPORT
    r.session_id = None
    r.turn_id = None
    r.caller_id = None
    return r


def _result(content, status="ok"):
    return {"status": status,
            "response": {"choices": [{"message": {"content": content}}],
                         "usage": {"prompt_tokens": 7, "completion_tokens": 4}}}


def _mock_state(backend_call):
    state = types.SimpleNamespace()
    for name in ("schema_detected", "schema_repaired", "schema_retry_recovered",
                 "schema_unrecoverable", "schema_invalid_stream", "schema_retry_inflight"):
        setattr(state, name, 0)
    state.schema_by_call_site = {}
    state.metrics = types.SimpleNamespace(record=lambda sample: None)
    state._persisted = []
    state.queue_db = types.SimpleNamespace(
        persist_complete=lambda *a, **k: state._persisted.append((a, k)))
    ep = types.SimpleNamespace(role="reasoner")
    state.config = types.SimpleNamespace(endpoints={"reasoner": ep})
    state.backend = types.SimpleNamespace(call=backend_call)
    return state


def _backend_returning(*bodies):
    seq = list(bodies)
    calls = []

    async def _call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        calls.append(payload)
        body = seq.pop(0) if seq else {"choices": [{"message": {"content": "{}"}}]}
        return types.SimpleNamespace(
            body=body, duration_s=0.3, input_tokens=8, output_tokens=5,
            finish_reason="stop")

    _call.calls = calls
    return _call


def _run(state, req, result):
    c = C(state)
    asyncio.run(c.maybe_repair_schema(req, result))


def _enforce():
    os.environ["COLLECTIVE_PROXY_SCHEMA_BACKSTOP"] = "1"
    os.environ.pop("COLLECTIVE_PROXY_SCHEMA_BACKSTOP_SHADOW", None)


def _clear_flags():
    os.environ.pop("COLLECTIVE_PROXY_SCHEMA_BACKSTOP", None)
    os.environ.pop("COLLECTIVE_PROXY_SCHEMA_BACKSTOP_SHADOW", None)


def test_flag_off_is_total_noop():
    _clear_flags()
    backend = _backend_returning()
    st = _mock_state(backend)
    res = _result('garbage not json')
    _run(st, _req(), res)
    assert st.schema_detected == 0
    assert backend.calls == []
    assert res["response"]["choices"][0]["message"]["content"] == 'garbage not json'


def test_valid_response_fast_path_noop():
    _enforce()
    try:
        backend = _backend_returning()
        st = _mock_state(backend)
        res = _result('{"a": 5}')
        _run(st, _req(), res)
    finally:
        _clear_flags()
    assert st.schema_detected == 0          # never flagged
    assert backend.calls == []              # backend never re-called


def test_trailing_prose_repaired_in_memory():
    _enforce()
    try:
        backend = _backend_returning()
        st = _mock_state(backend)
        res = _result('here: {"a": 5} ok?')
        _run(st, _req(), res)
    finally:
        _clear_flags()
    assert st.schema_detected == 1
    assert st.schema_repaired == 1
    assert backend.calls == []              # in-memory repair — no backend
    assert res["response"]["choices"][0]["message"]["content"] == '{"a": 5}'
    assert st._persisted                    # corrected row re-persisted
    assert st.schema_by_call_site["knowledge.extract"]["repaired"] == 1


def test_schema_miss_recovered_by_retry():
    _enforce()
    try:
        good = {"choices": [{"message": {"content": '{"a": 7}'}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 2}}
        backend = _backend_returning(good)
        st = _mock_state(backend)
        res = _result('{"a": "not-an-int"}')     # valid JSON, schema-invalid
        _run(st, _req(), res)
    finally:
        _clear_flags()
    assert st.schema_detected == 1
    assert st.schema_repaired == 0
    assert st.schema_retry_recovered == 1
    assert len(backend.calls) == 1                # exactly one retry
    # the retry fed the error back as an extra user turn
    assert backend.calls[0]["messages"][-1]["role"] == "user"
    assert res["response"]["choices"][0]["message"]["content"] == '{"a": 7}'
    assert st.schema_retry_inflight == 0          # released


def test_retry_still_invalid_fails_loud_deferrable():
    _enforce()
    try:
        stillbad = {"choices": [{"message": {"content": '{"a": "still-str"}'}}]}
        backend = _backend_returning(stillbad)
        st = _mock_state(backend)
        res = _result('{"a": "not-an-int"}')
        _run(st, _req(), res)
    finally:
        _clear_flags()
    assert st.schema_unrecoverable == 1
    assert res["status"] == "error"
    assert res["code"] == "schema_invalid"
    assert "response" not in res                  # body dropped, not handed back
    assert res.get("_schema_unrecoverable") is True
    assert st.schema_retry_inflight == 0


def test_shadow_detects_only():
    os.environ["COLLECTIVE_PROXY_SCHEMA_BACKSTOP"] = "1"
    os.environ["COLLECTIVE_PROXY_SCHEMA_BACKSTOP_SHADOW"] = "1"
    try:
        backend = _backend_returning()
        st = _mock_state(backend)
        res = _result('here: {"a": 5} ok?')
        _run(st, _req(), res)
    finally:
        _clear_flags()
    assert st.schema_detected == 1
    assert st.schema_repaired == 0                # not applied
    assert backend.calls == []                    # no retry in shadow
    assert res["response"]["choices"][0]["message"]["content"] == 'here: {"a": 5} ok?'


def test_retry_skipped_when_concurrency_saturated():
    _enforce()
    try:
        backend = _backend_returning()
        st = _mock_state(backend)
        st.schema_retry_inflight = 2              # cap hit
        res = _result('{"a": "not-an-int"}')
        _run(st, _req(), res)
    finally:
        _clear_flags()
    assert backend.calls == []                    # no retry attempted
    assert st.schema_unrecoverable == 1           # → fail-loud


def test_retry_skipped_when_deadline_exhausted():
    _enforce()
    try:
        backend = _backend_returning()
        st = _mock_state(backend)
        res = _result('{"a": "not-an-int"}')
        _run(st, _req(deadline_offset=-1.0), res)  # already past deadline
    finally:
        _clear_flags()
    assert backend.calls == []
    assert st.schema_unrecoverable == 1


def test_non_structured_request_skipped():
    _enforce()
    try:
        backend = _backend_returning()
        st = _mock_state(backend)
        req = _req()
        req.payload = {"messages": [{"role": "user", "content": "hi"}]}   # no schema/tools
        res = _result('garbage not json')
        _run(st, req, res)
    finally:
        _clear_flags()
    assert st.schema_detected == 0
    assert backend.calls == []


def test_tool_call_with_prose_content_not_flagged():
    # A PURE tool call (tools, no response_format) whose assistant message carries a
    # natural-language preamble alongside valid tool_calls must NOT be flagged — the
    # content isn't a JSON contract, only the tool args are.
    _enforce()
    try:
        backend = _backend_returning()
        st = _mock_state(backend)
        req = _req()
        req.payload = {"messages": [{"role": "user", "content": "weather?"}],
                       "tools": [{"type": "function", "function": {"name": "get"}}]}
        res = {"status": "ok", "response": {"choices": [{"message": {
            "content": "Let me look that up for you.",
            "tool_calls": [{"function": {"name": "get", "arguments": '{"city": "NYC"}'}}]}}]}}
        _run(st, req, res)
    finally:
        _clear_flags()
    assert st.schema_detected == 0                # not flagged
    assert backend.calls == []                    # no retry
    assert res["response"]["choices"][0]["message"]["content"] == "Let me look that up for you."


def test_fail_open_on_internal_error():
    _enforce()
    try:
        backend = _backend_returning()
        st = _mock_state(backend)
        # Corrupt the response shape so an internal access raises — the guard must
        # swallow it and leave `result` untouched (fail-open).
        res = {"status": "ok", "response": {"choices": "not-a-list"}}
        _run(st, _req(), res)
    finally:
        _clear_flags()
    assert res["status"] == "ok"                  # untouched — never broke the path


if __name__ == "__main__":  # plain-script runner
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"ok  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")


def _req_bare_grammar():
    """A raw GBNF grammar whose language is NOT JSON — the exact example named
    in `request_expects_json`'s docstring."""
    r = _req()
    r.payload = {
        "messages": [{"role": "user", "content": "Is the sky blue?"}],
        "grammar": 'root ::= ("yes" | "no")',
    }
    r.call_site = "agent.bare_grammar"
    return r


def test_bare_token_grammar_is_not_parse_gated():
    """REGRESSION: a bare-token GBNF grammar legitimately emits non-JSON, and
    the schema-backstop must not demand JSON of it.

    The backstop passed `request_is_structured()` as `expect_json_content`,
    but that predicate is true for ANY constrained output. A grammar
    constrains output to an arbitrary language, so `yes` is CORRECT — yet it
    was repair-failed, retry-failed and turned into a 502
    ("produced schema-invalid structured output"). Measured live 2026-07-31:
    GBNF worked direct to llama.cpp and 502'd through the proxy on all three
    tiers. `request_expects_json()` is the narrower predicate that already
    documents this exact case; the backstop must use it.
    """
    _enforce()
    try:
        backend = _backend_returning()
        st = _mock_state(backend)
        res = _result("yes")
        _run(st, _req_bare_grammar(), res)
    finally:
        _clear_flags()
    assert res["status"] == "ok", f"bare grammar was rejected: {res.get('error')}"
    assert res["response"]["choices"][0]["message"]["content"] == "yes"
    assert st.schema_detected == 0, "backstop must not even flag a non-JSON grammar"
    assert backend.calls == [], "must not burn a retry re-dispatching a valid answer"


def test_json_object_rooted_grammar_is_still_parse_gated():
    """The narrowing must NOT go too far: a grammar rooted at a JSON OBJECT
    still implies JSON content, so malformed output there must still be caught."""
    _enforce()
    try:
        backend = _backend_returning()
        st = _mock_state(backend)
        r = _req()
        r.payload = {
            "messages": [{"role": "user", "content": "give me a"}],
            "grammar": 'root ::= "{" ws "\\"a\\"" ws ":" ws number ws "}"\nws ::= [ \\t\\n]*\nnumber ::= [0-9]+',
        }
        res = _result("not json at all")
        _run(st, r, res)
    finally:
        _clear_flags()
    assert st.schema_detected == 1, "JSON-rooted grammar must still be parse-gated"
