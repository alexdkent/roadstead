"""A STRUCTURED response can be well-formed and still carry no answer.

Ledger `tier3-json-object-empty-brace`. On 2026-07-31 15:06:52 UTC tier3
restarted with ``disable_any_whitespace`` and every bare-``json_object``
request began returning the two characters ``{}``. Measured off the live
proxy's queue.db afterwards, over 2026-07-31 15:06 → 2026-08-01 23:03 UTC:

    tier3 structured requests BEFORE (07-31 00:00-14:59)   695, empty   0  =  0.0%
    tier3 structured requests AFTER  (07-31 15:00 onward) 3112, empty 1271 = 40.8%
    every other endpoint, same period                     max empty rate  0.0%

Nothing detected it for 31 hours, because ``{}`` is WELL-FORMED: it has
``finish_reason=stop``, no error field, non-empty content, fewer than the 40
words ``_is_degenerate_text`` needs, and it parses, so
``enforce_structured_validity`` passes it too. The only thing wrong with it is
that it carries no ANSWER — a property of the caller's contract, not of the
JSON.

These tests pin the four cases that matter, and the THIRD is the one that
decides whether this detector is usable at all: an extractor answering
"I found nothing" is legitimate and must never alarm.

Self-contained in the house style of test_json_object_guard.py: the real
``Correction`` methods are bound to a lightweight mock ``self`` carrying just
the ``.state`` fields they touch.
"""
import importlib
import logging
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # originfleet/
sys.path.insert(0, str(REPO))

correction = importlib.import_module("originfleet.llmproxy.correction")
lp_obs = importlib.import_module("originfleet.llmproxy.observability")
fw_obs = importlib.import_module("originfleet.framework.observability")
hooks = importlib.import_module("originfleet.llmproxy.hooks")
C = correction.Correction


# ---------------------------------------------------------------------------
# Fixtures — the four response shapes, and the request shapes that produce them
# ---------------------------------------------------------------------------

# 1. THE BUG. A bare json_object request; the proxy strips the constraint
#    (apply_json_object_guard) and flags json_object_stripped, which keeps
#    request_is_structured True. The backend returns the two-character document.
BARE_STRIPPED_PAYLOAD = {"messages": [{"role": "user", "content": "critique"}],
                         "max_tokens": 400}

# 2. VACUOUS. A declared schema with NO `required` — `{}` and
#    `{"voice_match": null}` both satisfy it, which is exactly why the fix had
#    to STRIP rather than substitute a permissive schema.
NO_REQUIRED_SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "critic", "schema": {
        "type": "object",
        "properties": {"voice_match": {"type": ["boolean", "null"]},
                       "notes": {"type": "string"}}}},
}

# 3. LEGITIMATELY EMPTY. A real contract: `facts` is REQUIRED, and an empty
#    list is a complete, correct answer meaning "I found nothing". This is a
#    healthy response and must be counted as such.
REQUIRED_SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "extract", "strict": True, "schema": {
        "type": "object",
        "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
        "required": ["facts"]}},
}


def _req(payload, *, endpoint="thinker", call_site="auto_approve.critic",
         stripped=False, ptype="chat_completion"):
    r = types.SimpleNamespace()
    r.payload = dict(payload)
    r.payload_type = ptype
    r.endpoint = endpoint
    r.stream = False
    r.request_id = "r1"
    r.agent_id = "sidekick"
    r.call_site = call_site
    r.json_object_stripped = stripped
    return r


def _result(content, *, status="ok", finish="stop"):
    return {"status": status, "response": {
        "choices": [{"message": {"role": "assistant", "content": content},
                     "finish_reason": finish}],
        "usage": {"completion_tokens": 2}}}


def _mock_self():
    state = types.SimpleNamespace()
    state.structured_empty_total = 0
    state.structured_empty_by_call_site = {}
    state.structured_empty_window = {}
    m = types.SimpleNamespace(state=state)
    for name in ("detect_structured_empty", "request_is_structured",
                 "extract_grammar"):
        setattr(m, name, getattr(C, name).__get__(m, C))
    return m


# ---------------------------------------------------------------------------
# The four cases
# ---------------------------------------------------------------------------

def test_case1_literal_empty_brace_is_detected():
    """THE BUG. `{}` on a structured request is a structured_empty event."""
    m = _mock_self()
    res = _result("{}")
    m.detect_structured_empty(_req(BARE_STRIPPED_PAYLOAD, stripped=True), res)
    assert m.state.structured_empty_total == 1
    assert m.state.structured_empty_by_call_site == {"auto_approve.critic": 1}
    rates = lp_obs.structured_empty_rates(m.state.structured_empty_window, 0.0)
    assert rates["thinker"]["empty"] == 1
    assert rates["thinker"]["n"] == 1


def test_case2_vacuously_satisfying_object_is_detected():
    """A schema with no `required` is satisfied by an object with nothing in
    it. That is the same defect wearing a schema, and it must still fire —
    otherwise "just add a schema" ships a no-op fix."""
    m = _mock_self()
    p = dict(BARE_STRIPPED_PAYLOAD, response_format=NO_REQUIRED_SCHEMA)
    m.detect_structured_empty(
        _req(p), _result('{"voice_match": null, "notes": ""}'))
    assert m.state.structured_empty_total == 1


def test_case3_legitimately_empty_but_valid_does_NOT_alarm():
    """THE CASE THAT DECIDES WHETHER THIS IS USABLE.

    An extractor answering "I found nothing" returns `{"facts": []}` against a
    schema that REQUIRES `facts`. The field it was asked for is present and
    its emptiness IS the answer. Nothing malfunctioned; alarming here would
    make the detector noise and it would be turned off.

    It is also counted as a HEALTHY denominator sample — a legitimate empty
    answer is evidence the endpoint is working, and excluding it would let a
    quiet-but-correct extractor drag the rate up."""
    m = _mock_self()
    p = dict(BARE_STRIPPED_PAYLOAD, response_format=REQUIRED_SCHEMA)
    m.detect_structured_empty(_req(p), _result('{"facts": []}'))
    assert m.state.structured_empty_total == 0
    rates = lp_obs.structured_empty_rates(m.state.structured_empty_window, 0.0)
    assert rates["thinker"] == {
        "n": 1, "empty": 0, "rate": 0.0, "top_call_site": None,
        "evaluated": False,
    }


def test_case4_normal_response_does_NOT_alarm():
    """A real answer is a healthy denominator sample and nothing else."""
    m = _mock_self()
    p = dict(BARE_STRIPPED_PAYLOAD, response_format=REQUIRED_SCHEMA)
    m.detect_structured_empty(_req(p), _result('{"facts": ["a", "b"]}'))
    assert m.state.structured_empty_total == 0
    rates = lp_obs.structured_empty_rates(m.state.structured_empty_window, 0.0)
    assert rates["thinker"]["n"] == 1 and rates["thinker"]["empty"] == 0


# ---------------------------------------------------------------------------
# Scope — what must NOT be counted
# ---------------------------------------------------------------------------

def test_unstructured_request_returning_empty_brace_is_ignored():
    """`{}` from a FREETEXT request is not a structured failure — the caller
    declared no contract and is not parsing fields out of it."""
    m = _mock_self()
    m.detect_structured_empty(_req(BARE_STRIPPED_PAYLOAD), _result("{}"))
    assert m.state.structured_empty_total == 0
    assert m.state.structured_empty_window == {}


def test_already_failed_response_is_neither_event_nor_sample():
    """A response an earlier guard already flipped to status=error is not a
    SILENT failure — the caller can see it. Counting it would let a noisy
    backend inflate the silent-failure rate."""
    m = _mock_self()
    m.detect_structured_empty(
        _req(BARE_STRIPPED_PAYLOAD, stripped=True),
        _result("{}", status="error"))
    assert m.state.structured_empty_total == 0
    assert m.state.structured_empty_window == {}


def test_schema_invalid_empty_object_is_left_to_the_schema_backstop():
    """`{}` against a schema that REQUIRES fields fails the schema outright —
    the backstop owns that, and double-counting it here would let genuine
    schema misses inflate the empty RATE."""
    m = _mock_self()
    p = dict(BARE_STRIPPED_PAYLOAD, response_format=REQUIRED_SCHEMA)
    m.detect_structured_empty(_req(p), _result("{}"))
    assert m.state.structured_empty_total == 0
    assert m.state.structured_empty_window == {}


def test_non_json_content_is_left_to_the_validity_floor():
    m = _mock_self()
    m.detect_structured_empty(
        _req(BARE_STRIPPED_PAYLOAD, stripped=True), _result("not json at all"))
    assert m.state.structured_empty_total == 0


def test_empty_string_content_is_the_empty_completion_gates_domain():
    m = _mock_self()
    m.detect_structured_empty(
        _req(BARE_STRIPPED_PAYLOAD, stripped=True), _result(""))
    assert m.state.structured_empty_total == 0
    assert m.state.structured_empty_window == {}


def test_false_value_is_a_real_answer_not_emptiness():
    """`false` and `0` are answers. Only null/""/[]/{} are emptiness."""
    m = _mock_self()
    p = dict(BARE_STRIPPED_PAYLOAD, response_format=NO_REQUIRED_SCHEMA)
    m.detect_structured_empty(_req(p), _result('{"voice_match": false}'))
    assert m.state.structured_empty_total == 0


# ---------------------------------------------------------------------------
# It is TELEMETRY, not a gate
# ---------------------------------------------------------------------------

def test_detector_never_mutates_the_response():
    """A retry loop here would turn a silent failure into an EXPENSIVE silent
    failure — the condition is a backend launch flag, so every retry re-earns
    the same `{}` at full cost."""
    m = _mock_self()
    res = _result("{}")
    before = repr(res)
    m.detect_structured_empty(_req(BARE_STRIPPED_PAYLOAD, stripped=True), res)
    assert repr(res) == before
    assert res["status"] == "ok"
    assert "error" not in res


def test_detector_is_total_on_garbage_input():
    """Fail-open in every direction — a broken req, a broken result, a state
    missing its fields. Telemetry must never break a response."""
    m = _mock_self()
    broken = types.SimpleNamespace()  # no attributes at all
    m.detect_structured_empty(broken, _result("{}"))
    m.detect_structured_empty(_req(BARE_STRIPPED_PAYLOAD, stripped=True), None)
    m.detect_structured_empty(_req(BARE_STRIPPED_PAYLOAD, stripped=True), {})
    bad = types.SimpleNamespace(state=types.SimpleNamespace())
    for name in ("detect_structured_empty", "request_is_structured",
                 "extract_grammar"):
        setattr(bad, name, getattr(C, name).__get__(bad, C))
    bad.detect_structured_empty(
        _req(BARE_STRIPPED_PAYLOAD, stripped=True), _result("{}"))


# ---------------------------------------------------------------------------
# The observability seam + the greppable marker
# ---------------------------------------------------------------------------

def test_event_reaches_the_framework_degradation_seam():
    """Same seam the comment critic already uses, so it also lands on the
    fleet-wide counter.

    Since the 2026-08-31 import sever, ``correction.py`` reports through
    ``llmproxy.hooks.degradation`` rather than importing the framework
    directly, and ``llmproxy/__main__.py`` registers the framework function as
    the sink at startup. This test wires the SAME sink production wires, so it
    still proves the event reaches the fleet-wide counter end to end."""
    fw_obs.reset_counters()
    hooks.set_degradation_sink(fw_obs.degradation)
    try:
        m = _mock_self()
        m.detect_structured_empty(_req(BARE_STRIPPED_PAYLOAD, stripped=True),
                                  _result("{}"))
        assert fw_obs.get_counter("llmproxy.structured_empty") == 1
    finally:
        hooks.set_degradation_sink(None)


def test_degradation_still_reported_with_no_sink_wired(caplog):
    """The other half of the sever: a STANDALONE deployment registers no sink,
    and the degradation must still be reported rather than silently dropped.
    Pins the built-in fallback so removing it can't pass unnoticed."""
    hooks.set_degradation_sink(None)
    with caplog.at_level(logging.WARNING, logger=hooks.logger.name):
        m = _mock_self()
        m.detect_structured_empty(_req(BARE_STRIPPED_PAYLOAD, stripped=True),
                                  _result("{}"))
    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "DEGRADATION" in line
    assert "component=llmproxy" in line
    assert "reason=structured_empty" in line


def test_event_carries_the_greppable_marker_and_caller_identity(caplog):
    with caplog.at_level(logging.WARNING, logger=correction.logger.name):
        m = _mock_self()
        m.detect_structured_empty(_req(BARE_STRIPPED_PAYLOAD, stripped=True),
                                  _result("{}"))
    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "LLMPROXY_STRUCTURED_EMPTY" in line
    assert "agent=sidekick" in line
    assert "call_site=auto_approve.critic" in line
    assert "model=thinker" in line


# ---------------------------------------------------------------------------
# Streaming is not a blind spot
# ---------------------------------------------------------------------------

def test_streaming_reassembly_is_detected_too():
    m = _mock_self()
    m.detect_structured_empty(
        _req(BARE_STRIPPED_PAYLOAD, stripped=True), {},
        content="{}", stream=True)
    assert m.state.structured_empty_total == 1
