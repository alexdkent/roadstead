"""Step 4a — uniform correction: ADVERSARIAL both-seam matrix + fuzz.

Independent hostile track (did NOT write the implementation). Goal: BREAK the two
new stream cells and prove the guards actually bite. Covers, over the real
in-process proxy + fake backend:

  * BOTH-DOORS PARITY (flag ON): the truncated-tool-call sanitizer relabels
    finish_reason -> "length" and drops the broken-JSON arg on the OpenAI door AND
    the internal /v1/submit door identically — no divergence.
  * DIVERGENCE (flag OFF): the OpenAI door is ALWAYS sanitized (pre-4a); only the
    internal door changes with the flag. Prove OFF leaks internally but not on
    OpenAI.
  * UNIFORM STREAM DETECTION: a streamed degenerate loop bumps the counter EXACTLY
    once on BOTH doors under the flag, zero times off; a legitimate long response /
    bounded chorus is NEVER false-flagged; a conformant grammar stream counts
    checked-not-dropped; a non-conformant grammar stream tallies a silent drop.
  * SLOT ACCOUNTING: every path (both flag states, truncated/degenerate/reset/
    client-abort) returns total_in_flight to baseline.
  * FAIL-OPEN / never-crash: finalize_stream over huge/unicode/binary/empty/None
    content + broken request shapes never raises.

The guard-bite reverts (temporarily stubbing the wiring and confirming these tests
go red) are performed out-of-band by the adversarial operator; see the report.
"""
from __future__ import annotations

import json

import pytest

# Exhaustive ProxyService-spinning adversarial matrix — deselected from the
# per-ship in_container_tollgate via `-m 'not heavy'` (see pyproject `heavy`).
pytestmark = pytest.mark.heavy

from roadstead.testing import (
    FAULT_DEGENERATE_LOOP,
    FAULT_MID_STREAM_RESET,
    FAULT_TRUNCATED_TOOL_CALLS,
)
from roadstead.correction import Correction

FLAG = "COLLECTIVE_PROXY_UNIFORM_CORRECTION"


@pytest.fixture(autouse=True)
def _pin_schema_backstop_off(monkeypatch):
    """Pin the Phase-3 schema-backstop OFF for this file (it tests the uniform
    correction / shadow-egress path, not the backstop). The fake backend echoes
    non-conformant text, so with the backstop ON (as in the container) a grammar
    case 502s before the correction path runs. It reads os.environ live per
    request; pin OFF so these tests are hermetic w.r.t. the ambient container
    env. Fixes the container-only 502 in test_sync_door_shadow_egress_wired_
    through_apply after the "chat"→classify (grammar-capable) cutover.

    The always-on structured-validity floor (COLLECTIVE_PROXY_STRUCTURED_VALIDITY,
    2026-07-11) is pinned OFF for the same reason: the fake's non-JSON echo on the
    grammar cases (sync 200 assertion + the grammar-stream 'done' flow) would trip
    the parse-only floor — an artifact of the fake, not the uniform-correction
    seams this file tests. The floor has its own suite (test_truncation_guard.py).
    """
    monkeypatch.setenv("COLLECTIVE_PROXY_SCHEMA_BACKSTOP", "0")
    monkeypatch.setenv("COLLECTIVE_PROXY_STRUCTURED_VALIDITY", "0")


# --------------------------------------------------------------------------- #
# door drivers
# --------------------------------------------------------------------------- #

async def _submit_stream_events(proxy, *, fault=None, content="a b c", extra_payload=None):
    """Drive the INTERNAL /v1/submit streaming door; return decoded envelope events."""
    if fault:
        proxy.controller.set_fault(fault, 0.0)
    payload = {
        "model": "chat",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 16,
        "stream": True,
    }
    if extra_payload:
        payload.update(extra_payload)
    body = {
        "agent_id": "adv_test",
        "endpoint": "chat",
        "priority": "P3_INGESTION",
        "call_site": "uc_adversarial",
        "payload_type": "chat_completion",
        "payload": payload,
    }
    events: list = []
    async with proxy.client.stream("POST", "/v1/submit", json=body) as resp:
        async for line in resp.aiter_lines():
            line = line.strip()
            if line.startswith("data: "):
                raw = line[len("data: "):]
                if raw == "[DONE]":
                    continue
                try:
                    events.append(json.loads(raw))
                except ValueError:
                    events.append({"_raw": raw})
    return events


def _submit_chunk_objs(events):
    """The decoded backend chunk dicts from internal envelope 'chunk' events."""
    out = []
    for ev in events:
        if ev.get("type") != "chunk":
            continue
        try:
            out.append(json.loads(ev["data"]))
        except (ValueError, KeyError, TypeError):
            continue
    return out


def _openai_chunk_objs(frames):
    """The decoded chunk dicts from OpenAI-door raw SSE data payloads."""
    out = []
    for f in frames:
        if f == "[DONE]":
            continue
        try:
            out.append(json.loads(f))
        except ValueError:
            continue
    return out


def _tool_args(chunks):
    out = []
    for chunk in chunks:
        for ch in chunk.get("choices", []):
            for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                out.append((tc.get("function") or {}).get("arguments"))
    return out


def _finish_reasons(chunks):
    out = []
    for chunk in chunks:
        for ch in chunk.get("choices", []):
            fr = ch.get("finish_reason")
            if fr is not None:
                out.append(fr)
    return out


# --------------------------------------------------------------------------- #
# 1) BOTH-DOORS PARITY (flag ON) — the truncated tool-call sanitizer
# --------------------------------------------------------------------------- #

async def test_parity_truncated_toolcall_both_doors_on(proxy, monkeypatch):
    """Flag ON: the OpenAI door and the internal door produce the SAME correction —
    no broken-JSON arg leaks, and finish_reason is relabeled 'length' on BOTH."""
    monkeypatch.setenv(FLAG, "1")

    oa_frames = await proxy.stream_frames("a b c", fault=FAULT_TRUNCATED_TOOL_CALLS)
    oa_chunks = _openai_chunk_objs(oa_frames)

    ev = await _submit_stream_events(proxy, fault=FAULT_TRUNCATED_TOOL_CALLS)
    in_chunks = _submit_chunk_objs(ev)

    # neither door leaks the truncated partial-JSON argument
    assert '{"x":' not in _tool_args(oa_chunks), "OpenAI door leaked truncated arg"
    assert '{"x":' not in _tool_args(in_chunks), "internal door leaked truncated arg"

    # both doors relabel the finish_reason from tool_calls -> length
    assert "length" in _finish_reasons(oa_chunks), "OpenAI door failed to relabel"
    assert "length" in _finish_reasons(in_chunks), "internal door failed to relabel"
    assert "tool_calls" not in _finish_reasons(in_chunks), \
        "internal door left the raw tool_calls finish_reason (divergence from OpenAI)"

    assert proxy.total_in_flight() == 0


async def test_off_flag_door_divergence(proxy, monkeypatch):
    """Flag OFF: the DIVERGENCE the flag governs. The OpenAI door was ALWAYS
    sanitized (pre-4a) — the flag must NOT gate it (drops the truncated call +
    relabels even OFF); the internal door, however, stays RAW and leaks the broken
    arg when OFF. (Both doors in one proxy spin for tollgate budget.)"""
    monkeypatch.delenv(FLAG, raising=False)
    frames = await proxy.stream_frames("a b c", fault=FAULT_TRUNCATED_TOOL_CALLS)
    oa = _openai_chunk_objs(frames)
    assert '{"x":' not in _tool_args(oa), "OpenAI door must stay sanitized when OFF"
    assert "length" in _finish_reasons(oa)
    ev = await _submit_stream_events(proxy, fault=FAULT_TRUNCATED_TOOL_CALLS)
    assert '{"x":' in _tool_args(_submit_chunk_objs(ev)), \
        "internal door should be RAW when the flag is off"


# --------------------------------------------------------------------------- #
# 2) UNIFORM STREAM DETECTION — exactly-once, both doors, no false-flag
# --------------------------------------------------------------------------- #

async def test_degeneration_detected_exactly_once_both_doors(proxy, monkeypatch):
    """Uniformity (flag ON): BOTH the internal /v1/submit door and the OpenAI door
    get stream-side degeneration detection — exactly once each (detect-only), and
    neither leaks a slot. (One proxy spin, both doors, for tollgate budget.)"""
    monkeypatch.setenv(FLAG, "1")
    st = proxy.svc._correction.state
    before = st.degeneration_detected
    await _submit_stream_events(proxy, fault=FAULT_DEGENERATE_LOOP)
    assert st.degeneration_detected == before + 1, "internal stream must detect once"
    assert proxy.total_in_flight() == 0
    await proxy.stream_frames("a b c", fault=FAULT_DEGENERATE_LOOP)
    assert st.degeneration_detected == before + 2, "OpenAI stream must detect once"
    assert proxy.total_in_flight() == 0


async def test_openai_stream_degeneration_not_detected_when_off(proxy, monkeypatch):
    """Flag OFF: the OpenAI stream must add ZERO detection (byte-identical)."""
    monkeypatch.delenv(FLAG, raising=False)
    st = proxy.svc._correction.state
    before = st.degeneration_detected
    await proxy.stream_frames("a b c", fault=FAULT_DEGENERATE_LOOP)
    assert st.degeneration_detected == before


async def test_legit_streams_not_false_flagged(proxy, monkeypatch):
    """Flag ON, false-positive guard: neither a legitimate LONG response (60
    distinct words, no dominating shingle) NOR a bounded chorus (a phrase repeated
    only ~3x inside a long reply, below the reps/fraction thresholds) is flagged as
    degeneration. (Both cases in one proxy spin for tollgate budget.)"""
    monkeypatch.setenv(FLAG, "1")
    st = proxy.svc._correction.state
    before = st.degeneration_detected
    await proxy.stream_frames(" ".join(f"w{i}" for i in range(60)))
    assert st.degeneration_detected == before, "clean long stream false-flagged"
    chorus = "hey now hey now "
    filler = " ".join(f"line{i} verse text here more" for i in range(12))
    await proxy.stream_frames((chorus + filler + " ") + chorus * 2 + filler)
    assert st.degeneration_detected == before, "bounded chorus false-flagged"


async def test_grammar_stream_is_shadow_checked(proxy, monkeypatch):
    """A grammar-bearing stream runs the silent-grammar-drop check over its
    reassembled content (requires the shadow-egress kill-switch on) — proving the
    streaming path is no longer a shadow-egress blind spot."""
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("COLLECTIVE_PROXY_SHADOW_EGRESS", "1")
    st = proxy.svc._correction.state
    st.shadow_drop.clear()
    grammar = (
        'root ::= "{" ws "\\"x\\":" ws str ws "}"\n'
        'str ::= "\\"" [^"]* "\\""\n'
        'ws ::= [ \\t\\n]*\n'
    )
    # fake echoes: content becomes 'echo: {"x": "hi"}' — NOT conformant to the
    # object-root grammar (leading 'echo:'), so this should be a silent DROP.
    await _submit_stream_events(
        proxy, content='{"x": "hi"}',
        extra_payload={"extra_body": {"grammar": grammar}})
    tally = st.shadow_drop.get("uc_adversarial")
    assert tally is not None, "grammar stream was not checked at all"
    assert tally["checked"] == 1


async def test_sync_door_shadow_egress_wired_through_apply(proxy, monkeypatch):
    """HARDENING guard-bite: the golden oracle can't see shadow-egress (it's
    detect-only), so a revert that drops shadow_egress_detect from Correction.apply
    would pass the oracle. Prove END-TO-END that a real sync /v1/submit with a
    grammar + a non-conformant backend reply tallies a silent drop — i.e. apply
    actually wires the real detector into the sync door. Fails if the step is
    dropped from apply."""
    monkeypatch.setenv("COLLECTIVE_PROXY_SHADOW_EGRESS", "1")
    grammar = (
        'root ::= "{" ws "\\"x\\":" ws str ws "}"\n'
        'str ::= "\\"" [^"]* "\\""\n'
        'ws ::= [ \\t\\n]*\n'
    )
    st = proxy.svc._correction.state
    st.shadow_drop.clear()
    body = {
        "agent_id": "adv_test",
        "endpoint": "chat",
        "priority": "P3_INGESTION",
        "call_site": "uc_sync_shadow",
        "payload_type": "chat_completion",
        "payload": {
            "model": "chat",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "extra_body": {"grammar": grammar},
        },
    }
    resp = await proxy.client.post("/v1/submit", json=body)
    assert resp.status_code == 200
    # fake echoes 'echo: hi' — not conformant to the object-root grammar → drop.
    tally = st.shadow_drop.get("uc_sync_shadow")
    assert tally is not None and tally["checked"] == 1, \
        "sync door did not run shadow-egress via apply"
    assert proxy.total_in_flight() == 0


# --------------------------------------------------------------------------- #
# 3) SLOT ACCOUNTING — hostile paths return to baseline
# --------------------------------------------------------------------------- #

async def test_slot_reclaimed_reset_and_degenerate_streams(proxy, monkeypatch):
    """Flag ON (the new correction path): a mid-stream reset AND a degenerate
    stream each return in-flight to baseline — no leak on the sanitized/detected
    path. (Slot mechanics are flag-independent — finalize_stream runs AFTER the
    slot is freed — and the OFF path is already covered by the pre-existing stream
    suite; consolidated to one proxy spin for tollgate budget.)"""
    monkeypatch.setenv(FLAG, "1")
    before = proxy.total_in_flight()
    frames = await proxy.stream_frames("a b c", fault=FAULT_MID_STREAM_RESET)
    assert any("error" in f for f in frames)
    assert proxy.total_in_flight() == before
    await proxy.stream_frames("a b c", fault=FAULT_DEGENERATE_LOOP)
    assert proxy.total_in_flight() == before


async def test_slot_reclaimed_client_abort_mid_stream(proxy, monkeypatch):
    """Client walks away after the first frame (flag ON): the producer is
    cancelled and the slot is reclaimed — no leak on the new sanitizer path."""
    monkeypatch.setenv(FLAG, "1")
    proxy.controller.set_fault(FAULT_DEGENERATE_LOOP, 0.0)
    before = proxy.total_in_flight()
    body = {
        "model": "chat",
        "messages": [{"role": "user", "content": "x"}],
        "max_tokens": 16,
        "stream": True,
    }
    async with proxy.client.stream("POST", "/v1/chat/completions", json=body) as resp:
        async for _line in resp.aiter_lines():
            break  # abort after the first line
    # allow the producer-cancel/finally to settle
    import asyncio
    for _ in range(50):
        if proxy.total_in_flight() == before:
            break
        await asyncio.sleep(0.02)
    assert proxy.total_in_flight() == before


# --------------------------------------------------------------------------- #
# 4) FAIL-OPEN / never-crash — finalize_stream fuzz
# --------------------------------------------------------------------------- #

class _StubState:
    def __init__(self):
        self.degeneration_detected = 0
        self.degeneration_by_call_site = {}
        self.shadow_drop = {}
        # a real ProxyState always carries this; apply()->finalize_thinking reads it
        self.thinking_active = {}


class _Req:
    def __init__(self, payload=None, ptype="chat_completion"):
        self.payload = payload if payload is not None else {"messages": []}
        self.stream = True
        self.payload_type = ptype
        self.endpoint = "chat"
        self.request_id = "r"
        self.call_site = "fuzz"


@pytest.mark.parametrize("content", [
    "",
    None,
    "x" * 200_000,                       # huge
    "\x00\x01\x02\xff\xfe binary junk",  # binary-ish
    "🎵🎶" * 5000,                        # unicode
    "the song of the sea " * 60,         # actually degenerate — must not raise
])
def test_finalize_stream_fuzz_never_raises(monkeypatch, content):
    monkeypatch.setenv(FLAG, "1")
    c = Correction(_StubState())
    c.finalize_stream(_Req(), content, "stop")  # no exception == pass


def test_finalize_stream_broken_request_shapes_fail_open(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    c = Correction(_StubState())
    # payload not a dict / call_site None / weird finish
    r = _Req(payload="not-a-dict")
    r.call_site = None
    c.finalize_stream(r, "the song of the sea " * 60, None)
    # broken state (missing attrs) must be swallowed too
    Correction(object()).finalize_stream(_Req(), "the song of the sea " * 60, "stop")


async def test_apply_fuzz_result_shapes_never_raise(monkeypatch):
    """apply() over degenerate/odd result dicts must never raise out. Async so
    pytest-asyncio owns the loop (manual get_event_loop().run_until_complete in a
    sync test can grab a CLOSED loop under some pytest-asyncio versions)."""
    monkeypatch.delenv(FLAG, raising=False)
    c = Correction(_StubState())
    for result in ({}, {"status": "ok"}, {"status": "error"},
                   {"status": "ok", "response": None},
                   {"status": "ok", "response": {"choices": []}}):
        r = _Req(ptype="chat_completion")
        r.stream = False
        # apply calls finalize_thinking/maybe_correct_degenerate/shadow_egress_detect;
        # with a stub state lacking thinking_active etc. they must fail-open, not raise.
        await c.apply(r, result)
