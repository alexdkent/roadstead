"""Meta-tests: prove the fake backend actually PRODUCES each south-face fault.

A green adversarial suite is only meaningful if the fault it claims to inject was
really emitted — otherwise "the proxy handled it" could just mean "the fault
never fired." These tests hit the fake backend **directly** (no proxy) over a
real socket, driving each fault via the ``X-Fault`` header, and assert the
pathological wire output is present.

This is the guardrail on the guardrail. Every entry in ``ALL_FAULTS`` must be
observably reproduced here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator, List

import httpx
import pytest

from roadstead.testing import (
    ALL_FAULTS,
    FakeBackend,
    FakeBackendServer,
    MidStreamReset,
)
from roadstead.testing import fake_backend as fb


@pytest.fixture
def server() -> Iterator[FakeBackendServer]:
    srv = FakeBackendServer(FakeBackend()).start()
    try:
        yield srv
    finally:
        srv.stop()


def _chat(srv: FakeBackendServer, fault: str, arg: float = 0.0,
          content: str = "hello world one two") -> httpx.Response:
    headers = {"X-Fault": fault}
    if arg:
        headers["X-Fault-Arg"] = str(arg)
    with httpx.Client(base_url=srv.url, timeout=10.0) as c:
        return c.post("/v1/chat/completions",
                      json={"messages": [{"role": "user", "content": content}]},
                      headers=headers)


def _stream_lines(srv: FakeBackendServer, fault: str, arg: float = 0.0,
                  content: str = "alpha beta gamma delta") -> List[str]:
    headers = {"X-Fault": fault}
    if arg:
        headers["X-Fault-Arg"] = str(arg)
    lines: List[str] = []
    with httpx.Client(base_url=srv.url, timeout=10.0) as c:
        with c.stream("POST", "/v1/chat/completions",
                      json={"stream": True,
                            "messages": [{"role": "user", "content": content}]},
                      headers=headers) as resp:
            for ln in resp.iter_lines():
                lines.append(ln)
    return lines


# --------------------------------------------------------------------------- #
# Coverage guard: every catalogued fault must have a meta-assertion here.
# --------------------------------------------------------------------------- #

def test_every_fault_is_meta_covered():
    covered = set(_COVERED)
    missing = set(ALL_FAULTS) - covered
    assert not missing, f"faults with no meta-test: {sorted(missing)}"


def test_every_fault_is_driven_through_the_proxy():
    """🚨 The second half, and the one this file could not see.

    Everything above proves a fault is EMITTED. That is necessary and not
    sufficient: a fault nothing drives through Roadstead is a pathology the
    fake can produce and the proxy has never been asked to survive, and it
    looks identical to a covered one from in here — `ALL_FAULTS` is full, every
    meta-test is green, and the fault library reads as an asset that is
    partly furniture.

    It found `slow_drain` on its first run: emitted, meta-asserted, and driven
    through the proxy by nothing at all. It was the false-positive control for
    the stall watchdogs — the case that tells "slow" from "stopped" — and its
    absence meant nothing checked that a healthy slow stream survives.

    🚨 This file is EXCLUDED from the scan, and that exclusion is the whole
    guard. Every fault appears here by construction, so counting these files
    makes the assertion vacuous in exactly the way it is meant to detect.

    Read by AST rather than by grep: a fault is referenced as `fb.FAULT_X`, as a
    bare `FAULT_X`, or as its own string value, and a substring sweep over the
    values alone matches prose — `"timeout"` and `"none"` are ordinary words.
    """
    import ast

    root = Path(__file__).resolve().parents[1]
    values = {getattr(fb, n): n for n in dir(fb)
              if n.startswith("FAULT_") and isinstance(getattr(fb, n), str)}
    # The files whose job is to enumerate the library rather than to use it.
    excluded = {Path(__file__).resolve(),
                root / "tests" / "test_testing_module_is_public.py"}

    driven: set[str] = set()
    for path in sorted(root.glob("tests/**/*.py")):
        if path.resolve() in excluded:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:            # not ours to police
            continue
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                name = values.get(node.value)
            if name and name.startswith("FAULT_") and hasattr(fb, name):
                driven.add(getattr(fb, name))

    undriven = sorted(set(ALL_FAULTS) - driven - set(_NOT_DRIVEN))
    assert not undriven, (
        "these faults are emitted by the fake and meta-asserted here, but no "
        "test drives them through the proxy — so nothing knows how Roadstead "
        f"behaves when they happen: {undriven}. Write that test, or add the "
        "fault to _NOT_DRIVEN with the reason it needs none.")

    # A `_NOT_DRIVEN` entry that has since grown a test is a stale excuse, and
    # excuses are how an allowlist stops meaning anything.
    stale = sorted(set(_NOT_DRIVEN) & driven)
    assert not stale, (
        f"_NOT_DRIVEN still excuses {stale}, which a test now drives — drop the "
        "entry")


#: Faults that deliberately need no proxy-level test, each with the reason the
#: absence is CORRECT rather than merely current. 🚨 Empty as of 2026-09-01, and
#: that is the point: it is here so a future exemption has to be argued in
#: writing rather than taken by leaving a fault out of the suite.
_NOT_DRIVEN: dict[str, str] = {}


# --------------------------------------------------------------------------- #
# Non-streaming faults
# --------------------------------------------------------------------------- #

def test_meta_none_happy(server):
    r = _chat(server, fb.FAULT_NONE)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"].startswith("echo:")


@pytest.mark.parametrize("fault,code", [
    (fb.FAULT_HTTP_400, 400), (fb.FAULT_HTTP_500, 500), (fb.FAULT_HTTP_503, 503),
])
def test_meta_http_status(server, fault, code):
    assert _chat(server, fault).status_code == code


def test_meta_truncated_json(server):
    r = _chat(server, fb.FAULT_TRUNCATED_JSON)
    assert r.status_code == 200
    with pytest.raises(json.JSONDecodeError):
        json.loads(r.text)


def test_meta_empty_completion(server):
    r = _chat(server, fb.FAULT_EMPTY_COMPLETION)
    assert r.status_code == 200
    msg = r.json()["choices"][0]["message"]
    assert (msg.get("content") or "") == ""
    assert not msg.get("tool_calls")


def test_meta_no_usage(server):
    r = _chat(server, fb.FAULT_NO_USAGE)
    assert r.status_code == 200
    assert "usage" not in r.json()


def test_meta_finish_length(server):
    r = _chat(server, fb.FAULT_FINISH_LENGTH)
    assert r.json()["choices"][0]["finish_reason"] == "length"


def test_meta_degenerate_loop(server):
    # must trip the proxy's detector: 6-word shingle repeating >= 6x over >= 40 words
    from roadstead.service import _is_degenerate_text
    text = _chat(server, fb.FAULT_DEGENERATE_LOOP).json()["choices"][0]["message"]["content"]
    assert _is_degenerate_text(text), "degenerate fault did not trip the real detector"


def test_meta_schema_valid_wrong(server):
    content = _chat(server, fb.FAULT_SCHEMA_VALID_WRONG).json()["choices"][0]["message"]["content"]
    obj = json.loads(content)  # valid JSON …
    assert "unexpected" in obj  # … but the wrong shape


def test_meta_schema_invalid(server):
    content = _chat(server, fb.FAULT_SCHEMA_INVALID).json()["choices"][0]["message"]["content"]
    # parseable JSON head + forbidden trailing prose → not a clean JSON document
    with pytest.raises(json.JSONDecodeError):
        json.loads(content)
    assert content.strip().startswith("{")


def test_meta_phantom_tool_calls(server):
    msg = _chat(server, fb.FAULT_PHANTOM_TOOL_CALLS).json()["choices"][0]["message"]
    tcs = msg.get("tool_calls")
    assert tcs, "expected tool_calls present"
    args = tcs[0]["function"]["arguments"]
    with pytest.raises(json.JSONDecodeError):
        json.loads(args)  # malformed / truncated arguments


def test_meta_timeout_delays(server):
    t0 = time.monotonic()
    _chat(server, fb.FAULT_TIMEOUT, arg=0.2)
    assert time.monotonic() - t0 >= 0.15


def test_meta_capacity_desync(server):
    # accept_limit defaults to 1; 3 concurrent → at least one 503.
    server.controller.set_fault(fb.FAULT_CAPACITY_DESYNC, arg=0.3)
    server.controller.accept_limit = 1
    import concurrent.futures

    def one():
        with httpx.Client(base_url=server.url, timeout=10.0) as c:
            return c.post("/v1/chat/completions",
                          json={"messages": [{"role": "user", "content": "x"}]}).status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        codes = list(ex.map(lambda _: one(), range(3)))
    assert 503 in codes, f"capacity desync produced no 503: {codes}"


# --------------------------------------------------------------------------- #
# Streaming faults
# --------------------------------------------------------------------------- #

def test_meta_ttft_stall(server):
    t0 = time.monotonic()
    lines = _stream_lines(server, fb.FAULT_TTFT_STALL, arg=0.2)
    # first data frame arrives only after the stall
    assert time.monotonic() - t0 >= 0.15
    assert any(ln.startswith("data: ") for ln in lines)


def test_meta_intertoken_stall(server):
    t0 = time.monotonic()
    _stream_lines(server, fb.FAULT_INTERTOKEN_STALL, arg=0.2)
    assert time.monotonic() - t0 >= 0.15


def test_meta_mid_stream_reset(server):
    # a real reset surfaces to httpx as a RemoteProtocolError (or a stream that
    # ends without [DONE]); either way the body is NOT completed normally.
    try:
        lines = _stream_lines(server, fb.FAULT_MID_STREAM_RESET)
    except httpx.RemoteProtocolError:
        return  # reset observed
    assert "data: [DONE]" not in lines, "reset stream should not complete with [DONE]"


def test_meta_partial_sse(server):
    lines = _stream_lines(server, fb.FAULT_PARTIAL_SSE)
    # at least one data frame is not parseable JSON (the injected partial)
    bad = 0
    for ln in lines:
        if ln.startswith("data: ") and ln[6:] != "[DONE]":
            try:
                json.loads(ln[6:])
            except json.JSONDecodeError:
                bad += 1
    assert bad >= 1, "expected a partial/unparseable SSE frame"


def test_meta_interleaved_sse(server):
    lines = _stream_lines(server, fb.FAULT_INTERLEAVED_SSE)
    assert any(ln.startswith(":") for ln in lines), "expected an SSE comment/keepalive line"
    assert "data: [DONE]" in lines


def test_meta_no_done(server):
    lines = _stream_lines(server, fb.FAULT_NO_DONE)
    assert "data: [DONE]" not in lines


def test_meta_slow_drain(server):
    t0 = time.monotonic()
    lines = _stream_lines(server, fb.FAULT_SLOW_DRAIN, arg=0.05)
    assert "data: [DONE]" in lines  # valid, just slow
    assert time.monotonic() - t0 >= 0.1


def test_meta_truncated_tool_calls(server):
    lines = _stream_lines(server, fb.FAULT_TRUNCATED_TOOL_CALLS)
    joined = "\n".join(lines)
    assert "tool_calls" in joined
    # the streamed argument JSON is cut off mid-object
    assert '"arguments": "{\\"x\\":"' in joined or '{"x":' in joined


# The set of faults each test above proves is emitted — kept in sync with
# ALL_FAULTS by test_every_fault_is_meta_covered.
_COVERED = (
    fb.FAULT_NONE,
    fb.FAULT_HTTP_400, fb.FAULT_HTTP_500, fb.FAULT_HTTP_503,
    fb.FAULT_TRUNCATED_JSON, fb.FAULT_EMPTY_COMPLETION, fb.FAULT_NO_USAGE,
    fb.FAULT_FINISH_LENGTH, fb.FAULT_DEGENERATE_LOOP,
    fb.FAULT_SCHEMA_VALID_WRONG, fb.FAULT_SCHEMA_INVALID, fb.FAULT_PHANTOM_TOOL_CALLS,
    fb.FAULT_TTFT_STALL, fb.FAULT_INTERTOKEN_STALL, fb.FAULT_MID_STREAM_RESET,
    fb.FAULT_PARTIAL_SSE, fb.FAULT_INTERLEAVED_SSE, fb.FAULT_NO_DONE, fb.FAULT_SLOW_DRAIN,
    fb.FAULT_TRUNCATED_TOOL_CALLS,
    fb.FAULT_TIMEOUT, fb.FAULT_CAPACITY_DESYNC,
)
