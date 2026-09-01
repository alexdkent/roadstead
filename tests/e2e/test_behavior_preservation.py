"""Behavior-preservation golden-fixture harness for the LLM proxy.

This is the **correctness oracle** guarding the upcoming behavior-preserving
refactor (the de-monolith of ``service.py``). It captures the CURRENT proxy's
*observable, caller-visible I/O* for a tight corpus of (door x mode x behavior)
cells and freezes it as ``golden/behavior_baseline.json``. The refactor must keep
that I/O byte-identical (after normalization); this test replays the corpus and
asserts equality.

WHAT IS JUDGED — observable caller-visible I/O ONLY (never internal "which guard
fired"): for each case we capture

  * the HTTP ``status_code``;
  * the response BODY (sync) or the ordered list of SSE ``data:`` payloads
    (streaming) — frame ORDER is behavior and is preserved;

and, separately (not frozen — it's an invariant, not an output), we assert the
slot-accounting invariant ``total_in_flight() == 0`` settles after every case.

NORMALIZATION / REDACTION (empirically derived — captured twice + diffed the raw
output; only fields that actually vary run-to-run, plus the spec-mandated
timestamp/uuid fields, are redacted so the baseline is stable):

  * ``request_id``            (anywhere)                 -> ``"<rid>"``
      the proxy mints a fresh uuid-ish id per request (the ONLY field that
      differed in the raw diff of the sync submit envelope).
  * any key ending ``_ms``    (anywhere)                 -> ``0``
      queue_wait_ms / backend_latency_ms — wall-clock latency, varies every run.
  * any key ending ``_ss``    (anywhere)                 -> ``0``
      estimated_cost_ss — slot-seconds cost, latency-derived.
  * ``slot_seconds``          (anywhere)                 -> ``0``
  * ``created`` / ``timestamp`` (anywhere)               -> ``0``
      OpenAI envelopes carry a unix ``created``; a real backend varies it.
  * ``id`` **only inside an OpenAI envelope dict** (a dict whose ``object`` is
    ``chat.completion`` / ``chat.completion.chunk`` / ``list``) -> ``"<id>"``
      the chatcmpl / chunk envelope id. Scoped to the envelope so nested
      ``tool_calls[].id`` (real behavior a refactor must preserve) stays intact.

  All dict keys are then emitted with ``sort_keys=True`` so ordering never
  affects equality. SSE frames are normalized per-frame with the same rules and
  kept in order; the ``[DONE]`` sentinel and any non-JSON (partial) frame are
  preserved verbatim.

TWO MODES:
  * CAPTURE  (``LLMPROXY_GOLDEN_CAPTURE=1``): drive every declared case through a
    fresh proxy and (re)write ``golden/behavior_baseline.json`` (pretty, sorted).
    Prints the case count. Run this ONLY to regenerate the baseline on purpose.
  * ASSERT   (default): parametrized over the frozen baseline; drives each case
    live and asserts the normalized result EQUALS the baseline (readable diff on
    mismatch), plus the no-leak invariant.

Determinism + hermeticity: reuses the Phase-T e2e proxy construction (in-process
``build_app`` + a real ``FakeBackendServer`` socket, backends repointed at the
fake). No GPU, no network.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
import pytest

from roadstead.__main__ import build_app

# Reuse the exact Phase-T e2e construction so capture-mode proxies are identical
# to the ``proxy`` fixture used in assert-mode (no drift between the two paths).
from tests.e2e.conftest import ProxyHarness, _repointed_config
from roadstead.testing import FakeBackend, FakeBackendServer
from tests.corpus.schemas import _CHAT_CASE_D


_BASELINE_PATH = Path(__file__).parent / "golden" / "behavior_baseline.json"
_CAPTURE = os.environ.get("LLMPROXY_GOLDEN_CAPTURE", "").strip() in ("1", "true", "yes", "on")
_INTERNAL_CLIENT = ("127.0.0.1", 41999)


@pytest.fixture(autouse=True)
def _pin_schema_backstop_off(monkeypatch):
    """Pin the Phase-3 schema-backstop OFF for this golden-corpus file.

    The fake backend echoes plain "echo: ..." text, so a grammar/schema case is
    intentionally non-conformant. The production schema-backstop
    (``ROADSTEAD_PROXY_SCHEMA_BACKSTOP``, ON in the container) would 502 it —
    an artifact of the fake backend, not the transform this corpus pins. The
    backstop has its own suite (test_schema_backstop*). The golden baseline was
    captured with the flag OFF (its default), so pin it OFF to keep this file
    hermetic w.r.t. the ambient container env (it reads os.environ live). Fixes
    the container-only 502 in the chat_sync_grammar case after "chat"→classify.

    The always-on structured-validity floor (ROADSTEAD_PROXY_STRUCTURED_VALIDITY,
    2026-07-11) is pinned OFF for the same reason: the fake's non-JSON echo on the
    grammar case would 502 under the parse-only floor — an artifact of the fake
    backend, not the transform this corpus pins. The guard has its own suite
    (test_truncation_guard.py). Env is read live per request, so fixture ordering
    vs proxy construction doesn't matter.
    """
    monkeypatch.setenv("ROADSTEAD_PROXY_SCHEMA_BACKSTOP", "0")
    monkeypatch.setenv("ROADSTEAD_PROXY_STRUCTURED_VALIDITY", "0")

_OPENAI_ENVELOPE_OBJECTS = ("chat.completion", "chat.completion.chunk", "list")


# --------------------------------------------------------------------------- #
# Normalization / redaction (see module docstring for the empirical rationale)
# --------------------------------------------------------------------------- #

def _normalize(obj: Any) -> Any:
    """Recursively redact volatile fields so the baseline is run-to-run stable."""
    if isinstance(obj, dict):
        is_openai_env = str(obj.get("object", "")) in _OPENAI_ENVELOPE_OBJECTS
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k == "request_id":
                out[k] = "<rid>"
            elif k == "id" and is_openai_env:
                out[k] = "<id>"
            elif k in ("created", "timestamp", "slot_seconds"):
                out[k] = 0
            elif isinstance(k, str) and (k.endswith("_ms") or k.endswith("_ss")):
                out[k] = 0
            else:
                out[k] = _normalize(v)
        return out
    if isinstance(obj, list):
        return [_normalize(x) for x in obj]
    return obj


def _normalize_frame(frame: str) -> str:
    """Normalize one SSE ``data:`` payload; ``[DONE]`` and unparseable (partial)
    frames pass through verbatim. Order is preserved by the caller."""
    if frame == "[DONE]":
        return frame
    try:
        parsed = json.loads(frame)
    except (json.JSONDecodeError, ValueError):
        return frame  # partial / non-JSON frame — behavior, kept as-is
    return json.dumps(_normalize(parsed), sort_keys=True)


def _normalize_body(text: str) -> Any:
    """Normalize a sync response body. Non-JSON bodies are kept as raw text."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {"_raw_text": text}
    return _normalize(parsed)


# --------------------------------------------------------------------------- #
# Corpus — one case per (door x mode x behavior) cell a refactor could regress.
# --------------------------------------------------------------------------- #

@dataclass
class GoldenCase:
    cid: str
    door: str                      # "chat" | "submit" | "embeddings"
    mode: str = "sync"             # "sync" | "stream"  (stream = chat only)
    fault: Optional[str] = None
    fault_arg: float = 0.0
    fault_max_hits: int = 0
    model: str = "chat"
    content: str = "hello world"
    timeout_s: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)   # merged into chat body
    submit_endpoint: str = "chat"
    embed_input: Any = field(default_factory=lambda: ["alpha", "beta"])


# Grammar-bearing routing sub-call (reuse the captured corpus fixture) — exercises
# the proxy's grammar-relocation / vLLM-normalization path. The fake echoes, so
# the observable body is a clean echo; the value is that the grammar path ran.
_GRAMMAR_EXTRA = {"extra_body": _CHAT_CASE_D["extra_body"]}


CASES: List[GoldenCase] = [
    # ---- OpenAI /v1/chat/completions — the byte-identical anchors + faults ----
    GoldenCase("chat_sync_happy", "chat", "sync"),
    GoldenCase("chat_sync_vllm_happy", "chat", "sync", model="tier3",
               content="vllm path"),
    GoldenCase("chat_stream_happy", "chat", "stream", content="one two three"),
    GoldenCase("chat_sync_grammar", "chat", "sync", model="tier1",
               content="play something mellow for the evening", extra=_GRAMMAR_EXTRA),
    GoldenCase("chat_sync_empty_completion", "chat", "sync",
               fault="empty_completion", fault_max_hits=1),
    GoldenCase("chat_sync_finish_length", "chat", "sync", fault="finish_length"),
    GoldenCase("chat_sync_degenerate_loop", "chat", "sync", fault="degenerate_loop"),
    GoldenCase("chat_sync_phantom_tool_calls", "chat", "sync",
               fault="phantom_tool_calls"),
    GoldenCase("chat_sync_http_500", "chat", "sync", fault="http_500"),
    GoldenCase("chat_sync_timeout", "chat", "sync", fault="timeout", fault_arg=2.0,
               timeout_s=0.5),
    GoldenCase("chat_stream_truncated_tool_calls", "chat", "stream",
               fault="truncated_tool_calls", content="call a tool"),
    # ---- the enriched Roadstead envelope (/rs/v1/chat) ----
    # Replaces the two `/v1/submit` cases: that door was removed with Workstream
    # C, and these freeze its successor's envelope — attribution and timing
    # included, because a disclosure that silently stops being emitted is the
    # failure the whole API exists to prevent.
    GoldenCase("rs_sync_happy", "rs", "sync", content="internal hi"),
    GoldenCase("rs_sync_http_500", "rs", "sync", fault="http_500",
               content="internal boom"),
    GoldenCase("rs_sync_by_intent", "rs", "sync", content="by intent",
               extra={"intent": "chat"}),
    GoldenCase("rs_stream_happy", "rs", "stream", content="internal stream"),
    # ---- OpenAI /v1/embeddings ----
    GoldenCase("embeddings_happy", "embeddings", "sync"),
    GoldenCase("embeddings_error", "embeddings", "sync", fault="http_500"),
]

CASES_BY_ID: Dict[str, GoldenCase] = {c.cid: c for c in CASES}
assert len(CASES_BY_ID) == len(CASES), "duplicate case id in CASES"


# --------------------------------------------------------------------------- #
# Proxy construction (mirrors tests/llmproxy/e2e/conftest.py::proxy exactly) so
# capture-mode and the determinism test spin identical services.
# --------------------------------------------------------------------------- #

@contextlib.asynccontextmanager
async def _spawn_proxy() -> AsyncIterator[ProxyHarness]:
    srv = FakeBackendServer(FakeBackend()).start()
    tmp = tempfile.TemporaryDirectory()
    try:
        config = _repointed_config(srv.host, srv.port, f"{tmp.name}/queue.db")
        app = build_app(config)
        svc = app.state.proxy_service

        async def _healthy(ep_cfg):  # circuit never trips on probe timing
            return True

        svc._backend.probe_health = _healthy
        await svc.startup()
        transport = httpx.ASGITransport(
            app=app, raise_app_exceptions=False, client=_INTERNAL_CLIENT)
        client = httpx.AsyncClient(transport=transport, base_url="http://proxy",
                                   timeout=30.0)
        try:
            yield ProxyHarness(app, svc, client, srv)
        finally:
            await client.aclose()
            await svc.shutdown()
    finally:
        srv.stop()
        tmp.cleanup()


# --------------------------------------------------------------------------- #
# Drive one case → normalized observable record.
# --------------------------------------------------------------------------- #

def _chat_body(case: GoldenCase, *, stream: bool) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": case.model,
        "messages": [{"role": "user", "content": case.content}],
        "max_tokens": 16,
        "stream": stream,
    }
    if case.timeout_s is not None:
        body["timeout_s"] = case.timeout_s
    if case.extra:
        body.update(case.extra)
    return body


def _rs_body(case: GoldenCase, *, stream: bool = False) -> Dict[str, Any]:
    """The enriched envelope. Routing declaration outside, model payload inside —
    the split that lets `deadline_s` exist without being forwarded to a backend
    that would reject the unknown field."""
    body: Dict[str, Any] = {
        "model": case.submit_endpoint,
        "priority": "P3_INGESTION",
        "call_site": "behavior_preservation",
        "payload": {
            "model": case.model,
            "messages": [{"role": "user", "content": case.content}],
            "max_tokens": 16,
            "stream": stream,
        },
    }
    if case.extra:
        # An `intent` case declares no pin; the extra replaces the pin rather
        # than sitting beside it, so the resolver is genuinely exercised.
        if "intent" in case.extra:
            body.pop("model", None)
        body.update(case.extra)
    return body


async def _drive(harness: ProxyHarness, case: GoldenCase) -> Dict[str, Any]:
    """Fire the case at the proxy front door; return the normalized record."""
    if case.fault:
        harness.controller.set_fault(case.fault, case.fault_arg, case.fault_max_hits)

    if case.door == "rs" and case.mode == "stream":
        frames: List[str] = []
        async with harness.client.stream(
                "POST", "/rs/v1/chat", json=_rs_body(case, stream=True)) as resp:
            status = resp.status_code
            async for line in resp.aiter_lines():
                line = line.strip()
                if line.startswith("data: "):
                    frames.append(line[len("data: "):])
        return {"status_code": status, "mode": "stream",
                "frames": [_normalize_frame(f) for f in frames]}

    if case.door == "chat" and case.mode == "stream":
        frames: List[str] = []
        status = None
        async with harness.client.stream(
                "POST", "/v1/chat/completions", json=_chat_body(case, stream=True)) as resp:
            status = resp.status_code
            async for line in resp.aiter_lines():
                line = line.strip()
                if line.startswith("data: "):
                    frames.append(line[len("data: "):])
        return {"status_code": status, "mode": "stream",
                "frames": [_normalize_frame(f) for f in frames]}

    if case.door == "chat":
        resp = await harness.client.post(
            "/v1/chat/completions", json=_chat_body(case, stream=False))
    elif case.door == "rs":
        resp = await harness.client.post("/rs/v1/chat", json=_rs_body(case))
    elif case.door == "embeddings":
        resp = await harness.client.post(
            "/v1/embeddings", json={"model": "bge-m3", "input": case.embed_input})
    else:  # pragma: no cover - guarded by construction
        raise ValueError(f"unknown door {case.door!r}")

    return {"status_code": resp.status_code, "mode": "sync",
            "body": _normalize_body(resp.text)}


async def _settled_no_leak(harness: ProxyHarness, timeout: float = 2.0) -> int:
    """Poll until in-flight returns to 0 (streaming/timeout paths settle a beat
    after the response). Returns the final in-flight count."""
    import asyncio
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if harness.total_in_flight() == 0:
            return 0
        await asyncio.sleep(0.02)
    return harness.total_in_flight()


# --------------------------------------------------------------------------- #
# CAPTURE mode — regenerate the frozen baseline.
# --------------------------------------------------------------------------- #

async def _capture_all() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for case in CASES:
        async with _spawn_proxy() as harness:
            record = await _drive(harness, case)
            leak = await _settled_no_leak(harness)
            assert leak == 0, f"{case.cid}: slot leak, in-flight={leak}"
            out[case.cid] = record
    return out


@pytest.mark.skipif(not _CAPTURE, reason="capture mode: set LLMPROXY_GOLDEN_CAPTURE=1")
async def test_capture_baseline():
    """Regenerate golden/behavior_baseline.json from the live proxy."""
    records = await _capture_all()
    _BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _BASELINE_PATH.write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n[golden] captured {len(records)} cases -> {_BASELINE_PATH}")
    assert set(records) == set(CASES_BY_ID), "captured set != declared corpus"


# --------------------------------------------------------------------------- #
# ASSERT mode (default) — replay each frozen case and require equality.
# --------------------------------------------------------------------------- #

def _load_baseline() -> Dict[str, Any]:
    if not _BASELINE_PATH.exists():
        raise AssertionError(
            f"baseline missing: {_BASELINE_PATH} — run with "
            f"LLMPROXY_GOLDEN_CAPTURE=1 to generate it")
    return json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))


def _first_diff(expected: Any, actual: Any, path: str = "") -> Optional[str]:
    """Return a readable description of the first differing field, or None."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        for k in expected:
            if k not in actual:
                return f"{path}.{k}: missing in live output"
            d = _first_diff(expected[k], actual[k], f"{path}.{k}")
            if d:
                return d
        for k in actual:
            if k not in expected:
                return f"{path}.{k}: unexpected in live output"
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return f"{path}: length {len(expected)} (baseline) != {len(actual)} (live)"
        for i, (e, a) in enumerate(zip(expected, actual)):
            d = _first_diff(e, a, f"{path}[{i}]")
            if d:
                return d
        return None
    if expected != actual:
        return f"{path}: baseline={expected!r} != live={actual!r}"
    return None


_BASELINE = None if _CAPTURE else _load_baseline()
_ASSERT_IDS = [] if _CAPTURE else sorted(_BASELINE.keys())


@pytest.mark.skipif(_CAPTURE, reason="capture mode active; assert replay skipped")
@pytest.mark.parametrize("cid", _ASSERT_IDS)
async def test_behavior_preserved(proxy, cid):
    """Live result for one frozen case must equal the baseline (normalized)."""
    assert cid in CASES_BY_ID, f"baseline case {cid!r} is not a declared corpus case"
    case = CASES_BY_ID[cid]
    expected = _BASELINE[cid]

    live = await _drive(proxy, case)
    leak = await _settled_no_leak(proxy)
    assert leak == 0, f"{cid}: slot leak, in-flight={leak}"

    diff = _first_diff(expected, live)
    assert diff is None, (
        f"behavior regression for case {cid!r}: {diff}\n"
        f"  baseline={json.dumps(expected, sort_keys=True)[:400]}\n"
        f"  live    ={json.dumps(live, sort_keys=True)[:400]}")


# --------------------------------------------------------------------------- #
# ORACLE SELF-GUARD — the baseline can't masquerade as "everything preserved".
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(_CAPTURE, reason="capture mode active")
def test_baseline_covers_every_declared_case():
    """Non-empty baseline that has exactly one entry per declared corpus case —
    so a partial/empty/stale baseline is caught, not silently trusted."""
    baseline = _load_baseline()
    assert baseline, "baseline is empty"
    declared = set(CASES_BY_ID)
    frozen = set(baseline)
    assert frozen == declared, (
        f"baseline drift: missing={sorted(declared - frozen)} "
        f"extra={sorted(frozen - declared)} — regenerate with "
        f"LLMPROXY_GOLDEN_CAPTURE=1")


@pytest.mark.skipif(_CAPTURE, reason="capture mode active")
async def test_capture_is_deterministic():
    """Capturing a representative case twice yields identical normalized output —
    proves no volatile field leaks past the redaction (the baseline is stable)."""
    # The enriched envelope, which now carries the most volatile fields: the
    # four blocks add timing, slot-seconds and a computed cost on top of what
    # the old submit envelope had, so if any of them leaks past the redaction
    # this is where it shows.
    case = CASES_BY_ID["rs_sync_happy"]
    async with _spawn_proxy() as h1:
        first = await _drive(h1, case)
        assert await _settled_no_leak(h1) == 0
    async with _spawn_proxy() as h2:
        second = await _drive(h2, case)
        assert await _settled_no_leak(h2) == 0
    assert first == second, (
        f"non-deterministic capture — volatile field leaked past redaction:\n"
        f"  first ={json.dumps(first, sort_keys=True)}\n"
        f"  second={json.dumps(second, sort_keys=True)}")
