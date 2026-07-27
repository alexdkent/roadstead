"""Phase 2a — per-caller prefix-cache attribution: INDEPENDENT ADVERSARIAL track.

The builder's happy-path + parity coverage lives in ``test_cache_attribution.py``.
THIS file attacks both seams and asserts the contract invariants hold under
hostile input. Written by the independent validation track — it trusts nothing.

Seams under attack:
  * SOUTH FACE — pathological backend ``usage`` shapes the proxy parses
    (``extract_cached_tokens`` + the sync/stream capture paths): huge/negative/
    float/bool/str/list/dict/NaN/Infinity ``cached_tokens``; ``prompt_tokens_details``
    as a non-dict/null/list; absent ``usage``; absent/zero ``prompt_tokens``.
    Invariant: the parser NEVER raises, the proxy NEVER hard-500s or leaks a slot,
    attribution degrades sanely.
  * NORTH FACE — hostile CALLER inputs can't corrupt attribution or crash capture.

Rollup invariants (tested at the DB layer, deterministically, no proxy spin):
  (a) NULL rows (llama.cpp / rejected-malformed) NEVER drag the hit rate toward 0;
  (b) division-by-zero guard when attributable_input_tokens == 0 → hit_rate None;
  (c) a real cold-0 (``cached_tokens=0``) IS in the ratio — distinct from NULL;
  (d) ``limit`` bounds the by_call_site list.

Lean-suite discipline (the llmproxy tollgate is near its 120s budget): the bulk of
the hostile matrix runs on the pure parser (no I/O) and on a bare ``PersistentQueue``
(sync writes, no async/proxy). Only a HANDFUL of tests spin the real proxy — each
drives several hostile values in one instance.
"""
from __future__ import annotations

import math
import tempfile

import pytest

# Exhaustive ProxyService-spinning adversarial matrix — deselected from the
# per-ship in_container_tollgate via `-m 'not heavy'` (see pyproject `heavy`).
pytestmark = pytest.mark.heavy

from originfleet.llmproxy.backend import extract_cached_tokens
from originfleet.llmproxy.queue import PersistentQueue
from originfleet.llmproxy.timeout_model import normalize_endpoint

from tests.llmproxy.fake_backend import _OMIT_USAGE, _UNSET  # noqa: F401  (imported for parity/clarity)


# --------------------------------------------------------------------------- #
# SOUTH FACE (unit) — the pure parser must NEVER raise on any hostile shape.
# --------------------------------------------------------------------------- #

# Shapes the parser handles TODAY (returns int-or-None, no raise). This set is
# also the GUARD-BITE target: strip the robustness from extract_cached_tokens
# (e.g. `return usage["prompt_tokens_details"]["cached_tokens"]`) and several of
# these raise KeyError/TypeError → the test goes red.
_SAFE_HOSTILE = [
    # cached_tokens value pathologies (inside a well-formed details dict)
    ({"prompt_tokens": 12, "prompt_tokens_details": {"cached_tokens": 10**9}}, 10**9),   # > prompt
    ({"prompt_tokens": 12, "prompt_tokens_details": {"cached_tokens": 2**60}}, 2**60),   # > 2**53
    ({"prompt_tokens": 12, "prompt_tokens_details": {"cached_tokens": -3}}, None),        # negative
    ({"prompt_tokens": 12, "prompt_tokens_details": {"cached_tokens": 3.9}}, 3),          # float → trunc
    ({"prompt_tokens": 12, "prompt_tokens_details": {"cached_tokens": True}}, None),       # bool
    ({"prompt_tokens": 12, "prompt_tokens_details": {"cached_tokens": False}}, None),      # bool
    ({"prompt_tokens": 12, "prompt_tokens_details": {"cached_tokens": "5"}}, None),        # str
    ({"prompt_tokens": 12, "prompt_tokens_details": {"cached_tokens": [1, 2]}}, None),     # list
    ({"prompt_tokens": 12, "prompt_tokens_details": {"cached_tokens": {"a": 1}}}, None),   # dict
    ({"prompt_tokens": 12, "prompt_tokens_details": {"cached_tokens": None}}, None),       # explicit null
    # prompt_tokens_details shape pathologies
    ({"prompt_tokens": 12, "prompt_tokens_details": [1, 2, 3]}, None),                     # list, not dict
    ({"prompt_tokens": 12, "prompt_tokens_details": None}, None),                          # null
    ({"prompt_tokens": 12, "prompt_tokens_details": "nope"}, None),                        # str
    ({"prompt_tokens": 12, "prompt_tokens_details": 7}, None),                             # int
    ({"prompt_tokens": 12, "prompt_tokens_details": {}}, None),                            # empty dict
    # top-level fallback + its pathologies
    ({"prompt_tokens": 12, "cached_tokens": 8}, 8),
    ({"prompt_tokens": 12, "cached_tokens": "8"}, None),
    ({"prompt_tokens": 12, "cached_tokens": [8]}, None),
    # usage-block shape pathologies
    ({}, None),
    ({"prompt_tokens": 12}, None),   # llama.cpp: absent entirely
    (None, None),
    ([1, 2, 3], None),
    ("notadict", None),
    (42, None),
]

# NaN / Infinity: JSON accepts them by default (both httpx .json() and our SSE
# json.loads), so they reach the parser over the wire; `int(float('nan'))` and
# `int(float('inf'))` RAISE (ValueError / OverflowError). This WAS a confirmed
# defect (an observability field turning a good completion into a caller-facing
# error); FIXED by the `math.isfinite()` guard in extract_cached_tokens — a
# non-finite cached_tokens now degrades to None (n/a), never raises. This test
# is the guard-bite regression: revert the guard and it fails.
_NAN_INF = [
    ({"prompt_tokens": 12, "prompt_tokens_details": {"cached_tokens": float("nan")}}, None),
    ({"prompt_tokens": 12, "prompt_tokens_details": {"cached_tokens": float("inf")}}, None),
    ({"prompt_tokens": 12, "prompt_tokens_details": {"cached_tokens": float("-inf")}}, None),
    ({"prompt_tokens": 12, "cached_tokens": float("nan")}, None),
]


@pytest.mark.parametrize("usage,expected", _SAFE_HOSTILE)
def test_extract_never_raises_on_hostile_shape(usage, expected):
    # Must not raise, and must return an int-or-None (never a float/str/etc.).
    result = extract_cached_tokens(usage)
    assert result == expected
    assert result is None or (isinstance(result, int) and not isinstance(result, bool))


@pytest.mark.parametrize("usage,expected", _NAN_INF)
def test_extract_never_raises_on_nan_inf(usage, expected):
    # Guarded by math.isfinite() — a non-finite cached_tokens degrades to None,
    # never raises. Revert that guard and this bites (guard-bite regression).
    result = extract_cached_tokens(usage)
    assert result == expected


# --------------------------------------------------------------------------- #
# ROLLUP INVARIANTS (DB layer) — deterministic, no proxy spin.
# --------------------------------------------------------------------------- #

@pytest.fixture
def qdb():
    """A bare PersistentQueue on a temp file. No async writer started → writes
    are synchronous (the legitimate pre-writer window), so persist_complete +
    cache_attribution run inline with no flushing/race."""
    with tempfile.TemporaryDirectory() as tmp:
        q = PersistentQueue(f"{tmp}/queue.db")
        try:
            yield q
        finally:
            q.close()


def _chat_row(q, rid, *, call_site="cs", endpoint="chat", cached=None,
              in_tok=12, kind="chat", status="ok"):
    q.persist_complete(
        rid, "agent", endpoint, call_site, 0,
        in_tok, 4, 0.1, 0.0, status, kind=kind, cached_tokens=cached)


def test_null_rows_never_drag_hit_rate(qdb):
    """THE property: a window mixing an attributed vLLM row (6/12) with many
    NULL (llama.cpp / rejected-malformed) rows yields fleet hit_rate 0.5 — the
    NULL rows are counted unattributed and EXCLUDED from the ratio, never
    dragging it toward 0 (the ~4.8% attribution artifact this whole feature
    fixes)."""
    _chat_row(qdb, "a1", cached=6, in_tok=12)          # attributed vLLM
    for i in range(6):                                  # NULL llama.cpp rows
        _chat_row(qdb, f"n{i}", cached=None, in_tok=12)
    attr = qdb.cache_attribution(window_s=3600)
    fleet = attr["fleet"]
    assert fleet["hit_rate"] == 0.5, "NULL rows must not enter the ratio"
    assert fleet["attributed_calls"] == 1
    assert fleet["unattributed_calls"] == 6
    assert fleet["attributable_input_tokens"] == 12   # only the attributed row's input


def test_cold_zero_is_in_ratio_but_null_is_not(qdb):
    """A real cold miss (cached_tokens=0) is ATTRIBUTED and enters the ratio;
    a NULL is not. 6/12 + 0/12 → 6/24 = 0.25; the NULL row changes nothing.
    This is the NULL-vs-0 distinction the persistence layer exists to preserve."""
    _chat_row(qdb, "hit", cached=6, in_tok=12)
    _chat_row(qdb, "cold", cached=0, in_tok=12)
    _chat_row(qdb, "null", cached=None, in_tok=12)
    fleet = qdb.cache_attribution(window_s=3600)["fleet"]
    assert fleet["attributed_calls"] == 2      # hit + cold (0 counts), NOT null
    assert fleet["unattributed_calls"] == 1
    assert fleet["cached_tokens"] == 6
    assert fleet["attributable_input_tokens"] == 24
    assert fleet["hit_rate"] == 0.25


def test_division_by_zero_guard(qdb):
    """attributable_input_tokens == 0 must yield hit_rate None, NOT a
    ZeroDivisionError — even when cached_tokens rows are present."""
    _chat_row(qdb, "z1", cached=0, in_tok=0)     # attributed, zero input
    _chat_row(qdb, "z2", cached=5, in_tok=0)     # attributed, zero input
    _chat_row(qdb, "z3", cached=3, in_tok=None)  # attributed, NULL input (COALESCE→0)
    fleet = qdb.cache_attribution(window_s=3600)["fleet"]
    assert fleet["attributed_calls"] == 3
    assert fleet["attributable_input_tokens"] == 0
    assert fleet["hit_rate"] is None


def test_all_null_endpoint_is_na_not_zero(qdb):
    """An endpoint that only ever saw NULL rows reads n/a, never 0%."""
    for i in range(4):
        _chat_row(qdb, f"l{i}", endpoint="companion", cached=None)
    ep = next(e for e in qdb.cache_attribution()["by_endpoint"]
              if e["endpoint"] == "companion")
    assert ep["hit_rate"] is None
    assert ep["attributed_calls"] == 0
    assert ep["unattributed_calls"] == 4


def test_limit_bounds_by_call_site(qdb):
    """limit caps the by_call_site list length (by_endpoint/fleet are unbounded)."""
    for i in range(5):
        _chat_row(qdb, f"c{i}", call_site=f"site{i}", cached=6, in_tok=12)
    attr = qdb.cache_attribution(window_s=3600, limit=2)
    assert len(attr["by_call_site"]) == 2
    # the unbounded endpoint rollup still saw all 5 calls
    fleet = attr["fleet"]
    assert fleet["calls"] == 5


def test_non_chat_kinds_excluded(qdb):
    """Only kind='chat' rows count — embed/rerank/other carry no prefix cache."""
    _chat_row(qdb, "chat", cached=6, in_tok=12, kind="chat")
    _chat_row(qdb, "embed", cached=6, in_tok=12, kind="embed")
    _chat_row(qdb, "rr", cached=6, in_tok=12, kind="rerank")
    fleet = qdb.cache_attribution(window_s=3600)["fleet"]
    assert fleet["calls"] == 1  # only the chat row


# --------------------------------------------------------------------------- #
# SOUTH FACE (E2E) — the real proxy must survive hostile backend usage:
# no hard-crash, no slot leak, attribution degrades sanely.  (few proxy spins)
# --------------------------------------------------------------------------- #

# JSON-safe hostile cached_tokens values (all round-trip through Starlette's
# JSONResponse). Each is a value the parser must handle WITHOUT failing the
# (otherwise valid) completion.
_JSON_SAFE_HOSTILE_CACHED = [10**9, 2**60, -5, 3.7, True, False, "5", [1, 2], {"a": 1}]


async def test_sync_seam_hostile_never_500_or_leak(proxy):
    """Drive a batch of hostile backend cached_tokens through the SYNC path in
    one proxy: each completion still succeeds (200 — the field is observability-
    only and must never fail a good response), and no slot leaks."""
    assert proxy.total_in_flight() == 0
    for val in _JSON_SAFE_HOSTILE_CACHED:
        proxy.controller.cached_tokens = val
        r = await proxy.chat("hi", model="chat", timeout_s=30)
        assert r.status_code == 200, f"hostile cached_tokens={val!r} failed the completion"
    # hostile prompt_tokens_details shapes via a verbatim usage override
    for ov in (
        {"prompt_tokens": 12, "prompt_tokens_details": [1, 2, 3]},   # list
        {"prompt_tokens": 12, "prompt_tokens_details": None},        # null
        {"prompt_tokens": 12, "prompt_tokens_details": "nope"},      # str
        {"prompt_tokens": 0, "prompt_tokens_details": {"cached_tokens": 4}},  # zero input
        {"completion_tokens": 3},                                    # prompt_tokens absent
    ):
        proxy.controller.cached_tokens = None
        proxy.controller.usage_override = ov
        r = await proxy.chat("hi", model="chat", timeout_s=30)
        assert r.status_code == 200, f"usage_override={ov!r} failed the completion"
    proxy.controller.usage_override = _UNSET
    # absent usage entirely (llama.cpp-ish) via the existing no_usage fault
    r = await proxy.chat("hi", model="chat", fault="no_usage", timeout_s=30)
    assert r.status_code == 200
    proxy.controller.set_fault("none")
    # no slot leaked across the whole hostile batch
    proxy.svc._queue_db.flush(timeout=5.0)
    assert proxy.total_in_flight() == 0
    # the endpoint stays 200 with a sane shape under all that hostile traffic
    resp = await proxy.client.get("/v1/fleet/cache-attribution?window=1h")
    assert resp.status_code == 200
    j = resp.json()
    assert {"window_s", "by_call_site", "by_endpoint", "fleet"} <= set(j)


async def test_sync_nan_infinity_no_hard_crash_no_leak(proxy):
    """NaN/Infinity cached_tokens on the SYNC wire (raw body — Starlette can't
    emit NaN, but json.loads accepts it). Two distinct concerns:
      - the observability PARSE no longer raises (math.isfinite guard →
        cached_tokens degrades to None), so it can't by itself error a good
        completion — proven by test_extract_never_raises_on_nan_inf + the
        streaming test (raw SSE passthrough, no re-serialization);
      - but a NaN echoed in the SYNC body is not JSON-serializable
        (Starlette JSONResponse allow_nan=False), so the response surfaces as a
        structured 5xx — a defensible outcome for a backend that emitted a
        non-JSON-compliant body, and pre-existing / out of Phase-2a scope.
    The invariant floor this test pins: a STRUCTURED HTTP response (no hang, no
    unhandled-crash backstop) and NO slot leak, whatever the status."""
    assert proxy.total_in_flight() == 0
    for tail in ("NaN", "Infinity", "-Infinity"):
        proxy.controller.raw_completion_text = (
            '{"id":"x","object":"chat.completion","choices":[{"index":0,'
            '"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":12,"completion_tokens":1,'
            '"prompt_tokens_details":{"cached_tokens":' + tail + '}}}')
        r = await proxy.chat("hi", model="chat", timeout_s=30)
        assert isinstance(r.status_code, int)   # structured response, not a hang
    proxy.controller.raw_completion_text = None
    proxy.svc._queue_db.flush(timeout=5.0)
    assert proxy.total_in_flight() == 0, "NaN/Inf cached_tokens leaked a slot"


async def test_stream_seam_hostile_never_leak(proxy):
    """Hostile cached_tokens on the STREAMING usage frame (NaN rides via
    json.dumps; a list-shaped details too). Content is still delivered and no
    slot leaks — the observability parse can't wedge the stream path."""
    assert proxy.total_in_flight() == 0
    # NaN in the streamed usage frame
    proxy.controller.cached_tokens = float("nan")
    frames = await proxy.stream_frames("a b c", model="chat", timeout_s=30)
    assert frames, "no stream frames delivered"
    # hostile details shape in the streamed usage frame
    proxy.controller.cached_tokens = None
    proxy.controller.usage_override = {
        "prompt_tokens": 12, "completion_tokens": 3, "prompt_tokens_details": [1, 2]}
    frames2 = await proxy.stream_frames("a b c", model="chat", timeout_s=30)
    assert frames2
    proxy.controller.usage_override = _UNSET
    proxy.svc._queue_db.flush(timeout=5.0)
    assert proxy.total_in_flight() == 0, "hostile stream usage leaked a slot"


# --------------------------------------------------------------------------- #
# NORTH FACE — hostile CALLER inputs can't corrupt attribution or crash capture.
# --------------------------------------------------------------------------- #

async def test_north_face_hostile_caller_cannot_corrupt_attribution(proxy):
    """A caller stuffs its OWN usage/cached_tokens/prompt_tokens_details/call_site
    into the request body, plus unicode + injection-y content. Attribution must
    reflect only the BACKEND-reported cached_tokens (6/12 = 0.5), never the
    caller's fabricated 999999; the endpoint stays 200; no leak."""
    proxy.controller.cached_tokens = 6   # the real backend truth
    hostile_extra = {
        "usage": {"prompt_tokens_details": {"cached_tokens": 999999}},
        "cached_tokens": 123456,
        "prompt_tokens_details": [1, 2, 3],
        "call_site": "../../etc/passwd",
        "agent_id": "'; DROP TABLE proxy_completions;--",
    }
    r = await proxy.chat("héllo \x00 <script> {{7*7}} 世界", model="chat",
                         timeout_s=30, extra=hostile_extra)
    assert r.status_code == 200
    proxy.svc._queue_db.flush(timeout=5.0)
    attr = proxy.svc._queue_db.cache_attribution(window_s=3600)
    # The proxy normalizes model="chat" to its endpoint CLASS, so attribution
    # rows land under the resolved name — derived, not hardcoded: that target has
    # moved twice (→ "classify" 2026-07-03, → "creative" 2026-07-11 boxa
    # consolidation). A bare `next()` on the stale name raised StopIteration
    # INSIDE an async test, which asyncio re-reports as an opaque
    # "coroutine raised StopIteration" — so the default below keeps a missing row
    # failing legibly.
    chat_class = normalize_endpoint("chat")
    chat_ep = next((e for e in attr["by_endpoint"]
                    if e["endpoint"] == chat_class), None)
    assert chat_ep is not None, (
        f"no attribution row for {chat_class!r}; "
        f"got {[e['endpoint'] for e in attr['by_endpoint']]}")
    assert chat_ep["cached_tokens"] == 6           # backend value, not 999999/123456
    assert chat_ep["attributable_input_tokens"] == 12
    assert chat_ep["hit_rate"] == 0.5
    assert proxy.total_in_flight() == 0
    # endpoint renders it fine
    resp = await proxy.client.get("/v1/fleet/cache-attribution")
    assert resp.status_code == 200


async def test_endpoint_hostile_query_params(proxy):
    """The /v1/fleet/cache-attribution endpoint tolerates garbage query params
    (non-numeric limit, junk window, negative/huge limit) → 200, sane shape."""
    proxy.controller.cached_tokens = 6
    await proxy.chat("hi", model="chat", timeout_s=30)
    proxy.svc._queue_db.flush(timeout=5.0)
    for qs in ("?limit=abc", "?limit=-5", "?limit=999999", "?window=garbage",
               "?window=-1h", "?limit=&window=", "?window=99999d"):
        resp = await proxy.client.get("/v1/fleet/cache-attribution" + qs)
        assert resp.status_code == 200, f"query {qs!r} did not return 200"
        j = resp.json()
        assert {"window_s", "by_call_site", "by_endpoint", "fleet"} <= set(j)
        assert isinstance(j["by_call_site"], list)
