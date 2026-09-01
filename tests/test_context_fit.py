"""The context-fit predicate — ONE copy, and the four gates that share it.

Until 2026-09-01 "does this request fit in this much context?" was written out
by hand four times: the admission gate (``lifecycle.handle_submit``), the
failover gate (``failover.plan``), the spill gate (``scheduler._admit``) and
the WAL-recovery shadow tally (``service``). Three of them agreed; the fourth
had silently diverged and was DEAD (see ``test_recovery_tally_fires_*`` below).
That is the reason this is a shared predicate and not four tidy copies: a
safety predicate written N times is N answers to one question, and the copy
that drifts is the one nobody watches.

🚨 What this file does NOT assert is that the four gates behave the same. They
must not — the denominator and the consequence are legitimately per-site, and
collapsing them would arm ``context_gate_enforce`` on gates nobody flipped it
for. The last section pins that difference SURVIVING the sharing.
"""

from __future__ import annotations

import ast
import asyncio
import itertools
import json
from pathlib import Path

import pytest

from roadstead.backend import BackendResponse
from roadstead.config import ProxyConfig
from roadstead.cost_model import (
    CONTEXT_OVERFLOW_MARKER, ContextFit, context_fit, estimate_input_tokens,
)
from roadstead.service import ProxyService
from tests.wire_contract import CONTEXT_OVERFLOW_MARKER as DOC_MARKER

PKG = Path(__file__).resolve().parent.parent / "roadstead"


# ---------------------------------------------------------------------------
# The conventions, each of which looks arbitrary until it bites
# ---------------------------------------------------------------------------

def _chat(chars: int, **extra) -> dict:
    return {"messages": [{"role": "user", "content": "x" * chars}], **extra}


def test_unknown_ceiling_admits():
    """0 means "not known", NOT "a ceiling of zero".

    vLLM publishes no per-slot context and a catalog may seed none. The obvious
    reading — 0 tokens available, refuse — would refuse every request to every
    endpoint whose capacity we never discovered, which is the majority of a
    fresh deployment. ``intent.py`` follows the same convention for
    ``min_context``.
    """
    assert context_fit(_chat(10_000_000), "chat_completion", 0).fits
    assert context_fit(_chat(10_000_000), "chat_completion", -1).fits


def test_non_chat_payloads_are_not_gated():
    """An embedding has no ``messages`` to walk, so the number would be
    meaningless rather than merely imprecise."""
    for ptype in ("embedding", "rerank", "", None):
        assert context_fit(_chat(10_000_000), ptype, 8192).fits


def test_max_tokens_counts_toward_the_ceiling_and_only_when_usable():
    """Reserved output shares the window with the input — a prompt that fits
    and a completion that does not is still a backend 400.

    Only a positive ``int`` counts: ``"512"``, ``None``, ``0`` and ``-5`` are
    all "unspecified" rather than an error, because refusing a request over a
    malformed hint would be a gate inventing a rejection of its own.
    """
    assert not context_fit(_chat(8, max_tokens=100), "chat_completion", 50).fits
    for junk in ("512", None, 0, -5, 1.5):
        assert context_fit(_chat(8, max_tokens=junk), "chat_completion", 50).fits


def test_the_boundary_admits_exactly_at_the_ceiling():
    fit = context_fit(_chat(400, max_tokens=10), "chat_completion", 110)
    assert (fit.est_in, fit.est_out) == (100, 10)
    assert fit.fits                                       # 110 <= 110
    assert not context_fit(_chat(400, max_tokens=11), "chat_completion", 110).fits


def test_the_estimator_is_skipped_when_the_gate_does_not_apply():
    """``est_in`` is 0, not a computed number nobody reads.

    This runs on the admission path of EVERY request; walking a 200K-char
    transcript to produce a value the caller discards is the kind of cost that
    only shows up under load.
    """
    fit = context_fit(_chat(10_000_000), "embedding", 8192)
    assert (fit.fits, fit.est_in, fit.est_out) == (True, 0, 0)


# ---------------------------------------------------------------------------
# The marker is WIRE CONTRACT, and it is defined once
# ---------------------------------------------------------------------------

def test_marker_matches_the_document():
    """Same two-ended pin the client SDK uses — transcribed from docs/api.md
    §2.2, not imported from the emitting side, so agreement is evidence."""
    assert CONTEXT_OVERFLOW_MARKER == DOC_MARKER


def test_marker_is_spelled_once_on_each_side_of_the_client_boundary():
    """🚨 An AST constant sweep, not a substring grep over source text.

    The failure this catches is a site that re-words its own overflow refusal
    and drops the marker, which stops chunking callers' re-chunk handling
    engaging — a silent regression to exactly the callers the gate was built
    for. Comments MENTIONING the marker are fine and several do; only a string
    LITERAL is a second spelling waiting to drift, so this walks
    ``ast.Constant`` and never the raw bytes.

    TWO spellings are expected, not one, and the second is doctrine rather than
    an oversight: ``roadstead.client`` imports NOTHING from the server, so it
    transcribes the marker from ``docs/api.md`` §2.2 the same way
    ``tests/wire_contract.py`` does. A client that imported the server's
    constant would agree with it by construction and could never catch a
    drift — which is the whole reason the boundary is AST-enforced. So the rule
    is one spelling per SIDE: the server's gates share ``cost_model``'s, the SDK
    keeps its own, and ``test_marker_matches_the_document`` pins them together
    through the document.
    """
    spellers = []
    for path in sorted(PKG.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if CONTEXT_OVERFLOW_MARKER in node.value:
                    spellers.append(f"{path.relative_to(PKG.parent)}:{node.lineno}")
    server = [s for s in spellers if not s.startswith("roadstead/client/")]
    client = [s for s in spellers if s.startswith("roadstead/client/")]
    assert server == [f"roadstead/cost_model.py:{_marker_lineno()}"], server
    assert len(client) == 1, client


def _marker_lineno() -> int:
    tree = ast.parse((PKG / "cost_model.py").read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and CONTEXT_OVERFLOW_MARKER in node.value):
            return node.lineno
    raise AssertionError("marker not defined in cost_model.py")


def test_no_gate_re_derives_the_arithmetic():
    """🚨 The predicate must not grow a fifth copy.

    A module that calls ``estimate_input_tokens`` AND compares the result
    against something is re-deriving the gate. The four gates now call
    ``context_fit``; the remaining ``estimate_input_tokens`` callers
    (``scheduler.enqueue``, ``lifecycle``'s reporting paths, ``enriched``) use
    the number as a MEASUREMENT and never as a ceiling test, which is why this
    looks for the comparison rather than the call.
    """
    offenders = []
    for path in sorted(PKG.rglob("*.py")):
        if path.name == "cost_model.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        # Names bound from an estimate_input_tokens() call, per function scope.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            call = node.value
            if not (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "estimate_input_tokens"):
                continue
            bound = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if not bound:
                continue
            for cmp_node in ast.walk(tree):
                if not isinstance(cmp_node, ast.Compare):
                    continue
                if any(isinstance(n, ast.Name) and n.id in bound
                       for n in ast.walk(cmp_node)):
                    offenders.append(
                        f"{path.relative_to(PKG.parent)}:{cmp_node.lineno}")
    assert not offenders, (
        "a context ceiling is being re-derived instead of calling "
        f"cost_model.context_fit: {offenders}")


# ---------------------------------------------------------------------------
# The three live gates agree — differentially, over a corpus
# ---------------------------------------------------------------------------

_CORPUS = [
    {},
    {"messages": []},
    _chat(2),
    _chat(40_000),
    _chat(40_000, max_tokens=512),
    _chat(8, max_tokens=1_000_000),
    {"system": "s" * 20_000, "messages": [{"role": "user", "content": "q"}]},
    {"messages": [{"role": "assistant", "content": None,
                   "tool_calls": [{"function": {"name": "f",
                                                "arguments": "a" * 30_000}}]}]},
    {"messages": [{"role": "user", "content": [{"type": "text", "text": "t" * 5_000},
                                               {"type": "image_url"}]}]},
    {"messages": "not a list"},
    {"messages": [None, 5, "raw string message"]},
]
_LIMITS = [0, 1, 512, 8192, 131_072]


def test_scheduler_spill_gate_is_the_shared_predicate():
    """``scheduler._fits_context`` is a view of ``context_fit``, not a copy."""
    from roadstead.scheduler import _fits_context

    class _R:
        def __init__(self, p, t):
            self.payload, self.payload_type = p, t

    for payload, limit, ptype in itertools.product(
            _CORPUS, _LIMITS, ["chat_completion", "embedding"]):
        assert (_fits_context(_R(payload, ptype), limit)
                == context_fit(payload, ptype, limit).fits)


def test_failover_refusal_carries_the_shared_arithmetic():
    """The refusal message is assembled from ``overflow_detail``, so the
    numbers in it cannot disagree with the numbers the gate decided on."""
    fit = context_fit(_chat(40_000, max_tokens=512), "chat_completion", 8192)
    detail = fit.overflow_detail("tier3")
    assert not fit.fits
    assert f"est {fit.est_in} input tokens" in detail
    assert f"max_tokens {fit.est_out}" in detail
    assert CONTEXT_OVERFLOW_MARKER in detail
    assert "8192/slot on tier3" in detail
    # est_in is the ESTIMATOR's number, not a second opinion.
    assert fit.est_in == estimate_input_tokens(_chat(40_000, max_tokens=512))


# ---------------------------------------------------------------------------
# 🚨 The consequences stay DIFFERENT — sharing the predicate did not fuse them
# ---------------------------------------------------------------------------

class _Req:
    class _Client:
        host = "127.0.0.1"

    client = _Client()
    headers: dict = {}


def _body(endpoint: str, n_chars: int, max_tokens: int = 100, timeout_s: float = 5.0):
    return {"agent_id": "a", "endpoint": endpoint, "priority": "P3_INGESTION",
            "call_site": "t", "payload_type": "chat_completion",
            "payload": {"messages": [{"role": "user", "content": "x" * n_chars}],
                        "max_tokens": max_tokens},
            "timeout_s": timeout_s}


def _ok_backend(svc):
    async def ok_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"content": "y"}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            duration_s=0.01, input_tokens=1, output_tokens=1, finish_reason="stop")
    svc._backend.call = ok_call


@pytest.mark.asyncio
async def test_admission_gate_still_shadows_by_default():
    """🚨 The load-bearing half of the refactor.

    The admission gate is the ONLY one with a shadow mode, and it is off-by-
    default: it counts and admits. Failover and spill refuse unconditionally.
    Had the shared predicate absorbed the consequence, this request would 422
    on a flag nobody flipped.
    """
    svc = ProxyService(ProxyConfig())
    _ok_backend(svc)
    await svc.startup()
    try:
        resp = await asyncio.wait_for(
            svc.handle_submit(_body("chat", 1_200_000, timeout_s=10.0), _Req()),
            timeout=10.0)
        assert resp.status_code == 200            # admitted
        assert svc._context_overflows["tier2"]["count"] == 1   # and counted
    finally:
        await svc.shutdown()


# ---------------------------------------------------------------------------
# The regression that started all this: the DEAD fourth copy
# ---------------------------------------------------------------------------

def test_recovered_requests_carry_no_cached_estimate():
    """The fact that made the recovery tally dead.

    ``est_input_tokens`` is cached by ``scheduler.enqueue``; ``recover_queued``
    builds a ``QueuedRequest`` straight from the WAL row and never reaches that
    path, so the field is its default 0. The old tally read
    ``req.est_input_tokens or 0`` and could therefore only fire when
    ``max_tokens`` ALONE exceeded the ceiling.

    Pinned as a fact rather than fixed in ``recover_queued``: the estimate is
    derivable from the payload, and re-deriving it is what ``context_fit``
    already does everywhere else.
    """
    import inspect
    from roadstead.queue import PersistentQueue
    src = inspect.getsource(PersistentQueue.recover_queued)
    assert "est_input_tokens" not in src


def test_recovery_tally_fires_for_an_oversized_recovered_request(tmp_path):
    """The bug, observed fixed.

    Before 2026-09-01 this counted 0: the recovery path estimated from a field
    the recovery path does not populate. It now estimates from the payload,
    like the other three gates.
    """
    from roadstead.queue import PersistentQueue
    from roadstead.scheduler import QueuedRequest

    db = str(tmp_path / "q.db")
    pq = PersistentQueue(db)
    # 1.2M chars ≈ 300K tokens, against tier2's 262144/slot.
    pq.persist_enqueue(QueuedRequest.create(
        agent_id="a", endpoint="tier2", priority="P3_INGESTION",
        call_site="t", payload_type="chat_completion",
        payload={"messages": [{"role": "user", "content": "x" * 1_200_000}],
                 "max_tokens": 100},
        timeout_s=600.0, caller_id="c", request_id="r1"))
    pq.close()

    svc = ProxyService(ProxyConfig(queue_db_path=db))
    asyncio.run(_recover_and_shutdown(svc))
    tally = svc._context_overflows.get("tier2")
    assert tally is not None, "recovery tally never fired — the dead copy is back"
    assert tally["count"] == 1
    assert tally["callers"] == {"c": 1}
    assert tally["max_est_in"] >= 299_000, tally


async def _recover_and_shutdown(svc):
    await svc.startup()
    await svc.shutdown()
