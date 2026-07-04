"""Phase 6 (M1) — context-window pre-admission gate.

The proxy knows each endpoint's live per-slot context and estimates input
tokens at submit; an oversized prompt should fail FAST with an actionable,
chunker-compatible message instead of queueing and dying as a backend 400.
Ships shadow-first (count + WARN, admit) behind the ``context_gate_enforce``
runtime flag. Pins: shadow admits+counts; enforce 422s with the canonical
context-overflow marker (so chunkers' re-chunk handling engages) + the
``context_overflow`` code; skips non-chat and 0-context endpoints; boundary
math on the real thinker/companion limits.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from originfleet.framework.token_budget import is_context_overflow_error
from originfleet.llmproxy.backend import BackendResponse
from originfleet.llmproxy.config import ProxyConfig
from originfleet.llmproxy.service import ProxyService


class _Req:
    class _Client:
        host = "127.0.0.1"

    client = _Client()
    headers: dict = {}


def _body(endpoint: str, n_chars: int, max_tokens: int = 100, timeout_s: float = 5.0):
    return {
        "agent_id": "a", "endpoint": endpoint, "priority": "P3_INGESTION",
        "call_site": "t", "payload_type": "chat_completion",
        "payload": {"messages": [{"role": "user", "content": "x" * n_chars}],
                    "max_tokens": max_tokens},
        "timeout_s": timeout_s,
    }


def _ok_backend(svc):
    async def ok_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"content": "y"}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            duration_s=0.01, input_tokens=1, output_tokens=1, finish_reason="stop")
    svc._backend.call = ok_call


@pytest.mark.asyncio
async def test_shadow_counts_but_admits():
    svc = ProxyService(ProxyConfig())
    _ok_backend(svc)
    await svc.startup()
    try:
        # chat (-> classify, 131072/slot after the 2026-07-04 kv-unified swap);
        # ~150K tokens ≈ 600K chars blows it.
        resp = await asyncio.wait_for(
            svc.handle_submit(_body("chat", 600_000, timeout_s=10.0), _Req()),
            timeout=10.0)
        assert resp.status_code == 200  # shadow: admitted (backend faked ok)
        tally = svc._context_overflows["classify"]
        assert tally["count"] == 1
        assert tally["callers"] == {"a": 1}
        assert tally["max_est_in"] >= 149_000
        status = json.loads((await svc.handle_status(_Req())).body)
        assert "classify" in status["reliability"]["context_overflows_shadow"]
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_enforce_422_with_chunker_marker_and_code():
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"context_gate_enforce": True})
    # classify 131072/slot (kv-unified, 2026-07-04) → ~150K tokens ≈ 600K chars.
    resp = await svc.handle_submit(_body("chat", 600_000), _Req())
    body = json.loads(resp.body)
    assert resp.status_code == 422
    assert body["code"] == "context_overflow"
    # The message must trip the chunkers' canonical overflow detector, exactly
    # like the backend's own overflow 400 does — otherwise the gate would
    # REGRESS chunking callers.
    assert is_context_overflow_error(ValueError(body["error"]))


@pytest.mark.asyncio
async def test_enforce_max_tokens_counts_toward_limit():
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"context_gate_enforce": True})
    # chat: 131072/slot (kv-unified, 2026-07-04). est_in ≈ 130000 fits alone;
    # +4000 max_tokens (→134000) overflows — proves max_tokens counts toward the limit.
    resp = await svc.handle_submit(
        _body("chat", 520_000, max_tokens=4000), _Req())
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_boundaries_thinker_vs_companion():
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"context_gate_enforce": True})
    _ok_backend(svc)
    await svc.startup()
    try:
        # Post-kv-unified (2026-07-04) both companion (composer 122B) and thinker
        # (reasoner) run at 131072/slot, so the gate is exercised PER-ENDPOINT
        # against each one's own live limit: an over-131072 prompt to companion
        # 422s; an under-131072 prompt to thinker admits.
        over = 600_000   # ~150K tokens > 131072
        under = 200_000  # ~50K tokens, comfortably inside 131072
        resp = await svc.handle_submit(_body("qwen-composer", over), _Req())
        assert resp.status_code == 422
        resp = await asyncio.wait_for(
            svc.handle_submit(_body("llama-thinker", under, timeout_s=10.0), _Req()),
            timeout=10.0)
        assert resp.status_code == 200
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_non_chat_and_unknown_context_skip_the_gate():
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"context_gate_enforce": True})
    _ok_backend(svc)
    await svc.startup()
    try:
        # Embedding payloads aren't chat completions — never gated (their
        # endpoint has context_per_slot but token estimation is chat-shaped).
        body = {
            "agent_id": "a", "endpoint": "bge-m3-embed", "priority": "P3_INGESTION",
            "call_site": "t", "payload_type": "embedding",
            "payload": {"inputs": ["x" * 200_000]}, "timeout_s": 10.0,
        }
        resp = await asyncio.wait_for(svc.handle_submit(body, _Req()), timeout=10.0)
        assert resp.status_code == 200
        # An endpoint with context_per_slot == 0 (nothing discovered/seeded)
        # must skip rather than reject everything.
        svc._config.endpoints["classify"].context_per_slot = 0
        resp = await asyncio.wait_for(
            svc.handle_submit(_body("chat", 200_000, timeout_s=10.0), _Req()),
            timeout=10.0)
        assert resp.status_code == 200
        assert "classify" not in svc._context_overflows
    finally:
        await svc.shutdown()
