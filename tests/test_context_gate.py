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
        # chat (-> tier2-chat, jetty :30000, since the 2026-08-19 tier2 split;
        # 262144/slot). ~300K tokens ≈ 1.2M chars blows it. The old numbers here
        # (600K chars against creative's 65536) went with the alias.
        resp = await asyncio.wait_for(
            svc.handle_submit(_body("chat", 1_200_000, timeout_s=10.0), _Req()),
            timeout=10.0)
        assert resp.status_code == 200  # shadow: admitted (backend faked ok)
        tally = svc._context_overflows["tier2-chat"]
        assert tally["count"] == 1
        assert tally["callers"] == {"a": 1}
        assert tally["max_est_in"] >= 299_000
        status = json.loads((await svc.handle_status(_Req())).body)
        assert "tier2-chat" in status["reliability"]["context_overflows_shadow"]
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_enforce_422_with_chunker_marker_and_code():
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"context_gate_enforce": True})
    # chat -> tier2-chat, 262144/slot (2026-08-19 split) → ~300K tokens ≈ 1.2M chars.
    resp = await svc.handle_submit(_body("chat", 1_200_000), _Req())
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
    # 🚨 THIS TEST WAS VACUOUS AND THE 2026-08-19 SPLIT EXPOSED IT. It claimed
    # "est_in ≈ 130000 fits alone; +4000 max_tokens overflows", sized against a
    # 131072/slot comment — but `chat` resolved to `creative`, which was trimmed
    # 131072 -> 65536 on 2026-07-11. est_in of 130000 was ALREADY double the real
    # limit, so the 422 came from the input alone and the max_tokens bump proved
    # nothing. A guard satisfiable by something OTHER than the thing it is about.
    #
    # Re-sized against tier2-chat's real 262144, and with the CONTROL asserted:
    # 1_048_000 chars = 262,000 est_in, which fits alone (+100 default max_tokens
    # = 262,100 <= 262,144) and only overflows once max_tokens=4000 is counted.
    # Without the control the repair could rot back into the same vacuum.
    # The control asserts the GATE's verdict, not the call's outcome: with no
    # faked backend an admitted request just times out, and a 200 here would be
    # asserting the wrong thing. What must be true is that the gate did not
    # reject it — no 422, no context_overflow code.
    ok = await svc.handle_submit(_body("chat", 1_048_000, timeout_s=0.05), _Req())
    assert ok.status_code != 422 and json.loads(ok.body).get("code") != "context_overflow", (
        "control: est_in must FIT alone, or the max_tokens assertion below is "
        "satisfied by the input size and proves nothing")
    resp = await svc.handle_submit(
        _body("chat", 1_048_000, max_tokens=4000), _Req())
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_boundaries_thinker_vs_creative():
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"context_gate_enforce": True})
    _ok_backend(svc)
    await svc.startup()
    try:
        # The gate is exercised PER-ENDPOINT against each one's own configured
        # limit — derived from context_per_slot at runtime so a capacity bump
        # (131072→262144 on 2026-07-09 when classify freed composer memory)
        # can't strand this test on a stale constant again: an over-limit
        # prompt to companion 422s; a comfortably-under prompt to thinker admits.
        # 2026-07-30: `qwen-composer` no longer names an endpoint ROLE — composer/companion
        # moved to tier3 (role `llama-thinker`, 700K). Deliberately NOT using `tier3-backup`
        # as the small side: its stanza is `status: on_demand`, which routes admission through
        # OnDemandManager.ensure_loaded BEFORE this gate, so it is a bad fixture here. `creative`
        # is an ACTIVE chat endpoint with a genuinely smaller window, which keeps this test doing
        # what it was written to do: prove the gate applies PER-ENDPOINT against each one's own
        # limit rather than one global constant.
        small_slot = next(ep.context_per_slot for ep in svc._config.endpoints.values()
                          if ep.role == "creative")
        over = small_slot * 4 + 200_000      # chars ≈ 4/token → safely past the slot limit
        under = 200_000                      # ~50K tokens, inside any live thinker limit
        resp = await svc.handle_submit(_body("creative", over), _Req())
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
        # `chat` aliases to the `tier2-chat` endpoint since the 2026-08-19 split.
        svc._config.endpoints["tier2-chat"].context_per_slot = 0
        resp = await asyncio.wait_for(
            svc.handle_submit(_body("chat", 200_000, timeout_s=10.0), _Req()),
            timeout=10.0)
        assert resp.status_code == 200
        assert "tier2-chat" not in svc._context_overflows
    finally:
        await svc.shutdown()
