"""Empty-completion (position-0-EOS) rescue — 2026-06-11.

Some prompts make the model emit EOS as its first token: a 1-token
"completion" with empty content, DETERMINISTIC for that prompt (the sidekick
craft_6 failure: 6/6 across temps and thinking modes). A plain same-bytes
retry can never recover it, so the in-proxy retry re-dispatches with
``min_tokens`` (masks EOS for the first N positions; vLLM honors it,
llama.cpp ignores unknown fields). Pins:

  - the retry payload carries min_tokens; the FIRST attempt never does;
  - ``req.payload`` (corpus-persisted) is never mutated;
  - recovery → ok envelope + attempts/recovered counters;
  - both-attempts-empty → 502 surfaces, recovered stays 0;
  - a non-empty transient (BackendUnavailable) retries WITHOUT min_tokens.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from originfleet.llmproxy.backend import (
    BackendError,
    BackendResponse,
    BackendUnavailable,
)
from originfleet.llmproxy.config import ProxyConfig
from originfleet.llmproxy.service import _EMPTY_RESCUE_MIN_TOKENS, ProxyService


class _Req:
    class _Client:
        host = "127.0.0.1"

    client = _Client()
    headers: dict = {}


def _body(timeout_s=20.0):
    return {
        "agent_id": "sidekick", "endpoint": "llama-thinker", "priority": "P2_POST_TURN",
        "call_site": "sidekick.compose_daily_song.craft_6", "payload_type": "chat_completion",
        "payload": {"messages": [{"role": "user", "content": "x"}],
                    "max_tokens": 4000, "temperature": 0.85},
        "timeout_s": timeout_s,
    }


def _ok_response():
    return BackendResponse(
        status_code=200,
        body={"choices": [{"message": {"content": "# The Quiet Takes Root"},
                           "finish_reason": "stop"}],
              "usage": {"prompt_tokens": 100, "completion_tokens": 696}},
        duration_s=0.05, input_tokens=100, output_tokens=696, finish_reason="stop")


def _empty_error():
    return BackendError(
        502, "backend llama-thinker returned empty completion "
             "(no content, output_tokens=1)")


@pytest.mark.asyncio
async def test_rescue_injects_min_tokens_on_retry_and_recovers():
    svc = ProxyService(ProxyConfig())
    seen: list[dict] = []

    async def fake_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        seen.append(payload)
        if len(seen) == 1:
            raise _empty_error()
        return _ok_response()

    svc._backend.call = fake_call
    await svc.startup()
    try:
        body = _body()
        resp = await asyncio.wait_for(svc.handle_submit(body, _Req()), timeout=15.0)
        result = json.loads(resp.body)
        assert resp.status_code == 200 and result["status"] == "ok"
        # First attempt: caller's bytes, no min_tokens. Retry: rescue payload.
        assert "min_tokens" not in seen[0]
        assert seen[1]["min_tokens"] == _EMPTY_RESCUE_MIN_TOKENS
        # Everything else in the retry payload is the caller's, unchanged.
        assert seen[1]["messages"] == seen[0]["messages"]
        assert seen[1]["max_tokens"] == 4000
        # The corpus payload was never mutated.
        assert "min_tokens" not in body["payload"]
        assert svc._empty_rescue_attempts == 1
        assert svc._empty_rescue_recovered == 1
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_both_attempts_empty_surfaces_502_unrecovered():
    svc = ProxyService(ProxyConfig())
    calls = {"n": 0}

    async def always_empty(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        calls["n"] += 1
        raise _empty_error()

    svc._backend.call = always_empty
    await svc.startup()
    try:
        resp = await asyncio.wait_for(
            svc.handle_submit(_body(), _Req()), timeout=15.0)
        result = json.loads(resp.body)
        assert resp.status_code == 502
        assert "empty completion" in result["error"]  # deferrable marker intact
        assert calls["n"] == 2  # one plain + one rescue attempt
        assert svc._empty_rescue_attempts == 1
        assert svc._empty_rescue_recovered == 0
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_non_empty_transient_retry_does_not_inject():
    svc = ProxyService(ProxyConfig())
    seen: list[dict] = []

    async def flaky(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        seen.append(payload)
        if len(seen) == 1:
            raise BackendUnavailable("backend llama-thinker unreachable: boom")
        return _ok_response()

    svc._backend.call = flaky
    await svc.startup()
    try:
        resp = await asyncio.wait_for(svc.handle_submit(_body(), _Req()), timeout=15.0)
        assert json.loads(resp.body)["status"] == "ok"
        assert len(seen) == 2
        assert "min_tokens" not in seen[1]  # plain retry, no rescue
        assert svc._empty_rescue_attempts == 0
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_status_exposes_rescue_counters():
    svc = ProxyService(ProxyConfig())
    status = json.loads((await svc.handle_status(_Req())).body)
    r = status["reliability"]
    assert r["empty_rescue_attempts"] == 0
    assert r["empty_rescue_recovered"] == 0
