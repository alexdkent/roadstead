"""WS1 — fail-loud contract through the proxy handle_submit path.

An invalid grammar must return a 422 grammar_invalid error and never be
enqueued/dispatched. A valid multi-line grammar must be normalized in
place so the dispatched payload is wire-correct.
"""

from __future__ import annotations

import json

import pytest

from roadstead.config import ProxyConfig
from roadstead.service import ProxyService


class _FakeRequest:
    class _Client:
        host = "127.0.0.1"
    client = _Client()


def _svc() -> ProxyService:
    return ProxyService(ProxyConfig())


@pytest.mark.asyncio
async def test_invalid_grammar_fails_loud_422():
    svc = _svc()
    body = {
        "agent_id": "forum-agent", "endpoint": "tier3",
        "priority": "P3_INGESTION", "call_site": "forum-agent.proposal_emitter",
        "payload_type": "chat_completion",
        "payload": {
            "messages": [{"role": "user", "content": "x"}],
            "extra_body": {"grammar": 'root ::= item ( ws "," ws item ){0,2000}\nitem ::= "x"\nws ::= [ ]*\n'},
        },
        "timeout_s": 10.0,
    }
    resp = await svc.handle_submit(body, _FakeRequest())
    assert resp.status_code == 422
    result = json.loads(resp.body.decode())
    assert result["status"] == "error"
    assert result["error"] == "grammar_invalid"
    assert "repetition_over_threshold" in result["detail"]


@pytest.mark.asyncio
async def test_valid_multiline_grammar_normalized_in_place():
    svc = _svc()
    grammar = (
        'root ::= "{" ws\n'
        '         "\\"a\\"" ws ":" ws string\n'
        '         ws "}"\n'
        'string ::= "\\"" "x" "\\""\n'
        'ws ::= [ \\t\\n]*\n'
    )
    payload = {
        "messages": [{"role": "user", "content": "x"}],
        "extra_body": {"grammar": grammar},
    }
    body = {
        "agent_id": "knowledge", "endpoint": "tier3",
        "priority": "P1_TURN_SUPPORT", "call_site": "knowledge.extract_entities",
        "payload_type": "chat_completion", "payload": payload, "timeout_s": 10.0,
    }
    # No backend in test → dispatch will fail, but grammar processing happens
    # first and mutates payload. We assert the mutation, not the dispatch.
    from roadstead.scheduler import QueuedRequest
    req = QueuedRequest.create(
        agent_id="knowledge", endpoint="tier3", priority="P1_TURN_SUPPORT",
        call_site="knowledge.extract_entities", payload_type="chat_completion",
        payload=payload, timeout_s=10.0,
    )
    err = svc._process_grammar(req)
    assert err is None
    normalized = req.payload["extra_body"]["grammar"]
    assert "::= (" in normalized  # root RHS wrapped


@pytest.mark.asyncio
async def test_no_grammar_passes_through():
    svc = _svc()
    from roadstead.scheduler import QueuedRequest
    req = QueuedRequest.create(
        agent_id="sidekick", endpoint="tier1", priority="P1_TURN_SUPPORT",
        call_site="sidekick.proxy_client", payload_type="chat_completion",
        payload={"messages": [{"role": "user", "content": "hi"}]}, timeout_s=10.0,
    )
    assert svc._process_grammar(req) is None


def test_grammar_result_cached():
    svc = _svc()
    from roadstead.scheduler import QueuedRequest
    payload = {
        "messages": [{"role": "user", "content": "x"}],
        "extra_body": {"grammar": 'root ::= "x"\n'},
    }
    req = QueuedRequest.create(
        agent_id="a", endpoint="tier3", priority="P1_TURN_SUPPORT",
        call_site="t", payload_type="chat_completion", payload=payload, timeout_s=10.0,
    )
    svc._process_grammar(req)
    assert len(svc._grammar_cache) == 1
