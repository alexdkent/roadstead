"""Cache-hit response must include "status": "ok".

Both ProxyLLMClient and NexusEmbedder._post_via_proxy check
`result.get("status") != "ok"` to detect success. A cache-hit
response missing that field is treated as an error by every caller.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from originfleet.llmproxy.coalesce import DeterministicCache
from originfleet.llmproxy.config import ProxyConfig
from originfleet.llmproxy.service import ProxyService


def _make_service() -> ProxyService:
    config = ProxyConfig()
    return ProxyService(config)


class _FakeRequest:
    class _Client:
        host = "127.0.0.1"
    client = _Client()


@pytest.mark.asyncio
async def test_cache_hit_has_status_ok():
    svc = _make_service()

    svc._cache.put("test_key", {"choices": [{"message": {"content": "cached"}}]})

    body = {
        "agent_id": "test_agent",
        "endpoint": "qwen-analyst",
        "priority": "P1_TURN_SUPPORT",
        "call_site": "test",
        "payload_type": "chat_completion",
        "payload": {
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
            "max_tokens": 64,
        },
        "timeout_s": 10.0,
    }

    original_cache_key = svc._cache.cache_key
    svc._cache.cache_key = lambda endpoint, payload: "test_key"

    resp = await svc.handle_submit(body, _FakeRequest())
    result = json.loads(resp.body.decode())

    assert result["status"] == "ok"
    assert result["cache_hit"] is True
    assert result["response"]["choices"][0]["message"]["content"] == "cached"
    assert result["queue_wait_ms"] == 0


@pytest.mark.asyncio
async def test_cache_hit_response_shape_matches_dispatch():
    """Cache-hit response must have the same top-level keys as a
    successful dispatch response (minus timing jitter)."""
    svc = _make_service()

    svc._cache.put("k2", {"id": "chatcmpl-abc"})
    svc._cache.cache_key = lambda endpoint, payload: "k2"

    body = {
        "agent_id": "test_agent",
        "endpoint": "qwen-analyst",
        "priority": "P2_POST_TURN",
        "call_site": "test",
        "payload_type": "chat_completion",
        "payload": {
            "messages": [{"role": "user", "content": "x"}],
            "temperature": 0,
            "max_tokens": 32,
        },
        "timeout_s": 5.0,
    }

    resp = await svc.handle_submit(body, _FakeRequest())
    result = json.loads(resp.body.decode())

    expected_keys = {
        "status", "request_id", "queue_wait_ms",
        "backend_latency_ms", "estimated_cost_ss",
        "response", "cache_hit",
    }
    assert set(result.keys()) == expected_keys


# --- Phase 1 hardening: full-payload cache key ------------------------------
#
# The old key enumerated fields (system/messages/grammar/max_tokens), so two
# temperature=0 requests with identical messages but different response_format
# / structured_outputs / tools COLLIDED — the second caller got the first's
# response (cache poisoning for structured extraction). The key now hashes the
# entire canonical payload.

def _base_payload():
    return {
        "messages": [{"role": "user", "content": "classify this"}],
        "temperature": 0,
        "max_tokens": 64,
    }


def test_cache_key_differs_on_response_format():
    cache = DeterministicCache()
    a = cache.cache_key("chat", _base_payload())
    b = cache.cache_key("chat", {
        **_base_payload(),
        "response_format": {"type": "json_schema", "json_schema": {"schema": {}}},
    })
    assert a and b and a != b


def test_cache_key_differs_on_structured_outputs_grammar():
    cache = DeterministicCache()
    a = cache.cache_key("thinker", {
        **_base_payload(), "structured_outputs": {"grammar": 'root ::= "a"'},
    })
    b = cache.cache_key("thinker", {
        **_base_payload(), "structured_outputs": {"grammar": 'root ::= "b"'},
    })
    assert a and b and a != b


def test_cache_key_differs_on_tools():
    cache = DeterministicCache()
    a = cache.cache_key("chat", {**_base_payload(), "tools": [{"type": "function",
        "function": {"name": "f1", "parameters": {}}}]})
    b = cache.cache_key("chat", {**_base_payload(), "tools": [{"type": "function",
        "function": {"name": "f2", "parameters": {}}}]})
    assert a and b and a != b


def test_cache_key_stable_for_identical_payload_and_gates_unchanged():
    cache = DeterministicCache()
    # Identical payloads (independent dicts, different key order) → same key.
    p1 = _base_payload()
    p2 = {"max_tokens": 64, "temperature": 0,
          "messages": [{"role": "user", "content": "classify this"}]}
    assert cache.cache_key("chat", p1) == cache.cache_key("chat", p2)
    # Same payload, different endpoint → different key.
    assert cache.cache_key("chat", p1) != cache.cache_key("companion", p1)
    # Cacheability gates unchanged: non-zero temperature / streaming → None.
    assert cache.cache_key("chat", {**p1, "temperature": 0.7}) is None
    assert cache.cache_key("chat", {**p1, "stream": True}) is None
