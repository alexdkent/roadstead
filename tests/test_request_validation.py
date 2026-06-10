"""Front-door request validation — a malformed field must never escape as an
unhandled 500 (regression for the `ValueError: unknown LLM priority
'P3_BACKGROUND'` storm, 2026-06-06).

Three tiers: deterministically correct a known-correctable class → soft-default
with a WARNING → (config-load only) hard-raise.
"""

from __future__ import annotations

import pytest

from originfleet.llmproxy.backend import BackendResponse
from originfleet.llmproxy.config import LLMPriority, ProxyConfig, normalize_endpoint
from originfleet.llmproxy.service import ProxyService, _to_float, _to_int
from originfleet.llmproxy import __main__ as proxy_main


# --- tier 1: deterministic correction ---------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("P3_BACKGROUND", LLMPriority.P3_INGESTION),   # the live bug: band-suffixed
    ("P0_REALTIME", LLMPriority.P0_REALTIME),       # exact member
    ("p2_post_turn", LLMPriority.P2_POST_TURN),     # case-insensitive member
    ("BACKGROUND", LLMPriority.P3_INGESTION),       # band name
    ("interactive", LLMPriority.P1_TURN_SUPPORT),   # band name
    ("foreground", LLMPriority.P2_POST_TURN),       # band name
    ("ingestion", LLMPriority.P3_INGESTION),        # legacy alias
    ("P4_anything", LLMPriority.P4_HYGIENE),        # numeric-prefix correction
    ("p1-turn", LLMPriority.P1_TURN_SUPPORT),       # numeric prefix, dash sep
    (2, LLMPriority.P2_POST_TURN),                  # valid int
    (None, LLMPriority.P1_TURN_SUPPORT),            # None → P1
])
def test_coerce_corrects_known_classes(value, expected):
    assert LLMPriority.coerce(value) == expected


def test_coerce_out_of_range_int_clamped():
    assert LLMPriority.coerce(99) == LLMPriority.P4_HYGIENE
    assert LLMPriority.coerce(-3) == LLMPriority.P0_REALTIME


# --- tier 2 vs tier 3: soft-default (request path) vs raise (config) ---------

def test_coerce_soft_defaults_when_default_given():
    # request path: unknown garbage must NOT raise — defaults + warns.
    assert LLMPriority.coerce("urgent!!", default=LLMPriority.P1_TURN_SUPPORT) \
        == LLMPriority.P1_TURN_SUPPORT


def test_coerce_raises_without_default():
    # config-load strictness preserved: a misconfigured value fails loud.
    with pytest.raises(ValueError):
        LLMPriority.coerce("not_a_priority")


# --- helpers + normalize_endpoint --------------------------------------------

def test_safe_numeric_helpers_default_on_garbage():
    assert _to_int("abc", 7) == 7
    assert _to_int(None, 5) == 5
    assert _to_int("12", 0) == 12
    assert _to_float("x", 1.5) == 1.5
    assert _to_float("3.0", 0.0) == 3.0


def test_normalize_endpoint_tolerates_non_string():
    # a non-string endpoint must not AttributeError (str() coercion).
    assert normalize_endpoint(123) == "123"


# --- integration: handle_submit with malformed fields must not 500 -----------

async def _started_service() -> ProxyService:
    svc = ProxyService(ProxyConfig())

    async def fake_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(status_code=200, body={
            "id": "x", "object": "chat.completion", "created": 1,
            "model": "llama-thinker",
            "choices": [{"index": 0, "message": {"role": "assistant",
                         "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }, duration_s=0.01, input_tokens=3, output_tokens=1)

    async def _none(*a, **k):
        return None

    svc._backend.call = fake_call
    svc._backend.probe_vllm_capacity = _none
    svc._backend.probe_props = _none
    svc._backend.probe_models = _none
    await svc.startup()
    return svc


class _FakeRequest:
    class _Client:
        host = "172.16.0.5"

    class _URL:
        path = "/v1/submit"

    client = _Client()
    headers: dict = {}
    method = "POST"
    url = _URL()


@pytest.mark.asyncio
async def test_submit_bad_priority_and_timeout_does_not_500():
    svc = await _started_service()
    try:
        body = {
            "agent_id": "test",
            "endpoint": "thinker",
            "priority": "P3_BACKGROUND",       # would have 500'd before
            "timeout_s": "soon",               # un-coercible → default
            "call_site": "test",
            "payload": {"model": "llama-thinker",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 8},
        }
        resp = await svc.handle_submit(body, _FakeRequest())
        assert resp.status_code == 200
    finally:
        await svc.shutdown()


# --- the global exception-handler backstop -----------------------------------

@pytest.mark.asyncio
async def test_invalid_json_handler_returns_400():
    resp = await proxy_main._on_invalid_json(_FakeRequest(), ValueError("bad"))
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_unhandled_handler_returns_500_envelope():
    resp = await proxy_main._on_unhandled(_FakeRequest(), RuntimeError("boom"))
    assert resp.status_code == 500


# --- Phase 3 hardening: unknown-endpoint gate on /v1/submit -------------------

import asyncio
import json as _json

from originfleet.llmproxy.service import ProxyService as _Svc
from originfleet.llmproxy.config import ProxyConfig as _Cfg


class _LoopbackReq:
    class _Client:
        host = "127.0.0.1"

    client = _Client()
    headers: dict = {}


def _submit_body(endpoint):
    return {
        "agent_id": "a", "endpoint": endpoint, "priority": "P3_INGESTION",
        "call_site": "t", "payload_type": "chat_completion",
        "payload": {"messages": [{"role": "user", "content": "x"}]},
        "timeout_s": 0.2,
    }


@pytest.mark.asyncio
async def test_unknown_endpoint_shadow_counts_and_proceeds():
    svc = _Svc(_Cfg())
    assert svc._flags.get("unknown_endpoint_enforce") is False
    resp = await svc.handle_submit(_submit_body("qwen-composr-typo"), _LoopbackReq())
    # Shadow: behaviour unchanged — the request waits out its (short) deadline
    # and 504s, but the counter recorded the offender for the flip check.
    assert resp.status_code == 504
    tally = svc._unknown_endpoint_submits["qwen-composr-typo"]
    assert tally["count"] == 1
    assert tally["callers"] == {"a": 1}
    status = _json.loads((await svc.handle_status(_LoopbackReq())).body)
    assert "qwen-composr-typo" in status["reliability"]["unknown_endpoint_submits"]


@pytest.mark.asyncio
async def test_unknown_endpoint_enforce_fast_404():
    svc = _Svc(_Cfg())
    svc._flags.set_many({"unknown_endpoint_enforce": True})
    import time as _t
    t0 = _t.monotonic()
    resp = await svc.handle_submit(_submit_body("qwen-composr-typo"), _LoopbackReq())
    assert resp.status_code == 404
    assert _t.monotonic() - t0 < 0.1  # fast-fail, not deadline-rot
    body = _json.loads(resp.body)
    assert body["code"] == "unknown_endpoint"
    assert "known" not in body  # message carries the list, envelope stays lean


@pytest.mark.asyncio
async def test_known_roles_and_aliases_pass_the_gate():
    svc = _Svc(_Cfg())
    svc._flags.set_many({"unknown_endpoint_enforce": True})

    async def ok_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        from originfleet.llmproxy.backend import BackendResponse
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"content": "y"}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            duration_s=0.01, input_tokens=1, output_tokens=1, finish_reason="stop")

    svc._backend.call = ok_call
    await svc.startup()
    try:
        for role in ("bge-m3-embed", "bge-reranker", "llama-thinker",
                     "qwen-composer", "nexus-chat", "gemma-greeter", "chat"):
            body = _submit_body(role)
            body["timeout_s"] = 10.0
            resp = await asyncio.wait_for(
                svc.handle_submit(body, _LoopbackReq()), timeout=10.0)
            assert resp.status_code == 200, f"role {role} blocked by the gate"
        assert svc._unknown_endpoint_submits == {}
    finally:
        await svc.shutdown()
