"""Model-authoritative routing + vLLM model-field normalization.

The proxy used to route purely by the caller's client role and forward the
payload's `model` unchanged — so a thinker-role client asking for
`model="tier2"` landed on the vLLM tier3 backend, which 404s on a
model name it doesn't serve. These tests pin the fix: the requested model
selects the endpoint, and vLLM backends receive their served model id.
"""

from __future__ import annotations

import logging

import pytest

from roadstead.providers import LLAMACPP, VLLM
from roadstead.config import EndpointConfig, ProxyConfig
from roadstead.scheduler import QueuedRequest
from roadstead.service import ProxyService


def _svc() -> ProxyService:
    return ProxyService(ProxyConfig())


# ----- _resolve_endpoint -----

def test_requested_model_wins_when_known_and_different():
    svc = _svc()
    ep = svc._resolve_endpoint({
        "endpoint": "tier3",  # tier3 client
        "payload_type": "chat_completion",
        "payload": {"model": "tier2"},  # but asks for tier2
    })
    # tier2 (30B) fully decommissioned 2026-07-03; after the 2026-07-11
    # boxa one-model consolidation its legacy alias (like classify/analyst/vision)
    # resolves to the `tier2` endpoint, not a distinct endpoint.
    assert ep == "tier2"  # routed to tier2, not tier3


def test_no_override_when_model_matches_submit_endpoint():
    svc = _svc()
    ep = svc._resolve_endpoint({
        "endpoint": "classify",
        "payload_type": "chat_completion",
        # both `classify` (submit endpoint) and `tier2` (model) alias to
        # `tier2` post-2026-07-11 consolidation → they match, no override.
        "payload": {"model": "tier2"},
    })
    assert ep == "tier2"


def test_falls_back_when_model_absent():
    svc = _svc()
    ep = svc._resolve_endpoint({
        "endpoint": "tier3",
        "payload_type": "chat_completion",
        "payload": {"messages": []},
    })
    assert ep == "tier3"


def test_falls_back_when_model_unknown():
    svc = _svc()
    ep = svc._resolve_endpoint({
        "endpoint": "tier3",
        "payload_type": "chat_completion",
        "payload": {"model": "gpt-4-turbo"},  # not a known endpoint
    })
    assert ep == "tier3"


def test_embeddings_and_rerank_untouched():
    svc = _svc()
    # embeddings: non-chat payload_type short-circuits
    assert svc._resolve_endpoint({
        "endpoint": "embed", "payload_type": "embedding",
        "payload": {"texts": ["x"]},
    }) == "embed"
    # rerank: model='bge' is not a known endpoint even if it were chat
    assert svc._resolve_endpoint({
        "endpoint": "rerank", "payload_type": "rerank",
        "payload": {"model": "bge"},
    }) == "rerank"


def test_reconcile_logs_loudly(caplog):
    svc = _svc()
    with caplog.at_level(logging.WARNING):
        svc._resolve_endpoint({
            "endpoint": "tier3",
            "payload_type": "chat_completion",
            "payload": {"model": "tier2"},
            "caller_id": "orchestrator/orchestrator.autonomous_chat-agent.reflect/s1",
            "call_site": "orchestrator.autonomous_chat-agent.reflect",
        })
    assert "route reconcile" in caplog.text
    assert "autonomous_chat-agent.reflect" in caplog.text


def test_resolved_endpoint_flows_into_queued_request():
    svc = _svc()
    body = {
        "endpoint": "tier3",
        "payload_type": "chat_completion",
        "payload": {"model": "tier2", "messages": []},
    }
    req = QueuedRequest.create(
        agent_id="notifier", endpoint=svc._resolve_endpoint(body),
        priority="P4_HYGIENE", call_site="orchestrator.autonomous_chat-agent.reflect",
        payload_type="chat_completion", payload=body["payload"],
    )
    assert req.endpoint == "tier2"


# ----- effective_model_id -----

def test_effective_model_id_falls_back_to_role():
    ep = EndpointConfig(endpoint_class="tier3", role="tier3")
    assert ep.effective_model_id == "tier3"
    ep.served_model_id = "qwen3.6-27b-nvfp4"
    assert ep.effective_model_id == "qwen3.6-27b-nvfp4"


# ----- vLLM model-field normalization -----

def test_vllm_forces_served_model_id():
    out = VLLM.prepare_chat_payload(
        {"model": "tier2", "messages": [{"role": "user", "content": "hi"}]},
        model_id="tier3")
    assert out["model"] == "tier3"


def test_llamacpp_model_left_untouched():
    payload = {"model": "tier2", "messages": [{"role": "user", "content": "hi"}]}
    out = LLAMACPP.prepare_chat_payload(payload, model_id="tier3")
    assert out["model"] == "tier2"  # llama.cpp ignores it; we don't touch it


def test_vllm_noop_when_model_already_correct():
    payload = {"model": "tier3", "messages": [{"role": "user", "content": "hi"}]}
    out = VLLM.prepare_chat_payload(payload, model_id="tier3")
    assert out["model"] == "tier3"
