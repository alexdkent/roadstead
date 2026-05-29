"""Regression tests for the system/grammar drop bug.

The proxy migration silently dropped top-level `system` and
`extra_body.grammar` because llama-server ignores both. These tests
lock the backend normalization that fixes it, the cache-key fix that
prevents system-blind cache collisions, and the test-harness A/B
normalization that keeps shadow comparisons valid.

This is the test that would have caught the original bug.
"""

from __future__ import annotations

from originfleet.llmproxy.backend import _normalize_chat_payload
from originfleet.llmproxy.coalesce import DeterministicCache


# --- backend normalization ---

def test_vllm_moves_grammar_to_structured_outputs():
    # vLLM ignores top-level `grammar`; it must land in structured_outputs.
    payload = {
        "model": "llama-thinker",
        "messages": [{"role": "user", "content": "extract"}],
        "extra_body": {"grammar": "root ::= object"},
    }
    out = _normalize_chat_payload(payload, vllm=True)
    assert "grammar" not in out
    assert out["structured_outputs"] == {"grammar": "root ::= object"}


def test_llamacpp_keeps_top_level_grammar():
    # Default (llama.cpp) backend: grammar stays top-level, no structured_outputs.
    payload = {
        "model": "llama-thinker",
        "messages": [{"role": "user", "content": "extract"}],
        "extra_body": {"grammar": "root ::= object"},
    }
    out = _normalize_chat_payload(payload)  # vllm=False
    assert out["grammar"] == "root ::= object"
    assert "structured_outputs" not in out


def test_normalize_inlines_system_into_messages():
    payload = {
        "model": "llama-thinker",
        "system": "You extract entities. Output JSON only.",
        "messages": [{"role": "user", "content": "Alex met Barbara."}],
    }
    out = _normalize_chat_payload(payload)
    assert "system" not in out
    assert out["messages"][0] == {
        "role": "system",
        "content": "You extract entities. Output JSON only.",
    }
    assert out["messages"][1]["role"] == "user"


def test_normalize_lifts_grammar_from_extra_body():
    payload = {
        "model": "llama-thinker",
        "messages": [{"role": "user", "content": "x"}],
        "extra_body": {"grammar": "root ::= object"},
    }
    out = _normalize_chat_payload(payload)
    assert "extra_body" not in out
    assert out["grammar"] == "root ::= object"


def test_normalize_handles_both_together():
    payload = {
        "system": "schema rules",
        "messages": [{"role": "user", "content": "content"}],
        "extra_body": {"grammar": "g"},
    }
    out = _normalize_chat_payload(payload)
    assert out["messages"][0]["content"] == "schema rules"
    assert out["grammar"] == "g"
    assert "system" not in out
    assert "extra_body" not in out


def test_normalize_noop_for_clean_payload():
    """A payload that's already wire-correct passes through untouched —
    this is the guard that makes the fix safe for working call sites."""
    payload = {
        "model": "llama-thinker",
        "messages": [
            {"role": "system", "content": "already here"},
            {"role": "user", "content": "x"},
        ],
        "grammar": "already top-level",
    }
    out = _normalize_chat_payload(payload)
    assert out == payload


def test_normalize_does_not_mutate_input():
    payload = {
        "system": "s",
        "messages": [{"role": "user", "content": "x"}],
    }
    _normalize_chat_payload(payload)
    # original must be unmodified (corpus capture stores req.payload)
    assert payload["system"] == "s"
    assert len(payload["messages"]) == 1


# --- vLLM thinking-leak fix: default chat_template_kwargs.enable_thinking off ---
# The thinker (Qwen3.6) emits chain-of-thought as prose with no <think> tags and
# no reasoning parser, so thinking-on leaks the "Here's a thinking process: ..."
# preamble into message.content and breaks structured parsers downstream.

def test_vllm_defaults_enable_thinking_false():
    payload = {"model": "llama-thinker", "messages": [{"role": "user", "content": "x"}]}
    out = _normalize_chat_payload(payload, vllm=True)
    assert out["chat_template_kwargs"]["enable_thinking"] is False


def test_llamacpp_does_not_touch_thinking():
    # vLLM-only: a clean llama.cpp payload passes through untouched.
    payload = {"model": "llama-thinker", "messages": [{"role": "user", "content": "x"}]}
    out = _normalize_chat_payload(payload)  # vllm=False
    assert "chat_template_kwargs" not in out
    assert out == payload


def test_vllm_preserves_caller_enable_thinking_top_level():
    payload = {
        "model": "llama-thinker",
        "messages": [{"role": "user", "content": "x"}],
        "chat_template_kwargs": {"enable_thinking": True},
    }
    out = _normalize_chat_payload(payload, vllm=True)
    assert out["chat_template_kwargs"]["enable_thinking"] is True


def test_vllm_preserves_caller_enable_thinking_in_extra_body():
    payload = {
        "model": "llama-thinker",
        "messages": [{"role": "user", "content": "x"}],
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }
    out = _normalize_chat_payload(payload, vllm=True)
    assert out["chat_template_kwargs"]["enable_thinking"] is True
    assert "extra_body" not in out


def test_vllm_thinking_default_does_not_mutate_input():
    payload = {"model": "llama-thinker", "messages": [{"role": "user", "content": "x"}]}
    _normalize_chat_payload(payload, vllm=True)
    assert "chat_template_kwargs" not in payload  # input untouched (corpus capture)


# --- cache key must reflect system (cache-poisoning fix) ---

def test_cache_key_differs_when_system_differs():
    cache = DeterministicCache()
    base_msgs = [{"role": "user", "content": "extract from this"}]
    k1 = cache.cache_key("thinker", {
        "temperature": 0, "messages": base_msgs,
        "system": "schema A", "max_tokens": 256,
    })
    k2 = cache.cache_key("thinker", {
        "temperature": 0, "messages": base_msgs,
        "system": "schema B", "max_tokens": 256,
    })
    assert k1 is not None and k2 is not None
    assert k1 != k2, "different system prompts must not collide in cache"


def test_cache_key_same_when_system_same():
    cache = DeterministicCache()
    payload = {
        "temperature": 0,
        "messages": [{"role": "user", "content": "x"}],
        "system": "same",
        "max_tokens": 256,
    }
    assert cache.cache_key("thinker", payload) == cache.cache_key("thinker", payload)


# --- test harness A/B applies normalization ---

def test_ab_harness_normalizes_before_shadow_dispatch():
    """The A/B path must normalize the corpus payload so the shadow
    backend gets system+grammar, matching what the proxy sends primary.
    Without this, shadow comparisons are invalid."""
    import inspect
    from originfleet.llmproxy import test_harness
    src = inspect.getsource(test_harness.ProxyTestHarness._ab_one)
    assert "_normalize_chat_payload" in src, (
        "A/B path must apply _normalize_chat_payload to the shadow payload"
    )
