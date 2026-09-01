"""Regression tests for the system/grammar drop bug.

The proxy migration silently dropped top-level `system` and
`extra_body.grammar` because llama-server ignores both. These tests
lock the backend normalization that fixes it, the cache-key fix that
prevents system-blind cache collisions, and the test-harness A/B
normalization that keeps shadow comparisons valid.

This is the test that would have caught the original bug.
"""

from __future__ import annotations

from roadstead.providers import LLAMACPP, VLLM
from roadstead.providers.payload import (
    _needs_alternation_fix,
    _normalize_strict_alternation,
    _translate_anthropic_image_blocks,
)
from roadstead.coalesce import DeterministicCache


# --- backend normalization ---

def test_vllm_moves_grammar_to_structured_outputs():
    # vLLM ignores top-level `grammar`; it must land in structured_outputs.
    payload = {
        "model": "llama-thinker",
        "messages": [{"role": "user", "content": "extract"}],
        "extra_body": {"grammar": "root ::= object"},
    }
    out = VLLM.prepare_chat_payload(payload)
    assert "grammar" not in out
    assert out["structured_outputs"] == {"grammar": "root ::= object"}


def test_llamacpp_keeps_top_level_grammar():
    # Default (llama.cpp) backend: grammar stays top-level, no structured_outputs.
    payload = {
        "model": "llama-thinker",
        "messages": [{"role": "user", "content": "extract"}],
        "extra_body": {"grammar": "root ::= object"},
    }
    out = LLAMACPP.prepare_chat_payload(payload)
    assert out["grammar"] == "root ::= object"
    assert "structured_outputs" not in out


def test_normalize_inlines_system_into_messages():
    payload = {
        "model": "llama-thinker",
        "system": "You extract entities. Output JSON only.",
        "messages": [{"role": "user", "content": "Alex met Barbara."}],
    }
    out = LLAMACPP.prepare_chat_payload(payload)
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
    out = LLAMACPP.prepare_chat_payload(payload)
    assert "extra_body" not in out
    assert out["grammar"] == "root ::= object"


def test_normalize_handles_both_together():
    payload = {
        "system": "schema rules",
        "messages": [{"role": "user", "content": "content"}],
        "extra_body": {"grammar": "g"},
    }
    out = LLAMACPP.prepare_chat_payload(payload)
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
    out = LLAMACPP.prepare_chat_payload(payload)
    assert out == payload


def test_normalize_does_not_mutate_input():
    payload = {
        "system": "s",
        "messages": [{"role": "user", "content": "x"}],
    }
    LLAMACPP.prepare_chat_payload(payload)
    # original must be unmodified (corpus capture stores req.payload)
    assert payload["system"] == "s"
    assert len(payload["messages"]) == 1


# --- adjacent-system coalesce (composer 122B template rejects >1 system msg) ---
# The Qwen3.5-122B chat template raises Jinja "System message must be at the
# beginning" on more than one system message, 400ing every dj crew turn. The
# llama.cpp path merges adjacent system messages; the vLLM path leaves them.

def test_coalesce_merges_adjacent_system_messages_llamacpp():
    payload = {
        "model": "qwen-composer",
        "messages": [
            {"role": "system", "content": "Crew roster + craft."},
            {"role": "system", "content": "You are Nils."},
            {"role": "user", "content": "tell a joke"},
        ],
    }
    out = LLAMACPP.prepare_chat_payload(payload)
    assert [m["role"] for m in out["messages"]] == ["system", "user"]
    assert out["messages"][0]["content"] == "Crew roster + craft.\n\nYou are Nils."


def test_coalesce_left_untouched_on_vllm():
    payload = {
        "model": "llama-thinker",
        "messages": [
            {"role": "system", "content": "A."},
            {"role": "system", "content": "B."},
            {"role": "user", "content": "x"},
        ],
    }
    out = VLLM.prepare_chat_payload(payload, model_id="llama-thinker")
    # vLLM (thinker) tolerates multiple system messages — leave them intact.
    assert [m["role"] for m in out["messages"]] == ["system", "system", "user"]


def test_coalesce_noop_for_single_system():
    payload = {
        "model": "qwen-composer",
        "messages": [
            {"role": "system", "content": "one"},
            {"role": "user", "content": "x"},
        ],
    }
    out = LLAMACPP.prepare_chat_payload(payload)
    assert out == payload


def test_coalesce_of_inlined_top_level_system():
    # Top-level `system` inlined in front of a messages list that already opens
    # with a system message must collapse to one (both transforms compose).
    payload = {
        "model": "qwen-composer",
        "system": "leading",
        "messages": [
            {"role": "system", "content": "persona"},
            {"role": "user", "content": "x"},
        ],
    }
    out = LLAMACPP.prepare_chat_payload(payload)
    assert "system" not in out
    assert [m["role"] for m in out["messages"]] == ["system", "user"]
    assert out["messages"][0]["content"] == "leading\n\npersona"


def test_coalesce_does_not_mutate_input():
    payload = {
        "model": "qwen-composer",
        "messages": [
            {"role": "system", "content": "A."},
            {"role": "system", "content": "B."},
            {"role": "user", "content": "x"},
        ],
    }
    LLAMACPP.prepare_chat_payload(payload)
    assert len(payload["messages"]) == 3
    assert payload["messages"][0]["content"] == "A."


def test_coalesce_preserves_non_string_system_block():
    # A multimodal/system block-list content must not be string-joined.
    block = {"role": "system", "content": [{"type": "text", "text": "x"}]}
    msgs = [block, {"role": "system", "content": "after"},
            {"role": "user", "content": "u"}]
    out = _normalize_strict_alternation(msgs)
    # The list-content system starts a fresh run; only compatible string runs merge.
    assert out[0]["content"] == [{"type": "text", "text": "x"}]
    assert [m["role"] for m in out] == ["system", "system", "user"]


# --- strict-alternation normalizer (Mistral/Ministral) regression suite ---
# _normalize_strict_alternation + _needs_alternation_fix are the LIVE path
# (the old _coalesce_*/_has_consecutive_system_messages wrappers were deleted
# 2026-07-12, D-1). These lock every shape a strict-alternation template
# rejects so the normalizer can't silently regress (D-2).

def test_needs_alternation_fix_detects_each_bad_shape():
    assert _needs_alternation_fix([
        {"role": "system"}, {"role": "system"}, {"role": "user"}])
    assert _needs_alternation_fix([
        {"role": "user"}, {"role": "user"}])
    assert _needs_alternation_fix([
        {"role": "user"}, {"role": "assistant"}, {"role": "assistant"}])
    # leading assistant (after the optional system) is invalid
    assert _needs_alternation_fix([
        {"role": "system"}, {"role": "assistant"}, {"role": "user"}])
    # already-valid alternation is not flagged
    assert not _needs_alternation_fix([
        {"role": "system"}, {"role": "user"}, {"role": "assistant"},
        {"role": "user"}])
    assert not _needs_alternation_fix([{"role": "user"}])
    assert not _needs_alternation_fix(None)


def test_normalize_coalesces_consecutive_user():
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    ]
    out = _normalize_strict_alternation(msgs)
    assert [m["role"] for m in out] == ["user"]
    assert out[0]["content"] == "first\n\nsecond"


def test_normalize_coalesces_consecutive_assistant():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a1"},
        {"role": "assistant", "content": "a2"},
    ]
    out = _normalize_strict_alternation(msgs)
    assert [m["role"] for m in out] == ["user", "assistant"]
    assert out[1]["content"] == "a1\n\na2"


def test_normalize_drops_leading_assistant():
    # An assistant before the first user turn (orphan history fragment) is
    # dropped; a leading system is kept.
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "orphan"},
        {"role": "user", "content": "u"},
    ]
    out = _normalize_strict_alternation(msgs)
    assert [m["role"] for m in out] == ["system", "user"]
    assert out[0]["content"] == "sys"
    assert out[1]["content"] == "u"


def test_normalize_tool_role_preserves_both_tool_call_ids():
    # Two consecutive tool messages must NOT be merged — each carries its own
    # tool_call_id, and a merge would drop the second (D-3a). BOTH survive.
    msgs = [
        {"role": "user", "content": "call two tools"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_a", "type": "function",
             "function": {"name": "f", "arguments": "{}"}},
            {"id": "call_b", "type": "function",
             "function": {"name": "g", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_a", "content": "result A"},
        {"role": "tool", "tool_call_id": "call_b", "content": "result B"},
    ]
    out = _normalize_strict_alternation(msgs)
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert len(tool_msgs) == 2, "consecutive tool messages must not be coalesced"
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_a", "call_b"]
    assert [m["content"] for m in tool_msgs] == ["result A", "result B"]


def test_normalize_empty_list():
    assert _normalize_strict_alternation([]) == []
    assert _normalize_strict_alternation(None) == []


def test_normalize_single_message():
    msgs = [{"role": "user", "content": "solo"}]
    out = _normalize_strict_alternation(msgs)
    assert out == [{"role": "user", "content": "solo"}]


def test_normalize_all_same_role():
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    out = _normalize_strict_alternation(msgs)
    assert [m["role"] for m in out] == ["user"]
    assert out[0]["content"] == "a\n\nb\n\nc"


def test_normalize_consecutive_user_via_full_payload_llamacpp():
    # End-to-end through _normalize_chat_payload (llama.cpp path): consecutive
    # user turns are coalesced so a Mistral/Ministral template doesn't 500.
    payload = {
        "model": "llama-thinker",
        "messages": [
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
        ],
    }
    out = LLAMACPP.prepare_chat_payload(payload)
    assert [m["role"] for m in out["messages"]] == ["user"]
    assert out["messages"][0]["content"] == "one\n\ntwo"


# --- vLLM thinking default: the switch the TARGET MODEL reads, set to off ---
# 🚨 REWRITTEN 2026-08-24. The header here used to say the thinker was Qwen3.6,
# that it emitted chain-of-thought as prose with no <think> tags, and that no
# reasoning parser was configured. All three were stale: tier3 has been
# DeepSeek-V4-Flash-0731 since 2026-08-23, served with `--reasoning-parser
# deepseek_v4`, and with thinking ON the split is clean (measured: the JSON
# probe returned `{"artist": "Miles Davis", "year": 1959}` in `content` with the
# deliberation in a separate `reasoning` field). The default is OFF because
# reasoning tokens are additive to a `max_tokens` every caller sized for the
# answer alone — not because reasoning corrupts output.
#
# Which KEY is injected is now declared per model (`policy.thinking_kwargs`),
# because the spelling belongs to the chat template: DeepSeek reads `thinking`,
# Qwen reads `enable_thinking`. Full coverage of that:
# tests/llmproxy/test_thinking_kwargs_are_family_aware.py.

def test_vllm_defaults_the_declared_switch_to_false():
    payload = {"model": "llama-thinker", "messages": [{"role": "user", "content": "x"}]}
    out = VLLM.prepare_chat_payload(payload,
                                  thinking_kwargs=("thinking", "enable_thinking"))
    assert out["chat_template_kwargs"]["thinking"] is False
    assert out["chat_template_kwargs"]["enable_thinking"] is False


def test_vllm_injects_nothing_when_the_model_declares_no_switch():
    """No declaration → no guess. An unmeasured template must not have a key
    driven at it on the strength of what the LAST model happened to read."""
    payload = {"model": "llama-thinker", "messages": [{"role": "user", "content": "x"}]}
    out = VLLM.prepare_chat_payload(payload)
    assert "chat_template_kwargs" not in out


def test_llamacpp_does_not_touch_thinking():
    # vLLM-only: a clean llama.cpp payload passes through untouched.
    payload = {"model": "llama-thinker", "messages": [{"role": "user", "content": "x"}]}
    out = LLAMACPP.prepare_chat_payload(payload)
    assert "chat_template_kwargs" not in out
    assert out == payload


def test_vllm_preserves_caller_enable_thinking_top_level():
    payload = {
        "model": "llama-thinker",
        "messages": [{"role": "user", "content": "x"}],
        "chat_template_kwargs": {"enable_thinking": True},
    }
    out = VLLM.prepare_chat_payload(payload,
                                  thinking_kwargs=("thinking", "enable_thinking"))
    assert out["chat_template_kwargs"]["enable_thinking"] is True
    assert "thinking" not in out["chat_template_kwargs"], (
        "a caller's pin must be left ALONE, not have the other family's switch "
        "appended alongside it")


def test_vllm_preserves_caller_enable_thinking_in_extra_body():
    payload = {
        "model": "llama-thinker",
        "messages": [{"role": "user", "content": "x"}],
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }
    out = VLLM.prepare_chat_payload(payload)
    assert out["chat_template_kwargs"]["enable_thinking"] is True
    assert "extra_body" not in out


def test_vllm_thinking_default_does_not_mutate_input():
    payload = {"model": "llama-thinker", "messages": [{"role": "user", "content": "x"}]}
    VLLM.prepare_chat_payload(payload)
    assert "chat_template_kwargs" not in payload  # input untouched (corpus capture)


# --- vision: Anthropic image blocks → OAI image_url ---
# ProxyLLMClient forwards vision messages in Anthropic shape ({type:image,
# source:{type:base64,...}}); both llama.cpp and vLLM reject that with
# "400 unsupported content[].type", silently dropping every image upload's
# description + OCR. The proxy must translate to {type:image_url, image_url:{url}}.

_ANTHROPIC_IMG_MSG = {
    "role": "user",
    "content": [
        {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": "QUJD",
        }},
        {"type": "text", "text": "Analyze the image above."},
    ],
}


def test_normalize_translates_anthropic_image_block():
    payload = {
        "model": "qwen-analyst",
        "system": "You are an image-vision assistant.",
        "messages": [_ANTHROPIC_IMG_MSG],
    }
    out = LLAMACPP.prepare_chat_payload(payload)
    # system inlined first; user message is second
    user = out["messages"][1]
    img = user["content"][0]
    assert img == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,QUJD"},
    }
    # the text block is preserved verbatim
    assert user["content"][1] == {"type": "text", "text": "Analyze the image above."}


def test_normalize_translates_image_without_system():
    # No system/grammar/extra_body — the presence of an image block alone
    # must defeat the early-return guard so the translation still runs.
    payload = {
        "model": "qwen-analyst",
        "messages": [_ANTHROPIC_IMG_MSG],
    }
    out = LLAMACPP.prepare_chat_payload(payload)
    assert out["messages"][0]["content"][0]["type"] == "image_url"
    assert (
        out["messages"][0]["content"][0]["image_url"]["url"]
        == "data:image/png;base64,QUJD"
    )


def test_normalize_vision_does_not_mutate_input():
    payload = {"model": "qwen-analyst", "messages": [_ANTHROPIC_IMG_MSG]}
    LLAMACPP.prepare_chat_payload(payload)
    # original Anthropic block untouched (corpus capture stores req.payload)
    assert payload["messages"][0]["content"][0]["type"] == "image"
    assert payload["messages"][0]["content"][0]["source"]["data"] == "QUJD"


def test_normalize_passthrough_existing_image_url():
    # A payload already in OAI image_url shape carries no Anthropic image
    # block, so it stays wire-correct and passes through untouched.
    payload = {
        "model": "qwen-analyst",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64,ZZZ"}},
                {"type": "text", "text": "hi"},
            ],
        }],
    }
    out = LLAMACPP.prepare_chat_payload(payload)
    assert out == payload


def test_translate_helper_handles_url_source():
    # An Anthropic image block with a url source (not base64) maps to image_url.
    msgs = [{
        "role": "user",
        "content": [{"type": "image", "source": {
            "type": "url", "url": "https://example.com/x.jpg",
        }}],
    }]
    out = _translate_anthropic_image_blocks(msgs)
    assert out[0]["content"][0] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/x.jpg"},
    }


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
    from roadstead import test_harness
    src = inspect.getsource(test_harness.ProxyTestHarness._ab_one)
    assert "prepare_chat_payload" in src, (
        "A/B path must apply the provider's prepare_chat_payload to the "
        "shadow payload"
    )
