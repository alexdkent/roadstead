"""Unit tests for the proxy thinking option (request-side enable + generous budget,
response-side deterministic structured-output recovery, fail-safe). Self-contained:
binds the real Correction methods (de-monolith Step 3 moved the thinking logic
from ProxyService to the Correction collaborator) to a lightweight mock `self`
with a `.state` (the Mac 3.9 conftest blocks pytest, so this runs as a plain
script too)."""
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

# Import the modules under test (package import so relative imports resolve).
service = importlib.import_module("roadstead.service")
config = importlib.import_module("roadstead.config")
correction = importlib.import_module("roadstead.correction")
C = correction.Correction

# ``lifecycle.py`` is read as SOURCE by the structural sweep below. The path
# used to be spelled out relative to a monorepo checkout, reaching into the host
# application's tree — the only reason this file could not run standalone. It is
# derived from the imported module now, so it cannot go stale again and cannot
# silently read the wrong tree.
LIFECYCLE_SRC = Path(importlib.import_module("roadstead.lifecycle").__file__)


def _req(payload, *, stream=False, ptype="chat_completion", endpoint="tier3",
         rid="r1", call_site="test"):
    r = types.SimpleNamespace()
    r.payload = payload; r.stream = stream; r.payload_type = ptype
    r.endpoint = endpoint; r.request_id = rid; r.call_site = call_site
    return r


def _mock_self(engine="vllm", thinking_kwargs=("thinking", "enable_thinking")):
    # Correction operates on a shared `self.state`; mimic just the fields the
    # thinking methods touch. The mock IS the Correction `self`.
    # `thinking_kwargs` mirrors models.yaml `policy.thinking_kwargs` — the
    # per-MODEL declaration of which chat-template variable switches reasoning.
    # Default here is the live tier3 (reasoner) declaration.
    state = types.SimpleNamespace()
    ep = types.SimpleNamespace(backend_engine=engine,
                               thinking_kwargs=tuple(thinking_kwargs))
    state.config = types.SimpleNamespace(endpoints={"tier3": ep})
    state.thinking_active = {}
    state.thinking_requests = state.thinking_clean = state.thinking_recovered = 0
    state.thinking_truncated = state.thinking_fallback = state.thinking_noop = 0
    m = types.SimpleNamespace(state=state)
    # bind the real Correction methods to the mock under their public names (so
    # internal cross-calls like self.thinking_allowed_keys resolve) AND under the
    # old underscore-prefixed names the test bodies call.
    for pub, priv in (("apply_thinking", "_apply_thinking"),
                      ("finalize_thinking", "_finalize_thinking"),
                      ("thinking_allowed_keys", "_thinking_allowed_keys"),
                      ("extract_grammar", "_extract_grammar")):
        bound = getattr(C, pub).__get__(m, C)
        setattr(m, pub, bound)
        setattr(m, priv, bound)
    return m


SCHEMA_RF = {"type": "json_schema", "json_schema": {"name": "v", "schema": {
    "type": "object", "properties": {"action": {}, "params": {}, "why": {}},
    "required": ["action", "params", "why"]}}}


def test_apply_thinking_opt_in_vllm():
    m = _mock_self("vllm")
    p = {"messages": [], "max_tokens": 800, "thinking": True, "response_format": SCHEMA_RF}
    m._apply_thinking(_req(p))
    assert p.get("chat_template_kwargs", {}).get("enable_thinking") is True
    assert p["max_tokens"] == 800 + config.thinking_reasoning_budget()  # generous bump
    assert "thinking" not in p  # control field stripped
    assert set(m.state.thinking_active["r1"]["allowed_keys"]) == {"action", "params", "why"}


def test_apply_thinking_transparent_without_optin():
    m = _mock_self("vllm")
    p = {"messages": [], "max_tokens": 800}
    m._apply_thinking(_req(p))
    assert "chat_template_kwargs" not in p
    assert p["max_tokens"] == 800
    assert m.state.thinking_active == {}


def test_apply_thinking_noop_when_the_model_declares_no_switch():
    """The opt-in bails on an UNDECLARED template, not on a non-vLLM engine.

    REGRESSION (2026-08-24): this test used to assert `noop_on_llamacpp`, and
    the production bail really was `if engine != "vllm": return`. That was
    wrong, not merely conservative — both llama.cpp chat endpoints separate
    reasoning into their own response field perfectly well (measured live:
    tier2 2,913 chars, tier2 2,572 chars, `content` clean in
    both), so `thinking: true` was a SILENT no-op on backends that support it.
    What the proxy actually cannot do is guess the switch's NAME, so that is
    what it now refuses on."""
    m = _mock_self("llama.cpp", thinking_kwargs=())
    p = {"messages": [], "max_tokens": 800, "thinking": True}
    m._apply_thinking(_req(p))
    assert "chat_template_kwargs" not in p and p["max_tokens"] == 800
    assert "thinking" not in p and m.state.thinking_active == {}  # still stripped, no-op


def test_apply_thinking_applies_on_a_declared_llamacpp_endpoint():
    """The mirror of the test above, and the behaviour change it protects: a
    llama.cpp endpoint that DECLARES its switch gets the opt-in, with the Qwen
    spelling and NOT DeepSeek's."""
    m = _mock_self("llama.cpp", thinking_kwargs=("enable_thinking",))
    p = {"messages": [], "max_tokens": 800, "thinking": True}
    m._apply_thinking(_req(p))
    ck = p["chat_template_kwargs"]
    assert ck.get("enable_thinking") is True
    assert "thinking" not in ck, (
        "`thinking` is a MEASURED no-op on Qwen templates (0 chars of reasoning "
        "on both tier2 endpoints). Sending it here would be cargo-culted from "
        "DeepSeek and would hide a wrong declaration.")
    assert p["max_tokens"] == 800 + config.thinking_reasoning_budget()


def test_apply_thinking_applies_on_streaming_requests():
    """REGRESSION (2026-08-02): apply_thinking used to `return` on req.stream, so a
    streaming caller's `thinking: true` was stripped and silently ignored — no error,
    just a non-thinking answer. That is the shape the Playground's thinking toggle
    would have shipped as a dead control. The request side is stream-agnostic now."""
    m = _mock_self("vllm")
    p = {"messages": [{"role": "user", "content": "why?"}], "max_tokens": 2048,
         "thinking": True}
    m._apply_thinking(_req(p, stream=True))
    assert p["chat_template_kwargs"]["enable_thinking"] is True
    assert p["max_tokens"] == 2048 + config.thinking_reasoning_budget()
    assert "thinking" not in p  # control field still stripped


def test_apply_thinking_streaming_does_not_register_for_finalize():
    """finalize_thinking rewrites a COMPLETE result dict and never runs over an SSE
    stream, so a streamed request must not be recorded in thinking_active — the
    entry would never be popped and would leak per request."""
    m = _mock_self("vllm")
    p = {"messages": [{"role": "user", "content": "why?"}], "max_tokens": 800,
         "thinking": True, "response_format": SCHEMA_RF}
    m._apply_thinking(_req(p, stream=True))
    assert p["chat_template_kwargs"]["enable_thinking"] is True   # applied...
    assert m.state.thinking_active == {}                          # ...but not registered
    # the sync twin of the same payload DOES register — proves the assertion above
    # is about streaming, not about the opt-in silently failing.
    m2 = _mock_self("vllm")
    p2 = {"messages": [{"role": "user", "content": "why?"}], "max_tokens": 800,
          "thinking": True, "response_format": SCHEMA_RF}
    m2._apply_thinking(_req(p2, stream=False))
    assert set(m2.state.thinking_active["r1"]["allowed_keys"]) == {"action", "params", "why"}


def test_apply_thinking_streaming_transparent_without_optin():
    """Zero blast radius for the streaming traffic that does NOT opt in — which is
    all of it today."""
    m = _mock_self("vllm")
    orig = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    p = {"messages": list(orig), "max_tokens": 800}
    m._apply_thinking(_req(p, stream=True))
    assert "chat_template_kwargs" not in p
    assert p["max_tokens"] == 800 and p["messages"] == orig
    assert m.state.thinking_active == {}


def test_finalize_recovers_brace_dup():
    m = _mock_self("vllm")
    m.state.thinking_active["r1"] = {"allowed_keys": ["action", "params", "why"]}
    bad = '\n\n{{"action": "comment", "params": {"x": 1}, "why": "ok"}'
    result = {"status": "ok", "response": {"choices": [
        {"message": {"content": bad, "reasoning": "...thought..."}, "finish_reason": "stop"}]}}
    m._finalize_thinking(_req({}, rid="r1"), result)
    out = result["response"]["choices"][0]["message"]["content"]
    assert json.loads(out) == {"action": "comment", "params": {"x": 1}, "why": "ok"}
    assert m.state.thinking_recovered == 1 and result["status"] == "ok"


def test_finalize_clean_passthrough():
    m = _mock_self("vllm")
    m.state.thinking_active["r1"] = {"allowed_keys": ["action", "params", "why"]}
    good = '{"action": "upvote", "params": {}, "why": "y"}'
    result = {"status": "ok", "response": {"choices": [
        {"message": {"content": good}, "finish_reason": "stop"}]}}
    m._finalize_thinking(_req({}, rid="r1"), result)
    assert m.state.thinking_clean == 1 and result["response"]["choices"][0]["message"]["content"] == good


def test_finalize_truncation_fails_safe():
    m = _mock_self("vllm")
    m.state.thinking_active["r1"] = {"allowed_keys": ["action", "params", "why"]}
    result = {"status": "ok", "response": {"choices": [
        {"message": {"content": ""}, "finish_reason": "length"}]}}
    m._finalize_thinking(_req({}, rid="r1"), result)
    assert result["status"] == "error" and "response" not in result
    assert m.state.thinking_truncated == 1


def test_finalize_noop_without_optin():
    m = _mock_self("vllm")  # nothing recorded in _thinking_active
    result = {"status": "ok", "response": {"choices": [{"message": {"content": "x"}}]}}
    m._finalize_thinking(_req({}, rid="rX"), result)
    assert result["status"] == "ok" and result["response"]["choices"][0]["message"]["content"] == "x"


def test_finalize_counts_noop_when_no_reasoning():
    """The guard that would have caught the laguna defect: schema-conformant
    content but ZERO reasoning = the opt-in silently did nothing."""
    m = _mock_self("vllm")
    m.state.thinking_active["r1"] = {"allowed_keys": ["action", "params", "why"]}
    good = '{"action": "upvote", "params": {}, "why": "y"}'
    result = {"status": "ok", "response": {"choices": [
        {"message": {"content": good, "reasoning": ""}, "finish_reason": "stop"}]}}
    m._finalize_thinking(_req({}, rid="r1"), result)
    assert m.state.thinking_noop == 1
    assert m.state.thinking_clean == 1  # conformant AND a no-op — the whole point


def test_finalize_counts_noop_when_reasoning_key_absent():
    m = _mock_self("vllm")
    m.state.thinking_active["r1"] = {"allowed_keys": []}
    result = {"status": "ok", "response": {"choices": [
        {"message": {"content": "hello"}, "finish_reason": "stop"}]}}
    m._finalize_thinking(_req({}, rid="r1"), result)
    assert m.state.thinking_noop == 1


def test_finalize_no_noop_when_reasoning_present():
    m = _mock_self("vllm")
    m.state.thinking_active["r1"] = {"allowed_keys": ["action", "params", "why"]}
    good = '{"action": "upvote", "params": {}, "why": "y"}'
    result = {"status": "ok", "response": {"choices": [
        {"message": {"content": good, "reasoning": "let me think..."},
         "finish_reason": "stop"}]}}
    m._finalize_thinking(_req({}, rid="r1"), result)
    assert m.state.thinking_noop == 0 and m.state.thinking_clean == 1


def test_finalize_no_noop_on_reasoning_content_alias():
    """Some vLLM builds emit `reasoning_content` instead of `reasoning`."""
    m = _mock_self("vllm")
    m.state.thinking_active["r1"] = {"allowed_keys": []}
    result = {"status": "ok", "response": {"choices": [
        {"message": {"content": "x", "reasoning_content": "thought"},
         "finish_reason": "stop"}]}}
    m._finalize_thinking(_req({}, rid="r1"), result)
    assert m.state.thinking_noop == 0


def test_finalize_noop_not_counted_without_optin():
    m = _mock_self("vllm")
    result = {"status": "ok", "response": {"choices": [{"message": {"content": "x"}}]}}
    m._finalize_thinking(_req({}, rid="rX"), result)
    assert m.state.thinking_noop == 0 and m.state.thinking_requests == 0


def test_budget_env_override():
    os.environ["ROADSTEAD_PROXY_THINKING_BUDGET"] = "12000"
    assert config.thinking_reasoning_budget() == 12000
    os.environ["ROADSTEAD_PROXY_THINKING"] = "0"
    assert config.thinking_enabled() is False
    os.environ.pop("ROADSTEAD_PROXY_THINKING_BUDGET"); os.environ.pop("ROADSTEAD_PROXY_THINKING")


def _mock_self_forced(reasoning=True, endpoint="tier2"):
    """Mock `self` for apply_forced_reasoning_budget: one endpoint carrying a
    capabilities dict."""
    state = types.SimpleNamespace()
    ep = types.SimpleNamespace(backend_engine="llama.cpp",
                               forces_reasoning=reasoning)
    state.config = types.SimpleNamespace(endpoints={endpoint: ep})
    m = types.SimpleNamespace(state=state)
    m.apply_forced_reasoning_budget = C.apply_forced_reasoning_budget.__get__(m, C)
    return m


def test_forced_reasoning_budget_bumps_small_cap():
    """A forced-reasoning endpoint gets reasoning headroom added to max_tokens so a
    small caller cap (crew turn ~240) can't be eaten by the un-disable-able CoT."""
    m = _mock_self_forced(reasoning=True)
    req = _req({"max_tokens": 240, "messages": []}, endpoint="tier2")
    m.apply_forced_reasoning_budget(req)
    assert req.payload["max_tokens"] == 240 + config.forced_reasoning_budget()


def test_forced_reasoning_budget_noop_when_not_reasoning():
    """An endpoint that does NOT force reasoning is left untouched."""
    m = _mock_self_forced(reasoning=False)
    req = _req({"max_tokens": 240, "messages": []}, endpoint="tier2")
    m.apply_forced_reasoning_budget(req)
    assert req.payload["max_tokens"] == 240


def test_forced_reasoning_budget_noop_without_cap():
    """No positive max_tokens → nothing to protect (model self-limits)."""
    m = _mock_self_forced(reasoning=True)
    req = _req({"messages": []}, endpoint="tier2")
    m.apply_forced_reasoning_budget(req)
    assert "max_tokens" not in req.payload


def test_forced_reasoning_budget_applies_even_to_tiny_cap():
    """A tiny cap (e.g. the dj warm-up ping's max_tokens=1) is NOT exempted: a
    forced-reasoning model spends its first tokens on the CoT, so an un-padded
    1-token cap returns EMPTY → the backend 502s and the warmer logs a false
    failure. Padding makes the warm-up succeed."""
    m = _mock_self_forced(reasoning=True)
    req = _req({"max_tokens": 1, "messages": [{"role": "user", "content": "ok"}]},
               endpoint="tier2")
    m.apply_forced_reasoning_budget(req)
    assert req.payload["max_tokens"] == 1 + config.forced_reasoning_budget()
    # streaming path is covered too (method is stream-agnostic)
    req2 = _req({"max_tokens": 300, "messages": []}, endpoint="tier2", stream=True)
    m.apply_forced_reasoning_budget(req2)
    assert req2.payload["max_tokens"] == 300 + config.forced_reasoning_budget()


def test_forced_reasoning_budget_env_override():
    os.environ["ROADSTEAD_PROXY_FORCED_REASONING_BUDGET"] = "700"
    try:
        assert config.forced_reasoning_budget() == 700
        m = _mock_self_forced(reasoning=True)
        req = _req({"max_tokens": 100, "messages": []}, endpoint="tier2")
        m.apply_forced_reasoning_budget(req)
        assert req.payload["max_tokens"] == 800
    finally:
        os.environ.pop("ROADSTEAD_PROXY_FORCED_REASONING_BUDGET")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)


# ---------------------------------------------------------------------------
# Wiring: apply_thinking must be REACHED on the streaming path
# ---------------------------------------------------------------------------
# A correct apply_thinking that only the sync branch calls is exactly the defect
# this change fixed — the unit tests above would have stayed green through it.
# handle_submit is a long async method over live proxy state (scheduler, futures,
# queue DB, health), so this asserts the CALL SITE structurally instead of
# standing up a fake proxy. Keyed on FUNCTION NAMES, never line numbers.

def _lifecycle_calls(fn_name):
    """Names of `self.correction.<x>(...)` calls made directly inside the named
    method of lifecycle.py (not nested defs)."""
    import ast
    src = LIFECYCLE_SRC.read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
            out = set()
            for sub in ast.walk(node):
                f = getattr(sub, "func", None)
                if (isinstance(f, ast.Attribute)
                        and isinstance(f.value, ast.Attribute)
                        and f.value.attr == "correction"):
                    out.add(f.attr)
            return out
    raise AssertionError(f"lifecycle.py has no method {fn_name!r} — rename? update this test")


def test_apply_thinking_is_called_before_the_stream_branch():
    """It must live in handle_submit (which owns the `if req.stream:` branch), so
    both lanes get it — the same placement apply_forced_reasoning_budget and
    apply_json_object_guard already use, and for the same reason."""
    assert "apply_thinking" in _lifecycle_calls("handle_submit"), (
        "correction.apply_thinking is not called in handle_submit — if it moved back "
        "into handle_sync_submit, `thinking: true` is a SILENT no-op for every "
        "streaming caller (the Playground's thinking toggle among them)")
    assert "apply_thinking" not in _lifecycle_calls("handle_sync_submit"), (
        "apply_thinking is called in BOTH handle_submit and handle_sync_submit — a "
        "sync request would get the reasoning budget added twice")


# ===========================================================================
# The template-KEY contract. Added 2026-08-22 after the tier3 cutover, because
# "thinking works" had only ever been verified by calling vLLM DIRECTLY on
# the backend directly — which bypasses the proxy, and the proxy is what constructs the
# switch. A layer test is not a journey test.
#
# The failure this pins is SILENT: the server pins a default
# (--default-chat-template-kwargs '{"thinking":false}') and the proxy must
# override it. Send the wrong key and the pinned default wins — no error, no
# 4xx, just an answer with no reasoning. Nothing downstream can tell that from
# a model that simply chose not to reason.
# ===========================================================================

def test_thinking_optin_sends_both_template_keys():
    """Laguna's template read `enable_thinking`; DeepSeek-V4-Flash-0731 reads
    `thinking`. BOTH must be sent, or the opt-in silently no-ops on whichever
    backend spells it the other way."""
    m = _mock_self("vllm")
    p = {"messages": [], "max_tokens": 800, "thinking": True}
    m._apply_thinking(_req(p))
    ck = p.get("chat_template_kwargs", {})
    assert ck.get("thinking") is True, (
        "DeepSeek-V4-Flash keys this as `thinking`, and its serve script pins "
        '`{"thinking": false}` as the default — omit it and that default wins, '
        "making the opt-in a SILENT no-op."
    )
    assert ck.get("enable_thinking") is True, (
        "`enable_thinking` is the older vLLM spelling; dropping it would silently "
        "disable the opt-in on any backend still using that key."
    )


def test_thinking_optin_sends_both_template_keys_on_streaming():
    """The Playground is the streaming consumer of this opt-in, so the key
    contract has to hold on the streaming path too."""
    m = _mock_self("vllm")
    p = {"messages": [], "max_tokens": 800, "thinking": True}
    m._apply_thinking(_req(p, stream=True))
    ck = p.get("chat_template_kwargs", {})
    assert ck.get("thinking") is True and ck.get("enable_thinking") is True
    # ...and a streamed request must still NOT register for finalize, or the
    # entry leaks: finalize_thinking cannot run over an SSE stream.
    assert m.state.thinking_active == {}


def test_thinking_keys_absent_without_optin():
    """No opt-in => neither key is set. If `thinking: true` leaked in by default,
    every ordinary tier3 call would spend its budget on reasoning BEFORE emitting
    content — measured empty-content 3/5 at max_tokens=200, and tier3's real mean
    output is 172 tokens."""
    m = _mock_self("vllm")
    p = {"messages": [], "max_tokens": 800}
    m._apply_thinking(_req(p))
    ck = p.get("chat_template_kwargs") or {}
    assert not ck.get("thinking")
    assert not ck.get("enable_thinking")
