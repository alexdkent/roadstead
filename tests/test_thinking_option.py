"""Unit tests for the proxy thinking option (request-side enable + generous budget,
response-side deterministic structured-output recovery, fail-safe). Self-contained:
binds the real LLMProxyService methods to a lightweight mock `self` (the Mac 3.9
conftest blocks pytest, so this runs as a plain script too)."""
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # originfleet/
sys.path.insert(0, str(REPO))

# Import the modules under test (package import so relative imports resolve).
service = importlib.import_module("originfleet.llmproxy.service")
config = importlib.import_module("originfleet.llmproxy.config")
S = service.ProxyService


def _req(payload, *, stream=False, ptype="chat_completion", endpoint="thinker",
         rid="r1", call_site="test"):
    r = types.SimpleNamespace()
    r.payload = payload; r.stream = stream; r.payload_type = ptype
    r.endpoint = endpoint; r.request_id = rid; r.call_site = call_site
    return r


def _mock_self(engine="vllm"):
    m = types.SimpleNamespace()
    ep = types.SimpleNamespace(backend_engine=engine)
    m._config = types.SimpleNamespace(endpoints={"thinker": ep})
    m._thinking_active = {}
    m._thinking_requests = m._thinking_clean = m._thinking_recovered = 0
    m._thinking_truncated = m._thinking_fallback = 0
    # bind the real methods
    for name in ("_apply_thinking", "_finalize_thinking", "_thinking_allowed_keys",
                 "_extract_grammar"):
        setattr(m, name, getattr(S, name).__get__(m, S.__class__))
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
    assert set(m._thinking_active["r1"]["allowed_keys"]) == {"action", "params", "why"}


def test_apply_thinking_transparent_without_optin():
    m = _mock_self("vllm")
    p = {"messages": [], "max_tokens": 800}
    m._apply_thinking(_req(p))
    assert "chat_template_kwargs" not in p
    assert p["max_tokens"] == 800
    assert m._thinking_active == {}


def test_apply_thinking_noop_on_llamacpp():
    m = _mock_self("llama.cpp")
    p = {"messages": [], "max_tokens": 800, "thinking": True}
    m._apply_thinking(_req(p))
    assert "chat_template_kwargs" not in p and p["max_tokens"] == 800
    assert "thinking" not in p and m._thinking_active == {}  # still stripped, no-op


def test_finalize_recovers_brace_dup():
    m = _mock_self("vllm")
    m._thinking_active["r1"] = {"allowed_keys": ["action", "params", "why"]}
    bad = '\n\n{{"action": "comment", "params": {"x": 1}, "why": "ok"}'
    result = {"status": "ok", "response": {"choices": [
        {"message": {"content": bad, "reasoning": "...thought..."}, "finish_reason": "stop"}]}}
    m._finalize_thinking(_req({}, rid="r1"), result)
    out = result["response"]["choices"][0]["message"]["content"]
    assert json.loads(out) == {"action": "comment", "params": {"x": 1}, "why": "ok"}
    assert m._thinking_recovered == 1 and result["status"] == "ok"


def test_finalize_clean_passthrough():
    m = _mock_self("vllm")
    m._thinking_active["r1"] = {"allowed_keys": ["action", "params", "why"]}
    good = '{"action": "upvote", "params": {}, "why": "y"}'
    result = {"status": "ok", "response": {"choices": [
        {"message": {"content": good}, "finish_reason": "stop"}]}}
    m._finalize_thinking(_req({}, rid="r1"), result)
    assert m._thinking_clean == 1 and result["response"]["choices"][0]["message"]["content"] == good


def test_finalize_truncation_fails_safe():
    m = _mock_self("vllm")
    m._thinking_active["r1"] = {"allowed_keys": ["action", "params", "why"]}
    result = {"status": "ok", "response": {"choices": [
        {"message": {"content": ""}, "finish_reason": "length"}]}}
    m._finalize_thinking(_req({}, rid="r1"), result)
    assert result["status"] == "error" and "response" not in result
    assert m._thinking_truncated == 1


def test_finalize_noop_without_optin():
    m = _mock_self("vllm")  # nothing recorded in _thinking_active
    result = {"status": "ok", "response": {"choices": [{"message": {"content": "x"}}]}}
    m._finalize_thinking(_req({}, rid="rX"), result)
    assert result["status"] == "ok" and result["response"]["choices"][0]["message"]["content"] == "x"


def test_budget_env_override():
    os.environ["COLLECTIVE_PROXY_THINKING_BUDGET"] = "12000"
    assert config.thinking_reasoning_budget() == 12000
    os.environ["COLLECTIVE_PROXY_THINKING"] = "0"
    assert config.thinking_enabled() is False
    os.environ.pop("COLLECTIVE_PROXY_THINKING_BUDGET"); os.environ.pop("COLLECTIVE_PROXY_THINKING")


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
