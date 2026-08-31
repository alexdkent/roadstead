"""The thinking switch is a per-MODEL chat-template variable, not a per-engine one.

WHY THIS FILE EXISTS. tier3 swapped Qwen3.6 -> DeepSeek-V4-Flash-0731 on
2026-08-23. The proxy's thinking logic did not follow: it hardcoded Qwen's
`enable_thinking` for every vLLM backend, and `_has_enable_thinking` could only
recognise that one spelling. Fifteen cutover gates passed. Every one of them
asserted that FRAMES existed — none asserted what was IN them, and the substrate
skill's own rule is that a green suite does not mean a capability works.

So these tests assert on CONTENT: which key lands in the payload, with which
value, for which declared family. A test that only checked
`"chat_template_kwargs" in payload` would have passed against the defect.

The live measurements the declarations encode (2026-08-24, through the proxy at
:42161, one probe per cell, `chat_template_kwargs` sent verbatim):

    endpoint       model               `thinking`    `enable_thinking`
    tier3          DeepSeek-V4-Flash   ON  (556 ch)  ON  (518 ch)
    tier2-analyst  Qwen3.8-27B         no-op (0 ch)  ON  (2913 ch)
    tier2-chat     Qwen3.6-35B         no-op (0 ch)  ON  (2572 ch)
"""
import importlib
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(REPO))

backend = importlib.import_module("roadstead.backend")
config = importlib.import_module("roadstead.config")
model_catalog = importlib.import_module("roadstead.model_catalog")
correction = importlib.import_module("roadstead.correction")
C = correction.Correction

_norm = backend._normalize_chat_payload

DEEPSEEK = ("thinking", "enable_thinking")   # tier3 (reasoner)
QWEN = ("enable_thinking",)                  # tier2-analyst / tier2-chat


# ---------------------------------------------------------------------------
# 1. The default injection emits the key the TARGET model actually reads.
# ---------------------------------------------------------------------------

def test_default_off_emits_the_deepseek_key_on_a_deepseek_endpoint():
    """THE BUG. Before this change the proxy emitted `enable_thinking` here
    because that is what Qwen read, and nothing tied the key to the model."""
    out = _norm({"messages": []}, vllm=True, thinking_kwargs=DEEPSEEK)
    ck = out["chat_template_kwargs"]
    assert ck["thinking"] is False, (
        "tier3 runs DeepSeek-V4-Flash, whose template + serve script both spell "
        "the switch `thinking`. Emitting only Qwen's key leaves the proxy's "
        "stated intent riding on the server default agreeing with it by luck.")
    assert ck["enable_thinking"] is False


def test_default_off_emits_only_the_qwen_key_on_a_qwen_endpoint():
    """The mirror. Sending DeepSeek's `thinking` to a Qwen template is a
    measured no-op (0 chars of reasoning on both tier2 endpoints), so shipping
    it would be noise that hides a wrong declaration."""
    out = _norm({"messages": []}, vllm=True, thinking_kwargs=QWEN)
    ck = out["chat_template_kwargs"]
    assert ck == {"enable_thinking": False}, ck


def test_undeclared_endpoint_gets_no_injection_at_all():
    """An endpoint whose template we have not measured gets NOTHING. Guessing a
    key is exactly how the tier3 swap went unnoticed; a missing declaration must
    fail visible-and-inert, not silently drive an unknown switch."""
    out = _norm({"messages": []}, vllm=True, thinking_kwargs=())
    assert "chat_template_kwargs" not in out


def test_llamacpp_is_never_injected_into():
    """Both tier2 backends pin `--chat-template-kwargs '{"enable_thinking":
    false}'` at launch, so OFF is already the server-side default there. The
    family-aware change must not start writing into Qwen-family payloads —
    that would be a live behaviour change to callers nobody asked for."""
    out = _norm({"messages": []}, vllm=False, thinking_kwargs=QWEN)
    assert "chat_template_kwargs" not in out


# ---------------------------------------------------------------------------
# 2. A caller's pin survives — for EITHER spelling.
# ---------------------------------------------------------------------------

def test_a_caller_pinning_the_deepseek_key_is_left_completely_alone():
    """THE SECOND HALF OF THE BUG. `_has_enable_thinking` knew one name, so a
    caller sending `{"thinking": true}` was invisible to it and the proxy
    appended a contradictory `enable_thinking: False`. It survived only because
    V4's template ORs the two names — a template that ANDed them would have
    turned every opt-in into a silent no-op."""
    out = _norm({"messages": [], "chat_template_kwargs": {"thinking": True}},
                vllm=True, thinking_kwargs=DEEPSEEK)
    assert out["chat_template_kwargs"] == {"thinking": True}, (
        "the proxy must not append a contradictory second switch to a payload "
        "the caller has already made up its mind about")


def test_a_caller_pinning_the_qwen_key_is_left_completely_alone():
    out = _norm({"messages": [],
                 "chat_template_kwargs": {"enable_thinking": True}},
                vllm=True, thinking_kwargs=DEEPSEEK)
    assert out["chat_template_kwargs"] == {"enable_thinking": True}


def test_a_caller_pinning_FALSE_is_also_left_alone():
    """A pin is a pin whatever its value — re-asserting a caller's `false` would
    be harmless today and is still wrong: it makes the payload the proxy sends
    differ from the payload the caller wrote."""
    out = _norm({"messages": [], "chat_template_kwargs": {"thinking": False}},
                vllm=True, thinking_kwargs=DEEPSEEK)
    assert out["chat_template_kwargs"] == {"thinking": False}


def test_a_pin_nested_in_extra_body_is_detected():
    """extra_body is merged into the top level further down `_normalize_chat_
    payload`; detection must see the pin BEFORE that merge or the injection
    races it."""
    out = _norm({"messages": [],
                 "extra_body": {"chat_template_kwargs": {"thinking": True}}},
                vllm=True, thinking_kwargs=DEEPSEEK)
    assert out["chat_template_kwargs"] == {"thinking": True}


def test_detection_covers_every_known_spelling_not_just_the_declared_one():
    """Detection is deliberately WIDER than injection. An endpoint declaring
    only Qwen's key must still notice a caller who pinned DeepSeek's — the
    caller may know something the declaration does not, and overriding them is
    the failure this whole change is about."""
    assert backend._THINKING_KWARG_NAMES >= {"thinking", "enable_thinking"}
    for name in backend._THINKING_KWARG_NAMES:
        out = _norm({"messages": [], "chat_template_kwargs": {name: True}},
                    vllm=True, thinking_kwargs=QWEN)
        assert out["chat_template_kwargs"] == {name: True}, name


# ---------------------------------------------------------------------------
# 3. The declaration is really wired from models.yaml (a key missing from
#    model_catalog's mapping is silently dropped — see disable_any_whitespace).
# ---------------------------------------------------------------------------

def test_models_yaml_thinking_kwargs_reach_endpoint_config():
    kwargs = model_catalog.build_endpoint_kwargs()
    by_class = {k: v for k, v in kwargs.items()}
    assert by_class["thinker"]["thinking_kwargs"] == DEEPSEEK, (
        "tier3's declaration did not survive the models.yaml -> EndpointConfig "
        "hop; the proxy would fall back to injecting nothing")
    for cls in ("creative", "tier2-chat"):
        assert by_class[cls]["thinking_kwargs"] == QWEN, cls
    ep = config.EndpointConfig(**by_class["thinker"])
    assert ep.thinking_kwargs == DEEPSEEK


def test_declared_keys_are_all_names_detection_knows():
    """A declaration naming a switch detection cannot see would let the proxy
    inject a key it would then fail to recognise as a caller's pin."""
    for cls, kw in model_catalog.build_endpoint_kwargs().items():
        for key in kw.get("thinking_kwargs", ()):  # noqa: B007
            assert key in backend._THINKING_KWARG_NAMES, (cls, key)


# ---------------------------------------------------------------------------
# 4. The opt-in sets the declared key — and its budget is per-request.
# ---------------------------------------------------------------------------

def _optin_self(thinking_kwargs, engine="vllm", endpoint="thinker"):
    state = types.SimpleNamespace()
    ep = types.SimpleNamespace(backend_engine=engine,
                               thinking_kwargs=tuple(thinking_kwargs))
    state.config = types.SimpleNamespace(endpoints={endpoint: ep})
    state.thinking_active = {}
    m = types.SimpleNamespace(state=state)
    for name in ("thinking_allowed_keys", "apply_thinking", "extract_grammar"):
        setattr(m, name, getattr(C, name).__get__(m, C))
    return m


def _req(payload, endpoint="thinker"):
    return types.SimpleNamespace(
        payload=payload, stream=False, payload_type="chat_completion",
        endpoint=endpoint, request_id="r1", call_site="test")


def test_optin_sets_the_declared_key_for_the_qwen_family():
    m = _optin_self(QWEN, engine="llama.cpp", endpoint="creative")
    p = {"messages": [], "max_tokens": 800, "thinking": True}
    m.apply_thinking(_req(p, endpoint="creative"))
    assert p["chat_template_kwargs"] == {"enable_thinking": True}


def test_optin_int_form_asks_for_a_smaller_reasoning_headroom():
    """THE LEVER THAT UNBLOCKS CHAT. The flat 8000 is right for a long-form
    author and ruinous for a chat turn: the model spends the budget it is given
    (measured — a ~60-token ask produced 5,576 completion tokens over 443.6s
    against 3.0s with thinking off). An interactive caller says how much."""
    m = _optin_self(DEEPSEEK)
    p = {"messages": [], "max_tokens": 1200, "thinking": 600}
    m.apply_thinking(_req(p))
    assert p["max_tokens"] == 1800, (
        "the int form must add exactly the requested headroom, not the flat "
        "default")
    assert p["chat_template_kwargs"]["thinking"] is True


def test_optin_int_form_can_only_ask_for_LESS_than_the_flat_default():
    """A caller must not be able to talk the proxy into a budget bigger than the
    operator-set one — the field is an economy lever, not an escape hatch."""
    m = _optin_self(DEEPSEEK)
    p = {"messages": [], "max_tokens": 1000, "thinking": 999_999}
    m.apply_thinking(_req(p))
    assert p["max_tokens"] == 1000 + config.thinking_reasoning_budget()


def test_optin_true_keeps_the_historic_flat_budget():
    """The Songs lane (`bccff2c45`) ships `thinking: true` on the one-shot song
    author and must be untouched by the int form's arrival."""
    m = _optin_self(DEEPSEEK)
    p = {"messages": [], "max_tokens": 4000, "thinking": True}
    m.apply_thinking(_req(p))
    assert p["max_tokens"] == 4000 + config.thinking_reasoning_budget()
    ck = p["chat_template_kwargs"]
    assert ck["thinking"] is True and ck["enable_thinking"] is True


def test_optin_zero_is_off_not_a_zero_budget():
    m = _optin_self(DEEPSEEK)
    p = {"messages": [], "max_tokens": 800, "thinking": 0}
    m.apply_thinking(_req(p))
    assert "chat_template_kwargs" not in p and p["max_tokens"] == 800


if __name__ == "__main__":   # runnable as a plain script (Mac 3.9 conftest)
    import traceback
    failed = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except Exception:
                failed += 1
                print(f"FAIL  {name}")
                traceback.print_exc()
    print("FAILED" if failed else "all passed")
    sys.exit(1 if failed else 0)
