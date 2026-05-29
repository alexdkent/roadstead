"""Phase 2 extend-only timeout auto-wire in ProxyLLMClient.

The proxy records, per (model, tier), a recommended timeout. With the auto-wire
on, each call raises its deadline to that recommendation when it needs more than
the caller/default — and never lowers it. This is what stops the flat default
from dropping long thinker-ingestion work mid-flight, with zero added timeout
risk to cells whose recommendation is below the default.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from originfleet.framework.llm_proxy_client import (
    ProxyLLMClient,
    _extend_only,
)


# --- pure extend-only policy ---

def test_extends_up_to_recommendation():
    # recommended above base, below cap → use recommendation
    assert _extend_only(300.0, 706.0, 1800.0) == 706.0


def test_never_lowers_below_base():
    # recommended below base → keep base (the whole point of "extend-only")
    assert _extend_only(300.0, 15.0, 1800.0) == 300.0


def test_caps_runaway_recommendation():
    # recommendation above the cap → clamp to cap
    assert _extend_only(300.0, 9000.0, 1800.0) == 1800.0


def test_high_base_above_cap_is_preserved():
    # a caller that already asked for more than the cap keeps its value
    assert _extend_only(3600.0, 700.0, 1800.0) == 3600.0


def test_missing_recommendation_is_failsafe():
    assert _extend_only(300.0, None, 1800.0) == 300.0
    assert _extend_only(300.0, 0.0, 1800.0) == 300.0


# --- recommendation fetch is fail-safe + parses the proxy shape ---

def _client_with_http(mock_http) -> ProxyLLMClient:
    c = ProxyLLMClient(role="llama-thinker", agent_id="test")
    c._http = mock_http
    return c


def test_recommended_timeout_parses_advice():
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"recommended_timeout_s": 706}
    http = MagicMock()
    http.get.return_value = resp
    c = _client_with_http(http)
    rec = c._recommended_timeout_s(
        "llama-thinker", "P3_INGESTION",
        [{"role": "user", "content": "x" * 4000}], 2000,
    )
    assert rec == 706.0
    # est_in derived from message chars / 4 (~1000), est_out from max_tokens.
    _, kwargs = http.get.call_args
    assert kwargs["params"]["est_out"] == 2000
    assert kwargs["params"]["est_in"] == 1000
    assert kwargs["timeout"] == 2.0  # bounded so the real call never blocks


def test_recommended_timeout_failsafe_on_error():
    http = MagicMock()
    http.get.side_effect = RuntimeError("proxy unreachable")
    c = _client_with_http(http)
    assert c._recommended_timeout_s("llama-thinker", "P1_TURN_SUPPORT", [], 0) is None


def test_recommended_timeout_failsafe_on_non_200():
    resp = MagicMock(status_code=400)
    http = MagicMock()
    http.get.return_value = resp
    c = _client_with_http(http)
    assert c._recommended_timeout_s("bogus-model", "P1_TURN_SUPPORT", [], 0) is None


# --- end-to-end: create() raises the submit + post timeout via the advice ---

def test_create_applies_extend_only(monkeypatch):
    # No-op the identity helper so the test doesn't depend on its internals.
    import originfleet.framework.llm_proxy_client as mod
    monkeypatch.setattr(mod, "resolve_identity", lambda *a, **k: {}, raising=False)

    advice = MagicMock(status_code=200)
    advice.json.return_value = {"recommended_timeout_s": 706}
    submit_resp = MagicMock(status_code=200)
    submit_resp.json.return_value = {
        "status": "ok",
        "response": {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
    }
    http = MagicMock()
    http.get.return_value = advice          # /v1/timeout-advice
    http.post.return_value = submit_resp     # /v1/submit

    c = _client_with_http(http)
    c.chat.completions.create(
        messages=[{"role": "user", "content": "compose"}],
        max_tokens=2000, timeout=300,   # caller asked for 300s
    )

    # The submit (proxy deadline) AND the httpx post() wait both got extended
    # to the recommendation, so the call actually waits long enough.
    _, post_kwargs = http.post.call_args
    assert post_kwargs["json"]["timeout_s"] == 706.0
    assert post_kwargs["timeout"] == 706.0
