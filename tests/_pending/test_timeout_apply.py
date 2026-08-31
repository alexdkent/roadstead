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


# --- shared sub-floor-honor policy (apply_extend_only / floor_for) ---

def test_floor_for_known_and_unknown():
    from originfleet.framework.timeout_advice import floor_for
    assert floor_for("llama-thinker") == 180.0   # normalizes to "thinker"
    assert floor_for("rerank") == 10.0            # isolated small-floor sanity check
    assert floor_for("gemma-greeter") == 60.0     # role → "gemma" class (E4B) since 2026-06-08 (was gemma-hot/8.0)
    assert floor_for("totally-unknown") == 60.0   # default floor


def test_apply_extend_only_honors_sub_floor_budget(monkeypatch):
    # A deliberate best-effort budget below the floor is honored, and the
    # recommendation is never even consulted (no backend over-run).
    import originfleet.framework.timeout_advice as ta
    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        return 9.0
    monkeypatch.setattr(ta, "recommended_timeout", _boom)

    # gemma-hot floor is 8.0; a 0.9s advisory budget stays 0.9s.
    assert ta.apply_extend_only("gemma-hot", "P1_TURN_SUPPORT", 100, 50, 0.9, cap_s=1800.0) == 0.9
    assert called["n"] == 0


def test_apply_extend_only_extends_at_or_above_floor(monkeypatch):
    import originfleet.framework.timeout_advice as ta
    monkeypatch.setattr(ta, "recommended_timeout", lambda *a, **k: 2126.0)
    # thinker floor 180; base 900 (>= floor) extends toward the recommendation,
    # bounded by the cap.
    assert ta.apply_extend_only("thinker", "P3_INGESTION", 8000, 1200, 900.0, cap_s=1800.0) == 1800.0
    # never lowers a high base
    assert ta.apply_extend_only("thinker", "P3_INGESTION", 8000, 1200, 3000.0, cap_s=1800.0) == 3000.0


def test_apply_extend_only_failsafe_on_no_recommendation(monkeypatch):
    import originfleet.framework.timeout_advice as ta
    monkeypatch.setattr(ta, "recommended_timeout", lambda *a, **k: 0.0)
    assert ta.apply_extend_only("thinker", "P3_INGESTION", 100, 100, 300.0, cap_s=1800.0) == 300.0


def test_create_honors_sub_floor_best_effort_budget(monkeypatch):
    # A best-effort advisory (timeout below the model floor) is NOT extended,
    # so the backend doesn't keep running work the caller abandons.
    import originfleet.framework.llm_proxy_client as mod
    monkeypatch.setattr(mod, "resolve_identity", lambda *a, **k: {}, raising=False)

    submit_resp = MagicMock(status_code=200)
    submit_resp.json.return_value = {
        "status": "ok",
        "response": {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
    }
    http = MagicMock()
    http.post.return_value = submit_resp

    c = ProxyLLMClient(role="gemma-greeter", agent_id="test")  # floor 8.0
    c._http = http
    c.chat.completions.create(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=64, timeout=0.9,   # deliberate sub-floor budget
    )
    _, post_kwargs = http.post.call_args
    assert post_kwargs["json"]["timeout_s"] == 0.9   # honored, not extended
    # advice GET must not have been consulted
    assert http.get.call_count == 0


# --- the llm_qos direct-submit path applies extend-only (the bypass fix) ---

def test_qos_direct_submit_applies_extend_only(monkeypatch):
    import originfleet.framework.llm_qos as qos
    import originfleet.framework.timeout_advice as ta
    monkeypatch.setattr(ta, "recommended_timeout", lambda *a, **k: 2126.0)
    monkeypatch.setattr(qos, "_APPLY_TIMEOUT_ADVICE", True, raising=False)
    monkeypatch.setattr(qos, "_TIMEOUT_EXTEND_CAP_S", 1800.0, raising=False)

    eff = qos._resolve_qos_timeout(
        "thinker", "P3_INGESTION",
        {"messages": [{"role": "user", "content": "x" * 4000}], "max_tokens": 1200},
        None,   # caller passed no timeout → must extend, not fall back to 180
    )
    assert eff == 1800.0   # was the artificial 180s before the fix


def test_qos_direct_submit_honors_sub_floor(monkeypatch):
    import originfleet.framework.llm_qos as qos
    import originfleet.framework.timeout_advice as ta
    monkeypatch.setattr(ta, "recommended_timeout", lambda *a, **k: 8.0)
    monkeypatch.setattr(qos, "_APPLY_TIMEOUT_ADVICE", True, raising=False)

    eff = qos._resolve_qos_timeout(
        "gemma-hot", "P1_TURN_SUPPORT",
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64},
        0.9,
    )
    assert eff == 0.9


# --- Phase 0 regression: inherited default vs explicit sub-floor deadline ---
# (regression_ledger: llmproxy-subfloor-default-cliff, 2026-07-05)

def _creative_client(http) -> ProxyLLMClient:
    c = ProxyLLMClient(role="creative", agent_id="test")  # class floor 900s
    c._http = http
    return c


def test_inherited_default_below_floor_is_extended_not_honored(monkeypatch):
    """The client's 300s CONSTRUCTOR DEFAULT sits below creative's 900s floor.
    It must NOT be treated as a deliberate sub-floor budget — extend-only runs
    and lifts the deadline to the recommendation, instead of pinning the call at
    300s (the bug that killed 36% of timeouts, all premature)."""
    import originfleet.framework.llm_proxy_client as mod
    monkeypatch.setattr(mod, "resolve_identity", lambda *a, **k: {}, raising=False)

    advice = MagicMock(status_code=200)
    advice.json.return_value = {"recommended_timeout_s": 900}
    submit_resp = MagicMock(status_code=200)
    submit_resp.json.return_value = {
        "status": "ok",
        "response": {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
    }
    http = MagicMock()
    http.get.return_value = advice
    http.post.return_value = submit_resp

    c = _creative_client(http)
    # NO explicit timeout= → base falls to the 300s constructor default.
    c.chat.completions.create(messages=[{"role": "user", "content": "compose"}], max_tokens=2000)

    _, post_kwargs = http.post.call_args
    assert post_kwargs["json"]["timeout_s"] == 900.0   # extended, NOT pinned at 300
    assert post_kwargs["timeout"] == 900.0
    assert http.get.called   # the recommendation WAS consulted


def test_explicit_sub_floor_deadline_is_still_honored(monkeypatch):
    """A caller that EXPLICITLY passes a tight deadline below the floor still
    gets it honored (deliberate best-effort budget) — and the recommendation is
    not even consulted, so no backend over-run. The fix narrows sub-floor honor
    to explicit deadlines only; it must not break the explicit case."""
    import originfleet.framework.llm_proxy_client as mod
    monkeypatch.setattr(mod, "resolve_identity", lambda *a, **k: {}, raising=False)

    submit_resp = MagicMock(status_code=200)
    submit_resp.json.return_value = {
        "status": "ok",
        "response": {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
    }
    http = MagicMock()
    http.post.return_value = submit_resp

    c = _creative_client(http)
    c.chat.completions.create(
        messages=[{"role": "user", "content": "quick"}], max_tokens=50, timeout=5,
    )

    _, post_kwargs = http.post.call_args
    assert post_kwargs["json"]["timeout_s"] == 5.0   # honored
    assert not http.get.called   # advice never fetched for an explicit sub-floor budget


# --- keepalive-disconnect resilience (2026-07-06) ---
# LEDGER: llmproxy-keepalive-disconnect

import httpx  # noqa: E402
import pytest  # noqa: E402

from originfleet.framework.nexus_errors import is_deferrable_llm_error  # noqa: E402
from originfleet.framework.llm_proxy_client import _CLIENT_KEEPALIVE_EXPIRY_S  # noqa: E402


def test_keepalive_invariant_client_below_server():
    """The client must retire idle connections BEFORE the proxy does, so a POST
    never reuses a server-closed keepalive socket (the RemoteProtocolError race).
    Pins client keepalive_expiry < server timeout_keep_alive with real margin."""
    from roadstead.__main__ import PROXY_SERVER_KEEPALIVE_S
    assert _CLIENT_KEEPALIVE_EXPIRY_S < PROXY_SERVER_KEEPALIVE_S
    # margin must exceed plausible clock/RTT jitter, not just be positive
    assert PROXY_SERVER_KEEPALIVE_S - _CLIENT_KEEPALIVE_EXPIRY_S >= 5.0


def test_client_keepalive_expiry_applied_to_pool():
    c = ProxyLLMClient(role="gemma-router", agent_id="test")
    assert c._http._transport._pool._keepalive_expiry == _CLIENT_KEEPALIVE_EXPIRY_S


@pytest.mark.parametrize("exc", [
    httpx.RemoteProtocolError("Server disconnected without sending a response."),
    httpx.ReadError("connection reset"),
    httpx.WriteError("broken pipe"),
])
def test_midflight_disconnect_normalized_to_deferrable(monkeypatch, exc):
    """A mid-flight transport drop on the submit POST must surface as a DEFERRABLE
    ConnectionError (not a raw httpx error) so every caller defers + retries."""
    import originfleet.framework.llm_proxy_client as mod
    monkeypatch.setattr(mod, "resolve_identity", lambda *a, **k: {}, raising=False)
    http = MagicMock()
    http.post.side_effect = exc
    c = _client_with_http(http)
    with pytest.raises(ConnectionError) as ei:
        c.chat.completions.create(messages=[{"role": "user", "content": "x"}], max_tokens=10)
    assert "connection lost" in str(ei.value).lower()
    # and the canonical fleet classifier treats it as deferrable (embeds orig text)
    assert is_deferrable_llm_error(ei.value)
