"""Unit tests for the egress degeneration guard
(`ProxyService._maybe_correct_degenerate` + `_is_degenerate_text`).

A repetition-LOOP response ("… the song of ## … the song of ## …") is a 200 with
syntactically-valid content, so the transient-error retry and the grammar egress
check both miss it. The guard detects it and re-dispatches with an anti-repetition
penalty. Invariants pinned here:
  - the detector flags a real loop but NOT a normal chorus / short reply / varied text;
  - on a degenerate response the guard re-dispatches (with frequency_penalty) and
    swaps in the first clean result, tallying recovered;
  - a clean response is a no-op (backend never re-called);
  - shadow mode detects-only (no re-dispatch); the kill-switch disables it entirely;
  - an unrecoverable loop is left untouched + flagged (never cached), fail-open.

Self-contained (binds the real method to a mock `self`); runs under pytest OR as a
plain script.
"""
import asyncio
import importlib
import os
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # originfleet/
sys.path.insert(0, str(REPO))

service = importlib.import_module("roadstead.service")
correction = importlib.import_module("roadstead.correction")
C = correction.Correction  # degeneration guard moved here in de-monolith Step 3

# A repetition loop: the same 6-word shingle dominates the whole output.
DEGEN = "the song of the night and " * 30
# A normal song: a chorus repeats a few times but is a tiny fraction of varied text.
CHORUS = "oh carry me home through the rain\n"
CLEAN = (
    "Salt crusted stone against the restless tide\n"
    "A solitary eye that never sleeps\n" + CHORUS +
    "Glass and neon in a slow parade\n"
    "Memories the passing years remade\n" + CHORUS +
    "Somewhere out there a stranger hears\n"
    "The same old melody across the years\n" + CHORUS
)


# ---- detector ----
def test_detects_repetition_loop():
    assert service._is_degenerate_text(DEGEN)


def test_passes_normal_chorus():
    assert not service._is_degenerate_text(CLEAN)


def test_passes_short_and_varied():
    assert not service._is_degenerate_text("Two short sentences. Nothing repeats here at all.")
    assert not service._is_degenerate_text(
        " ".join(f"distinct word number {i} here now" for i in range(40)))


def test_top_shingle_reps():
    reps, total = service._top_shingle_reps(DEGEN)
    assert reps >= 6 and total > 0


# ---- guard (async) ----
def _result(content, status="ok"):
    return {"status": status,
            "response": {"choices": [{"message": {"content": content}}]}}


def _req():
    r = types.SimpleNamespace()
    r.payload_type = "chat_completion"
    r.payload = {"messages": [], "max_tokens": 500, "temperature": 0.85}
    r.endpoint = "thinker"
    r.request_id = "rid-1"
    r.call_site = "sidekick.craft_song"
    r.timeout_deadline = time.monotonic() + 120
    # Phase 5 accounting fields (metrics sample + corrected-row persist).
    r.agent_id = "sidekick"
    r.priority = importlib.import_module(
        "roadstead.config").LLMPriority.P2_POST_TURN
    r.session_id = None
    r.turn_id = None
    r.caller_id = None
    return r


def _mock_self(backend_call):
    # Correction.maybe_correct_degenerate operates on self.state.* — mirror the
    # fields it touches.
    state = types.SimpleNamespace()
    state.degeneration_detected = 0
    state.degeneration_recovered = 0
    state.degeneration_unrecovered = 0
    state.degeneration_by_call_site = {}
    state.degen_redispatch_inflight = 0
    state.metrics = types.SimpleNamespace(record=lambda sample: None)
    m = types.SimpleNamespace(state=state)
    m._persisted = []
    state.queue_db = types.SimpleNamespace(
        persist_complete=lambda *a, **k: m._persisted.append((a, k)))
    ep = types.SimpleNamespace(role="thinker")
    state.config = types.SimpleNamespace(endpoints={"thinker": ep})
    state.backend = types.SimpleNamespace(call=backend_call)
    m._maybe_correct_degenerate = C.maybe_correct_degenerate.__get__(m, C)
    return m


def _backend_returning(*bodies):
    """An async backend.call that yields the given response bodies in order."""
    seq = list(bodies)
    calls = []

    async def _call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        calls.append(payload)
        body = seq.pop(0) if seq else {"choices": [{"message": {"content": DEGEN}}]}
        return types.SimpleNamespace(
            body=body, duration_s=0.4, input_tokens=10, output_tokens=20,
            finish_reason="stop")

    _call.calls = calls
    return _call


def test_degenerate_is_recovered_by_redispatch():
    clean_body = {"choices": [{"message": {"content": CLEAN}}]}
    backend = _backend_returning(clean_body)
    m = _mock_self(backend)
    res = _result(DEGEN)
    asyncio.run(m._maybe_correct_degenerate(_req(), res))
    assert m.state.degeneration_detected == 1
    assert m.state.degeneration_recovered == 1
    assert res["response"] is clean_body                      # swapped in
    assert backend.calls and "frequency_penalty" in backend.calls[0]


def test_clean_response_is_noop():
    backend = _backend_returning()  # would raise if popped wrongly; must NOT be called
    m = _mock_self(backend)
    res = _result(CLEAN)
    asyncio.run(m._maybe_correct_degenerate(_req(), res))
    assert m.state.degeneration_detected == 0
    assert backend.calls == []                                # backend never re-called


def test_shadow_mode_detects_only():
    backend = _backend_returning()
    m = _mock_self(backend)
    res = _result(DEGEN)
    os.environ["COLLECTIVE_PROXY_DEGENERATION_SHADOW"] = "1"
    try:
        asyncio.run(m._maybe_correct_degenerate(_req(), res))
    finally:
        os.environ.pop("COLLECTIVE_PROXY_DEGENERATION_SHADOW", None)
    assert m.state.degeneration_detected == 1
    assert m.state.degeneration_recovered == 0
    assert backend.calls == []                                # no re-dispatch in shadow
    assert res["response"]["choices"][0]["message"]["content"] == DEGEN


def test_kill_switch_disables():
    backend = _backend_returning()
    m = _mock_self(backend)
    res = _result(DEGEN)
    os.environ["COLLECTIVE_PROXY_DEGENERATION_GUARD"] = "off"
    try:
        asyncio.run(m._maybe_correct_degenerate(_req(), res))
    finally:
        os.environ.pop("COLLECTIVE_PROXY_DEGENERATION_GUARD", None)
    assert m.state.degeneration_detected == 0
    assert backend.calls == []


def test_unrecoverable_is_flagged_not_mutated():
    # Both re-dispatches still come back degenerate → leave original + flag.
    d1 = {"choices": [{"message": {"content": DEGEN}}]}
    d2 = {"choices": [{"message": {"content": DEGEN}}]}
    backend = _backend_returning(d1, d2)
    m = _mock_self(backend)
    res = _result(DEGEN)
    asyncio.run(m._maybe_correct_degenerate(_req(), res))
    assert m.state.degeneration_detected == 1
    assert m.state.degeneration_recovered == 0
    assert m.state.degeneration_unrecovered == 1
    assert res.get("_degenerate_unrecovered") is True
    assert res["response"]["choices"][0]["message"]["content"] == DEGEN  # untouched


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"ok {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")


# ---- Phase 5: re-dispatch accounting ----------------------------------------

def test_redispatch_concurrency_guard_fails_open():
    backend = _backend_returning()
    m = _mock_self(backend)
    m.state.degen_redispatch_inflight = 2  # two already running fleet-wide
    res = _result(DEGEN)
    asyncio.run(m._maybe_correct_degenerate(_req(), res))
    assert backend.calls == []                       # no third re-dispatch
    assert res["_degenerate_unrecovered"] is True    # fail-open, never cached
    assert m.state.degeneration_unrecovered == 1
    assert m.state.degen_redispatch_inflight == 2         # untouched


def test_recovery_persists_corrected_row_same_request_id():
    clean_body = {"choices": [{"message": {"content": CLEAN}}]}
    backend = _backend_returning(clean_body)
    m = _mock_self(backend)
    res = _result(DEGEN)
    asyncio.run(m._maybe_correct_degenerate(_req(), res))
    assert m.state.degeneration_recovered == 1
    assert len(m._persisted) == 1
    args, kwargs = m._persisted[0]
    assert args[0] == "rid-1"                        # SAME request_id → row replaced
    assert kwargs["response"] is clean_body          # the corrected body, not the garbage
    assert args[5] == 10 and args[6] == 20           # re-dispatch token counts
    assert m.state.degen_redispatch_inflight == 0         # released


def test_redispatch_metrics_sample_emitted():
    samples = []
    clean_body = {"choices": [{"message": {"content": CLEAN}}]}
    backend = _backend_returning(clean_body)
    m = _mock_self(backend)
    m.state.metrics = types.SimpleNamespace(record=samples.append)
    asyncio.run(m._maybe_correct_degenerate(_req(), _result(DEGEN)))
    assert len(samples) == 1
    assert samples[0].status == "degen_retry"
    assert samples[0].endpoint == "thinker"
