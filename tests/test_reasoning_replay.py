"""Reasoning replay (roadstead/reasoning_replay.py + Correction hooks).

Covers the module's own key/store logic in isolation, then the two
``Correction`` methods (``apply_reasoning_replay_restore`` /
``store_reasoning_replay``) through the same lightweight mock-``self``
pattern ``tests/test_thinking_option.py`` uses: bind the real bound methods
to a ``types.SimpleNamespace`` carrying just the ``.state`` fields they read.
"""
from __future__ import annotations

import importlib
import types

import pytest

reasoning_replay = importlib.import_module("roadstead.reasoning_replay")
correction = importlib.import_module("roadstead.correction")
model_catalog = importlib.import_module("roadstead.model_catalog")

ReasoningReplayStore = reasoning_replay.ReasoningReplayStore
replay_key = reasoning_replay.replay_key
accumulate_tool_call_deltas = reasoning_replay.accumulate_tool_call_deltas
C = correction.Correction


# --------------------------------------------------------------------------
# replay_key / normalization
# --------------------------------------------------------------------------

def test_key_is_stable_for_identical_input():
    msgs = [{"role": "user", "content": "hi"}]
    k1 = replay_key(msgs, None, "hello", None)
    k2 = replay_key(msgs, None, "hello", None)
    assert k1 == k2


def test_a_request_this_proxy_already_enriched_hashes_the_same():
    """The store's own normalization: an assistant history message a PRIOR
    restore attached `reasoning_content` to must hash identically to the
    client's original bytes — otherwise store/restore could never agree with
    themselves past the first turn of a conversation."""
    plain = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "and then?"},
    ]
    enriched = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "reasoning_content": "because..."},
        {"role": "user", "content": "and then?"},
    ]
    assert replay_key(plain, None, "next", None) == replay_key(enriched, None, "next", None)
    # Both spellings are stripped, not just one.
    enriched2 = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "reasoning": "because..."},
        {"role": "user", "content": "and then?"},
    ]
    assert replay_key(plain, None, "next", None) == replay_key(enriched2, None, "next", None)


def test_cross_conversation_isolation():
    """Two different conversations that produce the identical short assistant
    reply ("OK") must NOT collide — the key covers the whole prefix, not just
    the turn."""
    prefix_a = [{"role": "user", "content": "is the sky blue?"}]
    prefix_b = [{"role": "user", "content": "is water wet?"}]
    ka = replay_key(prefix_a, None, "OK", None)
    kb = replay_key(prefix_b, None, "OK", None)
    assert ka != kb


def test_system_field_participates_in_the_key():
    msgs = [{"role": "user", "content": "hi"}]
    k1 = replay_key(msgs, "be terse", "hello", None)
    k2 = replay_key(msgs, "be verbose", "hello", None)
    assert k1 != k2


def test_tool_calls_participate_in_the_key():
    msgs = [{"role": "user", "content": "weather?"}]
    tc1 = [{"id": "1", "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"NY"}'}}]
    tc2 = [{"id": "1", "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"LA"}'}}]
    assert replay_key(msgs, None, None, tc1) != replay_key(msgs, None, None, tc2)


# --------------------------------------------------------------------------
# accumulate_tool_call_deltas
# --------------------------------------------------------------------------

def test_accumulate_tool_call_deltas_concatenates_name_and_arguments():
    acc: dict[int, dict] = {}
    accumulate_tool_call_deltas(acc, [
        {"index": 0, "id": "call_1", "type": "function",
         "function": {"name": "get_", "arguments": ""}},
    ])
    accumulate_tool_call_deltas(acc, [
        {"index": 0, "function": {"name": "weather", "arguments": '{"city":'}},
    ])
    accumulate_tool_call_deltas(acc, [
        {"index": 0, "function": {"arguments": '"NY"}'}},
    ])
    assert acc[0] == {
        "id": "call_1", "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city":"NY"}'},
    }


def test_accumulate_tool_call_deltas_handles_multiple_indices():
    acc: dict[int, dict] = {}
    accumulate_tool_call_deltas(acc, [
        {"index": 0, "id": "a", "function": {"name": "f1", "arguments": ""}},
        {"index": 1, "id": "b", "function": {"name": "f2", "arguments": ""}},
    ])
    assert set(acc) == {0, 1}
    assert acc[0]["id"] == "a" and acc[1]["id"] == "b"


def test_accumulate_tool_call_deltas_ignores_garbage():
    acc: dict[int, dict] = {}
    accumulate_tool_call_deltas(acc, "not a list")
    accumulate_tool_call_deltas(acc, [None, 5, {"index": "x"}])
    assert acc == {} or 0 in acc  # never raises either way


# --------------------------------------------------------------------------
# ReasoningReplayStore — bounds
# --------------------------------------------------------------------------

def test_store_round_trip():
    s = ReasoningReplayStore()
    s.put("k1", "because reasons")
    assert s.get("k1") == "because reasons"
    assert s.get("nope") is None


def test_store_ignores_empty_reasoning():
    s = ReasoningReplayStore()
    s.put("k1", "")
    assert s.size == 0


def test_eviction_by_entry_count():
    s = ReasoningReplayStore(max_entries=3, max_bytes=10**9)
    for i in range(5):
        s.put(f"k{i}", "x" * 10)
    assert s.size == 3
    # LRU: the earliest keys are gone, the most recent survive.
    assert s.get("k0") is None and s.get("k4") == "x" * 10


def test_eviction_by_total_bytes():
    s = ReasoningReplayStore(max_entries=10**6, max_bytes=30)
    s.put("k0", "x" * 20)
    s.put("k1", "y" * 20)  # total would be 40 > 30 -> evicts k0
    assert s.get("k0") is None
    assert s.get("k1") == "y" * 20
    assert s.total_bytes <= 30


def test_a_single_entry_larger_than_the_whole_budget_is_not_stored():
    s = ReasoningReplayStore(max_entries=10, max_bytes=10)
    s.put("k0", "x" * 100)
    assert s.size == 0


def test_ttl_expiry(monkeypatch):
    s = ReasoningReplayStore(ttl_s=5.0)
    s.put("k0", "value")
    assert s.get("k0") == "value"
    # Advance the store's clock past the TTL. Offset from the REAL current
    # monotonic() rather than a fixed literal — on a long-uptime host the
    # process's monotonic clock can already exceed a small fixed constant.
    future = reasoning_replay.time.monotonic() + 10_000.0
    monkeypatch.setattr(reasoning_replay.time, "monotonic", lambda: future)
    assert s.get("k0") is None
    assert s.size == 0


def test_lru_touch_on_get():
    s = ReasoningReplayStore(max_entries=2, max_bytes=10**9)
    s.put("k0", "a")
    s.put("k1", "b")
    s.get("k0")            # touches k0 -> k1 becomes the LRU victim
    s.put("k2", "c")
    assert s.get("k0") == "a"
    assert s.get("k1") is None
    assert s.get("k2") == "c"


# --------------------------------------------------------------------------
# Correction.apply_reasoning_replay_restore / store_reasoning_replay
# --------------------------------------------------------------------------

def _req(payload, *, stream=False, ptype="chat_completion", endpoint="tier3",
         rid="r1", call_site="test"):
    r = types.SimpleNamespace()
    r.payload = payload; r.stream = stream; r.payload_type = ptype
    r.endpoint = endpoint; r.request_id = rid; r.call_site = call_site
    return r


def _mock_self(*, replay=True, store=None):
    ep = types.SimpleNamespace(replay_reasoning_history=replay)
    state = types.SimpleNamespace()
    state.config = types.SimpleNamespace(endpoints={"tier3": ep})
    state.reasoning_replay = store if store is not None else ReasoningReplayStore()
    state.reasoning_replay_stored = 0
    state.reasoning_replay_restored = 0
    state.reasoning_replay_miss = 0
    state.reasoning_replay_skipped = 0
    state.reasoning_replay_by_endpoint = {}
    m = types.SimpleNamespace(state=state)
    for name in ("apply_reasoning_replay_restore", "store_reasoning_replay"):
        setattr(m, name, getattr(C, name).__get__(m, C))
    return m


def test_store_then_restore_round_trip_non_streaming():
    m = _mock_self()
    turn1_messages = [{"role": "user", "content": "why is the sky blue?"}]
    m.store_reasoning_replay(
        _req({"messages": turn1_messages}), "scattering", None,
        "let me think about Rayleigh scattering")
    assert m.state.reasoning_replay_stored == 1

    # A LATER call replays that turn stripped to plain content, plus a new
    # user turn — exactly what a client that rebuilds history from
    # `message.content` alone sends.
    history = [
        {"role": "user", "content": "why is the sky blue?"},
        {"role": "assistant", "content": "scattering"},
        {"role": "user", "content": "and sunsets?"},
    ]
    req2 = _req({"messages": history})
    m.apply_reasoning_replay_restore(req2)
    assert req2.payload["messages"][1]["reasoning_content"] == (
        "let me think about Rayleigh scattering")
    assert m.state.reasoning_replay_restored == 1
    assert m.state.reasoning_replay_miss == 0
    # The ORIGINAL history list handed in must not have been mutated in place.
    assert "reasoning_content" not in history[1]


def test_restore_miss_leaves_message_untouched():
    m = _mock_self()
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    req = _req({"messages": history})
    m.apply_reasoning_replay_restore(req)
    assert "reasoning_content" not in req.payload["messages"][1]
    assert m.state.reasoning_replay_miss == 1
    assert m.state.reasoning_replay_restored == 0


def test_restore_leaves_a_message_that_already_carries_reasoning_untouched():
    m = _mock_self()
    # Prime the store so a hit WOULD be available, to prove the pre-existing
    # reasoning wins rather than being silently overwritten.
    m.store_reasoning_replay(
        _req({"messages": [{"role": "user", "content": "hi"}]}),
        "hello", None, "stored reasoning")
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "reasoning_content": "already here"},
    ]
    req = _req({"messages": history})
    m.apply_reasoning_replay_restore(req)
    assert req.payload["messages"][1]["reasoning_content"] == "already here"
    assert m.state.reasoning_replay_restored == 0
    assert m.state.reasoning_replay_skipped == 1


def test_cross_conversation_isolation_end_to_end():
    """Identical assistant content ('OK') in two different conversations must
    never cross-contaminate — a restore on one must not pull in the other's
    reasoning."""
    m = _mock_self()
    m.store_reasoning_replay(
        _req({"messages": [{"role": "user", "content": "is the sky blue?"}]}),
        "OK", None, "reasoning about the sky")
    m.store_reasoning_replay(
        _req({"messages": [{"role": "user", "content": "is water wet?"}]}),
        "OK", None, "reasoning about water")

    req = _req({"messages": [
        {"role": "user", "content": "is the sky blue?"},
        {"role": "assistant", "content": "OK"},
        {"role": "user", "content": "why?"},
    ]})
    m.apply_reasoning_replay_restore(req)
    assert req.payload["messages"][1]["reasoning_content"] == "reasoning about the sky"


def test_tool_call_turn_round_trips():
    m = _mock_self()
    tool_calls = [{"id": "call_1", "type": "function",
                   "function": {"name": "get_weather", "arguments": '{"city":"NY"}'}}]
    m.store_reasoning_replay(
        _req({"messages": [{"role": "user", "content": "weather in NY?"}]}),
        None, tool_calls, "deciding to call get_weather")
    history = [
        {"role": "user", "content": "weather in NY?"},
        {"role": "assistant", "content": None, "tool_calls": tool_calls},
        {"role": "tool", "content": "72F", "tool_call_id": "call_1"},
    ]
    req = _req({"messages": history})
    m.apply_reasoning_replay_restore(req)
    assert req.payload["messages"][1]["reasoning_content"] == "deciding to call get_weather"


def test_endpoint_not_opted_in_is_a_no_op_both_directions():
    m = _mock_self(replay=False)
    m.store_reasoning_replay(
        _req({"messages": [{"role": "user", "content": "hi"}]}), "hello", None,
        "some reasoning")
    assert m.state.reasoning_replay_stored == 0
    assert m.state.reasoning_replay.size == 0

    req = _req({"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]})
    m.apply_reasoning_replay_restore(req)
    assert "reasoning_content" not in req.payload["messages"][1]
    assert m.state.reasoning_replay_miss == 0  # never even looked


def test_store_skips_when_reasoning_is_empty():
    m = _mock_self()
    for empty in (None, "", "   "):
        m.store_reasoning_replay(
            _req({"messages": [{"role": "user", "content": "hi"}]}), "hello",
            None, empty)
    assert m.state.reasoning_replay_stored == 0
    assert m.state.reasoning_replay_skipped == 3


def test_restore_no_op_on_non_chat_payload_type():
    m = _mock_self()
    req = _req({"messages": [{"role": "assistant", "content": "x"}]},
               ptype="embedding")
    before = dict(req.payload)
    m.apply_reasoning_replay_restore(req)
    assert req.payload == before  # untouched


def test_errors_are_swallowed_not_raised():
    """A correction must never break the request/response path. Feed shapes
    that would raise inside a naive implementation (non-list messages, a
    non-dict payload) and confirm nothing propagates."""
    m = _mock_self()
    req = _req("not a dict")
    m.apply_reasoning_replay_restore(req)  # must not raise

    req2 = _req({"messages": "not a list"})
    m.apply_reasoning_replay_restore(req2)  # must not raise

    # store side: messages missing / wrong shape
    m.store_reasoning_replay(_req({"messages": None}), "c", None, "reasoning")
    m.store_reasoning_replay(_req("not a dict"), "c", None, "reasoning")
    assert m.state.reasoning_replay_stored == 0


# --------------------------------------------------------------------------
# Catalog wiring
# --------------------------------------------------------------------------

def test_replay_reasoning_history_is_in_the_policy_passthrough_allowlist():
    assert "replay_reasoning_history" in model_catalog._POLICY_PASSTHROUGH


def test_declared_flag_reaches_endpoint_config():
    """End to end through the real builder (same shape as
    test_reasoning_tuning.py::test_a_declared_stanza_reaches_endpoint_config):
    a key that is not read by build_endpoint_kwargs is silently dropped from
    models.yaml, which looks identical to a knob that was never load-bearing."""
    from roadstead.config import EndpointConfig

    entry = model_catalog.EndpointEntry(
        name="probe", provider="p", kind="chat",
        policy={"replay_reasoning_history": True},
    )
    kw = model_catalog.build_endpoint_kwargs(entries=[entry])["probe"]
    assert kw["replay_reasoning_history"] is True
    ep = EndpointConfig(**{k: v for k, v in kw.items()
                           if k in EndpointConfig.__dataclass_fields__})
    assert ep.replay_reasoning_history is True


def test_undeclared_endpoint_gets_the_false_default():
    entry = model_catalog.EndpointEntry(name="probe2", provider="p", kind="chat", policy={})
    kw = model_catalog.build_endpoint_kwargs(entries=[entry])["probe2"]
    assert "replay_reasoning_history" not in kw

