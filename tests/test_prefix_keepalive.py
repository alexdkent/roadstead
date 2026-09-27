"""Prefix keep-alive (roadstead/prefix_keepalive.py + its health.py/lifecycle.py
wiring).

Covers, in order: the module's own pure logic in isolation (skeleton
derivation/keying, accounting, trigger/idle/eviction, touch-payload shape);
the `models.yaml` policy passthrough (a key silently dropped here looks
exactly like a knob that was never load-bearing — the same failure mode
`test_goodput_wiring.py` guards); the poller-side scheduling/capacity gate in
`health.py`, using the same lightweight `types.SimpleNamespace` state stub
`test_goodput_wiring.py` uses; and one end-to-end pass through the real
`ProxyService` (`test_truncation_guard.py`'s harness) proving a real
completed request actually reaches the tracker and a due touch actually
fires through `backend.probe_prefix_touch`.
"""
from __future__ import annotations

import asyncio
import types

import pytest

from roadstead import model_catalog
from roadstead.agent_budget import BudgetManager
from roadstead.backend import BackendResponse
from roadstead.config import EndpointConfig, ProxyConfig
from roadstead.cost_model import CostModel
from roadstead.health import Health
from roadstead.scheduler import QueuedRequest, Scheduler
from roadstead.prefix_keepalive import (
    HIT_RATIO,
    PrefixKeepaliveTracker,
    TrackedPrefix,
    derive_skeleton,
    skeleton_key,
)
from roadstead.service import ProxyService

# --------------------------------------------------------------------------- #
# 1. derive_skeleton / skeleton_key — pure logic
# --------------------------------------------------------------------------- #

_SYS = {"role": "system", "content": "be terse"}
_TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]


def _payload(*, system=True, tools=True, model=None, ck=None, user="hi",
             sys_content=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": sys_content or _SYS["content"]})
    messages.append({"role": "user", "content": user})
    p = {"messages": messages}
    if tools:
        p["tools"] = list(_TOOLS)
    if model:
        p["model"] = model
    if ck:
        p["chat_template_kwargs"] = dict(ck)
    return p


def test_derive_skeleton_returns_none_with_no_system_and_no_tools():
    assert derive_skeleton({"messages": [{"role": "user", "content": "hi"}]}) is None


def test_derive_skeleton_returns_none_for_a_non_dict_payload():
    assert derive_skeleton(None) is None
    assert derive_skeleton("not a dict") is None


def test_derive_skeleton_keeps_only_the_LEADING_system_run():
    """A system message that arrives after a non-system one must not join the
    skeleton — only a CONTIGUOUS run from index 0 is "the leading prefix"."""
    messages = [
        dict(_SYS),
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "late injection"},
    ]
    sk = derive_skeleton({"messages": messages, "tools": _TOOLS})
    assert sk["system_messages"] == [_SYS]


def test_derive_skeleton_carries_tools_model_and_chat_template_kwargs():
    sk = derive_skeleton(_payload(model="glm", ck={"reasoning_effort": "low"}))
    assert sk["tools"] == _TOOLS
    assert sk["model"] == "glm"
    assert sk["chat_template_kwargs"] == {"reasoning_effort": "low"}


def test_skeleton_key_is_stable_across_the_TRAILING_user_turn():
    """Two payloads differing ONLY in their own trailing turn must key
    identically — that is the whole point of deriving a skeleton rather than
    hashing the payload whole."""
    a = derive_skeleton(_payload(user="what is the weather in NY?"))
    b = derive_skeleton(_payload(user="totally different question"))
    assert skeleton_key(a) == skeleton_key(b)


@pytest.mark.parametrize("mutate", [
    lambda p: p.__setitem__("tools", [{"type": "function",
                                       "function": {"name": "other"}}]),
    lambda p: p["messages"].__setitem__(0, {"role": "system", "content": "different"}),
    lambda p: p.__setitem__("model", "a-different-model"),
    lambda p: p.__setitem__("chat_template_kwargs", {"reasoning_effort": "high"}),
    lambda p: p.__setitem__("tool_choice", "required"),
])
def test_skeleton_key_differs_when_a_rendering_relevant_field_differs(mutate):
    base = _payload(model="glm", ck={"reasoning_effort": "low"})
    base["tool_choice"] = "auto"
    other = _payload(model="glm", ck={"reasoning_effort": "low"})
    other["tool_choice"] = "auto"
    mutate(other)
    assert skeleton_key(derive_skeleton(base)) != skeleton_key(derive_skeleton(other))


def test_tool_choice_is_carried_into_the_skeleton():
    sk = derive_skeleton({"messages": [{"role": "user", "content": "hi"}],
                          "tools": _TOOLS, "tool_choice": "required"})
    assert sk["tool_choice"] == "required"


# --------------------------------------------------------------------------- #
# 1b. derive_skeleton — the TWO system-prompt shapes, and the one it refuses
# --------------------------------------------------------------------------- #

def test_a_top_level_system_field_is_captured_and_keys_like_the_shape_it_is():
    """The Anthropic-shaped door: a top-level `system` string, folded into
    `messages` only at dispatch time by the provider — the skeleton must
    capture the RAW field, not pre-fold it itself (see the module
    docstring)."""
    payload = {"messages": [{"role": "user", "content": "hi"}],
              "system": "be terse", "tools": _TOOLS}
    sk = derive_skeleton(payload)
    assert sk is not None
    assert sk["top_level_system"] == "be terse"
    assert sk["system_messages"] == []


def test_the_two_system_shapes_key_differently_even_with_the_same_text():
    """A messages-array system message and a top-level `system` field are
    DIFFERENT wire shapes a provider treats differently (the array form is
    already positioned; the top-level form gets prepended by
    `prepare_chat_payload`) — collapsing them into one key would touch
    whichever shape happened to be captured, not the one a later real
    request actually sends."""
    as_message = derive_skeleton(
        {"messages": [{"role": "system", "content": "be terse"},
                      {"role": "user", "content": "hi"}]})
    as_top_level = derive_skeleton(
        {"messages": [{"role": "user", "content": "hi"}], "system": "be terse"})
    assert skeleton_key(as_message) != skeleton_key(as_top_level)


def test_a_top_level_system_field_alone_with_no_tools_is_still_worth_tracking():
    sk = derive_skeleton({"messages": [{"role": "user", "content": "hi"}],
                          "system": "be terse"})
    assert sk is not None
    assert sk["top_level_system"] == "be terse"


def test_a_system_prompt_hidden_inside_extra_body_is_refused_not_dropped():
    """No provider in this package folds `extra_body.system` the way a
    top-level `system` is (see providers/*.py's `prepare_chat_payload`) — a
    skeleton built from this shape would silently omit real prefix content,
    so it must refuse to track the prefix at all rather than track a
    skeleton that renders differently from what a real request sent."""
    payload = {"messages": [{"role": "user", "content": "hi"}],
              "extra_body": {"system": "be terse"}, "tools": _TOOLS}
    assert derive_skeleton(payload) is None


def test_the_touch_payload_sends_a_top_level_system_field_back_as_one():
    """The touch must reproduce the shape, not translate it — the SAME
    `prepare_chat_payload` call folds it identically either way (see
    `backend.probe_prefix_touch`); folding it here too would be a second
    implementation of that rule to keep in step with the first."""
    t = PrefixKeepaliveTracker()
    entry = _entry(skeleton={"system_messages": [], "top_level_system": "be terse",
                             "tools": None, "tool_choice": None, "model": None,
                             "chat_template_kwargs": None})
    payload = t.build_touch_payload(entry)
    assert payload["system"] == "be terse"
    assert "system" not in (payload["messages"][0] if payload["messages"] else {})


# --------------------------------------------------------------------------- #
# 2. PrefixKeepaliveTracker — accounting, trigger, idle, eviction
# --------------------------------------------------------------------------- #

def _ep(**over) -> EndpointConfig:
    kw = dict(endpoint_class="tier3", role="reasoner",
              prefix_keepalive_call_sites=("kv4.*",),
              prefix_keepalive_trigger_tokens=100,
              prefix_keepalive_idle_s=0,
              prefix_keepalive_max_prefixes=0)
    kw.update(over)
    return EndpointConfig(**kw)


def test_enabled_requires_BOTH_call_sites_and_a_positive_trigger():
    assert PrefixKeepaliveTracker.enabled(_ep()) is True
    assert PrefixKeepaliveTracker.enabled(_ep(prefix_keepalive_call_sites=())) is False
    assert PrefixKeepaliveTracker.enabled(
        _ep(prefix_keepalive_trigger_tokens=0)) is False


def test_matches_is_an_fnmatch_glob():
    ep = _ep(prefix_keepalive_call_sites=("hermes.*",))
    assert PrefixKeepaliveTracker.matches(ep, "hermes.openai_compat")
    assert not PrefixKeepaliveTracker.matches(ep, "kv4.judge")


def test_undeclared_endpoint_is_completely_inert():
    """Absent policy -> not even a bucket is created; observe_completion is a
    single attribute read and nothing more."""
    t = PrefixKeepaliveTracker()
    ep = EndpointConfig(endpoint_class="x", role="x")
    t.observe_completion("x", ep, call_site="kv4.judge", payload=_payload(),
                         status="ok", input_tokens=1000, cached_tokens=0, now=0.0)
    assert t.status_snapshot("x", 0.0) is None


def test_a_matching_ok_completion_creates_and_refreshes_a_tracked_prefix():
    t = PrefixKeepaliveTracker()
    ep = _ep()
    t.observe_completion("tier3", ep, call_site="kv4.judge", payload=_payload(),
                         status="ok", input_tokens=5000, cached_tokens=100, now=10.0)
    snap = t.status_snapshot("tier3", 10.0)
    assert len(snap["tracked"]) == 1
    row = snap["tracked"][0]
    assert row["call_site"] == "kv4.judge"
    assert row["approx_tokens"] == 5000
    assert row["tokens_since_touch"] == 0
    assert row["touch_in_flight"] is False


def test_a_non_matching_call_site_never_captures():
    t = PrefixKeepaliveTracker()
    ep = _ep()
    t.observe_completion("tier3", ep, call_site="other.thing", payload=_payload(),
                         status="ok", input_tokens=5000, cached_tokens=0, now=0.0)
    assert t.status_snapshot("tier3", 0.0) is None


def test_a_failed_completion_never_captures_even_from_a_matching_call_site():
    t = PrefixKeepaliveTracker()
    ep = _ep()
    t.observe_completion("tier3", ep, call_site="kv4.judge", payload=_payload(),
                         status="error", input_tokens=None, cached_tokens=None, now=0.0)
    assert t.status_snapshot("tier3", 0.0) is None


def test_every_completion_accounts_uncached_tokens_against_OTHER_prefixes():
    t = PrefixKeepaliveTracker()
    ep = _ep()
    t.observe_completion("tier3", ep, call_site="kv4.judge", payload=_payload(),
                         status="ok", input_tokens=1000, cached_tokens=900, now=0.0)
    # A DIFFERENT, non-matching call site's completion on the same endpoint.
    t.observe_completion("tier3", ep, call_site="other.thing", payload={},
                         status="ok", input_tokens=2000, cached_tokens=0, now=1.0)
    row = t.status_snapshot("tier3", 1.0)["tracked"][0]
    assert row["tokens_since_touch"] == 2000, (
        "the OTHER request's 2000 uncached prompt tokens must count against "
        "the tracked prefix's countdown")


def test_a_matching_completion_resets_its_OWN_countdown_not_just_creates():
    t = PrefixKeepaliveTracker()
    ep = _ep()
    t.observe_completion("tier3", ep, call_site="kv4.judge", payload=_payload(),
                         status="ok", input_tokens=1000, cached_tokens=900, now=0.0)
    t.observe_completion("tier3", ep, call_site="other.thing", payload={},
                         status="ok", input_tokens=5000, cached_tokens=0, now=1.0)
    # A SECOND real hit from the same call_site — must zero the countdown
    # this same request would otherwise itself have been charged for.
    t.observe_completion("tier3", ep, call_site="kv4.judge", payload=_payload(),
                         status="ok", input_tokens=1000, cached_tokens=900, now=2.0)
    row = t.status_snapshot("tier3", 2.0)["tracked"][0]
    assert row["tokens_since_touch"] == 0
    assert row["last_real_seen_age_s"] == 0.0


def test_accounting_is_skipped_when_either_token_count_is_unknown():
    """None means "cannot tell", not "no other traffic" — coercing it to 0
    would silently mean the countdown never has anything to count."""
    t = PrefixKeepaliveTracker()
    ep = _ep()
    t.observe_completion("tier3", ep, call_site="kv4.judge", payload=_payload(),
                         status="ok", input_tokens=1000, cached_tokens=900, now=0.0)
    t.observe_completion("tier3", ep, call_site="other.thing", payload={},
                         status="timeout", input_tokens=0, cached_tokens=None, now=1.0)
    row = t.status_snapshot("tier3", 1.0)["tracked"][0]
    assert row["tokens_since_touch"] == 0


def test_max_prefixes_evicts_LRU_by_last_REAL_use():
    t = PrefixKeepaliveTracker()
    ep = _ep(prefix_keepalive_call_sites=("kv4.judge", "kv4.plan", "kv4.review"),
             prefix_keepalive_max_prefixes=2)

    def _capture(call_site, sys_content, now):
        # A DISTINCT system message per site — the skeleton (not the
        # call_site, and not the trailing user turn) is what keys a tracked
        # prefix, so three genuinely different prefixes need three different
        # leading system prompts, not just three different call sites.
        t.observe_completion("tier3", ep, call_site=call_site,
                             payload=_payload(sys_content=sys_content), status="ok",
                             input_tokens=100, cached_tokens=0, now=now)

    _capture("kv4.judge", "prompt A", now=0.0)
    _capture("kv4.plan", "prompt B", now=1.0)
    _capture("kv4.review", "prompt C", now=2.0)  # evicts kv4.judge (oldest)

    sites = {row["call_site"] for row in t.status_snapshot("tier3", 2.0)["tracked"]}
    assert sites == {"kv4.plan", "kv4.review"}


def test_max_prefixes_zero_is_unbounded():
    t = PrefixKeepaliveTracker()
    ep = _ep(prefix_keepalive_call_sites=("kv4.*",), prefix_keepalive_max_prefixes=0)
    for i in range(5):
        t.observe_completion("tier3", ep, call_site=f"kv4.site{i}",
                             payload=_payload(sys_content=f"prompt {i}"), status="ok",
                             input_tokens=100, cached_tokens=0, now=float(i))
    assert len(t.status_snapshot("tier3", 5.0)["tracked"]) == 5


# --------------------------------------------------------------------------- #
# 3. due_for_touch — trigger + idle + in-flight
# --------------------------------------------------------------------------- #

def _tracked(t: PrefixKeepaliveTracker, ep_name="tier3"):
    return t._by_endpoint[ep_name]


def test_due_for_touch_requires_the_trigger_to_be_crossed():
    t = PrefixKeepaliveTracker()
    ep = _ep(prefix_keepalive_trigger_tokens=1000)
    t.observe_completion("tier3", ep, call_site="kv4.judge", payload=_payload(),
                         status="ok", input_tokens=100, cached_tokens=0, now=0.0)
    t.observe_completion("tier3", ep, call_site="other", payload={}, status="ok",
                         input_tokens=999, cached_tokens=0, now=1.0)
    assert t.due_for_touch("tier3", ep, now=1.0) == []
    t.observe_completion("tier3", ep, call_site="other", payload={}, status="ok",
                         input_tokens=1, cached_tokens=0, now=2.0)
    assert len(t.due_for_touch("tier3", ep, now=2.0)) == 1


def test_due_for_touch_excludes_a_prefix_past_its_idle_deadline():
    t = PrefixKeepaliveTracker()
    ep = _ep(prefix_keepalive_trigger_tokens=100, prefix_keepalive_idle_s=60)
    t.observe_completion("tier3", ep, call_site="kv4.judge", payload=_payload(),
                         status="ok", input_tokens=100, cached_tokens=0, now=0.0)
    t.observe_completion("tier3", ep, call_site="other", payload={}, status="ok",
                         input_tokens=200, cached_tokens=0, now=1.0)
    assert t.due_for_touch("tier3", ep, now=1.0 + 61) == [], (
        "nobody used this prefix in over the idle window — must not touch it")
    assert t.due_for_touch("tier3", ep, now=1.0 + 59) != []


def test_due_for_touch_excludes_a_touch_already_in_flight():
    t = PrefixKeepaliveTracker()
    ep = _ep(prefix_keepalive_trigger_tokens=100)
    t.observe_completion("tier3", ep, call_site="kv4.judge", payload=_payload(),
                         status="ok", input_tokens=100, cached_tokens=0, now=0.0)
    t.observe_completion("tier3", ep, call_site="other", payload={}, status="ok",
                         input_tokens=200, cached_tokens=0, now=1.0)
    entry = _tracked(t)[next(iter(_tracked(t)))]
    entry.touch_in_flight = True
    assert t.due_for_touch("tier3", ep, now=1.0) == []


def test_zero_trigger_never_arms_due_checking_even_with_a_tracked_prefix():
    t = PrefixKeepaliveTracker()
    ep = _ep(prefix_keepalive_trigger_tokens=100)
    t.observe_completion("tier3", ep, call_site="kv4.judge", payload=_payload(),
                         status="ok", input_tokens=100, cached_tokens=0, now=0.0)
    disabled = _ep(prefix_keepalive_trigger_tokens=0)
    assert t.due_for_touch("tier3", disabled, now=0.0) == []


# --------------------------------------------------------------------------- #
# 4. build_touch_payload / record_dispatch / record_result
# --------------------------------------------------------------------------- #

def _entry(**over):
    kw = dict(key="k", skeleton={"system_messages": [_SYS], "top_level_system": None,
                                 "tools": _TOOLS, "tool_choice": "required",
                                 "model": "glm", "chat_template_kwargs": {"a": 1}},
              call_site="kv4.judge", last_real_seen=0.0)
    kw.update(over)
    return TrackedPrefix(**kw)


def test_build_touch_payload_shape():
    t = PrefixKeepaliveTracker()
    entry = _entry()
    payload = t.build_touch_payload(entry)
    assert payload["messages"][0] == _SYS
    assert payload["messages"][1]["role"] == "user"
    assert payload["messages"][1]["content"].startswith("keepalive ")
    assert payload["tools"] == _TOOLS
    assert payload["tool_choice"] == "required"
    assert payload["model"] == "glm"
    assert payload["chat_template_kwargs"] == {"a": 1}
    assert payload["max_tokens"] > 0
    assert "system" not in payload, "no top-level system was tracked for this entry"


def test_build_touch_payload_nonce_is_unique_every_call():
    """THE load-bearing detail — see the module docstring for why an
    IDENTICAL repeated touch fails to refresh a new session's checkpoint."""
    t = PrefixKeepaliveTracker()
    entry = _entry()
    a = t.build_touch_payload(entry)["messages"][-1]["content"]
    b = t.build_touch_payload(entry)["messages"][-1]["content"]
    assert a != b


def test_record_dispatch_marks_in_flight_and_resets_the_counter():
    t = PrefixKeepaliveTracker()
    entry = _entry(tokens_since_touch=500.0)
    t.record_dispatch(entry)
    assert entry.touch_in_flight is True
    assert entry.tokens_since_touch == 0.0


@pytest.mark.parametrize("result,sent,hit,missed,skipped", [
    ("hit", 1, 1, 0, 0),
    ("missed", 1, 0, 1, 0),
    ("skipped", 0, 0, 0, 1),
])
def test_record_result_tallies_per_endpoint(result, sent, hit, missed, skipped):
    t = PrefixKeepaliveTracker()
    entry = _entry()
    entry.touch_in_flight = True
    t.record_result("tier3", entry, result)
    assert entry.touch_in_flight is False
    stats = t._stats["tier3"]
    assert (stats.sent, stats.hit, stats.missed, stats.skipped) == (
        sent, hit, missed, skipped)


def test_record_result_accepts_no_entry_for_a_skip_before_one_was_picked():
    t = PrefixKeepaliveTracker()
    t.record_result("tier3", None, "skipped")
    assert t._stats["tier3"].skipped == 1


def test_status_snapshot_is_none_when_nothing_ever_tracked_or_touched():
    t = PrefixKeepaliveTracker()
    assert t.status_snapshot("tier3", 0.0) is None


# --------------------------------------------------------------------------- #
# 5. models.yaml wiring — the policy keys must actually reach EndpointConfig
# --------------------------------------------------------------------------- #

_POLICY_KEYS = [k for k in model_catalog._POLICY_PASSTHROUGH
                if k.startswith("prefix_keepalive_")]


def test_policy_keys_reach_endpoint_config():
    assert _POLICY_KEYS == [
        "prefix_keepalive_trigger_tokens",
        "prefix_keepalive_idle_s",
        "prefix_keepalive_max_prefixes",
    ]
    policy = {
        "prefix_keepalive_call_sites": ["hermes.*", "kv4.judge"],
        "prefix_keepalive_trigger_tokens": 100_000,
        "prefix_keepalive_idle_s": 7200,
        "prefix_keepalive_max_prefixes": 8,
    }
    entry = model_catalog.EndpointEntry(
        name="probe", provider="p", kind="chat", policy=dict(policy))
    kw = model_catalog.build_endpoint_kwargs(entries=[entry])["probe"]
    assert kw["prefix_keepalive_call_sites"] == ("hermes.*", "kv4.judge")
    for key in _POLICY_KEYS:
        assert kw[key] == policy[key]

    ep = EndpointConfig(**{k: v for k, v in kw.items()
                           if k in EndpointConfig.__dataclass_fields__})
    assert ep.prefix_keepalive_call_sites == ("hermes.*", "kv4.judge")
    assert ep.prefix_keepalive_trigger_tokens == 100_000
    assert ep.prefix_keepalive_idle_s == 7200
    assert ep.prefix_keepalive_max_prefixes == 8


def test_undeclared_endpoint_ships_no_default():
    """🚨 Absent -> off. A default here would arm the tracker on every
    deployment that never asked for it."""
    ep = EndpointConfig(endpoint_class="x", role="x")
    assert ep.prefix_keepalive_call_sites == ()
    assert ep.prefix_keepalive_trigger_tokens == 0
    assert ep.prefix_keepalive_idle_s == 0
    assert ep.prefix_keepalive_max_prefixes == 0
    assert PrefixKeepaliveTracker.enabled(ep) is False


def test_a_single_string_call_sites_value_is_still_accepted():
    entry = model_catalog.EndpointEntry(
        name="probe", provider="p", kind="chat",
        policy={"prefix_keepalive_call_sites": "kv4.judge",
                "prefix_keepalive_trigger_tokens": 1})
    kw = model_catalog.build_endpoint_kwargs(entries=[entry])["probe"]
    assert kw["prefix_keepalive_call_sites"] == ("kv4.judge",)


def test_an_unknown_policy_key_is_still_reported_not_silently_dropped(monkeypatch):
    """`prefix_keepalive_call_sites` is handled OUTSIDE the passthrough loop —
    confirm it is in `_POLICY_HANDLED` so the unknown-key notice does not
    wrongly flag a key that IS load-bearing."""
    assert "prefix_keepalive_call_sites" in model_catalog._POLICY_HANDLED


# --------------------------------------------------------------------------- #
# 5b. Scheduler.background_available — the reusable capacity primitive
#
# health.py's gate MUST reuse this rather than re-deriving "is there room"
# from `effective_max_slots` — a real Scheduler proves it actually reads
# `fast_path_reserve_slots`/`background_cap_slots`, not a stub that always
# answers whatever a test wants.
# --------------------------------------------------------------------------- #

def _sched(max_slots=4, fast_path_reserve_slots=0):
    config = ProxyConfig()
    config.endpoints["tier3"].max_slots = max_slots
    config.endpoints["tier3"].fast_path_reserve_slots = fast_path_reserve_slots
    cm = CostModel()
    for ep, epc in config.endpoints.items():
        cm.register_endpoint(ep, epc.max_slots)
    bm = BudgetManager()
    bm.set_total_capacity(config.total_fleet_slots)
    return Scheduler(config, cm, bm)


def _bg_req(now, agent_id="bg-agent"):
    return QueuedRequest.create(
        agent_id=agent_id, endpoint="tier3", priority="P3_INGESTION",
        call_site="test", payload_type="chat_completion",
        payload={"messages": [{"role": "user", "content": "x"}], "max_tokens": 8},
        timeout_s=60.0, now=now)


def test_background_available_is_the_full_cap_when_the_endpoint_is_idle():
    sched = _sched(max_slots=4, fast_path_reserve_slots=1)
    # background_cap_slots = max(floor, effective_max_slots - reserve) = max(1, 3) = 3
    assert sched.background_available("tier3") == 3


def test_background_available_respects_the_fast_path_reserve():
    """🚨 THE DEFECT THE REVIEW FLAGGED. `tier3` declares `fast_path_reserve_
    slots: 1` in the shipped catalog: with 4 physical slots and 3 already
    occupied by BACKGROUND-band real traffic, one raw slot is free
    (4 - 3 = 1), but it is the slot the catalog reserved for interactive/
    fast-path traffic — comparing against the bare `effective_max_slots`
    would say "1 available" here, which is exactly wrong."""
    sched = _sched(max_slots=4, fast_path_reserve_slots=1)
    now = 1000.0
    for i in range(3):
        sched.enqueue(_bg_req(now + i * 0.001, agent_id=f"agent{i}"))
    decisions = sched.tick(now + 0.01)
    assert len(decisions) == 3, "all three background requests must dispatch"
    assert sched.active_count("tier3") == 3

    assert sched.background_available("tier3") == 0, (
        "the reserved slot must not read as background-available, even "
        "though a bare (max_slots - in_flight) would say 1")


def test_background_available_folds_in_extra_occupied():
    """A caller (a prefix-keepalive touch) that the scheduler cannot see for
    itself must not be invisible to the same gate real background traffic
    answers to — it shrinks the OVERALL free-slot ceiling exactly as a real
    dispatched request would (``min(effective_max_slots - occupied,
    background_cap_slots)``)."""
    sched = _sched(max_slots=5, fast_path_reserve_slots=1)
    # background_cap_slots = max(floor=1, effective_max_slots(5) - reserve(1)) = 4.
    assert sched.background_available("tier3") == 4
    assert sched.background_available("tier3", extra_occupied=2) == 3
    assert sched.background_available("tier3", extra_occupied=4) == 1
    assert sched.background_available("tier3", extra_occupied=5) == 0
    assert sched.background_available("tier3", extra_occupied=10) == 0, (
        "must never go negative")


def test_background_available_is_zero_for_an_unknown_endpoint():
    sched = _sched()
    assert sched.background_available("no-such-endpoint") == 0


# --------------------------------------------------------------------------- #
# 6. health.py scheduling — the capacity/health gate, without a real ProxyService
# --------------------------------------------------------------------------- #

def _health_state(**over):
    state = types.SimpleNamespace(
        endpoint_failure_times={}, endpoint_cooldown_until={},
        endpoint_cooldown_trips={}, cooldown_best_effort_skips={},
        paused_endpoints=set(), endpoint_health={},
        collapsed_endpoints=set(),
        goodput=None, goodput_verdicts={},
        prefix_keepalive=PrefixKeepaliveTracker(),
        prefix_keepalive_tasks={},
        draining=types.SimpleNamespace(is_set=lambda: False),
        scheduler=types.SimpleNamespace(
            endpoint_snapshot=lambda ep: {"in_flight": 0, "max_slots": 4, "queued": 0},
            background_available=lambda ep, extra_occupied=0: 4),
        on_demand=types.SimpleNamespace(manages=lambda ep: False),
        backend=None,
    )
    for k, v in over.items():
        setattr(state, k, v)
    return state


class _StubBackend:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = []

    async def probe_prefix_touch(self, ep_cfg, payload, timeout_s):
        self.calls.append(payload)
        return self._outcomes.pop(0) if self._outcomes else None


def _due_ep(**over):
    kw = dict(endpoint_class="tier3", role="reasoner",
              prefix_keepalive_call_sites=("kv4.*",),
              prefix_keepalive_trigger_tokens=100, max_slots=4)
    kw.update(over)
    return EndpointConfig(**kw)


async def _tick_and_await(health, ep_name, ep_cfg):
    """Fire the schedule pass and await every task it created, so a
    fire-and-forget touch is observable synchronously in a test."""
    health._schedule_prefix_keepalive_touches(ep_name, ep_cfg)
    tasks = list(health.state.prefix_keepalive_tasks.values())
    if tasks:
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_an_unconfigured_endpoint_is_never_scheduled():
    state = _health_state(backend=_StubBackend([]))
    health = Health(state)
    health._schedule_prefix_keepalive_touches(
        "tier3", EndpointConfig(endpoint_class="tier3", role="r"))
    assert state.backend.calls == []
    assert state.prefix_keepalive_tasks == {}


@pytest.mark.asyncio
async def test_draining_skips_scheduling_entirely():
    state = _health_state(backend=_StubBackend([{"prompt_tokens": 10, "cached_tokens": 10}]),
                          draining=types.SimpleNamespace(is_set=lambda: True))
    ep = _due_ep()
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="kv4.judge", payload=_payload(), status="ok",
        input_tokens=100, cached_tokens=0, now=0.0)
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="other", payload={}, status="ok",
        input_tokens=200, cached_tokens=0, now=1.0)
    health = Health(state)
    await _tick_and_await(health, "tier3", ep)
    assert state.backend.calls == []


@pytest.mark.asyncio
async def test_a_paused_endpoint_is_skipped_not_touched():
    state = _health_state(backend=_StubBackend([{"prompt_tokens": 10, "cached_tokens": 10}]),
                          paused_endpoints={"tier3"})
    ep = _due_ep()
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="kv4.judge", payload=_payload(), status="ok",
        input_tokens=100, cached_tokens=0, now=0.0)
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="other", payload={}, status="ok",
        input_tokens=200, cached_tokens=0, now=1.0)
    health = Health(state)
    await _tick_and_await(health, "tier3", ep)
    assert state.backend.calls == [], "a paused (drained) endpoint owes us no touch"


@pytest.mark.asyncio
async def test_a_full_background_gate_is_skipped_and_the_counter_is_NOT_reset():
    """"Full" is decided by `Scheduler.background_available` — the SAME gate
    real background traffic is held to — never by a bare `in_flight vs
    max_slots` comparison that would ignore `fast_path_reserve_slots`."""
    state = _health_state(
        backend=_StubBackend([{"prompt_tokens": 10, "cached_tokens": 10}]),
        scheduler=types.SimpleNamespace(
            endpoint_snapshot=lambda ep: {"queued": 0},
            background_available=lambda ep, extra_occupied=0: 0))
    ep = _due_ep()
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="kv4.judge", payload=_payload(), status="ok",
        input_tokens=100, cached_tokens=0, now=0.0)
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="other", payload={}, status="ok",
        input_tokens=200, cached_tokens=0, now=1.0)
    health = Health(state)
    await _tick_and_await(health, "tier3", ep)
    assert state.backend.calls == [], "full endpoint must not be touched"
    snap = state.prefix_keepalive.status_snapshot("tier3", 1.0)
    assert snap["touches_skipped"] == 1
    assert snap["tracked"][0]["tokens_since_touch"] == 200, (
        "a SKIP must not reset the countdown — the very next poller pass "
        "must retry, not wait out a fresh trigger")


@pytest.mark.asyncio
async def test_a_queued_real_request_skips_the_touch_entirely():
    """A real request already waiting for a slot must never wait even one
    tick longer because a touch got there first — checked BEFORE the
    capacity gate, and regardless of what that gate would have said."""
    state = _health_state(
        backend=_StubBackend([{"prompt_tokens": 10, "cached_tokens": 10}]),
        scheduler=types.SimpleNamespace(
            endpoint_snapshot=lambda ep: {"queued": 1},
            background_available=lambda ep, extra_occupied=0: 4))
    ep = _due_ep()
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="kv4.judge", payload=_payload(), status="ok",
        input_tokens=100, cached_tokens=0, now=0.0)
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="other", payload={}, status="ok",
        input_tokens=200, cached_tokens=0, now=1.0)
    health = Health(state)
    await _tick_and_await(health, "tier3", ep)
    assert state.backend.calls == [], "a queued real request must pre-empt any touch"
    snap = state.prefix_keepalive.status_snapshot("tier3", 1.0)
    assert snap["touches_skipped"] == 1
    assert snap["tracked"][0]["tokens_since_touch"] == 200, (
        "a SKIP must not reset the countdown")


@pytest.mark.asyncio
async def test_two_due_prefixes_in_one_tick_produce_exactly_one_touch():
    """Several tracked prefixes can cross their trigger in the same poller
    pass; only ONE touch is dispatched, and the other waits for the next
    tick rather than piling a second generation onto the endpoint."""
    state = _health_state(
        backend=_StubBackend([{"prompt_tokens": 10, "cached_tokens": 10}]))
    ep = _due_ep()
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="kv4.a", payload=_payload(sys_content="A"),
        status="ok", input_tokens=100, cached_tokens=0, now=0.0)
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="kv4.b", payload=_payload(sys_content="B"),
        status="ok", input_tokens=100, cached_tokens=0, now=1.0)
    # One other request pushes BOTH tracked prefixes' countdowns past the
    # trigger (100) at once.
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="other", payload={}, status="ok",
        input_tokens=200, cached_tokens=0, now=2.0)
    assert len(state.prefix_keepalive.due_for_touch("tier3", ep, now=2.0)) == 2, (
        "both prefixes must genuinely be due before the schedule pass runs")

    health = Health(state)
    health._schedule_prefix_keepalive_touches("tier3", ep)
    await asyncio.sleep(0)  # let the dispatched task reach record_dispatch
    assert len(state.backend.calls) == 1, "at most one touch per endpoint per tick"
    still_due = state.prefix_keepalive.due_for_touch("tier3", ep, now=2.0)
    assert len(still_due) == 1, "the other prefix must remain due for the next tick"
    # Let the in-flight touch finish so the test leaves nothing pending.
    await asyncio.gather(*state.prefix_keepalive_tasks.values())


@pytest.mark.asyncio
async def test_a_running_touch_on_this_endpoint_counts_against_the_gate():
    """`extra_occupied` passed to `Scheduler.background_available` must be
    THIS endpoint's own in-flight touch count, so a touch is never invisible
    to the same gate real background traffic is held to — independent of
    (and in addition to) the task-dict single-flight guard."""
    calls: list[int] = []

    def _bg(ep, extra_occupied=0):
        calls.append(extra_occupied)
        return max(0, 1 - extra_occupied)  # exactly one background slot, total

    state = _health_state(
        backend=_StubBackend([{"prompt_tokens": 10, "cached_tokens": 10}]),
        scheduler=types.SimpleNamespace(
            endpoint_snapshot=lambda ep: {"queued": 0},
            background_available=_bg))
    ep = _due_ep()
    # Prefix A: simulated as ALREADY mid-touch (no task ever created for it —
    # isolates the `extra_occupied` wiring from the task-dict guard above).
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="kv4.a", payload=_payload(sys_content="A"),
        status="ok", input_tokens=100, cached_tokens=0, now=0.0)
    entry_a = next(iter(state.prefix_keepalive._by_endpoint["tier3"].values()))
    entry_a.touch_in_flight = True
    # Prefix B: a DIFFERENT tracked prefix that IS due.
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="kv4.b", payload=_payload(sys_content="B"),
        status="ok", input_tokens=100, cached_tokens=0, now=1.0)
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="other", payload={}, status="ok",
        input_tokens=200, cached_tokens=0, now=2.0)

    health = Health(state)
    health._schedule_prefix_keepalive_touches("tier3", ep)
    assert calls == [1], "extra_occupied must equal prefix A's in-flight touch count"
    assert state.backend.calls == [], (
        "the gate (1 slot total, 1 already occupied) must refuse prefix B's touch")


@pytest.mark.asyncio
async def test_a_due_prefix_on_a_healthy_endpoint_with_room_is_touched():
    state = _health_state(
        backend=_StubBackend([{"prompt_tokens": 1000, "cached_tokens": 950}]))
    ep = _due_ep()
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="kv4.judge", payload=_payload(), status="ok",
        input_tokens=100, cached_tokens=0, now=0.0)
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="other", payload={}, status="ok",
        input_tokens=200, cached_tokens=0, now=1.0)
    health = Health(state)
    await _tick_and_await(health, "tier3", ep)
    assert len(state.backend.calls) == 1
    payload = state.backend.calls[0]
    assert payload["messages"][-1]["content"].startswith("keepalive ")
    snap = state.prefix_keepalive.status_snapshot("tier3", 1.0)
    assert snap["touches_sent"] == 1
    assert snap["touches_hit"] == 1, "950/1000 = 95% >= HIT_RATIO (90%)"
    assert snap["tracked"][0]["tokens_since_touch"] == 0, (
        "dispatch resets the countdown")


@pytest.mark.asyncio
async def test_a_mostly_uncached_touch_is_a_miss_not_a_hit():
    state = _health_state(
        backend=_StubBackend([{"prompt_tokens": 1000, "cached_tokens": 100}]))
    ep = _due_ep()
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="kv4.judge", payload=_payload(), status="ok",
        input_tokens=100, cached_tokens=0, now=0.0)
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="other", payload={}, status="ok",
        input_tokens=200, cached_tokens=0, now=1.0)
    health = Health(state)
    await _tick_and_await(health, "tier3", ep)
    snap = state.prefix_keepalive.status_snapshot("tier3", 1.0)
    assert snap["touches_missed"] == 1
    assert snap["touches_hit"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cached,expect_hit", [
    (1000, True),                                   # 100% cached
    (int(1000 * HIT_RATIO), True),                  # exactly at HIT_RATIO
    (int(1000 * HIT_RATIO) - 1, False),              # one token below it
])
async def test_the_hit_verdict_is_exactly_HIT_RATIO_of_the_touchs_OWN_prompt(
    cached, expect_hit,
):
    state = _health_state(
        backend=_StubBackend([{"prompt_tokens": 1000, "cached_tokens": cached}]))
    ep = _due_ep()
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="kv4.judge", payload=_payload(), status="ok",
        input_tokens=100, cached_tokens=0, now=0.0)
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="other", payload={}, status="ok",
        input_tokens=200, cached_tokens=0, now=1.0)
    health = Health(state)
    await _tick_and_await(health, "tier3", ep)
    snap = state.prefix_keepalive.status_snapshot("tier3", 1.0)
    assert (snap["touches_hit"] == 1) is expect_hit
    assert (snap["touches_missed"] == 1) is not expect_hit


@pytest.mark.asyncio
async def test_a_backend_that_never_answers_is_a_miss_never_an_exception():
    state = _health_state(backend=_StubBackend([None]))  # "cannot tell"
    ep = _due_ep()
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="kv4.judge", payload=_payload(), status="ok",
        input_tokens=100, cached_tokens=0, now=0.0)
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="other", payload={}, status="ok",
        input_tokens=200, cached_tokens=0, now=1.0)
    health = Health(state)
    await _tick_and_await(health, "tier3", ep)   # must not raise
    snap = state.prefix_keepalive.status_snapshot("tier3", 1.0)
    assert snap["touches_missed"] == 1


@pytest.mark.asyncio
async def test_a_touch_already_in_flight_is_not_dispatched_twice():
    """Two poller passes back to back while the first touch is still
    running must fire only ONE backend call."""
    class _SlowBackend:
        def __init__(self):
            self.calls = 0
            self.gate = asyncio.Event()

        async def probe_prefix_touch(self, ep_cfg, payload, timeout_s):
            self.calls += 1
            await self.gate.wait()
            return {"prompt_tokens": 10, "cached_tokens": 10}

    backend = _SlowBackend()
    state = _health_state(backend=backend)
    ep = _due_ep()
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="kv4.judge", payload=_payload(), status="ok",
        input_tokens=100, cached_tokens=0, now=0.0)
    state.prefix_keepalive.observe_completion(
        "tier3", ep, call_site="other", payload={}, status="ok",
        input_tokens=200, cached_tokens=0, now=1.0)
    health = Health(state)
    health._schedule_prefix_keepalive_touches("tier3", ep)
    health._schedule_prefix_keepalive_touches("tier3", ep)  # second pass, same tick
    await asyncio.sleep(0)  # let both scheduling calls run
    assert backend.calls == 1
    backend.gate.set()
    await asyncio.gather(*state.prefix_keepalive_tasks.values())


# --------------------------------------------------------------------------- #
# 7. End to end through the real ProxyService — proves the lifecycle wiring
#    (record_completion) actually reaches the tracker on a REAL completed
#    request, not just on a hand-built call.
# --------------------------------------------------------------------------- #

class _Req:
    class _Client:
        host = "127.0.0.1"

    client = _Client()
    headers: dict = {}


def _submit_body(call_site, tools=True):
    payload = {"messages": [dict(_SYS), {"role": "user", "content": "hi"}]}
    if tools:
        payload["tools"] = list(_TOOLS)
    return {
        "agent_id": "kv4", "endpoint": "tier3", "priority": "P3_INGESTION",
        "call_site": call_site, "payload_type": "chat_completion",
        "payload": payload, "timeout_s": 15.0,
    }


def _fake_resp(prompt_tokens, cached_tokens):
    async def fake_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"role": "assistant", "content": "ok"},
                               "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": prompt_tokens,
                           "completion_tokens": 1,
                           "prompt_tokens_details": {"cached_tokens": cached_tokens}}},
            duration_s=0.01, input_tokens=prompt_tokens, output_tokens=1,
            finish_reason="stop", cached_tokens=cached_tokens)
    return fake_call


@pytest.mark.asyncio
async def test_a_real_completed_request_reaches_the_tracker_and_a_due_touch_fires():
    svc = ProxyService(ProxyConfig())
    ep_cfg = svc._state.config.endpoints["tier3"]
    ep_cfg.prefix_keepalive_call_sites = ("kv4.judge",)
    ep_cfg.prefix_keepalive_trigger_tokens = 500
    ep_cfg.prefix_keepalive_idle_s = 0
    ep_cfg.max_slots = 4

    await svc.startup()
    try:
        # Turn 1: the call_site this endpoint keeps warm. Captures a skeleton.
        svc._backend.call = _fake_resp(prompt_tokens=5000, cached_tokens=100)
        resp = await asyncio.wait_for(
            svc.handle_submit(_submit_body("kv4.judge"), _Req()), timeout=5.0)
        assert resp.status_code == 200

        snap = svc._state.prefix_keepalive.status_snapshot("tier3", 0.0)
        assert snap is not None and len(snap["tracked"]) == 1
        assert snap["tracked"][0]["call_site"] == "kv4.judge"
        assert snap["tracked"][0]["tokens_since_touch"] == 0

        # Turn 2: a DIFFERENT, non-matching call_site with 1000 uncached
        # tokens — enough to cross the 500-token trigger.
        svc._backend.call = _fake_resp(prompt_tokens=1000, cached_tokens=0)
        resp2 = await asyncio.wait_for(
            svc.handle_submit(_submit_body("other.thing", tools=False), _Req()),
            timeout=5.0)
        assert resp2.status_code == 200

        snap = svc._state.prefix_keepalive.status_snapshot("tier3", 0.0)
        assert snap["tracked"][0]["tokens_since_touch"] == 1000

        # Now fire the poller-side schedule pass directly (no need to wait
        # out the real poller cadence) and await the touch it creates.
        svc._backend.probe_prefix_touch = _StubBackend(
            [{"prompt_tokens": 4800, "cached_tokens": 4750}]).probe_prefix_touch
        svc._health._schedule_prefix_keepalive_touches("tier3", ep_cfg)
        tasks = list(svc._state.prefix_keepalive_tasks.values())
        assert len(tasks) == 1, "one due prefix must produce exactly one task"
        await asyncio.gather(*tasks)

        snap = svc._state.prefix_keepalive.status_snapshot("tier3", 0.0)
        assert snap["touches_sent"] == 1
        assert snap["touches_hit"] == 1
        assert snap["tracked"][0]["tokens_since_touch"] == 0
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_an_endpoint_with_no_policy_declared_never_tracks_a_real_request():
    """Absent policy ⇒ inert, through the REAL lifecycle path — the strongest
    form of the "byte-identical when unconfigured" claim."""
    svc = ProxyService(ProxyConfig())
    await svc.startup()
    try:
        svc._backend.call = _fake_resp(prompt_tokens=5000, cached_tokens=0)
        await asyncio.wait_for(
            svc.handle_submit(_submit_body("kv4.judge"), _Req()), timeout=5.0)
        assert svc._state.prefix_keepalive.status_snapshot("tier3", 0.0) is None
    finally:
        await svc.shutdown()
