"""§ 9 — tier3 → tier2-analyst failover.

Covers the tests § 9.9 of `docs/anvil2_tier3_deepseek_v4_flash_plan_2026-08.md`
says this change owes, plus the two "silently dropped config key" guards the
surrounding code warns about in its own comments.

Built on a REAL ProxyService (real catalog, real config, real scheduler, real
Health) rather than a stub state. The whole failover is a routing decision made
from live config and live health, so a stub would mostly assert that the stub
was built correctly — and the two most likely failure modes here are exactly the
ones a stub hides: a config key that never reaches its dataclass, and an
endpoint name that resolves differently than the test assumed.
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]  # originfleet/
sys.path.insert(0, str(REPO))

from roadstead import failover as failover_mod  # noqa: E402
from roadstead.config import (  # noqa: E402
    LLMPriority,
    load_agent_configs,
    PriorityBand,
    ProxyConfig,
    normalize_endpoint,
)
from roadstead.model_catalog import build_endpoint_kwargs, load_catalog  # noqa: E402
from roadstead.scheduler import QueuedRequest  # noqa: E402
from roadstead.service import ProxyService  # noqa: E402

SRC = "thinker"      # tier3
TGT = "creative"     # tier2-analyst


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def svc(tmp_path):
    """A real ProxyService, never started (no loops, no sockets)."""
    # `agents=load_agent_configs()` mirrors __main__.py's boot: ProxyConfig
    # defaults `agents` to {}, so WITHOUT this every agent reads degrade_ok
    # False and the whole opt-in half of this suite would pass vacuously.
    s = ProxyService(ProxyConfig(
        queue_db_path=str(tmp_path / "queue.db"),
        agents=load_agent_configs(),
    ))
    yield s
    try:
        s._state.queue_db.close()
    except Exception:  # noqa: BLE001 — teardown best-effort
        pass


def _sick(svc, ep=SRC):
    """Trip ``ep``'s circuit the way the poller does."""
    svc._state.endpoint_health[ep] = {
        "healthy": False, "consecutive_failures": 3, "unhealthy_since": time.monotonic()}


def _well(svc, ep=SRC):
    svc._state.endpoint_health[ep] = {
        "healthy": True, "consecutive_failures": 0, "unhealthy_since": None}


def _req(agent="discord", endpoint=SRC, tokens=50, max_tokens=256,
         priority=LLMPriority.P1_TURN_SUPPORT):
    """A chat request whose estimated input size is controllable.

    ``estimate_input_tokens`` works off the payload's character count, so the
    prompt is sized in characters and the assertions below are written against
    the ESTIMATE, never against a hand-computed token count.
    """
    return QueuedRequest.create(
        agent_id=agent,
        endpoint=endpoint,
        priority=priority,
        call_site=f"{agent}.test",
        payload_type="chat_completion",
        payload={"model": endpoint, "max_tokens": max_tokens,
                 "messages": [{"role": "user", "content": "x" * (tokens * 4)}]},
        request_id=f"req_{uuid.uuid4().hex[:12]}",
    )


# ---------------------------------------------------------------------------
# config plumbing — the two keys that get silently dropped
# ---------------------------------------------------------------------------

def test_degrade_ok_reaches_agent_config():
    """`degrade_ok` in agents.yaml must reach AgentQuotaConfig.

    load_agent_configs() parses an explicit ALLOWLIST of keys; a knob added to
    the dataclass but not to that parser is settable in code and unreachable by
    the operator — which looks identical to a policy decision that the agent is
    not opted in.
    """
    cfgs = load_agent_configs()
    assert cfgs["discord"].degrade_ok is True, (
        "the day-1 opt-in seed did not survive the agents.yaml parser")
    # beacon joined the set on 2026-08-29 (operator decision, after a tier3
    # outage refused every one of its turns). Pinned by NAME because the agent
    # id is what Gate 1 reads: llmproxy/acl.py registers 10.0.0.23 as "beacon",
    # and a rename there would silently opt it back out.
    assert cfgs["beacon"].degrade_ok is True, (
        "beacon lost its tier3 failover opt-in — during a tier3 outage every "
        "Beacon turn goes back to a hard 503 instead of degrading to "
        "tier2-analyst")
    # Default-deny is the property that matters most, so assert it on a real
    # agent that IS in the file (i.e. the parser ran for it) rather than on an
    # absent key, which would pass even if the parser were dead.
    assert cfgs["sidekick"].degrade_ok is False


def test_dwell_reaches_endpoint_config():
    """`policy.failover_dwell_s` must reach EndpointConfig.

    build_endpoint_kwargs() copies policy keys through a hardcoded tuple; a key
    missing from it is dropped with no error, and the dwell would silently fall
    back to the dataclass default. models.yaml says so in its own comment.
    """
    kw = build_endpoint_kwargs()
    assert kw[SRC]["failover_dwell_s"] == 120


def test_fallback_resolves_to_a_failover_target():
    """The tier3 stanza's `fallback:` must arm a real failover pair."""
    eps = ProxyConfig().endpoints
    assert eps[SRC].failover_to == TGT
    # ...and the target must not itself declare one: failover never chains, and
    # a reciprocal pair would read as a loop to anyone auditing it.
    assert eps[TGT].failover_to == ""


def test_declared_fallbacks_resolve_to_live_endpoints():
    """DOCTRINE (§ 9.9). Every non-null `fallback:` on a routable stanza must
    name a LIVE, ACTIVE proxy endpoint.

    `fallback:` was advisory metadata nothing consumed for months, and the value
    it carried (`tier3_backup`) pointed at a stanza with `proxy_endpoint: false`
    — a declared backup that physically could not serve. The moment a field
    becomes load-bearing it needs a guard, or it rots back into a comment and
    the failover target silently becomes nothing.
    """
    cat = load_catalog()
    live = {e.endpoint_class for e in cat.proxy_endpoints()
            if e.endpoint_class and e.status == "active"}
    # Scoped to PROXY endpoints: those are the only stanzas whose `fallback:`
    # can arm a failover (build_endpoint_kwargs only walks proxy_endpoints).
    # A non-routable stanza's fallback stays advisory and is covered by
    # test_model_naming_doctrine.py::test_catalog_loads_and_is_consistent,
    # which asserts only that it resolves to something.
    for e in cat.proxy_endpoints():
        if not e.fallback:
            continue
        target = cat.entry(e.fallback)
        assert target is not None, (
            f"{e.name}: fallback {e.fallback!r} resolves to nothing")
        assert target.proxy_endpoint, (
            f"{e.name}: fallback {e.fallback!r} is not a proxy endpoint — it "
            f"cannot serve, so this arms a failover at a dead backend")
        assert target.endpoint_class in live, (
            f"{e.name}: fallback {e.fallback!r} -> class "
            f"{target.endpoint_class!r} is not live+active")
        assert target.endpoint_class != e.endpoint_class, (
            f"{e.name}: fallback points at its own endpoint class")


def test_failover_does_not_touch_endpoint_name_resolution():
    """REGRESSION (§ 9.3, ledger `endpoint-class-alias-collision`).

    The failover must NOT be implemented as an alias or a name rewrite. The
    submit path normalizes twice (resolve_endpoint, then QueuedRequest.create),
    so a non-idempotent normalize silently served every `tier3-backup` request
    from tier3 itself. Assert the property directly, and assert that a rerouted
    request's endpoint still round-trips.
    """
    for name in ("thinker", "creative", "tier3", "tier2", "tier2-analyst",
                 "reasoner", "companion"):
        once = normalize_endpoint(name)
        assert normalize_endpoint(once) == once, f"{name} is not idempotent"
    # `creative` must not have acquired an alias pointing back at the source.
    assert normalize_endpoint(TGT) == TGT
    assert normalize_endpoint(SRC) == SRC


# ---------------------------------------------------------------------------
# the state machine
# ---------------------------------------------------------------------------

def test_enters_degraded_only_when_source_sick_and_target_well(svc):
    fo = svc._state.failover
    fo.refresh()
    assert svc._state.degraded_endpoints == set()

    _sick(svc)
    fo.refresh()
    assert SRC in svc._state.degraded_endpoints

    # Target goes sick too -> we must not have entered in the first place.
    svc._state.degraded_endpoints.clear()
    svc._state.degraded_since.clear()
    _sick(svc, TGT)
    fo.refresh()
    assert SRC not in svc._state.degraded_endpoints, (
        "entered degraded mode with the failover target also down — that "
        "announces a degradation that cannot happen")


def test_an_operator_drain_trips_degraded_mode(svc):
    """An operator DRAIN reads unhealthy, so failover fires during planned
    maintenance. That is the intended behaviour and the biggest operational win
    here (a ~10-minute tier3 cold start becomes a degradation, not an outage) —
    pinned so nobody 'fixes' it later.
    """
    svc._state.paused_endpoints.add(SRC)
    svc._state.failover.refresh()
    assert SRC in svc._state.degraded_endpoints


def test_recovery_requires_health_and_drain_and_dwell(svc):
    """§ 9.7 — all THREE conditions, not any one of them."""
    fo = svc._state.failover
    _sick(svc)
    fo.refresh()
    t0 = svc._state.degraded_since[SRC]
    dwell = svc._state.config.endpoints[SRC].failover_dwell_s

    # 1. Healthy again, but still inside the dwell -> stay degraded.
    _well(svc)
    fo.refresh(t0 + dwell - 1)
    assert SRC in svc._state.degraded_endpoints, "left before the dwell elapsed"

    # 2. Past the dwell but a degraded request is still in flight -> stay.
    req = _req()
    fo.apply(req, TGT)
    svc._state.scheduler.enqueue(req)
    assert svc._state.scheduler.degraded_inflight(SRC) == 1
    fo.refresh(t0 + dwell + 1)
    assert SRC in svc._state.degraded_endpoints, "left before the cohort drained"

    # 3. Drained AND past the dwell -> leave.
    svc._state.scheduler.cancel(req.request_id)
    assert svc._state.scheduler.degraded_inflight(SRC) == 0
    fo.refresh(t0 + dwell + 1)
    assert SRC not in svc._state.degraded_endpoints

    # 4. And it does not leave while still unhealthy, however long it waits.
    _sick(svc)
    fo.refresh(t0 + dwell + 2)
    assert SRC in svc._state.degraded_endpoints
    fo.refresh(t0 + dwell * 10)
    assert SRC in svc._state.degraded_endpoints


# ---------------------------------------------------------------------------
# the two admission gates (§ 9.4 / § 9.5)
# ---------------------------------------------------------------------------

def test_opted_in_agent_is_rerouted(svc):
    _sick(svc)
    fo = svc._state.failover
    fo.refresh()
    plan = fo.plan(_req(agent="discord"))
    assert plan.rerouted and plan.target == TGT


def test_agent_without_degrade_ok_is_refused_not_degraded(svc):
    """DEFAULT-DENY. The branch most likely to rot into always-true."""
    _sick(svc)
    fo = svc._state.failover
    fo.refresh()
    for agent in ("knowledge_store", "sidekick", "some_agent_that_has_no_stanza"):
        req = _req(agent=agent)
        plan = fo.plan(req)
        assert not plan.rerouted, f"{agent} was degraded without opting in"
        assert plan.refusal_code == failover_mod.CODE_NOT_OPTED_IN
        fo.record_refusal(req, plan)
    assert (svc._state.degraded_refused[SRC][failover_mod.CODE_NOT_OPTED_IN]
            == 3), "refusals must be counted — an invisible gate is a dead one"


def test_plan_is_pure_and_does_not_count(svc):
    """plan() must not touch the counters.

    A BACKGROUND request that fails both gates is NOT refused — it falls
    through and queues on the unhealthy endpoint to defer until recovery, as it
    did before failover existed. If plan() counted, every one of those would be
    tallied as a refusal, and `degraded_refused` is precisely the number the
    operator reads to decide the opt-in set is too small.
    """
    _sick(svc)
    fo = svc._state.failover
    fo.refresh()
    for _ in range(5):
        assert fo.plan(_req(agent="knowledge_store")).refusal_code is not None
    assert svc._state.degraded_refused == {}, (
        "plan() counted a refusal nobody made")


def test_oversized_request_is_refused_not_truncated(svc):
    """§ 9.4 gate 1 — physics, not policy.

    Sized against the TARGET's live per-slot context, read from config rather
    than hardcoded: that number moved when the boxa's model was swapped
    (2026-08-19) and the plan doc's copy of it is already stale.
    """
    _sick(svc)
    fo = svc._state.failover
    fo.refresh()
    ctx = svc._state.config.endpoints[TGT].context_per_slot
    assert ctx > 0

    fits = _req(agent="discord", tokens=ctx // 4, max_tokens=256)
    assert fo.plan(fits).rerouted, "a request that fits was refused"

    too_big = _req(agent="discord", tokens=ctx + 5_000, max_tokens=256)
    plan = fo.plan(too_big)
    assert not plan.rerouted
    assert plan.refusal_code == failover_mod.CODE_CONTEXT_OVERFLOW
    # The refusal must NOT have shortened anything: a silently truncated prompt
    # answers a different question than the one that was asked.
    assert len(too_big.payload["messages"][0]["content"]) == (ctx + 5_000) * 4
    assert too_big.endpoint == SRC and too_big.degraded_from is None


def test_max_tokens_counts_toward_the_fit(svc):
    """input + max_tokens, the same predicate as the M1 context gate on the
    normal path — so the two cannot disagree about what fits."""
    _sick(svc)
    fo = svc._state.failover
    fo.refresh()
    ctx = svc._state.config.endpoints[TGT].context_per_slot
    near = _req(agent="discord", tokens=(ctx - 1000) // 4, max_tokens=64)
    assert fo.plan(near).rerouted
    same_input_huge_output = _req(
        agent="discord", tokens=(ctx - 1000) // 4, max_tokens=ctx)
    assert not fo.plan(same_input_huge_output).rerouted


def test_no_reroute_when_not_degraded(svc):
    """A healthy fleet must never execute a reroute, whatever the opt-in says."""
    fo = svc._state.failover
    fo.refresh()
    assert not fo.plan(_req(agent="discord")).rerouted


# ---------------------------------------------------------------------------
# apply() — what the reroute actually changes
# ---------------------------------------------------------------------------

def test_apply_repoints_routing_and_records_the_origin(svc):
    _sick(svc)
    fo = svc._state.failover
    fo.refresh()
    req = _req(agent="discord")
    deadline_before = req.timeout_deadline
    fo.apply(req, TGT)

    # The routing key moved, so every downstream consumer (queue, occupancy,
    # cost, dispatch's backend lookup, the served-model reported back) resolves
    # against the target with no special case.
    assert req.endpoint == TGT
    assert req.degraded_from == SRC
    assert req.ctx_per_slot_at_admission == svc._state.config.endpoints[TGT].context_per_slot
    # The deadline is deliberately NOT re-derived — it came from the SOURCE's
    # floors, and the source is the slower tier, so it is generous for the
    # target (wrong in the safe direction) and may be a caller's own contract.
    assert req.timeout_deadline == deadline_before
    assert svc._state.degraded_rerouted[SRC] == 1


def test_rerouted_request_queues_and_is_costed_on_the_target(svc):
    """The DRR scheduler is the contention mechanism (§ 9.4): rerouted work must
    land in the TARGET's queue and be accounted there, not stay on the dead
    endpoint's queue where nothing dispatches it."""
    _sick(svc)
    fo = svc._state.failover
    fo.refresh()
    req = _req(agent="discord")
    fo.apply(req, TGT)
    svc._state.scheduler.enqueue(req)
    assert svc._state.scheduler.queue_depth(TGT) == 1
    assert svc._state.scheduler.queue_depth(SRC) == 0
    assert svc._state.scheduler.degraded_inflight(SRC) == 1


# ---------------------------------------------------------------------------
# § 9.2 — fast_fail_interactive reroutes rather than releases
# ---------------------------------------------------------------------------

def test_fast_fail_interactive_reroutes_an_opted_in_request(svc):
    """The cohort already queued when the circuit trips never reaches the
    submit-path hook (that ran while the endpoint was still healthy), so
    without this they are the one group that fails during the very outage the
    failover exists to absorb."""
    st = svc._state
    req = _req(agent="discord", priority=LLMPriority.P0_REALTIME)
    assert req.band == PriorityBand.INTERACTIVE
    st.scheduler.enqueue(req)
    released: list = []
    st.resolve_error = lambda r, e: released.append((r.request_id, e))

    _sick(svc)
    svc._health.fast_fail_interactive(SRC)

    assert released == [], "an eligible request was released instead of rerouted"
    assert req.endpoint == TGT and req.degraded_from == SRC
    assert st.scheduler.queue_depth(TGT) == 1
    assert st.scheduler.queue_depth(SRC) == 0


def test_fast_fail_interactive_still_releases_a_refused_request(svc):
    st = svc._state
    req = _req(agent="knowledge_store", priority=LLMPriority.P0_REALTIME)
    st.scheduler.enqueue(req)
    released: list = []
    st.resolve_error = lambda r, e: released.append((r.request_id, e))

    _sick(svc)
    svc._health.fast_fail_interactive(SRC)

    assert len(released) == 1
    assert req.endpoint == SRC and req.degraded_from is None
    msg = released[0][1]
    # The message must say WHY it could not be degraded, not just that the
    # circuit is open — otherwise the opt-in set being too small is invisible.
    assert "degrade_ok" in msg
    # ...APPENDED to the pre-existing message, not replacing it: that base text
    # carries the marker `is_deferrable_llm_error` sniffs for, so substituting
    # it would turn a retryable deferral into a hard error for every caller
    # that could not be degraded.
    assert msg.startswith(f"backend {SRC} unavailable (circuit open)")


# ---------------------------------------------------------------------------
# § 9.6 — visibility
# ---------------------------------------------------------------------------

def test_status_exposes_a_set_not_a_per_endpoint_bool(svc):
    fo = svc._state.failover
    _sick(svc)
    fo.refresh()
    refused = _req(agent="knowledge_store")         # a refusal
    fo.record_refusal(refused, fo.plan(refused))
    req = _req(agent="discord")
    fo.apply(req, TGT)                            # a reroute

    s = fo.status()
    assert s["degraded_endpoints"] == [SRC]
    assert s["failover_pairs"] == {SRC: TGT}
    assert s["degraded_rerouted"] == {SRC: 1}
    assert s["degraded_refused"][SRC][failover_mod.CODE_NOT_OPTED_IN] == 1
    # 🚨 No per-endpoint `degraded` bool anywhere in the payload: the existing
    # `paused` bool is already known-unreliable and a second one with the same
    # bug just gives the operator a second thing to disbelieve.
    assert not any(isinstance(v, bool) for v in s.values())


def test_status_is_empty_and_cheap_when_healthy(svc):
    s = svc._state.failover.status()
    assert s["degraded_endpoints"] == []
    assert s["degraded_rerouted"] == {}
    assert s["degraded_refused"] == {}


def test_failover_target_is_never_on_demand(svc):
    """An ON-DEMAND endpoint must never be armed as a failover target.

    Ordering, not policy: handle_submit runs on_demand.ensure_loaded() for the
    endpoint the request ARRIVED for, and the reroute happens after that, at the
    circuit-breaker branch. A rerouted request would therefore be dispatched at
    a model nobody asked to load — a dispatch into an unloaded backend, which is
    not an error anyone would see as one.

    Nothing is on-demand today, so this is pinned by CONSTRUCTION rather than by
    observation: force the target on-demand and assert the pair disappears. A
    test that merely asserted "no current target is on-demand" would pass
    forever without the guard existing at all.
    """
    st = svc._state
    assert st.failover.pairs() == {SRC: TGT}
    real_manages = st.on_demand.manages
    st.on_demand.manages = lambda ep: ep == TGT
    try:
        assert st.failover.pairs() == {}, (
            "an on-demand endpoint was armed as a failover target")
        _sick(svc)
        st.failover.refresh()
        assert SRC not in st.degraded_endpoints
        assert not st.failover.plan(_req(agent="discord")).rerouted
    finally:
        st.on_demand.manages = real_manages


def test_a_background_request_is_not_counted_as_refused(svc):
    """The counting bug this design nearly shipped, pinned.

    Failover is evaluated for EVERY band (a background request that can be
    served now should be), but a background request that fails both gates is
    not turned away — it falls through the refusal branch and queues on the
    unhealthy endpoint to defer until recovery, exactly as before. Counting it
    would make `degraded_refused` answer a slightly different question than the
    one it is read for.
    """
    st = svc._state
    bg = _req(agent="knowledge_store", priority=LLMPriority.P3_INGESTION)
    assert bg.band == PriorityBand.BACKGROUND
    _sick(svc)
    st.failover.refresh()
    plan = st.failover.plan(bg)
    assert plan.refusal_code is not None      # it IS ineligible...
    # ...but nothing records it, because nothing refused it.
    assert st.degraded_refused == {}
