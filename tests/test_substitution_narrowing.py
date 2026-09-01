"""A request may DECLINE a substitution it was granted. It may never grant one.

Workstream C added two per-request flags — ``substitution.degrade`` and
``substitution.spill`` — beside the two the operator sets on an identity
(``degrade_ok``, ``spill_ok``). The whole of the doctrine is the direction:

🚨 **The operator GRANTS and the caller may only DECLINE.** Both gates take the
AND. A request that could set either flag True would let a caller award itself a
permission its operator withheld — the self-asserted ``agent_id`` bug in a
different costume, and for ``spill`` the consequence is money spent on somebody's
behalf without their say-so, which is not recoverable by replying to it.

The other direction is genuinely useful and is why the flags exist at all: one
confidential prompt on an identity that is otherwise happy to spill, or one
answer that must come from the big model on an identity that normally accepts a
degraded one.

Both gates are covered in **all four** combinations, because the failure that
matters is asymmetric — a gate that ignored the request flag entirely still
passes three of the four.
"""
from __future__ import annotations

import pytest

from roadstead.config import (
    AgentQuotaConfig,
    EndpointConfig,
    LLMPriority,
    ProxyConfig,
)
from roadstead.failover import CODE_NOT_OPTED_IN
from roadstead.scheduler import Admission, QueuedRequest


# ---------------------------------------------------------------------------
# spill — scheduler._admit
# ---------------------------------------------------------------------------

def _spill_config(*, spill_ok: bool) -> ProxyConfig:
    src = EndpointConfig(endpoint_class="src", role="src", max_slots=1,
                         context_per_slot=32768, spill_to="dst")
    dst = EndpointConfig(endpoint_class="dst", role="dst", max_slots=4,
                         context_per_slot=32768)
    cfg = ProxyConfig(endpoints={"src": src, "dst": dst})
    cfg.agents = {"a": AgentQuotaConfig(agent_id="a", spill_ok=spill_ok)}
    return cfg


def _req(**kw) -> QueuedRequest:
    return QueuedRequest.create(
        agent_id="a", endpoint="src", priority=LLMPriority.P1_TURN_SUPPORT,
        call_site="t", payload_type="chat_completion",
        payload={"messages": [{"role": "user", "content": "hi"}]},
        **kw)


def _admit(cfg: ProxyConfig, req: QueuedRequest) -> Admission:
    from roadstead.agent_budget import BudgetManager
    from roadstead.cost_model import CostModel
    from roadstead.scheduler import Scheduler

    sched = Scheduler(cfg, CostModel(), BudgetManager())
    sched.may_spend = lambda agent_id: True
    src = cfg.endpoints["src"]
    # Local FULL — the only state in which spill is even a question. Spill is
    # overflow: it is never the first answer, so this is where the decision
    # genuinely lives.
    sched._active["src"] = {"other": object()}
    return sched._admit("src", src, req, current_occupancy=1,
                        band_available=0, spill_target="dst")


@pytest.mark.parametrize("granted,declined,expected", [
    (True,  None,  Admission.SPILL),   # granted, no opinion → spill
    (True,  True,  Admission.SPILL),   # granted, agreed     → spill
    (True,  False, Admission.DEFER),   # granted, DECLINED   → wait locally
    (False, True,  Admission.DEFER),   # 🚨 NOT granted, request says yes → NO
    (False, None,  Admission.DEFER),
    (False, False, Admission.DEFER),
])
def test_spill_takes_the_and_of_the_operator_and_the_request(
        granted, declined, expected):
    """The `(False, True)` row is the one that matters.

    Mutation: make the gate an OR (`spill_ok or req.allow_spill`) and only that
    row fails — which is exactly why it is here and why three-of-four coverage
    would have shipped the bug.
    """
    cfg = _spill_config(spill_ok=granted)
    assert _admit(cfg, _req(allow_spill=declined)) is expected


def test_declining_spill_is_a_defer_and_never_an_error():
    """🚨 A caller opting out of paid capacity is not a refusal. The request
    keeps its place and is served locally when a slot frees — the same DEFER an
    un-opted-in caller has always received. Turning it into an error would make
    a confidentiality preference cost the caller its answer."""
    cfg = _spill_config(spill_ok=True)
    assert _admit(cfg, _req(allow_spill=False)) is Admission.DEFER


def test_local_capacity_is_still_tried_first_whatever_the_request_says(monkeypatch):
    """🚨 Nothing about substitution — or money — may appear above the local
    dispatch line in `_admit`. A caller that declined spill must not thereby
    lose a free local slot."""
    from roadstead.agent_budget import BudgetManager
    from roadstead.cost_model import CostModel
    from roadstead.scheduler import Scheduler

    cfg = _spill_config(spill_ok=True)
    sched = Scheduler(cfg, CostModel(), BudgetManager())
    sched.may_spend = lambda agent_id: True
    for declined in (None, True, False):
        verdict = sched._admit("src", cfg.endpoints["src"],
                               _req(allow_spill=declined),
                               current_occupancy=0, band_available=1,
                               spill_target="dst")
        assert verdict is Admission.DISPATCH


# ---------------------------------------------------------------------------
# degrade — failover.plan
# ---------------------------------------------------------------------------

class _StubHealth:
    def endpoint_healthy(self, ep: str) -> bool:
        return ep != "src"          # source sick, target well


class _StubOnDemand:
    def manages(self, ep: str) -> bool:
        return False


def _failover(*, degrade_ok: bool):
    from roadstead.failover import Failover

    src = EndpointConfig(endpoint_class="src", role="src", max_slots=1,
                         context_per_slot=32768, failover_to="dst")
    dst = EndpointConfig(endpoint_class="dst", role="dst", max_slots=4,
                         context_per_slot=32768)
    cfg = ProxyConfig(endpoints={"src": src, "dst": dst})
    cfg.agents = {"a": AgentQuotaConfig(agent_id="a", degrade_ok=degrade_ok)}

    state = type("S", (), {})()
    state.config = cfg
    state.on_demand = _StubOnDemand()
    state.degraded_endpoints = {"src"}
    state.degraded_since = {}
    state.degraded_rerouted = {}
    state.degraded_refused = {}
    state.scheduler = type("Q", (), {"degraded_inflight": lambda self, ep: 0})()
    return Failover(state, _StubHealth())


@pytest.mark.parametrize("granted,declined,rerouted", [
    (True,  None,  True),
    (True,  True,  True),
    (True,  False, False),   # granted, DECLINED → refused, not degraded
    (False, True,  False),   # 🚨 NOT granted, request says yes → NO
    (False, None,  False),
    (False, False, False),
])
def test_degrade_takes_the_and_of_the_operator_and_the_request(
        granted, declined, rerouted):
    fo = _failover(degrade_ok=granted)
    plan = fo.plan(_req(allow_degrade=declined))
    assert plan.rerouted is rerouted
    if not rerouted:
        # 🚨 The SAME refusal code either way. From the caller's side the
        # outcome is identical — a clean labelled 503 rather than a smaller
        # model's answer — and a second code would ask every existing client to
        # learn a distinction it cannot act on differently.
        assert plan.refusal_code == CODE_NOT_OPTED_IN


def test_declining_degrade_says_which_side_declined():
    """The refusal message is what an operator reads to decide whether the
    opt-in set is too small. "This caller is not enrolled" and "this caller
    enrolled and declined this one request" are different findings."""
    fo = _failover(degrade_ok=True)
    plan = fo.plan(_req(allow_degrade=False))
    assert "substitution.degrade" in plan.refusal_detail
    other = _failover(degrade_ok=False).plan(_req())
    assert "degrade_ok" in other.refusal_detail


# ---------------------------------------------------------------------------
# The two are separate questions, and stay separate
# ---------------------------------------------------------------------------

def test_neither_flag_defaults_from_the_other():
    """🚨 `spill_to` is not `failover_to` and `spill_ok` is not `degrade_ok`.
    Failover asks "this backend is DOWN, may a worse model answer" — a quality
    judgement. Spill asks "this backend is BUSY, may we pay somebody else" — a
    confidentiality-and-money one. The instinct to collapse them recurs."""
    req = _req(allow_degrade=False, allow_spill=None)
    assert req.allow_degrade is False and req.allow_spill is None
    req2 = _req(allow_spill=False)
    assert req2.allow_spill is False and req2.allow_degrade is None

    # And an identity granted one is not thereby granted the other.
    a = AgentQuotaConfig(agent_id="x", degrade_ok=True)
    assert a.spill_ok is False
    b = AgentQuotaConfig(agent_id="y", spill_ok=True)
    assert b.degrade_ok is False


def test_the_flags_reach_the_request_from_the_enriched_body():
    """The other half of the allowlist guard the repo keeps re-learning: a knob
    that is settable and never reaches the thing it configures looks exactly
    like a knob that was never load-bearing."""
    from roadstead.enriched import _substitution_request

    assert _substitution_request({}) == (None, None)
    assert _substitution_request({"substitution": {"degrade": False}}) == (False, None)
    assert _substitution_request({"substitution": {"spill": False}}) == (None, False)
    assert _substitution_request(
        {"substitution": {"degrade": True, "spill": False}}) == (True, False)

    # …and end to end, from the body to the field the gates read.
    req = QueuedRequest.create(
        agent_id="a", endpoint="src", priority=LLMPriority.P1_TURN_SUPPORT,
        call_site="t", payload_type="chat_completion", payload={},
        allow_degrade=False, allow_spill=False)
    assert req.allow_degrade is False and req.allow_spill is False
