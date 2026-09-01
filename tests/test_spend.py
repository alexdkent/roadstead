"""Money — pricing, metering, spill and the degrading threshold (Workstream D).

Four doctrines are pinned here, each of which is a decision rather than an
implementation detail, and each of which has been observed going red:

1. **Real money and avoided cost are never summed.** ``usage_rates.py`` prices a
   LOCAL endpoint at what renting it would have cost — a saving, not a bill. A
   remote provider's published price is an invoice. Both are USD per million
   tokens and nothing in the type system separates them, so ``spend.py`` does.
2. **A threshold DEGRADES and never rejects**, and specifically never costs a
   caller local capacity. Crossing a cap takes one priority band and paid spill.
   That is the whole penalty.
3. **Admission is ONE decision with three outcomes.** Spill is not a second pass
   that runs after the first gives up; it is a third answer to the same question
   about the same request, and local capacity is tried first for everybody.
4. **Spill and failover are different questions.** `spill_ok` is not
   `degrade_ok`, `spill_to` is not `failover_to`, and neither implies the other.
"""

from __future__ import annotations

import time
from typing import Optional

import pytest

from roadstead import model_catalog
from roadstead.agent_budget import BudgetManager
from roadstead.config import (
    AgentQuotaConfig,
    LLMPriority,
    PriorityBand,
    ProxyConfig,
    load_agent_configs,
)
from roadstead.cost_model import CostModel
from roadstead.providers.openrouter import OPENROUTER
from roadstead.scheduler import Admission, QueuedRequest, Scheduler
from roadstead.spend import (
    SOURCE_CONFIG,
    SOURCE_IMPUTED,
    SOURCE_PROVIDER,
    PriceBook,
    SpendLedger,
    SpendStanding,
    TokenPrice,
    day_bucket,
    declared_price,
    standing,
)

#: A plausible remote price, in the unit providers publish: USD per SINGLE
#: token, as a string. See `OpenRouterProvider._parse_pricing`.
_REMOTE_PROMPT = "0.0000005"      # $0.50 / Mtok
_REMOTE_COMPLETION = "0.0000015"  # $1.50 / Mtok

_REAL = TokenPrice(input_usd_per_mtok=0.5, output_usd_per_mtok=1.5,
                   source=SOURCE_PROVIDER, detail="test")


# ===========================================================================
# 1. The two kinds of money
# ===========================================================================

def test_an_imputed_price_is_not_money():
    """🚨 The doctrine this module exists for.

    A local call's cost is what renting the same class of model WOULD have cost.
    Nobody is billed for it, so it may never reach the spent total and may never
    reach a threshold — a threshold that counted it would throttle a caller for
    using capacity that is free and already paid for, which is the exact
    inversion of local-first.
    """
    ledger = SpendLedger()
    # `tier3` is priced by usage_rates.py's avoided-cost table, so this is the
    # imputed path with no configuration at all.
    assert ledger.prices.price("tier3").source == SOURCE_IMPUTED
    assert ledger.prices.price("tier3").real is False

    billed = ledger.charge("a", "tier3", 1_000_000, 1_000_000, now=0.0)
    acct = ledger.get("a")
    assert billed == 0.0, "an imputed price billed the caller"
    assert acct.spent_usd == 0.0
    assert acct.day_spent_usd == 0.0
    assert acct.avoided_usd > 0.0, "the saving was not recorded either"
    assert ledger.spent_today("a", now=0.0) == 0.0
    # …and the tokens are counted regardless of who pays: attribution is not
    # billing, and a local-only deployment still wants to know its volumes.
    assert acct.tokens_in == 1_000_000 and acct.tokens_out == 1_000_000


def test_a_real_price_is_money_and_lands_in_the_other_column():
    ledger = SpendLedger()
    ledger.prices.declare("spill-chat", _REAL)
    billed = ledger.charge("a", "spill-chat", 1_000_000, 2_000_000, now=0.0)
    acct = ledger.get("a")
    assert billed == pytest.approx(0.5 + 3.0)
    assert acct.spent_usd == pytest.approx(3.5)
    assert acct.day_spent_usd == pytest.approx(3.5)
    assert acct.avoided_usd == 0.0, "real spend leaked into the avoided column"
    assert acct.by_endpoint == {"spill-chat": pytest.approx(3.5)}


def test_the_two_totals_are_never_combined_by_the_ledger():
    """The counterweight: a caller doing both kinds of work keeps both numbers,
    separately, and no accessor adds them. Anything wanting a total has to say
    so at its own call site, where somebody can see it deciding."""
    ledger = SpendLedger()
    ledger.prices.declare("spill-chat", _REAL)
    ledger.charge("a", "tier3", 1_000_000, 0, now=0.0)        # imputed
    ledger.charge("a", "spill-chat", 1_000_000, 0, now=0.0)   # real
    acct = ledger.get("a")
    assert acct.spent_usd == pytest.approx(0.5)
    assert acct.avoided_usd > 0.0
    assert not hasattr(acct, "total_usd"), (
        "a combined total appeared — it is neither a bill nor a saving, and an "
        "operator reading it to decide whether a caller costs too much would be "
        "reading a number that means nothing")


# ===========================================================================
# 2. The price book's precedence
# ===========================================================================

def test_a_declared_price_beats_a_discovered_one():
    """🚨 An operator who wrote a number down knows something the catalogue does
    not — a negotiated rate, an internal chargeback. The reverse precedence
    would make the declaration unreachable, which is the `_POLICY_PASSTHROUGH`
    failure in a different costume: a knob that is settable and has no effect."""
    book = PriceBook()
    book.observe("spill-chat", TokenPrice(9.0, 9.0, source=SOURCE_PROVIDER))
    book.declare("spill-chat", TokenPrice(1.0, 1.0, source=SOURCE_CONFIG))
    assert book.price("spill-chat").input_usd_per_mtok == 1.0
    # …and the order of the two calls does not matter, because "most recently
    # written" is not the rule.
    book2 = PriceBook()
    book2.declare("spill-chat", TokenPrice(1.0, 1.0, source=SOURCE_CONFIG))
    book2.observe("spill-chat", TokenPrice(9.0, 9.0, source=SOURCE_PROVIDER))
    assert book2.price("spill-chat").input_usd_per_mtok == 1.0


def test_an_unpriced_endpoint_falls_through_to_imputed_and_never_to_none():
    book = PriceBook()
    assert book.price("tier3").source == SOURCE_IMPUTED
    # An endpoint even the imputed table does not know prices at zero, which is
    # what it has always done — not a crash and not a None to be checked for.
    unknown = book.price("no-such-endpoint")
    assert unknown.cost_usd(1_000_000, 1_000_000) == 0.0


def test_half_a_declaration_is_a_whole_declaration():
    """An operator who priced only the input side has said something true and
    specific. Filling the other half from the imputed table would mix a real
    rate and an avoided-cost one inside ONE price — the confusion this module is
    built to prevent, arriving through the back door."""
    class _Ep:
        input_usd_per_mtok = 2.0
        output_usd_per_mtok = None

    price = declared_price("x", _Ep())
    assert price is not None
    assert price.real is True
    assert price.input_usd_per_mtok == 2.0
    assert price.output_usd_per_mtok == 0.0


def test_an_endpoint_with_no_declaration_declares_nothing():
    class _Ep:
        input_usd_per_mtok = None
        output_usd_per_mtok = None

    assert declared_price("x", _Ep()) is None


# ===========================================================================
# 3. Reading a provider's published prices
# ===========================================================================

def test_openrouter_pricing_is_parsed_from_strings_and_scaled_to_per_mtok():
    """The values are STRINGS and per SINGLE token — strings because
    0.0000005 is exactly the magnitude where a JSON float starts losing digits,
    and per-token because that is the unit the upstream bills in."""
    report = OPENROUTER.parse_capacity({
        "model_id": "vendor/model",
        "models": {"data": [
            {"id": "someone/else", "context_length": 8192,
             "pricing": {"prompt": "0.09", "completion": "0.09"}},
            {"id": "vendor/model", "context_length": 131072,
             "pricing": {"prompt": _REMOTE_PROMPT,
                         "completion": _REMOTE_COMPLETION}},
        ]},
    })
    assert report is not None
    assert report.context_per_slot == 131072
    assert report.input_usd_per_mtok == pytest.approx(0.5)
    assert report.output_usd_per_mtok == pytest.approx(1.5)
    assert report.publishes_prices is True


def test_a_published_price_of_zero_is_a_real_price_and_is_kept():
    """🚨 A free model on a remote provider is still a remote model. Reporting
    "no price" for it would push it onto the imputed avoided-cost fallback,
    which would then credit the deployment with money SAVED for a call it made
    over the internet."""
    report = OPENROUTER.parse_capacity({
        "model_id": "vendor/free",
        "models": {"data": [{"id": "vendor/free", "context_length": 4096,
                             "pricing": {"prompt": "0", "completion": "0"}}]},
    })
    assert report.input_usd_per_mtok == 0.0
    assert report.output_usd_per_mtok == 0.0
    assert report.publishes_prices is True, (
        "a zero price read as unpublished — the endpoint would fall back to the "
        "imputed table and book a saving for a remote call")


@pytest.mark.parametrize("pricing", [
    None, {}, {"prompt": "not-a-number"}, {"prompt": None}, "0.5",
])
def test_an_unparseable_price_is_unpublished_rather_than_free(pricing):
    report = OPENROUTER.parse_capacity({
        "model_id": "vendor/model",
        "models": {"data": [{"id": "vendor/model", "context_length": 4096,
                             "pricing": pricing}]},
    })
    assert report is not None, "a bad price cost us the context ceiling too"
    assert report.input_usd_per_mtok is None


def test_each_half_of_a_price_is_parsed_independently():
    """A provider that publishes a prompt price and no completion price has told
    us half of something true, and half of something true is worth having."""
    report = OPENROUTER.parse_capacity({
        "model_id": "vendor/model",
        "models": {"data": [{"id": "vendor/model", "context_length": 4096,
                             "pricing": {"prompt": _REMOTE_PROMPT}}]},
    })
    assert report.input_usd_per_mtok == pytest.approx(0.5)
    assert report.output_usd_per_mtok is None


# ===========================================================================
# 4. The threshold: it degrades, it never rejects
# ===========================================================================

def test_an_uncapped_caller_is_never_over():
    st = SpendStanding(agent_id="a", spent_today_usd=1e9, cap_usd=None)
    assert st.over is False
    assert st.may_spill is True
    assert st.effective_priority(LLMPriority.P1_TURN_SUPPORT) is LLMPriority.P1_TURN_SUPPORT


def test_a_zero_cap_is_a_real_cap_and_not_an_absent_one():
    """🚨 `0.0` and `None` are one keystroke apart in YAML and mean opposite
    things: "no paid spend at all" versus "uncapped". A truthiness test here
    would silently turn the first into the second."""
    assert SpendStanding("a", spent_today_usd=0.0, cap_usd=0.0).over is True
    assert SpendStanding("a", spent_today_usd=0.0, cap_usd=None).over is False


@pytest.mark.parametrize("declared,expected", [
    (LLMPriority.P0_REALTIME, LLMPriority.P1_TURN_SUPPORT),
    (LLMPriority.P1_TURN_SUPPORT, LLMPriority.P2_POST_TURN),
    (LLMPriority.P3_INGESTION, LLMPriority.P4_HYGIENE),
    # 🚨 The floor. However far over the cap a caller is, the penalty stops
    # here: a proportional penalty would be unbounded, and an unbounded penalty
    # on a billing signal is a rejection wearing a different hat.
    (LLMPriority.P4_HYGIENE, LLMPriority.P4_HYGIENE),
])
def test_over_cap_demotes_exactly_one_band_and_floors(declared, expected):
    st = SpendStanding("a", spent_today_usd=10.0, cap_usd=1.0)
    assert st.effective_priority(declared) is expected


def test_over_cap_loses_spill_and_nothing_else():
    st = SpendStanding("a", spent_today_usd=10.0, cap_usd=1.0)
    assert st.may_spill is False
    # There is deliberately no `may_dispatch`, no `refused` and no error code on
    # this object. The absence is the doctrine: nothing here can turn a request
    # into a failure, so no amount of misreading it can either.
    assert not hasattr(st, "may_dispatch")


def test_the_day_window_rolls_over_on_READ_not_only_on_write():
    """A caller that crossed its cap yesterday and has not called since must not
    still be degraded today. Without a read-side roll-over a cap becomes
    permanent for exactly the callers that stopped calling — the ones it has
    already worked on."""
    ledger = SpendLedger()
    ledger.prices.declare("spill-chat", _REAL)
    yesterday = 0.0
    tomorrow = 86400.0 * 2
    ledger.charge("a", "spill-chat", 10_000_000, 0, now=yesterday)
    assert ledger.spent_today("a", now=yesterday) > 0
    assert ledger.spent_today("a", now=tomorrow) == 0.0
    assert standing(ledger, "a", 1.0, now=yesterday).over is True
    assert standing(ledger, "a", 1.0, now=tomorrow).over is False
    # The lifetime total is untouched by the window — it is a different question.
    assert ledger.get("a").spent_usd > 0


def test_the_day_bucket_is_utc_and_stable():
    assert day_bucket(0.0) == 0
    assert day_bucket(86399.0) == 0
    assert day_bucket(86400.0) == 1


# ===========================================================================
# 5. Admission: one decision, three outcomes
# ===========================================================================

def _sched(
    *,
    spill_to: str = "spill-chat",
    spill_ok: bool = True,
    may_spend: Optional[bool] = True,
    src_slots: int = 1,
    tgt_slots: int = 4,
):
    """A two-endpoint fleet: `tier1` local with `src_slots`, spilling to a
    `spill-chat` stand-in with `tgt_slots`. The shipped catalog's `spill-chat`
    is `planned` and therefore not routed, so the target is built here."""
    config = ProxyConfig()
    config.endpoints["tier1"].max_slots = src_slots
    config.endpoints["tier1"].spill_to = spill_to
    if "spill-chat" not in config.endpoints:
        import copy
        tgt = copy.deepcopy(config.endpoints["tier1"])
        tgt.endpoint_class = "spill-chat"
        tgt.role = "spill-chat"
        tgt.spill_to = ""
        config.endpoints["spill-chat"] = tgt
    config.endpoints["spill-chat"].max_slots = tgt_slots
    config.endpoints["spill-chat"].context_per_slot = 131072
    config.agents["spiller"] = AgentQuotaConfig(agent_id="spiller", spill_ok=spill_ok)
    cm = CostModel()
    for ep, epc in config.endpoints.items():
        cm.register_endpoint(ep, epc.max_slots)
    bm = BudgetManager()
    bm.set_total_capacity(config.total_fleet_slots)
    sched = Scheduler(config, cm, bm)
    if may_spend is not None:
        sched.may_spend = lambda _agent, allowed=may_spend: allowed
    return sched, config


def _req(agent="spiller", endpoint="tier1", max_tokens=256, now=None):
    return QueuedRequest.create(
        agent_id=agent, endpoint=endpoint, priority="P1_TURN_SUPPORT",
        call_site="t", payload_type="chat_completion",
        payload={"messages": [{"role": "user", "content": "hi"}],
                 "max_tokens": max_tokens},
        timeout_s=60.0, now=now if now is not None else time.monotonic(),
    )


def test_a_free_local_slot_dispatches_locally_for_everybody():
    """🚨 The first line of the doctrine. Nothing about money appears above the
    local-capacity test in `_admit`, so a caller over its cap, a caller with no
    `spill_ok` and a caller nobody configured all reach the same DISPATCH."""
    sched, config = _sched(spill_ok=False, may_spend=False)
    verdict = sched._admit("tier1", config.endpoints["tier1"], _req(),
                           current_occupancy=0, band_available=1,
                           spill_target="spill-chat")
    assert verdict is Admission.DISPATCH


def test_a_full_local_endpoint_spills_an_opted_in_caller():
    sched, config = _sched()
    verdict = sched._admit("tier1", config.endpoints["tier1"], _req(),
                           current_occupancy=1, band_available=0,
                           spill_target="spill-chat")
    assert verdict is Admission.SPILL


@pytest.mark.parametrize("kwargs,why", [
    (dict(spill_ok=False), "a caller that did not opt in was spilled"),
    (dict(may_spend=False), "a caller over its spend cap was spilled"),
    (dict(tgt_slots=0), "spilled into a target with no capacity of its own"),
])
def test_the_third_outcome_is_defer_and_defer_is_not_a_refusal(kwargs, why):
    """Every "no" here leaves the request QUEUED on its local endpoint, where it
    will be served when a slot frees. There is no fourth outcome and no error:
    `Admission` has three members and none of them is a refusal."""
    sched, config = _sched(**kwargs)
    verdict = sched._admit("tier1", config.endpoints["tier1"], _req(),
                           current_occupancy=1, band_available=0,
                           spill_target="spill-chat")
    assert verdict is Admission.DEFER, why
    assert set(Admission) == {Admission.DISPATCH, Admission.SPILL, Admission.DEFER}


def test_a_request_that_cannot_FIT_the_target_defers():
    """Physics, not policy, and the same predicate failover uses. Never truncate
    to make it fit — a silently shortened prompt answers a different question
    than the one that was asked, and would pass every check downstream."""
    sched, config = _sched()
    config.endpoints["spill-chat"].context_per_slot = 512
    verdict = sched._admit("tier1", config.endpoints["tier1"],
                           _req(max_tokens=100_000),
                           current_occupancy=1, band_available=0,
                           spill_target="spill-chat")
    assert verdict is Admission.DEFER


def test_an_unhealthy_spill_target_is_not_a_target():
    """Deferring is strictly better than queueing into a second dead backend:
    the local slot the request is waiting for will actually open."""
    sched, config = _sched()
    sched.is_endpoint_healthy = lambda ep: ep != "spill-chat"
    assert sched._spill_target(config.endpoints["tier1"]) == ""


def test_a_spill_target_that_is_not_a_routed_endpoint_is_inert():
    """The commonest way to get this wrong is pointing at one of the `planned`
    remote endpoints before the credential is in the environment. That has to be
    a no-op, not a 502 the first time the local tier fills up."""
    sched, config = _sched(spill_to="not-an-endpoint")
    assert sched._spill_target(config.endpoints["tier1"]) == ""
    verdict = sched._admit("tier1", config.endpoints["tier1"], _req(),
                           current_occupancy=1, band_available=0,
                           spill_target="")
    assert verdict is Admission.DEFER


def test_no_spend_authority_wired_means_unconstrained_not_forbidden():
    """🚨 The asymmetry. Authorisation to spend is `spill_ok` — an operator
    saying yes about a caller. The ledger is a THRESHOLD, and a threshold that
    is not wired up removes nothing. Making its absence a refusal would turn
    "the accounting is not plumbed in" into "this configured feature silently
    does nothing"."""
    sched, config = _sched(may_spend=None)
    assert sched.may_spend is None
    verdict = sched._admit("tier1", config.endpoints["tier1"], _req(),
                           current_occupancy=1, band_available=0,
                           spill_target="spill-chat")
    assert verdict is Admission.SPILL


# ===========================================================================
# 6. Spilling a request, end to end through the scheduler
# ===========================================================================

def test_spilling_repoints_the_request_and_records_where_it_came_from():
    sched, config = _sched(src_slots=1)
    now = time.monotonic()
    held, spilling = _req(now=now), _req(now=now)
    sched.enqueue(held)
    sched.enqueue(spilling)
    deadline_before = spilling.timeout_deadline

    seen: list = []
    sched.on_spill = lambda req, src, tgt: seen.append((req.request_id, src, tgt))
    sched.tick(now)

    assert spilling.endpoint == "spill-chat"
    assert spilling.spilled_from == "tier1"
    # 🚨 NOT `degraded_from`. A request can be both, and the two answer different
    # questions for whoever reads the completion row: "was this served by a
    # worse model" and "did this cost money".
    assert spilling.degraded_from is None
    assert seen == [(spilling.request_id, "tier1", "spill-chat")]
    assert sched.stats()["total_spilled"] == 1
    # The deadline was computed from the SOURCE's floors and is NOT re-derived —
    # same rule as failover.apply, and for the stronger reason that a caller may
    # have supplied it explicitly.
    assert spilling.timeout_deadline == deadline_before
    # It landed on the target's queue, and is dispatched from there.
    assert sched.queue_depth("spill-chat") + sched.active_count("spill-chat") == 1


def test_a_spilled_request_is_re_costed_against_the_endpoint_it_will_occupy():
    """DRR is charged at DISPATCH, so a spilled request has not been charged
    yet and must not be. Its slot-second estimate is re-made for the target,
    which matters because a remote endpoint's cost model is its own."""
    sched, config = _sched(src_slots=1)
    now = time.monotonic()
    sched.enqueue(_req(now=now))
    spilling = _req(now=now)
    sched.enqueue(spilling)
    before = spilling.estimated_cost_ss
    sched._spill(spilling, "spill-chat")
    assert spilling.estimated_cost_ss > 0
    assert before > 0  # both are real estimates; the point is it was re-made
    assert spilling.ctx_per_slot_at_admission == 131072


def test_an_endpoint_with_no_spill_target_runs_the_old_loop_exactly():
    """The behaviour-preservation claim, asserted rather than assumed: with no
    `spill_to`, a full endpoint dispatches nothing and defers everything, which
    is what the pre-Workstream-D `while available > 0` guard did."""
    sched, config = _sched(spill_to="", src_slots=1)
    now = time.monotonic()
    for _ in range(3):
        sched.enqueue(_req(now=now))
    sched.tick(now)
    assert sched.active_count("tier1") == 1
    assert sched.queue_depth("tier1") == 2
    assert sched.stats()["total_spilled"] == 0


# ===========================================================================
# 7. The three allowlist parsers — add the key AND its guard
# ===========================================================================

def test_declared_price_reaches_endpoint_config(tmp_path, monkeypatch):
    """`model_catalog._POLICY_PASSTHROUGH` drops an unknown key in silence, and
    that failure has now bitten more than once. Reporting a typo is not enough:
    a knob that is settable in code and unreachable from config looks exactly
    like a policy decision that nobody opted in."""
    src = (tmp_path / "models.yaml")
    src.write_text((
        "providers:\n"
        "  p:\n"
        "    engine: llama.cpp\n"
        "    host: 192.0.2.1\n"
        "    port: 9000\n"
        "endpoints:\n"
        "  priced:\n"
        "    provider: p\n"
        "    kind: chat\n"
        "    slots: 1\n"
        "    context_per_slot: 4096\n"
        "    policy:\n"
        "      input_usd_per_mtok: 1.25\n"
        "      output_usd_per_mtok: 3.5\n"
    ))
    monkeypatch.setenv("ROADSTEAD_MODELS_YAML", str(src))
    kwargs = model_catalog.build_endpoint_kwargs(
        model_catalog.load_catalog(src, force=True))
    assert kwargs["priced"]["input_usd_per_mtok"] == 1.25
    assert kwargs["priced"]["output_usd_per_mtok"] == 3.5


def test_spill_to_only_resolves_to_a_routed_endpoint(tmp_path):
    """Same shape as `failover_to`: a spill naming a `planned` entry resolves to
    nothing rather than arming an overflow path to a backend that cannot
    serve."""
    src = (tmp_path / "models.yaml")
    src.write_text((
        "providers:\n"
        "  p:\n"
        "    engine: llama.cpp\n"
        "    host: 192.0.2.1\n"
        "    port: 9000\n"
        "endpoints:\n"
        "  live:\n"
        "    provider: p\n"
        "    kind: chat\n"
        "    slots: 1\n"
        "    spill_to: someday\n"
        "  other:\n"
        "    provider: p\n"
        "    kind: chat\n"
        "    slots: 1\n"
        "    spill_to: live\n"
        "  someday:\n"
        "    provider: p\n"
        "    kind: chat\n"
        "    status: planned\n"
        "    slots: 1\n"
    ))
    kwargs = model_catalog.build_endpoint_kwargs(
        model_catalog.load_catalog(src, force=True))
    assert "someday" not in kwargs, "a planned endpoint became routable"
    assert kwargs["live"].get("spill_to", "") == "", (
        "spill_to resolved to a planned endpoint — the first time `live` fills "
        "up it would dispatch to a backend that cannot serve")
    assert kwargs["other"]["spill_to"] == "live"


def test_the_spill_knobs_reach_agent_config(tmp_path, monkeypatch):
    """`config.load_agent_configs` is the third allowlist parser, and it drops
    an unknown key silently too. Both halves matter: the value must arrive, and
    an absent one must keep its default."""
    f = tmp_path / "agents.yaml"
    f.write_text(
        "opted-in:\n"
        "  spill_ok: true\n"
        "  daily_spend_usd: 2.5\n"
        "uncapped:\n"
        "  spill_ok: true\n"
        "  daily_spend_usd: null\n"
        "zero-cap:\n"
        "  spill_ok: true\n"
        "  daily_spend_usd: 0\n"
        "plain:\n"
        "  weight: 1.0\n"
    )
    cfgs = load_agent_configs(f)
    assert cfgs["opted-in"].spill_ok is True
    assert cfgs["opted-in"].daily_spend_usd == 2.5
    # 🚨 An explicit null is "uncapped" and stays None; `0` is a real cap
    # meaning "no paid spend at all". One keystroke, opposite policies.
    assert cfgs["uncapped"].daily_spend_usd is None
    assert cfgs["zero-cap"].daily_spend_usd == 0.0
    assert cfgs["plain"].spill_ok is False, "spill defaulted to opted-in"
    assert cfgs["plain"].daily_spend_usd is None


def test_the_shipped_agents_example_opts_exactly_one_caller_into_spill():
    """A path that never fires is the guard-that-reads-green failure class, so
    the shipped example seeds one — and only one, because the other default that
    matters here is default-DENY."""
    cfgs = load_agent_configs()
    opted = sorted(a for a, c in cfgs.items() if c.spill_ok)
    assert opted == ["chat-assistant"]
    assert cfgs["chat-assistant"].daily_spend_usd is not None, (
        "the one caller allowed to spend money is unbounded")
    # The canonical NEVER: the heaviest caller by volume is the one an uncapped
    # spill would bankrupt fastest, and its work is the least urgent.
    assert cfgs["extractor"].spill_ok is False


def test_spill_ok_and_degrade_ok_are_separate_judgements():
    """🚨 Neither implies the other, so neither may default from the other.
    `degrade_ok` asks "may a WORSE model answer this"; `spill_ok` asks "may this
    leave the machine, at our expense". A caller whose work degrades happily may
    still be one whose prompts must never go to a third party."""
    both = AgentQuotaConfig(agent_id="x", degrade_ok=True)
    assert both.spill_ok is False, "degrade_ok leaked into spill_ok"
    other = AgentQuotaConfig(agent_id="y", spill_ok=True)
    assert other.degrade_ok is False, "spill_ok leaked into degrade_ok"


def test_spill_never_chains():
    """🚨 The same rule failover follows, for a sharper reason: two endpoints
    whose configs point at each other would hand one request back and forth a
    hop per tick, forever, overwriting `spilled_from` each time so nothing
    downstream could see it happening."""
    # Room on the target and every other gate open, so the chain guard is the
    # ONLY reason this can defer.
    sched, config = _sched(tgt_slots=4)
    config.endpoints["spill-chat"].spill_to = "tier1"
    already = _req()
    already.spilled_from = "somewhere"
    verdict = sched._admit("tier1", config.endpoints["tier1"], already,
                           current_occupancy=1, band_available=0,
                           spill_target="spill-chat")
    assert verdict is Admission.DEFER


# ===========================================================================
# 8. End to end: a request, a completion, an invoice
# ===========================================================================

def _service(monkeypatch, agents_yaml: str, tmp_path):
    from roadstead.backend import BackendResponse
    from roadstead.service import ProxyService

    f = tmp_path / "agents.yaml"
    f.write_text(agents_yaml)
    monkeypatch.setenv("LLM_PROXY_AGENTS_CONFIG", str(f))
    monkeypatch.delenv("ROADSTEAD_ACL", raising=False)
    svc = ProxyService(ProxyConfig(agents=load_agent_configs(f)))

    async def ok_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"content": "y"}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}},
            duration_s=0.01, input_tokens=1_000_000, output_tokens=0,
            finish_reason="stop")

    svc._backend.call = ok_call
    return svc


class _LoopbackReq:
    class _Client:
        host = "127.0.0.1"

    client = _Client()
    headers: dict = {}
    method = "POST"
    query_params: dict = {}


def _submit_body(agent_id: str, priority: str | None = None):
    body = {"agent_id": agent_id, "endpoint": "tier1", "call_site": "t",
            "payload_type": "chat_completion",
            "payload": {"messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 8}}
    if priority:
        body["priority"] = priority
    return body


@pytest.mark.asyncio
async def test_a_local_completion_is_metered_but_never_billed(monkeypatch, tmp_path):
    """The end-to-end shape of doctrine 1. A local call moves the token counters
    and the AVOIDED total, and leaves the caller's daily spend at zero — so a
    deployment that never touches a remote provider can never demote anybody."""
    import asyncio

    svc = _service(monkeypatch, "metered:\n  spill_ok: true\n  daily_spend_usd: 0.01\n",
                   tmp_path)
    await svc.startup()
    try:
        r = await asyncio.wait_for(
            svc.handle_submit(_submit_body("metered"), _LoopbackReq()), timeout=10.0)
        assert r.status_code == 200
        acct = svc._state.spend.get("metered")
        assert acct.tokens_in == 1_000_000
        assert acct.avoided_usd > 0.0
        assert acct.spent_usd == 0.0
        # …and therefore it is NOT over a cap of one cent, however much local
        # work it does.
        assert svc._state.spend_standing("metered").over is False
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_an_over_cap_caller_is_demoted_and_still_served(monkeypatch, tmp_path):
    """🚨 The end-to-end shape of doctrine 2, and the single most important
    assertion in this file: crossing a spend cap costs a band and NOT the
    answer. The request completes, from local capacity, with a 200."""
    import asyncio

    from roadstead import hooks

    svc = _service(monkeypatch, "spender:\n  spill_ok: true\n  daily_spend_usd: 1.0\n",
                   tmp_path)
    # Put the caller over its cap with REAL spend, the only kind that counts.
    svc._state.prices.declare("spill-chat", _REAL)
    svc._state.spend.charge("spender", "spill-chat", 10_000_000, 0)
    assert svc._state.spend_standing("spender").over is True

    reports: list = []
    hooks.set_degradation_sink(
        lambda **kw: reports.append(kw))
    # Capture what actually reached the scheduler: a demotion computed and not
    # applied is the shape this whole file is written against.
    from roadstead import scheduler as _sched
    created: list = []
    orig = _sched.QueuedRequest.create.__func__
    _sched.QueuedRequest.create = classmethod(
        lambda cls, **kw: created.append(orig(cls, **kw)) or created[-1])
    await svc.startup()
    try:
        r = await asyncio.wait_for(
            svc.handle_submit(_submit_body("spender", "P1_TURN_SUPPORT"),
                              _LoopbackReq()),
            timeout=10.0)
        assert r.status_code == 200, "an over-cap caller was refused"
        # 🚨 One band down ON THE QUEUED REQUEST, and served anyway. It declared
        # P1_TURN_SUPPORT (INTERACTIVE) and is scheduled at P2_POST_TURN
        # (FOREGROUND) — behind everyone inside their allowance, ahead of
        # nothing else being taken away.
        assert created[-1].priority is LLMPriority.P2_POST_TURN
        assert created[-1].band is PriorityBand.FOREGROUND
        assert svc._state.spend_may_spill("spender") is False
        # Reported once, through the integration seam — a caller that suddenly
        # waits longer with nothing logged is indistinguishable from a slow
        # backend.
        spend_reports = [r for r in reports if r.get("component") == "spend"]
        assert len(spend_reports) == 1
        assert spend_reports[0]["priority_effective"] == "P2_POST_TURN"
        # …and only once, however many requests it makes.
        svc._state.spend_demote("spender", LLMPriority.P1_TURN_SUPPORT)
        assert len([r for r in reports if r.get("component") == "spend"]) == 1
    finally:
        _sched.QueuedRequest.create = classmethod(orig)
        hooks.set_degradation_sink(None)
        await svc.shutdown()


@pytest.mark.asyncio
async def test_a_spilled_call_is_billed_at_the_endpoint_that_served_it(
    monkeypatch, tmp_path,
):
    """🚨 Metering is keyed on the class that ACTUALLY served.

    A spilled request has already been re-pointed at the remote endpoint, so it
    is priced at the remote rate. Pricing it at the endpoint it was submitted
    for would be the one arrangement guaranteed to under-report exactly the
    calls that cost real money — every local call would look priced and every
    remote one would book a saving.
    """
    import asyncio
    import copy

    from roadstead.backend import BackendResponse
    from roadstead.service import ProxyService

    f = tmp_path / "agents.yaml"
    f.write_text("spiller:\n  spill_ok: true\n")
    config = ProxyConfig(agents=load_agent_configs(f))
    # One local slot, and a routed remote target. The shipped catalog's
    # `spill-chat` is `planned` and therefore unrouted, so build one.
    config.endpoints["tier1"].max_slots = 1
    config.endpoints["tier1"].dispatch_concurrency_cap = 1
    config.endpoints["tier1"].spill_to = "spill-chat"
    tgt = copy.deepcopy(config.endpoints["tier1"])
    tgt.endpoint_class, tgt.role, tgt.spill_to = "spill-chat", "spill-chat", ""
    tgt.max_slots, tgt.dispatch_concurrency_cap = 4, 0
    tgt.context_per_slot = 131072
    config.endpoints["spill-chat"] = tgt

    svc = ProxyService(config)
    svc._state.prices.declare("spill-chat", _REAL)
    release = asyncio.Event()

    async def call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        if ep_cfg.endpoint_class == "tier1":
            await release.wait()          # hold the single local slot
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"content": "y"},
                               "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}},
            duration_s=0.01, input_tokens=1_000_000, output_tokens=0,
            finish_reason="stop")

    svc._backend.call = call
    await svc.startup()
    try:
        held = asyncio.create_task(
            svc.handle_submit(_submit_body("spiller"), _LoopbackReq()))
        # Let the first request occupy the only local slot before the second
        # arrives, so the second meets a genuinely FULL endpoint.
        for _ in range(200):
            await asyncio.sleep(0.005)
            if svc._scheduler.active_count("tier1") == 1:
                break
        assert svc._scheduler.active_count("tier1") == 1, "the slot never filled"

        spilled = await asyncio.wait_for(
            svc.handle_submit(_submit_body("spiller"), _LoopbackReq()), timeout=10.0)
        assert spilled.status_code == 200
        acct = svc._state.spend.get("spiller")
        assert "spill-chat" in acct.by_endpoint, (
            "the spilled call was not billed against the endpoint that served it")
        assert acct.spent_usd == pytest.approx(0.5)
        assert svc._state.spilled_from.get("tier1") == 1

        release.set()
        await asyncio.wait_for(held, timeout=10.0)
        # The LOCAL half of the same pair is still free — it went to the avoided
        # column and left the bill alone.
        assert svc._state.spend.get("spiller").spent_usd == pytest.approx(0.5)
        assert svc._state.spend.get("spiller").avoided_usd > 0.0
    finally:
        release.set()
        await svc.shutdown()


# ===========================================================================
# 9. docs/api.md §1.6 is executable too
# ===========================================================================

@pytest.fixture(scope="module")
def api_doc() -> str:
    from tests.wire_contract import API_DOC

    assert API_DOC.exists(), f"the wire contract is missing at {API_DOC}"
    return API_DOC.read_text(encoding="utf-8")


def test_the_three_outcomes_published_are_the_three_that_exist(api_doc):
    """§1.6 publishes a three-row table. A fourth outcome added to `Admission`
    without a row — or a row without a member — means the document and the code
    disagree about the central decision this gateway makes."""
    assert "### 1.6 Spend, spill, and what a threshold does" in api_doc, (
        "docs/api.md §1.6 moved or was renamed")
    for member in Admission:
        assert f"| **{member.value}** |" in api_doc, (
            f"admission outcome {member.value!r} is not published in §1.6")
    published = {
        line.split("**")[1]
        for line in api_doc.splitlines()
        if line.startswith("| **") and "**" in line[4:]
    }
    assert {m.value for m in Admission} <= published


def test_no_error_code_exists_for_a_spend_threshold(api_doc):
    """🚨 The absence is the contract, so it is asserted rather than assumed.

    §2.1 enumerates every machine-readable `code` a caller can receive. A spend
    threshold degrades and never rejects, so it must never add one — and the day
    somebody adds `spend_exceeded` or `over_budget` to that list, this fails and
    makes them say so out loud.
    """
    codes_section = api_doc.split("### 2.1 Codes", 1)[1].split("### 2.2", 1)[0]
    for forbidden in ("spend_exceeded", "over_budget", "budget_exceeded",
                      "quota_exceeded", "spill_denied", "payment_required"):
        assert forbidden not in codes_section, (
            f"a spend-related error code {forbidden!r} appeared in the error "
            "contract — a threshold that can refuse a request is no longer a "
            "threshold that degrades")
    assert "no error code exists for it" in api_doc


def test_an_endpoint_cannot_spill_to_itself():
    """It would re-enqueue the request onto the very queue being drained,
    forever. The catalog parser already refuses to resolve this, but the parser
    is not the only way an `EndpointConfig` gets built."""
    sched, config = _sched(spill_to="tier1")
    assert sched._spill_target(config.endpoints["tier1"]) == ""


def test_an_endpoint_with_no_slots_defers_rather_than_spilling():
    """An endpoint with no slots at all is MISCONFIGURED, not busy, and the
    difference matters: "busy" is the question spill answers. Spilling a config
    error would quietly convert it into an invoice."""
    sched, config = _sched(src_slots=0)
    verdict = sched._admit("tier1", config.endpoints["tier1"], _req(),
                           current_occupancy=0, band_available=1,
                           spill_target="spill-chat")
    assert verdict is Admission.DEFER


def test_a_snapshot_rolls_the_day_window_over_too():
    """The status page reads this. A caller that spent yesterday and has not
    called since would otherwise show yesterday's number as though it were
    today's — the same read-side roll-over `spent_today` does, in the one place
    an operator actually looks."""
    ledger = SpendLedger()
    ledger.prices.declare("spill-chat", _REAL)
    ledger.charge("a", "spill-chat", 1_000_000, 0, now=0.0)
    assert ledger.snapshot(now=0.0)[0]["day_spent_usd"] == pytest.approx(0.5)
    rolled = ledger.snapshot(now=86400.0 * 2)[0]
    assert rolled["day_spent_usd"] == 0.0
    # …and the lifetime total is untouched, because it answers another question.
    assert rolled["spent_usd"] == pytest.approx(0.5)
