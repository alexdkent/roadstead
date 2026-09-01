"""Today's spend survives a restart, for the same reason DRR balances do.

`SpendLedger` is in-memory and its day bucket reset to zero on every boot, so a
deploy at noon handed every caller its whole daily allowance a second time.

🚨 **`daily_spend_usd` is a DAY; the process was only ever measuring an UPTIME.**
The more a fleet ships, the less its spend cap means — worst precisely on the
deployment where somebody set the cap deliberately. DRR balances have survived a
restart since Phase 3.4 (`load_budgets`); this is the same argument about the
other per-caller quantity, and it had been missing since `spend.py` landed.

What is deliberately NOT changed: the threshold still only DEGRADES, and the
seed fails open. See `test_the_seed_can_never_take_the_proxy_down`.

Every guard here was observed going red by mutating the code it guards.
"""
from __future__ import annotations

import time

import pytest

from roadstead import spend as spend_mod
from roadstead.config import AgentQuotaConfig, ProxyConfig
from roadstead.service import ProxyService
from roadstead.spend import (
    SOURCE_PROVIDER, PriceBook, TokenPrice, day_start,
)


def _svc(tmp_path, **cfg) -> ProxyService:
    return ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db"), **cfg))


def _record(svc, agent, endpoint, tin, tout, *, request_id, when=None):
    """Write one completion row exactly as the completion path does."""
    svc._queue_db.persist_complete(
        request_id=request_id, agent_id=agent, endpoint=endpoint,
        call_site=f"{agent}.test", priority=1,
        input_tokens=tin, output_tokens=tout,
        duration_s=0.1, queue_wait_ms=0.0, status="ok",
    )
    if when is not None:
        svc._queue_db._w("UPDATE proxy_completions SET completed_at=? "
                         "WHERE request_id=?", (when, request_id))
    svc._queue_db.flush()


@pytest.fixture
def paid(monkeypatch):
    """Price every endpoint as a REAL cost, so spend is an invoice.

    A local endpoint prices as `avoided`, and avoided cost is never read by a
    threshold — a test that forgot this would assert on a number the product
    deliberately ignores.
    """
    def _price(self, endpoint: str) -> TokenPrice:
        # `real` is DERIVED from `source` — a provider-published price is an
        # invoice, an imputed one is a saving. Setting the source is the only
        # way to make this money, and that is the point of the design.
        return TokenPrice(input_usd_per_mtok=1000.0,
                          output_usd_per_mtok=1000.0, source=SOURCE_PROVIDER)
    monkeypatch.setattr(PriceBook, "price", _price)


async def _boot(svc):
    await svc.startup()
    try:
        yield
    finally:  # pragma: no cover
        pass


@pytest.mark.asyncio
async def test_todays_spend_is_restored_on_startup(tmp_path, paid):
    """The bug, stated as the thing an operator would see.

    Mutation: delete the seed block from `startup`. `spent_today` comes back
    0.0, the caller's cap is fresh, and nothing anywhere reports it.
    """
    svc = _svc(tmp_path)
    await svc.startup()
    _record(svc, "greedy", "tier1", 1_000_000, 1_000_000, request_id="r1")
    await svc.shutdown()

    # A different process object over the same database: a restart.
    again = _svc(tmp_path)
    assert again._state.spend.spent_today("greedy") == 0.0, "fixture stale"
    await again.startup()
    try:
        # 2 Mtok at $1000/Mtok = $2000.
        assert again._state.spend.spent_today("greedy") == pytest.approx(2000.0)
    finally:
        await again.shutdown()


@pytest.mark.asyncio
async def test_only_TODAYS_rows_are_restored(tmp_path, paid):
    """🚨 A cap is per-day. Replaying the whole table would make a caller's cap
    permanent — it could never come back under, because yesterday's spend never
    ages out.

    Mutation: pass 0 instead of `day_start(...)`. Yesterday's row is summed in,
    and the caller stays degraded forever.
    """
    svc = _svc(tmp_path)
    await svc.startup()
    yesterday = day_start(time.time()) - 3600.0
    _record(svc, "greedy", "tier1", 5_000_000, 0, request_id="old", when=yesterday)
    _record(svc, "greedy", "tier1", 1_000_000, 0, request_id="new")
    await svc.shutdown()

    again = _svc(tmp_path)
    await again.startup()
    try:
        assert again._state.spend.spent_today("greedy") == pytest.approx(1000.0)
    finally:
        await again.shutdown()


@pytest.mark.asyncio
async def test_the_request_COUNT_survives_too_not_just_the_money(tmp_path, paid):
    """The rollup is one row per caller-endpoint pair, so a naive replay would
    record N requests as 1 and the dashboard would under-report every restart.

    Mutation: drop `requests=row["requests"]`. The money is right and the
    request count is 1, which is the kind of wrong nobody notices.
    """
    svc = _svc(tmp_path)
    await svc.startup()
    for i in range(5):
        _record(svc, "chatty", "tier1", 1000, 1000, request_id=f"r{i}")
    await svc.shutdown()

    again = _svc(tmp_path)
    await again.startup()
    try:
        acct = again._state.spend.get("chatty")
        assert acct is not None and acct.requests == 5
        assert acct.tokens_in == 5000 and acct.tokens_out == 5000
    finally:
        await again.shutdown()


@pytest.mark.asyncio
async def test_a_restored_caller_is_actually_DEGRADED(tmp_path, paid):
    """The whole point: restoring the number has to move the decision, or it is
    a dashboard change wearing a correctness argument."""
    svc = _svc(tmp_path)
    svc._config.agents["greedy"] = AgentQuotaConfig(
        agent_id="greedy", daily_spend_usd=100.0)
    await svc.startup()
    _record(svc, "greedy", "tier1", 1_000_000, 0, request_id="r1")   # $1000
    await svc.shutdown()

    again = _svc(tmp_path)
    again._config.agents["greedy"] = AgentQuotaConfig(
        agent_id="greedy", daily_spend_usd=100.0)
    await again.startup()
    try:
        st = again._state.spend_standing("greedy")
        assert st.over is True
        # 🚨 And it still costs exactly one band and paid spill, never capacity.
        assert again._state.spend_may_spill("greedy") is False
    finally:
        await again.shutdown()


@pytest.mark.asyncio
async def test_the_seed_can_never_take_the_proxy_down(tmp_path, monkeypatch):
    """🚨 Fails OPEN, deliberately.

    The alternative is refusing to boot because we cannot prove a caller is over
    a threshold whose entire consequence is one priority band. Admission control
    is about capacity; a spend cap must not be able to take the proxy down any
    more than it can take a caller offline.

    Mutation: remove the `try`. A broken rollup query stops the process from
    starting at all — the worst possible failure for the least important number.
    """
    svc = _svc(tmp_path)

    def _boom(_since):
        raise RuntimeError("database is on fire")

    monkeypatch.setattr(svc._queue_db, "day_spend_rollup", _boom)
    await svc.startup()          # must not raise
    try:
        assert svc._state.spend.spent_today("anyone") == 0.0
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_two_endpoints_at_DIFFERENT_prices_are_priced_separately(
        tmp_path, monkeypatch):
    """🚨 Why the rollup groups by endpoint as well as by caller.

    This first SURVIVED the mutation `GROUP BY agent_id` — because the fixture
    above prices every endpoint the same, so collapsing the dimension changed
    nothing. It is not a cosmetic grouping: SQLite would hand back one arbitrary
    endpoint from the group and the caller's whole day would be re-priced at it.
    A caller that spent $1 on a cheap model and $1000 on an expensive one is the
    exact shape of a spend cap's day, and it would come back as $2 or $2000.

    Asserting a caller's total against two prices is what makes the grouping
    observable at all.
    """
    prices = {"tier1": 1.0, "tier2": 1000.0}

    def _price(self, endpoint: str) -> TokenPrice:
        per = prices.get(endpoint, 0.0)
        return TokenPrice(input_usd_per_mtok=per, output_usd_per_mtok=per,
                          source=SOURCE_PROVIDER)

    monkeypatch.setattr(PriceBook, "price", _price)

    svc = _svc(tmp_path)
    await svc.startup()
    _record(svc, "mixed", "tier1", 1_000_000, 0, request_id="cheap")   # $1
    _record(svc, "mixed", "tier2", 1_000_000, 0, request_id="dear")    # $1000
    await svc.shutdown()

    again = _svc(tmp_path)
    await again.startup()
    try:
        assert again._state.spend.spent_today("mixed") == pytest.approx(1001.0)
        # And the per-endpoint breakdown the readout shows survives too.
        acct = again._state.spend.get("mixed")
        assert acct.by_endpoint["tier1"] == pytest.approx(1.0)
        assert acct.by_endpoint["tier2"] == pytest.approx(1000.0)
    finally:
        await again.shutdown()


def test_the_rollup_is_aggregated_in_sql_not_in_python(tmp_path):
    """One row per caller-endpoint pair, not one per request. A day of traffic
    is millions of rows and startup reads this on the loop."""
    svc = _svc(tmp_path)
    for i in range(20):
        _record(svc, "a", "tier1", 10, 10, request_id=f"a{i}")
    for i in range(7):
        _record(svc, "b", "tier2", 10, 10, request_id=f"b{i}")

    rows = svc._queue_db.day_spend_rollup(0.0)
    assert len(rows) == 2, rows
    by_agent = {r["agent_id"]: r for r in rows}
    assert by_agent["a"]["requests"] == 20
    assert by_agent["b"]["requests"] == 7
    assert by_agent["a"]["input_tokens"] == 200


def test_day_start_and_day_bucket_cannot_disagree():
    """They are two answers to "where does a day begin", and they would drift
    apart at exactly one instant a day — the hardest possible time to notice.

    Mutation: define `day_start` with its own `//` arithmetic and a stray
    offset. This fails at the boundary rather than at a random hour.
    """
    now = time.time()
    for probe in (now, day_start(now), day_start(now) + 86399.999):
        assert spend_mod.day_bucket(probe) == spend_mod.day_bucket(now)
    assert spend_mod.day_bucket(day_start(now) - 0.001) == \
        spend_mod.day_bucket(now) - 1
