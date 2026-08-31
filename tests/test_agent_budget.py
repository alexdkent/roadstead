"""Tests for the DRR agent budget system."""

import time
import pytest

from roadstead.agent_budget import AgentBudget, BudgetManager


class TestAgentBudget:
    def test_charge_reduces_balance(self):
        b = AgentBudget(agent_id="a", balance=10.0)
        now = time.monotonic()
        b.charge(3.0, now)
        assert b.balance == 7.0
        assert b.total_consumed == 3.0
        assert b.total_requests == 1

    def test_charge_tracks_negative_since(self):
        b = AgentBudget(agent_id="a", balance=2.0)
        now = time.monotonic()
        b.charge(5.0, now)
        assert b.balance == -3.0
        assert b.negative_since == now

    def test_refund_increases_balance(self):
        b = AgentBudget(agent_id="a", balance=-5.0, negative_since=1.0)
        b.refund(10.0)
        assert b.balance == 5.0
        assert b.negative_since == 0  # cleared because balance >= 0

    def test_starvation_detection(self):
        now = time.monotonic()
        b = AgentBudget(agent_id="a", balance=-5.0, negative_since=now - 60)
        assert b.is_starving(now, starvation_timeout_s=30.0)
        assert not b.is_starving(now, starvation_timeout_s=120.0)


class TestBudgetManager:
    def test_replenishment(self):
        bm = BudgetManager()
        bm.set_total_capacity(10.0)
        now = time.monotonic()

        bm.get_or_create("a", weight=1.0, now=now)
        bm.get_or_create("b", weight=1.0, now=now)

        # After 1 second, each should gain 5.0 slot-seconds (10 total / 2 agents)
        bm.replenish(now + 1.0)
        assert abs(bm.agents["a"].balance - 5.0) < 0.1
        assert abs(bm.agents["b"].balance - 5.0) < 0.1

    def test_balance_cap(self):
        bm = BudgetManager()
        bm.set_total_capacity(100.0)
        now = time.monotonic()

        bm.get_or_create("a", weight=1.0, max_balance=10.0, now=now)

        # Replenish way past the cap
        bm.replenish(now + 100.0)
        assert bm.agents["a"].balance == 10.0

    def test_unequal_weights(self):
        bm = BudgetManager()
        bm.set_total_capacity(10.0)
        now = time.monotonic()

        bm.get_or_create("a", weight=2.0, now=now)
        bm.get_or_create("b", weight=1.0, now=now)

        bm.replenish(now + 1.0)
        # a gets 2/3 of 10 = 6.67, b gets 1/3 of 10 = 3.33
        assert abs(bm.agents["a"].balance - 6.67) < 0.1
        assert abs(bm.agents["b"].balance - 3.33) < 0.1

    def test_pick_agent_highest_balance(self):
        bm = BudgetManager()
        bm.set_total_capacity(10.0)
        now = time.monotonic()

        bm.get_or_create("a", now=now)
        bm.get_or_create("b", now=now)
        bm.agents["a"].balance = 5.0
        bm.agents["b"].balance = 10.0

        picked = bm.pick_agent(["a", "b"], now)
        assert picked == "b"

    def test_pick_agent_denied_service_gets_priority(self):
        # Starvation is keyed off real head-of-queue wait, not the sign of the
        # balance. "a" has waited past the timeout (denied service) → it wins
        # even with a deeply negative balance.
        bm = BudgetManager(starvation_timeout_s=1.0)
        bm.set_total_capacity(10.0)
        now = time.monotonic()

        bm.get_or_create("a", now=now)
        bm.get_or_create("b", now=now)
        bm.agents["a"].balance = -100.0
        bm.agents["b"].balance = 100.0

        picked = bm.pick_agent(["a", "b"], now, {"a": 10.0, "b": 0.1})
        assert picked == "a"

    def test_pick_agent_negative_balance_alone_is_not_starving(self):
        # Regression for the DRR starvation bug: a deeply negative balance that
        # has been negative "forever" (negative_since long ago) must NOT win
        # the override if the agent is actually being served (fresh head wait).
        bm = BudgetManager(starvation_timeout_s=1.0)
        bm.set_total_capacity(10.0)
        now = time.monotonic()

        bm.get_or_create("heavy", now=now)
        bm.get_or_create("light", now=now)
        bm.agents["heavy"].balance = -100.0
        bm.agents["heavy"].negative_since = now - 600.0  # "starving" under old rule
        bm.agents["light"].balance = 50.0

        # heavy is served continuously (fresh head); light has waited.
        picked = bm.pick_agent(["heavy", "light"], now, {"heavy": 0.1, "light": 5.0})
        assert picked == "light"

    def test_retroactive_adjust_refund(self):
        bm = BudgetManager()
        bm.set_total_capacity(10.0)
        now = time.monotonic()

        bm.get_or_create("a", now=now)
        bm.agents["a"].balance = 0.0
        bm.agents["a"].total_consumed = 10.0

        bm.retroactive_adjust("a", estimated=5.0, actual=3.0)
        assert bm.agents["a"].balance == 2.0  # refunded 2
        assert bm.agents["a"].total_consumed == 8.0

    def test_retroactive_adjust_surcharge(self):
        bm = BudgetManager()
        bm.set_total_capacity(10.0)
        now = time.monotonic()

        bm.get_or_create("a", now=now)
        bm.agents["a"].balance = 5.0
        bm.agents["a"].total_consumed = 10.0

        bm.retroactive_adjust("a", estimated=3.0, actual=5.0)
        assert bm.agents["a"].balance == 3.0  # charged 2 more
        assert bm.agents["a"].total_consumed == 12.0

    def test_snapshot_serializable(self):
        bm = BudgetManager()
        bm.set_total_capacity(10.0)
        now = time.monotonic()
        bm.get_or_create("a", now=now)
        snap = bm.snapshot()
        assert len(snap) == 1
        assert snap[0]["agent_id"] == "a"
        assert "balance_ss" in snap[0]


class TestPruneIdle:
    def test_prunes_only_idle_agents(self):
        bm = BudgetManager()
        bm.set_total_capacity(32.0)
        now = 1000.0
        # 'active' charged recently; 'stale' created long ago, never returned.
        bm.get_or_create("active", now=now)
        bm.get_or_create("stale", now=now - 7200.0)
        bm.charge("active", 1.0, now)               # active_at = now
        pruned = bm.prune_idle(now, idle_ttl_s=3600.0)
        assert pruned == ["stale"]
        assert "active" in bm.agents
        assert "stale" not in bm.agents

    def test_returning_agent_recreates_clean(self):
        bm = BudgetManager()
        bm.set_total_capacity(32.0)
        now = 1000.0
        bm.get_or_create("oneoff", now=now - 7200.0)
        bm.prune_idle(now, idle_ttl_s=3600.0)
        # If it ever returns, get_or_create rebuilds it at full (idle) balance.
        b = bm.get_or_create("oneoff", max_balance=60.0, now=now)
        assert b.agent_id == "oneoff"
        assert "oneoff" in bm.agents
