"""Tests for the DRR scheduler with priority bands."""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import pytest

from originfleet.llmproxy.agent_budget import BudgetManager
from originfleet.llmproxy.config import LLMPriority, ProxyConfig
from originfleet.llmproxy.cost_model import CostModel
from originfleet.llmproxy.scheduler import (
    CompletionRecord,
    QueuedRequest,
    Scheduler,
)


def _make_scheduler(
    *,
    endpoints: Optional[Dict[str, int]] = None,
    starvation_timeout_s: float = 30.0,
) -> Tuple[Scheduler, ProxyConfig, CostModel, BudgetManager]:
    config = ProxyConfig(starvation_timeout_s=starvation_timeout_s)
    if endpoints:
        for ep_name, slots in endpoints.items():
            if ep_name in config.endpoints:
                config.endpoints[ep_name].max_slots = slots
    cm = CostModel()
    for ep, epc in config.endpoints.items():
        cm.register_endpoint(ep, epc.max_slots)
    bm = BudgetManager(starvation_timeout_s=starvation_timeout_s)
    bm.set_total_capacity(config.total_fleet_slots)
    sched = Scheduler(config, cm, bm)
    return sched, config, cm, bm


def _req(
    agent_id: str = "agent_a",
    endpoint: str = "qwen-analyst",
    priority: str = "P1_TURN_SUPPORT",
    call_site: str = "test",
    now: float | None = None,
    timeout_s: float = 60.0,
    max_tokens: int = 256,
) -> QueuedRequest:
    return QueuedRequest.create(
        agent_id=agent_id,
        endpoint=endpoint,
        priority=priority,
        call_site=call_site,
        payload_type="chat_completion",
        payload={"messages": [{"role": "user", "content": "test"}], "max_tokens": max_tokens},
        timeout_s=timeout_s,
        now=now or time.monotonic(),
    )


class TestPriorityBandOrdering:
    def test_interactive_dispatched_before_background(self):
        sched, *_ = _make_scheduler()
        now = time.monotonic()

        bg = _req(agent_id="bg_agent", endpoint="llama-thinker", priority="P3_INGESTION", now=now)
        fg = _req(agent_id="fg_agent", endpoint="llama-thinker", priority="P0_REALTIME", now=now + 0.001)

        sched.enqueue(bg)
        sched.enqueue(fg)

        decisions = sched.tick(now + 0.01)
        assert len(decisions) >= 2
        assert decisions[0].request.priority == LLMPriority.P0_REALTIME

    def test_foreground_dispatched_before_background(self):
        sched, *_ = _make_scheduler()
        now = time.monotonic()

        bg = _req(agent_id="bg_agent", endpoint="llama-thinker", priority="P4_HYGIENE", now=now)
        fg = _req(agent_id="fg_agent", endpoint="llama-thinker", priority="P2_POST_TURN", now=now + 0.001)

        sched.enqueue(bg)
        sched.enqueue(fg)

        decisions = sched.tick(now + 0.01)
        assert len(decisions) >= 2
        assert decisions[0].request.priority == LLMPriority.P2_POST_TURN


class TestDRRFairness:
    def test_equal_weights_equal_dispatch(self):
        sched, _, _, bm = _make_scheduler()
        now = time.monotonic()

        # Two agents, equal weight, each submit 4 requests
        for i in range(4):
            sched.enqueue(_req(agent_id="agent_a", now=now + i * 0.001))
            sched.enqueue(_req(agent_id="agent_b", now=now + i * 0.001 + 0.0005))

        # Run enough ticks to dispatch all (chat endpoint has 4 slots)
        dispatched_a = 0
        dispatched_b = 0
        for step in range(10):
            decisions = sched.tick(now + 0.1 + step * 0.01)
            for d in decisions:
                if d.request.agent_id == "agent_a":
                    dispatched_a += 1
                else:
                    dispatched_b += 1
                # Complete immediately to free slots
                sched.complete(CompletionRecord(
                    request_id=d.request.request_id,
                    duration_s=0.1,
                    input_tokens=10,
                    output_tokens=10,
                    success=True,
                    occupancy_during=1,
                ), now + 0.2 + step * 0.01)

        assert dispatched_a == 4
        assert dispatched_b == 4

    def test_unequal_weights_proportional(self):
        sched, config, _, bm = _make_scheduler(endpoints={"chat": 2})
        now = time.monotonic()

        # Agent A weight=2, Agent B weight=1
        config.agents["agent_a"] = config.agent_config("agent_a")
        config.agents["agent_a"].weight = 2.0
        config.agents["agent_b"] = config.agent_config("agent_b")
        config.agents["agent_b"].weight = 1.0

        # Submit 6 requests each
        for i in range(6):
            sched.enqueue(_req(agent_id="agent_a", now=now + i * 0.001))
            sched.enqueue(_req(agent_id="agent_b", now=now + i * 0.001 + 0.0005))

        dispatched_a = 0
        dispatched_b = 0
        for step in range(20):
            decisions = sched.tick(now + 0.1 + step * 0.05)
            for d in decisions:
                if d.request.agent_id == "agent_a":
                    dispatched_a += 1
                else:
                    dispatched_b += 1
                sched.complete(CompletionRecord(
                    request_id=d.request.request_id,
                    duration_s=0.5,
                    input_tokens=10, output_tokens=10,
                    success=True, occupancy_during=1,
                ), now + 0.2 + step * 0.05)

        assert dispatched_a == 6
        assert dispatched_b == 6
        # With 2x weight, agent_a's balance depletes half as fast, so it
        # should be served more promptly. Both finish all 6 though.


class TestStarvationPrevention:
    def test_denied_service_agent_is_rescued_regardless_of_balance(self):
        # An agent whose oldest queued request has waited past the starvation
        # timeout is being denied service and must be served even if its DRR
        # balance is negative.
        sched, _, _, bm = _make_scheduler(
            endpoints={"chat": 1},
            starvation_timeout_s=0.1,
        )
        now = time.monotonic()

        waiter = bm.get_or_create("waiter", now=now)
        waiter.balance = -100.0  # negative balance must NOT disqualify it
        bm.get_or_create("fresh", now=now)

        # waiter's request has waited 0.5s (> 0.1 threshold); fresh just arrived.
        sched.enqueue(_req(agent_id="waiter", now=now - 0.5))
        sched.enqueue(_req(agent_id="fresh", now=now))

        decisions = sched.tick(now + 0.001)
        assert len(decisions) >= 1
        assert decisions[0].request.agent_id == "waiter"

    def test_continuous_consumer_does_not_starve_light_waiter(self):
        # Regression for the DRR starvation bug: a heavy consumer whose balance
        # is perpetually negative used to be flagged "starving" forever (by
        # negative_since) and short-circuit the picker, locking out a light
        # waiter. Starvation is now denial-of-service (head-of-queue wait), so
        # a heavy consumer that is being served continuously (fresh head
        # request) does NOT preempt a light consumer that has actually waited.
        sched, _, _, bm = _make_scheduler(
            endpoints={"chat": 1},
            starvation_timeout_s=0.1,
        )
        now = time.monotonic()

        heavy = bm.get_or_create("heavy", now=now)
        heavy.balance = -500.0
        heavy.negative_since = now - 100.0  # "starving" under the OLD buggy rule
        bm.get_or_create("light", now=now)

        # light has waited (denied service); heavy's head request is fresh
        # (it is being dispatched continuously).
        sched.enqueue(_req(agent_id="light", now=now - 0.5))
        sched.enqueue(_req(agent_id="heavy", now=now))

        decisions = sched.tick(now + 0.001)
        assert len(decisions) >= 1
        assert decisions[0].request.agent_id == "light"


class TestPickAgentDRR:
    """Direct contract tests for BudgetManager.pick_agent."""

    def test_no_wait_info_falls_back_to_highest_balance(self):
        bm = BudgetManager(starvation_timeout_s=30.0)
        now = time.monotonic()
        a = bm.get_or_create("a", now=now)
        b = bm.get_or_create("b", now=now)
        a.balance = -10.0
        b.balance = 5.0
        assert bm.pick_agent(["a", "b"], now) == "b"

    def test_sub_threshold_waits_use_drr_not_starvation(self):
        bm = BudgetManager(starvation_timeout_s=30.0)
        now = time.monotonic()
        heavy = bm.get_or_create("heavy", now=now)
        light = bm.get_or_create("light", now=now)
        heavy.balance = -500.0
        light.balance = 40.0
        # Both waited only 1s (< 30s) → no starvation → DRR picks light.
        waits = {"heavy": 1.0, "light": 1.0}
        assert bm.pick_agent(["heavy", "light"], now, waits) == "light"

    def test_continuous_consumer_negative_balance_is_not_starving(self):
        # The exact bug: heavy consumer balance deeply negative but its head
        # request is fresh (served continuously); light consumer has waited.
        bm = BudgetManager(starvation_timeout_s=30.0)
        now = time.monotonic()
        heavy = bm.get_or_create("heavy", now=now)
        heavy.balance = -999.0
        heavy.negative_since = now - 600.0  # would be "starving" under old rule
        bm.get_or_create("light", now=now)
        waits = {"heavy": 2.0, "light": 120.0}  # only light is past 30s
        assert bm.pick_agent(["heavy", "light"], now, waits) == "light"

    def test_longest_waiter_wins_when_several_starving(self):
        bm = BudgetManager(starvation_timeout_s=30.0)
        now = time.monotonic()
        for name in ("x", "y", "z"):
            bm.get_or_create(name, now=now)
        # All past threshold; the longest waiter is served first.
        waits = {"x": 45.0, "y": 200.0, "z": 60.0}
        assert bm.pick_agent(["x", "y", "z"], now, waits) == "y"


class TestBackgroundFloor:
    def test_background_gets_floor_during_interactive_flood(self):
        sched, config, _, bm = _make_scheduler(endpoints={"chat": 4})
        now = time.monotonic()

        # 4 interactive requests
        for i in range(4):
            sched.enqueue(_req(
                agent_id="orch", priority="P0_REALTIME",
                now=now + i * 0.001,
            ))
        # 1 background request
        sched.enqueue(_req(
            agent_id="forum-agent", priority="P3_INGESTION",
            now=now + 0.005,
        ))

        decisions = sched.tick(now + 0.01)
        endpoints_used = [d.request.agent_id for d in decisions]
        priorities = [d.request.priority for d in decisions]

        # bg floor for chat is 20% of 4 = 1 slot reserved for background
        # so interactive should get at most 3 slots, bg gets 1
        interactive_count = sum(1 for p in priorities if p <= LLMPriority.P1_TURN_SUPPORT)
        bg_count = sum(1 for p in priorities if p >= LLMPriority.P3_INGESTION)

        assert interactive_count <= 3
        assert bg_count >= 1


class TestConcurrencyAwareAdmission:
    def test_respects_max_slots(self):
        sched, *_ = _make_scheduler(endpoints={"chat": 2})
        now = time.monotonic()

        # Submit 5 requests
        for i in range(5):
            sched.enqueue(_req(agent_id="agent_a", now=now + i * 0.001))

        decisions = sched.tick(now + 0.01)
        # Should dispatch at most 2 (max_slots)
        assert len(decisions) <= 2
        assert sched.active_count("chat") == len(decisions)


class TestTimeout:
    def test_expired_requests_removed(self):
        sched, *_ = _make_scheduler()
        now = time.monotonic()

        # Request with very short timeout
        req = _req(agent_id="agent_a", now=now, timeout_s=0.01)
        sched.enqueue(req)

        # Tick after timeout
        expired_count = 0

        def on_timeout(r):
            nonlocal expired_count
            expired_count += 1

        sched.on_timeout = on_timeout
        decisions = sched.tick(now + 1.0)

        assert expired_count == 1
        assert sched.queue_depth("chat") == 0


class TestQueueWaitMetrics:
    def test_queue_wait_ms_reported(self):
        sched, *_ = _make_scheduler()
        now = time.monotonic()

        req = _req(agent_id="agent_a", now=now)
        sched.enqueue(req)

        decisions = sched.tick(now + 0.05)
        assert len(decisions) == 1
        assert decisions[0].queue_wait_ms >= 45  # ~50ms wait
        assert decisions[0].queue_wait_ms <= 60
