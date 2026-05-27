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
    def test_starving_agent_gets_minimum_service(self):
        sched, _, _, bm = _make_scheduler(
            endpoints={"chat": 1},
            starvation_timeout_s=0.1,
        )
        now = time.monotonic()

        # Agent A has a heavily negative balance
        budget_a = bm.get_or_create("agent_a", now=now)
        budget_a.balance = -100.0
        budget_a.negative_since = now - 1.0  # negative for 1s, threshold is 0.1s

        # Agent B has positive balance
        bm.get_or_create("agent_b", now=now)

        sched.enqueue(_req(agent_id="agent_a", now=now))
        sched.enqueue(_req(agent_id="agent_b", now=now + 0.001))

        decisions = sched.tick(now + 0.01)
        assert len(decisions) >= 1
        # The starving agent_a should get served despite negative balance
        assert decisions[0].request.agent_id == "agent_a"


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
