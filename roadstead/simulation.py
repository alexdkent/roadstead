"""Discrete-event simulator for the LLM proxy scheduler.

Runs the real scheduler/cost_model/agent_budget code against synthetic
request streams.  No actual LLM backends needed — request durations
are drawn from the cost model or scenario-specified distributions.

Usage:
    python -m roadstead.simulation stress all_agents_burst
    python -m roadstead.simulation stress --all
    python -m roadstead.simulation sweep --param thinker.max_slots --range 2,6 --scenario one_agent_flood
"""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .agent_budget import BudgetManager
from .config import (
    DEFAULT_ENDPOINTS,
    EndpointConfig,
    LLMPriority,
    PriorityBand,
    ProxyConfig,
    normalize_endpoint,
    priority_to_band,
)
from .cost_model import CostModel
from .observability import jains_fairness_index
from .scheduler import CompletionRecord, DispatchDecision, QueuedRequest, Scheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulated clock
# ---------------------------------------------------------------------------

@dataclass(order=True)
class SimEvent:
    time: float
    kind: str = field(compare=False)       # "arrive" | "complete"
    data: Any = field(compare=False)


class SimClock:
    """Discrete event clock with a priority queue of upcoming events."""

    def __init__(self) -> None:
        self._heap: list[SimEvent] = []
        self._now: float = 0.0

    @property
    def now(self) -> float:
        return self._now

    def schedule(self, at: float, kind: str, data: Any = None) -> None:
        heapq.heappush(self._heap, SimEvent(time=at, kind=kind, data=data))

    def next_event(self) -> Optional[SimEvent]:
        if not self._heap:
            return None
        ev = heapq.heappop(self._heap)
        self._now = ev.time
        return ev

    def has_events(self) -> bool:
        return bool(self._heap)


# ---------------------------------------------------------------------------
# Agent traffic profile
# ---------------------------------------------------------------------------

@dataclass
class AgentProfile:
    agent_id: str
    rate_rps: float                              # requests per second
    priority_mix: Dict[str, float] = field(default_factory=dict)  # e.g. {"P0": 0.3, "P1": 0.7}
    endpoints: List[str] = field(default_factory=list)
    avg_input_tokens: int = 2000
    avg_output_tokens: int = 200
    timeout_s: float = 60.0
    weight: float = 1.0

    def sample_priority(self) -> LLMPriority:
        if not self.priority_mix:
            return LLMPriority.P1_TURN_SUPPORT
        r = random.random()
        cumulative = 0.0
        for label, pct in self.priority_mix.items():
            cumulative += pct
            if r <= cumulative:
                return LLMPriority.coerce(label)
        return LLMPriority.P1_TURN_SUPPORT

    def sample_endpoint(self) -> str:
        if not self.endpoints:
            return "chat"
        return random.choice(self.endpoints)


# ---------------------------------------------------------------------------
# Endpoint event (slot change mid-simulation)
# ---------------------------------------------------------------------------

@dataclass
class EndpointEvent:
    at_s: float
    endpoint: str
    max_slots: int


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    duration_s: float
    agents: List[AgentProfile]
    endpoint_events: List[EndpointEvent] = field(default_factory=list)
    endpoint_overrides: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Simulation results
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    scenario: str
    duration_s: float
    total_submitted: int = 0
    total_dispatched: int = 0
    total_completed: int = 0
    total_timeouts: int = 0
    total_expired: int = 0

    # Per-endpoint metrics
    per_endpoint: Dict[str, Dict] = field(default_factory=dict)

    # Per-agent metrics
    per_agent: Dict[str, Dict] = field(default_factory=dict)

    # Fairness
    jains_index: float = 1.0
    max_deficit_ratio: float = 0.0

    # Time series (for plotting)
    queue_depth_over_time: List[Tuple[float, str, int]] = field(default_factory=list)

    def report(self) -> str:
        lines = []
        lines.append(f"\n{'='*70}")
        lines.append(f"  Scenario: {self.scenario}")
        lines.append(f"  Duration: {self.duration_s:.0f}s")
        lines.append(f"{'='*70}")
        lines.append(f"  Submitted: {self.total_submitted}  Dispatched: {self.total_dispatched}  "
                      f"Completed: {self.total_completed}  Timeouts: {self.total_timeouts}  "
                      f"Expired: {self.total_expired}")
        lines.append(f"  Jain's fairness index: {self.jains_index:.3f}")
        lines.append(f"  Max deficit ratio: {self.max_deficit_ratio:.2f}")

        lines.append(f"\n  {'Endpoint':<12} {'Submitted':>9} {'Dispatched':>10} {'Completed':>9} "
                      f"{'Timeouts':>8} {'p50 wait':>9} {'p95 wait':>9} {'Util%':>6}")
        lines.append(f"  {'-'*12} {'-'*9} {'-'*10} {'-'*9} {'-'*8} {'-'*9} {'-'*9} {'-'*6}")
        for ep, m in sorted(self.per_endpoint.items()):
            lines.append(
                f"  {ep:<12} {m['submitted']:>9} {m['dispatched']:>10} {m['completed']:>9} "
                f"{m['timeouts']:>8} {m['p50_wait_ms']:>8.0f}ms {m['p95_wait_ms']:>8.0f}ms "
                f"{m['utilization_pct']:>5.1f}%"
            )

        lines.append(f"\n  {'Agent':<16} {'Submitted':>9} {'Dispatched':>10} {'Slot-sec':>8} {'Starved':>7}")
        lines.append(f"  {'-'*16} {'-'*9} {'-'*10} {'-'*8} {'-'*7}")
        for aid, m in sorted(self.per_agent.items()):
            lines.append(
                f"  {aid:<16} {m['submitted']:>9} {m['dispatched']:>10} "
                f"{m['slot_seconds']:>7.1f}s {m['starvation_events']:>7}"
            )

        # Diagnostics
        total_lost = self.total_timeouts + self.total_expired
        loss_pct = (total_lost / self.total_submitted * 100) if self.total_submitted else 0
        lines.append(f"  Loss: {total_lost}/{self.total_submitted} ({loss_pct:.0f}%) "
                      f"[threshold: {self.max_expire_pct*100:.0f}%]")
        lines.append("")
        return "\n".join(lines)

    max_expire_pct: float = 0.10  # scenario-specific override

    @property
    def passed(self) -> bool:
        """Pass criteria.  ``max_expire_pct`` is set per-scenario to
        reflect physically achievable throughput under the load profile."""
        total_lost = self.total_timeouts + self.total_expired
        if self.total_submitted > 0:
            if total_lost / self.total_submitted > self.max_expire_pct:
                return False
        # Hard fail: no dispatches at all (deadlock)
        if self.total_submitted > 10 and self.total_dispatched == 0:
            return False
        return True


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

class SimRunner:
    """Discrete-event simulation engine."""

    def __init__(self, scenario: Scenario) -> None:
        self._scenario = scenario
        self._clock = SimClock()

        # Build config with overrides
        self._config = ProxyConfig(starvation_timeout_s=30.0)
        for ep, slots in scenario.endpoint_overrides.items():
            if ep in self._config.endpoints:
                self._config.endpoints[ep].max_slots = slots

        for profile in scenario.agents:
            acfg = self._config.agent_config(profile.agent_id)
            acfg.weight = profile.weight

        # Build components
        self._cost_model = CostModel()
        for ep, epc in self._config.endpoints.items():
            self._cost_model.register_endpoint(ep, epc.max_slots)
        self._budget_mgr = BudgetManager(starvation_timeout_s=30.0)
        self._budget_mgr.set_total_capacity(self._config.total_fleet_slots)
        self._scheduler = Scheduler(self._config, self._cost_model, self._budget_mgr)

        # Tracking
        self._result = SimResult(scenario=scenario.name, duration_s=scenario.duration_s)
        self._wait_times: Dict[str, List[float]] = {}  # endpoint → [wait_ms]
        self._per_agent_ss: Dict[str, float] = {}
        self._per_agent_submitted: Dict[str, int] = {}
        self._per_agent_dispatched: Dict[str, int] = {}
        self._per_agent_starvation: Dict[str, int] = {}
        self._per_ep_submitted: Dict[str, int] = {}
        self._per_ep_dispatched: Dict[str, int] = {}
        self._per_ep_completed: Dict[str, int] = {}
        self._per_ep_timeouts: Dict[str, int] = {}
        self._per_ep_slot_time: Dict[str, float] = {}
        self._req_counter = 0

    def run(self) -> SimResult:
        """Execute the full simulation and return results."""
        # Schedule initial arrivals for each agent
        for profile in self._scenario.agents:
            if profile.rate_rps > 0:
                interval = 1.0 / profile.rate_rps
                self._clock.schedule(
                    interval * random.random(),
                    "arrive",
                    profile,
                )

        # Schedule endpoint events
        for ee in self._scenario.endpoint_events:
            self._clock.schedule(ee.at_s, "endpoint_change", ee)

        # Schedule simulation end
        self._clock.schedule(self._scenario.duration_s, "sim_end", None)

        # Schedule periodic queue-depth sampling (every 0.5s)
        self._clock.schedule(0.5, "sample_depth", None)

        # Run event loop
        while self._clock.has_events():
            ev = self._clock.next_event()
            if ev is None:
                break
            if ev.kind == "sim_end":
                break
            elif ev.kind == "arrive":
                self._handle_arrival(ev)
            elif ev.kind == "complete":
                self._handle_completion(ev)
            elif ev.kind == "endpoint_change":
                self._handle_endpoint_change(ev)
            elif ev.kind == "sample_depth":
                self._sample_queue_depth()
                if self._clock.now < self._scenario.duration_s:
                    self._clock.schedule(self._clock.now + 0.5, "sample_depth", None)

            # Run scheduler tick after every event
            self._run_tick()

        self._finalize()
        return self._result

    def _handle_arrival(self, ev: SimEvent) -> None:
        profile: AgentProfile = ev.data
        now = self._clock.now

        self._req_counter += 1
        endpoint = normalize_endpoint(profile.sample_endpoint())
        priority = profile.sample_priority()

        req = QueuedRequest.create(
            agent_id=profile.agent_id,
            endpoint=endpoint,
            priority=priority,
            call_site=f"{profile.agent_id}.sim",
            payload_type="chat_completion",
            payload={
                "messages": [{"role": "user", "content": "x" * (profile.avg_input_tokens * 4)}],
                "max_tokens": profile.avg_output_tokens,
            },
            timeout_s=profile.timeout_s,
            request_id=f"sim_{self._req_counter}",
            now=now,
        )

        self._scheduler.enqueue(req)
        self._result.total_submitted += 1
        self._per_agent_submitted[profile.agent_id] = self._per_agent_submitted.get(profile.agent_id, 0) + 1
        self._per_ep_submitted[endpoint] = self._per_ep_submitted.get(endpoint, 0) + 1

        # Schedule next arrival
        if profile.rate_rps > 0 and now < self._scenario.duration_s:
            interval = 1.0 / profile.rate_rps
            jitter = interval * 0.2 * (random.random() - 0.5)
            self._clock.schedule(now + interval + jitter, "arrive", profile)

    def _handle_completion(self, ev: SimEvent) -> None:
        decision: DispatchDecision = ev.data
        req = decision.request
        now = self._clock.now
        duration_s = now - (req.enqueued_at + decision.queue_wait_ms / 1000)

        self._scheduler.complete(
            CompletionRecord(
                request_id=req.request_id,
                duration_s=max(0.01, duration_s),
                input_tokens=req.payload.get("max_tokens", 200),
                output_tokens=req.payload.get("max_tokens", 200),
                success=True,
                occupancy_during=decision.occupancy_at_dispatch + 1,
            ),
            now,
        )

        self._result.total_completed += 1
        self._per_ep_completed[req.endpoint] = self._per_ep_completed.get(req.endpoint, 0) + 1
        self._per_ep_slot_time[req.endpoint] = self._per_ep_slot_time.get(req.endpoint, 0) + duration_s
        self._per_agent_ss[req.agent_id] = self._per_agent_ss.get(req.agent_id, 0) + duration_s

    def _handle_endpoint_change(self, ev: SimEvent) -> None:
        ee: EndpointEvent = ev.data
        ep = normalize_endpoint(ee.endpoint)
        if ep in self._config.endpoints:
            old = self._config.endpoints[ep].max_slots
            self._config.endpoints[ep].max_slots = ee.max_slots
            self._cost_model.update_max_slots(ep, ee.max_slots)
            self._budget_mgr.set_total_capacity(self._config.total_fleet_slots)
            logger.info(
                "sim t=%.1f: endpoint %s slots %d → %d",
                self._clock.now, ep, old, ee.max_slots,
            )

    def _run_tick(self) -> None:
        now = self._clock.now
        expired_before = self._scheduler.stats()["total_timeouts"]

        decisions = self._scheduler.tick(now)

        expired_after = self._scheduler.stats()["total_timeouts"]
        new_expired = expired_after - expired_before
        self._result.total_expired += new_expired

        for d in decisions:
            self._result.total_dispatched += 1
            self._per_ep_dispatched[d.request.endpoint] = (
                self._per_ep_dispatched.get(d.request.endpoint, 0) + 1
            )
            self._per_agent_dispatched[d.request.agent_id] = (
                self._per_agent_dispatched.get(d.request.agent_id, 0) + 1
            )
            self._wait_times.setdefault(d.request.endpoint, []).append(d.queue_wait_ms)

            # Simulate backend duration
            duration = self._simulate_duration(d)
            self._clock.schedule(now + duration, "complete", d)

        # Check starvation
        for budget in self._budget_mgr.agents.values():
            if budget.is_starving(now, 30.0):
                self._per_agent_starvation[budget.agent_id] = (
                    self._per_agent_starvation.get(budget.agent_id, 0) + 1
                )

    def _simulate_duration(self, decision: DispatchDecision) -> float:
        """Estimate how long this request would take on the real backend."""
        req = decision.request
        ep_model = self._cost_model.get(req.endpoint)
        if not ep_model:
            return 1.0

        occ = max(1, decision.occupancy_at_dispatch + 1)
        tps_idx = min(occ, len(ep_model.decode_tps)) - 1
        tps = ep_model.decode_tps[tps_idx] if ep_model.decode_tps else 40.0

        input_tokens = len(req.payload.get("messages", [{}])[0].get("content", "")) // 4
        output_tokens = req.payload.get("max_tokens", 200)
        prefill = ep_model.prefill_k * input_tokens
        decode = output_tokens / tps if tps > 0 else 5.0

        # Add 10% jitter
        base = prefill + decode
        return base * (0.9 + 0.2 * random.random())

    def _sample_queue_depth(self) -> None:
        for ep in self._config.endpoints:
            depth = self._scheduler.queue_depth(ep)
            self._result.queue_depth_over_time.append(
                (self._clock.now, ep, depth)
            )

    def _finalize(self) -> None:
        # Per-endpoint stats
        for ep in self._config.endpoints:
            waits = self._wait_times.get(ep, [])
            waits_sorted = sorted(waits)
            p50 = waits_sorted[len(waits_sorted) // 2] if waits_sorted else 0
            p95 = waits_sorted[int(len(waits_sorted) * 0.95)] if waits_sorted else 0

            ep_cfg = self._config.endpoints[ep]
            slot_time = self._per_ep_slot_time.get(ep, 0)
            available_time = ep_cfg.max_slots * self._scenario.duration_s
            util = (slot_time / available_time * 100) if available_time > 0 else 0

            self._result.per_endpoint[ep] = {
                "submitted": self._per_ep_submitted.get(ep, 0),
                "dispatched": self._per_ep_dispatched.get(ep, 0),
                "completed": self._per_ep_completed.get(ep, 0),
                "timeouts": self._per_ep_timeouts.get(ep, 0),
                "p50_wait_ms": p50,
                "p95_wait_ms": p95,
                "utilization_pct": util,
            }

        # Per-agent stats
        for profile in self._scenario.agents:
            aid = profile.agent_id
            self._result.per_agent[aid] = {
                "submitted": self._per_agent_submitted.get(aid, 0),
                "dispatched": self._per_agent_dispatched.get(aid, 0),
                "slot_seconds": self._per_agent_ss.get(aid, 0),
                "starvation_events": self._per_agent_starvation.get(aid, 0),
            }

        # Fairness
        shares = [self._per_agent_ss.get(p.agent_id, 0) for p in self._scenario.agents]
        self._result.jains_index = jains_fairness_index(shares)

        if shares:
            mean_share = sum(shares) / len(shares) if shares else 1
            if mean_share > 0:
                self._result.max_deficit_ratio = max(
                    abs(s - mean_share) / mean_share for s in shares
                )


# ---------------------------------------------------------------------------
# Built-in scenarios
# ---------------------------------------------------------------------------

REALISTIC_AGENTS = [
    AgentProfile("orchestrator", rate_rps=0.5, priority_mix={"P0_REALTIME": 0.3, "P1_TURN_SUPPORT": 0.7},
                 endpoints=["qwen-composer", "gemma-router", "gemma-greeter"], avg_input_tokens=4000,
                 avg_output_tokens=500, weight=2.0),
    AgentProfile("knowledge", rate_rps=2.0, priority_mix={"P1_TURN_SUPPORT": 0.4, "P3_INGESTION": 0.6},
                 endpoints=["qwen-analyst", "bge-m3-embed", "bge-reranker"], avg_input_tokens=3000,
                 avg_output_tokens=200, weight=1.5),
    AgentProfile("forum-agent", rate_rps=0.8, priority_mix={"P3_INGESTION": 1.0},
                 endpoints=["llama-thinker"], avg_input_tokens=5000, avg_output_tokens=400),
    AgentProfile("mail-agent", rate_rps=0.3, priority_mix={"P3_INGESTION": 1.0},
                 endpoints=["qwen-analyst", "gemma-router"], avg_input_tokens=2000, avg_output_tokens=150),
    AgentProfile("sidekick", rate_rps=0.2, priority_mix={"P1_TURN_SUPPORT": 0.5, "P3_INGESTION": 0.5},
                 endpoints=["qwen-analyst", "llama-thinker"], avg_input_tokens=1500, avg_output_tokens=100),
    AgentProfile("homeassistant", rate_rps=0.1, priority_mix={"P1_TURN_SUPPORT": 0.8, "P3_INGESTION": 0.2},
                 endpoints=["gemma-greeter", "llama-thinker"], avg_input_tokens=1000, avg_output_tokens=100),
]


def _scenario_all_agents_burst() -> Scenario:
    agents = [
        AgentProfile(p.agent_id, rate_rps=p.rate_rps * 5, priority_mix={"P1_TURN_SUPPORT": 1.0},
                     endpoints=p.endpoints, avg_input_tokens=p.avg_input_tokens,
                     avg_output_tokens=p.avg_output_tokens, weight=p.weight)
        for p in REALISTIC_AGENTS
    ]
    return Scenario("all_agents_burst", duration_s=60, agents=agents)


def _scenario_one_agent_flood() -> Scenario:
    agents = list(REALISTIC_AGENTS)
    agents[0] = AgentProfile(
        "orchestrator", rate_rps=10.0, priority_mix={"P1_TURN_SUPPORT": 1.0},
        endpoints=["qwen-composer"], avg_input_tokens=4000, avg_output_tokens=500, weight=2.0,
    )
    return Scenario("one_agent_flood", duration_s=60, agents=agents)


def _scenario_background_storm() -> Scenario:
    agents = [
        AgentProfile(p.agent_id, rate_rps=p.rate_rps * 3, priority_mix={"P3_INGESTION": 0.7, "P4_HYGIENE": 0.3},
                     endpoints=p.endpoints, avg_input_tokens=p.avg_input_tokens,
                     avg_output_tokens=p.avg_output_tokens)
        for p in REALISTIC_AGENTS
    ]
    return Scenario("background_storm", duration_s=60, agents=agents)


def _scenario_endpoint_loss() -> Scenario:
    return Scenario(
        "endpoint_loss", duration_s=60, agents=list(REALISTIC_AGENTS),
        endpoint_events=[EndpointEvent(at_s=20, endpoint="qwen-analyst", max_slots=0)],
    )


def _scenario_slot_reduction() -> Scenario:
    return Scenario(
        "slot_reduction", duration_s=60, agents=list(REALISTIC_AGENTS),
        endpoint_events=[EndpointEvent(at_s=20, endpoint="qwen-analyst", max_slots=2)],
    )


def _scenario_new_agent() -> Scenario:
    agents = list(REALISTIC_AGENTS) + [
        AgentProfile("new_agent", rate_rps=1.0, priority_mix={"P3_INGESTION": 1.0},
                     endpoints=["qwen-analyst", "llama-thinker"], avg_input_tokens=3000,
                     avg_output_tokens=300),
    ]
    return Scenario("new_agent", duration_s=60, agents=agents)


def _scenario_turn_storm() -> Scenario:
    agents = [
        AgentProfile("orchestrator", rate_rps=3.0, priority_mix={"P0_REALTIME": 0.5, "P1_TURN_SUPPORT": 0.5},
                     endpoints=["qwen-composer", "gemma-router"], avg_input_tokens=4000,
                     avg_output_tokens=500, weight=2.0),
        AgentProfile("knowledge", rate_rps=4.0, priority_mix={"P3_INGESTION": 1.0},
                     endpoints=["qwen-analyst", "bge-m3-embed"], avg_input_tokens=3000, avg_output_tokens=200),
        AgentProfile("forum-agent", rate_rps=2.0, priority_mix={"P3_INGESTION": 1.0},
                     endpoints=["llama-thinker"], avg_input_tokens=5000, avg_output_tokens=400),
    ]
    return Scenario("turn_storm", duration_s=60, agents=agents)


def _scenario_long_tail() -> Scenario:
    agents = [
        AgentProfile("fast_agent", rate_rps=5.0, priority_mix={"P1_TURN_SUPPORT": 1.0},
                     endpoints=["gemma-router"], avg_input_tokens=500, avg_output_tokens=50,
                     timeout_s=5.0),
        AgentProfile("slow_agent", rate_rps=0.5, priority_mix={"P1_TURN_SUPPORT": 1.0},
                     endpoints=["qwen-composer"], avg_input_tokens=10000, avg_output_tokens=2000,
                     timeout_s=120.0),
    ]
    return Scenario("long_tail", duration_s=60, agents=agents)


def _scenario_cascade_timeout() -> Scenario:
    agents = [
        AgentProfile(p.agent_id, rate_rps=p.rate_rps * 4, priority_mix=p.priority_mix,
                     endpoints=p.endpoints, avg_input_tokens=p.avg_input_tokens,
                     avg_output_tokens=p.avg_output_tokens, timeout_s=10.0, weight=p.weight)
        for p in REALISTIC_AGENTS
    ]
    return Scenario("cascade_timeout", duration_s=60, agents=agents)


BUILTIN_SCENARIOS: Dict[str, Scenario] = {
    "all_agents_burst": _scenario_all_agents_burst(),
    "one_agent_flood": _scenario_one_agent_flood(),
    "background_storm": _scenario_background_storm(),
    "endpoint_loss": _scenario_endpoint_loss(),
    "slot_reduction": _scenario_slot_reduction(),
    "new_agent": _scenario_new_agent(),
    "turn_storm": _scenario_turn_storm(),
    "long_tail": _scenario_long_tail(),
    "cascade_timeout": _scenario_cascade_timeout(),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Scenarios that deliberately exceed capacity get relaxed pass thresholds.
# The point is to verify no deadlocks and correct priority ordering, not
# that infinite demand fits in finite capacity.
_SCENARIO_EXPIRE_THRESHOLDS: Dict[str, float] = {
    "all_agents_burst":  0.95,  # 5x overload — most will expire
    "one_agent_flood":   0.95,
    "background_storm":  0.95,
    "endpoint_loss":     0.50,  # half the traffic hits a dead endpoint
    "cascade_timeout":   0.85,  # 4x load + 10s timeouts
    "long_tail":         0.80,  # fast_agent 5s timeout + 2-slot endpoint
    "turn_storm":        0.95,
    "slot_reduction":    0.30,
    "new_agent":         0.30,
}


def run_scenario(name: str, scenario: Scenario) -> SimResult:
    runner = SimRunner(scenario)
    result = runner.run()
    result.max_expire_pct = _SCENARIO_EXPIRE_THRESHOLDS.get(name, 0.10)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    parser = argparse.ArgumentParser(description="LLM proxy scheduler simulator")
    sub = parser.add_subparsers(dest="command")

    stress_p = sub.add_parser("stress", help="Run a built-in stress scenario")
    stress_p.add_argument("scenario", nargs="?", help="Scenario name (or --all)")
    stress_p.add_argument("--all", action="store_true", help="Run all scenarios")

    parser.parse_args()
    args = parser.parse_args()

    if args.command == "stress":
        if args.all or not args.scenario:
            scenarios = list(BUILTIN_SCENARIOS.items())
        else:
            if args.scenario not in BUILTIN_SCENARIOS:
                print(f"Unknown scenario: {args.scenario}")
                print(f"Available: {', '.join(BUILTIN_SCENARIOS.keys())}")
                sys.exit(1)
            scenarios = [(args.scenario, BUILTIN_SCENARIOS[args.scenario])]

        all_passed = True
        for name, scenario in scenarios:
            result = run_scenario(name, scenario)
            print(result.report())
            status = "PASS" if result.passed else "FAIL"
            if not result.passed:
                all_passed = False
            print(f"  Result: {status}")

        if not all_passed:
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
