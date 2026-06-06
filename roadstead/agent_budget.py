"""Deficit-round-robin agent budget accounting.

Pure computation — no I/O, no framework imports.  All state is in the
``AgentBudget`` and ``BudgetManager`` classes; persistence is the
caller's responsibility.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class AgentBudget:
    """Token balance for one agent in the DRR scheme.

    Units are **slot-seconds**: a request that holds one slot for 2
    seconds costs 2.0 tokens.  The balance increases over time at
    ``replenish_rate`` (slot-seconds per wall-second) and decreases
    when the scheduler charges a dispatched request's cost.
    """
    agent_id: str
    weight: float = 1.0
    replenish_rate: float = 0.0       # set by BudgetManager.recalculate_rates()
    max_balance: float = 60.0         # cap to prevent unbounded accumulation
    balance: float = 0.0              # current DRR deficit counter
    total_consumed: float = 0.0       # lifetime consumed slot-seconds
    total_requests: int = 0
    last_replenish_at: float = 0.0    # monotonic timestamp
    last_active_at: float = 0.0       # monotonic ts of last charge (for idle prune)
    negative_since: float = 0.0       # monotonic ts when balance first went negative (0 = not negative)

    def charge(self, cost_ss: float, now: float) -> None:
        """Debit ``cost_ss`` slot-seconds from this agent's balance."""
        self.balance -= cost_ss
        self.total_consumed += cost_ss
        self.total_requests += 1
        self.last_active_at = now
        if self.balance < 0 and self.negative_since == 0:
            self.negative_since = now

    def refund(self, delta_ss: float) -> None:
        """Refund a cost over-estimate (actual < estimated)."""
        self.balance += delta_ss
        self.total_consumed -= delta_ss
        if self.balance >= 0:
            self.negative_since = 0

    def is_starving(self, now: float, starvation_timeout_s: float) -> bool:
        """True if balance has been continuously negative for longer than
        the starvation timeout.

        NOTE: this is a *balance-health* signal used only for observability /
        simulation metrics. It is NOT the dispatch starvation guard — that
        moved to ``BudgetManager.pick_agent`` keyed on real head-of-queue wait
        (denial of service), because a heavy consumer's balance is perpetually
        negative even while it is being served continuously."""
        if self.negative_since == 0:
            return False
        return (now - self.negative_since) >= starvation_timeout_s


class BudgetManager:
    """Manages DRR budgets for all known agents.

    Call ``replenish(now)`` on every scheduler tick to credit balances.
    Call ``recalculate_rates(total_capacity)`` whenever the fleet
    topology changes (e.g. an endpoint discovers more or fewer slots).
    """

    def __init__(self, starvation_timeout_s: float = 30.0) -> None:
        self._agents: dict[str, AgentBudget] = {}
        self._starvation_timeout_s = starvation_timeout_s
        self._total_capacity: float = 0.0

    @property
    def agents(self) -> dict[str, AgentBudget]:
        return self._agents

    def get_or_create(
        self,
        agent_id: str,
        *,
        weight: float = 1.0,
        max_balance: float = 60.0,
        now: float | None = None,
    ) -> AgentBudget:
        if agent_id not in self._agents:
            ts = now if now is not None else time.monotonic()
            self._agents[agent_id] = AgentBudget(
                agent_id=agent_id,
                weight=weight,
                max_balance=max_balance,
                last_replenish_at=ts,
                last_active_at=ts,
            )
            self._recalculate_rates()
        return self._agents[agent_id]

    def remove(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)
        self._recalculate_rates()

    def prune_idle(self, now: float, idle_ttl_s: float) -> list[str]:
        """Evict agents with no charged request in ``idle_ttl_s`` so the map
        can't grow unbounded with one-off ``agent_id``s (ad-hoc tools / smoke
        scripts / ``probe-*``). Safe: an idle agent's balance has fully
        replenished to ``max_balance``, so if it ever returns it is recreated
        at the identical state — no fairness loss. Recalculates rates once."""
        stale = [
            aid for aid, b in self._agents.items()
            if b.last_active_at and (now - b.last_active_at) > idle_ttl_s
        ]
        for aid in stale:
            self._agents.pop(aid, None)
        if stale:
            self._recalculate_rates()
        return stale

    def set_total_capacity(self, total_slots: float) -> None:
        """Update the total fleet capacity (sum of all endpoint max_slots)
        and recalculate per-agent replenishment rates."""
        self._total_capacity = total_slots
        self._recalculate_rates()

    def replenish(self, now: float) -> None:
        """Credit all agents' balances proportional to elapsed time."""
        for budget in self._agents.values():
            elapsed = now - budget.last_replenish_at
            if elapsed <= 0:
                budget.last_replenish_at = now
                continue
            credit = budget.replenish_rate * elapsed
            budget.balance = min(budget.max_balance, budget.balance + credit)
            budget.last_replenish_at = now
            if budget.balance >= 0:
                budget.negative_since = 0

    def charge(self, agent_id: str, cost_ss: float, now: float) -> None:
        budget = self._agents.get(agent_id)
        if budget:
            budget.charge(cost_ss, now)

    def retroactive_adjust(self, agent_id: str, estimated: float, actual: float) -> None:
        """Adjust balance for the difference between estimated and actual cost."""
        budget = self._agents.get(agent_id)
        if not budget:
            return
        delta = estimated - actual
        if delta > 0:
            budget.refund(delta)
        elif delta < 0:
            budget.balance += delta  # charge more (delta is negative)
            budget.total_consumed -= delta

    def pick_agent(
        self,
        candidates: list[str],
        now: float,
        wait_by_agent: dict[str, float] | None = None,
    ) -> str | None:
        """Select the next agent to serve from ``candidates`` using DRR.

        Starvation is measured by **denial of service** — how long an agent's
        oldest queued request has actually waited (``wait_by_agent``) — NOT by
        the sign of its DRR balance. A heavy consumer that is being served
        continuously has a deeply negative balance yet a *short* head-of-queue
        wait, so it is correctly NOT starving; a light consumer locked behind
        it accrues a long head-of-queue wait and gets rescued. (Keying
        starvation off ``balance < 0`` was the bug: a perpetually-negative
        heavy producer flagged itself "starving" forever and short-circuited
        the picker, starving everyone else on the shared endpoint.)

        When one or more candidates have waited past the starvation timeout,
        the pool is restricted to those and the longest-waiting one is served.
        Otherwise the agent with the highest weight-normalized balance wins.
        Returns ``None`` if ``candidates`` is empty.
        """
        if not candidates:
            return None

        # Denied-service escape hatch: serve the agent whose oldest queued
        # request has waited longest past the timeout, regardless of balance.
        if wait_by_agent:
            starving = [
                (wait_by_agent.get(a, 0.0), a)
                for a in candidates
                if wait_by_agent.get(a, 0.0) >= self._starvation_timeout_s
            ]
            if starving:
                return max(starving, key=lambda t: t[0])[1]

        best_id: str | None = None
        best_score: float = float("-inf")

        for agent_id in candidates:
            budget = self._agents.get(agent_id)
            if budget is None:
                continue
            # Score: balance normalized by weight (higher is more deserving)
            score = budget.balance / budget.weight if budget.weight > 0 else budget.balance
            if score > best_score:
                best_score = score
                best_id = agent_id

        return best_id

    def _recalculate_rates(self) -> None:
        if not self._agents or self._total_capacity <= 0:
            return
        total_weight = sum(b.weight for b in self._agents.values())
        if total_weight <= 0:
            return
        for budget in self._agents.values():
            budget.replenish_rate = self._total_capacity * (budget.weight / total_weight)

    def snapshot(self) -> list[dict]:
        """Return serialisable budget state for observability."""
        return [
            {
                "agent_id": b.agent_id,
                "weight": b.weight,
                "balance_ss": round(b.balance, 2),
                "replenish_rate_ss": round(b.replenish_rate, 3),
                "total_consumed_ss": round(b.total_consumed, 1),
                "total_requests": b.total_requests,
                "starving": b.negative_since > 0,
            }
            for b in self._agents.values()
        ]
