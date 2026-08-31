"""Phase 3 — streamline: dead-code removal + DRR-budget persistence.

  3.1 EndpointConfig.default_timeout_s deleted (dead, no callers).
  3.3 _total_completed counts only a real completion; EmbedCoalescer deleted.
  3.4 DRR balances persist + restore across a restart.
"""

from __future__ import annotations

from roadstead import coalesce
from roadstead.agent_budget import BudgetManager
from roadstead.config import EndpointConfig, ProxyConfig
from roadstead.cost_model import CostModel
from roadstead.queue import PersistentQueue
from roadstead.scheduler import CompletionRecord, Scheduler


def test_default_timeout_s_removed():
    ep = EndpointConfig(endpoint_class="x", role="y")
    assert not hasattr(ep, "default_timeout_s")


def test_embed_coalescer_removed():
    assert not hasattr(coalesce, "EmbedCoalescer")


def test_complete_unknown_request_does_not_count():
    sched = Scheduler(ProxyConfig(), CostModel(), BudgetManager())
    sched.complete(
        CompletionRecord(request_id="never-dispatched", duration_s=1.0,
                         input_tokens=0, output_tokens=0, success=True,
                         occupancy_during=0),
        now=0.0,
    )
    assert sched.stats()["total_completed"] == 0


def test_budget_save_load_roundtrip(tmp_path):
    db = str(tmp_path / "q.db")
    pq = PersistentQueue(db)  # no writer → synchronous
    pq.save_budgets([
        {"agent_id": "heavy", "weight": 2.0, "balance_ss": 15.0, "total_consumed_ss": 100.0},
        {"agent_id": "light", "weight": 1.0, "balance_ss": -3.0, "total_consumed_ss": 5.0},
    ])
    loaded = {b["agent_id"]: b for b in pq.load_budgets()}
    assert loaded["heavy"]["balance"] == 15.0
    assert loaded["heavy"]["weight"] == 2.0
    assert loaded["heavy"]["total_consumed"] == 100.0
    assert loaded["light"]["balance"] == -3.0
    pq.close()
