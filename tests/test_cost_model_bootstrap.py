"""Cost model bootstrap from proxy_completions table.

On every restart, the cost model loses all calibration (EWMA trackers,
per-call-site output lengths, decode TPS curves). The bootstrap replays
recent successful completions from the persistent queue DB so the model
starts calibrated instead of using static defaults.
"""

from __future__ import annotations

import sqlite3
import time

import pytest

from roadstead.config import ProxyConfig
from roadstead.service import ProxyService


def _seed_completions(conn: sqlite3.Connection, n: int = 20) -> None:
    """Insert synthetic completion rows into proxy_completions."""
    now = time.time()
    for i in range(n):
        conn.execute(
            "INSERT INTO proxy_completions "
            "(request_id, agent_id, endpoint, call_site, priority, "
            " input_tokens, output_tokens, duration_s, queue_wait_ms, "
            " status, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"req_{i:04d}", "forum-agent", "tier3",
                "forum-agent.proposal_emitter", 3,
                6000 + i * 100, 250 + i * 10,
                25.0 + i * 0.5, 5.0 + i,
                "ok", now - (n - i) * 60,
            ),
        )


def _register_endpoints(svc: ProxyService) -> None:
    """Register endpoints in the cost model (normally done by startup())."""
    for ep_name, ep_cfg in svc._config.endpoints.items():
        svc._cost_model.register_endpoint(ep_name, ep_cfg.max_slots)


def test_bootstrap_populates_cost_model(tmp_path):
    db_path = str(tmp_path / "queue.db")
    config = ProxyConfig(queue_db_path=db_path)
    svc = ProxyService(config)
    _register_endpoints(svc)

    _seed_completions(svc._queue_db._conn, n=20)

    svc._bootstrap_cost_model()

    model = svc._cost_model.get("tier3")
    assert model is not None
    assert "forum-agent.proposal_emitter" in model.output_length_ewma
    ewma = model.output_length_ewma["forum-agent.proposal_emitter"]
    assert ewma.sample_count == 20
    assert ewma.value > 0


def test_bootstrap_skips_errors_and_zero_tokens(tmp_path):
    db_path = str(tmp_path / "queue.db")
    config = ProxyConfig(queue_db_path=db_path)
    svc = ProxyService(config)
    _register_endpoints(svc)

    now = time.time()
    conn = svc._queue_db._conn
    conn.execute(
        "INSERT INTO proxy_completions "
        "(request_id, agent_id, endpoint, call_site, priority, "
        " input_tokens, output_tokens, duration_s, queue_wait_ms, "
        " status, completed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("req_err", "sidekick", "tier3", "sidekick.critic", 1,
         5000, 0, 45.0, 100.0, "error", now - 60),
    )
    conn.execute(
        "INSERT INTO proxy_completions "
        "(request_id, agent_id, endpoint, call_site, priority, "
        " input_tokens, output_tokens, duration_s, queue_wait_ms, "
        " status, completed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("req_zero", "sidekick", "tier3", "sidekick.critic", 1,
         5000, 0, 30.0, 50.0, "ok", now - 30),
    )

    svc._bootstrap_cost_model()

    model = svc._cost_model.get("tier3")
    assert model is not None
    assert model.output_length_ewma == {}


def test_bootstrap_no_db():
    config = ProxyConfig()
    svc = ProxyService(config)
    _register_endpoints(svc)
    svc._bootstrap_cost_model()
    model = svc._cost_model.get("tier3")
    assert model is not None
    assert model.output_length_ewma == {}
