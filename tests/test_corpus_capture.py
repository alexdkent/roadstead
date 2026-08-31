"""Payload + response capture in proxy_completions for test harness replay.

Step 1 of the test harness: every sync request's prompt and response
are persisted alongside timing data, enabling recorded-replay and A/B
backend comparison.
"""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from roadstead.config import ProxyConfig
from roadstead.service import ProxyService


def _register_endpoints(svc: ProxyService) -> None:
    for ep_name, ep_cfg in svc._config.endpoints.items():
        svc._cost_model.register_endpoint(ep_name, ep_cfg.max_slots)


class _FakeRequest:
    class _Client:
        host = "127.0.0.1"
    client = _Client()


@pytest.mark.asyncio
async def test_persist_complete_stores_payload_and_response(tmp_path):
    db_path = str(tmp_path / "queue.db")
    config = ProxyConfig(queue_db_path=db_path)
    svc = ProxyService(config)

    payload = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 64}
    response = {"choices": [{"message": {"content": "hi"}}]}

    svc._queue_db.persist_complete(
        "req_001", "sidekick", "thinker", "sidekick.critic", 1,
        500, 50, 2.5, 10.0, "ok",
        payload=payload, response=response,
    )

    rows = svc._queue_db._conn.execute(
        "SELECT payload_json, response_json FROM proxy_completions WHERE request_id='req_001'"
    ).fetchone()
    assert rows is not None
    assert json.loads(rows[0]) == payload
    assert json.loads(rows[1]) == response


def test_export_corpus_returns_payload_records(tmp_path):
    db_path = str(tmp_path / "queue.db")
    config = ProxyConfig(queue_db_path=db_path)
    svc = ProxyService(config)

    now = time.time()
    payload = {"messages": [{"role": "user", "content": "test"}]}
    response = {"choices": [{"message": {"content": "ok"}}]}

    svc._queue_db.persist_complete(
        "req_002", "forum-agent", "thinker", "forum-agent.proposal_emitter", 3,
        6000, 250, 30.0, 5.0, "ok",
        payload=payload, response=response,
    )
    svc._queue_db.persist_complete(
        "req_003", "forum-agent", "thinker", "forum-agent.proposal_emitter", 3,
        6000, 0, 45.0, 5.0, "error",
    )

    corpus = svc._queue_db.export_corpus(hours=1, endpoint="thinker")
    assert len(corpus) == 1
    assert corpus[0]["request_id"] == "req_002"
    assert corpus[0]["payload"] == payload
    assert corpus[0]["response"] == response


def test_export_corpus_filters_by_call_site(tmp_path):
    db_path = str(tmp_path / "queue.db")
    config = ProxyConfig(queue_db_path=db_path)
    svc = ProxyService(config)

    payload = {"messages": [{"role": "user", "content": "x"}]}
    svc._queue_db.persist_complete(
        "req_a", "forum-agent", "thinker", "forum-agent.proposal_emitter", 3,
        100, 50, 1.0, 1.0, "ok", payload=payload, response={},
    )
    svc._queue_db.persist_complete(
        "req_b", "sidekick", "thinker", "sidekick.critic", 1,
        100, 50, 1.0, 1.0, "ok", payload=payload, response={},
    )

    corpus = svc._queue_db.export_corpus(hours=1, call_site="forum-agent.*")
    assert len(corpus) == 1
    assert corpus[0]["call_site"] == "forum-agent.proposal_emitter"


def test_migration_adds_columns_to_existing_db(tmp_path):
    db_path = str(tmp_path / "old.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE proxy_completions (
        request_id TEXT PRIMARY KEY, agent_id TEXT, endpoint TEXT,
        call_site TEXT, priority INTEGER, input_tokens INTEGER,
        output_tokens INTEGER, duration_s REAL, queue_wait_ms REAL,
        status TEXT, completed_at REAL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS proxy_queue (
        request_id TEXT PRIMARY KEY, agent_id TEXT NOT NULL,
        endpoint TEXT NOT NULL, priority INTEGER NOT NULL,
        call_site TEXT NOT NULL, payload_type TEXT NOT NULL,
        payload_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
        enqueued_at REAL NOT NULL, timeout_deadline REAL NOT NULL,
        estimated_cost REAL, session_id TEXT, turn_id TEXT,
        stream INTEGER NOT NULL DEFAULT 0
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS proxy_agent_budgets (
        agent_id TEXT PRIMARY KEY, weight REAL NOT NULL DEFAULT 1.0,
        balance REAL NOT NULL DEFAULT 0.0,
        total_consumed REAL NOT NULL DEFAULT 0.0,
        last_replenish_at REAL NOT NULL
    )""")
    conn.close()

    config = ProxyConfig(queue_db_path=db_path)
    svc = ProxyService(config)

    cols = {r[1] for r in svc._queue_db._conn.execute(
        "PRAGMA table_info(proxy_completions)"
    ).fetchall()}
    assert "payload_json" in cols
    assert "response_json" in cols
