"""Endpoint tests for /v1/timeout-advice and its shadow-report.

Drives a real ProxyService over a seeded in-memory-ish queue DB and calls
the async handlers with a fake request, asserting shape, normalization,
input validation, and the shadow-report aggregation math.
"""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from originfleet.llmproxy.config import ProxyConfig
from originfleet.llmproxy.service import ProxyService


class _FakeRequest:
    def __init__(self, **params):
        # str-coerce like real query params; handlers call .get().
        self.query_params = {k: str(v) for k, v in params.items()}


def _seed_completions(conn: sqlite3.Connection, ep: str, priority: int, n: int):
    now = time.time()
    for i in range(n):
        conn.execute(
            "INSERT INTO proxy_completions "
            "(request_id, agent_id, endpoint, call_site, priority, "
            " input_tokens, output_tokens, duration_s, queue_wait_ms, "
            " status, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"{ep}_{i:04d}", "sidekick", ep, "sidekick.test", priority,
                2000, 256, 5.0 + i * 0.1, 200.0,
                "ok", now - (n - i),
            ),
        )


def _make_service(tmp_path) -> ProxyService:
    config = ProxyConfig(
        queue_db_path=str(tmp_path / "queue.db"),
        timeout_advice_min_samples=30,
    )
    return ProxyService(config)


@pytest.mark.asyncio
async def test_advice_returns_stats_when_seeded(tmp_path):
    svc = _make_service(tmp_path)
    _seed_completions(svc._queue_db._conn, "thinker", 1, 40)
    svc._bootstrap_timeout_model()

    resp = await svc.handle_timeout_advice(
        _FakeRequest(model="thinker", priority="P1_TURN_SUPPORT", est_in=2000, est_out=256)
    )
    assert resp.status_code == 200
    body = json.loads(resp.body.decode())
    assert body["model"] == "thinker"
    assert body["priority"] == "P1_TURN_SUPPORT"
    assert body["source"] != "floor"
    assert body["sample_count"] == 40
    # recommended is the conservative default; never below the floor.
    assert body["recommended_timeout_s"] >= 180
    for k in ("min_ms", "median_ms", "p95_ms", "recommended_ms"):
        assert k in body


@pytest.mark.asyncio
async def test_advice_normalizes_role_name(tmp_path):
    svc = _make_service(tmp_path)
    _seed_completions(svc._queue_db._conn, "thinker", 1, 35)
    svc._bootstrap_timeout_model()

    resp = await svc.handle_timeout_advice(
        _FakeRequest(model="llama-thinker", priority=1, est_in=2000, est_out=256)
    )
    body = json.loads(resp.body.decode())
    assert resp.status_code == 200
    assert body["model"] == "thinker"
    assert body["source"] != "floor"


@pytest.mark.asyncio
async def test_advice_cold_model_returns_floor(tmp_path):
    svc = _make_service(tmp_path)
    resp = await svc.handle_timeout_advice(
        _FakeRequest(model="gemma-hot", priority="P0_REALTIME", est_in=50, est_out=32)
    )
    body = json.loads(resp.body.decode())
    assert resp.status_code == 200
    assert body["source"] == "floor"
    assert body["recommended_timeout_s"] == 8  # gemma-hot floor


@pytest.mark.asyncio
async def test_advice_input_validation(tmp_path):
    svc = _make_service(tmp_path)

    missing = await svc.handle_timeout_advice(_FakeRequest(priority=1))
    assert missing.status_code == 400

    unknown = await svc.handle_timeout_advice(_FakeRequest(model="nope"))
    assert unknown.status_code == 400

    bad_pri = await svc.handle_timeout_advice(
        _FakeRequest(model="thinker", priority="P9_BOGUS")
    )
    assert bad_pri.status_code == 400

    bad_int = await svc.handle_timeout_advice(
        _FakeRequest(model="thinker", priority=1, est_in="abc")
    )
    assert bad_int.status_code == 400


@pytest.mark.asyncio
async def test_shadow_report_aggregates(tmp_path):
    svc = _make_service(tmp_path)
    conn = svc._queue_db._conn
    # 4 rows: 1 would-timeout, applied 300s, recommended 30s.
    for i in range(4):
        svc._queue_db.persist_timeout_shadow(
            request_id=f"s{i}", endpoint="thinker", priority=1,
            est_in=2000, est_out=256, actual_out=300,
            actual_total_ms=20000.0, applied_timeout_s=300.0,
            recommended_ms=30000.0, p95_ms=25000.0, median_ms=10000.0,
            min_ms=5000.0, source="cell",
            would_timeout=(i == 0),
        )

    resp = await svc.handle_timeout_shadow_report(_FakeRequest(hours=24))
    body = json.loads(resp.body.decode())
    assert resp.status_code == 200
    rows = body["report"]
    assert len(rows) == 1
    row = rows[0]
    assert row["endpoint"] == "thinker"
    assert row["samples"] == 4
    assert row["would_timeout"] == 1
    assert row["would_timeout_rate"] == 0.25
    # headroom = applied(300000ms) - recommended(30000ms) = 270000ms
    assert row["headroom_vs_applied_ms_p50"] == 270000.0
    assert row["sources"] == {"cell": 4}
