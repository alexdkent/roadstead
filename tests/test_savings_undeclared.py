"""`/v1/fleet/savings` names the endpoints nobody priced.

🚨 THE DEFECT: an endpoint with no pricing decision books $0 exactly as a
genuinely free one does, so the total keeps looking plausible while a lane is
missing from it. On the reference fleet ~$119 of ~$678 lifetime savings was
unbooked across ten names, and `is_declared_endpoint()` — written to catch it —
was called by nothing, so the check existed and never ran.

These drive the PRODUCER (`savings_summary`) and the ROUTE, because a helper
that is unit-tested and never reached is the defect being fixed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

import pytest

from roadstead import usage_rates
from roadstead.config import ProxyConfig
from roadstead.queue import PersistentQueue
from roadstead.service import ProxyService

DAY = 86400


@pytest.fixture(autouse=True)
def _fresh_warned_set(monkeypatch):
    monkeypatch.setattr(usage_rates, "_WARNED_UNDECLARED", set())
    monkeypatch.setattr(usage_rates, "_cap_notice_sent", False)


def _push(pq, rid, endpoint, in_tok, out_tok, *, kind="external"):
    pq.persist_external_call(
        request_id=rid, agent_id="pusher", endpoint=endpoint, call_site="site",
        kind=kind, input_tokens=in_tok, output_tokens=out_tok,
        duration_s=0.1, status="ok")


@pytest.fixture
def pq(tmp_path):
    q = PersistentQueue(str(tmp_path / "q.db"))
    try:
        yield q
    finally:
        q.close()


def test_an_unpriced_name_is_listed_with_its_volume(pq):
    _push(pq, "a1", "never-priced-unit", 1_000, 7)
    _push(pq, "a2", "never-priced-unit", 500, 3)
    _push(pq, "d1", "tier1", 1_000_000, 0, kind="chat")
    pq.flush(timeout=5.0)

    out = pq.savings_summary()
    assert out["undeclared"] == [{
        "endpoint": "never-priced-unit", "requests": 2,
        "input_tokens": 1_500, "output_tokens": 10,
    }]
    # ...and it still books $0 while the priced lane is untouched — pricing is
    # fail-open, the report is the loud half.
    by_ep = {r["endpoint"]: r for r in out["by_endpoint"]}
    assert by_ep["never-priced-unit"]["total_usd"] == 0.0
    assert by_ep["tier1"]["total_usd"] > 0.0


def test_a_deliberately_free_lane_is_not_reported(pq):
    """`stream` books $0 for a stated reason. Reporting it would bury the real
    misses under lanes that are decided — the report has to be worth reading."""
    _push(pq, "s1", "stream", 100, 1)
    _push(pq, "s2", "cortex-stream", 100, 1)
    _push(pq, "m1", "meshgen", 1, 0)
    pq.flush(timeout=5.0)
    assert pq.savings_summary()["undeclared"] == []


def test_the_lifetime_volume_survives_the_prune_but_requests_do_not(pq):
    """`input_tokens`/`output_tokens` are LIFETIME (the rollup keeps tokens);
    `requests` counts only what proxy_completions still holds. The payload
    documents that split, so pin it rather than let it drift into a claim."""
    now = time.time()
    pq._conn.execute(
        "INSERT INTO proxy_completions (request_id, agent_id, endpoint, "
        "call_site, priority, input_tokens, output_tokens, duration_s, "
        "queue_wait_ms, status, completed_at, kind) "
        "VALUES ('old','p','ghost-unit',  'site', 3, 900, 9, 0.1, 0, 'ok', ?, 'external')",
        (now - 40 * DAY,))
    pq.flush(timeout=5.0)
    pq.cleanup_old_completions(max_age_s=30 * DAY)
    assert pq._conn.execute(
        "SELECT COUNT(*) FROM proxy_completions").fetchone()[0] == 0

    (row,) = pq.savings_summary()["undeclared"]
    assert row["endpoint"] == "ghost-unit"
    assert row["input_tokens"] == 900 and row["output_tokens"] == 9
    assert row["requests"] == 0


def test_the_biggest_miss_comes_first(pq):
    _push(pq, "x1", "small-miss", 10, 0)
    _push(pq, "x2", "big-miss", 10_000, 0)
    pq.flush(timeout=5.0)
    assert [r["endpoint"] for r in pq.savings_summary()["undeclared"]] == [
        "big-miss", "small-miss"]


def test_first_sight_is_logged_at_warning_once(pq, caplog):
    _push(pq, "w1", "warn-me", 1, 1)
    pq.flush(timeout=5.0)
    with caplog.at_level(logging.WARNING, logger="roadstead.usage_rates"):
        pq.savings_summary()
        pq.savings_summary()          # a dashboard polls; the log must not
    hits = [r for r in caplog.records if "warn-me" in r.getMessage()]
    assert len(hits) == 1 and hits[0].levelno == logging.WARNING


def test_the_usage_rollup_warns_but_keeps_its_shape(pq, caplog):
    """`/v1/usage` prices too. It only logs; its payload is an API contract
    with no `undeclared` field to grow."""
    _push(pq, "u1", "usage-only-miss", 1, 1)
    pq.flush(timeout=5.0)
    with caplog.at_level(logging.WARNING, logger="roadstead.usage_rates"):
        rows = pq.usage_rollup(dimension="endpoint", hours=24.0)
    assert [r["key"] for r in rows] == ["usage-only-miss"]
    assert any("usage-only-miss" in r.getMessage() for r in caplog.records)


def test_the_route_carries_it(tmp_path):
    """The producer alone proves nothing about what the dashboard receives."""
    class _Req:
        query_params: dict = {}

    svc = ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db")))
    pq = svc._queue_db
    try:
        _push(pq, "r1", "route-miss", 40, 2)
        pq.flush(timeout=5.0)
        body = json.loads(asyncio.run(svc.handle_fleet_savings(_Req())).body)
        assert [r["endpoint"] for r in body["undeclared"]] == ["route-miss"]
    finally:
        pq.close()


def test_the_unopened_db_still_has_the_key(tmp_path):
    """A consumer iterating `undeclared` must not KeyError on a degraded read."""
    q = PersistentQueue(str(tmp_path / "gone.db"))
    q.close()
    q._conn = None
    assert q.savings_summary()["undeclared"] == []
