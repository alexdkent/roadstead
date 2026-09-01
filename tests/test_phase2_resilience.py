"""Phase 2 — resilience for the LLM proxy.

  2.1 bounded in-flight drain on shutdown + draining rejects new work.
  2.2 SQLite writes run on a dedicated writer thread; reads see them.
  2.3 retention sweep trims old rows.
  2.4 load-shed: non-interactive submits shed (429 + Retry-After) when saturated;
      interactive is never shed.
  2.5 alerting: a dead backend produces an endpoint_paused alert; the misleading
      balance-sign agent_starvation alert is gone.
  2.6 queue recovery rebases monotonic deadlines via wall-clock.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from roadstead.backend import BackendResponse
from roadstead.config import ProxyConfig
from roadstead.observability import check_alerts, RollingMetrics
from roadstead.queue import PersistentQueue
from roadstead.scheduler import QueuedRequest
from roadstead.service import ProxyService


class _FakeRequest:
    class _Client:
        host = "172.16.0.5"

    client = _Client()
    headers: dict = {}


_COMPLETION = {
    "id": "c", "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
}


# --- 2.6 queue recovery (wall-clock rebase) ---------------------------------

def _enqueue(pq, rid, timeout_s=60.0):
    req = QueuedRequest.create(
        agent_id="a", endpoint="tier3", priority="P3_INGESTION",
        call_site="t", payload_type="chat_completion",
        payload={"messages": []}, timeout_s=timeout_s,
        caller_id="goose/recipe", request_id=rid)
    pq.persist_enqueue(req)
    return req


def test_recovery_rebases_deadline_via_wallclock(tmp_path):
    db = str(tmp_path / "q.db")
    pq = PersistentQueue(db)
    _enqueue(pq, "r1", timeout_s=60.0)
    # Simulate 30s of wall-clock elapsed before the (restart) recovery.
    pq._conn.execute(
        "UPDATE proxy_queue SET enqueued_wall=? WHERE request_id=?",
        (time.time() - 30.0, "r1"))
    now = time.monotonic()
    recovered = pq.recover_queued(now)
    assert len(recovered) == 1
    r = recovered[0]
    # ~30s of the 60s budget remains → deadline ≈ now + 30.
    assert 28.0 < (r.timeout_deadline - now) < 32.0
    assert r.caller_id == "goose/recipe"   # caller_id now round-trips
    assert r.timeout_s == 60.0
    pq.close()


def test_recovery_discards_expired(tmp_path):
    db = str(tmp_path / "q.db")
    pq = PersistentQueue(db)
    _enqueue(pq, "r2", timeout_s=60.0)
    pq._conn.execute(
        "UPDATE proxy_queue SET enqueued_wall=? WHERE request_id=?",
        (time.time() - 100.0, "r2"))  # 100s elapsed > 60s budget → expired
    recovered = pq.recover_queued(time.monotonic())
    assert recovered == []
    pq.close()


# --- 2.2 writer thread ------------------------------------------------------

def test_async_writer_applies_writes(tmp_path):
    db = str(tmp_path / "q.db")
    pq = PersistentQueue(db)
    pq.start_async_writer()
    assert pq._writer is not None and pq._writer.is_alive()
    pq.persist_complete(
        "rid1", "agentA", "tier3", "site", 3, 10, 5, 0.2, 0.0, "ok",
        finish_reason="stop")
    pq.flush(timeout=5.0)
    rows = pq.recent_requests(10)
    assert any(r["request_id"] == "rid1" for r in rows), rows
    pq.close()
    assert not (pq._writer and pq._writer.is_alive())  # writer joined on close


# --- 2.3 retention sweep ----------------------------------------------------

def test_cleanup_removes_old_rows(tmp_path):
    db = str(tmp_path / "q.db")
    pq = PersistentQueue(db)  # no writer → synchronous
    pq.persist_complete("old", "a", "tier3", "s", 3, 1, 1, 0.1, 0.0, "ok")
    pq._conn.execute(
        "UPDATE proxy_completions SET completed_at=? WHERE request_id=?",
        (time.time() - 86400 * 30, "old"))  # 30 days old
    pq.persist_complete("fresh", "a", "tier3", "s", 3, 1, 1, 0.1, 0.0, "ok")
    pq.cleanup_old_completions(max_age_s=86400 * 7)
    ids = {r["request_id"] for r in pq.recent_requests(10)}
    assert "old" not in ids and "fresh" in ids
    pq.close()


# --- service integration harness --------------------------------------------

async def _started_svc():
    svc = ProxyService(ProxyConfig())

    async def ok_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(200, _COMPLETION, 0.01, 5, 2, finish_reason="stop")

    async def _none(*a, **k):
        return None

    svc._backend.call = ok_call
    svc._backend.probe_vllm_capacity = _none
    svc._backend.probe_props = _none
    svc._backend.probe_models = _none
    await svc.startup()
    return svc


def _body(priority, timeout_s=10.0):
    return {
        "agent_id": "a", "endpoint": "tier3", "priority": priority,
        "call_site": "t", "payload_type": "chat_completion",
        "payload": {"messages": [{"role": "user", "content": "x"}]},
        "timeout_s": timeout_s,
    }


# --- 2.4 load-shed ----------------------------------------------------------

@pytest.mark.asyncio
async def test_background_shed_with_retry_after():
    svc = await _started_svc()
    try:
        svc._shed_depth = 0  # any non-interactive submit sheds
        resp = await svc.handle_submit(_body("P3_INGESTION"), _FakeRequest())
        assert resp.status_code == 429
        assert int(resp.headers["Retry-After"]) >= 5
        env = json.loads(resp.body.decode())
        assert "backpressure" in env["error"]
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_interactive_never_shed():
    svc = await _started_svc()
    try:
        svc._shed_depth = 0  # would shed everything non-interactive
        resp = await svc.handle_submit(_body("P0_REALTIME"), _FakeRequest())
        assert resp.status_code == 200  # interactive served, not shed
    finally:
        await svc.shutdown()


# --- 2.5 alerting -----------------------------------------------------------

@pytest.mark.asyncio
async def test_unhealthy_backend_raises_endpoint_paused_alert():
    svc = await _started_svc()
    try:
        svc._endpoint_health["tier3"] = {
            "healthy": False, "consecutive_failures": 5, "unhealthy_since": time.monotonic()}
        svc._evaluate_alerts(time.monotonic())
        names = {a["name"] for a in svc._alerts}
        assert "endpoint_paused" in names, svc._alerts
    finally:
        await svc.shutdown()


def test_agent_starvation_alert_removed():
    # A negative-balance agent must NOT trigger agent_starvation (the DRR fix
    # made balance-sign starvation a lie).
    alerts = check_alerts(
        endpoint_snapshots={},
        agent_budgets=[{"agent_id": "heavy", "starving": True, "balance_ss": -50.0}],
        metrics=RollingMetrics(),
        cost_model_samples={},
        queue_wal_size=0,
        now=time.monotonic(),
    )
    assert not any(a.name == "agent_starvation" for a in alerts)


# --- 2.1 drain --------------------------------------------------------------

@pytest.mark.asyncio
async def test_draining_rejects_new_submits():
    svc = await _started_svc()
    try:
        svc._draining.set()
        resp = await svc.handle_submit(_body("P1_TURN_SUPPORT"), _FakeRequest())
        assert resp.status_code == 503
        assert "draining" in json.loads(resp.body.decode())["error"]
    finally:
        svc._draining.clear()
        await svc.shutdown()


@pytest.mark.asyncio
async def test_shutdown_drains_inflight_tasks():
    svc = await _started_svc()
    done = {"ran": False}

    async def quick():
        await asyncio.sleep(0.05)
        done["ran"] = True

    task = asyncio.create_task(quick())
    svc._inflight_tasks["x"] = task
    task.add_done_callback(lambda t: svc._inflight_tasks.pop("x", None))
    await svc.shutdown()        # should await the in-flight task (bounded)
    assert done["ran"] is True  # drained to completion, not dropped
