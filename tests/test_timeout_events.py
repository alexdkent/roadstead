"""Tests for timeout-event logging.

Three timeout paths used to be silent (admission expiry, client-wait, and
streaming) and backend timeouts were misclassified as generic errors —
so `status="timeout"` was never emitted. These tests pin the new
behavior: every timeout is recorded once, with load context, distinct
from errors, and the metrics counter becomes real.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from originfleet.llmproxy.config import ProxyConfig
from originfleet.llmproxy.scheduler import DispatchDecision, QueuedRequest
from originfleet.llmproxy.service import ProxyService


class _FakeRequest:
    def __init__(self, **params):
        self.query_params = {k: str(v) for k, v in params.items()}


def _svc(tmp_path) -> ProxyService:
    return ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db")))


def _req(endpoint: str = "thinker", *, timeout_s: float = 300.0) -> QueuedRequest:
    return QueuedRequest.create(
        agent_id="sidekick",
        endpoint=endpoint,
        priority="P1_TURN_SUPPORT",
        call_site="sidekick.test",
        payload_type="chat_completion",
        payload={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 256},
        timeout_s=timeout_s,
        now=time.monotonic(),
    )


def _rows(svc, request_id: str):
    return svc._queue_db._conn.execute(
        "SELECT layer, elapsed_s, under_recommended, in_flight, queued "
        "FROM proxy_timeouts WHERE request_id=?",
        (request_id,),
    ).fetchall()


# ----- recording -----

def test_client_wait_event_recorded_with_metrics(tmp_path):
    svc = _svc(tmp_path)
    req = _req()
    svc._record_timeout_event(req, layer="client_wait", elapsed_s=300.0)
    rows = _rows(svc, req.request_id)
    assert len(rows) == 1
    assert rows[0][0] == "client_wait"
    # client_wait doesn't flow through _record_completion, so the event
    # owns the metrics + log emission.
    assert svc._metrics.count(status="timeout") == 1


def test_backend_event_does_not_double_count_metrics(tmp_path):
    svc = _svc(tmp_path)
    req = _req()
    # backend layer pairs with _record_completion (which owns metrics),
    # so the event must NOT emit a second metrics sample.
    svc._record_timeout_event(
        req, layer="backend", elapsed_s=5.0, queue_wait_ms=100.0,
        emit_metrics_and_log=False,
    )
    assert len(_rows(svc, req.request_id)) == 1
    assert svc._metrics.count(status="timeout") == 0


def test_timeout_event_deduped_across_layers(tmp_path):
    svc = _svc(tmp_path)
    req = _req()
    svc._record_timeout_event(req, layer="client_wait", elapsed_s=300.0)
    svc._record_timeout_event(req, layer="admission", elapsed_s=300.0)
    assert len(_rows(svc, req.request_id)) == 1
    assert svc._metrics.count(status="timeout") == 1


def test_premature_flag_when_below_recommended(tmp_path):
    svc = _svc(tmp_path)
    # cold model → thinker recommended = 180s floor; a 5s timeout is premature.
    req = _req(timeout_s=5.0)
    svc._record_timeout_event(req, layer="client_wait", elapsed_s=5.0)
    rows = _rows(svc, req.request_id)
    assert rows[0][2] == 1  # under_recommended


@pytest.mark.asyncio
async def test_admission_timeout_records_and_releases_caller(tmp_path):
    svc = _svc(tmp_path)
    loop = asyncio.get_running_loop()
    req = _req()
    fut = loop.create_future()
    svc._pending_futures[req.request_id] = fut

    svc._on_admission_timeout(req)

    assert fut.done()
    assert fut.result()["status"] == "timeout"
    rows = _rows(svc, req.request_id)
    assert len(rows) == 1
    assert rows[0][0] == "admission"


def test_record_completion_timeout_status_distinct_from_error(tmp_path):
    svc = _svc(tmp_path)
    req = _req()
    decision = DispatchDecision(request=req, queue_wait_ms=100.0, occupancy_at_dispatch=2)
    svc._record_completion(req, decision, 5.0, 0, 0, "timeout")

    status = svc._queue_db._conn.execute(
        "SELECT status FROM proxy_completions WHERE request_id=?", (req.request_id,),
    ).fetchone()[0]
    assert status == "timeout"
    assert svc._metrics.count(status="timeout") == 1
    # non-ok → never folds into the latency distribution / shadow log.
    assert svc._queue_db._conn.execute(
        "SELECT 1 FROM proxy_timeout_shadow WHERE request_id=?", (req.request_id,),
    ).fetchall() == []


# ----- report -----

def test_timeouts_report_aggregates(tmp_path):
    svc = _svc(tmp_path)
    for i in range(3):
        svc._queue_db.persist_timeout_event(
            request_id=f"b{i}", endpoint="thinker", priority=1, agent_id="sidekick",
            call_site="x", layer="backend", elapsed_s=10.0 + i,
            applied_timeout_s=300.0, queue_wait_ms=50.0, in_flight=2, queued=5,
            max_slots=6, est_in=2000, est_out=256, recommended_ms=180000.0,
            under_recommended=(i < 2),
        )
    rep = svc._queue_db.timeouts_report(24)
    assert rep["total"] == 3
    assert rep["premature"] == 2
    assert len(rep["rows"]) == 1
    row = rep["rows"][0]
    assert (row["endpoint"], row["layer"], row["count"]) == ("thinker", "backend", 3)
    assert row["premature"] == 2
    assert row["avg_in_flight"] == 2.0
    assert row["avg_queued"] == 5.0


@pytest.mark.asyncio
async def test_timeouts_report_endpoint(tmp_path):
    svc = _svc(tmp_path)
    svc._queue_db.persist_timeout_event(
        request_id="a0", endpoint="thinker", priority=3, agent_id="sidekick",
        call_site="x", layer="admission", elapsed_s=12.0, applied_timeout_s=300.0,
        queue_wait_ms=None, in_flight=3, queued=9, max_slots=6, est_in=100,
        est_out=64, recommended_ms=180000.0, under_recommended=True,
    )
    resp = await svc.handle_timeouts_report(_FakeRequest(hours=24))
    body = json.loads(resp.body.decode())
    assert resp.status_code == 200
    assert body["total"] == 1
    assert body["premature"] == 1
    assert body["rows"][0]["layer"] == "admission"
