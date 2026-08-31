"""Tests for the timeout-shadow write path in ProxyService.

Every completed call feeds the timeout model and (when successful) writes
a shadow row recording what `recommended` would have been vs the actual
latency. The shadow path is observational and must never disturb the
caller/scheduler, so a fault in it is swallowed.
"""

from __future__ import annotations

import time

import pytest

from roadstead.config import ProxyConfig
from roadstead.scheduler import DispatchDecision, QueuedRequest
from roadstead.service import ProxyService


def _make_service(tmp_path) -> ProxyService:
    return ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "queue.db")))


def _make_req(endpoint: str, *, now: float, timeout_s: float = 300.0) -> QueuedRequest:
    return QueuedRequest.create(
        agent_id="sidekick",
        endpoint=endpoint,
        priority="P1_TURN_SUPPORT",
        call_site="sidekick.test",
        payload_type="chat_completion",
        payload={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64},
        timeout_s=timeout_s,
        now=now,
    )


def _shadow_rows(svc, request_id: str):
    return svc._queue_db._conn.execute(
        "SELECT actual_total_ms, recommended_ms, would_timeout, applied_timeout_s "
        "FROM proxy_timeout_shadow WHERE request_id=?",
        (request_id,),
    ).fetchall()


def test_shadow_row_written_and_model_fed(tmp_path):
    svc = _make_service(tmp_path)
    t0 = time.monotonic()
    req = _make_req("rerank", now=t0, timeout_s=300.0)

    # 12s end-to-end vs rerank's 10s floor → would_timeout True.
    svc._record_timeout_shadow(req, t0 + 12.0, duration_s=11.0, output_tokens=40, status="ok")

    rows = _shadow_rows(svc, req.request_id)
    assert len(rows) == 1
    actual_ms, recommended_ms, would_timeout, applied_s = rows[0]
    assert actual_ms == pytest.approx(12000.0, abs=50)
    assert recommended_ms == 10000.0  # cold → floor
    assert would_timeout == 1
    assert applied_s == 300.0
    # The model received the sample.
    assert svc._timeout_model.snapshot()["cells"] == 1


def test_no_shadow_row_for_non_ok(tmp_path):
    svc = _make_service(tmp_path)
    t0 = time.monotonic()
    req = _make_req("thinker", now=t0)

    svc._record_timeout_shadow(req, t0 + 5.0, duration_s=5.0, output_tokens=0, status="error")

    assert _shadow_rows(svc, req.request_id) == []
    # error sample is not folded into the distribution either.
    assert svc._timeout_model.snapshot()["cells"] == 0


def test_shadow_fault_does_not_propagate(tmp_path):
    svc = _make_service(tmp_path)
    t0 = time.monotonic()
    req = _make_req("thinker", now=t0)
    decision = DispatchDecision(request=req, queue_wait_ms=100.0, occupancy_at_dispatch=1)

    # Make the shadow computation blow up.
    def _boom(*a, **k):
        raise RuntimeError("synthetic shadow failure")

    svc._timeout_model.advise = _boom  # type: ignore[assignment]

    # Must not raise — the dispatch path completes regardless.
    svc._record_completion(req, decision, 5.0, 2000, 256, "ok")

    # Completion still persisted; no shadow row from the faulted path.
    completed = svc._queue_db._conn.execute(
        "SELECT 1 FROM proxy_completions WHERE request_id=?", (req.request_id,),
    ).fetchall()
    assert len(completed) == 1
    assert _shadow_rows(svc, req.request_id) == []
