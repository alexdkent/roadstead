"""PersistentQueue SQL aggregations — were untested (Phase 4.3 coverage)."""

from __future__ import annotations

from originfleet.llmproxy.queue import PersistentQueue


def _pc(pq, rid, status="ok", in_tok=10, out_tok=5, dur=1.0, qw=2.0):
    pq.persist_complete(rid, "agentA", "thinker", "site", 3,
                        in_tok, out_tok, dur, qw, status)


def test_history_buckets_aggregates(tmp_path):
    pq = PersistentQueue(str(tmp_path / "q.db"))
    _pc(pq, "r1", status="ok")
    _pc(pq, "r2", status="error")
    buckets = pq.history_buckets(hours=1, bucket_minutes=60)
    assert buckets
    ep = buckets[0]["per_endpoint"]["thinker"]
    assert ep["requests"] == 2 and ep["ok"] == 1 and ep["errors"] == 1
    pq.close()


def test_recent_requests(tmp_path):
    pq = PersistentQueue(str(tmp_path / "q.db"))
    _pc(pq, "r1")
    _pc(pq, "r2")
    rows = pq.recent_requests(10)
    ids = {r["request_id"] for r in rows}
    assert ids == {"r1", "r2"}
    pq.close()


def test_completions_for_calibration_filters(tmp_path):
    pq = PersistentQueue(str(tmp_path / "q.db"))
    _pc(pq, "ok1", status="ok", out_tok=5, dur=1.0)
    _pc(pq, "err1", status="error", out_tok=5, dur=1.0)   # excluded (not ok)
    _pc(pq, "zero", status="ok", out_tok=0, dur=1.0)      # excluded (0 out tokens)
    rows = pq.completions_for_calibration(hours=1)
    assert {r["call_site"] for r in rows}  # at least one
    # only the ok + non-zero-token row qualifies
    assert len(rows) == 1
    pq.close()


def test_timeouts_report(tmp_path):
    pq = PersistentQueue(str(tmp_path / "q.db"))
    pq.persist_timeout_event(
        request_id="t1", endpoint="thinker", priority=3, agent_id="a",
        call_site="s", layer="backend", elapsed_s=10.0, applied_timeout_s=60.0,
        queue_wait_ms=5.0, in_flight=2, queued=1, max_slots=32, est_in=100, est_out=50,
        recommended_ms=15000.0, under_recommended=True)
    rep = pq.timeouts_report(hours=1)
    assert rep["total"] == 1 and rep["premature"] == 1
    assert rep["rows"][0]["layer"] == "backend"
    pq.close()
