"""PersistentQueue SQL aggregations — were untested (Phase 4.3 coverage)."""

from __future__ import annotations

import queue as _queue

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


# --- Phase 5B.3: writer-thread hardening ------------------------------------

def test_writer_alive_true_in_prewriter_mode(tmp_path):
    # Before start_async_writer, writes run synchronously on the loop thread —
    # that's the LEGITIMATE single-threaded mode, reported healthy (not "dead").
    pq = PersistentQueue(str(tmp_path / "q.db"))
    assert pq.writer_alive() is True
    assert pq.writer_restarts() == 0
    pq.close()


def test_write_queue_is_bounded(tmp_path):
    pq = PersistentQueue(str(tmp_path / "q.db"))
    pq.start_async_writer()
    assert pq._write_q.maxsize == pq._write_q_maxsize > 0  # not unbounded
    pq.close()


def test_writer_restarts_on_death(tmp_path):
    pq = PersistentQueue(str(tmp_path / "q.db"))
    pq.start_async_writer()
    # Kill the writer (sentinel + join) — simulate a dead writer thread.
    pq._write_q.put(None)
    pq._writer.join(timeout=2.0)
    assert not pq._writer.is_alive()
    assert pq.writer_alive() is False  # detected dead
    # A write must self-heal: restart the writer + apply the write, NOT silently
    # run it on the loop thread.
    _pc(pq, "after_death")
    assert pq.writer_restarts() >= 1
    assert pq.writer_alive() is True
    pq.flush(timeout=5.0)
    assert any(r["request_id"] == "after_death" for r in pq.recent_requests(10))
    pq.close()


def test_write_queue_overflow_drops_not_blocks(tmp_path):
    # A full bounded queue must DROP + count, never block the event loop.
    pq = PersistentQueue(str(tmp_path / "q.db"))
    pq._write_q_maxsize = 2
    pq._write_q = _queue.Queue(maxsize=2)
    pq._writer_started_once = True

    class _AliveButIdle:  # looks alive, never drains the queue
        def is_alive(self):
            return True

    pq._writer = _AliveButIdle()
    for _ in range(5):  # 2 fit, 3 overflow
        pq._w("INSERT INTO proxy_completions(request_id) VALUES ('x')")
    assert pq.write_q_dropped() == 3  # dropped, and we got here (never blocked)
    pq._writer = None  # avoid close() touching the stub
    pq._writer_started_once = False
    pq.close()
