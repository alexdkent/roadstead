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


# --- persistence cleanup: payload retention + WAL/vacuum maintenance ---------

_BIG = {"messages": [{"role": "user", "content": "x" * 2000}]}


def _pc_big(pq, rid, status="ok"):
    """A completion carrying payload+response blobs (the bytes we shed)."""
    pq.persist_complete(rid, "agentA", "chat", "site", 3, 10, 5, 1.0, 2.0,
                        status, payload=_BIG, response=_BIG)


def test_cleanup_old_payloads_nulls_old_keeps_recent_and_metadata(tmp_path):
    import time
    pq = PersistentQueue(str(tmp_path / "q.db"))
    _pc_big(pq, "old")
    _pc_big(pq, "new")
    # Backdate "old" well past the retention window.
    pq._conn.execute(
        "UPDATE proxy_completions SET completed_at=? WHERE request_id=?",
        (time.time() - 100_000, "old"))
    pq.cleanup_old_payloads(max_age_s=3600)
    rows = {r[0]: r for r in pq._conn.execute(
        "SELECT request_id, payload_json, response_json FROM proxy_completions"
    ).fetchall()}
    # old: blobs gone; new: blobs intact.
    assert rows["old"][1] is None and rows["old"][2] is None
    assert rows["new"][1] is not None and rows["new"][2] is not None
    # The METADATA row for "old" is RETAINED (not deleted) and still queryable.
    assert pq._conn.execute(
        "SELECT COUNT(*) FROM proxy_completions").fetchone()[0] == 2
    assert any(r["request_id"] == "old" for r in pq.recent_requests(10))
    # Idempotent: a second sweep is a cheap no-op (guard clause), no error.
    pq.cleanup_old_payloads(max_age_s=3600)
    pq.close()


def test_open_sets_incremental_auto_vacuum(tmp_path):
    # On a fresh DB the _open PRAGMA takes effect immediately (2 == INCREMENTAL).
    pq = PersistentQueue(str(tmp_path / "q.db"))
    assert pq._conn.execute("PRAGMA auto_vacuum").fetchone()[0] == 2
    pq.close()


def test_checkpoint_truncate_shrinks_wal(tmp_path):
    pq = PersistentQueue(str(tmp_path / "q.db"))
    for i in range(500):
        _pc_big(pq, f"r{i}")
    before = pq.wal_size_bytes()
    assert before > 0
    pq.checkpoint_truncate()  # pre-writer → runs synchronously on the conn
    assert pq.wal_size_bytes() < before
    pq.close()


def test_incremental_vacuum_reduces_freelist(tmp_path):
    pq = PersistentQueue(str(tmp_path / "q.db"))
    for i in range(800):
        _pc_big(pq, f"r{i}")
    pq.cleanup_old_completions(max_age_s=-1)  # delete all → build freelist
    pq.checkpoint_truncate()
    before = pq.freelist_bytes()
    assert before > 0
    pq.incremental_vacuum(1_000_000)
    pq.checkpoint_truncate()
    assert pq.freelist_bytes() < before
    pq.close()


def test_vacuum_full_blocking_reclaims_and_threshold_gates(tmp_path):
    pq = PersistentQueue(str(tmp_path / "q.db"))
    for i in range(800):
        _pc_big(pq, f"r{i}")
    pq.cleanup_old_completions(max_age_s=-1)
    pq.checkpoint_truncate()
    # threshold 0 → vacuums; reclaims the dead pages built above.
    reclaimed = pq.vacuum_full_blocking(threshold_bytes=0)
    assert reclaimed > 0
    # An astronomically high threshold → skip (returns 0, no-op).
    assert pq.vacuum_full_blocking(threshold_bytes=10**12) == 0
    pq.close()


def test_vacuum_full_blocking_refuses_post_writer(tmp_path):
    import pytest
    pq = PersistentQueue(str(tmp_path / "q.db"))
    pq.start_async_writer()  # writer now owns the connection
    with pytest.raises(RuntimeError):
        pq.vacuum_full_blocking(0)
    pq.close()


def test_submit_writer_call_runs_on_writer_thread(tmp_path):
    import threading
    pq = PersistentQueue(str(tmp_path / "q.db"))
    pq.start_async_writer()
    seen = {}

    def op(conn):
        seen["tid"] = threading.get_ident()
        conn.execute("CREATE TABLE IF NOT EXISTS probe(x)")
        conn.execute("INSERT INTO probe VALUES (1)")

    pq._submit_writer_call(op)
    pq.flush(timeout=5.0)
    # Ran on the writer thread, NOT the caller (single-writer invariant).
    assert seen.get("tid") and seen["tid"] != threading.get_ident()
    assert pq._reader().execute("SELECT COUNT(*) FROM probe").fetchone()[0] == 1
    pq.close()


def test_recover_queued_drops_stream_rows(tmp_path):
    """Phase 2 hardening: a queued STREAMING request recovered after a restart
    has no SSE consumer (it died with the old process) — recovery must DROP it,
    not re-dispatch into a permanent scheduler-slot leak. Sync rows recover."""
    from originfleet.llmproxy.scheduler import QueuedRequest

    pq = PersistentQueue(str(tmp_path / "q.db"))
    sync_req = QueuedRequest.create(
        agent_id="a", endpoint="thinker", priority="P3_INGESTION",
        call_site="t", payload_type="chat_completion",
        payload={"messages": [{"role": "user", "content": "x"}]},
        timeout_s=120.0,
    )
    stream_req = QueuedRequest.create(
        agent_id="a", endpoint="thinker", priority="P3_INGESTION",
        call_site="t", payload_type="chat_completion",
        payload={"messages": [{"role": "user", "content": "y"}], "stream": True},
        timeout_s=120.0,
    )
    assert stream_req.stream and not sync_req.stream
    pq.persist_enqueue(sync_req)
    pq.persist_enqueue(stream_req)

    recovered = pq.recover_queued(now=0.0)
    assert [r.request_id for r in recovered] == [sync_req.request_id]
    # The stream row is gone from the table, not just unreturned.
    remaining = pq._conn.execute(
        "SELECT COUNT(*) FROM proxy_queue WHERE stream=1").fetchone()[0]
    assert remaining == 0
    pq.close()
