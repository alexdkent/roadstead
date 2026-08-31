"""§5c: heavy dashboard reads run off the event loop via asyncio.to_thread.

A sqlite connection can't be shared across threads, so `_reader()` must hand a
pool thread its OWN read connection (WAL permits concurrent readers) — never the
loop thread's `_read_conn`, and never the writer connection. The single-WRITER
invariant is preserved (these threads only SELECT).
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from roadstead.queue import PersistentQueue


def _make_started_queue(tmp_path) -> PersistentQueue:
    q = PersistentQueue(str(tmp_path / "queue.db"))
    q.start_async_writer()   # opens _read_conn, records the loop thread
    return q


@pytest.mark.asyncio
async def test_offloop_reader_uses_a_distinct_connection(tmp_path):
    q = _make_started_queue(tmp_path)
    try:
        loop_conn = q._reader()                      # on the loop thread
        assert loop_conn is q._read_conn
        pool_conn = await asyncio.to_thread(q._reader)   # on a pool thread
        # A different physical connection (thread-local), never the writer conn.
        assert pool_conn is not None
        assert pool_conn is not q._read_conn
        assert pool_conn is not q._conn
    finally:
        q.close()


@pytest.mark.asyncio
async def test_offloop_read_method_works(tmp_path):
    q = _make_started_queue(tmp_path)
    try:
        # recent_requests is one of the offloaded reads; it must return cleanly
        # when invoked on a pool thread (i.e. its _reader() resolves there).
        rows = await asyncio.to_thread(q.recent_requests, 10)
        assert isinstance(rows, list)
    finally:
        q.close()


@pytest.mark.asyncio
async def test_slow_offloop_read_does_not_block_loop(tmp_path):
    """A read dispatched to a thread must not stall the event loop: a tiny
    sleep concurrent with the read both finish ~together (not serialized)."""
    q = _make_started_queue(tmp_path)
    try:
        async def ticker():
            await asyncio.sleep(0.05)
            return "tick"

        read = asyncio.to_thread(q.recent_requests, 50)
        results = await asyncio.gather(ticker(), read)
        assert results[0] == "tick"
        assert isinstance(results[1], list)
    finally:
        q.close()
