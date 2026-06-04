"""Persistent queue with SQLite WAL backing.

In-memory for performance; WAL for crash recovery.  Queued requests
are persisted on enqueue and removed on dispatch or expiry.  In-flight
requests are NOT persisted — on crash, callers timeout and resubmit.
"""

from __future__ import annotations

import json
import logging
import queue as _queue
import sqlite3
import threading
import time
from pathlib import Path

from .config import LLMPriority, normalize_endpoint, priority_to_band
from .scheduler import QueuedRequest

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS proxy_queue (
    request_id     TEXT PRIMARY KEY,
    agent_id       TEXT NOT NULL,
    endpoint       TEXT NOT NULL,
    priority       INTEGER NOT NULL,
    call_site      TEXT NOT NULL,
    payload_type   TEXT NOT NULL,
    payload_json   TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'queued',
    enqueued_at    REAL NOT NULL,
    timeout_deadline REAL NOT NULL,
    estimated_cost REAL,
    session_id     TEXT,
    turn_id        TEXT,
    stream         INTEGER NOT NULL DEFAULT 0,
    -- Wall-clock recovery (Phase 2.6): enqueued_at/timeout_deadline are
    -- time.monotonic() (process-relative), meaningless after a restart. Persist
    -- wall-clock enqueue + the budget so recovery can rebase onto the new
    -- monotonic clock and discard genuinely-expired rows.
    enqueued_wall  REAL,
    timeout_s      REAL,
    caller_id      TEXT
);

CREATE TABLE IF NOT EXISTS proxy_agent_budgets (
    agent_id       TEXT PRIMARY KEY,
    weight         REAL NOT NULL DEFAULT 1.0,
    balance        REAL NOT NULL DEFAULT 0.0,
    total_consumed REAL NOT NULL DEFAULT 0.0,
    last_replenish_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS proxy_completions (
    request_id       TEXT PRIMARY KEY,
    agent_id         TEXT NOT NULL,
    endpoint         TEXT NOT NULL,
    call_site        TEXT NOT NULL,
    priority         INTEGER NOT NULL,
    input_tokens     INTEGER,
    output_tokens    INTEGER,
    duration_s       REAL,
    queue_wait_ms    REAL,
    status           TEXT NOT NULL,
    completed_at     REAL NOT NULL,
    payload_json     TEXT,
    response_json    TEXT,
    session_id       TEXT,
    turn_id          TEXT,
    caller_id        TEXT
);

CREATE TABLE IF NOT EXISTS proxy_timeout_shadow (
    request_id        TEXT PRIMARY KEY,
    completed_at      REAL NOT NULL,
    endpoint          TEXT NOT NULL,
    priority          INTEGER NOT NULL,
    est_in            INTEGER,
    est_out           INTEGER,
    actual_out        INTEGER,
    actual_total_ms   REAL,
    applied_timeout_s REAL,
    recommended_ms    REAL,
    p95_ms            REAL,
    median_ms         REAL,
    min_ms            REAL,
    source            TEXT,
    would_timeout     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS proxy_timeouts (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id        TEXT NOT NULL,
    occurred_at       REAL NOT NULL,
    endpoint          TEXT NOT NULL,
    priority          INTEGER NOT NULL,
    agent_id          TEXT,
    call_site         TEXT,
    layer             TEXT NOT NULL,
    elapsed_s         REAL,
    applied_timeout_s REAL,
    queue_wait_ms     REAL,
    in_flight         INTEGER,
    queued            INTEGER,
    max_slots         INTEGER,
    est_in            INTEGER,
    est_out           INTEGER,
    recommended_ms    REAL,
    under_recommended INTEGER,
    session_id        TEXT,
    turn_id           TEXT,
    caller_id         TEXT,
    context_window    INTEGER,
    context_used_pct  REAL
);

CREATE INDEX IF NOT EXISTS idx_pq_status ON proxy_queue(status);
CREATE INDEX IF NOT EXISTS idx_pc_completed ON proxy_completions(completed_at);
CREATE INDEX IF NOT EXISTS idx_pts_completed ON proxy_timeout_shadow(completed_at);
CREATE INDEX IF NOT EXISTS idx_pto_occurred ON proxy_timeouts(occurred_at);
"""


class PersistentQueue:
    """WAL-backed persistence for the scheduler queue.

    The in-memory ``Scheduler`` is the source of truth during normal
    operation.  This class writes to SQLite on enqueue/dispatch/expire
    so that on restart, pending requests can be recovered.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path) if db_path else ""
        self._conn: sqlite3.Connection | None = None
        self._read_conn: sqlite3.Connection | None = None
        self._write_q: "_queue.Queue | None" = None
        self._writer: threading.Thread | None = None
        # Phase 5B.3 writer-thread hardening:
        # _writer_started_once distinguishes the LEGITIMATE pre-writer sync-write
        # window (init / startup recovery / tests — single-threaded, safe) from
        # the BUG case where the writer was started and later DIED (a sync write
        # on the loop thread then violates the single-writer invariant + blocks
        # the event loop). A bounded queue prevents unbounded memory growth; a
        # watchdog restart + CRITICAL surface replaces the old silent fallback.
        self._writer_started_once = False
        self._writer_restarts = 0
        self._write_q_dropped = 0
        self._write_q_maxsize = 20000
        if self._db_path:
            self._conn = self._open(self._db_path)

    def _open(self, path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        # Incremental auto-vacuum MUST be set before journal_mode=WAL writes the
        # db header — on a fresh DB it then takes effect on the first CREATE; on a
        # legacy auto_vacuum=NONE DB it stays INERT until the one-time startup
        # VACUUM commits the mode change (see vacuum_full_blocking). Freed pages
        # (deletes + payload NULLs) are then returned to the OS via the poller's
        # incremental_vacuum.
        conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=2500")
        conn.executescript(_SCHEMA)
        self._migrate_completions(conn)
        self._migrate_timeouts(conn)
        self._migrate_queue(conn)
        return conn

    @staticmethod
    def _add_missing_columns(
        conn: sqlite3.Connection, table: str, columns: dict[str, str],
    ) -> None:
        """Idempotently ALTER-ADD any columns missing from ``table``."""
        existing = {r[1] for r in conn.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()}
        for name, decl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

    @classmethod
    def _migrate_completions(cls, conn: sqlite3.Connection) -> None:
        cls._add_missing_columns(conn, "proxy_completions", {
            "payload_json": "TEXT",
            "response_json": "TEXT",
            "session_id": "TEXT",
            "turn_id": "TEXT",
            "caller_id": "TEXT",
            "finish_reason": "TEXT",
        })

    @classmethod
    def _migrate_timeouts(cls, conn: sqlite3.Connection) -> None:
        cls._add_missing_columns(conn, "proxy_timeouts", {
            "session_id": "TEXT",
            "turn_id": "TEXT",
            "caller_id": "TEXT",
            "context_window": "INTEGER",
            "context_used_pct": "REAL",
        })

    @classmethod
    def _migrate_queue(cls, conn: sqlite3.Connection) -> None:
        cls._add_missing_columns(conn, "proxy_queue", {
            "enqueued_wall": "REAL",
            "timeout_s": "REAL",
            "caller_id": "TEXT",
        })

    # ----- async writer (Phase 2.2): writes run on a dedicated thread that owns
    # the write connection, so synchronous SQLite I/O never blocks the event
    # loop that schedules the whole fleet. Reads use a separate read-only
    # connection on the loop thread (WAL permits concurrent readers). Until the
    # writer is started (tests / startup recovery), writes execute synchronously
    # on the write connection — identical behaviour, no thread.

    def start_async_writer(self) -> None:
        """Open the read-only connection and start the writer thread. Called by
        the service AFTER startup recovery (which runs synchronously on the write
        connection while single-threaded)."""
        if not self._conn or self._writer is not None:
            return
        try:
            # A second connection for reads, used ONLY by the event-loop thread
            # (SELECTs only — never writes). NOT opened mode=ro: a read-only
            # connection on a WAL database can fail to read because it can't
            # build the -shm index. WAL permits multiple connections; the writer
            # thread owns self._conn, the loop owns self._read_conn — neither is
            # shared across threads, so no locking is needed.
            self._read_conn = sqlite3.connect(
                self._db_path, isolation_level=None, check_same_thread=False)
            self._read_conn.execute("PRAGMA busy_timeout=2500")
        except Exception as exc:  # noqa: BLE001
            logger.warning("read connection open failed (%s); reads use writer conn", exc)
            self._read_conn = None
        # Bounded (Phase 5B.3): an unbounded queue grows without limit if the
        # writer falls behind; 20k is far above any real burst (the writer keeps
        # the queue near-empty under normal load), so hitting the cap means
        # something is wrong — drop + surface rather than grow memory unboundedly.
        self._write_q = _queue.Queue(maxsize=self._write_q_maxsize)
        self._writer_started_once = True
        self._spawn_writer_thread()

    def _spawn_writer_thread(self) -> None:
        self._writer = threading.Thread(
            target=self._writer_loop, name="llmproxy-dbwriter", daemon=True)
        self._writer.start()

    def _writer_loop(self) -> None:
        while True:
            item = self._write_q.get()
            try:
                if item is None:  # shutdown sentinel
                    return
                if callable(item):
                    # Maintenance op that must run on the writer connection
                    # (wal_checkpoint/incremental_vacuum return rows _w can't
                    # carry). Same single-writer invariant as every other write.
                    item(self._conn)
                else:
                    sql, params = item
                    self._conn.execute(sql, params)
            except Exception as exc:  # noqa: BLE001 — never let a bad write kill the writer
                logger.warning("llmproxy db write failed: %s", exc)
            finally:
                self._write_q.task_done()

    def _w(self, sql: str, params: tuple = ()) -> None:
        """Submit a write.

        Three cases (Phase 5B.3):
          - writer alive  → enqueue (non-blocking; drop+count if the bounded
            queue is full, so we NEVER block the event loop on DB I/O);
          - pre-writer    → run synchronously on the write connection. LEGITIMATE
            and single-threaded (init / startup recovery / tests);
          - writer DIED   → watchdog: restart it once; only if that fails do we
            fall back to a sync write — LOUDLY (CRITICAL), never silently, since
            a loop-thread write violates the single-writer invariant.
        """
        if not self._conn:
            return
        if self._writer is not None and self._writer.is_alive():
            try:
                self._write_q.put_nowait((sql, params))
            except _queue.Full:
                self._write_q_dropped += 1
                if self._write_q_dropped % 100 == 1:
                    logger.error(
                        "llmproxy write queue full (maxsize=%d) — dropped %d "
                        "best-effort write(s); recovery/telemetry rows lost, "
                        "live serving unaffected",
                        self._write_q_maxsize, self._write_q_dropped,
                    )
            return
        if not self._writer_started_once:
            # Pre-writer window — single-threaded, safe.
            self._conn.execute(sql, params)
            return
        # Writer was started and DIED — try to self-heal once.
        logger.critical(
            "llmproxy db writer thread is DEAD — attempting restart (restarts=%d)",
            self._writer_restarts,
        )
        self._writer_restarts += 1
        try:
            self._spawn_writer_thread()
        except Exception as exc:  # noqa: BLE001
            logger.critical("llmproxy db writer restart FAILED: %s", exc)
        if self._writer is not None and self._writer.is_alive():
            try:
                self._write_q.put_nowait((sql, params))
                return
            except _queue.Full:
                self._write_q_dropped += 1
                return
        # Restart failed — degraded sync fallback (loud). Better than losing the
        # write entirely; the CRITICAL above pages the operator via log_scan.
        logger.critical(
            "llmproxy db writer unavailable — degraded SYNC write on loop thread")
        try:
            self._conn.execute(sql, params)
        except Exception as exc:  # noqa: BLE001
            logger.error("llmproxy degraded sync write failed: %s", exc)

    def _submit_writer_call(self, fn) -> None:
        """Run ``fn(conn)`` on the writer thread (or synchronously pre-writer).

        For maintenance PRAGMAs (wal_checkpoint, incremental_vacuum) that return
        rows and so can't go through ``_w`` (which only execute()s and discards
        them). Same three-case routing + single-writer invariant as ``_w``.
        """
        if not self._conn:
            return
        if self._writer is not None and self._writer.is_alive():
            try:
                self._write_q.put_nowait(fn)
            except _queue.Full:
                self._write_q_dropped += 1
            return
        if not self._writer_started_once:
            # Pre-writer window — single-threaded, safe.
            fn(self._conn)
            return
        # Writer started and DIED — best-effort maintenance runs synchronously
        # and loudly (rare; _w's self-heal path covers normal writes).
        logger.critical(
            "llmproxy db writer dead — running maintenance op synchronously")
        try:
            fn(self._conn)
        except Exception as exc:  # noqa: BLE001
            logger.error("llmproxy maintenance op failed: %s", exc)

    def writer_alive(self) -> bool:
        """True if the async writer thread is running (or not yet started)."""
        if not self._writer_started_once:
            return True  # pre-writer single-threaded mode is healthy
        return self._writer is not None and self._writer.is_alive()

    def writer_restarts(self) -> int:
        return self._writer_restarts

    def write_q_dropped(self) -> int:
        return self._write_q_dropped

    def _reader(self) -> "sqlite3.Connection | None":
        """Connection for read queries: the read-only conn on the loop thread,
        falling back to the write conn (pre-writer / if RO open failed)."""
        return self._read_conn or self._conn

    def wal_size_bytes(self) -> int:
        """Size of the WAL sidecar in bytes (0 if absent) — for the wal_growth
        alert (Phase 2.5)."""
        if not self._db_path:
            return 0
        try:
            import os as _os
            return _os.path.getsize(self._db_path + "-wal")
        except OSError:
            return 0

    def checkpoint_truncate(self) -> None:
        """TRUNCATE-checkpoint the WAL so the -wal sidecar returns to ~0 instead
        of camping at a burst high-water mark. The PRAGMA returns
        (busy, log, checkpointed) — consumed via fetchall(); a non-zero busy just
        means a reader held it this tick, it truncates on the next."""
        self._submit_writer_call(
            lambda c: c.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall())

    def incremental_vacuum(self, pages: int) -> None:
        """Return up to ``pages`` freelist pages to the OS. No-op unless
        auto_vacuum=INCREMENTAL is committed and free pages exist."""
        n = int(pages)
        self._submit_writer_call(
            lambda c: c.execute(f"PRAGMA incremental_vacuum({n})").fetchall())

    def freelist_bytes(self) -> int:
        """Dead (free) space in the main DB file, in bytes (read path)."""
        if not self._conn:
            return 0
        try:
            r = self._reader()
            fl = r.execute("PRAGMA freelist_count").fetchone()[0]
            ps = r.execute("PRAGMA page_size").fetchone()[0]
            return int(fl) * int(ps)
        except Exception:  # noqa: BLE001
            return 0

    def vacuum_full_blocking(self, threshold_bytes: int) -> int:
        """One-time reclaim: full VACUUM + WAL truncate when the freelist exceeds
        ``threshold_bytes``. MUST run pre-writer (single-threaded) — VACUUM takes
        an exclusive lock and rewrites the whole file, so it can never touch the
        live loop. Also commits a pending auto_vacuum=INCREMENTAL mode change.
        Returns bytes reclaimed (0 if skipped)."""
        if not self._conn:
            return 0
        if self._writer is not None:
            raise RuntimeError(
                "vacuum_full_blocking must run pre-writer (single-threaded)")
        ps = self._conn.execute("PRAGMA page_size").fetchone()[0]
        free_before = self._conn.execute(
            "PRAGMA freelist_count").fetchone()[0] * ps
        if free_before < threshold_bytes:
            return 0
        pages_before = self._conn.execute("PRAGMA page_count").fetchone()[0]
        self._conn.execute("VACUUM")
        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
        pages_after = self._conn.execute("PRAGMA page_count").fetchone()[0]
        return max(0, (pages_before - pages_after) * ps)

    def flush(self, timeout: float = 10.0) -> None:
        """Block until all queued writes have been applied (drain support)."""
        if self._write_q is None:
            return
        try:
            # queue.Queue has no join-with-timeout; poll unfinished_tasks.
            import time as _t
            deadline = _t.monotonic() + timeout
            while self._write_q.unfinished_tasks and _t.monotonic() < deadline:
                _t.sleep(0.02)
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        if self._writer is not None and self._writer.is_alive():
            self.flush(timeout=5.0)
            self._write_q.put(None)  # shutdown sentinel
            self._writer.join(timeout=5.0)
            self._writer = None
        if self._read_conn:
            try:
                self._read_conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._read_conn = None
        if self._conn:
            self._conn.close()
            self._conn = None

    # ----- write operations -----

    def persist_enqueue(self, req: QueuedRequest) -> None:
        self._w(
            "INSERT OR REPLACE INTO proxy_queue "
            "(request_id, agent_id, endpoint, priority, call_site, "
            " payload_type, payload_json, status, enqueued_at, "
            " timeout_deadline, estimated_cost, session_id, turn_id, stream, "
            " enqueued_wall, timeout_s, caller_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                req.request_id, req.agent_id, req.endpoint,
                int(req.priority), req.call_site,
                req.payload_type, json.dumps(req.payload, separators=(",", ":")),
                "queued", req.enqueued_at, req.timeout_deadline,
                req.estimated_cost_ss, req.session_id, req.turn_id,
                1 if req.stream else 0,
                time.time(), req.timeout_s, req.caller_id,
            ),
        )

    def persist_dispatch(self, request_id: str) -> None:
        self._w(
            "UPDATE proxy_queue SET status='dispatched' WHERE request_id=?",
            (request_id,),
        )

    def persist_complete(
        self,
        request_id: str,
        agent_id: str,
        endpoint: str,
        call_site: str,
        priority: int,
        input_tokens: int,
        output_tokens: int,
        duration_s: float,
        queue_wait_ms: float,
        status: str,
        payload: dict | None = None,
        response: dict | None = None,
        session_id: str | None = None,
        turn_id: str | None = None,
        caller_id: str | None = None,
        finish_reason: str | None = None,
    ) -> None:
        if not self._conn:
            return
        now = time.time()
        self._w(
            "DELETE FROM proxy_queue WHERE request_id=?",
            (request_id,),
        )
        payload_s = json.dumps(payload, separators=(",", ":")) if payload else None
        response_s = json.dumps(response, separators=(",", ":")) if response else None
        self._w(
            "INSERT OR REPLACE INTO proxy_completions "
            "(request_id, agent_id, endpoint, call_site, priority, "
            " input_tokens, output_tokens, duration_s, queue_wait_ms, "
            " status, completed_at, payload_json, response_json, "
            " session_id, turn_id, caller_id, finish_reason) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                request_id, agent_id, endpoint, call_site, priority,
                input_tokens, output_tokens, duration_s, queue_wait_ms,
                status, now, payload_s, response_s,
                session_id, turn_id, caller_id, finish_reason,
            ),
        )

    def persist_timeout_shadow(
        self,
        *,
        request_id: str,
        endpoint: str,
        priority: int,
        est_in: int,
        est_out: int,
        actual_out: int,
        actual_total_ms: float,
        applied_timeout_s: float,
        recommended_ms: float,
        p95_ms: float,
        median_ms: float,
        min_ms: float,
        source: str,
        would_timeout: bool,
    ) -> None:
        """Record what the timeout-advice model *would* have recommended
        for a completed request, alongside the actual latency and the
        timeout actually applied.  Observational only — never on the
        caller's critical path."""
        self._w(
            "INSERT OR REPLACE INTO proxy_timeout_shadow "
            "(request_id, completed_at, endpoint, priority, est_in, est_out, "
            " actual_out, actual_total_ms, applied_timeout_s, recommended_ms, "
            " p95_ms, median_ms, min_ms, source, would_timeout) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                request_id, time.time(), endpoint, int(priority),
                est_in, est_out, actual_out, actual_total_ms, applied_timeout_s,
                recommended_ms, p95_ms, median_ms, min_ms, source,
                1 if would_timeout else 0,
            ),
        )

    def persist_timeout_event(
        self,
        *,
        request_id: str,
        endpoint: str,
        priority: int,
        agent_id: str,
        call_site: str,
        layer: str,
        elapsed_s: float,
        applied_timeout_s: float,
        queue_wait_ms: float | None,
        in_flight: int,
        queued: int,
        max_slots: int,
        est_in: int,
        est_out: int,
        recommended_ms: float,
        under_recommended: bool,
        session_id: str | None = None,
        turn_id: str | None = None,
        caller_id: str | None = None,
        context_window: int = 0,
        context_used_pct: float | None = None,
    ) -> None:
        """Record a call that hit its timeout instead of finishing, with
        the load context at the moment it gave up. ``layer`` is one of
        ``admission`` (expired while queued), ``client_wait`` (the sync
        caller's deadline fired — work may still be in flight),
        ``backend`` (the model exceeded the deadline after dispatch), or
        ``stream``. ``under_recommended`` flags a timeout that fired below
        the data-driven recommended deadline (i.e. likely premature)."""
        self._w(
            "INSERT INTO proxy_timeouts "
            "(request_id, occurred_at, endpoint, priority, agent_id, call_site, "
            " layer, elapsed_s, applied_timeout_s, queue_wait_ms, in_flight, "
            " queued, max_slots, est_in, est_out, recommended_ms, under_recommended, "
            " session_id, turn_id, caller_id, context_window, context_used_pct) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                request_id, time.time(), endpoint, int(priority), agent_id, call_site,
                layer, elapsed_s, applied_timeout_s, queue_wait_ms, in_flight,
                queued, max_slots, est_in, est_out, recommended_ms,
                1 if under_recommended else 0,
                session_id, turn_id, caller_id, context_window, context_used_pct,
            ),
        )

    def persist_expire(self, request_id: str) -> None:
        self._w(
            "DELETE FROM proxy_queue WHERE request_id=?",
            (request_id,),
        )

    def persist_cancel(self, request_id: str) -> None:
        self.persist_expire(request_id)

    # ----- budget persistence -----

    def save_budgets(self, budgets: list[dict]) -> None:
        for b in budgets:
            self._w(
                "INSERT OR REPLACE INTO proxy_agent_budgets "
                "(agent_id, weight, balance, total_consumed, last_replenish_at) "
                "VALUES (?,?,?,?,?)",
                (
                    b["agent_id"], b.get("weight", 1.0),
                    b.get("balance_ss", 0.0), b.get("total_consumed_ss", 0.0),
                    time.time(),
                ),
            )

    def load_budgets(self) -> list[dict]:
        """Restore persisted DRR balances on startup (Phase 3.4) so fairness
        survives a restart instead of resetting to zero."""
        if not self._conn:
            return []
        rows = self._reader().execute(
            "SELECT agent_id, weight, balance, total_consumed "
            "FROM proxy_agent_budgets"
        ).fetchall()
        return [
            {"agent_id": r[0], "weight": r[1] or 1.0,
             "balance": r[2] or 0.0, "total_consumed": r[3] or 0.0}
            for r in rows
        ]

    # ----- recovery -----

    def recover_queued(self, now: float) -> list[QueuedRequest]:
        """Reload pending requests from WAL.  Discard any past their
        deadline.  In-flight requests at crash time are abandoned."""
        if not self._conn:
            return []

        # Clean up stale dispatched entries (in-flight at crash time)
        self._conn.execute("DELETE FROM proxy_queue WHERE status='dispatched'")

        rows = self._reader().execute(
            "SELECT request_id, agent_id, endpoint, priority, call_site, "
            "       payload_type, payload_json, enqueued_at, timeout_deadline, "
            "       estimated_cost, session_id, turn_id, stream, "
            "       enqueued_wall, timeout_s, caller_id "
            "FROM proxy_queue WHERE status='queued'"
        ).fetchall()

        recovered: list[QueuedRequest] = []
        discarded = 0
        wall_now = time.time()

        for row in rows:
            (rid, aid, ep, pri, cs, pt, pj, ea, td, ec, sid, tid, st,
             ew, ts_budget, cid) = row

            # Phase 2.6: enqueued_at/timeout_deadline are time.monotonic() from a
            # DEAD process — meaningless against this process's clock. Rebase via
            # wall-clock: how much of the budget remains after the elapsed wall
            # time, then re-anchor onto the new monotonic `now`. Rows written
            # before this migration (no enqueued_wall) are re-anchored fresh
            # rather than risk a bogus monotonic comparison mis-expiring them.
            budget = ts_budget if ts_budget else 180.0
            if ew:
                elapsed = max(0.0, wall_now - ew)
                remaining = budget - elapsed
                if remaining <= 0:
                    self._conn.execute(
                        "DELETE FROM proxy_queue WHERE request_id=?", (rid,))
                    discarded += 1
                    continue
                new_enqueued_at = now - elapsed
                new_deadline = now + remaining
            else:
                new_enqueued_at = now
                new_deadline = now + budget

            try:
                payload = json.loads(pj)
            except (json.JSONDecodeError, TypeError):
                payload = {}

            req = QueuedRequest(
                request_id=rid,
                agent_id=aid,
                endpoint=normalize_endpoint(ep),
                priority=LLMPriority(pri),
                band=priority_to_band(LLMPriority(pri)),
                call_site=cs,
                payload_type=pt,
                payload=payload,
                timeout_deadline=new_deadline,
                enqueued_at=new_enqueued_at,
                estimated_cost_ss=ec or 0.0,
                session_id=sid,
                turn_id=tid,
                caller_id=cid,
                timeout_s=budget,
                stream=bool(st),
            )
            recovered.append(req)

        if recovered or discarded:
            logger.info(
                "queue recovery: %d recovered, %d discarded (expired)",
                len(recovered), discarded,
            )

        return recovered

    def cleanup_old_completions(self, max_age_s: float = 86400 * 7) -> None:
        """Trim completions / timeout-shadow / timeouts past the retention
        window. Enqueued to the writer thread (off the event loop)."""
        if not self._conn:
            return
        cutoff = time.time() - max_age_s
        self._w("DELETE FROM proxy_completions WHERE completed_at < ?", (cutoff,))
        self._w("DELETE FROM proxy_timeout_shadow WHERE completed_at < ?", (cutoff,))
        self._w("DELETE FROM proxy_timeouts WHERE occurred_at < ?", (cutoff,))

    def cleanup_old_payloads(self, max_age_s: float) -> None:
        """NULL out payload/response bodies older than the window while KEEPING
        the completion metadata row (7-day retention is separate). The only
        readers of these blobs use recent rows (health-sweep last 100-300,
        replay/AB --hours 4). The IS NOT NULL guard keeps re-runs cheap. Freed
        pages go to the freelist → returned to the OS by incremental_vacuum."""
        if not self._conn:
            return
        cutoff = time.time() - max_age_s
        self._w(
            "UPDATE proxy_completions SET payload_json=NULL, response_json=NULL "
            "WHERE completed_at < ? AND "
            "(payload_json IS NOT NULL OR response_json IS NOT NULL)",
            (cutoff,),
        )

    def history_buckets(
        self, hours: float = 4.0, bucket_minutes: int = 5,
    ) -> list[dict]:
        """Return time-bucketed aggregates from proxy_completions."""
        if not self._conn:
            return []
        cutoff = time.time() - (hours * 3600)
        bucket_s = bucket_minutes * 60
        rows = self._reader().execute(
            "SELECT "
            "  CAST((completed_at - ?) / ? AS INTEGER) AS bucket_idx, "
            "  endpoint, agent_id, "
            "  COUNT(*) AS req, "
            "  SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END) AS ok, "
            "  SUM(CASE WHEN status!='ok' THEN 1 ELSE 0 END) AS errors, "
            "  AVG(duration_s) AS avg_dur, "
            "  SUM(input_tokens) AS total_in, "
            "  SUM(output_tokens) AS total_out, "
            "  SUM(duration_s) AS slot_seconds "
            "FROM proxy_completions "
            "WHERE completed_at >= ? "
            "GROUP BY bucket_idx, endpoint, agent_id "
            "ORDER BY bucket_idx",
            (cutoff, bucket_s, cutoff),
        ).fetchall()

        from datetime import datetime, timezone
        buckets_map: dict[int, dict] = {}
        for row in rows:
            idx, ep, aid, req, ok_count, err_count, avg_dur, t_in, t_out, ss = row
            if idx not in buckets_map:
                start = cutoff + idx * bucket_s
                buckets_map[idx] = {
                    "start": datetime.fromtimestamp(start, tz=timezone.utc).isoformat(),
                    "end": datetime.fromtimestamp(start + bucket_s, tz=timezone.utc).isoformat(),
                    "per_endpoint": {},
                    "per_agent": {},
                }
            b = buckets_map[idx]
            if ep not in b["per_endpoint"]:
                b["per_endpoint"][ep] = {
                    "requests": 0, "ok": 0, "errors": 0,
                    "avg_duration_s": 0, "total_in_tokens": 0,
                    "total_out_tokens": 0,
                }
            epm = b["per_endpoint"][ep]
            epm["requests"] += req
            epm["ok"] += ok_count
            epm["errors"] += err_count
            epm["avg_duration_s"] = round((avg_dur or 0), 2)
            epm["total_in_tokens"] += t_in or 0
            epm["total_out_tokens"] += t_out or 0

            if aid not in b["per_agent"]:
                b["per_agent"][aid] = {"requests": 0, "slot_seconds": 0}
            b["per_agent"][aid]["requests"] += req
            b["per_agent"][aid]["slot_seconds"] = round(
                b["per_agent"][aid]["slot_seconds"] + (ss or 0), 1,
            )

        return [buckets_map[k] for k in sorted(buckets_map)]

    def recent_requests(self, limit: int = 50) -> list[dict]:
        """Return the most recent completed requests for the feed."""
        if not self._conn:
            return []
        rows = self._reader().execute(
            "SELECT request_id, agent_id, endpoint, call_site, priority, "
            "       input_tokens, output_tokens, duration_s, queue_wait_ms, "
            "       status, completed_at "
            "FROM proxy_completions ORDER BY completed_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        from datetime import datetime, timezone
        return [
            {
                "request_id": r[0], "agent_id": r[1], "endpoint": r[2],
                "call_site": r[3], "priority": r[4], "input_tokens": r[5],
                "output_tokens": r[6], "duration_s": round(r[7] or 0, 2),
                "queue_wait_ms": round(r[8] or 0, 1),
                "status": r[9],
                "completed_at": datetime.fromtimestamp(
                    r[10], tz=timezone.utc,
                ).isoformat() if r[10] else None,
            }
            for r in rows
        ]

    # ----- query (for simulation / observability) -----

    def recent_completions(self, hours: float = 4.0) -> list[dict]:
        """Return recent completions for simulation replay."""
        if not self._conn:
            return []
        cutoff = time.time() - (hours * 3600)
        rows = self._reader().execute(
            "SELECT request_id, agent_id, endpoint, call_site, priority, "
            "       input_tokens, output_tokens, duration_s, queue_wait_ms, "
            "       status, completed_at "
            "FROM proxy_completions WHERE completed_at >= ? "
            "ORDER BY completed_at",
            (cutoff,),
        ).fetchall()
        return [
            {
                "request_id": r[0], "agent_id": r[1], "endpoint": r[2],
                "call_site": r[3], "priority": r[4], "input_tokens": r[5],
                "output_tokens": r[6], "duration_s": r[7],
                "queue_wait_ms": r[8], "status": r[9], "completed_at": r[10],
            }
            for r in rows
        ]

    def timeout_samples(self, hours: float = 168.0) -> list[dict]:
        """Successful completions for bootstrapping the timeout model.

        End-to-end latency is reconstructed downstream as
        ``duration_s*1000 + queue_wait_ms`` (the persisted columns; the
        live feed path uses the truer enqueue-to-now span)."""
        if not self._conn:
            return []
        cutoff = time.time() - (hours * 3600)
        rows = self._reader().execute(
            "SELECT endpoint, priority, input_tokens, output_tokens, "
            "       duration_s, queue_wait_ms "
            "FROM proxy_completions "
            "WHERE completed_at >= ? AND status = 'ok' AND duration_s > 0 "
            "ORDER BY completed_at",
            (cutoff,),
        ).fetchall()
        return [
            {
                "endpoint": r[0], "priority": r[1],
                "input_tokens": r[2] or 0, "output_tokens": r[3] or 0,
                "duration_s": r[4] or 0.0, "queue_wait_ms": r[5] or 0.0,
            }
            for r in rows
        ]

    def timeout_shadow_report(self, hours: float = 24.0) -> list[dict]:
        """Per-(endpoint, tier) summary of the timeout-shadow log:
        how often ``recommended`` would have fired, and how much headroom
        it reclaims versus the timeout actually applied."""
        if not self._conn:
            return []
        from .timeout_model import percentile

        cutoff = time.time() - (hours * 3600)
        rows = self._reader().execute(
            "SELECT endpoint, priority, actual_total_ms, applied_timeout_s, "
            "       recommended_ms, would_timeout, source "
            "FROM proxy_timeout_shadow WHERE completed_at >= ?",
            (cutoff,),
        ).fetchall()

        groups: dict[tuple[str, int], dict] = {}
        for ep, pri, actual_ms, applied_s, rec_ms, wt, source in rows:
            g = groups.setdefault((ep, pri), {
                "actual": [], "recommended": [], "headroom": [],
                "would_timeout": 0, "sources": {},
            })
            g["actual"].append(actual_ms or 0.0)
            g["recommended"].append(rec_ms or 0.0)
            g["headroom"].append((applied_s or 0.0) * 1000.0 - (rec_ms or 0.0))
            g["would_timeout"] += int(wt or 0)
            g["sources"][source] = g["sources"].get(source, 0) + 1

        out: list[dict] = []
        for (ep, pri), g in groups.items():
            n = len(g["actual"])
            actual = sorted(g["actual"])
            rec = sorted(g["recommended"])
            head = sorted(g["headroom"])
            out.append({
                "endpoint": ep,
                "priority": pri,
                "samples": n,
                "would_timeout": g["would_timeout"],
                "would_timeout_rate": round(g["would_timeout"] / n, 4) if n else 0.0,
                "recommended_ms_p50": round(percentile(rec, 50), 1),
                "recommended_ms_p95": round(percentile(rec, 95), 1),
                "actual_total_ms_p50": round(percentile(actual, 50), 1),
                "actual_total_ms_p95": round(percentile(actual, 95), 1),
                "headroom_vs_applied_ms_p50": round(percentile(head, 50), 1),
                "headroom_vs_applied_ms_p95": round(percentile(head, 95), 1),
                "sources": g["sources"],
            })
        out.sort(key=lambda r: (r["endpoint"], r["priority"]))
        return out

    def timeouts_report(self, hours: float = 24.0) -> dict:
        """Per-(endpoint, tier, layer) summary of timeout events: how
        often calls give up instead of finishing, the load when they do,
        and how many fired below the data-driven recommended deadline
        (premature). Also returns a grand total."""
        if not self._conn:
            return {"total": 0, "premature": 0, "rows": []}
        from .timeout_model import percentile

        cutoff = time.time() - (hours * 3600)
        rows = self._reader().execute(
            "SELECT endpoint, priority, layer, elapsed_s, in_flight, queued, "
            "       under_recommended, recommended_ms, caller_id, context_used_pct "
            "FROM proxy_timeouts WHERE occurred_at >= ?",
            (cutoff,),
        ).fetchall()

        groups: dict[tuple[str, int, str], dict] = {}
        total = 0
        premature = 0
        for ep, pri, layer, elapsed, in_flight, queued, under, rec_ms, caller, ctx_pct in rows:
            total += 1
            premature += int(under or 0)
            g = groups.setdefault((ep, pri, layer), {
                "elapsed": [], "in_flight": [], "queued": [],
                "premature": 0, "recommended": [], "callers": {}, "ctx_pct": [],
            })
            g["elapsed"].append(elapsed or 0.0)
            g["in_flight"].append(in_flight or 0)
            g["queued"].append(queued or 0)
            g["premature"] += int(under or 0)
            g["recommended"].append(rec_ms or 0.0)
            if caller:
                g["callers"][caller] = g["callers"].get(caller, 0) + 1
            if ctx_pct is not None:
                g["ctx_pct"].append(ctx_pct)

        out: list[dict] = []
        for (ep, pri, layer), g in groups.items():
            n = len(g["elapsed"])
            elapsed = sorted(g["elapsed"])
            top_callers = dict(sorted(
                g["callers"].items(), key=lambda kv: -kv[1],
            )[:3])
            out.append({
                "endpoint": ep,
                "priority": pri,
                "layer": layer,
                "count": n,
                "premature": g["premature"],
                "elapsed_s_p50": round(percentile(elapsed, 50), 2),
                "elapsed_s_p95": round(percentile(elapsed, 95), 2),
                "avg_in_flight": round(sum(g["in_flight"]) / n, 1) if n else 0.0,
                "avg_queued": round(sum(g["queued"]) / n, 1) if n else 0.0,
                "recommended_ms_p50": round(percentile(sorted(g["recommended"]), 50), 1),
                "avg_context_used_pct": (
                    round(sum(g["ctx_pct"]) / len(g["ctx_pct"]), 1)
                    if g["ctx_pct"] else None
                ),
                "top_callers": top_callers,
            })
        out.sort(key=lambda r: (-r["count"], r["endpoint"], r["priority"], r["layer"]))
        return {"total": total, "premature": premature, "rows": out}

    def completions_for_calibration(self, hours: float = 24.0) -> list[dict]:
        """Return successful completions with non-zero token counts
        for cost model bootstrap on startup."""
        if not self._conn:
            return []
        cutoff = time.time() - (hours * 3600)
        rows = self._reader().execute(
            "SELECT endpoint, call_site, input_tokens, output_tokens, duration_s "
            "FROM proxy_completions "
            "WHERE completed_at >= ? AND status = 'ok' "
            "  AND output_tokens > 0 AND duration_s > 0 "
            "ORDER BY completed_at",
            (cutoff,),
        ).fetchall()
        return [
            {
                "endpoint": r[0], "call_site": r[1],
                "input_tokens": r[2], "output_tokens": r[3],
                "duration_s": r[4],
            }
            for r in rows
        ]

    def export_corpus(
        self,
        hours: float = 24.0,
        endpoint: str | None = None,
        call_site: str | None = None,
    ) -> list[dict]:
        """Export replay-ready corpus records with full payloads.

        Returns only successful completions that have payload_json
        recorded. Used by the test harness for recorded-replay and
        A/B backend comparison modes.
        """
        if not self._conn:
            return []
        cutoff = time.time() - (hours * 3600)
        where = ["completed_at >= ?", "status = 'ok'", "payload_json IS NOT NULL"]
        params: list = [cutoff]
        if endpoint:
            where.append("endpoint = ?")
            params.append(endpoint)
        if call_site:
            where.append("call_site LIKE ?")
            params.append(call_site.replace("*", "%"))
        rows = self._reader().execute(
            "SELECT request_id, agent_id, endpoint, call_site, priority, "
            "       input_tokens, output_tokens, duration_s, queue_wait_ms, "
            "       completed_at, payload_json, response_json "
            "FROM proxy_completions "
            f"WHERE {' AND '.join(where)} "
            "ORDER BY completed_at",
            params,
        ).fetchall()
        return [
            {
                "request_id": r[0], "agent_id": r[1], "endpoint": r[2],
                "call_site": r[3], "priority": r[4],
                "input_tokens": r[5], "output_tokens": r[6],
                "duration_s": r[7], "queue_wait_ms": r[8],
                "completed_at": r[9],
                "payload": json.loads(r[10]) if r[10] else None,
                "response": json.loads(r[11]) if r[11] else None,
            }
            for r in rows
        ]
