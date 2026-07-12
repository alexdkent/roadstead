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
from typing import Any

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
    -- Phase 2a prefix-cache attribution: prompt tokens the backend served from
    -- its KV prefix cache (vLLM usage.prompt_tokens_details.cached_tokens).
    -- NULL when the backend doesn't report it (llama.cpp) — distinct from 0
    -- (a real cold miss) so the per-caller rollup marks it n/a, not 0% hit.
    cached_tokens    INTEGER,
    duration_s       REAL,
    queue_wait_ms    REAL,
    status           TEXT NOT NULL,
    completed_at     REAL NOT NULL,
    payload_json     TEXT,
    response_json    TEXT,
    session_id       TEXT,
    turn_id          TEXT,
    caller_id        TEXT,
    -- 'llm' for native proxy traffic (chat/embed/rerank/vision), or the
    -- non-LLM service class ('audio'/'imagegen'/'ocr'/'translate') for
    -- calls pushed in via /v1/calls/log. NULL on pre-migration rows.
    kind             TEXT
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

-- Operator maintenance windows: deliberate backend restarts/drains. Timeout
-- events that fall inside a window are tagged ``planned`` by timeouts_report,
-- so an intentional restart reads as PLANNED instead of looking like an
-- incident. Written by the pause/resume drain control (source='drain') and the
-- manual /v1/admin/maintenance annotate API (source='manual'). ``endpoint`` is
-- an endpoint class or '*' (all endpoints); ``ended_at`` NULL = still open.
CREATE TABLE IF NOT EXISTS proxy_maintenance (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint    TEXT NOT NULL,
    started_at  REAL NOT NULL,
    ended_at    REAL,
    reason      TEXT,
    operator    TEXT,
    source      TEXT
);

CREATE TABLE IF NOT EXISTS proxy_cache_stats (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_at   REAL NOT NULL,        -- wall-clock of the periodic tick
    endpoint      TEXT NOT NULL,        -- endpoint_class (creative/thinker/chat/...)
    cum_hits      INTEGER,              -- vLLM prefix_cache_hits_total (cumulative; NULL=n/a)
    cum_queries   INTEGER,              -- vLLM prefix_cache_queries_total (cumulative; NULL=n/a)
    screen_json   TEXT                  -- JSON: per-call_site cache-ability rows for this endpoint
);

-- Durable shadow/flip-gate evidence (audit 2026-07-02): the in-memory
-- context-overflow tally + cache-drift dedup map reset on every ship-driven
-- proxy restart (2 boots/day is normal), so a "review the shadow window,
-- then flip" decision could never accumulate its evidence. Aggregates only —
-- one row per (endpoint, caller) / one KV row — not an event log.
CREATE TABLE IF NOT EXISTS proxy_context_overflows (
    endpoint   TEXT NOT NULL,
    caller     TEXT NOT NULL,
    count      INTEGER NOT NULL DEFAULT 0,
    max_est_in INTEGER NOT NULL DEFAULT 0,
    last_at    REAL NOT NULL,
    PRIMARY KEY (endpoint, caller)
);

CREATE TABLE IF NOT EXISTS proxy_kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL       -- JSON blob
);

CREATE INDEX IF NOT EXISTS idx_pq_status ON proxy_queue(status);
CREATE INDEX IF NOT EXISTS idx_pc_completed ON proxy_completions(completed_at);
CREATE INDEX IF NOT EXISTS idx_pts_completed ON proxy_timeout_shadow(completed_at);
CREATE INDEX IF NOT EXISTS idx_pto_occurred ON proxy_timeouts(occurred_at);
CREATE INDEX IF NOT EXISTS idx_pm_endpoint_started ON proxy_maintenance(endpoint, started_at);
CREATE INDEX IF NOT EXISTS idx_pcs_endpoint_snap ON proxy_cache_stats(endpoint, snapshot_at);
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
        # Off-loop dashboard reads (§5c): heavy GROUP-BY aggregations over the
        # whole-fleet completions table run via asyncio.to_thread so they never
        # block the event loop that schedules all fleet LLM traffic. A sqlite
        # connection can't be shared across threads, so each pool thread gets
        # its OWN read connection here (WAL permits concurrent readers). The
        # loop thread keeps using _read_conn; the single-WRITER-thread invariant
        # is untouched — these are read-only.
        self._tlocal = threading.local()
        self._loop_thread: threading.Thread | None = None
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
            "kind": "TEXT",
            "cached_tokens": "INTEGER",  # Phase 2a prefix-cache attribution
        })
        # Index supporting the fleet-usage rollups (group by endpoint over a
        # recent window) now that the table holds whole-fleet call metrics.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pc_endpoint_completed "
            "ON proxy_completions(endpoint, completed_at)")

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
        # Record the loop thread so _reader() can hand off-loop (to_thread)
        # dashboard reads their own per-thread connection (§5c).
        self._loop_thread = threading.current_thread()
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
        """Connection for read queries.

        Loop thread (and the pre-writer single-threaded window): the dedicated
        read-only conn, falling back to the write conn. Off-loop pool threads
        (dashboard aggregations dispatched via ``asyncio.to_thread``, §5c): a
        per-thread connection — a sqlite connection can't be shared across
        threads, and WAL permits concurrent readers. Read-only, so the
        single-WRITER-thread invariant is preserved."""
        rconn = self._read_conn or self._conn
        lt = self._loop_thread
        if lt is None or threading.current_thread() is lt:
            return rconn
        conn = getattr(self._tlocal, "conn", None)
        if conn is None and self._db_path:
            conn = sqlite3.connect(
                self._db_path, isolation_level=None, check_same_thread=False)
            conn.execute("PRAGMA busy_timeout=2500")
            self._tlocal.conn = conn
        return conn or rconn

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
        # freelist_count only sees whole free pages. cleanup_old_payloads NULLs
        # payload bodies IN PLACE, leaving intra-page fragmentation that never
        # reaches the freelist — the file camped at a ~930MB high-water mark
        # (~59% dead) while the freelist read 32MB, so this gate could never
        # fire (audit 2026-07-02). Add dbstat's per-page unused bytes when the
        # module is available (startup-only full scan, pre-writer — acceptable).
        try:
            unused = self._conn.execute(
                "SELECT COALESCE(SUM(unused), 0) FROM dbstat").fetchone()[0]
            free_before += int(unused or 0)
        except sqlite3.Error:
            pass  # dbstat not compiled in — fall back to freelist-only
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
        kind: str | None = "llm",
        cached_tokens: int | None = None,
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
            " input_tokens, output_tokens, cached_tokens, duration_s, queue_wait_ms, "
            " status, completed_at, payload_json, response_json, "
            " session_id, turn_id, caller_id, finish_reason, kind) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                request_id, agent_id, endpoint, call_site, priority,
                input_tokens, output_tokens, cached_tokens, duration_s, queue_wait_ms,
                status, now, payload_s, response_s,
                session_id, turn_id, caller_id, finish_reason, kind,
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

    # ----- maintenance windows (planned-restart annotation) -----

    @staticmethod
    def _norm_maint_endpoint(endpoint: str) -> str:
        """'*' (all endpoints) passes through; everything else normalizes to its
        QoS endpoint class so a window keyed by a role ('llama-thinker') matches
        the class ('thinker') that timeout events are recorded under."""
        return "*" if str(endpoint) == "*" else normalize_endpoint(endpoint)

    def maintenance_open(
        self, *, endpoint: str, reason: str = "", operator: str = "",
        source: str = "manual", started_at: float | None = None,
    ) -> None:
        """Open an (unbounded) maintenance window for ``endpoint`` (a class, or
        '*' for all). Timeout events inside the window are tagged ``planned`` in
        timeouts_report. Stays open (ended_at NULL) until ``maintenance_close``."""
        self._w(
            "INSERT INTO proxy_maintenance "
            "(endpoint, started_at, ended_at, reason, operator, source) "
            "VALUES (?,?,?,?,?,?)",
            (self._norm_maint_endpoint(endpoint),
             started_at if started_at is not None else time.time(),
             None, reason or None, operator or None, source),
        )

    def maintenance_close(
        self, *, endpoint: str, ended_at: float | None = None,
    ) -> None:
        """Close any open maintenance window(s) for ``endpoint`` (or '*')."""
        self._w(
            "UPDATE proxy_maintenance SET ended_at=? "
            "WHERE endpoint=? AND ended_at IS NULL",
            (ended_at if ended_at is not None else time.time(),
             self._norm_maint_endpoint(endpoint)),
        )

    def maintenance_record(
        self, *, endpoint: str, started_at: float, ended_at: float,
        reason: str = "", operator: str = "", source: str = "manual",
    ) -> None:
        """Record a CLOSED window — e.g. a backdated annotation of a restart
        already performed outside the drain path."""
        self._w(
            "INSERT INTO proxy_maintenance "
            "(endpoint, started_at, ended_at, reason, operator, source) "
            "VALUES (?,?,?,?,?,?)",
            (self._norm_maint_endpoint(endpoint), started_at, ended_at,
             reason or None, operator or None, source),
        )

    def maintenance_windows(self, hours: float = 24.0) -> list[dict]:
        """Maintenance windows overlapping the last ``hours``. An open window
        (ended_at NULL) is treated as extending to now."""
        if not self._conn:
            return []
        now = time.time()
        cutoff = now - (hours * 3600)
        rows = self._reader().execute(
            "SELECT id, endpoint, started_at, ended_at, reason, operator, source "
            "FROM proxy_maintenance WHERE COALESCE(ended_at, ?) >= ? "
            "ORDER BY started_at",
            (now, cutoff),
        ).fetchall()
        return [
            {
                "id": r[0], "endpoint": r[1], "started_at": r[2],
                "ended_at": r[3], "reason": r[4], "operator": r[5],
                "source": r[6],
            }
            for r in rows
        ]

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

    # ----- durable shadow/flip-gate evidence (audit 2026-07-02) -----

    def record_context_overflow(self, endpoint: str, caller: str,
                                est_in: int) -> None:
        """Upsert one (endpoint, caller) overflow aggregate. Rare events
        (a few/day) — one writer-queue op each, no hot-path cost."""
        self._w(
            "INSERT INTO proxy_context_overflows "
            "(endpoint, caller, count, max_est_in, last_at) VALUES (?,?,1,?,?) "
            "ON CONFLICT(endpoint, caller) DO UPDATE SET "
            "count = count + 1, "
            "max_est_in = MAX(max_est_in, excluded.max_est_in), "
            "last_at = excluded.last_at",
            (endpoint, caller, int(est_in), time.time()),
        )

    def load_context_overflows(self) -> dict[str, dict]:
        """Rebuild the /v1/status ``context_overflows_shadow`` shape from the
        durable aggregates (startup seed)."""
        if not self._conn:
            return {}
        rows = self._reader().execute(
            "SELECT endpoint, caller, count, max_est_in "
            "FROM proxy_context_overflows").fetchall()
        out: dict[str, dict] = {}
        for ep, caller, count, max_in in rows:
            tally = out.setdefault(ep, {"count": 0, "callers": {}, "max_est_in": 0})
            tally["count"] += int(count or 0)
            tally["callers"][caller] = int(count or 0)
            tally["max_est_in"] = max(tally["max_est_in"], int(max_in or 0))
        return out

    def kv_set(self, key: str, value: Any) -> None:
        """Persist a small JSON blob (e.g. the cache-drift dedup map)."""
        self._w(
            "INSERT INTO proxy_kv (key, value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )

    def kv_get(self, key: str, default: Any = None) -> Any:
        if not self._conn:
            return default
        row = self._reader().execute(
            "SELECT value FROM proxy_kv WHERE key = ?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return default

    def delete_budgets(self, agent_ids: list[str]) -> None:
        """Remove pruned one-off agent_ids from the persisted budget table.
        Without this, ``INSERT OR REPLACE`` never deletes and ``load_budgets``
        resurrects every historical id on each boot — 78 immortal ghosts were
        diluting real agents' replenish rates (audit 2026-07-02)."""
        for aid in agent_ids:
            self._w("DELETE FROM proxy_agent_budgets WHERE agent_id = ?", (aid,))

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

        # Drop queued STREAMING requests outright: their SSE consumer died
        # with the old process, so a recovered dispatch has nobody to relay
        # to — _execute_streaming would early-return without recording a
        # completion and the scheduler slot leaked PERMANENTLY (audit
        # 2026-06-10). Sync recovery stays: it completes + records, and a
        # temp-0 result still populates the cache.
        dropped_streams = self._conn.execute(
            "SELECT COUNT(*) FROM proxy_queue WHERE status='queued' AND stream=1"
        ).fetchone()[0]
        if dropped_streams:
            self._conn.execute(
                "DELETE FROM proxy_queue WHERE status='queued' AND stream=1")
            logger.info(
                "queue recovery: dropped %d queued streaming request(s) "
                "(consumer died with the old process)", dropped_streams)

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

    # ----- prefix-cache observability (snapshots + screen payloads) -----
    def cache_screen_payloads(
        self, endpoint: str, hours: float = 168.0, limit: int = 2000,
    ) -> list[tuple[str, str]]:
        """(call_site, prompt-prefix) rows for one endpoint over the window, most
        recent first. payload_json is truncated to 16 KB — bounds memory and is
        ample for the LCP/Jaccard screen (avglen truncates the same, so aligned
        big prompts still read as high-LCP)."""
        if not self._conn:
            return []
        cutoff = time.time() - hours * 3600.0
        rows = self._reader().execute(
            "SELECT call_site, substr(payload_json,1,16000) "
            "FROM proxy_completions "
            "WHERE endpoint = ? AND payload_json IS NOT NULL AND completed_at > ? "
            "ORDER BY completed_at DESC LIMIT ?",
            (endpoint, cutoff, limit),
        ).fetchall()
        return [(r[0], r[1] or "") for r in rows]

    def persist_cache_snapshot(
        self, *, snapshot_at: float, endpoint: str,
        cum_hits: int | None, cum_queries: int | None, screen_json: str,
    ) -> None:
        """Record one periodic prefix-cache snapshot for an endpoint."""
        self._w(
            "INSERT INTO proxy_cache_stats "
            "(snapshot_at, endpoint, cum_hits, cum_queries, screen_json) "
            "VALUES (?,?,?,?,?)",
            (snapshot_at, endpoint, cum_hits, cum_queries, screen_json),
        )

    def prune_cache_stats(self, older_than_s: float = 30 * 86400.0) -> None:
        self._w("DELETE FROM proxy_cache_stats WHERE snapshot_at < ?",
                (time.time() - older_than_s,))

    def cache_stats_snapshots(self, window_s: float = 7 * 86400.0) -> list[dict]:
        """All snapshots within the window, oldest→newest (drives latest +
        windowed-rate-delta + trend + drift in the handler)."""
        if not self._conn:
            return []
        cutoff = time.time() - window_s
        rows = self._reader().execute(
            "SELECT snapshot_at, endpoint, cum_hits, cum_queries, screen_json "
            "FROM proxy_cache_stats WHERE snapshot_at > ? ORDER BY snapshot_at ASC",
            (cutoff,),
        ).fetchall()
        out = []
        for r in rows:
            try:
                screen = json.loads(r[4]) if r[4] else []
            except Exception:
                screen = []
            out.append({
                "snapshot_at": r[0], "endpoint": r[1],
                "cum_hits": r[2], "cum_queries": r[3], "screen": screen,
            })
        return out

    # ----- non-LLM ingest (Phase 1: proxy = fleet call-metrics authority) -----

    def persist_external_call(
        self,
        *,
        request_id: str,
        agent_id: str,
        endpoint: str,
        call_site: str,
        kind: str,
        input_tokens: int,
        output_tokens: int,
        duration_s: float,
        status: str,
        priority: int = int(LLMPriority.P2_POST_TURN),
        caller_id: str | None = None,
    ) -> None:
        """Record a non-LLM service call (audio/imagegen/ocr/translate) that
        never traversed the scheduler. Lands in proxy_completions tagged by
        ``kind`` so the fleet rollups treat it as just another completion.
        No payload/response bodies (no replay value)."""
        if not self._conn:
            return
        self._w(
            "INSERT OR REPLACE INTO proxy_completions "
            "(request_id, agent_id, endpoint, call_site, priority, "
            " input_tokens, output_tokens, duration_s, queue_wait_ms, "
            " status, completed_at, caller_id, kind) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                request_id, agent_id, normalize_endpoint(endpoint), call_site,
                int(priority), int(input_tokens), int(output_tokens),
                float(duration_s), 0.0, status, time.time(), caller_id, kind,
            ),
        )

    # ----- fleet usage rollups (Phase 1) -----

    def fleet_activity(self, window_s: int = 86400, bin_s: int = 600) -> dict:
        """Fleet-wide binned call activity for the top-of-page strip:
        per-bin {n, fails, tokens_in, tokens_out, p95}, plus a per-endpoint
        breakdown over the last hour. The USAGE half of the host daemon's
        ``fleet_activity`` (host hardware series stays on the daemon)."""
        if not self._conn:
            return {"window_s": window_s, "bin_s": bin_s, "calls": [], "by_endpoint_1h": []}
        from .timeout_model import percentile
        now = time.time()
        since = now - window_s
        rows = self._reader().execute(
            "SELECT CAST(completed_at / ? AS INTEGER) * ? AS bucket_ts, "
            "       COUNT(*) AS n, "
            "       SUM(CASE WHEN status!='ok' THEN 1 ELSE 0 END) AS fails, "
            "       SUM(COALESCE(input_tokens,0)) AS tin, "
            "       SUM(COALESCE(output_tokens,0)) AS tout, "
            "       GROUP_CONCAT(CAST(duration_s*1000 AS INTEGER)) AS lats "
            "FROM proxy_completions WHERE completed_at >= ? "
            "GROUP BY bucket_ts ORDER BY bucket_ts ASC",
            (bin_s, bin_s, since),
        ).fetchall()
        calls = []
        for bucket_ts, n, fails, tin, tout, lats in rows:
            lat_list = sorted(int(x) for x in (lats or "").split(",") if x)
            calls.append({
                "ts": int(bucket_ts), "n": int(n), "fails": int(fails or 0),
                "tokens_in": int(tin or 0), "tokens_out": int(tout or 0),
                "p95": round(percentile(lat_list, 95), 1) if lat_list else 0.0,
            })
        cutoff_1h = now - 3600
        ep_rows = self._reader().execute(
            "SELECT endpoint, COUNT(*) AS n, "
            "       SUM(CASE WHEN status!='ok' THEN 1 ELSE 0 END) AS fails, "
            "       GROUP_CONCAT(CAST(duration_s*1000 AS INTEGER)) AS lats "
            "FROM proxy_completions WHERE completed_at >= ? "
            "GROUP BY endpoint ORDER BY n DESC LIMIT 16",
            (cutoff_1h,),
        ).fetchall()
        by_endpoint = []
        for ep, n, fails, lats in ep_rows:
            lat_list = sorted(int(x) for x in (lats or "").split(",") if x)
            by_endpoint.append({
                "endpoint": ep, "n": int(n), "fails": int(fails or 0),
                "p95": round(percentile(lat_list, 95), 1) if lat_list else 0.0,
            })
        return {
            "window_s": window_s, "bin_s": bin_s, "now": now,
            "calls": calls, "by_endpoint_1h": by_endpoint,
        }

    def endpoint_series(
        self, endpoint: str, window_s: int = 86400, bin_s: int = 600,
    ) -> dict:
        """Per-endpoint binned call stats for the detail-modal charts
        (the USAGE half of the daemon's ``unit_series``): per bin
        {n, fail_pct, tokens_in, tokens_out, p50, p95, p99, avg_in_toks}."""
        if not self._conn:
            return {"endpoint": endpoint, "window_s": window_s, "bin_s": bin_s, "calls_series": []}
        from .timeout_model import percentile
        ep = normalize_endpoint(endpoint)
        now = time.time()
        since = now - window_s
        rows = self._reader().execute(
            "SELECT CAST(completed_at / ? AS INTEGER) * ? AS bucket_ts, "
            "       COUNT(*) AS n, "
            "       SUM(CASE WHEN status!='ok' THEN 1 ELSE 0 END) AS fails, "
            "       SUM(COALESCE(input_tokens,0)) AS tin, "
            "       SUM(COALESCE(output_tokens,0)) AS tout, "
            "       AVG(input_tokens) AS avg_in, "
            "       GROUP_CONCAT(CAST(duration_s*1000 AS INTEGER)) AS lats "
            "FROM proxy_completions "
            "WHERE completed_at >= ? AND endpoint = ? "
            "GROUP BY bucket_ts ORDER BY bucket_ts ASC",
            (bin_s, bin_s, since, ep),
        ).fetchall()
        series = []
        for bucket_ts, n, fails, tin, tout, avg_in, lats in rows:
            lat_list = sorted(int(x) for x in (lats or "").split(",") if x)
            series.append({
                "ts": int(bucket_ts), "n": int(n),
                "fail_pct": round(100.0 * int(fails or 0) / int(n), 1) if n else 0.0,
                "tokens_in": int(tin or 0), "tokens_out": int(tout or 0),
                "p50": round(percentile(lat_list, 50), 1) if lat_list else None,
                "p95": round(percentile(lat_list, 95), 1) if lat_list else None,
                "p99": round(percentile(lat_list, 99), 1) if lat_list else None,
                "avg_in_toks": int(round(avg_in)) if avg_in is not None else None,
            })
        return {"endpoint": ep, "window_s": window_s, "bin_s": bin_s,
                "now": now, "calls_series": series}

    def cache_attribution(self, window_s: int = 3600, limit: int = 40) -> dict:
        """Phase 2a — per-caller prefix-cache attribution (the Tier-2 the
        observability doc specced but never built).

        Hit rate = ``sum(cached_tokens) / sum(input_tokens)`` computed ONLY over
        rows where the backend actually reported ``cached_tokens`` (vLLM). Rows
        where it is NULL (llama.cpp exposes no such counter) are EXCLUDED from
        the ratio and surfaced separately as ``unattributed_calls`` — so a
        non-reporting backend can never masquerade as a 0% hit rate and drag the
        number down (the exact ~4.8% attribution artifact this fixes).

        Returns per-(call_site,endpoint) rows ranked by volume, a per-endpoint
        rollup, and a fleet total. ``hit_rate`` is ``None`` (n/a) whenever no
        call in the group carried an attributable count."""
        empty = {"window_s": window_s, "now": None, "by_call_site": [],
                 "by_endpoint": [], "fleet": None}
        if not self._conn:
            return empty
        now = time.time()
        since = now - window_s
        # attributable_in only counts input_tokens on rows that reported cache
        # data, so the ratio's denominator matches its numerator's population.
        _attr_in = ("SUM(CASE WHEN cached_tokens IS NOT NULL "
                    "THEN COALESCE(input_tokens,0) ELSE 0 END)")
        select_tail = (
            "COUNT(*) AS calls, "
            "COUNT(cached_tokens) AS attributed_calls, "
            "COALESCE(SUM(cached_tokens),0) AS cached_in, "
            f"{_attr_in} AS attributable_in ")

        def _row(cols: tuple, label_keys: list[str]) -> dict:
            *labels, calls, attributed, cached_in, attributable_in = cols
            attributable_in = int(attributable_in or 0)
            cached_in = int(cached_in or 0)
            out = {k: v for k, v in zip(label_keys, labels)}
            out.update({
                "calls": int(calls or 0),
                "attributed_calls": int(attributed or 0),
                "unattributed_calls": int(calls or 0) - int(attributed or 0),
                "cached_tokens": cached_in,
                "attributable_input_tokens": attributable_in,
                "hit_rate": (round(cached_in / attributable_in, 4)
                             if attributable_in > 0 else None),
            })
            return out

        # kind='chat' is the prefix-cacheable LLM class — cached_tokens only ever
        # rides chat usage (vLLM); embed/rerank + non-LLM external calls have no
        # prefix cache and are correctly excluded.
        where = "FROM proxy_completions WHERE completed_at >= ? AND kind = 'chat' "
        cs_rows = self._reader().execute(
            "SELECT call_site, endpoint, " + select_tail + where +
            "GROUP BY call_site, endpoint ORDER BY calls DESC LIMIT ?",
            (since, limit),
        ).fetchall()
        ep_rows = self._reader().execute(
            "SELECT endpoint, " + select_tail + where +
            "GROUP BY endpoint ORDER BY calls DESC",
            (since,),
        ).fetchall()
        fleet_row = self._reader().execute(
            "SELECT 'fleet', " + select_tail + where,
            (since,),
        ).fetchone()
        return {
            "window_s": window_s, "now": now,
            "by_call_site": [_row(r, ["call_site", "endpoint"]) for r in cs_rows],
            "by_endpoint": [_row(r, ["endpoint"]) for r in ep_rows],
            "fleet": _row(fleet_row, ["scope"]) if fleet_row else None,
        }

    def top_callers(self, window_s: int = 3600, per_endpoint: int = 5) -> dict:
        """Top calling agents per endpoint over a recent window:
        ``{providers: {endpoint: [{agent, n, tokens_in, tokens_out}]}}``."""
        if not self._conn:
            return {"window_s": window_s, "providers": {}}
        now = time.time()
        since = now - window_s
        rows = self._reader().execute(
            "SELECT endpoint, agent_id, COUNT(*) AS n, "
            "       SUM(COALESCE(input_tokens,0)) AS tin, "
            "       SUM(COALESCE(output_tokens,0)) AS tout "
            "FROM proxy_completions WHERE completed_at >= ? "
            "GROUP BY endpoint, agent_id ORDER BY endpoint ASC, n DESC",
            (since,),
        ).fetchall()
        merged: dict[str, list[dict]] = {}
        for ep, agent, n, tin, tout in rows:
            merged.setdefault(ep, [])
            if len(merged[ep]) < per_endpoint:
                merged[ep].append({
                    "agent": agent or "—", "n": int(n),
                    "tokens_in": int(tin or 0), "tokens_out": int(tout or 0),
                })
        return {"window_s": window_s, "now": now, "providers": merged}

    def usage_rollup(self, dimension: str = "agent", hours: float = 24.0) -> list[dict]:
        """Per-(agent|call_site|endpoint) usage rollup over a window:
        requests, ok/errors, tokens, cloud-equivalent cost, p50/p95 latency.

        Cost is the cloud-equivalent we did NOT pay (rates keyed per endpoint),
        computed by grouping on (dim, endpoint) and summing up to the dim — so
        an agent that spans endpoints gets each endpoint's rate. NOTE: the
        proxy only sees LOCAL traffic, so this is savings, not Anthropic spend."""
        if not self._conn:
            return []
        from .timeout_model import percentile
        from .usage_rates import cloud_cost_usd
        col = {
            "agent": "agent_id", "call_site": "call_site",
            "endpoint": "endpoint", "provider": "endpoint",
        }.get(dimension, "agent_id")
        cutoff = time.time() - (hours * 3600)
        rows = self._reader().execute(
            f"SELECT {col} AS dim, endpoint, COUNT(*) AS n, "
            "       SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END) AS ok, "
            "       SUM(COALESCE(input_tokens,0)) AS tin, "
            "       SUM(COALESCE(output_tokens,0)) AS tout, "
            "       GROUP_CONCAT(CAST((duration_s*1000 + COALESCE(queue_wait_ms,0)) AS INTEGER)) AS lats "
            f"FROM proxy_completions WHERE completed_at >= ? GROUP BY {col}, endpoint",
            (cutoff,),
        ).fetchall()
        agg: dict[str, dict] = {}
        for dim, endpoint, n, ok, tin, tout, lats in rows:
            key = dim if dim is not None else "—"
            a = agg.setdefault(key, {
                "key": key, "requests": 0, "ok": 0, "errors": 0,
                "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "_lats": [],
            })
            a["requests"] += int(n)
            a["ok"] += int(ok or 0)
            a["errors"] += int(n) - int(ok or 0)
            a["tokens_in"] += int(tin or 0)
            a["tokens_out"] += int(tout or 0)
            a["cost_usd"] += cloud_cost_usd(endpoint or "", tin, tout)
            a["_lats"].extend(int(x) for x in (lats or "").split(",") if x)
        out: list[dict] = []
        for a in agg.values():
            lat_list = sorted(a.pop("_lats"))
            a["cost_usd"] = round(a["cost_usd"], 4)
            a["p50_ms"] = round(percentile(lat_list, 50), 1) if lat_list else 0.0
            a["p95_ms"] = round(percentile(lat_list, 95), 1) if lat_list else 0.0
            out.append(a)
        out.sort(key=lambda r: -r["requests"])
        return out

    def savings_summary(self, today_start: float | None = None) -> dict:
        """Cloud-equivalent cost avoided by running locally, within the
        completion-retention window. Returns today + total (all retained)
        USD and token totals, plus a per-endpoint breakdown. Ported from the
        host daemon's ``fleet_savings`` (now proxy-authoritative)."""
        if not self._conn:
            return {"today_usd": 0.0, "total_usd": 0.0, "by_endpoint": []}
        from .usage_rates import cloud_cost_usd
        if today_start is None:
            lt = time.localtime()
            today_start = time.mktime((
                lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
        rows = self._reader().execute(
            "SELECT endpoint, "
            "  SUM(CASE WHEN completed_at >= ? THEN COALESCE(input_tokens,0) ELSE 0 END) AS in_today, "
            "  SUM(CASE WHEN completed_at >= ? THEN COALESCE(output_tokens,0) ELSE 0 END) AS out_today, "
            "  SUM(COALESCE(input_tokens,0)) AS in_total, "
            "  SUM(COALESCE(output_tokens,0)) AS out_total "
            "FROM proxy_completions GROUP BY endpoint",
            (today_start, today_start),
        ).fetchall()
        today_usd = total_usd = 0.0
        today_in = today_out = total_in = total_out = 0
        by_endpoint: list[dict] = []
        for ep, in_today, out_today, in_total, out_total in rows:
            in_today, out_today = int(in_today or 0), int(out_today or 0)
            in_total, out_total = int(in_total or 0), int(out_total or 0)
            today_in += in_today; today_out += out_today
            total_in += in_total; total_out += out_total
            t_today = cloud_cost_usd(ep or "", in_today, out_today)
            t_total = cloud_cost_usd(ep or "", in_total, out_total)
            today_usd += t_today; total_usd += t_total
            by_endpoint.append({
                "endpoint": ep, "today_usd": round(t_today, 4),
                "total_usd": round(t_total, 4),
                "tokens_in": in_total, "tokens_out": out_total,
            })
        by_endpoint.sort(key=lambda x: -x["total_usd"])
        return {
            "today_usd": round(today_usd, 2), "total_usd": round(total_usd, 2),
            "today_tokens_in": today_in, "today_tokens_out": today_out,
            "total_tokens_in": total_in, "total_tokens_out": total_out,
            "today_start": int(today_start), "by_endpoint": by_endpoint,
        }

    # ----- query (for simulation / observability) -----

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

        # Survivorship-bias fix (2026-06-06): the shadow log only records
        # status==ok completions, so ``would_timeout`` is computed over
        # survivors and reads a misleading ~0 even while requests are actually
        # timing out. Join in the ACTUAL timeout counts (the censored samples)
        # from proxy_timeouts for the same window so the report is honest.
        actual_to: dict[tuple[str, int], int] = {}
        for ep, pri, cnt in self._reader().execute(
            "SELECT endpoint, priority, COUNT(*) FROM proxy_timeouts "
            "WHERE occurred_at >= ? GROUP BY endpoint, priority",
            (cutoff,),
        ).fetchall():
            actual_to[(ep, pri)] = cnt

        out: list[dict] = []
        for key in set(groups) | set(actual_to):
            ep, pri = key
            g = groups.get(key)
            n_to = actual_to.get(key, 0)
            n = len(g["actual"]) if g else 0
            # observed_timeout_rate counts the censored timeouts in the
            # denominator (completions + timeouts) — the true rate, vs the
            # survivor-only would_timeout_rate.
            denom = n + n_to
            row = {
                "endpoint": ep,
                "priority": pri,
                "samples": n,
                "actual_timeouts": n_to,
                "would_timeout": g["would_timeout"] if g else 0,
                "would_timeout_rate": round(g["would_timeout"] / n, 4) if n else 0.0,
                "observed_timeout_rate": round(n_to / denom, 4) if denom else 0.0,
            }
            if g:
                rec = sorted(g["recommended"])
                actual = sorted(g["actual"])
                head = sorted(g["headroom"])
                row.update({
                    "recommended_ms_p50": round(percentile(rec, 50), 1),
                    "recommended_ms_p95": round(percentile(rec, 95), 1),
                    "actual_total_ms_p50": round(percentile(actual, 50), 1),
                    "actual_total_ms_p95": round(percentile(actual, 95), 1),
                    "headroom_vs_applied_ms_p50": round(percentile(head, 50), 1),
                    "headroom_vs_applied_ms_p95": round(percentile(head, 95), 1),
                    "sources": g["sources"],
                })
            out.append(row)
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

        now = time.time()
        cutoff = now - (hours * 3600)
        rows = self._reader().execute(
            "SELECT endpoint, priority, layer, elapsed_s, in_flight, queued, "
            "       under_recommended, recommended_ms, caller_id, context_used_pct, "
            "       occurred_at "
            "FROM proxy_timeouts WHERE occurred_at >= ?",
            (cutoff,),
        ).fetchall()

        # Tag each event that falls inside an operator maintenance window
        # (deliberate restart/drain) so a planned burst doesn't read as an
        # incident. A window matches its own endpoint class or '*' (all).
        windows = self.maintenance_windows(hours)

        def _planned(ep: str, ts: float) -> bool:
            for w in windows:
                if w["endpoint"] not in (ep, "*"):
                    continue
                end = w["ended_at"] if w["ended_at"] is not None else now
                if w["started_at"] <= ts <= end:
                    return True
            return False

        groups: dict[tuple[str, int, str], dict] = {}
        total = 0
        premature = 0
        planned_total = 0
        for ep, pri, layer, elapsed, in_flight, queued, under, rec_ms, caller, ctx_pct, occurred in rows:
            total += 1
            premature += int(under or 0)
            is_planned = _planned(ep, occurred or 0.0)
            planned_total += int(is_planned)
            g = groups.setdefault((ep, pri, layer), {
                "elapsed": [], "in_flight": [], "queued": [],
                "premature": 0, "planned": 0, "recommended": [],
                "callers": {}, "ctx_pct": [],
            })
            g["elapsed"].append(elapsed or 0.0)
            g["in_flight"].append(in_flight or 0)
            g["queued"].append(queued or 0)
            g["premature"] += int(under or 0)
            g["planned"] += int(is_planned)
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
                "planned": g["planned"],
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
        return {
            "total": total,
            "premature": premature,
            "planned": planned_total,
            "rows": out,
            "maintenance_windows": windows,
        }

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
