"""Persistent queue with SQLite WAL backing.

In-memory for performance; WAL for crash recovery.  Queued requests
are persisted on enqueue and removed on dispatch or expiry.  In-flight
requests are NOT persisted — on crash, callers timeout and resubmit.
"""

from __future__ import annotations

import json
import logging
import sqlite3
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
    stream         INTEGER NOT NULL DEFAULT 0
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
    completed_at     REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pq_status ON proxy_queue(status);
CREATE INDEX IF NOT EXISTS idx_pc_completed ON proxy_completions(completed_at);
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
        if self._db_path:
            self._conn = self._open(self._db_path)

    def _open(self, path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=2500")
        conn.executescript(_SCHEMA)
        return conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ----- write operations -----

    def persist_enqueue(self, req: QueuedRequest) -> None:
        if not self._conn:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO proxy_queue "
            "(request_id, agent_id, endpoint, priority, call_site, "
            " payload_type, payload_json, status, enqueued_at, "
            " timeout_deadline, estimated_cost, session_id, turn_id, stream) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                req.request_id, req.agent_id, req.endpoint,
                int(req.priority), req.call_site,
                req.payload_type, json.dumps(req.payload, separators=(",", ":")),
                "queued", req.enqueued_at, req.timeout_deadline,
                req.estimated_cost_ss, req.session_id, req.turn_id,
                1 if req.stream else 0,
            ),
        )

    def persist_dispatch(self, request_id: str) -> None:
        if not self._conn:
            return
        self._conn.execute(
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
    ) -> None:
        if not self._conn:
            return
        now = time.time()
        self._conn.execute(
            "DELETE FROM proxy_queue WHERE request_id=?",
            (request_id,),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO proxy_completions "
            "(request_id, agent_id, endpoint, call_site, priority, "
            " input_tokens, output_tokens, duration_s, queue_wait_ms, "
            " status, completed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                request_id, agent_id, endpoint, call_site, priority,
                input_tokens, output_tokens, duration_s, queue_wait_ms,
                status, now,
            ),
        )

    def persist_expire(self, request_id: str) -> None:
        if not self._conn:
            return
        self._conn.execute(
            "DELETE FROM proxy_queue WHERE request_id=?",
            (request_id,),
        )

    def persist_cancel(self, request_id: str) -> None:
        self.persist_expire(request_id)

    # ----- budget persistence -----

    def save_budgets(self, budgets: list[dict]) -> None:
        if not self._conn:
            return
        for b in budgets:
            self._conn.execute(
                "INSERT OR REPLACE INTO proxy_agent_budgets "
                "(agent_id, weight, balance, total_consumed, last_replenish_at) "
                "VALUES (?,?,?,?,?)",
                (
                    b["agent_id"], b.get("weight", 1.0),
                    b.get("balance_ss", 0.0), b.get("total_consumed_ss", 0.0),
                    time.monotonic(),
                ),
            )

    # ----- recovery -----

    def recover_queued(self, now: float) -> list[QueuedRequest]:
        """Reload pending requests from WAL.  Discard any past their
        deadline.  In-flight requests at crash time are abandoned."""
        if not self._conn:
            return []

        # Clean up stale dispatched entries (in-flight at crash time)
        self._conn.execute("DELETE FROM proxy_queue WHERE status='dispatched'")

        rows = self._conn.execute(
            "SELECT request_id, agent_id, endpoint, priority, call_site, "
            "       payload_type, payload_json, enqueued_at, timeout_deadline, "
            "       estimated_cost, session_id, turn_id, stream "
            "FROM proxy_queue WHERE status='queued'"
        ).fetchall()

        recovered: list[QueuedRequest] = []
        discarded = 0

        for row in rows:
            (rid, aid, ep, pri, cs, pt, pj, ea, td, ec, sid, tid, st) = row
            if td <= now:
                self._conn.execute(
                    "DELETE FROM proxy_queue WHERE request_id=?", (rid,),
                )
                discarded += 1
                continue

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
                timeout_deadline=td,
                enqueued_at=ea,
                estimated_cost_ss=ec or 0.0,
                session_id=sid,
                turn_id=tid,
                stream=bool(st),
            )
            recovered.append(req)

        if recovered or discarded:
            logger.info(
                "queue recovery: %d recovered, %d discarded (expired)",
                len(recovered), discarded,
            )

        return recovered

    def cleanup_old_completions(self, max_age_s: float = 86400 * 7) -> int:
        """Remove completion records older than max_age_s."""
        if not self._conn:
            return 0
        cutoff = time.time() - max_age_s
        cursor = self._conn.execute(
            "DELETE FROM proxy_completions WHERE completed_at < ?",
            (cutoff,),
        )
        return cursor.rowcount

    # ----- query (for simulation / observability) -----

    def recent_completions(self, hours: float = 4.0) -> list[dict]:
        """Return recent completions for simulation replay."""
        if not self._conn:
            return []
        cutoff = time.time() - (hours * 3600)
        rows = self._conn.execute(
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
