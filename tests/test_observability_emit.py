"""RequestLogger: the JSONL is authoritative; only failures hit the text log.

Per the persistence-cleanup change, ``RequestLogger.log()`` writes every record
to the JSONL but only re-emits genuine failures (error/timeout/truncated)
through the Python logger — ok/cancelled are silent there (no more triplication
into llmproxy.log + docker logs).
"""

from __future__ import annotations

import logging

from originfleet.llmproxy.observability import RequestLogger, RequestLogRecord


def _rec(status: str) -> RequestLogRecord:
    return RequestLogRecord(
        ts="t", request_id="r", agent_id="a", endpoint="chat", call_site="s",
        priority="P3_INGESTION", band="background", payload_type="chat_completion",
        input_tokens=1, output_tokens=1, estimated_cost_ss=0.0, actual_cost_ss=0.0,
        queue_wait_ms=0.0, backend_latency_ms=0.0, total_latency_ms=0.0,
        occupancy_at_dispatch=0, status=status)


def test_ok_is_silent_failures_warn(tmp_path, caplog):
    rl = RequestLogger(str(tmp_path / "req.jsonl"))
    with caplog.at_level(logging.WARNING, logger="originfleet.llmproxy.observability"):
        rl.log(_rec("ok"))
        rl.log(_rec("cancelled"))   # benign client disconnect — silent in text log
        rl.log(_rec("error"))
        rl.log(_rec("truncated"))
    emitted = [r for r in caplog.records if r.getMessage().startswith("req:")]
    # Only the two genuine failures surfaced, at WARNING.
    assert len(emitted) == 2
    assert all(r.levelno == logging.WARNING for r in emitted)
    # Every record was written to the JSONL regardless of status (authoritative).
    lines = (tmp_path / "req.jsonl").read_text().strip().splitlines()
    assert len(lines) == 4
    rl.close()
