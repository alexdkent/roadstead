"""Per-caller, per-window backend-stall lookup (`stall_aborts` + its route).

`/v1/timeouts` aggregates by (endpoint, tier, layer) and answers "is the fleet
under pressure?". It cannot answer the question a single consumer has — "while
MY run was alive, did the backend stall underneath it?" — because the grouping
has already discarded which caller and which second.

That question is load-bearing: a pool job killed by an upstream stall and a
genuine agent-side hang produce the identical terminal detail, and only this
lookup separates them. So these tests pin the three ways it could quietly
answer the WRONG question and still return a plausible number: the wrong
caller's stall, a stall from outside the window, and an abort that was OUR
capacity decision rather than the backend dying.

No network, no clock dependence — every timestamp is written explicitly.
"""

from __future__ import annotations

import json

import pytest

from originfleet.llmproxy.config import ProxyConfig
from originfleet.llmproxy.service import ProxyService


class _FakeRequest:
    def __init__(self, **params):
        self.query_params = {k: str(v) for k, v in params.items()}


def _svc(tmp_path) -> ProxyService:
    return ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db")))


def _event(svc, *, request_id, occurred_at, caller_id, abort_reason,
           endpoint="thinker", layer="stream"):
    """Write one proxy_timeouts row with an EXPLICIT occurred_at.

    `persist_timeout_event` stamps `time.time()` itself, which would make every
    window assertion here depend on the wall clock. The row is written through
    the real insert first (so the column set stays honest), then its timestamp
    is pinned.
    """
    svc._queue_db.persist_timeout_event(
        request_id=request_id, endpoint=endpoint, priority=3, agent_id="pool_broker",
        call_site="pool_broker.dispatch", layer=layer, elapsed_s=31.0,
        applied_timeout_s=300.0, queue_wait_ms=0.0, in_flight=1, queued=0,
        max_slots=4, est_in=9000, est_out=800, recommended_ms=180000.0,
        under_recommended=False, caller_id=caller_id, abort_reason=abort_reason,
    )
    svc._queue_db._conn.execute(
        "UPDATE proxy_timeouts SET occurred_at = ? WHERE request_id = ?",
        (occurred_at, request_id),
    )


# ----- what it must FIND -------------------------------------------------

def test_returns_the_matching_stall(tmp_path):
    svc = _svc(tmp_path)
    _event(svc, request_id="s1", occurred_at=1000.0,
           caller_id="pool-analyst", abort_reason="stall")

    rows = svc._queue_db.stall_aborts("pool-analyst", 900.0, 1100.0)

    assert len(rows) == 1
    assert rows[0]["request_id"] == "s1"
    assert rows[0]["abort_reason"] == "stall"
    assert rows[0]["caller_id"] == "pool-analyst"
    assert rows[0]["endpoint"] == "thinker"


def test_prefix_matching_is_deliberate(tmp_path):
    """Matching is a PREFIX, not equality.

    Measured against the live queue.db 2026-08-16, the stored `caller_id` for
    pool work is the bare instance name (`pool-analyst` / `pool-observer` /
    `pool-effector`) with no suffix — equality would work today. The prefix is
    there so that a caller id which later grows a per-run or per-call-site
    suffix (as `call_site` already has: `pool-analyst.openai_compat`) does not
    silently stop matching and turn every stall into "no evidence" — a
    regression that would be invisible, because "no stall" is the answer this
    lookup gives when it cannot tell.
    """
    svc = _svc(tmp_path)
    _event(svc, request_id="bare", occurred_at=1000.0,
           caller_id="pool-analyst", abort_reason="stall")
    _event(svc, request_id="suffixed", occurred_at=1001.0,
           caller_id="pool-analyst.openai_compat", abort_reason="stall")

    rows = svc._queue_db.stall_aborts("pool-analyst", 900.0, 1100.0)
    assert [r["request_id"] for r in rows] == ["bare", "suffixed"]


def test_ttft_counts_as_a_backend_stall(tmp_path):
    """Never emitting a first token is the same fault caught earlier."""
    svc = _svc(tmp_path)
    _event(svc, request_id="s1", occurred_at=1000.0,
           caller_id="pool-analyst", abort_reason="ttft")

    rows = svc._queue_db.stall_aborts("pool-analyst", 900.0, 1100.0)
    assert [r["abort_reason"] for r in rows] == ["ttft"]


# ----- what it must NOT find ---------------------------------------------

@pytest.mark.parametrize("reason", ["hard_cap", "caller_deadline"])
def test_our_own_capacity_aborts_are_not_backend_stalls(tmp_path, reason):
    """`hard_cap`/`caller_deadline` are US cutting off healthy work.

    Counting them would let a deliberate capacity decision masquerade as the
    substrate failing — the exact inversion this lookup exists to prevent.
    """
    svc = _svc(tmp_path)
    _event(svc, request_id="s1", occurred_at=1000.0,
           caller_id="pool-analyst", abort_reason=reason)

    assert svc._queue_db.stall_aborts("pool-analyst", 900.0, 1100.0) == []


def test_a_null_abort_reason_is_not_a_stall(tmp_path):
    svc = _svc(tmp_path)
    _event(svc, request_id="s1", occurred_at=1000.0,
           caller_id="pool-analyst", abort_reason=None, layer="admission")

    assert svc._queue_db.stall_aborts("pool-analyst", 900.0, 1100.0) == []


def test_another_callers_stall_is_not_evidence(tmp_path):
    svc = _svc(tmp_path)
    _event(svc, request_id="s1", occurred_at=1000.0,
           caller_id="discord.initiative", abort_reason="stall")
    _event(svc, request_id="s2", occurred_at=1000.0,
           caller_id="pool-observer", abort_reason="stall")

    rows = svc._queue_db.stall_aborts("pool-analyst", 900.0, 1100.0)
    assert rows == []


def test_null_caller_id_never_matches(tmp_path):
    svc = _svc(tmp_path)
    _event(svc, request_id="s1", occurred_at=1000.0,
           caller_id=None, abort_reason="stall")

    assert svc._queue_db.stall_aborts("pool-analyst", 900.0, 1100.0) == []


def test_empty_prefix_matches_nothing_rather_than_everything(tmp_path):
    """An empty caller must not hand back the whole fleet's stalls."""
    svc = _svc(tmp_path)
    _event(svc, request_id="s1", occurred_at=1000.0,
           caller_id="discord.initiative", abort_reason="stall")

    assert svc._queue_db.stall_aborts("", 0.0, 1e12) == []


def test_underscore_is_not_a_wildcard(tmp_path):
    """`_` is a LIKE single-char wildcard; unescaped it over-matches."""
    svc = _svc(tmp_path)
    _event(svc, request_id="s1", occurred_at=1000.0,
           caller_id="poolXbroker.dispatch", abort_reason="stall")

    assert svc._queue_db.stall_aborts("pool_broker", 900.0, 1100.0) == []


def test_empty_when_nothing_matches_at_all(tmp_path):
    svc = _svc(tmp_path)
    assert svc._queue_db.stall_aborts("pool-analyst", 0.0, 1e12) == []


# ----- the window boundaries ---------------------------------------------

def test_window_excludes_before_and_after(tmp_path):
    svc = _svc(tmp_path)
    _event(svc, request_id="before", occurred_at=899.0,
           caller_id="pool-analyst", abort_reason="stall")
    _event(svc, request_id="inside", occurred_at=1000.0,
           caller_id="pool-analyst", abort_reason="stall")
    _event(svc, request_id="after", occurred_at=1101.0,
           caller_id="pool-analyst", abort_reason="stall")

    rows = svc._queue_db.stall_aborts("pool-analyst", 900.0, 1100.0)
    assert [r["request_id"] for r in rows] == ["inside"]


def test_window_is_inclusive_on_both_ends(tmp_path):
    """A stall recorded on the exact terminalisation second still counts.

    The proxy writes the row when it kills the stream — which is the same
    instant the consumer downstream observes as its own failure. An exclusive
    bound would drop precisely the row that matters most.
    """
    svc = _svc(tmp_path)
    _event(svc, request_id="lo", occurred_at=900.0,
           caller_id="pool-analyst", abort_reason="stall")
    _event(svc, request_id="hi", occurred_at=1100.0,
           caller_id="pool-analyst", abort_reason="stall")

    rows = svc._queue_db.stall_aborts("pool-analyst", 900.0, 1100.0)
    assert [r["request_id"] for r in rows] == ["lo", "hi"]


def test_rows_are_ordered_by_time(tmp_path):
    svc = _svc(tmp_path)
    _event(svc, request_id="late", occurred_at=1050.0,
           caller_id="pool-analyst", abort_reason="stall")
    _event(svc, request_id="early", occurred_at=950.0,
           caller_id="pool-analyst", abort_reason="stall")

    rows = svc._queue_db.stall_aborts("pool-analyst", 900.0, 1100.0)
    assert [r["request_id"] for r in rows] == ["early", "late"]


# ----- the route ----------------------------------------------------------

@pytest.mark.asyncio
async def test_route_returns_the_rows_not_merely_a_200(tmp_path):
    svc = _svc(tmp_path)
    _event(svc, request_id="s1", occurred_at=1000.0,
           caller_id="pool-analyst:job-abc", abort_reason="stall")
    _event(svc, request_id="other", occurred_at=1000.0,
           caller_id="discord.initiative", abort_reason="stall")

    resp = await svc.handle_stall_aborts(
        _FakeRequest(caller="pool-analyst", since=900.0, until=1100.0))
    body = json.loads(resp.body.decode())

    assert resp.status_code == 200
    assert body["caller"] == "pool-analyst"
    assert body["count"] == 1
    assert len(body["rows"]) == 1
    row = body["rows"][0]
    assert row["request_id"] == "s1"
    assert row["abort_reason"] == "stall"
    assert row["caller_id"] == "pool-analyst:job-abc"
    assert row["endpoint"] == "thinker"
    assert row["occurred_at"] == 1000.0


@pytest.mark.asyncio
async def test_route_is_registered_on_the_real_app(tmp_path):
    """A handler nobody can reach is a module that was written and never called."""
    from originfleet.llmproxy.routes import make_routes

    paths = {r.path for r in make_routes(_svc(tmp_path))}
    assert "/v1/timeouts/stalls" in paths


@pytest.mark.asyncio
async def test_route_reports_no_evidence_rather_than_erroring(tmp_path):
    """An unknown caller / empty window is `count: 0`, never a 4xx.

    The consumer treats "no stall" and "cannot tell" identically — falling
    through to today's behaviour — so an error here would buy nothing and
    could only break a best-effort call site.
    """
    svc = _svc(tmp_path)

    resp = await svc.handle_stall_aborts(_FakeRequest())
    body = json.loads(resp.body.decode())
    assert resp.status_code == 200
    assert body["count"] == 0
    assert body["rows"] == []
