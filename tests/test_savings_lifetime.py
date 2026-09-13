"""The savings TOTAL is a LIFETIME figure, and the prune must not eat it.

`savings_summary` used to report `total_usd` / `total_tokens_*` as a bare SUM
over `proxy_completions`, a table `cleanup_old_completions` trims at
`completions_retention_s` (30 days in production). So the "Saved total" the SPA
renders was a rolling 30-day window that plateaued once the window filled — it
was checked against production on 2026-09-12 and the live 554.31 total matched
the sum of the last 31 daily buckets, 554.32.

The fix is `proxy_savings_daily`: never pruned, tiny (endpoints x days), and
holding TOKENS rather than dollars so the whole figure stays priced at ONE set
of rates and can be recomputed when those rates change.

The risk it introduces is the reason for most of what is below. `_w` DROPS a
write when its bounded queue is full and is not transactional with the next
`_w`, so a dropped finalisation followed by a landed DELETE would destroy those
tokens with nothing left to rebuild them from. Two properties make that safe and
each has a test here: finalisation is monotone and idempotent, and the DELETE
refuses to touch a row whose bucket is not already in the rollup.
"""

from __future__ import annotations

import time

import pytest

from roadstead.queue import PersistentQueue

DAY = 86400.0
# 1M input tokens on tier1 == $0.04 (usage_rates: 0.04/M in), which makes every
# dollar figure below readable as "how many million tokens survived".
MTOK = 1_000_000


def _at(pq, rid, when, endpoint="tier1", in_tok=MTOK, out_tok=0):
    """Persist a completion and force its `completed_at` to `when`.

    persist_complete stamps wall-clock now; every test here needs rows in
    specific UTC day buckets, so the timestamp is rewritten in place.
    """
    pq.persist_complete(rid, "knowledge", endpoint, "site", 3,
                        in_tok, out_tok, 1.0, 2.0, "ok", kind="chat")
    pq._conn.execute(
        "UPDATE proxy_completions SET completed_at=? WHERE request_id=?",
        (when, rid))


def _bucket(ts: float) -> int:
    return int(ts // DAY) * int(DAY)


@pytest.fixture()
def pq(tmp_path):
    q = PersistentQueue(str(tmp_path / "q.db"))
    yield q
    q.close()


def test_lifetime_total_survives_the_prune(pq):
    # Three days of traffic, two of them beyond a 30-day retention window.
    now = time.time()
    _at(pq, "old1", now - 40 * DAY)
    _at(pq, "old2", now - 35 * DAY)
    _at(pq, "recent", now - 2 * DAY)
    before = pq.savings_summary(today_start=now + DAY)  # nothing counts as today
    assert before["total_tokens_in"] == 3 * MTOK
    assert before["total_usd"] == pytest.approx(0.12, abs=1e-6)

    pq.cleanup_old_completions(max_age_s=30 * DAY)

    # The rows really went — otherwise this test proves nothing about pruning.
    remaining = pq._conn.execute(
        "SELECT COUNT(*) FROM proxy_completions").fetchone()[0]
    assert remaining == 1

    after = pq.savings_summary(today_start=now + DAY)
    assert after["total_tokens_in"] == before["total_tokens_in"]
    assert after["total_tokens_out"] == before["total_tokens_out"]
    assert after["total_usd"] == before["total_usd"]
    # ...and the per-endpoint row keeps the lifetime tokens, not the survivors.
    assert [r["tokens_in"] for r in after["by_endpoint"]] == [3 * MTOK]


def test_finalising_twice_does_not_double_count(pq):
    now = time.time()
    _at(pq, "old", now - 40 * DAY)
    _at(pq, "recent", now - 1 * DAY)
    pq.cleanup_old_completions(max_age_s=30 * DAY)
    once = pq.savings_summary(today_start=now + DAY)
    pq.cleanup_old_completions(max_age_s=30 * DAY)
    twice = pq.savings_summary(today_start=now + DAY)
    assert twice["total_tokens_in"] == once["total_tokens_in"] == 2 * MTOK
    assert twice["total_usd"] == once["total_usd"]
    # One row per (endpoint, day) — the upsert updates, it does not append.
    assert pq._conn.execute(
        "SELECT COUNT(*) FROM proxy_savings_daily").fetchone()[0] == 2


def test_a_finalised_day_and_today_are_each_counted_exactly_once(pq):
    """The rollup and the live table overlap on every unpruned day; the total
    must reconcile them, not add them up."""
    now = time.time()
    today_start = float(_bucket(now))
    _at(pq, "past", today_start - 12 * 3600)     # yesterday
    _at(pq, "today", today_start + 60)
    # Finalise with a retention so long that NOTHING is pruned: both days are
    # now present in the rollup AND still live. Naive addition would double.
    pq.cleanup_old_completions(max_age_s=3650 * DAY)
    assert pq._conn.execute(
        "SELECT COUNT(*) FROM proxy_completions").fetchone()[0] == 2

    s = pq.savings_summary(today_start=today_start)
    assert s["total_tokens_in"] == 2 * MTOK
    assert s["today_tokens_in"] == MTOK
    assert sum(r["tokens_in"] for r in s["by_endpoint"]) == 2 * MTOK


def test_a_partially_pruned_bucket_is_not_clobbered_downward(pq):
    """The freeze rule. A bucket the prune has eaten into recomputes LOW from
    the live table; that recompute must never overwrite the correct value an
    earlier sweep stored, or the lifetime total shrinks with nothing to
    rebuild it from."""
    now = time.time()
    day_start = float(_bucket(now - 30 * DAY))
    _at(pq, "early", day_start + 2 * 3600)       # 02:00 — falls before the cutoff
    _at(pq, "late", day_start + 20 * 3600)       # 20:00 — falls after it
    # A retention whose cutoff lands at 12:00 inside that day, straddling it.
    straddle = now - (day_start + 12 * 3600)

    pq.cleanup_old_completions(max_age_s=straddle)
    stored = pq._conn.execute(
        "SELECT tokens_in FROM proxy_savings_daily WHERE day=?",
        (int(day_start),)).fetchone()
    assert stored == (2 * MTOK,), "the full day must be finalised before it is cut"
    assert pq._conn.execute(
        "SELECT COUNT(*) FROM proxy_completions").fetchone()[0] == 1

    # Second sweep: the 02:00 row is gone, so the live table can only account
    # for half the day. The stored value must not follow it down.
    pq.cleanup_old_completions(max_age_s=straddle)
    assert pq._conn.execute(
        "SELECT tokens_in FROM proxy_savings_daily WHERE day=?",
        (int(day_start),)).fetchone() == (2 * MTOK,)
    assert pq.savings_summary(today_start=now + DAY)["total_tokens_in"] == 2 * MTOK


def test_a_dropped_finalisation_write_loses_nothing(pq, monkeypatch):
    """`_w` drops a write when its bounded queue is full — deliberately, so DB
    I/O can never block the event loop. The DELETE that follows must therefore
    not assume the finalisation landed: it is bounded by what is actually in
    the rollup, evaluated inside the statement, so an unfinalised row survives
    to be pruned by the next sweep."""
    now = time.time()
    _at(pq, "old", now - 40 * DAY)
    _at(pq, "recent", now - 1 * DAY)
    expected = pq.savings_summary(today_start=now + DAY)["total_tokens_in"]
    assert expected == 2 * MTOK

    real_w = pq._w
    dropped = []

    def drop_finalisations(sql, params=()):
        if "INSERT INTO proxy_savings_daily" in sql:
            dropped.append(sql)          # the _queue.Full branch of _w
            return
        real_w(sql, params)

    monkeypatch.setattr(pq, "_w", drop_finalisations)
    pq.cleanup_old_completions(max_age_s=30 * DAY)
    assert dropped, "the finalisation write was not the one intercepted"

    # The DELETE ran with an empty rollup and must have refused every row.
    assert pq._conn.execute(
        "SELECT COUNT(*) FROM proxy_completions").fetchone()[0] == 2
    assert pq.savings_summary(today_start=now + DAY)["total_tokens_in"] == expected

    monkeypatch.setattr(pq, "_w", real_w)
    pq.cleanup_old_completions(max_age_s=30 * DAY)
    assert pq._conn.execute(
        "SELECT COUNT(*) FROM proxy_completions").fetchone()[0] == 1
    assert pq.savings_summary(today_start=now + DAY)["total_tokens_in"] == expected


def test_today_keeps_its_own_boundary_and_its_own_answer(pq):
    """Regression guard on the half that must NOT move: `today_*` is still the
    live rows since `today_start`, and neither the rollup nor a prune of older
    days may touch it."""
    now = time.time()
    today_start = float(_bucket(now))
    _at(pq, "old", now - 40 * DAY)
    _at(pq, "yesterday", today_start - 3600)
    _at(pq, "today", today_start + 3600, in_tok=2 * MTOK)

    before = pq.savings_summary(today_start=today_start)
    assert before["today_tokens_in"] == 2 * MTOK
    assert before["today_tokens_out"] == 0
    assert before["today_usd"] == pytest.approx(0.08, abs=1e-6)
    assert before["today_start"] == int(today_start)

    pq.cleanup_old_completions(max_age_s=30 * DAY)
    after = pq.savings_summary(today_start=today_start)
    assert after["today_usd"] == before["today_usd"]
    assert after["today_tokens_in"] == before["today_tokens_in"]
    assert after["today_tokens_out"] == before["today_tokens_out"]
    # today_usd counts ONLY today; the lifetime total counts all four M tokens.
    assert after["total_tokens_in"] == 4 * MTOK
    assert after["today_usd"] < after["total_usd"]


def test_migration_backfills_a_lifetime_from_a_pre_existing_table(tmp_path):
    """The upgrade path: a DB that predates the rollup gets one, populated from
    whatever completions it still holds, on the next open."""
    path = str(tmp_path / "q.db")
    now = time.time()
    first = PersistentQueue(path)
    _at(first, "a", now - 10 * DAY)
    _at(first, "b", now - 3 * DAY)
    assert first._conn.execute(
        "SELECT COUNT(*) FROM proxy_savings_daily").fetchone()[0] == 0
    first.close()

    reopened = PersistentQueue(path)
    backfilled = reopened._conn.execute(
        "SELECT day, tokens_in FROM proxy_savings_daily ORDER BY day").fetchall()
    assert [r[1] for r in backfilled] == [MTOK, MTOK]
    assert [r[0] for r in backfilled] == [_bucket(now - 10 * DAY),
                                          _bucket(now - 3 * DAY)]
    # It is a LIFETIME: prune everything and the total still stands.
    reopened.cleanup_old_completions(max_age_s=DAY)
    assert reopened._conn.execute(
        "SELECT COUNT(*) FROM proxy_completions").fetchone()[0] == 0
    assert reopened.savings_summary(
        today_start=now + DAY)["total_tokens_in"] == 2 * MTOK

    # ...and a second open does not re-derive it from the now-empty table.
    reopened.close()
    third = PersistentQueue(path)
    assert third.savings_summary(
        today_start=now + DAY)["total_tokens_in"] == 2 * MTOK
    third.close()


# --- the route, not just the producer ---------------------------------------

class _FakeRequest:
    """The one attribute this read-only handler touches (as in
    `test_fleet_analytics_schema.py`)."""

    def __init__(self, **params):
        self.query_params = {k: str(v) for k, v in params.items()}


def test_the_route_serves_the_lifetime_total_after_a_prune(tmp_path):
    """A unit-tested producer proves nothing about what the dashboard receives.
    Drive `GET /v1/fleet/savings` through its real handler, over a DB whose
    older completions have been pruned away."""
    import asyncio
    import json

    from roadstead.config import ProxyConfig
    from roadstead.service import ProxyService

    svc = ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db")))
    pq = svc._queue_db
    try:
        now = time.time()
        _at(pq, "old", now - 40 * DAY)
        _at(pq, "recent", now - 1 * DAY)
        pq.cleanup_old_completions(max_age_s=30 * DAY)
        assert pq._conn.execute(
            "SELECT COUNT(*) FROM proxy_completions").fetchone()[0] == 1

        resp = asyncio.run(svc.handle_fleet_savings(
            _FakeRequest(since=int(now + DAY))))
        body = json.loads(resp.body)
        assert body["total_tokens_in"] == 2 * MTOK
        assert body["total_usd"] == pytest.approx(0.08, abs=1e-6)
        assert body["today_usd"] == 0.0
        assert [r["tokens_in"] for r in body["by_endpoint"]] == [2 * MTOK]
    finally:
        pq.close()
