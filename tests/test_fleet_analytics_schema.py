"""The analytics schemas in `docs/api.md` §3.6 and §3.12, pinned to reality.

These payloads have external consumers — in the origin fleet a web UI reads
them — so their key names are public API. Both sections document them to column
level; this file drives the REAL producers (§3.6) and the REAL handlers (§3.12)
against a seeded `queue.db` and asserts the key sets match, **in both
directions**:

  * every field the code emits is documented (nothing ships undocumented);
  * every field the doc documents is emitted (the doc cannot describe a field
    that was renamed or dropped).

One direction alone is the usual failure. A doc-to-code check passes while new
undocumented fields accumulate; a code-to-doc check passes while the doc grows
fiction. Together they make the sections mechanically true rather than prose.

The parsing is deliberately dumb — it reads the markdown tables under each
`####` heading of ONE section — so writing the doc badly fails loudly here
rather than quietly weakening the check.

🚨 **§3.12 is pinned against the HANDLERS, not the producers, and that is not
tidiness.** Three of its seven routes ship a shape no producer returns:
`/v1/history` is wrapped in a `buckets` envelope, `/v1/inflight` has its `ts`
stamped on the way out, and `/v1/fleet/cache-attribution` gains `hit_rate_source`
from the gap-filling overlay. A producer-level pin would have called all three
fields fiction — or, worse, passed while the doc quietly described something no
consumer ever receives.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time

import pytest

from roadstead.config import ProxyConfig
from roadstead.queue import PersistentQueue
from roadstead.scheduler import QueuedRequest
from roadstead.service import ProxyService

from tests.wire_contract import API_DOC


# --------------------------------------------------------------------------
# Parse §3.6 / §3.12 out of docs/api.md
# --------------------------------------------------------------------------

_FIELD_ROW = re.compile(r"^\|\s*`([A-Za-z_][A-Za-z0-9_]*)`\s*\|")

#: The two sections this file reads back, by their exact `###` headings.
_S36 = "### 3.6 `/v1/fleet/*` analytics — response schemas"
_S312 = "### 3.12 The rest of the analytics plane — live, historical and forensic reads"


def _documented_tables(section: str) -> dict[str, list[set[str]]]:
    """{`####` heading -> [field set per markdown TABLE, in document order]}.

    Bounded at the next `###`, so a heading in one section can never satisfy a
    lookup meant for the other — the sections document adjacent routes with
    similar names, which is exactly where a silently-widened match hides.

    Per-table, deliberately. An earlier version pooled every table under a
    heading into one set, and mutation-testing caught it being weaker than
    advertised: deleting `p95` from the `calls[]` table still "passed" because
    `by_endpoint_1h[]` documents a field of the same name. Pooling makes the
    doc-documents-a-field-that-does-not-exist direction work and quietly breaks
    the field-exists-but-is-undocumented direction. Compare shape to shape.
    """
    doc = API_DOC.read_text(encoding="utf-8")
    rest = doc[doc.index(section) + len(section):]
    ends = [i for i in (rest.find("\n### "), rest.find("\n## ")) if i != -1]
    section_text = rest[:min(ends)] if ends else rest

    out: dict[str, list[set[str]]] = {}
    current: str | None = None
    in_table = False
    for line in section_text.splitlines():
        if line.startswith("#### "):
            current = line[len("#### "):].strip()
            out.setdefault(current, [])
            in_table = False
            continue
        if current is None:
            continue
        if not line.startswith("|"):
            in_table = False
            continue
        if not in_table:
            out[current].append(set())   # a new table begins
            in_table = True
        m = _FIELD_ROW.match(line)
        if m:
            out[current][-1].add(m.group(1))
    # drop header/separator-only blocks that yielded nothing
    return {k: [t for t in v if t] for k, v in out.items()}


def _tables_for(substring: str, section: str = _S36) -> list[set[str]]:
    docs = _documented_tables(section)
    matches = [k for k in docs if substring in k]
    assert len(matches) == 1, (
        f"expected exactly one heading containing {substring!r} in "
        f"{section[:8]}, found {matches} — headings: {sorted(docs)}")
    return docs[matches[0]]


def _tables_312(substring: str) -> list[set[str]]:
    return _tables_for(substring, _S312)


def test_the_section_parses_into_the_expected_shape():
    """🚨 Refuse to pass on an empty set. If the heading text or the table
    format changes, every check below would sail through on empty sets."""
    docs = _documented_tables(_S36)
    assert len(docs) >= 4, f"expected >=4 subsections in §3.6, found {sorted(docs)}"
    expected_tables = {"fleet/activity": 3, "fleet/savings": 2, "v1/usage": 2}
    for substring, n in expected_tables.items():
        tables = _tables_for(substring)
        assert len(tables) == n, (
            f"§3.6 '{substring}' should document {n} tables (a top-level shape "
            f"plus its nested rows), found {len(tables)}: {tables}")
        for t in tables:
            assert len(t) >= 3, f"§3.6 '{substring}' has a thin table: {t}"


# --------------------------------------------------------------------------
# A seeded DB with enough shape to exercise every branch
# --------------------------------------------------------------------------

@pytest.fixture
def seeded(tmp_path):
    pq = PersistentQueue(str(tmp_path / "q.db"))
    for i in range(6):
        # Non-zero queue_wait_ms on every row: the latency-basis test below is
        # vacuous without it, since that is the only term separating the two.
        pq.persist_complete(
            f"r{i}", "sidekick" if i % 2 else "forum-agent",
            "tier3" if i % 3 else "tier2", "site", 3,
            100 + i, 20 + i, 0.5 + i * 0.1, 25.0 + 10.0 * i,
            "ok" if i != 4 else "error", kind="chat")
    pq.flush(timeout=5.0)
    try:
        yield pq
    finally:
        pq.close()


def _assert_matches(actual: set[str], documented: set[str], what: str,
                    section: str = "§3.6"):
    undocumented = actual - documented
    fictional = documented - actual
    assert not undocumented, (
        f"{what}: emits {sorted(undocumented)}, which docs/api.md {section} does "
        f"not document — these are public API, add them")
    assert not fictional, (
        f"{what}: docs/api.md {section} documents {sorted(fictional)}, which is "
        f"not emitted — renamed or dropped without updating the contract")


def test_fleet_activity_matches_the_documented_schema(seeded):
    data = seeded.fleet_activity(window_s=86400, bin_s=600)
    top, calls, by_ep = _tables_for("fleet/activity")
    assert data["calls"], "seed produced no bins — the check would be vacuous"
    assert data["by_endpoint_1h"], "seed produced no endpoint rows"
    _assert_matches(set(data), top, "fleet_activity (top level)")
    _assert_matches(set(data["calls"][0]), calls, "fleet_activity calls[]")
    _assert_matches(set(data["by_endpoint_1h"][0]), by_ep,
                    "fleet_activity by_endpoint_1h[]")


def test_savings_summary_matches_the_documented_schema(seeded):
    data = seeded.savings_summary()
    top, by_ep = _tables_for("fleet/savings")
    assert data["by_endpoint"], "seed produced no endpoint rows"
    _assert_matches(set(data), top, "savings_summary (top level)")
    _assert_matches(set(data["by_endpoint"][0]), by_ep,
                    "savings_summary by_endpoint[]")


def test_usage_rollup_matches_the_documented_schema(seeded):
    rows = seeded.usage_rollup(dimension="agent", hours=24.0)
    assert rows, "seed produced no rows — the check would be vacuous"
    envelope_doc, rows_doc = _tables_for("v1/usage")
    # The envelope is built in the handler, not the producer; its three keys are
    # a literal there, so pin them as a literal here rather than inventing a
    # Request to drive handle_usage for three strings.
    _assert_matches({"dimension", "hours", "rows"}, envelope_doc,
                    "/v1/usage envelope")
    _assert_matches(set(rows[0]), rows_doc, "usage_rollup rows[]")


# --------------------------------------------------------------------------
# The documented invariants that are easy to get wrong
# --------------------------------------------------------------------------

def test_the_unopened_db_shapes_are_the_narrower_ones(tmp_path):
    """§3.6's last subsection: with no connection these early-out to a SMALLER
    shape. A consumer assuming `now` or `today_start` is always present gets a
    KeyError, not a degraded value — documented because it is easy to hit in a
    test double and never in production."""
    pq = PersistentQueue(str(tmp_path / "gone.db"))
    pq.close()
    pq._conn = None

    activity = pq.fleet_activity()
    assert set(activity) == {"window_s", "bin_s", "calls", "by_endpoint_1h"}
    assert "now" not in activity

    savings = pq.savings_summary()
    assert set(savings) == {"today_usd", "total_usd", "by_endpoint"}
    assert "today_start" not in savings

    assert pq.usage_rollup() == []


def test_the_em_dash_sentinel_is_unreachable_by_construction(tmp_path):
    """§3.6: `usage_rollup` falls back to `"—"` on a NULL dimension.

    That fallback is DEFENSIVE, not reachable: all three groupable columns are
    `NOT NULL`, so `persist_complete` rejects the row before the rollup ever
    sees it. Asserted here rather than left implied, because the honest reading
    of the code ("this handles NULLs") and the honest reading of the schema
    ("there are no NULLs") disagree, and only one of them is checkable.

    🚨 If a migration makes any of these nullable, this test fails — and at that
    point the `"—"` row in §3.6 stops being trivia and starts being contract.
    """
    pq = PersistentQueue(str(tmp_path / "q.db"))
    try:
        cols = {
            row[1]: bool(row[3])  # name -> notnull
            for row in pq._reader().execute(
                "PRAGMA table_info(proxy_completions)").fetchall()
        }
        for groupable in ("agent_id", "endpoint", "call_site"):
            assert cols.get(groupable) is True, (
                f"proxy_completions.{groupable} is no longer NOT NULL — the "
                f'"—" fallback in usage_rollup is now reachable, so §3.6 needs '
                f"a real test of it rather than this one")

        # And prove the constraint actually bites, rather than trusting PRAGMA.
        with pytest.raises(sqlite3.IntegrityError):
            pq._conn.execute(
                "INSERT INTO proxy_completions "
                "(request_id, agent_id, endpoint, call_site, priority, status, "
                " completed_at) VALUES (?,?,?,?,?,?,?)",
                ("x", None, "tier3", "site", 3, "ok", time.time()))
    finally:
        pq.close()


def test_latency_bases_differ_between_the_two_rollups(seeded):
    """§3.6 documents that `usage_rollup`'s p50/p95 INCLUDE queue wait while
    `fleet_activity`'s p95 does not. Same corpus, so with non-zero queue waits
    the former must come out strictly higher — otherwise one of them silently
    changed basis and every latency comparison across the two is wrong."""
    activity = seeded.fleet_activity(window_s=86400, bin_s=86400)
    rollup = seeded.usage_rollup(dimension="endpoint", hours=24.0)
    assert activity["calls"] and rollup
    worst_activity = max(c["p95"] for c in activity["calls"])
    worst_rollup = max(r["p95_ms"] for r in rollup)
    assert worst_rollup > worst_activity, (
        f"usage_rollup p95 {worst_rollup} should exceed fleet_activity p95 "
        f"{worst_activity} because it adds queue_wait_ms")


# --------------------------------------------------------------------------
# §3.12 — the other seven, pinned at the HANDLER
# --------------------------------------------------------------------------

class _FakeRequest:
    """The one attribute these read-only handlers touch."""

    def __init__(self, **params):
        self.query_params = {k: str(v) for k, v in params.items()}


async def _body(coro) -> dict:
    return json.loads((await coro).body)


@pytest.fixture
def analytics(tmp_path):
    """A ProxyService with something for every §3.12 surface to report.

    Not started: `startup()` dials backends and starts the poller, and none of
    these handlers needs either. What startup would have done for them is done
    here explicitly — registering the cost model's endpoints — so the fixture
    says out loud what each surface depends on.

    🚨 Every seeded completion has a NULL `cached_tokens` (the shape a backend
    that publishes no per-request counter produces), and `endpoint_cache_hit_rate`
    is populated for both endpoints. That is what makes the attribution overlay
    fire on EVERY `by_endpoint` row and on `fleet`, so `hit_rate_source` — which
    §3.12 documents as present only on an overlaid row — is really emitted here
    rather than taken on trust. Its conditionality is the other half, and lives
    in `test_cache_attribution_overlay.py`.
    """
    svc = ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db")))
    pq = svc._queue_db
    for i in range(6):
        pq.persist_complete(
            f"r{i}", "sidekick" if i % 2 else "forum-agent",
            "tier3" if i % 3 else "tier2", "site", 3,
            100 + i, 20 + i, 0.5 + i * 0.1, 25.0 + 10.0 * i,
            "ok" if i != 4 else "error", kind="chat")
    pq.flush(timeout=5.0)
    pq.persist_timeout_event(
        request_id="s1", endpoint="tier3", priority=3, agent_id="forum-agent",
        call_site="forum-agent.chat", layer="stream", elapsed_s=31.0,
        applied_timeout_s=300.0, queue_wait_ms=0.0, in_flight=1, queued=0,
        max_slots=4, est_in=900, est_out=80, recommended_ms=180000.0,
        under_recommended=False, caller_id="pool-analyst", abort_reason="stall")
    svc._state.endpoint_cache_hit_rate = {
        "tier3": {"hit_rate": 0.61, "queries": 10,
                  "source": "backend_prefix_cache_metrics"},
        "tier2": {"hit_rate": 0.42, "queries": 5,
                  "source": "backend_prefix_cache_metrics"},
    }
    # One dispatched request, so /v1/inflight has a row rather than an empty
    # list — the shape that carries eleven of the fields §3.12 documents.
    now = time.monotonic()
    svc._scheduler.enqueue(QueuedRequest.create(
        agent_id="forum-agent", endpoint="tier3", priority="P1_TURN_SUPPORT",
        call_site="forum-agent.chat", payload_type="chat_completion",
        payload={"messages": [{"role": "user", "content": "x"}], "max_tokens": 64},
        timeout_s=60.0, now=now))
    assert svc._scheduler.tick(now), "nothing dispatched — /v1/inflight would be empty"
    svc._cost_model.register_endpoint("tier3", 4)
    svc._cost_model.record_completion("tier3", "forum-agent.chat", 100, 40, 1.5, 1)
    try:
        yield svc
    finally:
        pq.close()


def test_the_312_section_parses_into_the_expected_shape():
    """The §3.6 guard, again for §3.12: refuse to check anything against an
    empty set. Every count below is a shape (a top level plus its nested rows),
    so a table lost to a formatting slip fails here and not silently later."""
    docs = _documented_tables(_S312)
    expected = {
        "v1/inflight": 3,                 # top level, requests[], per_endpoint{}
        "v1/history": 4,                  # envelope, buckets[], per_endpoint{}, per_agent{}
        "v1/series": 2,
        "cost-model": 3,                  # the endpoint entry, prefill_ewma, per_call_site{}
        "top-callers": 2,
        "fleet/cache-attribution": 4,     # top level, by_call_site[], by_endpoint[], fleet
        "timeouts/stalls": 2,
    }
    assert len(docs) >= len(expected), f"§3.12 headings: {sorted(docs)}"
    for substring, n in expected.items():
        tables = _tables_312(substring)
        assert len(tables) == n, (
            f"§3.12 '{substring}' should document {n} tables, found "
            f"{len(tables)}: {tables}")
        # No blanket per-table floor here, unlike §3.6: `/v1/history`'s envelope
        # documents exactly one field (`buckets`) and is not thin, it is small.
        # An EMPTY table is still a parse failure, and the per-heading total
        # catches a section quietly hollowed out.
        for t in tables:
            assert t, f"§3.12 '{substring}' has an empty table"
        assert sum(len(t) for t in tables) >= 6, (
            f"§3.12 '{substring}' documents only "
            f"{sum(len(t) for t in tables)} fields in total")


async def test_inflight_matches_the_documented_schema(analytics):
    data = await _body(analytics.handle_inflight(_FakeRequest()))
    top, requests, per_ep = _tables_312("v1/inflight")
    assert data["requests"], "nothing in flight — the check would be vacuous"
    assert data["per_endpoint"], "no endpoints — the check would be vacuous"
    _assert_matches(set(data), top, "inflight (top level)", "§3.12")
    _assert_matches(set(data["requests"][0]), requests, "inflight requests[]", "§3.12")
    _assert_matches(set(next(iter(data["per_endpoint"].values()))), per_ep,
                    "inflight per_endpoint{}", "§3.12")


async def test_history_matches_the_documented_schema(analytics):
    data = await _body(analytics.handle_history(_FakeRequest()))
    envelope, buckets, per_ep, per_agent = _tables_312("v1/history")
    assert data["buckets"], "seed produced no buckets"
    bucket = data["buckets"][0]
    assert bucket["per_endpoint"] and bucket["per_agent"]
    _assert_matches(set(data), envelope, "/v1/history envelope", "§3.12")
    _assert_matches(set(bucket), buckets, "history buckets[]", "§3.12")
    _assert_matches(set(next(iter(bucket["per_endpoint"].values()))), per_ep,
                    "history per_endpoint{}", "§3.12")
    _assert_matches(set(next(iter(bucket["per_agent"].values()))), per_agent,
                    "history per_agent{}", "§3.12")


async def test_series_matches_the_documented_schema(analytics):
    data = await _body(analytics.handle_series(_FakeRequest(endpoint="tier3")))
    top, series = _tables_312("v1/series")
    assert data["calls_series"], "seed produced no bins"
    _assert_matches(set(data), top, "series (top level)", "§3.12")
    _assert_matches(set(data["calls_series"][0]), series, "series calls_series[]", "§3.12")


async def test_series_without_an_endpoint_is_a_400(analytics):
    """§3.12: the one refusal in the section. A missing `endpoint` cannot
    degrade to a fleet-wide series — that is a different question with a
    plausible-looking answer."""
    resp = await analytics.handle_series(_FakeRequest())
    assert resp.status_code == 400
    assert json.loads(resp.body) == {"error": "endpoint query param required"}


async def test_cost_model_matches_the_documented_schema(analytics):
    data = await _body(analytics.handle_cost_model(_FakeRequest()))
    entry_doc, ewma_doc, call_site_doc = _tables_312("cost-model")
    assert data, "no endpoint registered — the check would be vacuous"
    entry = data["tier3"]
    assert entry["per_call_site"], "no call_site observed"
    _assert_matches(set(entry), entry_doc, "cost-model entry", "§3.12")
    _assert_matches(set(entry["prefill_ewma"]), ewma_doc,
                    "cost-model prefill_ewma", "§3.12")
    _assert_matches(set(next(iter(entry["per_call_site"].values()))), call_site_doc,
                    "cost-model per_call_site{}", "§3.12")


async def test_top_callers_matches_the_documented_schema(analytics):
    data = await _body(analytics.handle_top_callers(_FakeRequest()))
    top, caller = _tables_312("top-callers")
    assert data["providers"], "seed produced no endpoint rows"
    _assert_matches(set(data), top, "top_callers (top level)", "§3.12")
    _assert_matches(set(next(iter(data["providers"].values()))[0]), caller,
                    "top_callers providers[][]", "§3.12")


async def test_cache_attribution_matches_the_documented_schema(analytics):
    data = await _body(analytics.handle_cache_attribution(_FakeRequest()))
    top, by_call_site, by_endpoint, fleet = _tables_312("fleet/cache-attribution")
    assert data["by_call_site"] and data["by_endpoint"] and data["fleet"]
    # The overlay must have fired, or the `hit_rate_source` row below would be
    # checked against a payload that never had the chance to carry it.
    assert all("hit_rate_source" in r for r in data["by_endpoint"])
    _assert_matches(set(data), top, "cache_attribution (top level)", "§3.12")
    _assert_matches(set(data["by_call_site"][0]), by_call_site,
                    "cache_attribution by_call_site[]", "§3.12")
    _assert_matches(set(data["by_endpoint"][0]), by_endpoint,
                    "cache_attribution by_endpoint[]", "§3.12")
    _assert_matches(set(data["fleet"]), fleet, "cache_attribution fleet", "§3.12")


async def test_stall_aborts_matches_the_documented_schema(analytics):
    data = await _body(analytics.handle_stall_aborts(
        _FakeRequest(caller="pool-analyst", since=0, until=time.time() + 60)))
    envelope, rows = _tables_312("timeouts/stalls")
    assert data["rows"], "seed produced no stall rows"
    _assert_matches(set(data), envelope, "/v1/timeouts/stalls envelope", "§3.12")
    _assert_matches(set(data["rows"][0]), rows, "stall_aborts rows[]", "§3.12")


def test_the_312_unopened_db_shapes_are_the_narrower_ones(tmp_path):
    """§3.12's last subsection, the same trap as §3.6's: with no connection
    three of these early-out to a SMALLER shape and one goes null rather than
    absent. A consumer reaching for `now` gets a KeyError from two of them and
    a None from the third, which is exactly the sort of difference a test
    double never shows you."""
    pq = PersistentQueue(str(tmp_path / "gone.db"))
    pq.close()
    pq._conn = None

    series = pq.endpoint_series("tier3")
    assert set(series) == {"endpoint", "window_s", "bin_s", "calls_series"}
    assert "now" not in series

    top = pq.top_callers()
    assert set(top) == {"window_s", "providers"}
    assert "now" not in top

    attribution = pq.cache_attribution()
    assert set(attribution) == {"window_s", "now", "by_call_site",
                                "by_endpoint", "fleet"}
    assert attribution["now"] is None and attribution["fleet"] is None

    assert pq.history_buckets() == []
    assert pq.stall_aborts("pool-analyst", 0.0, 1.0) == []
