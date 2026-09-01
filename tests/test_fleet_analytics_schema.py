"""The `/v1/fleet/*` analytics schemas in `docs/api.md` §3.6, pinned to reality.

These payloads have external consumers — in the origin fleet a web UI reads
them — so their key names are public API. §3.6 documents them to column level;
this file drives the REAL producers against a seeded `queue.db` and asserts the
key sets match, **in both directions**:

  * every field the producer emits is documented (nothing ships undocumented);
  * every field §3.6 documents is emitted (the doc cannot describe a field that
    was renamed or dropped).

One direction alone is the usual failure. A doc-to-code check passes while new
undocumented fields accumulate; a code-to-doc check passes while the doc grows
fiction. Together they make §3.6 mechanically true rather than prose.

The parsing is deliberately dumb — it reads the markdown tables under each
`####` heading in §3.6 — so writing the doc badly fails loudly here rather than
quietly weakening the check.
"""
from __future__ import annotations

import re
import sqlite3
import time

import pytest

from roadstead.queue import PersistentQueue

from tests.wire_contract import API_DOC


# --------------------------------------------------------------------------
# Parse §3.6 out of docs/api.md
# --------------------------------------------------------------------------

_FIELD_ROW = re.compile(r"^\|\s*`([A-Za-z_][A-Za-z0-9_]*)`\s*\|")


def _documented_tables() -> dict[str, list[set[str]]]:
    """{`####` heading -> [field set per markdown TABLE, in document order]}.

    Per-table, deliberately. An earlier version pooled every table under a
    heading into one set, and mutation-testing caught it being weaker than
    advertised: deleting `p95` from the `calls[]` table still "passed" because
    `by_endpoint_1h[]` documents a field of the same name. Pooling makes the
    doc-documents-a-field-that-does-not-exist direction work and quietly breaks
    the field-exists-but-is-undocumented direction. Compare shape to shape.
    """
    doc = API_DOC.read_text(encoding="utf-8")
    start = doc.index("### 3.6 `/v1/fleet/*` analytics — response schemas")
    section = doc[start:doc.index("\n## ", start)]

    out: dict[str, list[set[str]]] = {}
    current: str | None = None
    in_table = False
    for line in section.splitlines():
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


def _tables_for(substring: str) -> list[set[str]]:
    docs = _documented_tables()
    matches = [k for k in docs if substring in k]
    assert len(matches) == 1, (
        f"expected exactly one §3.6 heading containing {substring!r}, "
        f"found {matches} — headings: {sorted(docs)}")
    return docs[matches[0]]


def test_the_section_parses_into_the_expected_shape():
    """🚨 Refuse to pass on an empty set. If the heading text or the table
    format changes, every check below would sail through on empty sets."""
    docs = _documented_tables()
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


def _assert_matches(actual: set[str], documented: set[str], what: str):
    undocumented = actual - documented
    fictional = documented - actual
    assert not undocumented, (
        f"{what}: emits {sorted(undocumented)}, which docs/api.md §3.6 does not "
        f"document — these are public API, add them")
    assert not fictional, (
        f"{what}: docs/api.md §3.6 documents {sorted(fictional)}, which is not "
        f"emitted — renamed or dropped without updating the contract")


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
