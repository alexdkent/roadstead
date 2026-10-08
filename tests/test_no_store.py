"""No-store — the policy, the grant loader, the store, and the guards on both.

The wire-level contract (every door, streaming, every surface) is
``tests/e2e/test_no_store.py``. This file pins what that cannot see:

  * the decision table, as a pure function — each refusal reason in the order an
    operator would fix them;
  * the GRANT LOADER — the operator-gating half. A grant is three keys read
    together (`content_no_store: allowed`, `approved_by: operator`, an ISO
    `approved_on`); anything less is not a grant AND says so. Absent ⇒ off;
  * the completion store — what `persist_complete` drops, what it keeps, and
    that a legacy database migrates;
  * two source guards, because both halves of this feature fail SILENTLY:
    every `persist_complete` caller must say what its outcome is (a call site
    that forgets would store a no-store request's content with no error), and
    the grant must be read in exactly the places that decide what it means.
"""
from __future__ import annotations

import ast
import datetime
import json
import logging
import sqlite3
import types
from pathlib import Path

import pytest

from roadstead import hooks, no_store
from roadstead.acl import IPIdentityMap
from roadstead.config import AgentQuotaConfig, ProxyConfig, load_agent_configs
from roadstead.correction import Correction
from roadstead.queue import PersistentQueue
from roadstead.reasoning_replay import ReasoningReplayStore
from roadstead.scheduler import QueuedRequest

PKG = Path(__file__).resolve().parent.parent / "roadstead"
CANARY = "zxq-canary-7f3a9c-do-not-store"


# --------------------------------------------------------------------------- #
# The header
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw,expected", [
    (None, False), ("", False), ("   ", False),
    ("content", True), ("Content", True), ("  CONTENT ", True),
])
def test_parse_header(raw, expected):
    assert no_store.parse_header(raw) is expected


@pytest.mark.parametrize("raw", ["true", "1", "yes", "all", "metadata", "content,x"])
def test_an_undefined_value_raises_rather_than_being_ignored(raw):
    with pytest.raises(no_store.InvalidNoStoreHeader):
        no_store.parse_header(raw)


def test_read_header_tolerates_a_plain_dict_double_and_a_missing_attribute():
    assert no_store.read_header(types.SimpleNamespace(
        headers={"X-Roadstead-No-Store": "content"})) == "content"
    assert no_store.read_header(types.SimpleNamespace(
        headers={"x-roadstead-no-store": "content"})) == "content"
    assert no_store.read_header(types.SimpleNamespace()) is None
    assert no_store.read_header(types.SimpleNamespace(headers={})) is None


# --------------------------------------------------------------------------- #
# The decision
# --------------------------------------------------------------------------- #

def _decide(**kw):
    base = dict(requested=True, authenticated=True, agent_id="caller-a",
                granted=True, shared_identities=frozenset({"internal"}))
    base.update(kw)
    return no_store.decide(**base)


def test_nothing_asked_is_never_a_decision():
    d = _decide(requested=False)
    assert d == no_store.NOT_REQUESTED and not d.requested and not d.honoured


def test_all_three_conditions_honour():
    d = _decide()
    assert d.honoured and d.outcome == "honoured"


@pytest.mark.parametrize("kw,reason", [
    (dict(authenticated=False), "not_authenticated"),
    (dict(agent_id="internal"), "shared_identity"),
    (dict(granted=False), "not_granted"),
])
def test_each_missing_condition_refuses_with_its_own_reason(kw, reason):
    d = _decide(**kw)
    assert d.requested and not d.honoured
    assert d.outcome == f"refused:{reason}" and d.reason == reason


def test_the_reasons_are_ordered_by_what_the_operator_should_fix_first():
    """An unauthenticated caller under a shared, ungranted name is told about
    the credential — the grant is irrelevant until it has one."""
    d = _decide(authenticated=False, agent_id="internal", granted=False)
    assert d.reason == "not_authenticated"
    d = _decide(agent_id="internal", granted=False)
    assert d.reason == "shared_identity"


def test_an_address_registered_identity_is_shared():
    acl = IPIdentityMap()
    assert acl.address_identities() == frozenset({"internal"})
    acl.register("203.0.113.0/24", "shared-label")
    acl.register("203.0.113.9", "one-host")
    assert acl.address_identities() == {"internal", "shared-label", "one-host"}


def test_outcome_helpers_read_a_stub_as_never_asked_and_a_mock_as_no_grant():
    assert no_store.outcome_of(types.SimpleNamespace()) == ""
    assert not no_store.engaged(types.SimpleNamespace())
    # A truthy attribute of the wrong type must not read as a grant.
    assert no_store.outcome_of(types.SimpleNamespace(no_store_outcome=True)) == ""
    assert not no_store.engaged(types.SimpleNamespace(no_store_outcome=1))
    assert no_store.engaged(types.SimpleNamespace(no_store_outcome="honoured"))
    assert not no_store.engaged(
        types.SimpleNamespace(no_store_outcome="refused:not_granted"))


def test_redact():
    assert no_store.redact(True, "secret") == no_store.WITHHELD
    assert no_store.redact(False, "secret") == "secret"


# --------------------------------------------------------------------------- #
# What is recorded about content that is not stored
# --------------------------------------------------------------------------- #

def test_content_summary_describes_the_shape_and_carries_no_content():
    payload = {
        "model": "tier3", "max_tokens": 64,
        "messages": [
            {"role": "system", "content": f"sys {CANARY}"},
            {"role": "user", "content": f"hello {CANARY}"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"function": {"name": "lookup_account",
                              "arguments": f'{{"q": "{CANARY}"}}'}}]},
            {"role": "tool", "content": f"result {CANARY}"},
            {"role": "user", "content": "again"},
            {"role": f"role-{CANARY}", "content": "x"},   # a hostile role
        ],
        "tools": [{"type": "function", "function": {"name": "lookup_account"}},
                  {"type": "function", "function": {"name": f"bad name {CANARY}"}}],
    }
    response = {"model": "served-model-1", "choices": [{"message": {
        "content": f"answer {CANARY}", "tool_calls": [
            {"function": {"name": "lookup_account", "arguments": "{}"}}]}}]}
    meta = no_store.content_summary(payload, response)
    assert CANARY not in json.dumps(meta)
    assert meta["message_count"] == 6
    assert meta["roles"] == {"system": 1, "user": 2, "assistant": 1, "tool": 1,
                             "other": 1}
    assert meta["tools_declared"] == ["lookup_account", "other"]
    assert meta["tool_calls"] == ["lookup_account"]
    assert meta["payload_bytes"] == len(json.dumps(payload, separators=(",", ":")))
    assert meta["response_bytes"] > 0 and meta["choice_count"] == 1
    assert meta["model"] == "served-model-1"


def test_content_summary_is_total_over_odd_shapes():
    for payload, response in [(None, None), ({}, {}), ("x", 3),
                              ({"messages": "no"}, {"choices": "no"}),
                              ({"messages": [1, None, "s"]}, {"model": {"a": 1}})]:
        no_store.content_summary(payload, response)    # must not raise


def test_a_streamed_rows_provenance_pair_survives_without_a_fake_size():
    meta = no_store.content_summary(
        {"messages": [{"role": "user", "content": CANARY}]},
        {"model": "m", "model_source": "backend_echo"})
    assert meta["model"] == "m" and meta["model_source"] == "backend_echo"
    assert "response_bytes" not in meta


# --------------------------------------------------------------------------- #
# The grant loader — the operator-gating half
# --------------------------------------------------------------------------- #

def _load(tmp_path, text):
    p = tmp_path / "agents.yaml"
    p.write_text(text)
    hooks.clear_config_notices()
    return load_agent_configs(p)


GOOD = """\
caller-a:
  content_no_store: allowed
  approved_by: operator
  approved_on: 2026-01-31
"""


def test_a_complete_stanza_is_a_grant(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    cfg = _load(tmp_path, GOOD)
    assert cfg["caller-a"].content_no_store is True
    assert hooks.config_notices() == []
    assert "NO-STORE GRANT agent=caller-a approved_by=operator" in caplog.text


def test_the_default_is_off_and_the_shipped_example_grants_nothing():
    assert AgentQuotaConfig(agent_id="x").content_no_store is False
    shipped = load_agent_configs(PKG / "agents.yaml")
    assert shipped and not any(c.content_no_store for c in shipped.values())
    # …and states the rule where an operator copying the example will read it.
    assert "approved_by: operator" in (PKG / "agents.yaml").read_text()


@pytest.mark.parametrize("stanza,why", [
    ("content_no_store: allowed\n  approved_on: 2026-01-31", "approved_by"),
    ("content_no_store: allowed\n  approved_by: operator", "approved_on"),
    ("content_no_store: allowed", "approved_by"),
    ("content_no_store: allowed\n  approved_by: alice\n  approved_on: 2026-01-31",
     "approved_by"),
    ("content_no_store: allowed\n  approved_by: Operator\n  approved_on: 2026-01-31",
     "approved_by"),
    ("content_no_store: allowed\n  approved_by: operator\n  approved_on: yesterday",
     "approved_on"),
    ("content_no_store: allowed\n  approved_by: operator\n  approved_on: 2999-01-01",
     "future"),
    ('content_no_store: allowed\n  approved_by: operator\n  approved_on: "2026-13-45"',
     "approved_on"),
    # YAML booleans are too easy to type by accident for a privilege.
    ("content_no_store: true\n  approved_by: operator\n  approved_on: 2026-01-31",
     "'allowed'"),
    ("content_no_store: yes\n  approved_by: operator\n  approved_on: 2026-01-31",
     "'allowed'"),
    ("content_no_store: granted\n  approved_by: operator\n  approved_on: 2026-01-31",
     "'allowed'"),
])
def test_anything_less_than_the_whole_stanza_is_not_a_grant_and_says_so(
        tmp_path, caplog, stanza, why):
    caplog.set_level(logging.WARNING)
    cfg = _load(tmp_path, f"caller-a:\n  {stanza}\n")
    assert cfg["caller-a"].content_no_store is False
    notices = hooks.config_notices()
    assert len(notices) == 1
    n = notices[0]
    assert n["subject"] == "caller-a" and n["problem"] == "invalid_grant"
    assert why in n["detail"]
    assert "NO-STORE GRANT" not in caplog.text
    assert "CONFIG NOTICE" in caplog.text


def test_the_approval_keys_alone_ask_for_nothing_and_report_nothing(tmp_path):
    cfg = _load(tmp_path, "caller-a:\n  approved_by: operator\n"
                          "  approved_on: 2026-01-31\n")
    assert cfg["caller-a"].content_no_store is False
    assert hooks.config_notices() == []        # nothing was asked for


@pytest.mark.parametrize("value", ["false", "null"])
def test_an_explicit_off_is_silent(tmp_path, value):
    cfg = _load(tmp_path, f"caller-a:\n  content_no_store: {value}\n")
    assert cfg["caller-a"].content_no_store is False
    assert hooks.config_notices() == []


def test_the_builtin_shared_identity_is_never_a_grantee(tmp_path):
    cfg = _load(tmp_path, GOOD.replace("caller-a", "internal"))
    assert cfg["internal"].content_no_store is False
    assert "built-in shared identity" in hooks.config_notices()[0]["detail"]


def test_a_yaml_datetime_is_accepted_as_a_date(tmp_path):
    cfg = _load(tmp_path, GOOD.replace("2026-01-31", "2026-01-31T10:00:00"))
    assert cfg["caller-a"].content_no_store is True


def test_the_three_keys_are_known_to_the_unknown_key_notice(tmp_path):
    """They are in the allowlist, so a correctly written grant raises no
    `unknown_key` notice (which would read as a typo to the operator)."""
    _load(tmp_path, GOOD)
    assert not [n for n in hooks.config_notices() if n["problem"] == "unknown_key"]


# --------------------------------------------------------------------------- #
# The admin plane cannot grant it
# --------------------------------------------------------------------------- #

def test_the_quota_patch_refuses_the_grant_and_says_where_it_lives():
    from roadstead.management import Invalid, validate_quota_patch
    for field in ("content_no_store", "approved_by", "approved_on"):
        with pytest.raises(Invalid, match="agents config file"):
            validate_quota_patch({field: "allowed"})
    with pytest.raises(Invalid, match="agents config file"):
        validate_quota_patch({"weight": 2.0, "content_no_store": "allowed"})


def test_the_runtime_overlay_cannot_carry_a_grant(tmp_path):
    """Even a hand-edited overlay file: `apply` writes only EDITABLE fields."""
    from roadstead.management import AdminOverlay
    store = tmp_path / "overlay.json"
    store.write_text(json.dumps({"agents": {
        "caller-a": {"weight": 3.0, "content_no_store": True}}}))
    overlay = AdminOverlay(str(store))
    cfg = ProxyConfig()
    reg = types.SimpleNamespace(
        register=lambda **k: None, revoke=lambda *a: None,
        set_expiry=lambda *a: None)
    overlay.apply(reg, cfg)
    assert cfg.agents["caller-a"].weight == 3.0
    assert cfg.agents["caller-a"].content_no_store is False


# --------------------------------------------------------------------------- #
# The completion store
# --------------------------------------------------------------------------- #

PAYLOAD = {"messages": [{"role": "user", "content": CANARY}], "max_tokens": 8}
RESPONSE = {"model": "m", "choices": [{"message": {"content": f"re: {CANARY}"}}]}


def _complete(db, rid, outcome):
    db.persist_complete(
        rid, "caller-a", "tier3", "caller-a.cs", 1, 11, 7, 0.5, 2.0, "ok",
        payload=PAYLOAD, response=RESPONSE, finish_reason="stop",
        cached_tokens=3, no_store=outcome)
    db.flush()


def _row(db, rid):
    db._conn.row_factory = sqlite3.Row
    try:
        return dict(db._conn.execute(
            "SELECT * FROM proxy_completions WHERE request_id=?", (rid,)).fetchone())
    finally:
        db._conn.row_factory = None


def test_an_honoured_completion_drops_content_and_keeps_everything_else(tmp_path):
    db = PersistentQueue(str(tmp_path / "q.db"))
    _complete(db, "r1", "honoured")
    row = _row(db, "r1")
    assert row["payload_json"] is None and row["response_json"] is None
    assert (row["agent_id"], row["endpoint"], row["call_site"], row["status"],
            row["finish_reason"], row["input_tokens"], row["output_tokens"],
            row["cached_tokens"], row["duration_s"], row["queue_wait_ms"],
            row["no_store"]) == (
        "caller-a", "tier3", "caller-a.cs", "ok", "stop", 11, 7, 3, 0.5, 2.0,
        "honoured")
    meta = json.loads(row["content_meta"])
    assert meta["message_count"] == 1 and meta["roles"] == {"user": 1}
    assert meta["payload_bytes"] > 0 and meta["response_bytes"] > 0
    assert CANARY not in json.dumps(row)
    # Not reachable by the replay corpus or the cache-ability screen either.
    assert db.export_corpus(hours=1) == []


@pytest.mark.parametrize("outcome", [None, "", "refused:not_granted",
                                     "refused:not_authenticated"])
def test_everything_else_is_stored_in_full(tmp_path, outcome):
    db = PersistentQueue(str(tmp_path / "q.db"))
    _complete(db, "r1", outcome)
    row = _row(db, "r1")
    assert json.loads(row["payload_json"]) == PAYLOAD
    assert json.loads(row["response_json"]) == RESPONSE
    assert row["no_store"] == (outcome or None) and row["content_meta"] is None


def test_a_corrected_row_rewrite_keeps_the_flag(tmp_path):
    """`INSERT OR REPLACE` on request_id: a second write (degeneration retry,
    schema repair) must not restore the content the first one withheld."""
    db = PersistentQueue(str(tmp_path / "q.db"))
    _complete(db, "r1", "honoured")
    _complete(db, "r1", "honoured")
    assert _row(db, "r1")["payload_json"] is None


def _qreq(outcome):
    return QueuedRequest.create(
        agent_id="caller-a", endpoint="tier3", priority=1, call_site="cs",
        payload_type="chat_completion", payload=dict(PAYLOAD),
        no_store_outcome=outcome)


@pytest.mark.parametrize("outcome,rows", [
    ("honoured", 0), ("", 1), ("refused:not_granted", 1)])
def test_the_restart_wal_row_exists_only_when_content_may_be_stored(
        tmp_path, outcome, rows):
    db = PersistentQueue(str(tmp_path / "q.db"))
    db.persist_enqueue(_qreq(outcome))
    db.flush()
    assert db._conn.execute("SELECT COUNT(*) FROM proxy_queue").fetchone()[0] == rows
    raw = (tmp_path / "q.db-wal")
    blob = (raw.read_bytes() if raw.exists() else b"") + (tmp_path / "q.db").read_bytes()
    assert (CANARY.encode() in blob) == bool(rows)


def test_a_database_from_before_this_change_migrates(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE proxy_completions (request_id TEXT PRIMARY KEY, "
        "agent_id TEXT NOT NULL, endpoint TEXT NOT NULL, call_site TEXT NOT NULL, "
        "priority INTEGER NOT NULL, input_tokens INTEGER, output_tokens INTEGER, "
        "duration_s REAL, queue_wait_ms REAL, status TEXT NOT NULL, "
        "completed_at REAL NOT NULL, payload_json TEXT, response_json TEXT)")
    conn.execute("INSERT INTO proxy_completions VALUES "
                 "('old','a','tier3','cs',1,1,1,1.0,1.0,'ok',1.0,'{}','{}')")
    conn.commit()
    conn.close()
    db = PersistentQueue(str(path))
    cols = {r[1] for r in db._conn.execute("PRAGMA table_info(proxy_completions)")}
    assert {"no_store", "content_meta"} <= cols
    assert _row(db, "old")["no_store"] is None
    _complete(db, "new", "honoured")
    assert _row(db, "new")["no_store"] == "honoured"


# --------------------------------------------------------------------------- #
# The in-memory stores, at the method level
# --------------------------------------------------------------------------- #

def _replay_self():
    ep = types.SimpleNamespace(replay_reasoning_history=True)
    state = types.SimpleNamespace(
        config=types.SimpleNamespace(endpoints={"tier3": ep}),
        reasoning_replay=ReasoningReplayStore(), reasoning_replay_stored=0,
        reasoning_replay_restored=0, reasoning_replay_miss=0,
        reasoning_replay_skipped=0, reasoning_replay_by_endpoint={})
    m = types.SimpleNamespace(state=state)
    for name in ("apply_reasoning_replay_restore", "store_reasoning_replay"):
        setattr(m, name, getattr(Correction, name).__get__(m, Correction))
    return m


def _stub(payload, outcome=""):
    return types.SimpleNamespace(
        payload=payload, stream=False, payload_type="chat_completion",
        endpoint="tier3", request_id="r", call_site="cs", no_store_outcome=outcome)


def test_replay_neither_stores_nor_restores_for_no_store():
    m = _replay_self()
    turn = [{"role": "user", "content": "q"}]
    m.store_reasoning_replay(_stub({"messages": turn}, "honoured"),
                             "a", None, "thinking")
    assert m.state.reasoning_replay.size == 0 and m.state.reasoning_replay_stored == 0
    # control: the same call, not no-store, IS stored
    m.store_reasoning_replay(_stub({"messages": turn}), "a", None, "thinking")
    assert m.state.reasoning_replay_stored == 1
    hist = turn + [{"role": "assistant", "content": "a"},
                   {"role": "user", "content": "q2"}]
    req = _stub({"messages": hist}, "honoured")
    m.apply_reasoning_replay_restore(req)
    assert "reasoning_content" not in req.payload["messages"][1]
    assert m.state.reasoning_replay_restored == 0 and m.state.reasoning_replay_miss == 0
    req = _stub({"messages": hist})
    m.apply_reasoning_replay_restore(req)
    assert req.payload["messages"][1]["reasoning_content"] == "thinking"


# --------------------------------------------------------------------------- #
# Source guards
# --------------------------------------------------------------------------- #

def _calls(name):
    for path in sorted(PKG.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == name):
                yield path.relative_to(PKG), node


def test_every_persist_complete_caller_states_its_no_store_outcome():
    """🚨 The completion-row half of the guarantee lives in `persist_complete`,
    but only for a caller that PASSES the outcome. A call site added later that
    forgets would store a no-store request's content, with no error and no log —
    and the e2e matrix would stay green because it only drives the sites that
    existed. Passing `None` is allowed; omitting the keyword is not."""
    sites = list(_calls("persist_complete"))
    assert len(sites) >= 4, "the call sites moved — this guard measures nothing"
    missing = [f"{p}:{n.lineno}" for p, n in sites
               if "no_store" not in {k.arg for k in n.keywords}]
    assert not missing, (
        "persist_complete called without no_store=: " + ", ".join(missing))


def test_the_grant_is_read_only_where_its_meaning_is_decided():
    """The same single-decider rule `Principal.may_assert` follows. A second
    reader of `content_no_store` is a second opinion about who may opt out."""
    allowed = {Path("config.py"), Path("lifecycle.py"), Path("management.py"),
               Path("no_store.py")}
    readers = set()
    for path in PKG.rglob("*.py"):
        tree = ast.parse(path.read_text())
        if any(isinstance(n, ast.Attribute) and n.attr == "content_no_store"
               for n in ast.walk(tree)):
            readers.add(path.relative_to(PKG))
    assert readers, "nothing reads the grant — this guard measures nothing"
    assert readers <= allowed, f"unexpected readers: {readers - allowed}"


def test_no_log_call_formats_a_raw_backend_exception_without_redaction():
    """The sites this change redacted, pinned by source: an `exc` passed raw to
    a logger in the dispatch paths (a backend error body can quote the prompt).
    A NEW such site would be caught by the next reader of this list, not by a
    test that drives it — so name them."""
    redacted_sites = {
        ("lifecycle.py", "dispatch %s failed"),
        ("lifecycle.py", "engine 500 — retrying once"),
        ("lifecycle.py", "transient backend error on %s"),
        ("correction.py", "degeneration re-dispatch %d failed"),
        ("correction.py", "schema-backstop retry dispatch failed"),
        ("correction.py", "schema-backstop UNRECOVERABLE"),
        ("correction.py", "schema-backstop WOULD retry+fail"),
        ("correction.py", '"ROADSTEAD_STRUCTURED_EMPTY '),
        ("correction.py", "GRAMMAR INVALID"),
    }
    found = set()
    for fname in ("lifecycle.py", "correction.py"):
        src = (PKG / fname).read_text()
        for _, marker in [s for s in redacted_sites if s[0] == fname]:
            i = src.index(marker)
            window = src[i: i + 900]
            assert "redact(" in window, f"{fname}: {marker!r} is not redacted"
            found.add((fname, marker))
    assert found == redacted_sites
