"""Fast direct units for surfaces that previously had only indirect/e2e or
heavy-tier coverage (audit 2026-07-02, dimension I): ProxyState construction
invariants, model_catalog resolution behavior, the cache-drift-alarm
kill-switch guard-bite, and the audit's new DRR-ghost / durable-shadow
persistence behaviors (queue KV + context-overflow aggregates + budget
delete/starving semantics).
"""
from __future__ import annotations

import time

from roadstead import model_catalog as mc
from roadstead.agent_budget import BudgetManager
from roadstead.config import ProxyConfig, cache_drift_alarm_enabled
from roadstead.queue import PersistentQueue
from roadstead.state import ProxyState


# ---- ProxyState construction invariants (was: only exercised by a heavy e2e) --

def test_proxystate_initial_invariants():
    st = ProxyState(ProxyConfig())
    # Observability containers exist and start empty — http_handlers reads
    # them unconditionally on /v1/status and /metrics.
    assert st.alerts == [] and st.alert_logged == set()
    assert st.context_overflows == {}
    assert st.cache_drift_alerted == {} and st.cache_drift_current == []
    assert st.admin_ips_seen == {}
    # Correction counters start at zero (frozen white-box surface).
    assert st.degeneration_detected == 0 and st.degeneration_recovered == 0


# ---- model_catalog resolution behavior (was: naming-doctrine only) -----------

def test_catalog_resolution_canonical_alias_unknown():
    cat = mc.load_catalog()
    eps = cat.proxy_endpoints()
    assert eps, "models.yaml must define proxy endpoints"
    e0 = eps[0]
    # A canonical name resolves to itself; its aliases resolve to it.
    assert cat.canonical(e0.name) == e0.name
    for alias in e0.aliases:
        assert cat.canonical(alias) == e0.name, alias
    # entry() follows the same resolution; unknown names resolve to None.
    assert cat.entry(e0.name) is e0
    assert cat.canonical("no-such-model-xyz") is None
    assert cat.entry("no-such-model-xyz") is None
    # host_ip maps through the hosts table for every proxy endpoint.
    for e in eps:
        assert cat.host_ip(e.name), f"{e.name} has no resolvable host ip"


def test_coerce_entry_retired_key_sets_status():
    # F-2 (audit 2026-07-12): a stanza carrying a `retired:` key is retired even
    # without an explicit `status:` — it must NOT silently default to "active".
    e = mc._coerce_entry("old-model", {"kind": "chat", "retired": "2026-07-11"})
    assert e.status == "retired"
    # No retired key → the historic "active" default is unchanged.
    e2 = mc._coerce_entry("live-model", {"kind": "chat"})
    assert e2.status == "active"
    # An explicit status still wins over the retired-key inference.
    e3 = mc._coerce_entry("odd", {"kind": "chat", "retired": "2026-01-01",
                                  "status": "on_demand"})
    assert e3.status == "on_demand"


def test_endpoint_kwargs_derive_from_catalog():
    kwargs = mc.build_endpoint_kwargs()
    assert kwargs, "derived endpoint kwargs must not be empty"
    for ep_name, kw in kwargs.items():
        assert kw.get("host"), (ep_name, kw)
        assert isinstance(kw.get("max_slots"), int) and kw["max_slots"] >= 1
        assert isinstance(kw.get("context_per_slot"), int) and kw["context_per_slot"] > 0


# ---- cache-drift alarm kill-switch (guard-bite) ------------------------------

def test_cache_drift_alarm_kill_switch(monkeypatch):
    monkeypatch.delenv("COLLECTIVE_PROXY_CACHE_DRIFT_ALARM", raising=False)
    assert cache_drift_alarm_enabled() is True  # default ON (observability-only)
    monkeypatch.setenv("COLLECTIVE_PROXY_CACHE_DRIFT_ALARM", "0")
    assert cache_drift_alarm_enabled() is False
    monkeypatch.setenv("COLLECTIVE_PROXY_CACHE_DRIFT_ALARM", "off")
    assert cache_drift_alarm_enabled() is False


# ---- durable shadow evidence (audit 2026-07-02, C1) --------------------------

def test_queue_kv_and_context_overflow_roundtrip(tmp_path):
    q = PersistentQueue(tmp_path / "q.db")
    try:
        # KV blob round-trip (cache-drift dedup map shape: [[cs, ep, ts], ...]).
        q.kv_set("cache_drift_alerted", [["a.b", "creative", 123.0]])
        assert q.kv_get("cache_drift_alerted") == [["a.b", "creative", 123.0]]
        assert q.kv_get("missing", "dflt") == "dflt"
        # Context-overflow aggregate: upsert accumulates count + max_est_in.
        q.record_context_overflow("chat", "x/y/z", 20_000)
        q.record_context_overflow("chat", "x/y/z", 25_000)
        q.record_context_overflow("thinker", "other", 90_000)
        q.flush()
        shadow = q.load_context_overflows()
        assert shadow["chat"]["count"] == 2
        assert shadow["chat"]["callers"] == {"x/y/z": 2}
        assert shadow["chat"]["max_est_in"] == 25_000
        assert shadow["thinker"]["count"] == 1
    finally:
        q.close()


# ---- DRR hygiene (audit 2026-07-02, C6) --------------------------------------

def test_budget_delete_and_prune_roundtrip(tmp_path):
    q = PersistentQueue(tmp_path / "b.db")
    try:
        q.save_budgets([
            {"agent_id": "ghost", "weight": 1.0, "balance_ss": 60.0,
             "total_consumed_ss": 5.0},
            {"agent_id": "debtor", "weight": 1.0, "balance_ss": -12.0,
             "total_consumed_ss": 100.0},
        ])
        q.flush()
        assert {r["agent_id"] for r in q.load_budgets()} == {"ghost", "debtor"}
        q.delete_budgets(["ghost"])
        q.flush()
        assert {r["agent_id"] for r in q.load_budgets()} == {"debtor"}
    finally:
        q.close()


def test_starving_reflects_denied_service_not_balance_sign():
    mgr = BudgetManager(starvation_timeout_s=30.0)
    mgr.set_total_capacity(10.0)
    heavy = mgr.get_or_create("heavy")
    light = mgr.get_or_create("light")
    now = time.monotonic()
    # Heavy consumer: deeply negative balance but being SERVED (short wait).
    heavy.balance = -500.0
    heavy.negative_since = now - 3600.0
    # pick_agent records the observed head-of-queue waits.
    mgr.pick_agent(["heavy", "light"], now,
                   wait_by_agent={"heavy": 0.5, "light": 45.0})
    snap = {b["agent_id"]: b for b in mgr.snapshot()}
    assert snap["heavy"]["starving"] is False  # negative ≠ starving
    assert snap["light"]["starving"] is True   # denied service ≥ timeout
    _ = light  # silence unused warning
