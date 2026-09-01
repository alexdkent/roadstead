"""Step 4c — vLLM max_slots shadow-drift reconciler.

Drives the real Health.evaluate_alerts over a started proxy. One proxy spin,
all cases (matching == silent, drift == warns, flag-off == silent, llama.cpp
never warns, guard-bite) — kept to a single fixture start for the tollgate budget.

The reconciler is SHADOW: it only appends a `max_slots_drift` WARNING to
state.alerts (surfaced on /v1/status); it never changes admission.
"""
from __future__ import annotations

import time


def _drift_alerts(state):
    return [a for a in state.alerts if a.get("name") == "max_slots_drift"]


async def test_max_slots_drift_reconciler(proxy, monkeypatch):
    monkeypatch.delenv("ROADSTEAD_PROXY_MAX_SLOTS_RECONCILE", raising=False)  # default ON
    health = proxy.svc._health
    state = proxy.svc._correction.state
    eps = state.config.endpoints

    # 1) Seeded config matches documented launch values → NO drift alert.
    health.evaluate_alerts(time.monotonic())
    assert _drift_alerts(state) == [], "false drift with matching values"

    # 2) Introduce a drift on the vLLM tier3 (the 32-vs-20 class): admitted
    #    max_slots diverges from documented --max-num-seqs → WARNING fires, naming
    #    both numbers. Admission is NOT touched (shadow).
    tier3 = eps["tier3"]
    # 20 = the Tier-3 backend's real --max-num-seqs (serve_tier3_prod.sh). History: 20 under the
    # retired dense Qwen3.6-27B -> 16 on 2026-07-30 when tier3 became Laguna S 2.1 INT4 -> back to
    # 20 on 2026-07-31 (operator request; the 26 GiB KV pool is sized by --kv-cache-memory and is
    # unchanged, so this only widens the scheduler's concurrency cap).
    # Pinned deliberately rather than read from config: this seed is the ONLY source of the
    # proxy's admission ceiling (vLLM does not expose --max-num-seqs over its API), so a silent
    # edit to models.yaml should break a test, not drift unnoticed into over-admission.
    assert tier3.documented_max_num_seqs == 6   # V4-Flash TP=2 profile (was 20 on Laguna)
    admitted_before = tier3.max_slots
    tier3.max_slots = 32
    health.evaluate_alerts(time.monotonic())
    drift = _drift_alerts(state)
    assert len(drift) == 1, f"drift not detected: {state.alerts}"
    assert drift[0]["severity"] == "WARNING"
    assert "tier3" in drift[0]["detail"] and "32" in drift[0]["detail"] \
        and "6" in drift[0]["detail"]
    # shadow: the reconciler never rewrote admission
    assert tier3.max_slots == 32

    # 3) Guard-bite via the kill-switch: flag OFF → the drift is NOT reported even
    #    though it still exists (proves the check is what emits the alert).
    monkeypatch.setenv("ROADSTEAD_PROXY_MAX_SLOTS_RECONCILE", "0")
    health.evaluate_alerts(time.monotonic())
    assert _drift_alerts(state) == [], "drift alerted while kill-switch OFF"
    monkeypatch.setenv("ROADSTEAD_PROXY_MAX_SLOTS_RECONCILE", "1")

    # 4) A llama.cpp endpoint (documented_max_num_seqs == 0) is NEVER reconciled,
    #    even if its max_slots is odd — the guard is vLLM-only.
    # After the 2026-07-11 boxa consolidation the old classify/chat class is an
    # alias of the `tier2` endpoint, which DOES carry documented_max_num_seqs.
    # `tier3` (nexus llama.cpp, documented == 0) used to be the example here,
    # but that class was removed 2026-08-02 — its name collided with the
    # `tier3` ALIAS of tier3 (ledger `endpoint-class-alias-collision`).
    # `rerank` is now the live llama.cpp endpoint with documented == 0.
    tier3.max_slots = admitted_before  # restore the drift
    llamacpp_ep = eps["rerank"]
    assert llamacpp_ep.documented_max_num_seqs == 0
    llamacpp_ep.max_slots = 999
    health.evaluate_alerts(time.monotonic())
    assert _drift_alerts(state) == [], "llama.cpp endpoint should never drift-alert"
