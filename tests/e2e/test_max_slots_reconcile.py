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
    monkeypatch.delenv("COLLECTIVE_PROXY_MAX_SLOTS_RECONCILE", raising=False)  # default ON
    health = proxy.svc._health
    state = proxy.svc._correction.state
    eps = state.config.endpoints

    # 1) Seeded config matches documented launch values → NO drift alert.
    health.evaluate_alerts(time.monotonic())
    assert _drift_alerts(state) == [], "false drift with matching values"

    # 2) Introduce a drift on the vLLM thinker (the 32-vs-20 class): admitted
    #    max_slots diverges from documented --max-num-seqs → WARNING fires, naming
    #    both numbers. Admission is NOT touched (shadow).
    thinker = eps["thinker"]
    assert thinker.documented_max_num_seqs == 20
    admitted_before = thinker.max_slots
    thinker.max_slots = 32
    health.evaluate_alerts(time.monotonic())
    drift = _drift_alerts(state)
    assert len(drift) == 1, f"drift not detected: {state.alerts}"
    assert drift[0]["severity"] == "WARNING"
    assert "thinker" in drift[0]["detail"] and "32" in drift[0]["detail"] \
        and "20" in drift[0]["detail"]
    # shadow: the reconciler never rewrote admission
    assert thinker.max_slots == 32

    # 3) Guard-bite via the kill-switch: flag OFF → the drift is NOT reported even
    #    though it still exists (proves the check is what emits the alert).
    monkeypatch.setenv("COLLECTIVE_PROXY_MAX_SLOTS_RECONCILE", "0")
    health.evaluate_alerts(time.monotonic())
    assert _drift_alerts(state) == [], "drift alerted while kill-switch OFF"
    monkeypatch.setenv("COLLECTIVE_PROXY_MAX_SLOTS_RECONCILE", "1")

    # 4) A llama.cpp endpoint (documented_max_num_seqs == 0) is NEVER reconciled,
    #    even if its max_slots is odd — the guard is vLLM-only.
    thinker.max_slots = admitted_before  # restore the drift
    chat = eps["chat"]
    assert chat.documented_max_num_seqs == 0
    chat.max_slots = 999
    health.evaluate_alerts(time.monotonic())
    assert _drift_alerts(state) == [], "llama.cpp endpoint should never drift-alert"
