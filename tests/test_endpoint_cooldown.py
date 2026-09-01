"""Step 4b — rate-windowed per-endpoint cooldown (BUILDER unit tests).

Exercises Health.record_dispatch_failure + the endpoint_healthy cooldown gate
against a light stub state (no proxy fixture → zero startup cost). Covers: both
flags off == no-op, backend-fault classification (5xx/timeout/unavailable count,
4xx does NOT), the allowed-fails threshold, sliding-window pruning, shadow (count
but never pull), enforce (pull + auto-recover), and guard-bite. The adversarial
track owns the hostile fault-injection E2E matrix.

Self-binds the real Health methods to a stub state (the Mac 3.9 conftest blocks
pytest, so it also runs as a plain script).


2026-07-30: these drove the identifier "tier3". That role resolves to the
anvil tier3 (Laguna) endpoint whose class is "tier3", so failures landed in
the tier3 bucket. They were switched to "tier3-backup" on the belief that the
nexus 122B "still OWNS the `tier3` endpoint class".

2026-08-02: that belief was the bug. `tier3-backup` normalized to `tier3`,
which was ALSO an alias of tier3 — so a second normalization pass collapsed it to
`tier3` and the proxy silently served the backup off tier3 (ledger
`endpoint-class-alias-collision`). These tests only passed because they asserted
on the intermediate `tier3` bucket, which is exactly the state that should
never have existed. The 122B is no longer a proxy endpoint at all.

Now driven by "tier2" — a real, live, non-thinker endpoint class — so the
cooldown mechanics are exercised on a genuine class and stay isolated from the
"tier3" assertions elsewhere in this file. The mechanics under test are
unchanged throughout.
"""
from __future__ import annotations

import importlib
import sys
import time
import types
from pathlib import Path

health_mod = importlib.import_module("roadstead.health")
backend_mod = importlib.import_module("roadstead.backend")
# ``lifecycle.py`` is read as SOURCE by the structural sweep below. The path
# used to be spelled out relative to a monorepo checkout, reaching into the host
# application's tree — the only reason this file could not run standalone. It is
# derived from the imported module now, so it cannot go stale again and cannot
# silently read the wrong tree.
LIFECYCLE_SRC = Path(importlib.import_module("roadstead.lifecycle").__file__)

Health = health_mod.Health
BackendError = backend_mod.BackendError
BackendTimeout = backend_mod.BackendTimeout
BackendUnavailable = backend_mod.BackendUnavailable

ENFORCE = "ROADSTEAD_PROXY_ENDPOINT_COOLDOWN"
SHADOW = "ROADSTEAD_PROXY_ENDPOINT_COOLDOWN_SHADOW"
ALLOWED = "ROADSTEAD_PROXY_COOLDOWN_ALLOWED_FAILS"
WINDOW = "ROADSTEAD_PROXY_COOLDOWN_WINDOW_S"
DURATION = "ROADSTEAD_PROXY_COOLDOWN_DURATION_S"


def _state():
    return types.SimpleNamespace(
        endpoint_failure_times={},
        endpoint_cooldown_until={},
        endpoint_cooldown_trips={},
        cooldown_best_effort_skips={},
        paused_endpoints=set(),
        endpoint_health={},
        on_demand=types.SimpleNamespace(manages=lambda ep: False),
        # fast_fail_interactive reads scheduler.queued_requests(ep, bands)
        scheduler=types.SimpleNamespace(queued_requests=lambda ep, bands: []),
        # § 9 failover: None here means "no failover armed", which is what this
        # suite wants — it exercises the COOLDOWN mechanics, and a live failover
        # would reroute the very requests it asserts are released. The real
        # ProxyState always has a Failover; a stub declaring it None is an
        # explicit choice, not an accident of the stub being thin.
        failover=None,
    )


def _health():
    return Health(_state())


def _fail(n=1, status=500):
    return BackendError(status, "boom")


def _all_off(mp):
    mp.delenv(ENFORCE, raising=False)
    mp.delenv(SHADOW, raising=False)


# --- flags off == byte-identical -------------------------------------------

def test_both_flags_off_is_noop(monkeypatch):
    _all_off(monkeypatch)
    h = _health()
    for _ in range(10):
        h.record_dispatch_failure("tier2", _fail())
    assert h.state.endpoint_failure_times == {}
    assert h.state.endpoint_cooldown_trips == {}
    assert h.endpoint_healthy("tier2") is True


# --- backend-fault classification ------------------------------------------

def test_4xx_does_not_count(monkeypatch):
    monkeypatch.setenv(SHADOW, "1")
    monkeypatch.delenv(ENFORCE, raising=False)
    h = _health()
    for _ in range(10):
        h.record_dispatch_failure("tier2", BackendError(400, "bad request"))
    assert h.state.endpoint_cooldown_trips == {}  # never trips on client errors
    assert "tier2" not in h.state.endpoint_failure_times or \
        h.state.endpoint_failure_times.get("tier2") == []


def test_timeout_and_unavailable_count(monkeypatch):
    monkeypatch.setenv(SHADOW, "1")
    monkeypatch.delenv(ENFORCE, raising=False)
    monkeypatch.setenv(ALLOWED, "2")
    h = _health()
    h.record_dispatch_failure("tier3", BackendTimeout("stall"))     # 504
    h.record_dispatch_failure("tier3", BackendUnavailable("down"))  # 503
    assert h.state.endpoint_cooldown_trips.get("tier3") == 1


# --- threshold + window -----------------------------------------------------

def test_below_threshold_no_trip(monkeypatch):
    monkeypatch.setenv(SHADOW, "1")
    monkeypatch.setenv(ALLOWED, "4")
    h = _health()
    for _ in range(3):
        h.record_dispatch_failure("tier2", _fail())
    assert h.state.endpoint_cooldown_trips == {}
    h.record_dispatch_failure("tier2", _fail())  # the 4th trips
    assert h.state.endpoint_cooldown_trips.get("tier2") == 1


def test_stale_failures_pruned_from_window(monkeypatch):
    monkeypatch.setenv(SHADOW, "1")
    monkeypatch.setenv(ALLOWED, "4")
    monkeypatch.setenv(WINDOW, "1")  # 1s window
    h = _health()
    # 3 old failures (well outside the 1s window) + 1 fresh → prune leaves 1.
    h.state.endpoint_failure_times["tier2"] = [time.monotonic() - 100] * 3
    h.record_dispatch_failure("tier2", _fail())
    assert h.state.endpoint_cooldown_trips == {}  # stale ones don't count


# --- shadow: count but never pull ------------------------------------------

def test_shadow_counts_but_does_not_pull(monkeypatch):
    monkeypatch.setenv(SHADOW, "1")
    monkeypatch.delenv(ENFORCE, raising=False)
    monkeypatch.setenv(ALLOWED, "3")
    h = _health()
    for _ in range(3):
        h.record_dispatch_failure("tier2", _fail())
    assert h.state.endpoint_cooldown_trips.get("tier2") == 1
    assert "tier2" not in h.state.endpoint_cooldown_until  # NOT cooled
    assert h.endpoint_healthy("tier2") is True             # NOT pulled


# --- enforce: pull + auto-recover ------------------------------------------

def test_enforce_pulls_then_auto_recovers(monkeypatch):
    monkeypatch.setenv(ENFORCE, "1")
    monkeypatch.setenv(ALLOWED, "3")
    monkeypatch.setenv(DURATION, "30")
    h = _health()
    for _ in range(3):
        h.record_dispatch_failure("tier2", _fail())
    assert h.state.endpoint_cooldown_trips.get("tier2") == 1
    assert "tier2" in h.state.endpoint_cooldown_until
    assert h.endpoint_healthy("tier2") is False  # cooled → deferred
    # Simulate the cooldown expiring → auto-recovery (next dispatch tick).
    h.state.endpoint_cooldown_until["tier2"] = time.monotonic() - 1
    assert h.endpoint_healthy("tier2") is True


def test_enforce_resets_window_after_trip(monkeypatch):
    """After a trip the failure window is cleared so a fresh burst is needed to
    re-cool (no immediate re-trip on the next single failure)."""
    monkeypatch.setenv(ENFORCE, "1")
    monkeypatch.setenv(ALLOWED, "3")
    h = _health()
    for _ in range(3):
        h.record_dispatch_failure("tier2", _fail())
    assert h.state.endpoint_failure_times.get("tier2") == []  # window reset


# --- guard-bite -------------------------------------------------------------

def test_guard_bite_enforce_off_never_pulls(monkeypatch):
    """With enforce OFF, even a fully-tripped (shadow) endpoint stays healthy —
    proving the endpoint_healthy cooldown gate is what pulls it. Revert that gate
    and this test fails."""
    monkeypatch.setenv(SHADOW, "1")
    monkeypatch.delenv(ENFORCE, raising=False)
    monkeypatch.setenv(ALLOWED, "2")
    h = _health()
    # Force a would-be-active cooldown window directly, then confirm OFF ignores it.
    h.state.endpoint_cooldown_until["tier2"] = time.monotonic() + 999
    assert h.endpoint_healthy("tier2") is True  # enforce off → gate skipped


if __name__ == "__main__":  # plain-script mode (Mac 3.9)
    import os

    class _MP:
        def setenv(self, k, v): os.environ[k] = v
        def delenv(self, k, raising=True): os.environ.pop(k, None)

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        # each test manages its own env; reset the cooldown envs between tests
        for k in (ENFORCE, SHADOW, ALLOWED, WINDOW, DURATION):
            os.environ.pop(k, None)
        fn(_MP())
        passed += 1
        print(f"ok {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")


# --- best-effort sub-floor timeouts must NOT cool the endpoint --------------
# Regression, 2026-08-20. `orchestrator.gemma_greeter_advisory` hands the backend
# a deadline far under the size-aware recommended time. When that deadline fired
# it was counted as a backend fault, and 4 of them inside 60s cooled the WHOLE
# tier1 endpoint for 30s — fast-failing unrelated P1 Discord turn-path callers
# with 503 circuit_open. 143 such timeouts were logged, 91 on 2026-08-20 alone,
# tripping 8 endpoint-wide cooldowns in one day. The identical
# `_timeout_below_recommended` predicate already gated the endpoint_stalled
# heuristic since 2026-07-05; it was simply never wired into the cooldown.

def test_best_effort_timeout_never_trips_cooldown(monkeypatch):
    monkeypatch.setenv(SHADOW, "1")
    monkeypatch.setenv(ALLOWED, "2")
    h = _health()
    for _ in range(20):
        h.record_dispatch_failure("tier1", BackendTimeout("gave up"), best_effort=True)
    assert h.state.endpoint_cooldown_trips == {}
    assert h.state.endpoint_failure_times.get("tier1", []) == []


def test_best_effort_skip_is_counted_so_the_guard_is_observable(monkeypatch):
    monkeypatch.setenv(SHADOW, "1")
    h = _health()
    for _ in range(3):
        h.record_dispatch_failure("tier1", BackendTimeout("gave up"), best_effort=True)
    # A guard nobody can see firing is indistinguishable from one that is dead.
    assert h.state.cooldown_best_effort_skips.get("tier1") == 3


def test_genuine_faults_still_cool_the_same_endpoint(monkeypatch):
    """ANTI-VACUITY. The exclusion must narrow the cooldown, not delete it —
    a version that simply stopped counting timeouts would pass the test above."""
    monkeypatch.setenv(SHADOW, "1")
    monkeypatch.setenv(ALLOWED, "2")
    h = _health()
    h.record_dispatch_failure("tier1", BackendTimeout("gave up"), best_effort=True)
    assert h.state.endpoint_cooldown_trips == {}          # excluded
    h.record_dispatch_failure("tier1", BackendTimeout("real stall"))
    h.record_dispatch_failure("tier1", BackendTimeout("real stall"))
    assert h.state.endpoint_cooldown_trips.get("tier1") == 1   # still cools


def test_best_effort_defaults_false_so_existing_callers_are_unchanged(monkeypatch):
    """ANTI-VACUITY. The keyword must be opt-in: an un-updated call site keeps
    counting exactly as before."""
    monkeypatch.setenv(SHADOW, "1")
    monkeypatch.setenv(ALLOWED, "2")
    h = _health()
    h.record_dispatch_failure("tier2", BackendTimeout("stall"))
    h.record_dispatch_failure("tier2", BackendTimeout("stall"))
    assert h.state.endpoint_cooldown_trips.get("tier2") == 1
    assert h.state.cooldown_best_effort_skips == {}


def test_best_effort_4xx_is_classified_out_before_the_skip_tally(monkeypatch):
    """A caller error is not a best-effort give-up; it must not inflate the
    skip counter, or the counter stops meaning what /v1/status says it means."""
    monkeypatch.setenv(SHADOW, "1")
    h = _health()
    h.record_dispatch_failure("tier1", BackendError(400, "bad"), best_effort=True)
    assert h.state.cooldown_best_effort_skips == {}
    assert h.state.endpoint_cooldown_trips == {}


def test_both_flags_off_still_noop_for_best_effort(monkeypatch):
    _all_off(monkeypatch)
    h = _health()
    h.record_dispatch_failure("tier1", BackendTimeout("x"), best_effort=True)
    assert h.state.cooldown_best_effort_skips == {}


# --- the WIRING, not just the mechanism ------------------------------------
# The bug was never in the predicate — it was that the predicate was not passed
# at the timeout call sites. Assert that statically so a future edit that drops
# the keyword fails here instead of silently restoring the outage.

def test_every_backend_timeout_handler_passes_best_effort():
    import ast
    src = LIFECYCLE_SRC.read_text()
    tree = ast.parse(src)
    offenders, checked = [], 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        names = set()
        t = node.type
        for n in (t.elts if isinstance(t, ast.Tuple) else [t] if t else []):
            if isinstance(n, ast.Name):
                names.add(n.id)
        if "BackendTimeout" not in names:
            continue
        for call in ast.walk(node):
            if (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "record_dispatch_failure"):
                checked += 1
                if not any(kw.arg == "best_effort" for kw in call.keywords):
                    offenders.append(node.lineno)
    assert checked >= 2, (
        f"expected >=2 record_dispatch_failure calls inside BackendTimeout "
        f"handlers, found {checked} — the sweep has gone blind, not green")
    assert not offenders, (
        f"record_dispatch_failure inside a BackendTimeout handler at line(s) "
        f"{offenders} omits best_effort= — a sub-floor caller give-up will cool "
        f"the endpoint for everyone again")
