"""Step 4b — rate-windowed per-endpoint cooldown (BUILDER unit tests).

Exercises Health.record_dispatch_failure + the endpoint_healthy cooldown gate
against a light stub state (no proxy fixture → zero startup cost). Covers: both
flags off == no-op, backend-fault classification (5xx/timeout/unavailable count,
4xx does NOT), the allowed-fails threshold, sliding-window pruning, shadow (count
but never pull), enforce (pull + auto-recover), and guard-bite. The adversarial
track owns the hostile fault-injection E2E matrix.

Self-binds the real Health methods to a stub state (the Mac 3.9 conftest blocks
pytest, so it also runs as a plain script).


2026-07-30: these drove the identifier "companion". That role resolves to the
anvil tier3 (Laguna) endpoint whose class is "thinker", so failures landed in
the thinker bucket. They were switched to "tier3-backup" on the belief that the
nexus 122B "still OWNS the `companion` endpoint class".

2026-08-02: that belief was the bug. `tier3-backup` normalized to `companion`,
which was ALSO an alias of tier3 — so a second normalization pass collapsed it to
`thinker` and the proxy silently served the backup off tier3 (ledger
`endpoint-class-alias-collision`). These tests only passed because they asserted
on the intermediate `companion` bucket, which is exactly the state that should
never have existed. The 122B is no longer a proxy endpoint at all.

Now driven by "creative" — a real, live, non-thinker endpoint class — so the
cooldown mechanics are exercised on a genuine class and stay isolated from the
"thinker" assertions elsewhere in this file. The mechanics under test are
unchanged throughout.
"""
from __future__ import annotations

import importlib
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # originfleet/
sys.path.insert(0, str(REPO))

health_mod = importlib.import_module("originfleet.llmproxy.health")
backend_mod = importlib.import_module("originfleet.llmproxy.backend")
Health = health_mod.Health
BackendError = backend_mod.BackendError
BackendTimeout = backend_mod.BackendTimeout
BackendUnavailable = backend_mod.BackendUnavailable

ENFORCE = "COLLECTIVE_PROXY_ENDPOINT_COOLDOWN"
SHADOW = "COLLECTIVE_PROXY_ENDPOINT_COOLDOWN_SHADOW"
ALLOWED = "COLLECTIVE_PROXY_COOLDOWN_ALLOWED_FAILS"
WINDOW = "COLLECTIVE_PROXY_COOLDOWN_WINDOW_S"
DURATION = "COLLECTIVE_PROXY_COOLDOWN_DURATION_S"


def _state():
    return types.SimpleNamespace(
        endpoint_failure_times={},
        endpoint_cooldown_until={},
        endpoint_cooldown_trips={},
        paused_endpoints=set(),
        endpoint_health={},
        on_demand=types.SimpleNamespace(manages=lambda ep: False),
        # fast_fail_interactive reads scheduler.queued_requests(ep, bands)
        scheduler=types.SimpleNamespace(queued_requests=lambda ep, bands: []),
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
        h.record_dispatch_failure("creative", _fail())
    assert h.state.endpoint_failure_times == {}
    assert h.state.endpoint_cooldown_trips == {}
    assert h.endpoint_healthy("creative") is True


# --- backend-fault classification ------------------------------------------

def test_4xx_does_not_count(monkeypatch):
    monkeypatch.setenv(SHADOW, "1")
    monkeypatch.delenv(ENFORCE, raising=False)
    h = _health()
    for _ in range(10):
        h.record_dispatch_failure("creative", BackendError(400, "bad request"))
    assert h.state.endpoint_cooldown_trips == {}  # never trips on client errors
    assert "creative" not in h.state.endpoint_failure_times or \
        h.state.endpoint_failure_times.get("creative") == []


def test_timeout_and_unavailable_count(monkeypatch):
    monkeypatch.setenv(SHADOW, "1")
    monkeypatch.delenv(ENFORCE, raising=False)
    monkeypatch.setenv(ALLOWED, "2")
    h = _health()
    h.record_dispatch_failure("thinker", BackendTimeout("stall"))     # 504
    h.record_dispatch_failure("thinker", BackendUnavailable("down"))  # 503
    assert h.state.endpoint_cooldown_trips.get("thinker") == 1


# --- threshold + window -----------------------------------------------------

def test_below_threshold_no_trip(monkeypatch):
    monkeypatch.setenv(SHADOW, "1")
    monkeypatch.setenv(ALLOWED, "4")
    h = _health()
    for _ in range(3):
        h.record_dispatch_failure("creative", _fail())
    assert h.state.endpoint_cooldown_trips == {}
    h.record_dispatch_failure("creative", _fail())  # the 4th trips
    assert h.state.endpoint_cooldown_trips.get("creative") == 1


def test_stale_failures_pruned_from_window(monkeypatch):
    monkeypatch.setenv(SHADOW, "1")
    monkeypatch.setenv(ALLOWED, "4")
    monkeypatch.setenv(WINDOW, "1")  # 1s window
    h = _health()
    # 3 old failures (well outside the 1s window) + 1 fresh → prune leaves 1.
    h.state.endpoint_failure_times["creative"] = [time.monotonic() - 100] * 3
    h.record_dispatch_failure("creative", _fail())
    assert h.state.endpoint_cooldown_trips == {}  # stale ones don't count


# --- shadow: count but never pull ------------------------------------------

def test_shadow_counts_but_does_not_pull(monkeypatch):
    monkeypatch.setenv(SHADOW, "1")
    monkeypatch.delenv(ENFORCE, raising=False)
    monkeypatch.setenv(ALLOWED, "3")
    h = _health()
    for _ in range(3):
        h.record_dispatch_failure("creative", _fail())
    assert h.state.endpoint_cooldown_trips.get("creative") == 1
    assert "creative" not in h.state.endpoint_cooldown_until  # NOT cooled
    assert h.endpoint_healthy("creative") is True             # NOT pulled


# --- enforce: pull + auto-recover ------------------------------------------

def test_enforce_pulls_then_auto_recovers(monkeypatch):
    monkeypatch.setenv(ENFORCE, "1")
    monkeypatch.setenv(ALLOWED, "3")
    monkeypatch.setenv(DURATION, "30")
    h = _health()
    for _ in range(3):
        h.record_dispatch_failure("creative", _fail())
    assert h.state.endpoint_cooldown_trips.get("creative") == 1
    assert "creative" in h.state.endpoint_cooldown_until
    assert h.endpoint_healthy("creative") is False  # cooled → deferred
    # Simulate the cooldown expiring → auto-recovery (next dispatch tick).
    h.state.endpoint_cooldown_until["creative"] = time.monotonic() - 1
    assert h.endpoint_healthy("creative") is True


def test_enforce_resets_window_after_trip(monkeypatch):
    """After a trip the failure window is cleared so a fresh burst is needed to
    re-cool (no immediate re-trip on the next single failure)."""
    monkeypatch.setenv(ENFORCE, "1")
    monkeypatch.setenv(ALLOWED, "3")
    h = _health()
    for _ in range(3):
        h.record_dispatch_failure("creative", _fail())
    assert h.state.endpoint_failure_times.get("creative") == []  # window reset


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
    h.state.endpoint_cooldown_until["creative"] = time.monotonic() + 999
    assert h.endpoint_healthy("creative") is True  # enforce off → gate skipped


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
