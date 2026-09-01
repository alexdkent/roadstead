"""The rate ledger's own memory, and the clock it is pruned on.

``RateLedger.prune`` shipped with Workstream I carrying a docstring that said
"Called from the maintenance tick". Nothing called it. Both dicts behind the
rate threshold are keyed by ``agent_id``, which is a CALLER-SUPPLIED string on
the address path (``docs/api.md`` §1.5 — an address only fills in an
``agent_id`` the body omitted), so they grew one entry per distinct name ever
seen, without bound, under a caller's control.

🚨 That is worse here than it looks, because of what this threshold is. A rate
threshold that DEGRADES rather than rejects is deliberately not a defence
against a runaway caller — which is exactly why its own bookkeeping must not
become one.

Every guard below was observed going red by mutating the code it guards.
"""
from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

from roadstead import health as health_mod
from roadstead.config import ProxyConfig
from roadstead.rate import WINDOW_S, RateLedger
from roadstead.service import ProxyService


@pytest.fixture
def proxy_state(tmp_path):
    """The real `ProxyState`, off a real `ProxyService`.

    Built rather than faked because the two dicts under test live on it and the
    method under test coordinates them — a stand-in would be asserting that this
    file's own model of the state is consistent with itself.
    """
    return ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db")))._state


def test_prune_drops_only_callers_whose_window_has_emptied():
    led = RateLedger()
    now = 1_000_000.0
    led.record("quiet", now)
    led.record("busy", now)
    # `quiet` said nothing more; `busy` kept going.
    later = now + WINDOW_S + 1
    led.record("busy", later)

    assert led.prune(later) == 1
    assert set(led.windows) == {"busy"}
    # And a caller still inside its window is never dropped.
    assert led.prune(later) == 0


def test_the_ledger_does_not_grow_without_bound(proxy_state):
    """The bug, stated as the thing a soak would eventually have shown.

    Mutation: drop the `prune_rate_state` call from the poller tick. The dict
    keeps every name, which is the pre-fix behaviour.
    """
    st = proxy_state
    base = time.time()
    for i in range(500):
        st.record_request(f"caller-{i}")
    assert len(st.rate.windows) == 500

    # Wind every window past the horizon by pruning at a later wall time.
    dropped = _prune_at(st, base + WINDOW_S + 1)
    assert dropped == 500
    assert st.rate.windows == {}


def test_forgetting_a_caller_forgets_that_we_noted_its_demotion(proxy_state):
    """The second dict, and the one a reader forgets exists.

    Mutation: prune only `rate.windows`. `rate_demotion_noted` keeps every name
    forever — a slower leak through the same caller-supplied key, and one that
    would also suppress a legitimate second notice.
    """
    st = proxy_state
    st.record_request("noisy")
    st.rate_demotion_noted["noisy"] = 20260901
    st.rate_demotion_noted["long-gone"] = 20260801

    _prune_at(st, time.time() + WINDOW_S + 1)
    assert st.rate.windows == {}
    assert st.rate_demotion_noted == {}


def test_a_live_caller_keeps_its_demotion_note(proxy_state):
    """The other half: pruning must not re-arm a notice for a caller that is
    still going, or the operator gets the same line every poller tick.

    🚨 A stale caller AND a live one, deliberately. This first SURVIVED the
    mutation `rate_demotion_noted.clear()`, because the version that asserted
    only `prune_rate_state() == 0` never got past the `if dropped:` guard — it
    proved that a prune which does nothing changes nothing, which is not the
    claim. A sweep of both dicts is only observable when something is actually
    being swept.
    """
    st = proxy_state
    now = time.time()
    st.rate.record("gone", now - WINDOW_S - 10)      # already outside the window
    st.record_request("still-here")                  # inside it
    st.rate_demotion_noted["gone"] = 20260801
    st.rate_demotion_noted["still-here"] = 20260901

    assert st.prune_rate_state() == 1, "the stale caller must be swept"
    assert set(st.rate.windows) == {"still-here"}
    assert st.rate_demotion_noted == {"still-here": 20260901}


# ---------------------------------------------------------------------------
# 🚨 The clock
# ---------------------------------------------------------------------------

def test_pruning_uses_WALL_time_because_recording_does(proxy_state):
    """🚨 The trap this fix could have walked into, pinned so it cannot be
    re-introduced.

    It sits one line below ``timeout_model.prune(mono)``, which takes the
    poller's MONOTONIC clock, and the obvious thing is to pass the same `mono`.
    ``RateLedger.record`` stamps ``time.time()``, so a monotonic `now` compares a
    process uptime against epoch timestamps: the cutoff lands decades before
    every sample, nothing is ever stale, and prune returns 0 forever while
    looking perfectly wired. A leak fixed by a call that does nothing is worse
    than the leak, because the call is evidence it was handled.

    Mutation: `now = time.monotonic()` in `prune_rate_state`. Nothing is ever
    dropped and this fails by assertion.
    """
    st = proxy_state
    st.record_request("someone")
    # A monotonic clock on any machine that has not been up for decades is far
    # smaller than an epoch timestamp — which is the whole failure.
    assert time.monotonic() < time.time() - WINDOW_S

    # Prune with the real wall clock advanced past the window.
    assert _prune_at(st, time.time() + WINDOW_S + 1) == 1
    assert st.rate.windows == {}


def test_the_poller_tick_actually_calls_it():
    """A source read, because the failure mode is an absent call — and an absent
    call is exactly what the whole of this file exists about. It went unnoticed
    for a workstream while the method carried a docstring claiming otherwise.

    An AST walk rather than a substring sweep: `prune_rate_state` appears in
    prose in two comments, and a grep would be satisfied by either.
    """
    tree = ast.parse(Path(health_mod.__file__).read_text())
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "prune_rate_state" in called, (
        "health.py no longer calls prune_rate_state — the rate ledger grows one "
        "entry per caller-supplied agent_id, without bound")


def test_the_method_is_armed_against_the_concurrency_invariant():
    """It mutates single-loop state from a SECOND loop-side writer (the poller,
    beside the submit path). CLAUDE.md's rule is that such a method goes in
    STATE_METHODS, or the soak watches less than it reports."""
    from tests.loop_affinity import STATE_METHODS

    # Asserted before indexing: without this the guard fails with a bare
    # KeyError, which is half a guard — it reports a crash rather than the rule.
    assert "rate" in STATE_METHODS, (
        "the rate ledger is not armed in loop_affinity.STATE_METHODS, so the "
        "soak watches less single-loop state than its count reports")
    assert {"record", "prune"} <= set(STATE_METHODS["rate"])


def _prune_at(st, when: float) -> int:
    """Run the real prune with the wall clock moved to ``when``."""
    import roadstead.state as state_mod

    real = state_mod.time.time
    state_mod.time.time = lambda: when          # type: ignore[assignment]
    try:
        return st.prune_rate_state()
    finally:
        state_mod.time.time = real              # type: ignore[assignment]
