"""Guard-bite proof: a green suite must mean the guard actually FIRED, not that
the fault never occurred. We take one real correction guard — the egress
degeneration guard — and show that:

  * with the guard PRESENT, a transient degenerate backend response is corrected
    (re-dispatched) into clean output; and
  * with the guard REVERTED (stubbed to a no-op), the *same* input flows the
    degenerate loop straight through to the caller — i.e. the guard's test bites.

If this ever passes with the guard reverted, the harness is blind and every
"handled" verdict elsewhere is suspect.
"""

from __future__ import annotations

import pytest

from roadstead import correction as correction_mod
from roadstead import service as service_mod  # noqa: F401
from roadstead.service import _is_degenerate_text

from tests.fake_backend import FAULT_DEGENERATE_LOOP


async def test_degeneration_guard_present_corrects(proxy):
    # Transient degeneration: the first dispatch loops, a re-dispatch recovers.
    proxy.controller.set_fault(FAULT_DEGENERATE_LOOP, max_hits=1)
    resp = await proxy.chat("sing me something")
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    # the guard re-dispatched → clean, non-degenerate output reached the caller
    assert not _is_degenerate_text(content)
    assert content.startswith("echo:")
    # and the proxy recorded a recovery
    assert proxy.svc._degeneration_recovered >= 1
    assert proxy.total_in_flight() == 0


async def test_degeneration_guard_reverted_leaks_bad_output(proxy, monkeypatch):
    # Revert the guard: stub the enable-check to False so no correction runs.
    # The degeneration guard moved to the Correction collaborator (de-monolith
    # Step 3); its enable-check resolves in correction's namespace now.
    monkeypatch.setattr(correction_mod, "degeneration_guard_enabled", lambda: False)
    proxy.controller.set_fault(FAULT_DEGENERATE_LOOP, max_hits=1)
    resp = await proxy.chat("sing me something")
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    # With the guard gone, the degenerate loop is handed straight to the caller —
    # this is the failure the guard's presence prevents (guard-bite proven).
    assert _is_degenerate_text(content), (
        "expected the reverted guard to leak degenerate output — if this is "
        "clean, the guard was never the thing correcting it")
    assert proxy.total_in_flight() == 0
