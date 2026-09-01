"""The shutdown budget — bounded, and the ceiling published in one place.

SIGTERM is the correct signal and that question is closed: the drain persists
the DRR budget row and the completion row even for a straggler it had to cancel,
which is exactly what SIGKILL loses (`docs/ledger.md`). What was still wrong on
2026-09-01 is that **nothing enforced a ceiling**.

The drain was bounded. Everything after it was not — and one of those steps,
`OnDemandManager.close`, makes a NETWORK call per held GPU lease, so a wedged
dispatcher could hang shutdown for as long as it liked, past every budget an
operator had computed. The measured 78.25s was a measurement of a tail that
happened to be fast, never a bound, and the 90s container stop-grace derived
from it inherited that.

🚨 That is why the recommended stop-grace went UP rather than down. 108 is the
first one that is a ceiling instead of an observation.

Three things are pinned here:

1. **The arithmetic**, so the published total cannot drift from the phases.
2. **The ceiling holds** — a wedged teardown does not extend shutdown.
3. **One source of truth**: uvicorn's argument, the Dockerfile label and the
   docs all read the same number rather than three people computing it.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

import pytest

from roadstead.config import ProxyConfig
from roadstead.service import (
    RECOMMENDED_STOP_GRACE_S,
    SHUTDOWN_DEADLINE_S,
    UVICORN_GRACEFUL_S,
    ProxyService,
    _CANCEL_UNWIND_S,
    _DRAIN_DEADLINE_S,
    _QUEUE_CLOSE_S,
    _STOP_GRACE_MARGIN_S,
    _TEARDOWN_CLOSE_S,
)

_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1 · The arithmetic
# ---------------------------------------------------------------------------

def test_the_published_ceiling_is_the_sum_of_the_phases():
    """🚨 There is no outer `wait_for` around shutdown, deliberately — one would
    cancel `queue_db.close()` mid-flush, which is the SIGKILL failure the drain
    exists to avoid. So the ceiling is TRUE BY ARITHMETIC, and the arithmetic is
    pinned here: a phase whose budget grows without the total growing would make
    the published number a lie in exactly the silent direction.
    """
    assert SHUTDOWN_DEADLINE_S == (
        _DRAIN_DEADLINE_S + _CANCEL_UNWIND_S + _TEARDOWN_CLOSE_S + _QUEUE_CLOSE_S)
    # 🚨 What this actually catches is somebody HARD-CODING the total. Growing a
    # phase cannot break the identity — the total is computed from the phases —
    # and that is the point: the number that moves when a phase moves is the
    # published one, which `test_the_dockerfile_label_agrees_with_the_computed_number`
    # then catches against the image label. The two guards are a pair, and a
    # mutation of a phase budget alone survives this one on purpose.
    assert not re.search(r"^SHUTDOWN_DEADLINE_S\s*=\s*[\d.]+\s*$",
                         (_ROOT / "roadstead" / "service.py").read_text(), re.M), (
        "SHUTDOWN_DEADLINE_S is a literal — it must be the SUM of the phases, "
        "or a phase can grow without the published ceiling moving")


def test_the_stop_grace_covers_both_serial_budgets():
    """🚨 The SUM, not the larger of the two.

    uvicorn's budget bounds in-flight HTTP connections; only once it expires
    does uvicorn send `lifespan.shutdown`, and it never bounds that at all. An
    operator who took the larger number would set a stop-grace that SIGKILLs the
    proxy mid-drain in exactly the case the drain exists for.
    """
    assert RECOMMENDED_STOP_GRACE_S == int(
        UVICORN_GRACEFUL_S + SHUTDOWN_DEADLINE_S + _STOP_GRACE_MARGIN_S)
    assert RECOMMENDED_STOP_GRACE_S > UVICORN_GRACEFUL_S + SHUTDOWN_DEADLINE_S
    # And it is not the nesting the old derivation assumed.
    assert RECOMMENDED_STOP_GRACE_S > max(UVICORN_GRACEFUL_S, SHUTDOWN_DEADLINE_S) * 2


def test_uvicorn_is_no_longer_derived_from_the_drain_deadline():
    """The old value was `_DRAIN_DEADLINE_S + 18`, carrying a comment that said
    uvicorn's budget must exceed the drain "or it hard-kills the process
    mid-flush". It does not: uvicorn hands over to the lifespan shutdown. A
    derivation that encodes a false nesting is worse than an independent number,
    because it looks like it is keeping two things in step.
    """
    import ast
    source = (_ROOT / "roadstead" / "__main__.py").read_text()
    assert "timeout_graceful_shutdown=int(UVICORN_GRACEFUL_S)" in source
    # 🚨 AST, not a substring sweep. The comment above that line EXPLAINS the
    # old derivation and has to keep saying so — a guard that forbade the words
    # would delete the reasoning to satisfy itself, which is how a comment
    # recording why something is the way it is gets lost.
    tree = ast.parse(source)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "_DRAIN_DEADLINE_S" not in used, (
        "__main__ is deriving uvicorn's budget from the drain again")


# ---------------------------------------------------------------------------
# 2 · The ceiling actually holds
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_wedged_lease_release_does_not_hang_shutdown(tmp_path):
    """🚨 THE hole this closes.

    `OnDemandManager.close` makes a network call per held GPU lease. A dispatcher
    that stops answering used to hang the lifespan shutdown indefinitely — past
    the operator's stop-grace, at which point the container SIGKILLs a process
    that had not yet flushed anything.

    Bounded now, and the flush still happens: a lost lease costs its TTL and
    nothing else, while a lost flush costs the budgets and completions that are
    the entire reason for draining.
    """
    svc = ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db")))
    await svc.startup()

    wedged = asyncio.Event()          # never set

    async def hang():
        await wedged.wait()

    svc._on_demand.close = hang

    started = time.monotonic()
    await svc.shutdown()
    elapsed = time.monotonic() - started

    # It waited for the bounded teardown and then carried on.
    assert _TEARDOWN_CLOSE_S <= elapsed < _TEARDOWN_CLOSE_S + _QUEUE_CLOSE_S + 5
    # 🚨 And the part that matters ran: the queue is closed, which means the
    # writer was flushed and joined rather than abandoned.
    assert svc._queue_db._conn is None


@pytest.mark.asyncio
async def test_an_idle_shutdown_is_still_immediate(tmp_path):
    """The bound is a ceiling, not a floor. A quiet proxy must not start paying
    48 seconds to exit because the worst case was given a budget."""
    svc = ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db")))
    await svc.startup()
    started = time.monotonic()
    await svc.shutdown()
    assert time.monotonic() - started < 5.0


# ---------------------------------------------------------------------------
# 3 · One source of truth
# ---------------------------------------------------------------------------

def test_the_dockerfile_label_agrees_with_the_computed_number():
    """🚨 Three places used to hold this number and one of them was a comment.

    An image whose label understates the requirement is worse than one with no
    label: an operator who reads it gets a stop-grace that truncates the drain,
    and believes they took the documented advice.
    """
    dockerfile = (_ROOT / "Dockerfile").read_text()
    label = re.search(
        r'org\.roadstead\.required-stop-grace-period-seconds="(\d+)"', dockerfile)
    assert label, "the Dockerfile no longer carries the stop-grace label"
    assert int(label.group(1)) >= RECOMMENDED_STOP_GRACE_S, (
        f"the image label says {label.group(1)}s but the computed ceiling is "
        f"{RECOMMENDED_STOP_GRACE_S}s")
    # The prose has to agree with the label, or a reader takes the prose.
    assert f"--stop-timeout {RECOMMENDED_STOP_GRACE_S}" in dockerfile
    assert f"stop_grace_period: {RECOMMENDED_STOP_GRACE_S}s" in dockerfile


def test_the_ledger_records_that_the_number_moved_and_why():
    """A stop-grace that grows is the kind of change an operator has to act on,
    so it goes in the ledger rather than only in a diff."""
    ledger = (_ROOT / "docs" / "ledger.md").read_text()
    # 🚨 Every stated REQUIREMENT, not just "does the number appear somewhere".
    # The first version of this test passed while the requirement line said ≥90,
    # because the reasoning paragraph beside it happened to mention 108. A
    # document that states a smaller requirement than the code enforces sends an
    # operator to a stop-grace that truncates the drain.
    stated = [int(n) for n in re.findall(
        r"stop-grace-period must be ≥(\d+)s", ledger)]
    assert stated, "docs/ledger.md no longer states a stop-grace requirement"
    for value in stated:
        assert value >= RECOMMENDED_STOP_GRACE_S, (
            f"docs/ledger.md requires ≥{value}s but the computed ceiling is "
            f"{RECOMMENDED_STOP_GRACE_S}s")


def test_the_startup_log_states_the_budget():
    """The number is needed by whoever writes the container's stop-grace, and
    they are not reading the source. Said once, at startup."""
    source = (_ROOT / "roadstead" / "service.py").read_text()
    assert "stop-grace-period to at least %ds" in source
    assert "RECOMMENDED_STOP_GRACE_S," in source
