"""The tmpfs scratch basetemp must bound what it RETAINS, not trust the cap.

Regression guard for 2026-08-18: the container's 64 MB ``/dev/shm`` sat 100%
full of six retained pytest roots. The redirect had been sized against one
run's peak (12.9 MB) while pytest retains three roots and the per-run peak had
grown to ~20 MB. Once full, the free-space guard stopped choosing the basetemp,
so pytest's reaper — which only runs inside the basetemp in use — never ran
there again and the 64 MB was pinned permanently. Cost: the llmproxy tollgate
lost its tmpfs speedup (20s -> 71s) with nothing to signal why.

These pin the two properties that keep it bounded: finished roots are reaped,
and a root a LIVE session owns is never touched.
"""

from __future__ import annotations

import os
import time

from tests.conftest import _PRUNE_MIN_AGE_S, _prune_stale_scratch


def _make_root(basetemp, name: str, *, locked: bool, age_s: float):
    """Build one numbered pytest root the way pytest lays it out."""
    root = basetemp / "pytest-of-root" / name
    root.mkdir(parents=True)
    (root / "test_something0").mkdir()
    (root / "test_something0" / "queue.db").write_bytes(b"x" * 1024)
    if locked:
        (root / ".lock").write_text("")
    stamp = time.time() - age_s
    os.utime(root, (stamp, stamp))
    return root


def test_finished_roots_are_reaped(tmp_path):
    """A root with no .lock, past the age floor, is pytest-finished — reap it."""
    stale = _make_root(tmp_path, "pytest-56", locked=False, age_s=_PRUNE_MIN_AGE_S * 2)

    assert _prune_stale_scratch(str(tmp_path)) == 1
    assert not stale.exists()


def test_a_live_session_is_never_reaped(tmp_path):
    """Two builders can ship at once. Neither may delete the other's scratch.

    A live session is identified two ways, and BOTH must hold it: its ``.lock``
    (pytest removes that only on clean exit) and, for the sliver of time before
    the lock is written, the age floor.
    """
    locked = _make_root(tmp_path, "pytest-57", locked=True, age_s=_PRUNE_MIN_AGE_S * 2)
    just_born = _make_root(tmp_path, "pytest-58", locked=False, age_s=0)

    assert _prune_stale_scratch(str(tmp_path)) == 0
    assert (locked / "test_something0" / "queue.db").exists()
    assert (just_born / "test_something0" / "queue.db").exists()


def test_retention_cannot_latch_the_basetemp_full(tmp_path):
    """The bound holds across repeated runs — the failure that actually bit.

    Three finished runs are exactly what pytest's default retention keeps, and
    3 x ~20 MB is what overflowed the 64 MB cap. After a reap the retained set
    must be empty, so free space is recovered before the guard measures it.
    """
    for i, name in enumerate(("pytest-59", "pytest-60", "pytest-61")):
        _make_root(tmp_path, name, locked=False, age_s=_PRUNE_MIN_AGE_S + 60 * i)

    assert _prune_stale_scratch(str(tmp_path)) == 3
    assert list((tmp_path / "pytest-of-root").iterdir()) == []


def test_reaping_never_raises_on_a_hostile_basetemp(tmp_path):
    """Failing to reap costs wall-clock; it must never take the suite down."""
    assert _prune_stale_scratch(str(tmp_path / "does-not-exist")) == 0

    dangling = tmp_path / "pytest-of-root"
    dangling.mkdir()
    (dangling / "pytest-62").symlink_to(tmp_path / "nowhere")
    assert _prune_stale_scratch(str(tmp_path)) == 0
