"""llmproxy test hygiene: no test may touch the real network.

The capacity poller probes backends (/health for skip_discovery shims,
/props + /v1/models otherwise) starting with its FIRST iteration after
``ProxyService.startup()`` — which most tests call. This autouse fixture
stubs every probe at the BackendClientPool *class* level so a test that
forgets to stub them can never issue a real HTTP call to the inference
hosts (10.0.0.3/.6 are reachable from the dev Mac AND the container, so a
leak would silently "work" locally and flake elsewhere).

Tests that exercise probe behaviour set ``svc._backend.probe_* = ...``
INSTANCE attributes, which shadow these class stubs — the existing pattern
keeps working unchanged.
"""

from __future__ import annotations

# Pin the STDLIB ``queue`` into sys.modules before any test module (or a lazily
# imported pytest plugin) does ``from queue import Queue`` while the inner
# ``originfleet/llmproxy`` dir is on sys.path[0] — there it would shadow stdlib
# with ``llmproxy/queue.py`` (whose relative ``from .config import`` then fails
# with "attempted relative import with no known parent package"). This bites only
# where a hypothesis-style plugin is installed (the dev Mac); the container has no
# such plugin, so this is a harmless no-op there. conftest imports run before
# collection, when sys.path[0] is still the stdlib-clean rootdir.
import queue  # noqa: F401

import glob
import os
import shutil
import tempfile
import time

import pytest

from originfleet.llmproxy.backend import BackendClientPool


# --- tmpfs redirect for test scratch (SQLite fsync avoidance) --------------
# Every E2E ``proxy`` fixture builds a real ``queue.db`` and every unit test
# uses pytest's ``tmp_path``; both root at ``tempfile.gettempdir()``. In the
# container that default is the docker overlay, which on this nasbox host lives
# on a parity-ARRAY disk (measured 30-80ms/fsync in a quiet window). SQLite WAL
# checkpoints + VACUUM fsync there dominate the suite wall-clock and — worse —
# balloon ~2x under host load, which is what pushes the tollgate at the budget.
# Redirect the scratch root to tmpfs (``/dev/shm`` = 0ms fsync) when present.
# tmpfs supports fsync/WAL/VACUUM (fsync is a no-op) so this is behaviour-
# equivalent for what the durability tests assert (logical outcomes, not
# timing). Falls back to the OS default where ``/dev/shm`` is absent or not
# writable (e.g. macOS dev), so it is a silent no-op there. Must run at
# conftest-import time — before pytest's tmp_path_factory resolves its base.
# The suite's own scratch high-water mark is ~15-20 MB; require comfortably
# more than that free before choosing tmpfs over the (slower, roomy) default.
_MIN_SHM_FREE_BYTES = 32 * 1024 * 1024

# --- bounding the scratch we retain ---------------------------------------
# 🚨 A CAP IS NOT A BOUND. The commit that introduced this redirect sized the
# 64 MB tmpfs against ONE run's peak ("peak working set: 12.9 MB, existing 64M
# shm is ample; no resize") — but pytest retains the last THREE numbered roots,
# and the per-run peak has since grown to ~20 MB as Phase 3/4/5 added tests.
# 3 x 20 MB against a 64 MB cap fills it and KEEPS it full. Worse, the state
# LATCHES: once free space drops under the guard above we stop choosing this
# basetemp, so pytest's own reaper — which only runs inside the basetemp in
# use — never runs here again and the 64 MB is pinned forever. Measured
# 2026-08-18: six retained roots, `shm 64M 64M 4.0K 100%`, tollgate back to
# 71s from 20s. So we bound what we retain instead of trusting the cap.
#
# We are the sole owner of this basetemp, so we may reap it. A root is ours to
# delete when pytest has finished with it: pytest writes a ``.lock`` into each
# numbered dir and removes it on clean session exit, so "no .lock" means "no
# live session". The age floor closes the one race that leaves — pytest creates
# the numbered dir a hair before it writes the lock (``make_numbered_dir`` then
# ``create_cleanup_lock``), so a root that young may belong to a session
# starting concurrently. That window is microseconds; keep the floor SHORT.
# 🚨 Do not raise it "to be safe" — the floor is the one thing that can still
# let retention outrun the cap. A run leaves ~20 MB and takes ~30s, so a 10
# MINUTE floor would pin ~20 roots during back-to-back iteration and put us
# straight back in the 100%-full state this reaper exists to prevent. At 60s
# the worst case is two retained roots (~40 MB); the free-space guard below
# then sends that one run to the slower OS default and the next run reaps
# normally. Slow and self-healing, never latched.
_PRUNE_MIN_AGE_S = 60


def _prune_stale_scratch(basetemp: str, *, now: float | None = None) -> int:
    """Delete finished pytest roots under our own basetemp. Returns the count.

    Never raises: a failure to reap costs wall-clock (we fall back to the OS
    default) and must never take the suite down with it.
    """
    now = time.time() if now is None else now
    reaped = 0
    for root in glob.glob(os.path.join(basetemp, "pytest-of-*", "pytest-*")):
        try:
            if not os.path.isdir(root) or os.path.islink(root):
                continue
            if os.path.exists(os.path.join(root, ".lock")):
                continue  # a live session owns it
            if now - os.stat(root).st_mtime < _PRUNE_MIN_AGE_S:
                continue  # may be a session that has not written its lock yet
            shutil.rmtree(root, ignore_errors=True)
            reaped += 1
        except OSError:
            continue
    return reaped


def _redirect_scratch_to_tmpfs() -> None:
    shm = "/dev/shm"
    if not os.path.isdir(shm) or not os.access(shm, os.W_OK):
        return
    # Reap BEFORE measuring free space — the whole point is that the space the
    # last run retained is space this run may have back. Reaping after the
    # check would preserve exactly the latch described above.
    _prune_stale_scratch(os.path.join(shm, "llmproxy-tests"))
    # 🚨 A FULL tmpfs is still a writable directory, so the writability check
    # above does not catch it. Without this guard the redirect proceeds, every
    # scratch SQLite open dies with "database or disk is full", and pytest
    # reports it as ~59 unrelated failures across test_timeout_events.py and
    # test_timeout_shadow.py — which reads as a code regression and sends you
    # looking for one. Measured in-container 2026-08-18: /dev/shm is 64 MB
    # there (`shm 64M 64M 4.0K 100%`) and the pytest tmpdirs pytest itself
    # retains fill it, after which EVERY subsequent tollgate run fails this
    # way. Falling back to the OS default costs wall-clock and nothing else.
    # With the reaper above this is now a BACKSTOP, not the primary defence:
    # it still catches a genuine outsider filling the tmpfs, but our own
    # retention can no longer be what trips it.
    try:
        st = os.statvfs(shm)
        free = st.f_bavail * st.f_frsize
    except OSError:
        return
    if free < _MIN_SHM_FREE_BYTES:
        return
    d = os.path.join(shm, "llmproxy-tests")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return
    tempfile.tempdir = d
    os.environ["TMPDIR"] = d


_redirect_scratch_to_tmpfs()


#: Every ``BackendClientPool.probe_*`` method, and the value its stub returns.
#:
#: 🚨 THIS LIST IS LOAD-BEARING AND IT USED TO FAIL OPEN. `_no_network_probes`
#: stubs by name, so a probe added later was simply absent and therefore NOT
#: stubbed — it made REAL calls to real fleet hosts from a unit test. Caught
#: 2026-08-24 when `probe_model_fingerprint` and `probe_thinking_switch`
#: landed: the e2e suite went 1.3s -> 61-181s because the poller was dialling
#: 10.0.0.3, and `probe_thinking_switch` would have run a real GENERATION
#: against live tier3 from `pytest`. The tests still PASSED — only teardown
#: timed out — which is exactly how a gap like this survives unnoticed.
#:
#: The list stays (blocking the transport wholesale breaks the ~84 tests that
#: legitimately drive `call`/`stream` through a fake client), but it no longer
#: fails open: `test_model_swap_guards.py::test_every_backend_probe_is_stubbed`
#: asserts this covers every `probe_*` on the class, so a new probe fails a
#: test on the day it is written.
STUBBED_PROBES = {
    "probe_props": None,
    "probe_models": None,
    "probe_vllm_capacity": None,
    "probe_health": False,
    "probe_model_fingerprint": None,
    "probe_thinking_switch": None,
}


@pytest.fixture(autouse=True)
def _no_network_probes(monkeypatch):
    """Keep the unit suite off the network. See STUBBED_PROBES above.

    A test that supplies its OWN fake client still exercises the real parsing
    code: it sets `_client_for` on the pool INSTANCE, and calls the probe
    directly rather than through the poller, so these class-level stubs are
    bypassed exactly where they should be.
    """
    def _stub(value):
        async def _probe(self, ep_cfg, *args, **kwargs):
            return value
        return _probe

    for name, value in STUBBED_PROBES.items():
        if hasattr(BackendClientPool, name):
            monkeypatch.setattr(BackendClientPool, name, _stub(value))
    yield


@pytest.fixture(autouse=True)
def _fast_retry_backoff(monkeypatch):
    """Collapse the transient-retry backoff for the whole suite. Production
    sleeps ``_RETRY_BACKOFF_S`` (0.5s) between dispatch retries; every
    retry-path test (empty-rescue, transient-unavailable, backend-timeout
    retry) pays that real wall-clock, and there are many — ~3-4s of pure sleep
    across the suite that a loaded-container tollgate can ill afford. No test
    asserts the backoff DURATION (only retry OUTCOMES + slot accounting), so
    shrinking it is behaviour-equivalent for what's under test. lifecycle reads
    the module global at call time, so patching the module attribute takes
    effect. Doctrine: shrink the corpus/waits, never raise the budget."""
    monkeypatch.setattr(
        "originfleet.llmproxy.lifecycle._RETRY_BACKOFF_S", 0.02, raising=False)
    yield
