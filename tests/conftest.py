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

import os
import tempfile

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
def _redirect_scratch_to_tmpfs() -> None:
    shm = "/dev/shm"
    if not os.path.isdir(shm) or not os.access(shm, os.W_OK):
        return
    d = os.path.join(shm, "llmproxy-tests")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return
    tempfile.tempdir = d
    os.environ["TMPDIR"] = d


_redirect_scratch_to_tmpfs()


@pytest.fixture(autouse=True)
def _no_network_probes(monkeypatch):
    async def _none(self, ep_cfg):
        return None

    async def _down(self, ep_cfg):
        return False

    monkeypatch.setattr(BackendClientPool, "probe_props", _none)
    monkeypatch.setattr(BackendClientPool, "probe_models", _none)
    monkeypatch.setattr(BackendClientPool, "probe_vllm_capacity", _none)
    monkeypatch.setattr(BackendClientPool, "probe_health", _down)
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
