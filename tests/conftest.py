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

import pytest

from originfleet.llmproxy.backend import BackendClientPool


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
