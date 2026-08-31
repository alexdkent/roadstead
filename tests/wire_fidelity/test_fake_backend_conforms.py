"""The fake backend must satisfy the same wire contract a real engine does.

Runs on the default path. Its value is not that the fake works — other tests
cover that — but that it is held to `conformance.py`, the *same* assertions
`test_real_engine.py` runs against a live `llama-server`. When the two disagree,
the fake has drifted from the thing it claims to imitate, and every test that
trusts it has been proving something about a fiction.
"""
from __future__ import annotations

import pytest

from roadstead.testing import FakeBackend, FakeBackendServer

from . import conformance as wire


@pytest.fixture(params=["llama.cpp", "vllm"])
def engine_server(request):
    srv = FakeBackendServer(FakeBackend(engine=request.param)).start()
    try:
        yield request.param, srv
    finally:
        srv.stop()


def test_health(engine_server):
    _engine, srv = engine_server
    wire.check_health(srv.url)


def test_models(engine_server):
    engine, srv = engine_server
    wire.check_models(srv.url, expect_max_model_len=(engine == "vllm"))


def test_sync_completion(engine_server):
    _engine, srv = engine_server
    wire.check_sync_completion(srv.url, srv.controller.served_model_id)


def test_stream_terminal_chunk_rides_alone(engine_server):
    """The fake must emit the shape the proxy's repair layer is written around.
    If it emitted the other legal shape, every streaming test would exercise the
    splitting path and none the passthrough."""
    _engine, srv = engine_server
    shape = wire.check_stream_terminal_shape(srv.url, srv.controller.served_model_id)
    assert shape == wire.TERMINAL_ALONE, shape


def test_props_capacity_is_discoverable():
    """llama.cpp only. `/props` is the entire basis of runtime slot discovery."""
    srv = FakeBackendServer(FakeBackend(props_n_parallel=4, props_n_ctx=32768)).start()
    try:
        props = wire.check_props_shape(srv.url)
        assert wire.discovered_slots(props) == 4
        per_slot, source = wire.discovered_context_per_slot(props)
        assert source == "default_generation_settings.n_ctx"
        assert per_slot == 32768
    finally:
        srv.stop()


def test_the_default_props_shape_matches_the_verified_real_one():
    """Resolved 2026-08-31 against a real llama-server (b5350).

    This test used to document a gap: the fake emitted the same number at both
    `default_generation_settings.n_ctx` (read as PER-SLOT) and top-level
    `n_ctx` (DIVIDED by the slot count, i.e. read as an aggregate), which cannot
    be right for both. The real engine settled it more sharply than expected —
    it publishes **neither** a top-level `n_ctx` nor a `n_parallel` nor a
    `slots` list. The fake was simply more generous than reality.

    🚨 Being more generous is the dangerous direction. Capacity discovery
    PREFERS `default_generation_settings.n_parallel` and falls back to
    `total_slots`; the real engine only has the latter. Under the old shape the
    fallback that real discovery entirely depends on was never exercised, so
    deleting it would have left the suite green and broken discovery in
    production.
    """
    srv = FakeBackendServer(FakeBackend(props_n_parallel=4, props_n_ctx=32768)).start()
    try:
        props = wire.check_props_shape(srv.url)
        assert props["default_generation_settings"]["n_ctx"] == 32768
        assert props["total_slots"] == 4
        for absent in ("n_ctx", "slots"):
            assert absent not in props, (
                f"the fake publishes top-level {absent!r}, which llama.cpp "
                f"b5350 does not — re-verify against a real engine before "
                f"widening the shape")
        assert "n_parallel" not in props["default_generation_settings"], (
            "the fake publishes default_generation_settings.n_parallel, which "
            "b5350 does not — restoring it hides the total_slots fallback that "
            "real capacity discovery depends on")
        # And the narrow shape still yields the right answers.
        assert wire.discovered_slots(props) == 4
        assert wire.discovered_context_per_slot(props) == (
            32768, "default_generation_settings.n_ctx")
    finally:
        srv.stop()


def test_the_legacy_profile_still_drives_the_aggregate_fallback():
    """`health.py` keeps a top-level-`n_ctx` fallback for "older/different
    builds". No observed engine publishes it, so it can only be exercised
    deliberately — which is what the legacy profile is for. Without this the
    fallback would be untestable through the fake and quietly rot."""
    srv = FakeBackendServer(FakeBackend(
        props_n_parallel=4, props_n_ctx=32768, props_profile="legacy")).start()
    try:
        props = wire.check_props_shape(srv.url)
        assert props["n_ctx"] == 32768
        assert props["default_generation_settings"]["n_parallel"] == 4
        assert len(props["slots"]) == 4
        assert wire.discovered_slots(props) == 4
    finally:
        srv.stop()
