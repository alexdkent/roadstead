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


def test_the_fakes_top_level_n_ctx_is_a_known_fidelity_gap():
    """🚨 A gap recorded, not papered over.

    The fake emits `props_n_ctx` at BOTH `default_generation_settings.n_ctx`
    (which Roadstead reads as PER-SLOT) and top-level `n_ctx` (which Roadstead
    divides by the slot count, i.e. reads as an AGGREGATE). The same number
    cannot be both. Nothing breaks today because the reader prefers the former
    and never reaches the fallback — but a test that exercised the fallback
    against this fake would compute 32768/4 = 8192 and call it per-slot, while
    the fake also claims per-slot is 32768.

    It is NOT "fixed" here by making the top level an aggregate, because
    `health.py` says outright that it is "unconfirmed whether it's ever
    populated as an aggregate" by a real build. Inventing a shape for the fake
    would make the fake authoritative over the engine, which is backwards.

    `test_real_engine.py::test_record_the_n_ctx_units` is what settles it. Until
    it has been run against a real `llama-server`, this test documents the
    inconsistency so nobody builds on either reading.
    """
    srv = FakeBackendServer(FakeBackend(props_n_parallel=4, props_n_ctx=32768)).start()
    try:
        props = wire.check_props_shape(srv.url)
        assert props["default_generation_settings"]["n_ctx"] == 32768
        assert props["n_ctx"] == 32768, (
            "the fake's top-level n_ctx changed — if this was a deliberate "
            "fidelity fix, it needs evidence from a real engine, and this test "
            "plus the note in tests/wire_fidelity/README.md should be updated")
        assert props["n_ctx"] // wire.discovered_slots(props) != 32768, (
            "the two readings coincide, so the gap this test documents is gone")
    finally:
        srv.stop()
