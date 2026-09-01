"""The same wire contract, against a REAL engine. Off the default path.

Purpose, from the extraction plan (`docs/history.md`): "catch the day an engine changes its
`/props` shape". Everything else in the suite runs against `roadstead.testing`,
which encodes what engines did on the day it was written — a snapshot with no
alarm on it. This is the alarm.

Deliberately few tests and deliberately slow: it needs a real `llama-server`
holding real weights. See `README.md` in this directory; `compose.yaml` brings
one up with a small model.

    ROADSTEAD_WIRE_FIDELITY_URL=http://127.0.0.1:18080 \\
        pytest -m wire_fidelity tests/wire_fidelity/

Skips cleanly when no engine is reachable, so it never breaks a normal run.
"""
from __future__ import annotations

import os

import pytest

from . import conformance as wire

pytestmark = pytest.mark.wire_fidelity

#: The compose file launches with these, and several assertions depend on it.
#: Change one, change the other.
COMPOSE_CTX_SIZE = 8192
COMPOSE_PARALLEL = 4
EXPECTED_PER_SLOT = COMPOSE_CTX_SIZE // COMPOSE_PARALLEL   # 2048


def _base_url() -> str:
    url = os.environ.get("ROADSTEAD_WIRE_FIDELITY_URL")
    if not url:
        pytest.skip("set ROADSTEAD_WIRE_FIDELITY_URL to a running llama-server "
                    "(see tests/wire_fidelity/README.md)")
    return url.rstrip("/")


def _launched_by_compose() -> bool:
    """Only the launch-config-dependent assertions need this."""
    return os.environ.get("ROADSTEAD_WIRE_FIDELITY_COMPOSE") == "1"


@pytest.fixture(scope="module")
def base_url() -> str:
    url = _base_url()
    try:
        wire.check_health(url)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no engine reachable at {url}: {exc}")
    return url


@pytest.fixture(scope="module")
def model_id(base_url: str) -> str:
    return wire.check_models(base_url, expect_max_model_len=False)["id"]


def test_props_still_publishes_a_slot_count(base_url):
    """If this fails, runtime capacity discovery has silently reverted to the
    CONFIGURED slot count — the proxy keeps working and admits against a number
    nobody verified. That is the failure this whole file exists for."""
    props = wire.check_props_shape(base_url)
    slots = wire.discovered_slots(props)
    assert slots >= 1
    if _launched_by_compose():
        assert slots == COMPOSE_PARALLEL, (
            f"launched with --parallel {COMPOSE_PARALLEL} but /props reports "
            f"{slots} slots")


def test_generation_settings_n_ctx_is_per_slot_not_aggregate(base_url):
    """🚨 The regression that quartered every multi-slot endpoint.

    `health.py` divides the TOP-LEVEL `n_ctx` by the slot count but takes
    `default_generation_settings.n_ctx` as-is. If a build ever reported the
    aggregate in the latter, every multi-slot endpoint's context_per_slot would
    be N times too large and the context gate would admit requests that cannot
    fit. Requires the known launch config to be meaningful.
    """
    if not _launched_by_compose():
        pytest.skip("needs the known --ctx-size/--parallel from compose.yaml")
    props = wire.check_props_shape(base_url)
    per_slot, source = wire.discovered_context_per_slot(props)
    assert source == "default_generation_settings.n_ctx", (
        f"llama.cpp stopped publishing default_generation_settings.n_ctx; "
        f"discovery fell back to {source}")
    assert per_slot == EXPECTED_PER_SLOT, (
        f"launched --ctx-size {COMPOSE_CTX_SIZE} --parallel {COMPOSE_PARALLEL}, "
        f"so per-slot context should be {EXPECTED_PER_SLOT}; /props reports "
        f"{per_slot}. If this is now the AGGREGATE, health.py's handling of "
        f"default_generation_settings.n_ctx is wrong for this build.")


def test_top_level_n_ctx_is_absent_not_an_aggregate(base_url):
    """The open question, settled 2026-08-31 — and settled differently than
    either candidate answer.

    `health.py` reads `default_generation_settings.n_ctx` as-is (per-slot) but
    DIVIDES a top-level `props["n_ctx"]` by the slot count, i.e. reads it as an
    aggregate, while noting it is "unconfirmed whether it's ever populated as an
    aggregate". The answer from b5350 is that **it is not populated at all** —
    there is no top-level `n_ctx` key. The fallback is dead code against a
    current build, kept only for older ones.

    That also settled the fake's fidelity gap: it was emitting a field the real
    engine does not have. `roadstead.testing` now defaults to this narrow shape.

    🚨 If this ever fails because a top-level `n_ctx` reappeared, do NOT assume
    the divide-by-slots reading is right. Print the value and compare it against
    --ctx-size and --parallel first: at 8192/4, an aggregate reads 8192 and a
    per-slot reads 2048, and the divide would quarter every multi-slot endpoint.
    """
    props = wire.check_props_shape(base_url)
    gen = (props.get("default_generation_settings") or {}).get("n_ctx")
    top = props.get("n_ctx")
    print(f"\n/props at --ctx-size {COMPOSE_CTX_SIZE} --parallel {COMPOSE_PARALLEL}:"
          f"\n  top-level n_ctx                   = {top!r}"
          f"\n  default_generation_settings.n_ctx = {gen!r}"
          f"\n  top-level keys                    = {sorted(props)}")
    assert gen == EXPECTED_PER_SLOT, "the per-slot field is the load-bearing one"
    assert "n_ctx" not in props, (
        f"a top-level n_ctx has reappeared ({top!r}). health.py would DIVIDE it "
        f"by {COMPOSE_PARALLEL}; if it is really an aggregate that is correct, "
        f"and if it is per-slot that quarters the endpoint's context. Read the "
        f"docstring before changing anything.")


def test_slot_count_comes_from_total_slots_not_n_parallel(base_url):
    """Capacity discovery PREFERS `default_generation_settings.n_parallel` and
    falls back to `total_slots`. b5350 publishes only the latter — so the
    fallback is not a legacy nicety, it is the ONLY path that works against a
    current engine. Deleting it would leave the suite green (the fake used to
    publish both) and break real discovery.
    """
    props = wire.check_props_shape(base_url)
    gen = props.get("default_generation_settings") or {}
    assert gen.get("n_parallel") is None, (
        "b5350 did not publish default_generation_settings.n_parallel; if a "
        "build now does, the preference order in health.py starts mattering")
    assert isinstance(props.get("total_slots"), int), (
        "total_slots is gone AND n_parallel was never there — capacity "
        "discovery has no source and silently keeps the configured value")
    assert wire.discovered_slots(props) == COMPOSE_PARALLEL


def test_models_publishes_a_served_id(base_url):
    entry = wire.check_models(base_url, expect_max_model_len=False)
    assert entry["id"]


def test_sync_completion_shape(base_url, model_id):
    wire.check_sync_completion(base_url, model_id)


def test_terminal_chunk_shape_is_one_of_the_two_handled(base_url, model_id):
    """Both known shapes are handled — `finish_reason` alone on a chunk with an
    empty delta, or riding along with content (which the proxy splits). A THIRD
    shape would pass through the repair layer untouched, so fail on one."""
    shape = wire.check_stream_terminal_shape(base_url, model_id)
    assert shape in (wire.TERMINAL_ALONE, wire.TERMINAL_WITH_CONTENT), shape
    print(f"\nreal engine terminal-chunk shape: {shape}")
