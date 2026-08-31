"""The same wire contract, against a REAL engine. Off the default path.

Purpose, from `docs/handoff.md` Phase 2: "catch the day an engine changes its
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


def test_record_the_n_ctx_units(base_url, record_property):
    """Settle the open question `health.py` flags as "unconfirmed".

    The top-level `props["n_ctx"]` is divided by the slot count on the fallback
    path, i.e. assumed to be an aggregate — with a comment saying it is
    unconfirmed whether any build populates it that way. The fake backend emits
    the SAME value in both places, which cannot be right for both readings.

    This does not assert a preference. It records what a real engine actually
    does, so the answer comes from the engine rather than from the fake.
    """
    if not _launched_by_compose():
        pytest.skip("needs the known --ctx-size/--parallel from compose.yaml")
    props = wire.check_props_shape(base_url)
    top = props.get("n_ctx")
    gen = (props.get("default_generation_settings") or {}).get("n_ctx")
    record_property("top_level_n_ctx", top)
    record_property("generation_settings_n_ctx", gen)
    print(f"\n/props n_ctx units, --ctx-size {COMPOSE_CTX_SIZE} "
          f"--parallel {COMPOSE_PARALLEL}:\n"
          f"  top-level n_ctx                        = {top}\n"
          f"  default_generation_settings.n_ctx      = {gen}\n"
          f"  => top-level is "
          f"{'AGGREGATE' if top == COMPOSE_CTX_SIZE else 'PER-SLOT' if top == EXPECTED_PER_SLOT else 'NEITHER — investigate'}")
    assert gen == EXPECTED_PER_SLOT, "the per-slot field is the load-bearing one"


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
