"""The wire contract against a REAL vLLM. The other half of the alarm.

`test_real_engine.py` audits the fake against a real `llama-server`. This file
does the same for the engine on the other side of the capacity asymmetry — and
until 2026-09-01 nothing in this project had ever run against a real vLLM at
all, so every vLLM claim in `providers/vllm.py` was asserted rather than
measured.

**Why it needs its own file rather than a parameter on the other one.** The two
engines are audited for *opposite* things. llama.cpp is checked for what it
publishes (`total_slots`, per-slot `n_ctx`) because discovery depends on those
fields existing. vLLM is checked for what it does NOT publish, because the
decision that rests on it — concurrency stays config-seeded, with a drift alert
— is only correct while the absence holds. An absence is invisible to the fake
backend, which publishes what it is told to and can never disagree.

Read-only by default. The discovery half is GETs, so it is safe to point at a
backend that is serving real traffic; the two inference assertions cost tokens
on somebody's endpoint and are opt-in separately:

    ROADSTEAD_WIRE_FIDELITY_VLLM_URL=http://<host>:<port> \\
        pytest -m wire_fidelity tests/wire_fidelity/test_real_vllm.py

    # …and, only where spending a few tokens is acceptable:
    ROADSTEAD_WIRE_FIDELITY_VLLM_INFERENCE=1

Skips cleanly when no engine is reachable, so a normal run is unaffected.
"""
from __future__ import annotations

import os

import pytest

from roadstead.providers.vllm import VLLM

from . import conformance as wire

pytestmark = pytest.mark.wire_fidelity


def _base_url() -> str:
    url = os.environ.get("ROADSTEAD_WIRE_FIDELITY_VLLM_URL")
    if not url:
        pytest.skip("set ROADSTEAD_WIRE_FIDELITY_VLLM_URL to a running vLLM "
                    "(see tests/wire_fidelity/README.md)")
    return url.rstrip("/")


def _inference_allowed() -> bool:
    """The discovery assertions are GETs and cost nothing. These POST.

    Deliberately a SECOND opt-in rather than part of the URL: the only real
    vLLM within reach of this project is one serving live traffic, and
    "somebody pointed the test at an engine" is not consent to generate on it.
    """
    return os.environ.get("ROADSTEAD_WIRE_FIDELITY_VLLM_INFERENCE") == "1"


@pytest.fixture(scope="module")
def base_url() -> str:
    url = _base_url()
    try:
        wire.check_health(url)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no vLLM reachable at {url}: {exc}")
    return url


@pytest.fixture(scope="module")
def models_entry(base_url: str) -> dict:
    return wire.check_models(base_url, expect_max_model_len=True)


@pytest.fixture(scope="module")
def metrics_text(base_url: str) -> str:
    return wire.check_no_published_concurrency(base_url)


# ---------------------------------------------------------------------------
# The asymmetry, run through the code that DECIDES it
# ---------------------------------------------------------------------------

def test_capacity_report_carries_a_context_ceiling_and_no_slot_count(models_entry):
    """The whole asymmetry, in one assertion, through the real parser.

    Deliberately driven through `VLLMProvider.parse_capacity` rather than
    re-read off the JSON here: the descriptor is a claim about what the proxy
    will DO with a real response, and asserting on the fields beside the parser
    would let the two drift apart while both looked audited.
    """
    report = VLLM.parse_capacity(models_entry)

    assert report.slots is None, (
        f"parse_capacity discovered {report.slots} slots from a real vLLM. "
        f"max_slots is config-seeded on the grounds that it cannot; if that is "
        f"now false, the drift alert is measuring the wrong thing.")
    assert isinstance(report.context_per_slot, int) and report.context_per_slot > 0
    assert report.context_per_slot == models_entry["max_model_len"], (
        "max_model_len is the per-request ceiling DIRECTLY — it is not an "
        "aggregate to divide by a slot count, which is exactly the mistake the "
        "llama.cpp path has to make on purpose.")


def test_no_surface_publishes_the_concurrency_the_seed_stands_in_for(
        models_entry, metrics_text):
    """`--max-num-seqs` is on no readable surface — the measurement behind
    `publishes_slot_count=False`.

    🚨 This takes `models_entry` it does not otherwise use, and that is
    load-bearing: the fixture is what establishes the target is a vLLM at all
    (llama.cpp publishes no `max_model_len`). Without it this test passes
    against a llama.cpp engine — which it did, when first written — because
    llama.cpp's `/metrics` has no `max_num_seqs` either, and a guard that holds
    against the engine it was written to distinguish is not a guard.

    The descriptor booleans themselves are pinned in the default suite
    (`tests/test_provider_interface.py`), where they need no engine. What can
    only be checked here is that a real engine still declines to publish the
    number they stand in for.
    """
    assert models_entry["max_model_len"] > 0
    # check_no_published_concurrency did the searching and raises with the
    # offending name; reaching here means every surface we read stayed silent.
    assert "vllm:" in metrics_text, (
        "/metrics carries no vllm: series at all — this is not the engine this "
        "file audits, and its silence about max_num_seqs proves nothing.")


def test_props_and_slots_do_not_exist_here(base_url):
    """llama.cpp's only two slot sources, absent — the engine half of the
    asymmetry. If either starts answering, discovery became possible."""
    assert wire.check_endpoint_absent(base_url, "/props") == 404
    assert wire.check_endpoint_absent(base_url, "/slots") == 404


def test_prefix_cache_counters_are_published_under_the_names_we_read(metrics_text, base_url):
    """`publishes_prefix_cache_metrics=True`, confirmed by name."""
    queries, hits = wire.check_prefix_cache_metrics(base_url, metrics_text)
    assert queries >= 0 and hits >= 0
    assert hits <= queries, (
        f"prefix cache reports more hits ({hits}) than queries ({queries}) — "
        f"compute_cache_stats divides one by the other")


def test_fingerprint_reads_the_weights_not_the_served_alias(models_entry):
    """`probe_model_fingerprint` prefers `root` precisely because `id` is the
    operator's `--served-model-name` and survives a weights swap.

    This is the claim that a stable `id` cannot detect a model change, checked
    on an engine where both fields are real.
    """
    root = models_entry.get("root")
    assert isinstance(root, str) and root.strip(), (
        "data[0].root is absent, so probe_model_fingerprint falls back to "
        "`meta` — which vLLM does not publish either, leaving weight-swap "
        "detection blind on this engine.")
    assert root != models_entry["id"], (
        "root and id are the same string, so the fingerprint carries no more "
        "signal than the alias and a swap under a stable served name is "
        "undetectable — the exact failure probe_model_fingerprint exists for.")


# ---------------------------------------------------------------------------
# Inference — opt-in, because the only reachable vLLM serves real traffic
# ---------------------------------------------------------------------------

def test_sync_completion_shape(base_url, models_entry):
    if not _inference_allowed():
        pytest.skip("set ROADSTEAD_WIRE_FIDELITY_VLLM_INFERENCE=1 to generate "
                    "on this endpoint (it may be serving real traffic)")
    wire.check_sync_completion(base_url, models_entry["id"])


def test_terminal_chunk_shape_is_one_of_the_two_handled(base_url, models_entry):
    """The `finish_reason`-rides-alone rule, on the second engine.

    It was established on llama.cpp. The correction layer applies it to every
    backend, so a vLLM that ended streams differently would be repaired against
    a rule measured somewhere else.
    """
    if not _inference_allowed():
        pytest.skip("set ROADSTEAD_WIRE_FIDELITY_VLLM_INFERENCE=1 to generate "
                    "on this endpoint (it may be serving real traffic)")
    shape = wire.check_stream_terminal_shape(base_url, models_entry["id"])
    print(f"\n  vLLM terminal chunk shape: {shape}")
    assert shape in (wire.TERMINAL_ALONE, wire.TERMINAL_WITH_CONTENT)
