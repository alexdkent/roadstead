"""The provider interface — resolution, the declared asymmetry, and the rule
that behaviour reads a CAPABILITY rather than an engine name.

Extracted from `backend.py` on 2026-08-31 (roadmap Workstream A). The split is
supposed to be behaviour-preserving, and the payload/discovery regressions that
prove it already exist elsewhere (test_payload_normalization,
test_thinking_kwargs_are_family_aware, test_context_discovery, and the model-swap
guards). What is NOT covered by those, and is covered here, is the seam itself:

  * an engine string that nobody anticipated must still resolve to something —
    every branch this package replaced was written `== "vllm"`, so an unknown
    engine has always taken the llama.cpp path and a config typo has never taken
    an endpoint offline;
  * the llama.cpp/vLLM capacity asymmetry is a DECLARATION, not a comment. It is
    the reason vLLM concurrency stays config-seeded (CLAUDE.md), so it should
    fail here if someone "tidies" it into symmetry;
  * nothing outside the config plumbing may go back to comparing
    `backend_engine` against a literal.
"""
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from roadstead.config import EndpointConfig
from roadstead.providers import (
    LLAMACPP,
    VLLM,
    Provider,
    provider_for,
    provider_for_engine,
)

_ROOT = Path(__file__).resolve().parents[1]
_PKG = _ROOT / "roadstead"


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine,expected", [
    ("llama.cpp", "llama.cpp"),
    ("vllm", "vllm"),
    ("VLLM", "vllm"),                 # models.yaml is written for humans
    ("llama.cpp (Vulkan)", "llama.cpp"),   # a build note, not an engine
    ("shim", "llama.cpp"),            # embed/rerank shims: see below
    ("", "llama.cpp"),
    (None, "llama.cpp"),
    ("something-nobody-wrote-yet", "llama.cpp"),
])
def test_engine_strings_resolve_the_way_the_old_branches_did(engine, expected):
    assert provider_for_engine(engine).name == expected


def test_shim_endpoints_are_not_a_provider_kind():
    """`backend_engine: shim` marks the non-OpenAI embed/rerank FastAPI shims,
    and it has never been what keeps the poller off them — `skip_discovery` is,
    and health checks that BEFORE asking a provider anything. Folding the two
    into one concept would silently change which endpoints get probed."""
    ep = EndpointConfig(endpoint_class="embed", role="embed",
                        backend_engine="shim", skip_discovery=True)
    assert provider_for(ep) is LLAMACPP
    assert ep.skip_discovery, "the discovery switch is per-endpoint, not per-engine"


def test_providers_are_stateless_singletons():
    """One instance per engine is shared by every endpoint on the single event
    loop (CLAUDE.md: no locks, one thread). Per-request state on a provider
    would be a data race nothing in this suite could catch."""
    a = EndpointConfig(endpoint_class="a", role="a", backend_engine="vllm")
    b = EndpointConfig(endpoint_class="b", role="b", backend_engine="vllm")
    assert provider_for(a) is provider_for(b) is VLLM
    assert not getattr(VLLM, "__dict__", {}), (
        "a provider grew instance state — descriptors are class attributes and "
        "must stay that way")
    with pytest.raises(dataclasses.FrozenInstanceError):
        VLLM.descriptor.name = "something-else"      # type: ignore[misc]


# ---------------------------------------------------------------------------
# The asymmetry, declared
# ---------------------------------------------------------------------------

def test_capacity_discovery_is_asymmetric_on_purpose():
    """🚨 CLAUDE.md, "Engine-behaviour findings": llama.cpp /props yields real
    n_parallel and per-slot n_ctx; vLLM exposes only max_model_len and keeps
    --max-num-seqs off the API, so vLLM concurrency stays CONFIG-SEEDED with a
    drift alert. That is a property of the engines, not an oversight — if this
    ever reads as symmetric, someone has either found a new vLLM endpoint or
    made a number up."""
    assert LLAMACPP.descriptor.publishes_slot_count is True
    assert LLAMACPP.descriptor.publishes_slot_context is True
    assert VLLM.descriptor.publishes_slot_count is False
    assert VLLM.descriptor.publishes_context_ceiling is True


def test_vllm_capacity_never_claims_a_slot_count():
    """The parse must leave slots absent rather than invent one: `None` means
    "cannot tell", and health leaves the configured value standing on it."""
    report = VLLM.parse_capacity({"max_model_len": 131072})
    assert report.slots is None
    assert report.context_per_slot == 131072


def test_llamacpp_capacity_reads_per_slot_context_as_is():
    """The 2026-07-03 regression in one line: `default_generation_settings.n_ctx`
    is ALREADY per-slot, and dividing it again quartered the context-gate's view
    of every multi-slot endpoint. Applied end-to-end in test_context_discovery."""
    report = LLAMACPP.parse_capacity(
        {"default_generation_settings": {"n_parallel": 4, "n_ctx": 32768}})
    assert report.slots == 4
    assert report.context_per_slot == 32768


def test_a_props_body_with_nothing_in_it_reports_nothing_not_zero():
    report = LLAMACPP.parse_capacity({"default_generation_settings": {}})
    assert report.slots is None and report.context_per_slot is None


# ---------------------------------------------------------------------------
# The rule: capabilities, not engine names
# ---------------------------------------------------------------------------

#: Where an engine NAME is legitimately a value rather than a decision: the
#: catalog reads it out of models.yaml, config stores it, and the registry maps
#: it. Everywhere else, a behavioural branch must ask the descriptor.
_ENGINE_NAME_MAY_APPEAR_IN = {"config.py", "model_catalog.py"}


def _modules_comparing_backend_engine() -> list[str]:
    out = []
    for path in sorted(_PKG.rglob("*.py")):
        if path.parent.name == "providers" or path.name in _ENGINE_NAME_MAY_APPEAR_IN:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            left = node.left
            if isinstance(left, ast.Attribute) and left.attr == "backend_engine":
                out.append(f"{path.relative_to(_ROOT)}:{node.lineno}")
    return out


def test_no_module_branches_on_an_engine_name():
    """Every one of these used to be `ep_cfg.backend_engine == "vllm"`, and each
    one MEANT something narrower: "publishes prefix-cache counters", "mislabels a
    truncated tool call", "has no /props". A third engine would have had to be
    added to each site by hand, and the sites that were missed would have failed
    silently, in one direction only. Ask the descriptor."""
    offenders = _modules_comparing_backend_engine()
    assert not offenders, (
        "these compare backend_engine against a literal instead of reading a "
        f"ProviderDescriptor capability: {offenders}")


def test_every_descriptor_field_is_declared_by_every_provider():
    """A descriptor field a provider does not set reads as False by DEFAULT,
    which is indistinguishable from "nobody thought about it". Every provider
    must state every field in its own source — including the ones that are
    False, because "measured and it does not" and "never considered" are
    different claims and only one of them is safe to build on."""
    fields = {f.name for f in dataclasses.fields(LLAMACPP.descriptor)}
    fields -= {"name", "kind"}
    modules = sorted(
        p.name for p in (_PKG / "providers").glob("*.py")
        if p.name not in {"__init__.py", "base.py", "payload.py"})
    assert len(modules) >= 3, (
        f"the sweep found only {modules} — it has gone blind, not green")
    for module in modules:
        src = (_PKG / "providers" / module).read_text(encoding="utf-8")
        missing = sorted(f for f in fields if f"{f}=" not in src)
        assert not missing, (
            f"providers/{module} does not state {missing} — it would default to "
            f"False, which is a claim nobody made")


def test_the_interface_is_abstract_enough_to_extend():
    """The point of the split is the provider that does not exist yet
    (OpenRouter, roadmap Workstream A → D). Anything a remote provider must
    override has to be abstract, or it will inherit a local assumption."""
    assert Provider.__abstractmethods__ >= {
        "prepare_chat_payload", "discover_capacity"}
