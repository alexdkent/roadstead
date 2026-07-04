"""Regression: apply_discovered_props() must not re-divide an already
per-slot n_ctx.

Found 2026-07-03 while investigating why the Inference page showed 8192
context for composer/classify (both configured --ctx-size 131072 --parallel
4 = 32768/slot): default_generation_settings.n_ctx in llama.cpp's /props IS
the per-slot value already, but the old code unconditionally divided it by
n_parallel again, silently quartering context_per_slot for every multi-slot
llama.cpp endpoint. That value feeds the context-gate admission check, so
this could wrongly reject requests that actually fit.
"""
from __future__ import annotations

from originfleet.llmproxy.config import EndpointConfig
from originfleet.llmproxy.health import Health


def _ep(max_slots: int = 4) -> EndpointConfig:
    return EndpointConfig(endpoint_class="classify", role="qwen-classify", max_slots=max_slots)


def test_llama_cpp_per_slot_n_ctx_not_redivided():
    """default_generation_settings.n_ctx is per-slot already (confirmed live
    against b9849: --ctx-size 131072 --parallel 4 reports n_ctx=32768, not
    131072) -- must be taken as-is, not divided by n_parallel again."""
    # Health.__new__ skips __init__ (no ProxyState needed) -- discovered
    # n_parallel is set to match ep.max_slots below so the slot-reconcile
    # branch (which DOES touch self.state.cost_model/budget_mgr) is never
    # entered; only the context-size path under test runs.
    health = Health.__new__(Health)
    ep = _ep(max_slots=4)
    props = {
        "default_generation_settings": {"n_parallel": 4, "n_ctx": 32768},
    }
    health.apply_discovered_props("classify", ep, props)
    assert ep.max_slots == 4
    assert ep.context_per_slot == 32768  # NOT 32768 // 4 == 8192


def test_top_level_n_ctx_fallback_still_divided():
    """The top-level props["n_ctx"] fallback (older/different builds) is a
    distinct, unconfirmed-semantics path -- kept dividing by n_parallel since
    no live build was found that populates it, so this pins the existing
    (pre-fix) behavior rather than asserting it's definitely correct."""
    health = Health.__new__(Health)
    ep = _ep(max_slots=4)
    props = {"total_slots": 4, "n_ctx": 131072}
    health.apply_discovered_props("classify", ep, props)
    assert ep.max_slots == 4
    assert ep.context_per_slot == 32768


def test_single_slot_per_slot_n_ctx_unaffected():
    """max_slots=1 made the bug invisible (n // 1 == n) -- pin that the fix
    doesn't change single-slot behavior."""
    health = Health.__new__(Health)
    ep = _ep(max_slots=1)
    props = {
        "default_generation_settings": {"n_parallel": 1, "n_ctx": 8192},
    }
    health.apply_discovered_props("classify", ep, props)
    assert ep.context_per_slot == 8192
