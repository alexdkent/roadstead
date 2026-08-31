"""The boxa analyst endpoint is named `tier2-analyst`; `creative` is its RETIRED name.

The 2026-08-18 rename moved the catalog stanza KEY to `tier2-analyst` and left
two `creative` strings behind — `endpoint_class` and `role`. Both are
deliberate and load-bearing, but nothing said so, and on 2026-08-23 a session
read `/v1/status` and the timeouts ledger (which report the raw
`endpoint_class`) and reported four stalls against an endpoint called
`creative` — a name the operator had retired.

This file pins the COUPLING rather than the string. A future migration that
renames `endpoint_class` everywhere it is persisted or hardcoded will pass; a
partial one — the only kind that actually hurts — fails here with the checklist.

What each `creative` is for:
  * `role`           — the PRE-DISCOVERY wire model id (`effective_model_id` is
    `served_model_id or role`). The boxa container is launched `-a creative`.
  * `endpoint_class` — the proxy's endpoint identity, PERSISTED into every
    proxy_completions / proxy_timeouts row and hardcoded in kv4.
  * the `creative` ALIAS — live callers pass it as the model name today.
"""

from __future__ import annotations

import pytest

from roadstead import cache_stats, model_catalog, usage_rates

NAME = "tier2-analyst"


@pytest.fixture(scope="module")
def entry():
    e = model_catalog.load_catalog().models.get(NAME)
    assert e is not None, (
        f"the catalog stanza key IS the endpoint's name; {NAME!r} must exist")
    return e


def test_the_endpoints_name_is_the_stanza_key(entry):
    assert entry.name == NAME


def test_creative_remains_a_routable_alias(entry):
    """Live callers pass `creative` as the model name RIGHT NOW — investor's
    INVESTOR_RESEARCH_ROLE default, knowledge_v3's KV3_EPISODE_MODEL /
    KV3_VERIFY_MODEL defaults, a kv4 budget row, the edward-sandbox skel
    scripts, and the infra/nasbox-vllm-xpu benches. Dropping the alias while
    "cleaning up the retired name" 404s every one of them."""
    assert "creative" in entry.aliases, entry.aliases
    assert NAME in entry.aliases, (
        "the stanza KEY never enters the dispatch map — build_role_aliases "
        "iterates aliases only, so the current name must be listed too "
        "(measured 2026-08-18: with the key alone, a call for tier2-analyst "
        "got 404 unknown_endpoint)")


def test_creative_remains_a_served_model_name(entry):
    """The boxa llama.cpp container is launched `-a creative`. This list is what
    the naming doctrine compares against the real server, so the retired name
    stays here for as long as the container is started that way."""
    assert "creative" in entry.served_model_names, entry.served_model_names


def test_endpoint_class_still_prices_for_billing(entry):
    """Whatever `endpoint_class` is, usage_rates MUST resolve it.

    Renaming the class is a data migration: `proxy_completions` rows carry the
    string they were logged with, and usage_rates says in its own comments that
    it must keep resolving retired strings or billing queries over historical
    data raise on an unknown endpoint. If you rename the class, add the new
    string to `_ENDPOINT_CLASS` and KEEP the old one.
    """
    assert usage_rates._ENDPOINT_CLASS.get(entry.endpoint_class, "missing") != "missing", (
        f"endpoint_class {entry.endpoint_class!r} has no usage_rates mapping — "
        f"cost attribution for this endpoint is silently unpriced")


def test_endpoint_class_is_a_production_chat_class_for_kv4(entry):
    """kv4 hardcodes the class set it may dispatch chat traffic at. A rename
    that misses this frozenset does not error — kv4 just quietly stops counting
    this endpoint as an independent production model, which is a gate changing
    behaviour with nothing red."""
    from originfleet.agents.knowledge_store.roles import PRODUCTION_CHAT_CLASSES
    assert entry.endpoint_class in PRODUCTION_CHAT_CLASSES, (
        f"{entry.endpoint_class!r} missing from kv4 PRODUCTION_CHAT_CLASSES "
        f"({sorted(PRODUCTION_CHAT_CLASSES)})")


def test_the_display_label_shows_the_CURRENT_name(entry):
    """The one operator surface that is already right, and must stay right: the
    Inference page labels the class from the catalog KEY, so it reads
    `tier2-analyst` even though the raw class is the retired string."""
    labels = cache_stats.chat_endpoint_labels()
    assert labels.get(entry.endpoint_class) == NAME, labels
