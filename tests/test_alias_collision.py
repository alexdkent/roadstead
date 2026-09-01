"""An alias resolves to exactly one endpoint, and a second claim is REPORTED.

Transplanted from the origin monorepo's regression ledger
(`models-yaml-alias-collision`, 2026-07-30) during the ledger transplant, and it
turned out not to be a transplant at all: the origin **raised** on a duplicate
alias, and this repo resolved it **silently by file order**. The lesson survived
the extraction; the guard did not.

🚨 Neither behaviour is right on its own:

* Raising took a whole gateway down at import — an alias moved between two
  stanzas without deleting the old one crash-looped the process to FATAL. This
  repo's own rule elsewhere is that a typo must not stop a fleet booting.
* Resolving silently means an operator's name routes somewhere they did not
  intend, forever, with no signal — the declared-vs-in-force gap that every
  expensive failure in `docs/ledger.md` lives in.

So it loads, and it reports. Every guard below was observed going red.
"""
from __future__ import annotations

import textwrap

import pytest

from roadstead import hooks, model_catalog


@pytest.fixture(autouse=True)
def _clean_notices():
    hooks.clear_config_notices()
    yield
    hooks.clear_config_notices()


def _load(body: str, tmp_path):
    """Write a catalog whose `endpoints:` are the dedented `body`, re-indented.

    Re-indented explicitly rather than concatenated: dedenting the joined text
    takes the common prefix of both halves and quietly lands the endpoints at
    column zero, where they become top-level keys and `endpoints:` is empty. The
    catalog then loads fine with no endpoints and every notice assertion here
    passes vacuously — which is how the first version of this file "passed".
    """
    stanzas = textwrap.indent(textwrap.dedent(body).strip("\n"), "  ")
    p = tmp_path / "models.yaml"
    p.write_text(
        "providers:\n"
        "  local: {engine: llama.cpp, host: 192.0.2.11, port: 8080}\n"
        "endpoints:\n" + stanzas + "\n")
    cat = model_catalog.load_catalog(p, force=True)
    assert cat.endpoints, "the fixture wrote no endpoints — check the indenting"
    return cat


def _notices(problem: str) -> list[dict]:
    return [n for n in hooks.config_notices() if n["problem"] == problem]


def test_a_duplicate_alias_still_loads(tmp_path):
    """The origin RAISED here, at import, and took the process with it — a
    crash-loop to FATAL that needed a hand-cleared state file. A config typo
    must not be able to do that; same rule as a dropped `policy:` key.

    Mutation: raise on a duplicate, as the origin did. This fails with the
    ValueError rather than passing, which is the whole behavioural difference.
    """
    cat = _load("""
      composer: {provider: local, kind: chat, aliases: [tier3]}
      reasoner: {provider: local, kind: chat, aliases: [tier3]}
    """, tmp_path)
    # It loaded, and both endpoints are routable under their own names — the
    # collision costs one alias, never an endpoint.
    assert set(cat.endpoints) == {"composer", "reasoner"}
    assert cat.canonical("composer") == "composer"
    assert cat.canonical("reasoner") == "reasoner"


def test_a_duplicate_alias_resolves_by_file_order_and_says_so(tmp_path):
    """Mutation: drop the `duplicate` notice. Resolution is unchanged and the
    suite is otherwise green — which is exactly the state this repo was in."""
    cat = _load("""
      composer: {provider: local, kind: chat, aliases: [tier3, comp]}
      reasoner: {provider: local, kind: chat, aliases: [tier3]}
    """, tmp_path)
    assert cat.canonical("tier3") == "composer"

    n = _notices("duplicate")
    assert len(n) == 1, hooks.config_notices()
    assert n[0]["subject"] == "endpoints.aliases.tier3"
    assert set(n[0]["keys"]) == {"composer", "reasoner"}
    # 🚨 The winner is named in a FIELD, not only in the prose. "These two
    # collided" without saying which one wins leaves the operator to work out
    # our file order for themselves — and a consumer of the gap view reads the
    # field. Asserting only on the sentence let a mutation that emptied the
    # field survive, because the sentence interpolates the same value.
    assert n[0]["in_force"] == "composer"
    assert "composer" in n[0]["detail"]


def test_an_alias_shadowed_by_another_ENDPOINTS_CLASS_is_reported(tmp_path):
    """The other half of the ambiguity, and the one a reader misses.

    Class and role are registered after aliases so a class is always reachable
    by its own name — deliberate, and it stays. But the alias it silently kills
    was still written by somebody.

    Mutation: drop the second loop. The alias is dead and nothing says so.
    """
    cat = _load("""
      reasoner: {provider: local, kind: chat}
      shadowy:  {provider: local, kind: chat, aliases: [reasoner]}
    """, tmp_path)
    assert cat.canonical("reasoner") == "reasoner"

    n = _notices("shadowed")
    assert len(n) == 1, hooks.config_notices()
    assert n[0]["subject"] == "endpoints.shadowy.aliases"
    assert n[0]["keys"] == ["reasoner"]
    assert n[0]["in_force"] == "reasoner"


def test_an_endpoints_own_name_as_its_own_alias_is_not_a_collision(tmp_path):
    """🚨 The false positive that would make the notice worthless.

    An alias equal to its own class maps to itself, which is harmless and
    common. Reporting it would put a line in the operator's gap view on every
    boot, and a view that cries wolf is not read.

    Mutation: report whenever `by_name[alias] != e.name` is false too, i.e. drop
    the `winner != e.name` test. Every self-alias fires.
    """
    _load("""
      solo: {provider: local, kind: chat, aliases: [solo, spare]}
    """, tmp_path)
    assert _notices("shadowed") == []
    assert _notices("duplicate") == []


def test_a_clean_catalog_reports_nothing(tmp_path):
    _load("""
      one: {provider: local, kind: chat, aliases: [uno]}
      two: {provider: local, kind: chat, aliases: [dos]}
    """, tmp_path)
    assert hooks.config_notices() == []


def test_the_shipped_example_has_no_alias_collisions():
    """`models.yaml` is a worked example that cannot rot: if the example itself
    tripped this, every deployment would learn to ignore the notice."""
    hooks.clear_config_notices()
    model_catalog.load_catalog(force=True)
    assert _notices("duplicate") == []
    assert _notices("shadowed") == []


def test_a_duplicate_is_reported_once_not_twice(tmp_path):
    """Both loops can see the same alias. Reporting it from each would double
    every line in the gap view, and a duplicated diagnostic reads as two faults.

    Mutation: drop the `continue` that skips an already-reported alias.
    """
    _load("""
      a: {provider: local, kind: chat, aliases: [shared]}
      b: {provider: local, kind: chat, aliases: [shared]}
    """, tmp_path)
    assert len(hooks.config_notices()) == 1, hooks.config_notices()
