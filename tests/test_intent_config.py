"""The intent vocabulary in CONFIG, and the negative constraint.

Two halves of the same question — *who owns the words a caller may use* — and
one line runs between them:

  A **profile** is shared, published, operator-written vocabulary, so it is
  expressed in declared capabilities and can never name an endpoint. A profile
  naming endpoints would be a second routing table to keep in step with
  ``endpoints:``, and it would break on every fleet spelled differently from the
  example's.

  An **intent** is one caller's words about one request, and it may name
  endpoints, because ``pin`` already does. ``exclude`` adds nothing a caller
  could not already say; it says the same kind of thing in the other direction.

Every guard here was observed going red by mutating the code it guards.
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest
import yaml

from roadstead import hooks, model_catalog
from roadstead.intent import (
    BUILTIN_PROFILES,
    REJECT_EXCLUDED,
    REJECT_UNKNOWN_ENDPOINT,
    ModelFacts,
    IntentError,
    Profile,
    parse_intent,
    resolve,
)

SRC = Path(model_catalog.__file__)


def facts(name, **kw) -> ModelFacts:
    base = dict(kind="chat", routed=True, healthy=True, max_slots=4,
                capabilities=frozenset({"streaming"}), context=32768)
    base.update(kw)
    base["capabilities"] = frozenset(base["capabilities"])
    return ModelFacts(endpoint=name, **base)


def catalog_from(*parts, tmp_path) -> "model_catalog.Catalog":
    """Load a catalog from YAML fragments, bypassing the per-path cache.

    Each fragment is dedented SEPARATELY: dedenting the concatenation takes the
    common prefix of both, which silently nests one stanza inside the other.
    """
    p = tmp_path / "models.yaml"
    p.write_text("\n".join(textwrap.dedent(part).strip("\n") for part in parts))
    return model_catalog.load_catalog(p, force=True)


BASE = """
    hosts: {llm: 192.0.2.11}
    providers:
      local: {engine: llama.cpp, host: llm, port: 8080}
    endpoints:
      tier1: {provider: local, kind: chat, slots: 4}
"""


@pytest.fixture(autouse=True)
def _clean_notices():
    hooks.clear_config_notices()
    yield
    hooks.clear_config_notices()


# ---------------------------------------------------------------------------
# The table comes out of config
# ---------------------------------------------------------------------------

def test_a_deployment_can_add_a_profile(tmp_path):
    cat = catalog_from(BASE, """
        intents:
          cheap-bulk:
            kind: chat
            prefer: capacity
            requires: [streaming]
            summary: Nobody is waiting.
    """, tmp_path=tmp_path)
    p = cat.intents["cheap-bulk"]
    assert p.prefer == "capacity"
    assert p.requires == frozenset({"streaming"})
    assert p.summary == "Nobody is waiting."
    # The disclosure that tells a caller this word is not ours.
    assert p.source == "models.yaml"


def test_the_table_is_layered_over_the_builtins_not_swapped_for_them(tmp_path):
    """🚨 A file that defines ONE profile has said nothing about the other nine.

    Mutation: return only the config profiles from ``_build_intents``. Nine
    published names vanish and every caller coding against them breaks, which
    no test asserting only the new name would have caught.
    """
    cat = catalog_from(BASE, """
        intents:
          cheap-bulk: {kind: chat, prefer: capacity}
    """, tmp_path=tmp_path)
    assert set(cat.intents) == set(BUILTIN_PROFILES) | {"cheap-bulk"}
    assert cat.intents["reasoning"] == BUILTIN_PROFILES["reasoning"]


def test_an_override_replaces_the_builtin_and_says_so(tmp_path):
    cat = catalog_from(BASE, """
        intents:
          reasoning:
            kind: chat
            prefer: context
            summary: On this fleet, reasoning means the deep one.
    """, tmp_path=tmp_path)
    assert cat.intents["reasoning"].prefer == "context"
    assert cat.intents["reasoning"].source == "models.yaml"
    assert cat.intents["reasoning"] != BUILTIN_PROFILES["reasoning"]


def test_no_intents_section_leaves_exactly_the_builtins(tmp_path):
    assert catalog_from(BASE, tmp_path=tmp_path).intents == BUILTIN_PROFILES


# ---------------------------------------------------------------------------
# 🚨 A profile can never name an endpoint
# ---------------------------------------------------------------------------

def test_no_profile_field_can_name_an_endpoint(tmp_path):
    """The rule, checked two ways: no such field exists, and writing one has no
    effect beyond a notice.

    Asserted against the FIELD LIST as well as the behaviour because the list is
    what a future change would edit — a new ``endpoints:`` key would sail past a
    test that only sent one and watched it be dropped.
    """
    assert not any(
        word in f
        for f in model_catalog._PROFILE_FIELDS
        for word in ("endpoint", "pin", "model", "class", "role", "alias")
    ), model_catalog._PROFILE_FIELDS

    cat = catalog_from(BASE, """
        intents:
          sneaky:
            kind: chat
            endpoints: [tier1]
            pin: tier1
    """, tmp_path=tmp_path)
    # The profile still loads — the unknown keys are dropped, not fatal — and
    # carries no trace of the endpoint it tried to name.
    assert "sneaky" in cat.intents
    assert not any("tier1" in str(v) for v in vars(cat.intents["sneaky"]).values())

    notice = _one_notice("intents.sneaky", "unknown_key")
    assert set(notice["keys"]) == {"endpoints", "pin"}


def test_the_profile_dataclass_has_no_endpoint_shaped_field():
    """Same rule one level down: ``Profile`` itself must stay expressible in
    capabilities, or the config allowlist is guarding a door in a wall that has
    a second door.
    """
    assert not any(
        word in name
        for name in Profile.__dataclass_fields__
        for word in ("endpoint", "pin", "exclude", "class", "role", "alias")
    ), sorted(Profile.__dataclass_fields__)


def test_a_profile_stanza_cannot_carry_exclude(tmp_path):
    """``exclude`` is a REQUEST field. In a profile it would be a table of
    endpoints under another name, so it is dropped and reported."""
    cat = catalog_from(BASE, """
        intents:
          avoidant: {kind: chat, exclude: [tier1]}
    """, tmp_path=tmp_path)
    assert cat.intents["avoidant"].requires == frozenset()
    assert "exclude" in _one_notice("intents.avoidant", "unknown_key")["keys"]


# ---------------------------------------------------------------------------
# An unusable stanza is refused, not offered
# ---------------------------------------------------------------------------

def _one_notice(subject: str, problem: str) -> dict:
    matches = [n for n in hooks.config_notices()
               if n["subject"] == subject and n["problem"] == problem]
    assert len(matches) == 1, [
        (n["subject"], n["problem"]) for n in hooks.config_notices()]
    return matches[0]


@pytest.mark.parametrize("stanza, bad", [
    ("{kind: chat, requires: [telepathy]}", "telepathy"),
    ("{kind: divination}", "divination"),
    ("{kind: chat, prefer: quality}", "quality"),
])
def test_an_unsatisfiable_stanza_is_refused_and_reported(stanza, bad, tmp_path):
    """🚨 Refused rather than registered, and the difference is where it sends
    the caller.

    A profile requiring a capability nothing can declare resolves to nothing on
    every request, and the caller reads "no endpoint satisfies requires=[…]" —
    a sentence about the fleet, for a fault in a config file. Refused, they get
    "unknown intent", which points at the vocabulary, where the fault is.

    Mutation: return the Profile anyway instead of None. The name reappears in
    the published table and the caller is sent to look at the fleet.
    """
    cat = catalog_from(BASE, f"intents:\n  odd: {stanza}", tmp_path=tmp_path)
    assert "odd" not in cat.intents
    assert bad in _one_notice("intents.odd", "unusable")["detail"]


def test_a_refused_override_does_not_leave_the_builtin_standing(tmp_path):
    """🚨 The operator wrote their own ``reasoning`` and it was unusable. Leaving
    ours in force under their name means they read their summary in the file and
    callers get ours on the wire — the exact declared-vs-in-force gap the
    management plane exists to close, created by us.
    """
    cat = catalog_from(BASE, """
        intents:
          reasoning: {kind: chat, requires: [telepathy]}
    """, tmp_path=tmp_path)
    assert "reasoning" not in cat.intents
    _one_notice("intents.reasoning", "unusable")


# ---------------------------------------------------------------------------
# The negative constraint
# ---------------------------------------------------------------------------

FLEET = [facts("tier1"), facts("tier2"), facts("tier3", capabilities={"reasoning"})]


def test_exclude_removes_a_candidate_and_says_why():
    res = resolve(parse_intent({"kind": "chat", "exclude": ["tier1"]}), FLEET)
    assert res.endpoint == "tier2"
    assert [(r.endpoint, r.reason) for r in res.rejected] == [
        ("tier1", REJECT_EXCLUDED)]


def test_exclude_alone_is_a_complete_declaration():
    """"Anything but tier1" is a whole statement of where a request may go.

    Requiring a positive declaration beside it would make a caller enumerate the
    endpoints it WOULD take, which is the routing table we refuse to put in a
    caller.

    Asserted in two steps on purpose. The mutation here — dropping `exclude`
    from the "must declare an intent" check — makes ``parse_intent`` RAISE, and
    a one-liner would have reported that as an uncaught IntentError from
    intent.py rather than as this guard failing. A guard whose red is an error
    somewhere else is half a guard.
    """
    try:
        intent = parse_intent({"exclude": ["tier1"]})
    except IntentError as exc:  # pragma: no cover - the mutation's path
        pytest.fail(f"`exclude` alone was refused as no declaration at all: {exc}")
    assert intent.exclude == frozenset({"tier1"})
    assert resolve(intent, FLEET).endpoint == "tier2"


def test_exclude_is_normalized_exactly_as_a_pin_is():
    """An exclusion compared literally would not exclude anything on a fleet
    where the caller knows the endpoint by an alias — and would route the
    request to precisely the endpoint it was written to avoid.

    Mutation: drop the ``normalize`` call on the exclusion. The alias stops
    matching, tier3 is served, and nothing anywhere reports a problem.
    """
    intent = parse_intent({"exclude": ["ALIAS"]},
                          normalize=lambda n: "tier3" if n == "ALIAS" else n)
    assert intent.exclude == frozenset({"tier3"})
    res = resolve(intent, FLEET)
    assert res.endpoint != "tier3"


def test_excluding_everything_names_the_exclusion_as_the_cause():
    res = resolve(parse_intent({"exclude": ["tier1", "tier2", "tier3"]}), FLEET)
    assert not res.ok
    assert {r.reason for r in res.rejected} == {REJECT_EXCLUDED}
    # The caller must be able to see it was their own constraint, not the fleet.
    assert "exclude" in res.failure_message()


def test_exclusion_is_reported_ahead_of_any_property_of_the_endpoint():
    """A caller who excluded an endpoint that is ALSO unrouted needs the reason
    they can act on, not a fact about our fleet that reads as though the
    exclusion had not been understood."""
    fleet = [facts("tier1"), facts("dead", routed=False)]
    res = resolve(parse_intent({"exclude": ["dead"]}), fleet)
    assert [(r.endpoint, r.reason) for r in res.rejected] == [
        ("dead", REJECT_EXCLUDED)]


def test_an_exclusion_naming_nothing_is_refused_not_ignored():
    """🚨 The whole doctrine of the field, and the one that looks wrong.

    Such an exclusion is NOT satisfied trivially: a name resolving to nothing is
    either "not in this fleet" or "in this fleet under a spelling you got
    wrong", and from here they are identical bytes. Serving the second sends the
    request to exactly the endpoint the exclusion existed to avoid, and reports
    success.

    Mutation: drop the refusal and carry it as a warning. tier1 is served, the
    caller sees a 200, and the guard that made the field worth having is gone.
    """
    res = resolve(parse_intent({"exclude": ["tierX"]}), FLEET)
    assert not res.ok
    assert res.unmatched_exclusions == ("tierX",)
    assert {r.reason for r in res.rejected} == {REJECT_UNKNOWN_ENDPOINT}
    msg = res.failure_message()
    assert "tierX" in msg and "exclude" in msg


def test_the_refusal_names_the_spelling_the_caller_ACTUALLY_SENT():
    """🚨 Found by running it, not by reading it.

    ``normalize`` lower-cases and strips, so an unresolvable name does NOT come
    back unchanged — a caller who wrote ``"tierX"`` was told ``'tierx'``, a word
    it never sent, in a message whose entire job is helping it find a typo. The
    same reason ``pin_as_written`` exists, missed once in the other direction.

    Mutation: report ``intent.exclude`` (the normalized set) instead of the
    pairing. The refusal still fires and still names something plausible, which
    is exactly why the suite was green while the message was wrong.
    """
    intent = parse_intent({"exclude": ["  TierX  "]}, normalize=str.lower)
    res = resolve(intent, FLEET)
    assert not res.ok
    assert res.unmatched_exclusions == ("TierX",)
    assert "TierX" in res.failure_message()
    # And the matching still happened on the normalized form.
    assert intent.exclude == frozenset({"tierx"})


def test_a_matched_exclusion_written_in_another_case_is_not_a_typo():
    """The other half: normalizing to something real must not then be reported
    as unmatched. Mutation: compare the WRITTEN name against the fleet."""
    res = resolve(parse_intent({"exclude": ["TIER1"]}, normalize=str.lower),
                  FLEET)
    assert res.ok and res.endpoint == "tier2"
    assert res.unmatched_exclusions == ()


def test_an_unknown_exclusion_is_refused_even_when_something_could_serve():
    """The point of the rule: there IS an answer available, and we refuse anyway
    rather than serve one the caller cannot trust."""
    assert not resolve(parse_intent({"intent": "chat", "exclude": ["typo"]}),
                       FLEET).ok


def test_a_pin_the_request_also_excludes_is_a_caller_contradiction():
    with pytest.raises(IntentError) as exc:
        parse_intent({"model": "tier1", "exclude": ["tier1"]})
    assert "exclude" in str(exc.value) and "tier1" in str(exc.value)


def test_a_pin_survives_an_exclusion_of_something_else():
    res = resolve(parse_intent({"model": "tier1", "exclude": ["tier2"]}), FLEET)
    assert res.endpoint == "tier1"


def test_exclude_must_be_a_list_of_names():
    with pytest.raises(IntentError):
        parse_intent({"exclude": {"not": "a list"}})
    # A bare string is the obvious single-item spelling and is accepted, the
    # same as `requires`.
    assert parse_intent({"exclude": "tier1"}).exclude == frozenset({"tier1"})


# ---------------------------------------------------------------------------
# The seam is actually used
# ---------------------------------------------------------------------------

def test_the_enriched_api_resolves_against_the_catalog_table_not_the_builtins():
    """🚨 The whole point of the section. ``enriched.py`` must pass the catalog's
    table to ``parse_intent``; passing nothing silently falls back to
    ``BUILTIN_PROFILES`` and a configured profile becomes "unknown intent".

    An AST check rather than a call count: the failure is a MISSING keyword
    argument, which no runtime assertion on the default catalog would notice,
    because the default catalog's table is a superset of the built-ins.
    """
    from roadstead import enriched

    tree = ast.parse(Path(enriched.__file__).read_text())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "parse_intent"]
    assert calls, "enriched.py no longer calls parse_intent"
    for call in calls:
        assert "profiles" in {kw.arg for kw in call.keywords}, (
            f"parse_intent at line {call.lineno} does not pass `profiles=` — a "
            "configured intent would resolve as unknown")


def test_the_shipped_example_exercises_the_config_path():
    """`models.yaml` is a worked example that cannot rot, so the parser must run
    on every boot rather than only in this file."""
    shipped = yaml.safe_load(
        (Path(model_catalog.__file__).parent / "models.yaml").read_text())
    assert shipped.get("intents"), "the example catalog no longer has intents:"
    cat = model_catalog.load_catalog(force=True)
    added = set(cat.intents) - set(BUILTIN_PROFILES)
    assert added, "the example's intents: no longer adds anything"
    assert all(cat.intents[n].source == "models.yaml" for n in added)
