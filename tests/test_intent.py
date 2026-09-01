"""``intent.py`` — the fifth pure-computation module, tested as one.

No proxy, no sockets, no catalog: every test here builds :class:`ModelFacts` by
hand and asserts what ``resolve`` does with them. That is the point of keeping
the decision separable from the lookup — a fleet shape that would take an hour
to stand up is four lines of dataclass here.

Each guard below was observed going red by mutating the code it guards; the
mutation is named in the docstring where it is not obvious.
"""
from __future__ import annotations

import pytest

from roadstead.config import normalize_endpoint
from roadstead.intent import (
    BUILTIN_PROFILES,
    DEFAULT_PREFERENCE,
    KNOWN_CAPABILITIES,
    PREFERENCES,
    REJECT_CONTEXT_TOO_SMALL,
    REJECT_MISSING_CAPABILITY,
    REJECT_NOT_ROUTED,
    REJECT_UNKNOWN_ENDPOINT,
    REJECT_WRONG_KIND,
    Intent,
    IntentError,
    ModelFacts,
    parse_intent,
    resolve,
)


def facts(name, **kw) -> ModelFacts:
    base = dict(kind="chat", routed=True, healthy=True, max_slots=4,
                capabilities=frozenset({"streaming"}), context=32768)
    base.update(kw)
    if isinstance(base.get("capabilities"), (set, list, tuple)):
        base["capabilities"] = frozenset(base["capabilities"])
    return ModelFacts(endpoint=name, **base)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_a_request_that_declares_nothing_is_refused_with_the_vocabulary():
    """An empty intent is a caller error, and the error has to teach.

    The profile table is extensible per deployment, so a caller literally cannot
    know the vocabulary without being told it — an error that says only
    'unknown' sends them to read somebody else's config file.
    """
    with pytest.raises(IntentError) as exc:
        parse_intent({})
    assert "intent" in str(exc.value) and "reasoning" in str(exc.value)


def test_an_unknown_capability_is_refused_rather_than_matching_nothing():
    """🚨 The closed-set half of the KNOWN_CAPABILITIES asymmetry.

    `requires: ["visoin"]` would otherwise resolve to 'no endpoint can do this'
    — an empty result indistinguishable from a fleet with no vision model, so
    the caller goes hunting for hardware it already has.
    """
    with pytest.raises(IntentError) as exc:
        parse_intent({"requires": ["visoin"]})
    assert "visoin" in str(exc.value) and "vision" in str(exc.value)


def test_an_explicit_prefer_beats_the_profile_that_carries_one():
    """A profile is a bundle of defaults, not a lock.

    Mutation: make the profile win (`prefer = profile.prefer or prefer`) and
    this fails — which is the `_POLICY_PASSTHROUGH` failure in a new costume, a
    knob that is settable and inert.
    """
    assert BUILTIN_PROFILES["fast-chat"].prefer == "latency"
    got = parse_intent({"intent": "fast-chat", "prefer": "capacity"})
    assert got.prefer == "capacity"
    assert parse_intent({"intent": "fast-chat"}).prefer == "latency"


def test_a_profile_and_extra_requirements_compose():
    got = parse_intent({"intent": "reasoning", "requires": ["vision"]})
    assert got.requires == frozenset({"reasoning", "vision"})


def test_a_pin_is_normalized_but_echoed_as_written():
    """🚨 Both halves matter and they pull in opposite directions.

    NORMALIZED, or `model: "chat"` is refused on a fleet where `chat` is exactly
    how half the callers' configs spell tier2. ECHOED AS WRITTEN, or the
    disclosure hands the caller a canonical name it has never used — which reads
    as a substitution rather than as the same endpoint under its real name.
    """
    got = parse_intent({"model": "chat"}, normalize=normalize_endpoint)
    assert got.pin == normalize_endpoint("chat") == "tier2"
    assert got.pin_as_written == "chat"
    assert got.declared == "chat"


@pytest.mark.parametrize("body,bad", [
    ({"intent": "reasonning"}, "reasonning"),
    ({"model": "x", "kind": "chatt"}, "chatt"),
    ({"model": "x", "prefer": "quality"}, "quality"),
    ({"model": "x", "min_context": "lots"}, "min_context"),
    ({"model": "x", "requires": 7}, "requires"),
])
def test_every_malformed_declaration_names_what_was_wrong(body, bad):
    with pytest.raises(IntentError) as exc:
        parse_intent(body)
    assert bad in str(exc.value)


def test_quality_is_not_a_preference():
    """🚨 Deliberate absence. We cannot measure model quality from inside a
    gateway, and a key spelled `quality` that resolved to 'biggest context'
    would be read by a caller as a promise about answers."""
    assert "quality" not in PREFERENCES
    assert DEFAULT_PREFERENCE in PREFERENCES


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def test_a_pin_narrows_and_does_not_reject_the_rest_of_the_fleet():
    """The rejection list is a diagnostic; one row per endpoint in the fleet is
    not. This grew with the catalog and buried the row that explained the
    refusal."""
    res = resolve(Intent(pin="tier2"),
                  [facts("tier1"), facts("tier2"), facts("tier3")])
    assert res.endpoint == "tier2"
    assert res.rejected == ()


def test_a_pin_at_a_name_that_does_not_exist_says_so():
    res = resolve(Intent(pin="tier9", pin_as_written="tier9"), [facts("tier1")])
    assert not res.ok
    assert [r.reason for r in res.rejected] == [REJECT_UNKNOWN_ENDPOINT]
    assert "unknown model" in res.failure_message()


def test_a_pin_that_cannot_meet_the_requirement_is_refused_not_substituted():
    """🚨 The load-bearing half of "a pin is a constraint on routing".

    Serving this from the vision-capable endpoint next door would be the silent
    substitution the whole API exists to make impossible — the caller pinned
    tier2 and would have no way to tell it got tier3.
    """
    res = resolve(
        Intent(pin="tier2", requires=frozenset({"vision"})),
        [facts("tier2", capabilities={"streaming"}),
         facts("tier3", capabilities={"streaming", "vision"})])
    assert not res.ok
    assert [r.reason for r in res.rejected] == [REJECT_MISSING_CAPABILITY]
    assert "vision" in res.failure_message()


def test_an_unrouted_endpoint_is_never_resolved_to():
    """`planned` and `retired` document a name without serving it — which is
    exactly what the example catalog's two remote endpoints are, so this branch
    runs on every boot."""
    res = resolve(Intent(), [facts("spill-chat", routed=False)])
    assert not res.ok
    assert res.rejected[0].reason == REJECT_NOT_ROUTED


def test_kind_filters_and_its_rejections_are_kept_out_of_the_message():
    res = resolve(Intent(kind="chat"),
                  [facts("embed", kind="embed"), facts("rerank", kind="rerank")])
    assert not res.ok
    assert {r.reason for r in res.rejected} == {REJECT_WRONG_KIND}
    # Every embedder fails every chat request, forever. Listing them buries the
    # near-miss the caller is actually looking for.
    assert res.near_misses == ()
    assert "embed" not in res.failure_message()


def test_a_context_of_zero_admits_because_it_means_not_known():
    """Same convention as `scheduler._fits_context`: a ceiling we have not
    discovered is not a ceiling of zero."""
    res = resolve(Intent(min_context=100_000),
                  [facts("undiscovered", context=0)])
    assert res.endpoint == "undiscovered"
    res2 = resolve(Intent(min_context=100_000), [facts("small", context=8192)])
    assert not res2.ok
    assert res2.rejected[0].reason == REJECT_CONTEXT_TOO_SMALL


# ---------------------------------------------------------------------------
# Ranking — the two doctrine rules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prefer", sorted(PREFERENCES))
def test_a_real_cost_endpoint_sorts_last_under_every_preference(prefer):
    """🚨 THE local-first rule, and it is checked under every preference on
    purpose.

    The remote endpoint below wins on every axis: faster, emptier, deeper
    context, cheaper per token. It still must not be chosen, because intent
    resolution preferring remote capacity would make traffic leave the machine
    on the ORDINARY path rather than only when local said no — remote capacity
    is overflow, and the operator would discover the difference on an invoice.

    Mutation: drop the `real_cost` term from the sort key and every parameter of
    this test fails.
    """
    local = facts("local", typical_ms=5000.0, max_slots=1, in_flight=1,
                  context=8192, input_usd_per_mtok=9.0, output_usd_per_mtok=9.0)
    remote = facts("remote", typical_ms=50.0, max_slots=99, context=1_000_000,
                   input_usd_per_mtok=0.01, output_usd_per_mtok=0.01,
                   real_cost=True)
    res = resolve(Intent(prefer=prefer), [remote, local])
    assert res.endpoint == "local"
    # …but still REACHABLE when nothing local satisfies the requirement, which
    # is what makes it overflow rather than forbidden.
    only_remote = resolve(Intent(prefer=prefer), [remote])
    assert only_remote.endpoint == "remote"


def test_an_unmeasured_endpoint_is_treated_as_slow_not_fast():
    """🚨 The obvious ascending sort ranks the endpoint we know LEAST about
    first — preferring a backend because there is no evidence about it.

    Mutation: `return self.typical_ms` from `latency_key` and this fails.
    """
    measured = facts("measured", typical_ms=900.0)
    never = facts("never", typical_ms=0.0)
    assert resolve(Intent(prefer="latency"),
                   [never, measured]).endpoint == "measured"


def test_cold_start_is_deterministic_rather_than_arbitrary():
    """With no samples anywhere, everything ties and the name breaks it — so two
    identical requests do not oscillate between equal candidates."""
    fleet = [facts("b"), facts("c"), facts("a")]
    first = resolve(Intent(prefer="latency"), fleet)
    second = resolve(Intent(prefer="latency"), list(reversed(fleet)))
    assert first.endpoint == second.endpoint == "a"
    assert first.ranked == second.ranked


def test_health_breaks_a_tie_but_never_filters():
    """An unhealthy endpoint that is the ONLY candidate still wins — what
    happens to a request aimed at a sick backend is owned by the circuit breaker
    and failover, and a second opinion here is one of them being wrong."""
    sick = facts("sick", healthy=False)
    well = facts("well", healthy=True)
    assert resolve(Intent(), [sick, well]).endpoint == "well"
    assert resolve(Intent(), [sick]).endpoint == "sick"


def test_each_preference_orders_by_the_thing_it_names():
    fleet = [
        facts("fast", typical_ms=100.0, max_slots=1, in_flight=1, context=8192,
              input_usd_per_mtok=5.0),
        facts("roomy", typical_ms=9000.0, max_slots=8, context=16384,
              input_usd_per_mtok=3.0),
        facts("deep", typical_ms=9000.0, max_slots=1, in_flight=1,
              context=1_000_000, input_usd_per_mtok=4.0),
        facts("cheap", typical_ms=9000.0, max_slots=1, in_flight=1,
              context=8192, input_usd_per_mtok=0.1),
    ]
    assert resolve(Intent(prefer="latency"), fleet).endpoint == "fast"
    assert resolve(Intent(prefer="capacity"), fleet).endpoint == "roomy"
    assert resolve(Intent(prefer="context"), fleet).endpoint == "deep"
    assert resolve(Intent(prefer="cost"), fleet).endpoint == "cheap"
    # balanced: somewhere with room now beats somewhere fast but full, because a
    # queue wait is latency that `typical_ms` does not contain.
    assert resolve(Intent(prefer="balanced"), fleet).endpoint == "roomy"


def test_the_alternatives_are_the_fallthrough_order():
    """`/rs/v1/plan` publishes these, so a caller can understand a decision
    rather than only receive it."""
    res = resolve(Intent(prefer="latency"), [
        facts("c", typical_ms=300.0),
        facts("a", typical_ms=100.0),
        facts("b", typical_ms=200.0),
    ])
    assert res.ranked == ("a", "b", "c")


# ---------------------------------------------------------------------------
# The profile table
# ---------------------------------------------------------------------------

def test_no_builtin_profile_names_an_endpoint():
    """🚨 Profiles are expressed in DECLARED CAPABILITIES only.

    One that named endpoints would be a third routing table to keep in step with
    models.yaml, and it would break on every fleet whose classes are not spelled
    like the example's — which is every fleet but ours.
    """
    for p in BUILTIN_PROFILES.values():
        assert p.requires <= KNOWN_CAPABILITIES, p
        assert p.prefer in PREFERENCES, p
        assert p.summary, f"{p.name} has no published summary"


def test_the_shipped_catalog_can_satisfy_every_chat_profile():
    """The example catalog is also the shipped DEFAULT, so a profile no shipped
    endpoint can serve is a vocabulary that does not work out of the box."""
    from roadstead.config import DEFAULT_ENDPOINTS

    fleet = [ModelFacts(endpoint=name, kind=cfg.kind,
                        capabilities=cfg.capabilities,
                        context=cfg.context_per_slot, max_slots=cfg.max_slots)
             for name, cfg in DEFAULT_ENDPOINTS.items()]
    unservable = [p.name for p in BUILTIN_PROFILES.values()
                  if not resolve(parse_intent({"intent": p.name}), fleet).ok]
    assert not unservable, (
        f"the shipped catalog cannot serve {unservable} — a built-in profile "
        f"that no shipped endpoint satisfies is a vocabulary that does not work "
        f"out of the box")
