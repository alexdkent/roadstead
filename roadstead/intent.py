"""Intent — what a caller ASKED FOR, resolved to an endpoint that can serve it.

**The fifth pure-computation module.** No I/O, no clock, no state: it is handed
an immutable snapshot of what every candidate endpoint is and is currently like
(:class:`ModelFacts`), and returns which one should serve and why every other
one did not. Everything that has to look something up — the catalog, the health
poller, the scheduler's occupancy, the learned latency distribution, the price
book — happens in ``enriched.py``, which assembles the facts and calls in here.
Keeping the decision separable from the lookup is what makes it testable against
a fleet that does not exist.

---

## The idea

Today a caller names an endpoint class (``tier3``) or one of its aliases. That
is a *pin*: it says which box to use, which means the caller has to know what the
fleet looks like and re-learn it every time the fleet changes. The roadmap's
model abstraction inverts it — **a caller declares a capability and Roadstead
owns the choice of model, provider and moment** — without taking the pin away,
because a caller that genuinely must have one specific model is not wrong.

So there are two shapes and they are the *same* decision:

* **an intent** (``intent: "reasoning"``, or a bare set of requirements) — a
  constraint on what the endpoint must be able to do. Roadstead picks.
* **a pin** (``model: "tier3"``) — a constraint on which endpoint it is. There
  is only one candidate, and it still has to satisfy the requirements: a pin at
  a text-only endpoint with ``requires: ["vision"]`` is a contradiction and is
  refused rather than quietly served.

🚨 **Resolution is not substitution.** Choosing ``tier2`` for
``intent: "fast-chat"`` is Roadstead doing the job it was asked to do; nothing
was promised and nothing was swapped. Substitution is what happens *later* to a
request whose endpoint was already decided — failover (this backend is DOWN) or
spill (this backend is FULL) — and only that is disclosed as a substitution.
Conflating the two would make ``substituted: true`` fire on every intent-routed
call and therefore mean nothing.

---

## Two rules that look arbitrary until they bite

🚨 **A real-cost endpoint always sorts last, under every preference.** Local
capacity is the design center and remote capacity is *overflow*
(``docs/roadmap.md``). If intent resolution were allowed to prefer a remote
endpoint because it happened to be faster or emptier, "intent" would become a
back door around the entire spill doctrine: traffic would leave the machine on
the ordinary path rather than only when local capacity said no, and the operator
would find out on an invoice. So the money question is answered *before* the
preference is consulted, not by it. A remote endpoint is reachable through an
intent only when no local candidate satisfies the requirements at all — which is
exactly what "overflow" means.

🚨 **An endpoint with no latency samples is treated as SLOW, not as fast.**
``typical_ms`` of 0.0 means *we have never measured this*, and the obvious
ascending sort would rank the endpoint we know least about first — preferring a
backend *because* there is no evidence about it. Unknown sorts as ``inf``. At
cold start every candidate is unknown, everything ties, and the stable name
tie-break decides; that is deterministic and honest, which is the most a
resolver can be before it has seen a single request.

---

## Health is a ranking key, never a filter

An unhealthy endpoint is still a legitimate answer — it may be the only one that
can serve the request at all, and what happens to a request aimed at a sick
backend is already owned end to end by ``health.py`` (the circuit breaker) and
``failover.py`` (the opted-in degrade). Filtering here would put a *second*
opinion about backend health on the request path, and two places deciding the
same thing separately is how one of them ends up wrong. So health only breaks a
tie between candidates that could both serve.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: The capability names an endpoint may declare in ``models.yaml``, and the only
#: ones a caller may require.
#:
#: 🚨 This is a CLOSED set on the request side and an OPEN one on the config
#: side, deliberately. A caller asking for a capability nobody has ever heard of
#: is a typo that would otherwise resolve to "no endpoint can do this" — an
#: empty result that looks exactly like a fleet with no vision model, so the
#: caller would go hunting for hardware it already has. An operator declaring an
#: unknown capability gets a load-time warning and keeps serving, because the
#: catalog is theirs and a capability we have not thought of yet is a thing they
#: are allowed to write down.
KNOWN_CAPABILITIES: frozenset[str] = frozenset({
    "vision",
    "reasoning",
    "streaming",
    "tool_calling",
    "structured_output",
})

#: What a routable unit of capacity *is*. Mirrored from the catalog's ``kind``.
KNOWN_KINDS: frozenset[str] = frozenset({"chat", "embed", "rerank"})

#: How to order candidates that all satisfy the requirements.
#:
#: 🚨 There is deliberately no ``quality`` preference. We cannot measure model
#: quality from inside a gateway, and a preference key that resolves to "the one
#: with the biggest context" while being spelled "quality" is a claim the code
#: cannot support — the caller would reasonably read it as a promise about
#: answers. Every name below is something the proxy actually observes.
PREFERENCES: frozenset[str] = frozenset({
    "balanced",   # somewhere with room now; then the fastest we have measured
    "latency",    # lowest learned median latency
    "capacity",   # most free slots — the shortest wait, not the fastest model
    "context",    # largest context window
    "cost",       # cheapest per token (local is free, so local still wins)
})

DEFAULT_PREFERENCE = "balanced"


@dataclass(frozen=True)
class Profile:
    """A named bundle of requirements + a preference. The caller-facing word.

    Profiles are how ``intent: "reasoning"`` becomes something the resolver can
    act on, and they are expressed **entirely in terms of declared capabilities**
    — never a list of endpoint names. A profile that named endpoints would be a
    third routing table to keep in step with ``models.yaml``, and it would break
    the moment somebody's fleet used different class names, which is every fleet
    but the example one.
    """

    #: 🚨 There is deliberately NO ``exclude`` here, although :class:`Intent`
    #: has one. A profile is *shared, static vocabulary* — it is published to
    #: every caller and, once it is configurable, written by an operator — so a
    #: profile naming endpoints is the third routing table this class exists to
    #: refuse. An intent's ``exclude`` is one caller's own words about one
    #: request, exactly as ``pin`` already is, and names no endpoint the caller
    #: could not already have named. The line is between *config* and *request*,
    #: not between positive and negative.

    name: str
    kind: str = "chat"
    requires: frozenset[str] = frozenset()
    prefer: str = DEFAULT_PREFERENCE
    #: One line, published on ``GET /rs/v1/models`` so a caller can discover the
    #: vocabulary instead of reading this file.
    summary: str = ""
    #: Where this profile came from — ``"builtin"`` or the config file that
    #: overrode it. Published, because a caller whose fleet redefined
    #: ``reasoning`` is reading a word that no longer means what our docs say.
    source: str = "builtin"


#: The built-in profiles. They ship so the abstraction works out of the box
#: against any catalog, including one whose endpoints are named nothing like the
#: example's — which is why every one of them is expressed in capabilities the
#: catalog declares rather than in class names only our example uses.
#:
#: ⚠️ Built-in is currently ALL they are: ``resolve_profile`` and ``parse_intent``
#: both take a table so a deployment's own can be threaded through, but nothing
#: reads one out of ``models.yaml`` yet. Open in ``docs/roadmap.md`` under C. The
#: seam is here rather than added later on purpose — a caller-facing vocabulary
#: that starts hard-coded and becomes configurable breaks anyone who read it off
#: the source instead of off ``GET /rs/v1/models``.
BUILTIN_PROFILES: dict[str, Profile] = {
    p.name: p for p in (
        Profile("fast-chat", kind="chat", prefer="latency",
                summary="A conversational turn where someone is waiting."),
        Profile("chat", kind="chat", prefer="balanced",
                summary="An ordinary chat completion, no special requirement."),
        Profile("reasoning", kind="chat", requires=frozenset({"reasoning"}),
                prefer="balanced",
                summary="A model that can think before it answers."),
        Profile("vision", kind="chat", requires=frozenset({"vision"}),
                prefer="balanced",
                summary="A model that can read an image."),
        Profile("tools", kind="chat", requires=frozenset({"tool_calling"}),
                prefer="balanced",
                summary="A model that can call tools."),
        Profile("structured", kind="chat",
                requires=frozenset({"structured_output"}), prefer="balanced",
                summary="A model that can be held to a schema or a grammar."),
        Profile("long-context", kind="chat", prefer="context",
                summary="The deepest context window available."),
        Profile("embed", kind="embed", prefer="balanced",
                summary="Text embeddings."),
        Profile("rerank", kind="rerank", prefer="balanced",
                summary="Cross-encoder reranking."),
    )
}


# ---------------------------------------------------------------------------
# What the resolver is allowed to know
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelFacts:
    """One candidate endpoint, frozen at the moment the decision is made.

    Everything here is either config (what the endpoint IS) or a cheap in-memory
    read (what it is currently LIKE). Nothing in this dataclass may require a
    network call to populate — the enriched API answers on the request path, and
    a resolver that probed a backend to decide where to send a request would be
    paying a round trip to save one.
    """

    endpoint: str
    kind: str = "chat"
    provider: str = ""
    engine: str = ""
    context: int = 0
    capabilities: frozenset[str] = frozenset()
    #: ``active``/``on_demand`` endpoints are routed; ``planned``/``retired``
    #: document a name without serving it and can never be resolved to.
    routed: bool = True
    healthy: bool = True
    max_slots: int = 0
    in_flight: int = 0
    queued: int = 0
    #: Learned median latency for this endpoint, ms. 0.0 means NO SAMPLES — see
    #: the module docstring; it is treated as slow, never as fast.
    typical_ms: float = 0.0
    input_usd_per_mtok: float = 0.0
    output_usd_per_mtok: float = 0.0
    #: ``TokenPrice.real`` — whether this endpoint's price is an INVOICE rather
    #: than an avoided cost. The local-first sort key. Read off the price, never
    #: off "is the provider remote": an operator may declare a real price on a
    #: local endpoint they are internally charged for, and the money question is
    #: the same one either way.
    real_cost: bool = False
    #: ``TokenPrice.source`` / ``.detail`` — WHERE the price came from, and the
    #: free-text provenance behind it. Carried so that `GET /rs/v1/models`
    #: publishes the same price block a chat envelope does; the SDK has one
    #: `Price` view for both, and until 2026-09-01 this row was missing two of
    #: its five fields, so `ModelInfo.price.source` read empty for every
    #: endpoint. `source` is the field that separates a price the provider
    #: PUBLISHED from one we imputed — the same "measured or guessed"
    #: distinction the management plane reports for slot counts, and the one a
    #: caller choosing where to send work most needs.
    #:
    #: 🚨 Spelled here as a literal because this module imports nothing from the
    #: package and that is worth more than sharing a constant. The default is
    #: `spend.SOURCE_IMPUTED`, and `test_intent.py` pins the two together.
    price_source: str = "imputed"
    price_detail: str = ""

    @property
    def free_slots(self) -> int:
        return max(0, self.max_slots - self.in_flight)

    @property
    def latency_key(self) -> float:
        """``typical_ms``, with *unknown* sorting last rather than first."""
        return self.typical_ms if self.typical_ms > 0 else math.inf

    @property
    def price_key(self) -> float:
        return self.input_usd_per_mtok + self.output_usd_per_mtok

    def as_dict(self) -> dict:
        """The ``GET /rs/v1/models`` row. Field names are wire contract."""
        return {
            "endpoint": self.endpoint,
            "kind": self.kind,
            "provider": self.provider,
            "engine": self.engine,
            "context": self.context,
            "capabilities": sorted(self.capabilities),
            "routed": self.routed,
            "healthy": self.healthy,
            "max_slots": self.max_slots,
            "in_flight": self.in_flight,
            "queued": self.queued,
            "free_slots": self.free_slots,
            # Null rather than 0.0 when unmeasured: a client that plotted 0 ms
            # would draw an infinitely fast model out of an absence of evidence,
            # which is the same mistake `latency_key` exists to prevent.
            "typical_ms": round(self.typical_ms, 1) if self.typical_ms > 0 else None,
            "price": {
                "input_usd_per_mtok": self.input_usd_per_mtok,
                "output_usd_per_mtok": self.output_usd_per_mtok,
                # 🚨 `real` is the whole of `spend.py`'s doctrine in one bool:
                # true means an invoice, false means a cost avoided. A consumer
                # that adds the two columns together has the bug that module
                # exists to prevent.
                "real": self.real_cost,
                "source": self.price_source,
                "detail": self.price_detail,
            },
        }


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Intent:
    """A caller's declared routing constraint, normalized.

    ``pin`` and ``profile`` are not alternatives to each other so much as two
    depths of the same statement — a pin says *which*, a profile says *what for*,
    and a caller may send both (pin ``tier3``, profile ``vision``) meaning "this
    endpoint, and it had better be able to see". Extra ``requires`` compose on
    top of whatever the profile asked for.
    """

    #: The pinned endpoint, NORMALIZED to its class. "" = none.
    pin: str = ""
    #: The caller's own spelling of that pin — an alias, a role, a legacy name.
    #: Echoed back rather than the class, because a caller recognises the word it
    #: sent, and telling it the canonical name it has never used reads as a
    #: substitution rather than as the same endpoint under its real name.
    pin_as_written: str = ""
    #: A named profile from the profile table. "" = none.
    profile: str = ""
    #: Capabilities required on top of the profile's own.
    requires: frozenset[str] = frozenset()
    kind: str = "chat"
    min_context: int = 0
    prefer: str = DEFAULT_PREFERENCE

    #: Endpoints the caller will NOT accept, normalized to their classes.
    #: A negative constraint, and the thing a caller working around one bad
    #: model actually wants — the alternative is pinning every endpoint they
    #: *would* take, which is a routing table in the caller.
    #:
    #: A concrete field rather than a property over ``exclude_pairs`` because
    #: ``resolve`` tests membership once per candidate; rebuilding a frozenset
    #: inside that loop would make a diagnostic field cost O(n²).
    exclude: frozenset[str] = frozenset()
    #: ``(normalized, as the caller wrote it)`` per exclusion.
    #:
    #: 🚨 The pairing is kept, not just the two sets, because a refusal has to
    #: name the caller's OWN spelling. ``normalize`` lower-cases and strips, so
    #: an unknown name does NOT come back unchanged: a caller who wrote
    #: ``"tierX"`` was told ``'tierx'`` — a word it never sent, about a mistake
    #: it is trying to find. Same reason ``pin_as_written`` exists.
    exclude_pairs: tuple[tuple[str, str], ...] = ()

    @property
    def exclude_as_written(self) -> tuple[str, ...]:
        """The caller's own spellings, in the order they sent them. Cold path —
        disclosures and refusals only."""
        return tuple(written for _, written in self.exclude_pairs)

    @property
    def declared(self) -> str:
        """What to echo back as ``attribution.requested`` — the caller's own
        words, in the order of specificity they used."""
        return self.pin_as_written or self.pin or self.profile or (
            "+".join(sorted(self.requires)) if self.requires else self.kind)


class IntentError(ValueError):
    """A caller's intent cannot be understood. Always the caller's fault, always
    deterministic, so it is a 400 and never deferrable."""


def resolve_profile(
    name: str, profiles: dict[str, Profile] | None = None,
) -> Profile:
    """Look a profile up, or raise :class:`IntentError` naming the known ones.

    Listing the alternatives in the error is not politeness. The table is a
    parameter — this deployment's, not a global — so an error that says only
    "unknown" sends the caller to read somebody else's config file to find out
    what it may ask for.
    """
    table = BUILTIN_PROFILES if profiles is None else profiles
    key = str(name or "").strip()
    if key in table:
        return table[key]
    raise IntentError(
        f"unknown intent {key!r} — known intents: {sorted(table)}")


def parse_intent(
    body: dict, *, profiles: dict[str, Profile] | None = None,
    normalize: Callable[[str], str] | None = None,
) -> Intent:
    """Build an :class:`Intent` from an enriched request body.

    Pure, and deliberately separate from the HTTP handler: the awkward parts of
    this API are all in here (what wins when a profile and explicit requirements
    disagree, what an unknown capability does), and they are worth testing
    without a socket.

    ``normalize`` maps a caller's spelling of an endpoint to its class, and is
    injected rather than imported because this module holds no catalog. 🚨 It has
    to be applied: every alias, role and legacy name resolves to a class
    (``config.normalize_endpoint``), and a pin compared literally would refuse
    ``model: "chat"`` on a fleet where ``chat`` is exactly how tier2 is spelled
    in half the callers' configs. ``Intent.declared`` keeps the caller's ORIGINAL
    spelling for the disclosure, because that is the word they will recognise.
    """
    table = BUILTIN_PROFILES if profiles is None else profiles

    raw_pin = str(body.get("model") or "").strip()
    pin = normalize(raw_pin) if (normalize and raw_pin) else raw_pin
    profile_name = str(body.get("intent") or "").strip()

    # 🚨 ``exclude`` is a declaration in its own right. "anything but tier2" is
    # a complete statement of where a request may go — it is what a caller
    # working around one bad model has to say, and requiring them to also name
    # something positive would make them enumerate the endpoints they *would*
    # take, which is the routing table we refuse to put in a caller.
    raw_exclude = body.get("exclude")
    if raw_exclude is None:
        excluded_as_written: tuple[str, ...] = ()
    else:
        if isinstance(raw_exclude, str):
            raw_exclude = [raw_exclude]
        if not isinstance(raw_exclude, (list, tuple)):
            raise IntentError(
                "`exclude` must be a list of endpoint names, e.g. "
                '["tier2", "some-alias"]')
        excluded_as_written = tuple(
            n for n in (str(x).strip() for x in raw_exclude) if n)
    # Normalized through the SAME callable as the pin. An exclusion compared
    # literally would fail to exclude anything on a fleet where the caller
    # knows the endpoint by an alias — and a negative constraint that silently
    # matches nothing routes the request to precisely the endpoint it was
    # written to avoid.
    exclude_pairs = tuple(
        ((normalize(n) if normalize else n), n) for n in excluded_as_written)
    exclude = {norm for norm, _ in exclude_pairs}

    if not pin and not profile_name and body.get("requires") is None \
            and not exclude:
        raise IntentError(
            "a request must declare an intent — send `intent` (a capability "
            f"profile: {sorted(table)}), `model` (a specific endpoint), "
            "`requires` (a list of capabilities), or `exclude` (endpoints it "
            "must not use)")

    # A pin and an exclusion of the same endpoint is a caller contradiction:
    # deterministic, entirely visible in the request, and satisfiable by
    # nothing. Refused here rather than resolved to an empty candidate set,
    # because "no endpoint satisfies model='tier2'" would send the caller
    # looking at the fleet for a fault that is in their own body.
    if pin and pin in exclude:
        raise IntentError(
            f"`model` and `exclude` name the same endpoint "
            f"({raw_pin!r} resolves to {pin!r}) — a request cannot both "
            "require and refuse it")

    kind = str(body.get("kind") or "").strip()
    prefer = str(body.get("prefer") or "").strip()
    requires: set[str] = set()

    profile = None
    if profile_name:
        profile = resolve_profile(profile_name, table)
        requires |= set(profile.requires)
        kind = kind or profile.kind
        # 🚨 An explicit `prefer` WINS over the profile's. The profile is a
        # bundle of sensible defaults, not a lock: a caller who says
        # `intent: "reasoning", prefer: "capacity"` has told us something the
        # profile author could not know — that right now they would rather wait
        # less than think harder. The reverse precedence would make `prefer`
        # settable and inert, which is the `_POLICY_PASSTHROUGH` failure again.
        prefer = prefer or profile.prefer

    raw_requires = body.get("requires")
    if raw_requires is not None:
        if isinstance(raw_requires, str):
            raw_requires = [raw_requires]
        if not isinstance(raw_requires, (list, tuple)):
            raise IntentError(
                "`requires` must be a list of capability names, e.g. "
                '["vision", "tool_calling"]')
        for cap in raw_requires:
            name = str(cap).strip()
            if name not in KNOWN_CAPABILITIES:
                raise IntentError(
                    f"unknown capability {name!r} — known capabilities: "
                    f"{sorted(KNOWN_CAPABILITIES)}")
            requires.add(name)

    kind = kind or "chat"
    if kind not in KNOWN_KINDS:
        raise IntentError(
            f"unknown kind {kind!r} — known kinds: {sorted(KNOWN_KINDS)}")

    prefer = prefer or DEFAULT_PREFERENCE
    if prefer not in PREFERENCES:
        raise IntentError(
            f"unknown preference {prefer!r} — known preferences: "
            f"{sorted(PREFERENCES)}")

    raw_ctx = body.get("min_context")
    try:
        min_context = int(raw_ctx or 0)
    except (TypeError, ValueError):
        raise IntentError(
            f"`min_context` must be an integer number of tokens, got "
            f"{raw_ctx!r}") from None
    if min_context < 0:
        raise IntentError("`min_context` must not be negative")

    return Intent(
        pin=pin,
        pin_as_written=raw_pin,
        profile=profile.name if profile is not None else "",
        requires=frozenset(requires),
        kind=kind,
        min_context=min_context,
        prefer=prefer,
        exclude=frozenset(exclude),
        exclude_pairs=exclude_pairs,
    )


# ---------------------------------------------------------------------------
# The answer
# ---------------------------------------------------------------------------

#: Why a candidate was excluded. Machine-readable, published on
#: ``POST /rs/v1/plan`` so an operator can see WHICH constraint emptied the set —
#: "no endpoint matched" is an answer nobody can act on.
REJECT_NOT_ROUTED = "not_routed"
REJECT_WRONG_KIND = "wrong_kind"
REJECT_MISSING_CAPABILITY = "missing_capability"
REJECT_CONTEXT_TOO_SMALL = "context_too_small"
#: The caller said not this one. Distinct from every other reason here in that
#: it is a property of the REQUEST rather than of the endpoint — which is why it
#: is checked first and reported per excluded endpoint: unlike a pin, an
#: exclusion names only what the caller typed, so the rows are bounded by the
#: request and are exactly the confirmation that the exclusion was understood.
REJECT_EXCLUDED = "excluded"
#: The pinned name resolves to nothing at all. Distinct from every reason above,
#: which are all "this endpoint exists and cannot serve you": this one is "you
#: named something that is not here", and it sends the caller to a different fix.
REJECT_UNKNOWN_ENDPOINT = "unknown_endpoint"


@dataclass(frozen=True)
class Rejection:
    endpoint: str
    reason: str
    detail: str = ""

    def as_dict(self) -> dict:
        return {"endpoint": self.endpoint, "reason": self.reason,
                "detail": self.detail}


@dataclass(frozen=True)
class Resolution:
    """Which endpoint should serve, plus every one that could not and why."""

    intent: Intent
    endpoint: str | None = None
    #: Ranked, best first. The chosen one is ``ranked[0]``; the rest are the
    #: alternatives, which is what makes ``/rs/v1/plan`` worth calling.
    ranked: tuple[str, ...] = ()
    rejected: tuple[Rejection, ...] = ()
    #: Exclusions that named no endpoint in this fleet. Non-empty ⟹ not ``ok``.
    #:
    #: 🚨 **This REFUSES; it does not warn.** The tempting argument is that such
    #: an exclusion is honoured trivially — the endpoint it forbids is absent,
    #: so the constraint holds — and that argument is wrong, because it assumes
    #: the reading the proxy cannot check. A name that resolves to nothing is
    #: either "not in this fleet" or "in this fleet under a spelling you got
    #: wrong", and from here those are the same bytes. Serving the second one
    #: sends the request to precisely the endpoint the caller wrote the
    #: exclusion to avoid, and reports success. That is the ``finish_reason``
    #: repair that became a silencer, and the grammar OpenRouter refuses rather
    #: than drops: never collapse "we checked" with "we could not tell".
    #:
    #: It is also what a ``pin`` already does. An unknown pin is a 404; an
    #: unknown exclusion is the same mistake, made about the same catalog, in
    #: the other direction — and a caller that must spell an endpoint correctly
    #: to demand it should not be able to misspell one to avoid it.
    unmatched_exclusions: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.endpoint is not None

    def failure_message(self) -> str:
        """A refusal a caller can act on: what was asked, and what stopped each
        candidate. Empty when the resolution succeeded."""
        if self.ok:
            return ""
        want = []
        if self.intent.pin:
            want.append(f"model={self.intent.pin!r}")
        if self.intent.profile:
            want.append(f"intent={self.intent.profile!r}")
        if self.intent.requires:
            want.append(f"requires={sorted(self.intent.requires)}")
        if self.intent.min_context:
            want.append(f"min_context={self.intent.min_context}")
        if self.intent.exclude:
            want.append(
                f"exclude={sorted(self.intent.exclude_as_written or self.intent.exclude)}")
        want.append(f"kind={self.intent.kind!r}")
        if self.unmatched_exclusions:
            # Deliberately NOT the pin's sentence, although it carries the same
            # code: a caller reading "unknown model" after excluding one would
            # look at what it asked for rather than at what it refused.
            names = ", ".join(repr(n) for n in self.unmatched_exclusions)
            return (f"`exclude` names {names} — no such endpoint, role or "
                    "alias, so the exclusion cannot be checked. Remove it, or "
                    "correct it to a name this fleet serves.")
        if any(r.reason == REJECT_UNKNOWN_ENDPOINT for r in self.rejected):
            # A pin at a name that does not exist. Deliberately worded like the
            # OpenAI door's own unknown-model error, and carrying the same
            # `unknown_endpoint` code, so a caller that already classifies one
            # does not have to learn a second spelling of the same mistake.
            return (f"unknown model {self.intent.pin_as_written or self.intent.pin!r}"
                    " — no such endpoint, role or alias")
        detail = "; ".join(f"{r.endpoint}: {r.detail or r.reason}"
                           for r in self.near_misses)
        return (f"no endpoint satisfies {', '.join(want)}"
                + (f" — {detail}" if detail else ""))

    @property
    def near_misses(self) -> tuple[Rejection, ...]:
        """The rejections worth showing a human.

        An endpoint of the wrong KIND is noise — every embedder fails every chat
        request, forever, and listing them buries the one endpoint that was the
        right shape and missed by a single capability. That one is what the
        caller is looking for.
        """
        return tuple(r for r in self.rejected if r.reason != REJECT_WRONG_KIND)


def _preference_key(prefer: str, f: ModelFacts) -> tuple:
    if prefer == "latency":
        return (f.latency_key, -f.free_slots)
    if prefer == "capacity":
        return (-f.free_slots, f.latency_key)
    if prefer == "context":
        return (-f.context, f.latency_key)
    if prefer == "cost":
        return (f.price_key, f.latency_key)
    # balanced: somewhere with room right now beats somewhere fast but full,
    # because a queue wait is added latency that the latency estimate does not
    # contain — `typical_ms` is measured from dispatch, not from arrival.
    return (0 if f.free_slots > 0 else 1, f.latency_key, -f.free_slots)


def resolve(intent: Intent, facts: list[ModelFacts] | tuple[ModelFacts, ...]) -> Resolution:
    """THE decision: which endpoint serves this intent.

    Pure. Given the same intent and the same facts it returns the same answer,
    including the same ordering of the alternatives — which is what lets
    ``/rs/v1/plan`` promise that a plan and the call that follows it agree,
    provided the fleet did not move underneath them.
    """
    pin = intent.pin.strip()
    eligible: list[ModelFacts] = []
    rejected: list[Rejection] = []

    # Which of the caller's exclusions named nothing that is here — checked
    # FIRST, because it refuses (see `Resolution.unmatched_exclusions`).
    #
    # Measured against the WHOLE fleet, not the candidate set. A pin narrows
    # `candidates`, so an exclusion naming a real endpoint the pin had already
    # removed would look unmatched here and 404 a request that is entirely
    # correct — `model: tier1, exclude: [tier2]` refused for naming tier2,
    # which is right there in the catalog.
    #
    # The normalized name is the right thing to report as well as to test. An
    # alias the fleet knows normalizes to its class and is found here; one it
    # does not know is returned unchanged by `normalize`, so what comes back is
    # the caller's own spelling, which is the word they need to see.
    present = {f.endpoint for f in facts}
    # Reported AS WRITTEN, via the pairing — see ``Intent.exclude_pairs``. The
    # normalized form is what we matched on and is not what the caller sent.
    unmatched = tuple(
        written for norm, written in intent.exclude_pairs if norm not in present)
    if unmatched:
        return Resolution(intent=intent, unmatched_exclusions=unmatched,
                          rejected=tuple(
                              Rejection(n, REJECT_UNKNOWN_ENDPOINT,
                                        "no such endpoint to exclude")
                              for n in unmatched))

    # 🚨 A pin NARROWS the candidate set; it does not reject the others. Rejecting
    # them produced one "not the pinned endpoint" row per endpoint in the fleet —
    # a diagnostic that grows with the catalog and says nothing, burying the row
    # that actually explains the refusal. What a caller needs to be told about a
    # pin is either "that name is not here" or "it is here and cannot do this".
    candidates = [f for f in facts if f.endpoint == pin] if pin else list(facts)
    if pin and not candidates:
        return Resolution(intent=intent, rejected=(
            Rejection(intent.pin_as_written or pin, REJECT_UNKNOWN_ENDPOINT,
                      "no such endpoint"),))

    for f in candidates:
        if f.endpoint in intent.exclude:
            # First, ahead of every property of the endpoint itself. If a
            # caller excluded an endpoint that is also unrouted and also the
            # wrong kind, the answer they need is the one they can act on —
            # "because you said so" — not a fact about our fleet that would
            # read as though the exclusion had not been understood.
            rejected.append(Rejection(f.endpoint, REJECT_EXCLUDED,
                                      "excluded by the request"))
            continue
        if not f.routed:
            # `planned` and `retired` entries exist to document a name and a
            # shape. Resolving to one would dispatch at a backend nobody has a
            # credential for — which is precisely what the example catalog's two
            # remote endpoints are, so this branch is exercised on every boot.
            rejected.append(Rejection(f.endpoint, REJECT_NOT_ROUTED,
                                      "endpoint is not routed"))
            continue
        if f.kind != intent.kind:
            rejected.append(Rejection(
                f.endpoint, REJECT_WRONG_KIND,
                f"kind is {f.kind!r}, not {intent.kind!r}"))
            continue
        missing = sorted(intent.requires - f.capabilities)
        if missing:
            rejected.append(Rejection(
                f.endpoint, REJECT_MISSING_CAPABILITY,
                f"does not declare {', '.join(missing)}"))
            continue
        if intent.min_context and f.context and f.context < intent.min_context:
            # A context of 0 means "not known" and admits, the same convention
            # as `scheduler._fits_context`: a ceiling we have not discovered is
            # not a ceiling of zero.
            rejected.append(Rejection(
                f.endpoint, REJECT_CONTEXT_TOO_SMALL,
                f"context {f.context} < required {intent.min_context}"))
            continue
        eligible.append(f)

    if not eligible:
        return Resolution(intent=intent, rejected=tuple(rejected))

    ranked = sorted(
        eligible,
        key=lambda f: (
            # 🚨 Money first, ALWAYS, under every preference. See the module
            # docstring: a preference that could pull traffic off the machine
            # would make remote capacity the ordinary path instead of overflow.
            1 if f.real_cost else 0,
            0 if f.healthy else 1,
            *_preference_key(intent.prefer, f),
            # Stable across boots so `data[0]` is a defensible default and two
            # identical requests do not oscillate between equal candidates.
            f.endpoint,
        ),
    )
    return Resolution(
        intent=intent,
        endpoint=ranked[0].endpoint,
        ranked=tuple(f.endpoint for f in ranked),
        rejected=tuple(rejected),
    )
