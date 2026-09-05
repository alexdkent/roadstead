"""The enriched Roadstead API — the north face that is not OpenAI's.

``/rs/v1/*``. Three routes, and between them they answer the three questions a
caller of a *capacity-aware* gateway has and cannot ask an OpenAI-shaped API:

===========================  ==================================================
``GET  /rs/v1/models``       what can serve me, what can it do, what is it like
                             right now, and what does it cost
``POST /rs/v1/plan``         where would this go, how long should I allow, and
                             what would it cost — **without dispatching**
``POST /rs/v1/chat``         do it, and tell me what actually happened
===========================  ==================================================

---

## Why a second API and not more fields on the first one

🚨 **Enrichment never appears inside an OpenAI-shaped body.** A client that
validates against OpenAI's schema must not break because it pointed at
Roadstead, and "we only *added* fields" is not a defence — strict validators
reject unknown keys, and the ones that do not will happily hand an extra key to
a caller that then depends on it from a server that is not us. So the OpenAI
door stays byte-identical and its enrichment rides in ``X-Roadstead-*`` response
headers (see ``ENRICHMENT_HEADERS``), which are ignorable by construction.

Everything that does not fit in a header lives here.

## The four things it carries that ``/v1/submit`` could not

**Enriched model information.** Not a config readout: what an endpoint *is*
(context, capabilities, provider, engine) and what it is *currently like* (queue
depth, free slots, health, learned median latency), plus its price and — the
part ``spend.py`` exists for — whether that price is an invoice or a cost
avoided.

**Timing, on both sides of the call.** Before: the recommended deadline for
*this* call, from the learned distribution, so a caller stops guessing 180s.
After: queue wait, TTFT, decode, total, against what was predicted.

**Prioritisation as a declaration.** ``priority`` and ``interactive`` are what
the caller states; the number is ours. That was already true and was buried in
a body field nobody documented as intent.

**Attribution.** Which endpoint served, on which provider, at what cost — and
whether that differed from what was asked for. 🚨 A caller that did not opt into
substitution is never silently given something else, and a caller that did is
always told.

---

## The one thing deliberately NOT in the response

🚨 **The effective priority, and therefore the spend demotion.** ``docs/api.md``
§1.6: *"A caller cannot observe its own demotion in a response."* A demoted
caller is still served, still from local capacity, and its own band is not
information it can act on — while publishing it would turn a threshold that
"never rejects" into one every client could detect and branch on, which is a
rejection with extra steps. It is an operator fact and it goes to the operator,
through the degradation seam and ``/v1/status``. So this module publishes no
band, no priority and no queue position, and ``test_enriched_api.py`` fails if
one appears.

---

## Where the work happens

Almost nowhere here. ``handle_rs_chat`` resolves an intent, translates the
enriched envelope into the internal submit dict, and hands it to
``Lifecycle.handle_submit`` — the *same* admission, DRR, grammar, cache,
correction and telemetry path the OpenAI doors use. A second hot path would be a
second set of bugs, and the whole argument for one admission decision with three
outcomes (``scheduler.Admission``) applies just as hard one layer up.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .config import LLMPriority, ROLE_TO_CLASS, normalize_endpoint
from .cost_model import estimate_input_tokens
from .intent import (
    IntentError,
    ModelFacts,
    parse_intent,
    resolve,
)

if TYPE_CHECKING:
    from .health import Health
    from .lifecycle import Lifecycle
    from .state import ProxyState

logger = logging.getLogger(__name__)

#: The route prefix. Its OWN version, separate from ``/v1/*``: the OpenAI-compat
#: surface is versioned by OpenAI and this one is versioned by us, and pinning
#: them together would mean either following someone else's version number or
#: publishing a ``/v2/chat/completions`` that is not OpenAI's v2.
PREFIX = "/rs/v1"

#: Which response shape ``Lifecycle.handle_submit`` serializes into. Two values,
#: because there are two north faces; the choice varies nothing but the bytes.
WIRE_OPENAI = "openai"
WIRE_ENRICHED = "enriched"

#: Enrichment on the OPENAI door, which may carry no extra body fields.
#:
#: 🚨 Only what is known *before* the body starts, because a streaming response's
#: headers are on the wire before the first token — so the served endpoint is the
#: one admission chose, and a later failover or spill is NOT reflected here. That
#: is a real limit of the header channel and it is why the enriched API exists:
#: a caller that needs to know what actually served must ask for it on ``/rs/v1``.
#: Advertising a substitution here that a stream might contradict would be worse
#: than advertising nothing.
ENRICHMENT_HEADERS = {
    "request_id": "X-Roadstead-Request-Id",
    "endpoint": "X-Roadstead-Endpoint",
    "deadline_s": "X-Roadstead-Deadline-S",
    "deadline_source": "X-Roadstead-Deadline-Source",
    # Audit P2, 2026-09-04. Comma-separated correction tokens (see
    # ``Correction.corrections_applied``) — omitted entirely when empty, same
    # rule as every other absent enrichment header. On a STREAMING response
    # this can only ever carry ``json_object_stripped`` (known at admission);
    # a sync response may add whatever `corrections_applied` found once the
    # backend has actually answered.
    "corrected": "X-Roadstead-Corrected",
}


def _error(code: str, message: str, status: int, **extra) -> JSONResponse:
    """The enriched error envelope.

    🚨 ``code`` and ``error`` stay at the TOP level and keep the exact spellings
    ``docs/api.md`` §2.1 publishes. The enriched API is a new shape, not a new
    taxonomy: a caller's deferrable-vs-not classification must give the same
    answer on both doors, and §2.2's marker substrings live in ``error``.
    """
    return JSONResponse(
        {"status": "error", "code": code, "error": message, **extra},
        status_code=status)


class EnrichedApi:
    """``/rs/v1`` over the shared ProxyState. A translator, not a second path."""

    #: How long a memoised ``typical_ms`` stays good. Short enough that a
    #: reader never sees a stale fleet, long enough that a burst of requests
    #: pays for one pass rather than N.
    _TYPICAL_MS_TTL_S = 1.0

    def __init__(self, state: "ProxyState", lifecycle: "Lifecycle",
                 health: "Health") -> None:
        self.state = state
        self.lifecycle = lifecycle
        self.health = health
        #: endpoint -> learned median ms, and when the map was built.
        #: 🚨 Memoised because ``facts()`` runs on EVERY `/rs/v1/chat` request to
        #: resolve one intent, and it called ``timeout_model.advise`` once per
        #: endpoint to fill this one field — N ladder walks to answer a question
        #: about the whole fleet, on the hot path, for a number that is a median
        #: over thousands of samples and cannot meaningfully move between two
        #: requests a millisecond apart.
        #:
        #: 🚨 NOT dropped, and not made optional. `prefer=latency` and
        #: `prefer=balanced` rank on it, and an endpoint with no samples sorts
        #: as SLOW — so a `facts()` that omitted it would silently re-rank every
        #: intent-routed request rather than merely losing a display field.
        self._typical_ms: dict[str, float] = {}
        self._typical_ms_at: float = 0.0

    def _typical_ms_for(self, names: "list[str]") -> dict[str, float]:
        """The learned median per endpoint, rebuilt at most once per TTL.

        One pass over the fleet, on the loop, memoised — rather than one ladder
        walk per endpoint per request. A rebuild that raises leaves the previous
        map in place and falls back to 0.0 for anything missing, because a model
        readout must never 500 a listing.
        """
        now = time.monotonic()
        if (self._typical_ms
                and now - self._typical_ms_at < self._TYPICAL_MS_TTL_S
                and all(n in self._typical_ms for n in names)):
            return self._typical_ms
        fresh: dict[str, float] = {}
        for name in names:
            # The learned median for the endpoint as a whole — the widest cell
            # the timeout model has, because a per-(tier, size) figure would be
            # answering a question about a request we have not been given yet.
            try:
                advice = self.state.timeout_model.advise(
                    name, int(LLMPriority.P1_TURN_SUPPORT), 0, 0)
                fresh[name] = (float(advice.get("median_ms") or 0.0)
                               if advice.get("sample_count") else 0.0)
            except Exception:  # noqa: BLE001 — a readout must not 500 a listing
                fresh[name] = 0.0
        self._typical_ms = fresh
        self._typical_ms_at = now
        return fresh

    # ---------------------------------------------------------------- facts

    def _aliases_for(self, endpoint: str) -> list[str]:
        """Every name that resolves to ``endpoint``, so a caller can pin using
        whichever one its config already holds instead of learning ours."""
        return sorted({name for name, cls in ROLE_TO_CLASS.items()
                       if cls == endpoint and name != endpoint})

    def profiles(self) -> dict:
        """The intent vocabulary in force — built-ins with this deployment's
        ``intents:`` layered over them.

        Read from the catalog on every call rather than cached on the instance,
        for the same reason ``facts()`` is: ``load_catalog`` already memoises
        per resolved path, so this costs a dict lookup, and a table pinned at
        construction would survive a deliberate reload and quietly serve the
        vocabulary the process booted with.
        """
        from .model_catalog import load_catalog

        return load_catalog().intents

    def facts(self) -> list[ModelFacts]:
        """A snapshot of every endpoint in the catalog, routed or not.

        Assembled here rather than in ``intent.py`` because every line of it is a
        lookup, and the resolver's whole value is that it does none. Cheap: all
        in-memory reads (config, the scheduler's occupancy dicts, the health
        map, the timeout model's aggregate, the price book), so it is safe on the
        request path and safe on the loop.
        """
        from .model_catalog import load_catalog

        cat = load_catalog()
        # 🚨 The UNION, not the catalog alone. `config.endpoints` is normally
        # derived from the catalog, but it is a plain dict a deployment (or a
        # test) may hold entries in that the catalog never named — and an
        # endpoint the scheduler will happily dispatch to must be reachable by
        # intent, or it becomes capacity only a pin can find. The catalog's
        # order is kept first so the common case reads the same.
        names = list(cat.endpoints) + [
            n for n in self.state.config.endpoints if n not in cat.endpoints]
        typical = self._typical_ms_for(names)
        out: list[ModelFacts] = []
        for name in names:
            entry = cat.endpoints.get(name)
            ep_cfg = self.state.config.endpoints.get(name)
            provider = cat.provider(entry.provider) if entry else None
            snap = (self.state.scheduler.endpoint_snapshot(name)
                    if ep_cfg is not None else
                    {"max_slots": 0, "in_flight": 0, "queued": 0})
            price = self.state.prices.price(name)
            typical_ms = typical.get(name, 0.0)
            # An endpoint present in the routing table is ROUTED whatever the
            # catalog says — the scheduler will dispatch to it, and reporting
            # otherwise would describe a fleet Roadstead is not running.
            #
            # 🚨 That is what this line MEANT since it was written, and not what
            # it did: it consulted `entry.routed` whenever a catalog entry
            # existed, i.e. almost always. The two could not disagree, because
            # `config.endpoints` is built from `cat.routed()` at startup — so
            # the wrong branch was unreachable and the comment above went
            # unchallenged. Roadmap J1 makes them disagree on purpose: an
            # endpoint promoted at runtime is in the routing table while the
            # catalog still says `planned`. Consulting the catalog then reported
            # a promoted endpoint as unrouted on `/rs/v1/models`, and
            # `intent.py` filters on exactly this field — so a pin to it 404'd
            # while the management plane said it was live.
            #
            # `config.endpoints` IS the routing table. There is nothing else to
            # ask.
            routed = ep_cfg is not None
            out.append(ModelFacts(
                endpoint=name,
                kind=(ep_cfg.kind if ep_cfg else entry.kind if entry else "chat"),
                provider=(entry.provider if entry else ""),
                engine=(provider.engine if provider
                        else ep_cfg.backend_engine if ep_cfg else ""),
                context=(ep_cfg.context_per_slot if ep_cfg
                         else entry.context_per_slot if entry else 0),
                capabilities=(
                    ep_cfg.capabilities if ep_cfg
                    else frozenset(k for k, v in entry.capabilities.items() if v)
                    if entry else frozenset()),
                routed=routed,
                # An unrouted endpoint has no health to report and no poller
                # looking at it; calling it healthy would put a green light on a
                # name nobody can dispatch to.
                healthy=(self.health.endpoint_healthy(name)
                         if routed and ep_cfg is not None else False),
                max_slots=int(snap.get("max_slots") or 0),
                in_flight=int(snap.get("in_flight") or 0),
                queued=int(snap.get("queued") or 0),
                typical_ms=typical_ms,
                input_usd_per_mtok=price.input_usd_per_mtok,
                output_usd_per_mtok=price.output_usd_per_mtok,
                real_cost=price.real,
                price_source=price.source,
                price_detail=price.detail,
            ))
        out.sort(key=lambda f: f.endpoint)
        return out

    # ---------------------------------------------- GET /rs/v1/models

    async def handle_rs_models(self, request: Request) -> Response:
        """The enriched catalogue: what exists, what it can do, what it is like.

        Gated like every other door. It is a map of the fleet — slot counts,
        occupancy, health and prices — and an unenrolled caller has no more
        business reading that than it has dispatching to it.
        """
        resolved = self.state.identity.resolve(request)
        if not resolved.ok:
            d = resolved.denial
            return _error(d.code, d.message, d.status)
        rows = [f.as_dict() for f in self.facts()]
        for row in rows:
            row["aliases"] = self._aliases_for(row["endpoint"])
        return JSONResponse({
            "object": "roadstead.models",
            "models": rows,
            # The intent vocabulary, published rather than documented — an
            # "unknown intent" error is a poor place to learn one. Asking is
            # now the only correct way to get it: the table is the built-ins
            # with this deployment's `intents:` layered over them, so a caller
            # that hard-coded our list is reading a vocabulary that may not be
            # the one in force here.
            #
            # 🚨 `source` is the disclosure that makes an override visible. A
            # fleet may redefine `reasoning` to mean its own thing, and a
            # caller reading our documentation for that word would otherwise
            # have no way to tell that it no longer applies.
            "intents": [
                {"name": p.name, "kind": p.kind,
                 "requires": sorted(p.requires), "prefer": p.prefer,
                 "summary": p.summary, "source": p.source}
                for p in sorted(self.profiles().values(), key=lambda p: p.name)
            ],
        })

    # ------------------------------------------------ POST /rs/v1/plan

    async def handle_rs_plan(self, body: dict, request: Request) -> Response:
        """Resolve an intent and price the call — without making it.

        🚨 The point of this route is that it runs the SAME
        ``intent.resolve`` the dispatching route runs, over the same facts. A
        planner that approximated the router would be a second answer to the
        question the router is about to answer differently, and a caller would
        have no way to tell which one lied.

        Same discipline for the PRIORITY BAND the timing is computed at:
        ``Lifecycle.resolve_declared_priority`` is the one place that decision
        is made, shared with ``handle_submit`` — a plan resolved at a
        different band than the call it precedes would predict against the
        wrong ``timeout_model`` cell and the wrong interactive/long-form
        ceiling.
        """
        resolved_id = self.state.identity.resolve(request)
        if not resolved_id.ok:
            d = resolved_id.denial
            return _error(d.code, d.message, d.status)
        # Delegated here too, and not as a courtesy: §1.7 promises a plan and
        # the call that follows it agree, and `_substitution_policy` reads the
        # caller's degrade/spill opt-in off the agent config. A plan resolved as
        # the key's own identity would answer for a different caller than the
        # one about to dispatch.
        delegated = self.state.identity.delegate(
            resolved_id.principal, body.get("agent_id"))
        if not delegated.ok:
            d = delegated.denial
            return _error(d.code, d.message, d.status)
        agent_id = delegated.principal.agent_id

        try:
            intent = parse_intent(body, normalize=normalize_endpoint,
                                  profiles=self.profiles())
        except IntentError as exc:
            return _error("invalid_request_error", str(exc), 400)

        facts = self.facts()
        res = resolve(intent, facts)
        if not res.ok:
            return _error("unknown_endpoint", res.failure_message(), 404,
                          considered=[r.as_dict() for r in res.near_misses])

        payload = body.get("payload") if isinstance(body.get("payload"), dict) else {}
        est_in = int(body.get("est_in") or 0) or estimate_input_tokens(payload)
        mt = payload.get("max_tokens")
        est_out = int(body.get("est_out") or 0) or (
            mt if isinstance(mt, int) and mt > 0 else 0)
        priority = self.lifecycle.resolve_declared_priority(
            body, agent_id, delegated.principal)

        advice = self.state.effective_timeout_advice(
            res.endpoint, int(priority), est_in, est_out)
        by_ep = {f.endpoint: f for f in facts}
        chosen = by_ep[res.endpoint]
        price = self.state.prices.price(res.endpoint)
        return JSONResponse({
            "object": "roadstead.plan",
            "requested": intent.declared,
            "endpoint": res.endpoint,
            # The runners-up, in the order the router would fall through them.
            # Publishing them is what lets a caller understand a decision rather
            # than only receive it.
            "alternatives": list(res.ranked[1:]),
            "considered": [r.as_dict() for r in res.near_misses],
            "model": chosen.as_dict(),
            "timing": {
                "recommended_deadline_s": advice.get("recommended_timeout_s"),
                "predicted_ms": advice.get("recommended_ms"),
                "median_ms": advice.get("median_ms"),
                "p95_ms": advice.get("p95_ms"),
                "sample_count": advice.get("sample_count"),
                # `floor` means the deadline came from the class floor, not from
                # evidence — see docs/api.md §1.4. Worth surfacing: a caller that
                # sees it knows its number is a guarantee, not a measurement.
                "source": advice.get("source"),
            },
            "cost": {
                "estimated_usd": round(price.cost_usd(est_in, est_out), 6),
                **price.as_dict(),
            },
            # Whether this caller may currently be substituted, so a plan can say
            # "and if it is busy, here is what happens". Read off the same config
            # the two gates read.
            "substitution": self._substitution_policy(agent_id, res.endpoint),
        })

    def _substitution_policy(self, agent_id: str, endpoint: str) -> dict:
        cfg = self.state.config.agent_config(agent_id)
        ep_cfg = self.state.config.endpoints.get(endpoint)
        # 🚨 Two questions, two answers, and they never default from each other:
        # `degrade` is "this backend is DOWN, may a smaller model answer" and
        # `spill` is "this backend is BUSY, may we pay somebody else". Reporting
        # them as one field is the collapse docs/internals.md warns recurs.
        return {
            "degrade": {
                "allowed": bool(cfg.degrade_ok),
                "target": (ep_cfg.failover_to if ep_cfg else "") or None,
            },
            "spill": {
                "allowed": bool(cfg.spill_ok
                                and self.state.spend_may_spill(agent_id)),
                "target": (ep_cfg.spill_to if ep_cfg else "") or None,
            },
        }

    # ------------------------------------------------ POST /rs/v1/chat

    async def handle_rs_chat(self, body: dict, request: Request) -> Response:
        """The enriched call. Resolves intent, then joins the one hot path."""
        resolved_id = self.state.identity.resolve(request)
        if not resolved_id.ok:
            d = resolved_id.denial
            return _error(d.code, d.message, d.status)
        # 🚨 A body-declared `agent_id` is applied ONLY where the credential's
        # own allowlist permits it, and `identity.delegate` is the only thing
        # that knows. This door read no identity from the body at all until
        # 2026-09-02, and the reason it did not is unchanged — a caller must not
        # be able to claim a better DRR weight. What changed is that a key can
        # now GRANT the names it may wear, which is the operator making the
        # claim rather than the caller.
        delegated = self.state.identity.delegate(
            resolved_id.principal, body.get("agent_id"))
        if not delegated.ok:
            d = delegated.denial
            return _error(d.code, d.message, d.status)
        principal = delegated.principal

        payload = body.get("payload")
        if not isinstance(payload, dict):
            return _error(
                "invalid_request_error",
                "`payload` must be an object carrying the model request "
                "(messages, max_tokens, stream, …)", 400)

        try:
            intent = parse_intent(body, normalize=normalize_endpoint,
                                  profiles=self.profiles())
        except IntentError as exc:
            return _error("invalid_request_error", str(exc), 400)

        res = resolve(intent, self.facts())
        if not res.ok:
            # 🚨 A refusal, never a fallback. A pin that cannot be satisfied and
            # an intent nothing matches are both *deterministic* caller errors:
            # serving them from whatever happened to be nearest is the silent
            # substitution this API exists to make impossible.
            return _error("unknown_endpoint", res.failure_message(), 404,
                          considered=[r.as_dict() for r in res.near_misses])

        try:
            allow_degrade, allow_spill = _substitution_request(body)
        except IntentError as exc:
            return _error("invalid_request_error", str(exc), 400)

        # The payload's own `model` is overwritten with the resolved class so
        # ``Lifecycle.resolve_endpoint`` — which reconciles a submit endpoint
        # against a payload model — agrees with the decision already made here
        # rather than silently re-routing away from it. The backend never sees
        # this value: ``backend.py`` substitutes ``effective_model_id``.
        payload = {**payload, "model": res.endpoint}

        submit: dict = {
            "agent_id": principal.agent_id,
            # 🚨 The caller's own word, carried separately BECAUSE the line
            # above has already overwritten it with the resolved identity. The
            # two differing is the whole thing being disclosed, so the raw one
            # cannot be reconstructed downstream — it has to travel.
            "declared_agent_id": str(body.get("agent_id") or ""),
            "endpoint": res.endpoint,
            "requested": intent.declared,
            # Only what the CALLER declared. `None` when it declared nothing,
            # in which case the key is omitted below and `handle_submit`'s one
            # precedence resolves the band — credential, then the agent's own
            # configured default. Pre-filling the identity's band here is what
            # made `agents.yaml`'s `default_priority` dead.
            "priority": _priority_for(body),
            "call_site": str(body.get("call_site")
                             or f"{principal.agent_id}.rs"),
            "caller_id": body.get("caller_id") or principal.agent_id,
            "payload_type": str(body.get("payload_type") or "chat_completion"),
            "payload": payload,
            "session_id": body.get("session_id"),
            "turn_id": body.get("turn_id"),
            "request_id": body.get("request_id"),
            "allow_degrade": allow_degrade,
            "allow_spill": allow_spill,
        }
        # `deadline_s` is the enriched spelling of `timeout_s`. Omitted entirely
        # when the caller expressed no opinion, so the computed default applies —
        # passing a null would look like a supplied deadline of nothing.
        if submit["priority"] is None:
            del submit["priority"]
        deadline = body.get("deadline_s", request.headers.get("X-Timeout-S"))
        if deadline is not None:
            submit["timeout_s"] = deadline

        return await self.lifecycle.handle_submit(submit, request, wire=WIRE_ENRICHED)


def _priority_for(body: dict) -> LLMPriority | None:
    """The band THIS REQUEST declared, or ``None`` if it declared none.

    ``interactive`` is the enriched API's first-class spelling of the thing
    ``priority`` has always encoded and never named: whether somebody is waiting.
    It is a coarse control on purpose — a caller that wants the precise band
    still sends ``priority``, which wins, because it says strictly more.

    🚨 Returns ``None`` rather than the identity's band. It used to fall back to
    ``principal.priority``, which reached ``handle_submit`` indistinguishable
    from a band the caller had asked for — and so shadowed the agent's own
    configured ``default_priority`` completely. Whose default applies is one
    question with one answer, and it is answered in ``handle_submit``.
    """
    if body.get("priority") is not None:
        return LLMPriority.coerce(body["priority"],
                                  default=LLMPriority.P1_TURN_SUPPORT)
    interactive = body.get("interactive")
    if interactive is True:
        return LLMPriority.P1_TURN_SUPPORT
    if interactive is False:
        return LLMPriority.P3_INGESTION
    return None


def _substitution_request(body: dict) -> tuple[bool | None, bool | None]:
    """Parse the per-request substitution narrowing.

    🚨 Returns ``None`` for "not declared" and only ever narrows downstream. A
    ``true`` here is accepted and recorded but grants nothing on its own — the
    gates take the AND with the operator's opt-in. Accepting it rather than
    rejecting it is deliberate: a caller that says "yes, spilling is fine by me"
    is making a true statement about itself, and refusing the request over a
    permission it does not control would be an error nobody can fix.
    """
    raw = body.get("substitution")
    if raw is None:
        return None, None
    if not isinstance(raw, dict):
        raise IntentError(
            '`substitution` must be an object, e.g. {"degrade": false, '
            '"spill": false}')
    out: list[bool | None] = []
    for key in ("degrade", "spill"):
        val = raw.get(key)
        if val is None:
            out.append(None)
        elif isinstance(val, bool):
            out.append(val)
        else:
            raise IntentError(
                f"`substitution.{key}` must be true or false, got {val!r}")
    return out[0], out[1]


# ---------------------------------------------------------------------------
# Response assembly — shared by the enriched door and the OpenAI door's headers
# ---------------------------------------------------------------------------

#: What moved a request off the endpoint that was chosen for it. Two values,
#: never one: see ``docs/api.md`` §1.6 and ``docs/internals.md`` — failover answers "this
#: backend is DOWN" and spill answers "this backend is FULL", and a caller reacts
#: to them differently (one got a worse answer, one got an invoice).
SUBSTITUTION_FAILOVER = "failover"
SUBSTITUTION_SPILL = "spill"


def attribution(state: "ProxyState", req) -> dict:
    """Who served this request, and whether that is who was asked for."""
    ep_cfg = state.config.endpoints.get(req.endpoint)
    price = state.prices.price(req.endpoint)
    substitution = None
    if req.spilled_from:
        substitution = SUBSTITUTION_SPILL
    elif req.degraded_from:
        substitution = SUBSTITUTION_FAILOVER
    return {
        "requested": req.requested,
        "resolved": req.routed_to,
        "endpoint": req.endpoint,
        # Derived from the two recorded fields rather than from `substitution`
        # being non-null, so the bool cannot drift from the label.
        "substituted": bool(req.routed_to and req.routed_to != req.endpoint),
        "substitution": substitution,
        "provider": _provider_name(req.endpoint),
        "engine": (ep_cfg.backend_engine if ep_cfg else ""),
        # The model the BACKEND is serving, as discovery found it — not the
        # endpoint class, and not what the caller asked for. Reading it back is
        # the fleet's own rule for trusting a rename.
        "model": (ep_cfg.effective_model_id if ep_cfg else ""),
    }


def _provider_name(endpoint: str) -> str:
    from .model_catalog import load_catalog
    entry = load_catalog().entry(endpoint)
    return entry.provider if entry else ""


def identity_block(req) -> dict:
    """Who this call was BILLED to, and whether that is who the caller asked for.

    🚨 The point of it is the second half. A credential with no delegation grant
    IGNORES a body-declared ``agent_id`` (``docs/api.md`` §1.5 rule 3) — right,
    because refusing would break every caller carrying a ``/v1/submit`` habit
    over a claim that was already inert, and a **silencer** if it were also
    invisible: the caller asks to be ``chat-agent``, the work is billed to the
    credential, and both cases return 200 with identical bodies. The same shape
    as the ``finish_reason`` repair that had to learn to keep quiet only when it
    could tell.

    So the resolved identity is ALWAYS reported (it is a fact, and cheap), and
    ``honoured`` appears only when the caller declared something — exactly as
    ``substituted`` is reported only for a move that actually happened, rather
    than firing on every intent-routed call and meaning nothing.

    🚨 Nothing here leaks a band, a queue position or a demotion (§1.6). The
    ``agent_id`` is either the caller's own word or the name on the credential
    it presented; neither is news to the caller.
    """
    block: dict = {"agent_id": req.agent_id}
    declared = getattr(req, "declared_agent_id", "") or ""
    if declared:
        block["declared"] = declared
        block["honoured"] = declared == req.agent_id
    return block


def cost_block(state: "ProxyState", endpoint: str,
               input_tokens: int, output_tokens: int) -> dict:
    """What this call cost, in the two kinds of money that are never summed.

    🚨 ``spent_usd`` is an invoice and ``avoided_usd`` is a saving. They are both
    USD and nothing in the type system separates them, which is why
    ``TokenPrice.real`` does — and why they are reported in two fields whose
    values a consumer must never add. Exactly one of them is ever non-zero for a
    given call, and which one is a property of the price, not of the endpoint.
    """
    price = state.prices.price(endpoint)
    amount = round(price.cost_usd(input_tokens, output_tokens), 6)
    return {
        "spent_usd": amount if price.real else 0.0,
        "avoided_usd": 0.0 if price.real else amount,
        "price": price.as_dict(),
    }


def timing_block(req, *, queue_wait_ms: float, backend_latency_ms: float,
                 ttft_ms: float | None = None,
                 predicted_ms: float | None = None) -> dict:
    """When things happened, against what was predicted."""
    total = float(queue_wait_ms) + float(backend_latency_ms)
    return {
        "queue_wait_ms": round(float(queue_wait_ms), 1),
        "backend_latency_ms": round(float(backend_latency_ms), 1),
        # Null rather than 0.0 for a non-streaming call: there is no first token
        # to time, and a zero would read as an instantaneous one.
        "ttft_ms": (round(float(ttft_ms), 1)
                    if ttft_ms not in (None, 0) else None),
        "total_ms": round(total, 1),
        "deadline_s": round(float(req.timeout_s), 3),
        # 🚨 Which SIDE chose the deadline, which is a contract boundary rather
        # than a detail: a caller's own deadline is a hard wall we keep to the
        # letter, one we computed is a budget the streaming path may extend
        # while tokens are demonstrably still arriving.
        "deadline_source": "computed" if req.deadline_is_default else "caller",
        "predicted_ms": (round(float(predicted_ms), 1)
                         if predicted_ms else None),
    }


def enrichment_headers(req, corrections: list[str] | None = None) -> dict[str, str]:
    """The ``X-Roadstead-*`` headers for the OpenAI door. See ENRICHMENT_HEADERS.

    ``corrections`` is the caller's own list (``Correction.corrections_applied``)
    — passed in rather than recomputed here so a sync caller (who has a
    completed ``result``) and a streaming caller (admission time only, no
    ``result`` yet) each supply exactly what they know. Omitted from the
    headers entirely when empty, the same rule every other enrichment header
    already follows.
    """
    headers = {
        ENRICHMENT_HEADERS["request_id"]: req.request_id,
        ENRICHMENT_HEADERS["endpoint"]: req.endpoint,
        ENRICHMENT_HEADERS["deadline_s"]: f"{req.timeout_s:.3f}",
        ENRICHMENT_HEADERS["deadline_source"]: (
            "computed" if req.deadline_is_default else "caller"),
    }
    if corrections:
        headers[ENRICHMENT_HEADERS["corrected"]] = ",".join(corrections)
    return headers
