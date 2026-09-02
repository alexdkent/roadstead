"""The catalog — what backends exist, how to reach them, and what they are for.

``models.yaml`` (next to this file) has two sections, and the split is the whole
design:

**`providers:`** — *how to reach a backend and how to speak to it.* One entry per
backend process or remote service: its engine (which adapter in
``roadstead/providers/`` drives it), its address, and its credential if it needs
one. A local llama.cpp server IS a provider — the address belongs here for the
same reason OpenRouter's ``base_url`` does, and mixing the two placements was the
first thing to go when this file was redesigned.

**`endpoints:`** — *a routable unit of capacity, with policy.* Slots, per-slot
context, timeout floors and ceilings, capabilities, band behaviour, failover.
Each names the provider it lives behind.

That gives the asymmetry a shape: a local provider hosts **one** endpoint,
because a llama.cpp or vLLM server serves one model; a remote provider hosts
**many**, because it fronts a catalogue and the credential and base URL are the
same for all of them. Declaring the connection once is why the sections are
separate rather than one flat list.

**The data here is an EXAMPLE.** The schema is Roadstead's contract; the fleet it
describes is invented, and its addresses are RFC 5737 documentation addresses.
Point ``ROADSTEAD_MODELS_YAML`` at your own file.

Pure: only ``yaml``, stdlib, ``roadstead.providers`` and ``roadstead.intent``
(neither of which imports anything back). ``config`` imports the ``build_*``
helpers from here, never the reverse.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import hooks
from .intent import (
    BUILTIN_PROFILES,
    DEFAULT_PREFERENCE,
    KNOWN_CAPABILITIES,
    KNOWN_KINDS,
    PREFERENCES,
    Profile,
)
from .providers import DEFAULT_PROVIDER, provider_for_engine

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parent / "models.yaml"


@dataclass(frozen=True)
class ProviderEntry:
    """One backend process, or one remote service. Connection + engine, nothing
    about policy — that is the endpoint's business."""

    name: str
    #: Which adapter drives it: ``llama.cpp`` | ``vllm`` | ``openrouter`` |
    #: ``shim``. Resolved through ``roadstead.providers``; an unknown value
    #: falls back to llama.cpp rather than failing (see that package).
    engine: str = ""
    #: Address, local form. ``host`` may be a name in the top-level ``hosts:``
    #: table or a literal address.
    host: str = ""
    port: int = 0
    #: Address, general form. Wins over host/port, and is the only form that can
    #: express a scheme or a base path.
    base_url: str = ""
    #: NAME of the environment variable holding the API key — never the key.
    api_key_env: str = ""
    notes: str = ""


@dataclass(frozen=True)
class EndpointEntry:
    """One routable unit of capacity. The YAML key is its endpoint class, which
    is what callers address and what the scheduler accounts against."""

    #: The YAML key: the endpoint class.
    name: str
    #: Which provider serves it.
    provider: str = ""
    #: ``chat`` | ``embed`` | ``rerank``. Read by the cache-stats screen, which
    #: only makes sense for chat.
    kind: str = "chat"
    #: ``active`` | ``on_demand`` | ``planned`` | ``retired``. Only active and
    #: on_demand are routed; the others document a name without serving it.
    status: str = "active"
    #: Telemetry/model role string. Defaults to the endpoint class.
    role: str = ""
    #: Every legacy or shorthand name that must resolve here.
    aliases: tuple[str, ...] = ()
    #: For a provider fronting a CATALOGUE, the model slug to route on. Ignored
    #: for a provider that serves one model and will name it — that is
    #: discovered, and a config value would go stale silently.
    model: str = ""
    context_per_slot: int = 0
    slots: int = 0
    timeout_floor_s: float = 0.0
    timeout_ceiling_s: float = 0.0
    stream_hard_cap_s: float = 0.0
    #: The endpoint class to degrade to when this one is unhealthy.
    failover_to: str = ""
    #: The endpoint class to spill to when this one is FULL. A different
    #: question from ``failover_to`` — see ``EndpointConfig.spill_to``.
    spill_to: str = ""
    capabilities: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def all_names(self) -> tuple[str, ...]:
        """Endpoint class + role + every alias (deduped, order-stable)."""
        out: list[str] = []
        for n in (self.name, self.role, *self.aliases):
            if n and n not in out:
                out.append(n)
        return tuple(out)

    @property
    def routed(self) -> bool:
        return self.status in ("active", "on_demand")


@dataclass(frozen=True)
class Catalog:
    providers: dict[str, ProviderEntry]
    endpoints: dict[str, EndpointEntry]
    hosts: dict[str, str]
    _by_name: dict[str, str]          # any name/alias/role → endpoint class
    #: The intent vocabulary in force: the built-ins, with this deployment's
    #: ``intents:`` stanzas layered over them by name. Built once at load
    #: rather than per request, because it is read on the hot path.
    intents: dict[str, Profile] = field(default_factory=dict)

    # ---- resolution ----
    def canonical(self, name: str) -> str | None:
        return self._by_name.get(str(name).strip())

    def entry(self, name: str) -> EndpointEntry | None:
        c = self.canonical(name)
        return self.endpoints.get(c) if c else None

    def provider(self, name: str) -> ProviderEntry | None:
        return self.providers.get(str(name).strip())

    def provider_for_endpoint(self, name: str) -> ProviderEntry | None:
        e = self.entry(name)
        return self.provider(e.provider) if e else None

    def host_ip(self, name: str) -> str | None:
        """Resolved address of the provider behind ``name`` — a name from the
        ``hosts:`` table, a literal address, or None for a remote provider,
        which has a base URL rather than a host."""
        p = self.provider_for_endpoint(name)
        if p is None or not p.host:
            return None
        return self.hosts.get(p.host, p.host)

    # ---- views ----
    def by_kind(self, *kinds: str) -> list[EndpointEntry]:
        return [e for e in self.endpoints.values() if e.kind in kinds]

    def routed(self) -> list[EndpointEntry]:
        """Endpoints that actually carry traffic. A ``planned`` entry documents
        a name and a shape without being dispatched to — which is how the
        example catalog can show a remote provider's 1:N shape without putting
        an endpoint nobody has a credential for into the routing table."""
        return [e for e in self.endpoints.values() if e.routed]


def _tuple(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    val = raw.get(key) or ()
    if isinstance(val, str):
        val = [val]
    return tuple(str(v).strip() for v in val if str(v).strip())


def _coerce_provider(name: str, raw: dict[str, Any]) -> ProviderEntry:
    return ProviderEntry(
        name=name,
        engine=raw.get("engine", ""),
        host=raw.get("host", ""),
        port=int(raw.get("port", 0) or 0),
        base_url=str(raw.get("base_url", "") or "").rstrip("/"),
        api_key_env=raw.get("api_key_env", ""),
        notes=raw.get("notes", ""),
    )


def _coerce_endpoint(name: str, raw: dict[str, Any]) -> EndpointEntry:
    # A stanza carrying a `retired:` date is retired even without an explicit
    # `status:` — otherwise an entry left in the file with only `retired:` set
    # defaults to "active" and reads as live (audit 2026-07-12, F-2). An
    # explicit `status:` still wins, so a re-activation does not need the audit
    # trail deleted.
    status = raw.get("status") or ("retired" if raw.get("retired") else "active")
    return EndpointEntry(
        name=name,
        provider=raw.get("provider", ""),
        kind=raw.get("kind", "chat"),
        status=status,
        role=raw.get("role", "") or name,
        aliases=_tuple(raw, "aliases"),
        model=raw.get("model", ""),
        context_per_slot=int(raw.get("context_per_slot", 0) or 0),
        slots=int(raw.get("slots", 0) or 0),
        timeout_floor_s=float(raw.get("timeout_floor_s", 0) or 0),
        timeout_ceiling_s=float(raw.get("timeout_ceiling_s", 0) or 0),
        stream_hard_cap_s=float(raw.get("stream_hard_cap_s", 0) or 0),
        failover_to=raw.get("failover_to", ""),
        spill_to=raw.get("spill_to", ""),
        capabilities=dict(raw.get("capabilities") or {}),
        policy=dict(raw.get("policy") or {}),
        notes=raw.get("notes", ""),
    )


#: Fields an ``intents:`` stanza may set. The mirror of ``Profile``, minus its
#: ``name`` (the stanza key) and its ``source`` (which we assign).
#:
#: 🚨 **There is no field here that names an endpoint, and adding one would be
#: the bug.** A profile is shared vocabulary — published to every caller, and
#: from here on written by an operator — so one that named endpoints would be a
#: second routing table to keep in step with ``endpoints:`` below it, and it
#: would break the moment a fleet's classes are spelled differently from the
#: example's. That is the whole reason profiles are expressed in declared
#: CAPABILITIES. A caller may still refuse a specific endpoint per request
#: (``exclude``, ``intent.py``); that is one caller's words about one call, not
#: a table. Guarded by test_intent_config.py.
_PROFILE_FIELDS = ("kind", "requires", "prefer", "summary")


def _coerce_profile(name: str, raw: dict[str, Any]) -> Profile | None:
    """One ``intents:`` stanza → a :class:`Profile`, or ``None`` if unusable.

    🚨 An unusable stanza is REFUSED rather than registered, and the difference
    matters to the caller: a profile requiring a capability nothing can declare
    would resolve to nothing on every request, and the caller would read "no
    endpoint satisfies requires=[...]" — a sentence about the fleet, for a fault
    in the config file. Refused, they get "unknown intent 'x' — known intents:
    [...]", which points at the vocabulary, where the fault actually is. Either
    way the operator gets the notice.
    """
    unknown = sorted(set(raw) - set(_PROFILE_FIELDS))
    if unknown:
        hooks.config_notice(
            source="models.yaml", subject=f"intents.{name}",
            problem="unknown_key",
            detail=(f"intent key(s) {unknown} are not read by any code and "
                    f"have NO effect on this profile"),
            keys=unknown, known=sorted(_PROFILE_FIELDS))

    kind = str(raw.get("kind") or "chat").strip()
    if kind not in KNOWN_KINDS:
        hooks.config_notice(
            source="models.yaml", subject=f"intents.{name}",
            problem="unusable",
            detail=(f"kind {kind!r} is not a routable kind, so this profile is "
                    f"NOT offered — callers asking for {name!r} get an unknown-"
                    f"intent error naming the ones that are"),
            known=sorted(KNOWN_KINDS))
        return None

    prefer = str(raw.get("prefer") or DEFAULT_PREFERENCE).strip()
    if prefer not in PREFERENCES:
        hooks.config_notice(
            source="models.yaml", subject=f"intents.{name}",
            problem="unusable",
            detail=(f"prefer {prefer!r} is not a known preference, so this "
                    f"profile is NOT offered"),
            known=sorted(PREFERENCES))
        return None

    requires = _tuple(raw, "requires")
    bad = sorted(set(requires) - KNOWN_CAPABILITIES)
    if bad:
        hooks.config_notice(
            source="models.yaml", subject=f"intents.{name}",
            problem="unusable",
            detail=(f"requires {bad} — no endpoint can declare a capability "
                    f"the proxy does not know, so this profile would match "
                    f"nothing on every request; it is NOT offered"),
            keys=bad, known=sorted(KNOWN_CAPABILITIES))
        return None

    return Profile(
        name=name, kind=kind, requires=frozenset(requires), prefer=prefer,
        summary=str(raw.get("summary") or "").strip(),
        source="models.yaml",
    )


def _build_intents(raw: dict[str, Any]) -> dict[str, Profile]:
    """The built-ins with this deployment's own layered over them.

    🚨 Layered, not replaced. A file that defines one profile does not delete
    the other nine — a deployment adding ``cheap-bulk`` has said nothing about
    ``reasoning``, and reading it as a whole-table replacement would silently
    empty a vocabulary that `GET /rs/v1/models` publishes and callers code
    against. Overriding by NAME is deliberate and is how a fleet whose
    ``reasoning`` means something particular says so; it is disclosed through
    each profile's ``source``, published beside it.
    """
    table = dict(BUILTIN_PROFILES)
    for name, body in (raw.get("intents") or {}).items():
        key = str(name).strip()
        if not key:
            continue
        profile = _coerce_profile(key, dict(body or {}))
        if profile is None:
            # A refused stanza must not leave the BUILT-IN of the same name in
            # force under the operator's spelling: they would be reading their
            # own summary in the config file and getting ours on the wire.
            table.pop(key, None)
            continue
        table[key] = profile
    return table


_cache: dict[str, Catalog] = {}


def load_catalog(path: str | os.PathLike | None = None, *, force: bool = False) -> Catalog:
    """Parse ``models.yaml`` (cached per resolved path)."""
    p = Path(path or os.environ.get("ROADSTEAD_MODELS_YAML") or _DEFAULT_PATH)
    key = str(p.resolve())
    if not force and key in _cache:
        return _cache[key]
    if not p.exists():
        raise FileNotFoundError(f"model catalog not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}

    providers = {name: _coerce_provider(name, body or {})
                 for name, body in (raw.get("providers") or {}).items()}
    endpoints = {name: _coerce_endpoint(name, body or {})
                 for name, body in (raw.get("endpoints") or {}).items()}

    by_name: dict[str, str] = {}
    #: alias -> the endpoints that claimed it, in file order.
    alias_claims: dict[str, list[str]] = {}
    for e in endpoints.values():
        for alias in e.aliases:
            alias_claims.setdefault(alias, []).append(e.name)
            by_name.setdefault(alias, e.name)
    # Class and role are registered AFTER the aliases so neither can be
    # shadowed by another endpoint's alias — a collision there once routed a
    # whole class to the wrong backend (ledger `endpoint-class-alias-collision`).
    for e in endpoints.values():
        by_name[e.role] = e.name
    for e in endpoints.values():
        by_name[e.name] = e.name

    # 🚨 An alias resolves to exactly ONE endpoint, and until 2026-09-01 a second
    # claim on it was resolved silently by file order. The origin monorepo
    # RAISED here, and that took a whole gateway down at import when somebody
    # moved an alias between two stanzas without deleting the old one — so
    # refusing to load is not the answer either (a typo must not stop a fleet
    # booting; same rule as `policy:` below). Reporting is. The operator wrote a
    # name that routes somewhere they did not intend, which is precisely the
    # declared-vs-in-force gap `hooks.config_notice` exists to close.
    #
    # A MOVE is a delete plus an add — the ledger's lesson, and the shape that
    # produces this every time.
    for alias, claimants in alias_claims.items():
        if len(claimants) > 1:
            hooks.config_notice(
                source="models.yaml", subject=f"endpoints.aliases.{alias}",
                problem="duplicate",
                detail=(f"alias {alias!r} is claimed by {claimants} — an alias "
                        f"resolves to exactly one endpoint, so it routes to "
                        f"{by_name[alias]!r} and the other claim(s) have NO "
                        f"effect. A move is a delete plus an add"),
                # `in_force` rather than `known`: every other notice uses
                # `known` for the set of VALID keys, and the question here
                # is not which names are legal but which endpoint the name
                # actually reaches. That is the plane's own vocabulary.
                keys=claimants, in_force=by_name[alias])

    # The other half of the same ambiguity: an alias that is some OTHER
    # endpoint's class or role is overwritten by the two loops above. That
    # ordering is deliberate and stays — a class must be reachable by its own
    # name — but the alias it silently kills was still written by somebody.
    for e in endpoints.values():
        for alias in e.aliases:
            winner = by_name.get(alias)
            if winner is not None and winner != e.name:
                if alias in alias_claims and len(alias_claims[alias]) > 1:
                    continue                      # already reported above
                hooks.config_notice(
                    source="models.yaml", subject=f"endpoints.{e.name}.aliases",
                    problem="shadowed",
                    detail=(f"alias {alias!r} on {e.name!r} is also the class or "
                            f"role of {winner!r}, which wins — so this alias has "
                            f"NO effect and callers using it reach {winner!r}"),
                    keys=[alias], in_force=winner)

    cat = Catalog(providers=providers, endpoints=endpoints,
                  hosts=dict(raw.get("hosts") or {}), _by_name=by_name,
                  intents=_build_intents(raw))
    _cache[key] = cat
    return cat


#: Endpoint ``policy.*`` keys copied straight through to ``EndpointConfig``.
#:
#: ⚠️ A key that is NOT here is SILENTLY DROPPED from models.yaml. Each of the
#: guards named below exists because that failure looks exactly like a knob
#: that was never load-bearing.
_POLICY_PASSTHROUGH = (
    "background_floor_pct",
    "fast_path_reserve_slots",
    "dispatch_concurrency_cap",
    "slot_affinity",
    "skip_discovery",
    "min_expected_slots",
    # The vLLM backend's real --max-num-seqs launch cap, mirrored from the serve
    # script (vLLM does not expose it) — drives the shadow max_slots-drift
    # reconciler.
    "documented_max_num_seqs",
    # The backend's `--structured-outputs-config {"disable_any_whitespace":true}`
    # launch flag, mirrored from the serve script (vLLM does not expose it
    # either) — drives Correction.apply_json_object_guard, which strips a bare
    # `response_format:{"type":"json_object"}` before it reaches a
    # whitespace-banned grammar and greedily returns `{}`. Guarded by
    # test_json_object_guard.py::test_models_yaml_flag_reaches_endpoint_config.
    "disable_any_whitespace",
    # Operator-declared token price, USD per MILLION tokens. Beats a
    # provider-published price (spend.PriceBook explains why) and is the only
    # way to price an endpoint whose backend publishes nothing. Guarded by
    # test_spend.py::test_declared_price_reaches_endpoint_config.
    "input_usd_per_mtok",
    "output_usd_per_mtok",
    # Fraction of max_tokens allowed for REASONING on a backend launched with
    # `--reasoning-config` (vLLM will not honour `thinking_token_budget` without
    # it, and 400s the request instead). Absent/0 = inject nothing, which is
    # what every endpoint except a reasoner wants.
    "thinking_budget_ratio",
    # Marks the endpoint backing the conversational lane, which is what
    # `/readyz` fails closed on. Guarded by
    # test_readyz.py::test_readiness_critical_flag_reaches_endpoint_config.
    "readiness_critical",
    # Minimum dwell in degraded mode before flipping back. Guarded by
    # test_failover.py::test_dwell_reaches_endpoint_config.
    "failover_dwell_s",
)


#: ``policy:`` keys read OUTSIDE the passthrough loop above, because their YAML
#: shape is not their ``EndpointConfig`` shape. Listed here so the unknown-key
#: notice does not report a key that is, in fact, load-bearing.
_POLICY_HANDLED = frozenset({"model_fingerprint", "thinking_kwargs"})


def build_endpoint_kwargs(cat: Catalog | None = None,
                          entries: list["EndpointEntry"] | None = None,
                          ) -> dict[str, dict[str, Any]]:
    """Per-endpoint-class kwargs for constructing proxy ``EndpointConfig``.

    Keyed by endpoint class; one entry per ROUTED endpoint. ``config`` builds
    ``EndpointConfig(**kwargs)`` from this (EndpointConfig lives there, so we
    return plain kwargs to avoid a cycle).

    ``entries`` overrides *which* entries are built, and exists for one caller:
    the management plane promoting a `planned` endpoint into service
    (roadmap J1). It needs the identical kwargs for an entry that is by
    definition not in ``cat.routed()`` yet, and the alternative — a second
    builder — is the "two places deciding the same thing" shape this repo keeps
    paying for. 🚨 It changes only the SELECTION; every rule below is the same
    code on the same catalog, so a promoted endpoint is configured exactly as a
    restart would have configured it.
    """
    cat = cat or load_catalog()
    out: dict[str, dict[str, Any]] = {}
    for e in (cat.routed() if entries is None else entries):
        pol = e.policy
        kw: dict[str, Any] = {
            "endpoint_class": e.name,
            "role": e.role,
            "max_slots": e.slots,
            "context_per_slot": e.context_per_slot,
        }
        p = cat.provider(e.provider) or ProviderEntry(name="")
        # --- connection, entirely from the provider ---
        if p.base_url:
            kw["base_url"] = p.base_url
        else:
            kw["host"] = cat.hosts.get(p.host, p.host)
            kw["port"] = p.port
        if p.api_key_env:
            kw["api_key_env"] = p.api_key_env
        # Mirror the engine whenever it names a provider that is NOT the
        # default. `shim` and a decorated `llama.cpp (Vulkan)` resolve to the
        # default and are not mirrored, so EndpointConfig keeps its own default.
        provider = provider_for_engine(p.engine)
        if provider is not DEFAULT_PROVIDER:
            kw["backend_engine"] = p.engine
        # --- what model this endpoint is ---
        if e.model and not provider.descriptor.publishes_served_model_id:
            # A provider fronting a catalogue cannot be asked which model it
            # serves — the answer is "hundreds" — so the pin is config. A
            # provider that serves ONE model and names it is discovered
            # instead, and seeding it from config would go stale in silence.
            kw["served_model_id"] = e.model
        if e.status == "on_demand":
            # Not always-resident, so health must not page when it is simply
            # not loaded (health.Health skips probing when on_demand and not
            # loaded). Without this the catalog's `status: on_demand` never
            # reached EndpointConfig and a deliberately-stopped backend logged
            # CRITICAL "UNHEALTHY — 3 consecutive probe failures" every cycle.
            kw["on_demand"] = True
        # --- capabilities ---
        # The declaration itself, carried whole. Everything below this line
        # DERIVES a narrower fact from one of these; nothing replaces it.
        # `kind` rides along for the same reason — see EndpointConfig.kind.
        kw["kind"] = e.kind
        declared = frozenset(
            name for name, on in e.capabilities.items()
            if on and isinstance(name, str))
        kw["capabilities"] = declared
        unknown = sorted(declared - KNOWN_CAPABILITIES)
        if unknown:
            # WARN and keep serving. The catalog is the operator's, and a
            # capability we have not thought of yet is a thing they are allowed
            # to write down — but the commonest reason for one is a typo
            # (`tool_call`, `structured_outputs`), and a mistyped capability is
            # invisible in exactly the way a dropped `policy.*` key is: the
            # endpoint keeps working and quietly never matches the intent that
            # was supposed to reach it.
            logger.warning(
                "models.yaml endpoint %r declares unknown capabilities %s — "
                "known: %s. Kept verbatim, but no intent will match one: check "
                "for a typo.", e.name, unknown, sorted(KNOWN_CAPABILITIES))
        if (e.capabilities.get("reasoning")
                and not provider.descriptor.reasoning_is_switchable):
            # A reasoning model on a backend with no proxy-side kill switch
            # (llama.cpp) emits its CoT UNCONDITIONALLY — unlike vLLM, where
            # reasoning is OFF by default and only a per-request `thinking:true`
            # opt-in turns it on, via apply_thinking which adds its OWN budget.
            # So only the forced case needs the submit-time answer-headroom
            # reserve (forced_reasoning_budget); a vLLM reasoning endpoint must
            # NOT get it.
            kw["forces_reasoning"] = True
        if e.capabilities.get("vision"):
            # Mirror the declaration so the submit path can READ it. This was
            # pure documentation once — nothing consulted it — and an alias
            # move sent an image-describe call to a box with no mmproj for a
            # day without a single test or alert noticing. Ledger:
            # `a-role-rename-carried-vision-to-a-text-only-box`.
            kw["vision"] = True
        # --- policy passthrough ---
        for src in _POLICY_PASSTHROUGH:
            if src in pol:
                kw[src] = pol[src]
        # 🚨 Everything else in `policy:` is DROPPED, and a dropped knob looks
        # exactly like a knob that was never load-bearing. Reported (never
        # raised — a typo must not stop a fleet booting) and retained, so
        # `GET /rs/v1/admin/providers` can answer "what did I write that is not
        # in force?" without anybody grepping a startup log. See hooks.py.
        unknown_policy = sorted(set(pol) - set(_POLICY_PASSTHROUGH) - _POLICY_HANDLED)
        if unknown_policy:
            hooks.config_notice(
                source="models.yaml",
                subject=f"endpoints.{e.name}.policy",
                problem="unknown_key",
                detail=(f"policy key(s) {unknown_policy} are not read by any "
                        f"code and have NO effect on this endpoint"),
                keys=unknown_policy,
                known=sorted(set(_POLICY_PASSTHROUGH) | _POLICY_HANDLED),
            )
        # Declared per MODEL and checked against what the backend actually
        # serves; see health.model_swap_alerts.
        raw_fp = pol.get("model_fingerprint")
        if isinstance(raw_fp, str) and raw_fp.strip():
            kw["model_fingerprint"] = raw_fp.strip()
        # Not in the passthrough loop because YAML hands us a LIST and
        # EndpointConfig holds a tuple — a list would be mutable state shared
        # across endpoints. Declares which chat-template variable(s) switch
        # reasoning for THIS model. Guarded by
        # test_thinking_kwargs_are_family_aware.py.
        raw_tk = pol.get("thinking_kwargs")
        if isinstance(raw_tk, (list, tuple)):
            kw["thinking_kwargs"] = tuple(
                str(k) for k in raw_tk if isinstance(k, str) and k.strip())
        elif isinstance(raw_tk, str) and raw_tk.strip():
            kw["thinking_kwargs"] = (raw_tk.strip(),)
        out[e.name] = kw

    # Make `failover_to:` load-bearing, and only when the target is itself a
    # routed endpoint. A failover naming a retired or planned entry resolves to
    # nothing rather than arming a degrade to a backend that cannot serve.
    #
    # 🚨 Deliberately NOT an alias: `normalize_endpoint()` is untouched and stays
    # idempotent. Registering the target as an alias of the source is what once
    # made every backup request route to the primary it was meant to replace
    # (ledger `endpoint-class-alias-collision`).
    for e in cat.routed():
        target = e.failover_to
        if not target or e.name not in out or target == e.name:
            continue
        if target in out:
            out[e.name]["failover_to"] = target

    # Same treatment for `spill_to:`, and for the same reason: a spill target
    # that is not itself routed must resolve to nothing rather than arming an
    # overflow path to a backend that cannot serve. The commonest way to get
    # this wrong is pointing at one of the `planned` remote endpoints before the
    # credential is in the environment — `routed()` is what makes that a no-op
    # instead of a 502 the first time the local tier fills up.
    for e in cat.routed():
        target = e.spill_to
        if not target or e.name not in out or target == e.name:
            continue
        if target in out:
            out[e.name]["spill_to"] = target
    return out


def build_role_to_class(cat: Catalog | None = None) -> dict[str, str]:
    """role string OR alias → endpoint class.

    Covers aliases as well as roles: when a name is retired and becomes a pure
    alias of a different endpoint, callers passing the legacy name as
    ``endpoint=`` need it to keep resolving. Role wins over alias on collision
    (written second so an alias cannot overwrite it).
    """
    cat = cat or load_catalog()
    out: dict[str, str] = {}
    for e in cat.endpoints.values():
        for a in e.aliases:
            out[a] = e.name
    for e in cat.endpoints.values():
        out[e.role] = e.name
        out[e.name] = e.name
    return out


def build_class_to_role(cat: Catalog | None = None) -> dict[str, str]:
    """endpoint class → its role string."""
    cat = cat or load_catalog()
    return {e.name: e.role for e in cat.endpoints.values()}


def _class_map(cat: Catalog, attr: str) -> dict[str, float]:
    return {e.name: float(getattr(e, attr))
            for e in cat.routed() if getattr(e, attr) and getattr(e, attr) > 0}


def build_class_floors(cat: Catalog | None = None) -> dict[str, float]:
    """endpoint class → per-class timeout floor (seconds).

    This is what makes the catalog the ENFORCED authority for timeout floors.
    ``TimeoutModel`` is seeded from it, so a floor bump in the yaml actually
    changes enforcement instead of being an inert field. Classes absent here
    fall back to ``timeout_model.FLOOR_S``.
    """
    return _class_map(cat or load_catalog(), "timeout_floor_s")


def build_class_ceilings(cat: Catalog | None = None) -> dict[str, float]:
    """endpoint class → per-class timeout CEILING (seconds).

    The per-class override that lets an inherently long-running class keep a
    generous upper bound even on an interactive tier, overriding the band in
    ``timeout_model.resolve_ceiling_s``. Absent → the tier band.
    """
    return _class_map(cat or load_catalog(), "timeout_ceiling_s")


def build_class_stream_hard_caps(cat: Catalog | None = None) -> dict[str, float]:
    """endpoint class → absolute streaming hard cap (seconds).

    Bounds how far a PROXY-CHOSEN streaming deadline may be extended while the
    stream is still emitting tokens; an explicit caller deadline never consults
    it. Absent → the band default in ``constants._STREAM_HARD_CAP_*``.
    """
    return _class_map(cat or load_catalog(), "stream_hard_cap_s")
