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

Pure: only ``yaml``, stdlib, and ``roadstead.providers`` (which imports nothing
back). ``config`` imports the ``build_*`` helpers from here, never the reverse.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .providers import DEFAULT_PROVIDER, provider_for_engine

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
        capabilities=dict(raw.get("capabilities") or {}),
        policy=dict(raw.get("policy") or {}),
        notes=raw.get("notes", ""),
    )


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
    for e in endpoints.values():
        for alias in e.aliases:
            by_name.setdefault(alias, e.name)
    # Class and role are registered AFTER the aliases so neither can be
    # shadowed by another endpoint's alias — a collision there once routed a
    # whole class to the wrong backend (ledger `endpoint-class-alias-collision`).
    for e in endpoints.values():
        by_name[e.role] = e.name
    for e in endpoints.values():
        by_name[e.name] = e.name

    cat = Catalog(providers=providers, endpoints=endpoints,
                  hosts=dict(raw.get("hosts") or {}), _by_name=by_name)
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


def build_endpoint_kwargs(cat: Catalog | None = None) -> dict[str, dict[str, Any]]:
    """Per-endpoint-class kwargs for constructing proxy ``EndpointConfig``.

    Keyed by endpoint class; one entry per ROUTED endpoint. ``config`` builds
    ``EndpointConfig(**kwargs)`` from this (EndpointConfig lives there, so we
    return plain kwargs to avoid a cycle).
    """
    cat = cat or load_catalog()
    out: dict[str, dict[str, Any]] = {}
    for e in cat.routed():
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
