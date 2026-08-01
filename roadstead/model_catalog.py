"""Loader for the fleet model-naming authority (``models.yaml``).

``models.yaml`` (next to this file) is THE single source of truth for every
model the fleet runs — names, aliases, hosts/ports, capabilities and policy.
This module parses it once and exposes:

  * :class:`ModelEntry` / :func:`load_catalog` — the typed catalog.
  * ``canonical`` / ``entry`` / ``resolve_*`` — name resolution helpers.
  * ``build_*`` functions — derive the legacy structures that used to be
    hand-maintained copies (proxy ``DEFAULT_ENDPOINTS`` kwargs,
    ``ROLE_TO_CLASS``, the framework port + alias maps, ``VALID_PROVIDERS``,
    the per-provider context windows, the telemetry port→role map).

Pure: only ``yaml`` + stdlib. Deliberately does NOT import ``framework`` or
``llmproxy.config`` so any consumer can import it without a cycle (the proxy
``config`` module imports the ``build_*`` helpers from here, not vice-versa).

Override the file path with ``COLLECTIVE_MODELS_YAML``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parent / "models.yaml"


@dataclass(frozen=True)
class ModelEntry:
    """One model in the authority. ``name`` is the function-based canonical id."""

    name: str
    kind: str
    status: str
    role: str                         # proxy/telemetry role string (pre-rename canonical)
    aliases: tuple[str, ...] = ()
    proxy_endpoint: bool = False
    endpoint_class: str = ""
    host: str = ""
    port: int = 0
    wyoming_port: int = 0
    backend_engine: str = ""
    # --- off-package derive fields (telemetry UNITS + dispatcher registry) ---
    systemd_unit: str = ""          # box unit name (no .service) for telemetry UNITS
    probe_kind: str = ""            # telemetry probe type: vllm | llama | infinity | health
    dispatcher_unit_key: str = ""   # anvil-dispatcher MODEL_REGISTRY key (media/creative)
    served_model_names: tuple[str, ...] = ()
    model_family: str = ""
    quant: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    context_per_slot: int = 0
    context_window: int = 0
    slots: int = 0
    timeout_floor_s: float = 0.0
    timeout_ceiling_s: float = 0.0
    policy: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    dispatcher_capability: str = ""
    dispatcher_capability_aliases: tuple[str, ...] = ()
    unit: str = ""
    manage_process: bool = True
    health_path: str = ""
    warmup_sec: int = 0
    token_speed: dict[str, Any] = field(default_factory=dict)
    purpose: str = ""
    when_used: tuple[str, ...] = ()
    # RESERVED: active but pinned to a single consumer (e.g. the chat loop) — not a
    # general-purpose role. The Inference page still shows it (it IS live), but the
    # Model Playground filters it out so it can't be manually driven.
    reserved: bool = False
    fallback: str | None = None
    notes: str = ""

    @property
    def all_names(self) -> tuple[str, ...]:
        """Canonical name + role + every alias (deduped, order-stable)."""
        out: list[str] = []
        for n in (self.name, self.role, *self.aliases):
            if n and n not in out:
                out.append(n)
        return tuple(out)


@dataclass(frozen=True)
class Catalog:
    models: dict[str, ModelEntry]
    hosts: dict[str, str]
    extra_context_windows: dict[str, int]
    decommissioned: dict[str, dict[str, Any]]
    _by_name: dict[str, str]          # any name/alias/role → canonical

    # ---- resolution ----
    def canonical(self, name: str) -> str | None:
        return self._by_name.get(str(name).strip())

    def entry(self, name: str) -> ModelEntry | None:
        c = self.canonical(name)
        return self.models.get(c) if c else None

    def host_ip(self, name: str) -> str | None:
        e = self.entry(name)
        return self.hosts.get(e.host) if e else None

    # ---- views ----
    def by_kind(self, *kinds: str) -> list[ModelEntry]:
        return [e for e in self.models.values() if e.kind in kinds]

    def chat_roles(self) -> list[ModelEntry]:
        return self.by_kind("chat")

    def media(self) -> list[ModelEntry]:
        return self.by_kind("media")

    def proxy_endpoints(self) -> list[ModelEntry]:
        return [e for e in self.models.values() if e.proxy_endpoint]


def _coerce_entry(name: str, raw: dict[str, Any]) -> ModelEntry:
    def _t(key: str) -> tuple:
        v = raw.get(key) or []
        return tuple(v)

    # A stanza that carries a `retired:` key (a retirement date) is retired,
    # even if it lacks an explicit `status:` — otherwise a model left in the
    # `models:` block with only `retired:` set silently defaults to "active"
    # and reads as live (audit 2026-07-12, F-2). An explicit `status:` still
    # wins (lets a re-activation override without deleting the audit trail).
    status = raw.get("status") or ("retired" if raw.get("retired") else "active")

    return ModelEntry(
        name=name,
        kind=raw.get("kind", ""),
        status=status,
        role=raw.get("role", name),
        aliases=_t("aliases"),
        proxy_endpoint=bool(raw.get("proxy_endpoint", False)),
        endpoint_class=raw.get("endpoint_class", ""),
        host=raw.get("host", ""),
        port=int(raw.get("port", 0) or 0),
        wyoming_port=int(raw.get("wyoming_port", 0) or 0),
        backend_engine=raw.get("backend_engine", ""),
        systemd_unit=raw.get("systemd_unit", ""),
        probe_kind=raw.get("probe_kind", ""),
        dispatcher_unit_key=raw.get("dispatcher_unit_key", ""),
        served_model_names=_t("served_model_names"),
        model_family=raw.get("model_family", ""),
        quant=raw.get("quant", ""),
        params=dict(raw.get("params") or {}),
        context_per_slot=int(raw.get("context_per_slot", 0) or 0),
        context_window=int(raw.get("context_window", 0) or 0),
        slots=int(raw.get("slots", 0) or 0),
        timeout_floor_s=float(raw.get("timeout_floor_s", 0) or 0),
        timeout_ceiling_s=float(raw.get("timeout_ceiling_s", 0) or 0),
        policy=dict(raw.get("policy") or {}),
        capabilities=dict(raw.get("capabilities") or {}),
        dispatcher_capability=raw.get("dispatcher_capability", ""),
        dispatcher_capability_aliases=_t("dispatcher_capability_aliases"),
        unit=raw.get("unit", ""),
        manage_process=bool(raw.get("manage_process", True)),
        health_path=raw.get("health_path", ""),
        warmup_sec=int(raw.get("warmup_sec", 0) or 0),
        token_speed=dict(raw.get("token_speed") or {}),
        purpose=raw.get("purpose", ""),
        when_used=_t("when_used"),
        reserved=bool(raw.get("reserved", False)),
        fallback=raw.get("fallback"),
        notes=raw.get("notes", "") or "",
    )


_cache: dict[str, Catalog] = {}


def load_catalog(path: str | os.PathLike | None = None, *, force: bool = False) -> Catalog:
    """Parse ``models.yaml`` (cached per resolved path)."""
    p = Path(path or os.environ.get("COLLECTIVE_MODELS_YAML") or _DEFAULT_PATH)
    key = str(p.resolve())
    if not force and key in _cache:
        return _cache[key]
    if not p.exists():
        raise FileNotFoundError(f"model catalog not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    models = {name: _coerce_entry(name, body or {}) for name, body in (raw.get("models") or {}).items()}

    by_name: dict[str, str] = {}
    for name, e in models.items():
        for n in e.all_names:
            prev = by_name.get(n)
            if prev and prev != name:
                raise ValueError(f"name collision: {n!r} maps to both {prev!r} and {name!r}")
            by_name[n] = name

    cat = Catalog(
        models=models,
        hosts=dict(raw.get("meta", {}).get("hosts") or {}),
        extra_context_windows=dict(raw.get("meta", {}).get("extra_context_windows") or {}),
        decommissioned=dict(raw.get("decommissioned") or {}),
        _by_name=by_name,
    )
    _cache[key] = cat
    return cat


# ===========================================================================
# Builders — derive the legacy structures from the catalog.
# These are the functions the rest of the codebase calls instead of holding
# their own literal copies.
# ===========================================================================

def build_endpoint_kwargs(cat: Catalog | None = None) -> dict[str, dict[str, Any]]:
    """Per-endpoint-class kwargs for constructing proxy ``EndpointConfig``.

    Keyed by ``endpoint_class``; one entry per ``proxy_endpoint: true`` role.
    The proxy ``config`` module builds ``EndpointConfig(**kwargs)`` from this
    (EndpointConfig lives there, so we return plain kwargs to avoid a cycle).
    """
    cat = cat or load_catalog()
    out: dict[str, dict[str, Any]] = {}
    for e in cat.proxy_endpoints():
        pol = e.policy
        kw: dict[str, Any] = {
            "endpoint_class": e.endpoint_class,
            "role": e.role,
            "max_slots": e.slots,
            "context_per_slot": e.context_per_slot,
            "host": cat.hosts.get(e.host, e.host),
            "port": e.port,
        }
        if e.status == "on_demand":
            # An on_demand endpoint is NOT always-resident, so health must not page when it is
            # simply not loaded (health.Health skips probing when on_demand and not loaded).
            # Without this the catalog's `status: on_demand` never reached EndpointConfig and a
            # deliberately-stopped backend logged CRITICAL "UNHEALTHY — 3 consecutive probe
            # failures" every cycle. Found 2026-07-31 after tier3-backup was stopped per §2b.
            kw["on_demand"] = True
        if e.backend_engine == "vllm":
            kw["backend_engine"] = "vllm"
        if e.capabilities.get("reasoning") and e.backend_engine != "vllm":
            # A non-vLLM (llama.cpp) reasoning model emits its CoT UNCONDITIONALLY —
            # there is no proxy-side kill switch (unlike vLLM, where reasoning is OFF
            # by default and only a per-request `thinking:true` opt-in turns it on, via
            # apply_thinking which adds its OWN budget). So only the forced/llama.cpp
            # case needs the submit-time answer-headroom reserve (forced_reasoning_budget);
            # a vLLM reasoning endpoint (e.g. the thinker) must NOT get it. e.g. creative
            # / Trinity-Mini.
            kw["forces_reasoning"] = True
        for src, dst in (
            ("background_floor_pct", "background_floor_pct"),
            ("fast_path_reserve_slots", "fast_path_reserve_slots"),
            ("dispatch_concurrency_cap", "dispatch_concurrency_cap"),
            ("slot_affinity", "slot_affinity"),
            ("skip_discovery", "skip_discovery"),
            # Step 4c: the vLLM backend's real --max-num-seqs launch cap, mirrored
            # from the serve script (vLLM doesn't expose it) — drives the shadow
            # max_slots-drift reconciler.
            ("documented_max_num_seqs", "documented_max_num_seqs"),
            # 2026-08-01: the backend's `--structured-outputs-config
            # {"disable_any_whitespace":true}` launch flag, mirrored from the
            # serve script (vLLM doesn't expose it either) — drives
            # Correction.apply_json_object_guard, which strips a bare
            # `response_format:{"type":"json_object"}` before it reaches a
            # whitespace-banned grammar and greedily returns `{}`.
            # ⚠️ A key that is NOT in this tuple list is SILENTLY DROPPED from
            # models.yaml — see the `min_expected_slots` note in models.yaml.
            # test_json_object_guard.py::test_models_yaml_flag_reaches_endpoint_config
            # is the guard that this one is really wired.
            ("disable_any_whitespace", "disable_any_whitespace"),
        ):
            if src in pol:
                kw[dst] = pol[src]
        out[e.endpoint_class] = kw
    return out


def build_role_to_class(cat: Catalog | None = None) -> dict[str, str]:
    """role string OR alias → endpoint class (mirrors the old ROLE_TO_CLASS).

    Must cover aliases too, not just the canonical ``role:`` string: when a
    role is retired and its old name becomes a pure alias of a different
    model (e.g. ``qwen-analyst``/``chat`` -> ``classify`` after the
    2026-07-03 decommission), callers passing the legacy name as
    ``endpoint=`` need it to keep resolving. Role wins over alias on
    collision (checked second so it can't be overwritten by another entry's
    alias)."""
    cat = cat or load_catalog()
    out: dict[str, str] = {}
    for e in cat.models.values():
        if not e.endpoint_class:
            continue
        for a in e.aliases:
            out[a] = e.endpoint_class
        out[e.role] = e.endpoint_class
    return out


def build_class_to_role(cat: Catalog | None = None) -> dict[str, str]:
    """endpoint class → its representative (proxy-endpoint-owner) role string.

    Deterministic — uses the ``proxy_endpoint: true`` owner, not a reverse-dict
    last-wins (so ``gemma`` → ``gemma-router``, never the shared ``gemma-greeter``)."""
    cat = cat or load_catalog()
    return {e.endpoint_class: e.role for e in cat.proxy_endpoints() if e.endpoint_class}


def build_class_floors(cat: Catalog | None = None) -> dict[str, float]:
    """endpoint class → per-class timeout floor (seconds), from ``timeout_floor_s``.

    This is what makes ``models.yaml`` the ENFORCED authority for timeout floors
    (it declares itself the single source of truth). ``TimeoutModel`` is seeded
    from this so a floor bump in the yaml — e.g. classify 45→180 / composer
    180→360 for the 2026-07 nexus model swap — actually changes enforcement
    instead of being an inert field. Only proxy-endpoint owners with a positive
    floor are emitted; classes absent here fall back to ``timeout_model.FLOOR_S``.
    Deterministic (proxy-endpoint owner per class, like ``build_class_to_role``)."""
    cat = cat or load_catalog()
    out: dict[str, float] = {}
    for e in cat.proxy_endpoints():
        if e.endpoint_class and e.timeout_floor_s and e.timeout_floor_s > 0:
            out[e.endpoint_class] = float(e.timeout_floor_s)
    return out


def build_class_ceilings(cat: Catalog | None = None) -> dict[str, float]:
    """endpoint class → per-class timeout CEILING (seconds), from
    ``timeout_ceiling_s``.  The per-role override that lets an inherently
    long-running class (e.g. ``creative`` song-compose, the generation roles)
    keep a generous upper bound even on an interactive tier, overriding the
    interactive/background tier band in ``timeout_model.resolve_ceiling_s``.
    Only proxy-endpoint owners with a positive ceiling are emitted; classes
    absent here fall back to the tier band.  Deterministic (proxy-endpoint
    owner per class, like ``build_class_floors``)."""
    cat = cat or load_catalog()
    out: dict[str, float] = {}
    for e in cat.proxy_endpoints():
        if e.endpoint_class and e.timeout_ceiling_s and e.timeout_ceiling_s > 0:
            out[e.endpoint_class] = float(e.timeout_ceiling_s)
    return out


def build_role_aliases(cat: Catalog | None = None) -> dict[str, str]:
    """alias → role string (mirrors the old _ROLE_ALIASES). Every non-role
    alias of every entry resolves to that entry's role string."""
    cat = cat or load_catalog()
    out: dict[str, str] = {}
    for e in cat.models.values():
        for a in e.aliases:
            if a != e.role:
                out[a] = e.role
    return out


def build_host_ports(cat: Catalog | None = None, host: str = "") -> dict[str, int]:
    """role string → port for every model on ``host`` (nexus/anvil/nasbox)."""
    cat = cat or load_catalog()
    return {e.role: e.port for e in cat.models.values() if e.host == host and e.port}


def build_context_windows(cat: Catalog | None = None) -> dict[str, int]:
    """provider/role name → context window (mirrors _DEFAULT_CONTEXT_WINDOW).

    Includes the canonical name, the role string and every alias, plus the
    ``meta.extra_context_windows`` overrides preserved verbatim (anthropic +
    legacy nexus-* values that differ from their live role)."""
    cat = cat or load_catalog()
    out: dict[str, int] = {}
    for e in cat.by_kind("chat", "embed", "rerank"):
        cw = e.context_window or e.context_per_slot
        if not cw:
            continue
        for n in e.all_names:
            out.setdefault(n, cw)
    # explicit overrides win (legacy values that must stay verbatim)
    out.update(cat.extra_context_windows)
    return out


def build_valid_providers(cat: Catalog | None = None) -> frozenset[str]:
    """All resolvable provider names (canonical + role + aliases) across every
    entry, plus the legacy extra_context_windows keys and ``anthropic``
    (mirrors VALID_PROVIDERS — a permissive allowlist for policy/DB rows)."""
    cat = cat or load_catalog()
    names: set[str] = {"anthropic"}
    for e in cat.models.values():
        names.update(e.all_names)
    names.update(cat.extra_context_windows.keys())
    return frozenset(names)


def build_telemetry_units(cat: Catalog | None = None, host: str = "") -> list[tuple[str, str, int, str]]:
    """(role, systemd_unit, port, probe_kind) for every telemetry-probed model on
    ``host`` (nexus/anvil). Drives the per-host telemetry UNITS tables + health-verifier
    health. Scope: chat/embed/rerank/ocr/tts roles with their OWN backend on the
    box (classifier shares router's E4B → excluded); media is dispatcher-managed
    (not telemetry); ``planned`` roles are excluded. Order is catalog order."""
    cat = cat or load_catalog()
    probed = {"chat", "embed", "rerank", "ocr", "tts"}
    out: list[tuple[str, str, int, str]] = []
    for e in cat.models.values():
        if e.host != host or e.status not in ("active", "on_demand"):
            continue
        if e.kind not in probed or not e.systemd_unit:
            continue
        if e.kind == "chat" and not e.proxy_endpoint:
            continue  # shares another role's backend (classifier -> router E4B)
        out.append((e.role, e.systemd_unit, e.port, e.probe_kind or "health"))
    return out


def build_dispatcher_entries(cat: Catalog | None = None) -> list[dict[str, Any]]:
    """One dict per anvil-dispatcher-managed model (media + the pinned creative
    LLM) — everything needed to build MODEL_REGISTRY + DEFAULT_POLICY. Excludes
    ``planned`` roles (e.g. video). The dispatcher constructs its own ModelMeta
    from these (ModelMeta lives there, so we return plain dicts)."""
    cat = cat or load_catalog()
    out: list[dict[str, Any]] = []
    for e in cat.models.values():
        if e.host != "anvil" or not e.dispatcher_capability:
            continue
        if e.status not in ("active", "on_demand"):
            continue
        out.append({
            "key": e.dispatcher_unit_key or e.role,
            "unit": e.unit,
            "port": e.port,
            "health_path": e.health_path or "/health",
            "warmup_sec": e.warmup_sec or 60,
            "manage_process": e.manage_process,
            "capability": e.dispatcher_capability,
            "capability_aliases": list(e.dispatcher_capability_aliases),
        })
    return out


def build_port_to_role(cat: Catalog | None = None) -> dict[str, str]:
    """port (str) → role string for telemetry call-log attribution.

    Active + on_demand models only; the role string is the telemetry name.
    wyoming_port maps to ``<role>-wyoming`` for the stream relay."""
    cat = cat or load_catalog()
    out: dict[str, str] = {}
    for e in cat.models.values():
        if e.port:
            out.setdefault(str(e.port), e.role)
        if e.wyoming_port:
            out.setdefault(str(e.wyoming_port), f"{e.role}-wyoming")
    return out
