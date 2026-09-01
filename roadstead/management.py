"""The management plane — read first, control second (roadmap Workstream E).

Everything B and D built is now managed by editing a file and restarting the
process. That is workable for a fleet one person runs and unworkable for
anything else, and it is the last unstarted workstream because there was nothing
worth managing until keys, quotas, budgets and providers all existed.

---

## What this surface is FOR

An operator's questions are not the caller's questions one level up. A caller
asks *what can serve me, how long will it take, what did it cost* — that is
``/rs/v1/models`` and the enriched envelope. An operator asks something the
package could not answer at all:

🚨 **"What did I write that is not in force?"**

That is the thesis of this module, and every view here is shaped by it. A
gateway's configuration is a pile of YAML, environment variables and discovered
facts that agree with each other most of the time, and every expensive failure in
``docs/ledger.md`` lives in a gap between two of them:

* a ``policy:`` key that is not in ``_POLICY_PASSTHROUGH`` and was dropped in
  silence — indistinguishable, from outside, from a knob that was never
  load-bearing;
* an endpoint whose declared ``slots:`` was overwritten by discovery, or was not
  because the engine publishes nothing;
* a declared model fingerprint the backend stopped matching;
* an ``api_key_env`` naming a variable nobody exported, so an endpoint fails its
  health probe for a reason no health page states;
* a caller whose quota stanza has a typo, so it looks exactly like a caller who
  never opted in.

So the views here report the **declared** value beside the one **in force**, and
``GET /rs/v1/admin/config`` reads back every ``hooks.config_notice`` — the
retained record of things an operator wrote that no code reads. A surface that
merely echoed ``models.yaml`` back would be a worse version of ``cat``.

## Where it lives, and why not on ``/v1``

``/rs/v1/admin/*``. ``/v1`` is versioned by OpenAI (``docs/api.md`` §1.7), and the
management API is the surface most likely to need its own second version — it is
the one that grows with the product rather than with a published standard. The
four control routes that predate this module (``pause``/``resume``/``flags``/
``maintenance``) are served at BOTH spellings: they keep ``/v1/admin/*`` because
§3 published them there and external consumers read them, and they gain the
``/rs/v1/admin/*`` spelling so an operator has one prefix rather than two. Same
handlers, same gate, no behaviour difference.

## Doctrine

🚨 **A management surface never emits a credential — not the key, not the
digest.** ``KeyRegistry.snapshot`` already refuses to print a digest, and the
reason generalises: a digest is not a secret cryptographically but it IS a
working credential for anyone who can compute one, so an endpoint that prints it
turns a read-only surface into a key store. The same rule covers
``api_key_env``: the read plane publishes the variable's NAME and whether it
resolved, never its value. ``tests/test_management_plane.py`` sweeps every
response body for both.

🚨 **A runtime edit never rewrites the operator's config file.** ``models.yaml``,
``agents.yaml`` and a keys file are hand-written, commented and usually in
version control; a process that rewrote one would destroy the comments, race the
operator's own editor, and make "who changed this" unanswerable. Runtime changes
go to a separate JSON overlay (:class:`AdminOverlay`) which is *layered over* the
files at startup — so what an operator wrote stays exactly as they wrote it, and
what the API changed is visible as its own list.

🚨 **A control action that cannot be persisted still takes effect, and says so.**
The overlay is unwritable in an embedded deployment, on a read-only disk, or in a
test. Refusing a revocation on those grounds is a correctness argument answered,
in the moment, by a breach — so the mutation applies in memory and the response
carries ``persisted: false`` with the reason. An operator who reads that knows
the change is live and will not survive a restart, which is a fact they can act
on. This is why §3 mints **no new error code** for it (and none at all — the same
decision as §1.6 and §1.7).

🚨 **An edit changes policy, never history.** Lowering a spend cap does not
retroactively charge anybody, raising one does not refund; a weight change moves
the replenish RATE and leaves the deficit already run
(``BudgetManager.reweight``). The alternative — clearing a balance on a config
edit — hands a fresh allowance to precisely the caller an operator is reweighting
because it consumes too much.

🚨 **The write boundary enforces the same doctrine the read path does.** An
unknown field in a PATCH is a 400 that names the known set, never a silent drop:
this is the surface whose entire purpose is to expose that failure, so
reproducing it here would be self-defeating. And a threshold written through this
API can only ever DEGRADE — §1.6's rule is not re-implemented at the boundary, it
is inherited, because there is no field with which to express a rejection.

🚨 **Keys stay FLAT, and the roadmap's "multi-tenancy depth" question is answered
by the shape that already exists.** The budget holder is the ``agent_id``, not
the key — so several keys naming one ``agent_id`` already give a team one quota,
one DRR share and one spend cap, with per-key revocation and per-key priority.
That is what nesting was wanted for. ``GET /rs/v1/admin/keys`` groups by
``agent_id`` to make the structure visible rather than inferable.

## Concurrency

🚨 The objects mutated here — the key registry, ``config.agents``, the DRR
budgets — are read on the hot path, on the loop thread, with no locks
(``CLAUDE.md``). So **the mutation happens on the loop and only the file write
goes off it**, via ``asyncio.to_thread``. That is deliberately stricter than
``flags.py``, which runs its whole ``set_many`` off-loop: a flag dict is written
once in a blue moon and read as a plain lookup, whereas a registry write racing
``resolve()`` on every request is the exact interleaving the invariant exists to
forbid. ``tests/loop_affinity.py`` arms the new mutators, so a future handler
that gets this wrong fails the soak instead of drifting a budget.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from . import hooks, model_catalog
from .config import AgentQuotaConfig, LLMPriority
from .enriched import _error
from .providers import provider_for_engine
from .spend import declared_price

if TYPE_CHECKING:  # pragma: no cover — typing only
    from .config import ProxyConfig
    from .http_handlers import ProxyHttpHandlers
    from .identity import KeyRegistry
    from .state import ProxyState

logger = logging.getLogger(__name__)

#: The management plane's own prefix. ``/rs/v1`` is Roadstead's north face and is
#: versioned by us; see the module docstring for why admin does not stay on the
#: OpenAI-versioned one.
PREFIX = "/rs/v1/admin"

#: Prefix on a generated secret, so a leaked string is identifiable as a
#: Roadstead credential in a log or a paste and can be revoked by shape.
_SECRET_PREFIX = "rs-"

#: Bytes of entropy behind a generated key. 32 bytes ≈ 43 urlsafe characters.
_SECRET_BYTES = 32

#: Quota fields a ``PATCH /rs/v1/admin/callers/{agent_id}`` may set — the write
#: half of ``config._AGENT_CONFIG_FIELDS``, and pinned against it, because a knob
#: an operator can write in the file and not through the API (or the reverse) is
#: the same class of surprise as one nothing reads at all.
#: The wire names are the ``agents.yaml`` names and the ``AgentQuotaConfig``
#: attribute names, identically — so an operator can copy a line between a file
#: and a PATCH body without translating it, and so no mapping table exists to
#: drift.
EDITABLE_QUOTA_FIELDS = (
    "weight", "max_balance_ss", "default_priority",
    "degrade_ok", "spill_ok", "daily_spend_usd",
)

#: Fields a ``POST /rs/v1/admin/keys`` accepts.
_KEY_CREATE_FIELDS = frozenset({
    "agent_id", "priority", "min_timeout_s", "admin", "id", "key_sha256",
})


# ---------------------------------------------------------------------------
# The overlay — what the API changed, layered over what the operator wrote
# ---------------------------------------------------------------------------

class AdminOverlay:
    """Runtime changes, persisted beside the config rather than into it.

    One JSON file with three sections:

    ``keys``     runtime-enrolled credentials, as digests (never secrets).
    ``revoked``  key ids tombstoned at runtime — including ones declared in the
                 environment or in a keys file, which this layer cannot edit but
                 must still be able to switch off. See :meth:`apply`.
    ``agents``   per-caller quota overrides, field by field.

    ``path=None`` keeps everything in memory: the changes apply, nothing
    survives a restart, and every response says so.
    """

    #: Top-level keys the file may carry. An unknown one is REPORTED rather than
    #: dropped, for the reason this whole module exists.
    _FILE_SECTIONS = frozenset({"version", "keys", "revoked", "agents"})

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self.keys: list[dict] = []
        self.revoked: list[str] = []
        self.agents: dict[str, dict] = {}
        #: Per-agent values the CONFIG FILE set, captured in :meth:`apply`
        #: before any override lands. What makes "declared vs in force"
        #: reportable after the two have been merged into one object.
        self.declared_agents: dict[str, dict] = {}
        self._load()

    # ---- persistence ----------------------------------------------------

    @property
    def path(self) -> str:
        return str(self._path) if self._path else ""

    @property
    def writable(self) -> bool:
        """Whether a change made now would survive a restart."""
        return self.unwritable_reason() is None

    def unwritable_reason(self) -> str | None:
        """Why persistence would fail, in words an operator can act on."""
        if self._path is None:
            return ("no admin store is configured (ROADSTEAD_ADMIN_STORE); "
                    "changes apply immediately and are lost on restart")
        parent = self._path.parent
        if not parent.exists():
            return f"{parent} does not exist"
        if not os.access(parent, os.W_OK):
            return f"{parent} is not writable"
        if self._path.exists() and not os.access(self._path, os.W_OK):
            return f"{self._path} is not writable"
        return None

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
        except Exception as exc:  # noqa: BLE001 — a corrupt store must not block startup
            logger.error("admin store load failed at %s (%s); starting with an "
                         "empty overlay — the CONFIG FILES are unaffected",
                         self._path, exc)
            return
        if not isinstance(raw, dict):
            logger.error("admin store %s is not a JSON object; ignoring",
                         self._path)
            return
        unknown = sorted(set(raw) - self._FILE_SECTIONS)
        if unknown:
            hooks.config_notice(
                source=str(self._path),
                subject="(top level)",
                problem="unknown_key",
                detail=(f"admin store section(s) {unknown} are not read and have "
                        f"NO effect"),
                keys=unknown, known=sorted(self._FILE_SECTIONS),
            )
        self.keys = [k for k in raw.get("keys", []) if isinstance(k, dict)]
        self.revoked = [str(k) for k in raw.get("revoked", [])]
        agents = raw.get("agents", {})
        if isinstance(agents, dict):
            self.agents = {str(k): dict(v) for k, v in agents.items()
                           if isinstance(v, dict)}
        logger.info("admin overlay: %d runtime key(s), %d revocation(s), "
                    "%d caller override(s) from %s",
                    len(self.keys), len(self.revoked), len(self.agents),
                    self._path)

    def persist(self) -> None:
        """Rewrite the store atomically. Blocking — call via ``to_thread``.

        Never raises: an unwritable disk degrades to in-memory, which the
        handlers report as ``persisted: false`` rather than as a failure.
        """
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(
                {
                    "version": 1,
                    "keys": self.keys,
                    "revoked": self.revoked,
                    "agents": self.agents,
                },
                indent=1, sort_keys=True,
            ))
            os.replace(tmp, self._path)
        except Exception as exc:  # noqa: BLE001
            logger.error("admin store persist failed at %s: %s", self._path, exc)

    # ---- application ----------------------------------------------------

    def apply(self, registry: "KeyRegistry", config: "ProxyConfig") -> None:
        """Layer the overlay over a freshly-loaded registry and config.

        Order matters and is doctrine: **enrol, then revoke**. A key id present
        in both sections is revoked — a tombstone is a later statement than the
        enrolment it follows, and the opposite order would make a revocation
        silently undone by the record of the key it revoked.
        """
        for entry in self.keys:
            registry.register(
                agent_id=str(entry.get("agent_id") or ""),
                key_sha256=str(entry.get("key_sha256") or "") or None,
                priority=_coerce_priority(entry.get("priority")),
                min_timeout_s=(None if entry.get("min_timeout_s") is None
                               else float(entry["min_timeout_s"])),
                admin=bool(entry.get("admin", False)),
                key_id=str(entry["id"]) if entry.get("id") else None,
                source="runtime",
            )
        for key_id in self.revoked:
            registry.revoke(key_id)

        # Capture what the FILES said before overriding, so the read plane can
        # show both. A field equal to its dataclass default is reported as
        # undeclared: `load_agent_configs` only sets what was written, so the
        # two coincide except for someone writing a value that happens to equal
        # the default — harmless, and not worth a second parse of the YAML.
        self.declared_agents = {
            agent_id: _non_default_quota(cfg)
            for agent_id, cfg in config.agents.items()
        }
        for agent_id, fields in self.agents.items():
            cfg = config.agent_config(agent_id)
            for name, value in fields.items():
                if name in EDITABLE_QUOTA_FIELDS:
                    setattr(cfg, name, _coerce_quota(name, value))

    # ---- mutation -------------------------------------------------------

    def add_key(self, record: dict) -> None:
        self.keys.append(record)
        # An id being re-enrolled is no longer revoked. Without this, a key id
        # reused after a revocation would be tombstoned on the next restart and
        # work fine until then — a credential that stops working at a moment
        # unrelated to anything anyone did.
        self.revoked = [k for k in self.revoked if k != record.get("id")]

    def revoke_key(self, key_id: str) -> None:
        self.keys = [k for k in self.keys if k.get("id") != key_id]
        if key_id not in self.revoked:
            self.revoked.append(key_id)

    def set_agent(self, agent_id: str, fields: dict) -> None:
        self.agents.setdefault(agent_id, {}).update(fields)


def _coerce_priority(raw: Any) -> LLMPriority:
    if raw is None:
        return LLMPriority.P3_INGESTION
    try:
        return LLMPriority.coerce(raw)
    except ValueError:
        return LLMPriority.P3_INGESTION


def _coerce_quota(name: str, value: Any) -> Any:
    if name == "default_priority":
        return LLMPriority.coerce(value)
    if name in ("degrade_ok", "spill_ok"):
        return bool(value)
    if name == "daily_spend_usd":
        return None if value is None else float(value)
    return float(value)


def _non_default_quota(cfg: Any) -> dict:
    """Every quota field on ``cfg`` that differs from the dataclass default."""
    defaults = AgentQuotaConfig(agent_id=cfg.agent_id)
    out: dict[str, Any] = {}
    for name in EDITABLE_QUOTA_FIELDS:
        value = getattr(cfg, name)
        if value != getattr(defaults, name):
            out[name] = value.name if isinstance(value, LLMPriority) else value
    return out


# ---------------------------------------------------------------------------
# Validation — the write boundary
# ---------------------------------------------------------------------------

class Invalid(ValueError):
    """A rejected write, with the message the operator should read."""


def validate_quota_patch(body: Any) -> dict:
    """Coerce and check a caller-quota PATCH body. Raises :class:`Invalid`.

    🚨 An unknown field is REJECTED and the known set is named. Three loaders in
    this package drop unknown keys and each has cost something; this surface
    exists to expose that failure, so repeating it here would be a joke at the
    operator's expense.
    """
    if not isinstance(body, dict) or not body:
        raise Invalid("expected a non-empty JSON object of quota fields")
    unknown = sorted(set(body) - set(EDITABLE_QUOTA_FIELDS))
    if unknown:
        raise Invalid(f"unknown quota field(s) {unknown}; editable fields are "
                      f"{list(EDITABLE_QUOTA_FIELDS)}")
    out: dict[str, Any] = {}
    for name, value in body.items():
        if name == "default_priority":
            try:
                out[name] = LLMPriority.coerce(value).name
            except (ValueError, TypeError) as exc:
                raise Invalid(f"default_priority: {exc}") from exc
        elif name in ("degrade_ok", "spill_ok"):
            if not isinstance(value, bool):
                raise Invalid(f"{name} must be a JSON boolean")
            out[name] = value
        elif name == "daily_spend_usd":
            # 🚨 `null` is not `0`. None means uncapped; 0.0 is a real cap
            # meaning "no paid spend at all" (see AgentQuotaConfig). They are one
            # keystroke apart and mean opposite things, so the JSON null survives
            # rather than being coerced through float().
            if value is None:
                out[name] = None
            else:
                out[name] = _positive_number(name, value, allow_zero=True)
        else:
            out[name] = _positive_number(name, value, allow_zero=False)
    return out


def _positive_number(name: str, value: Any, *, allow_zero: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Invalid(f"{name} must be a number")
    number = float(value)
    if number < 0 or (number == 0 and not allow_zero):
        raise Invalid(f"{name} must be {'>= 0' if allow_zero else '> 0'}")
    return number


def validate_key_create(body: Any) -> dict:
    """Coerce and check a key-enrolment body. Raises :class:`Invalid`.

    🚨 There is no ``key`` field: a plaintext secret is never ACCEPTED over this
    door, only ever returned once by it. Sending one would put a live credential
    in an access log, a proxy buffer and a shell history, which is the failure
    ``key_sha256:`` exists in the keys file to avoid. An operator migrating an
    existing key sends its digest.
    """
    if not isinstance(body, dict):
        raise Invalid("expected a JSON object")
    unknown = sorted(set(body) - _KEY_CREATE_FIELDS)
    if unknown:
        raise Invalid(f"unknown field(s) {unknown}; accepted fields are "
                      f"{sorted(_KEY_CREATE_FIELDS)}")
    agent_id = str(body.get("agent_id") or "").strip()
    if not agent_id:
        raise Invalid("agent_id is required — it is the fair-share key, the "
                      "quota holder and the budget holder")
    out: dict[str, Any] = {"agent_id": agent_id}
    if body.get("priority") is not None:
        try:
            out["priority"] = LLMPriority.coerce(body["priority"]).name
        except (ValueError, TypeError) as exc:
            raise Invalid(f"priority: {exc}") from exc
    if body.get("min_timeout_s") is not None:
        out["min_timeout_s"] = _positive_number(
            "min_timeout_s", body["min_timeout_s"], allow_zero=False)
    if "admin" in body:
        if not isinstance(body["admin"], bool):
            raise Invalid("admin must be a JSON boolean")
        out["admin"] = body["admin"]
    if body.get("id") is not None:
        key_id = str(body["id"]).strip()
        if not key_id:
            raise Invalid("id must be a non-empty string when given")
        out["id"] = key_id
    if body.get("key_sha256") is not None:
        digest = str(body["key_sha256"]).strip().lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise Invalid("key_sha256 must be 64 hex characters")
        out["key_sha256"] = digest
    return out


# ---------------------------------------------------------------------------
# The API
# ---------------------------------------------------------------------------

class ManagementApi:
    """Handlers for ``/rs/v1/admin/*``.

    Holds the admin gate by reference rather than reimplementing it: the gate is
    three lines and the repo already carries a note about a predicate written out
    three times.
    """

    def __init__(self, state: "ProxyState", http: "ProxyHttpHandlers") -> None:
        self.state = state
        self._http = http
        # Serialises overlay writes. An asyncio.Lock, NOT a threading one: it
        # orders two coroutines' read-modify-write of the same file on one loop,
        # and holds across the `to_thread` that does the I/O. It guards no
        # in-memory scheduler state, which is what CLAUDE.md forbids locking.
        self._store_lock = asyncio.Lock()

    # ---- gate -----------------------------------------------------------

    def _gate(self, route: str, request: Request) -> Response | None:
        remote_ip = _remote_ip(request)
        self._http.audit_admin_ip(route, remote_ip)
        return self._http.deny_non_admin(request, remote_ip)

    async def _persist(self) -> dict:
        """Persist the overlay off-loop and describe the outcome.

        🚨 The caller has ALREADY applied the change in memory. This only decides
        whether it survives a restart — see the module docstring on why an
        unpersistable change is still a change.
        """
        overlay = self.state.admin_overlay
        reason = overlay.unwritable_reason()
        if reason is not None:
            return {"persisted": False, "reason": reason}
        await asyncio.to_thread(overlay.persist)
        # Re-check: `persist` swallows its own errors so a bad disk cannot break
        # a response, which means "it ran" is not "it worked".
        return {"persisted": overlay.unwritable_reason() is None,
                "store": overlay.path}

    # ---- config: what you wrote vs what is in force ----------------------

    async def handle_admin_config(self, request: Request) -> Response:
        """GET — the configuration sources, and everything written that is not
        in force.

        The single most useful page here, and the reason the notice seam exists:
        a startup WARNING is read by the operator who booted the process, and the
        question "why does this knob do nothing" is asked by a different one, a
        week later.
        """
        denied = self._gate(f"{PREFIX}/config", request)
        if denied is not None:
            return denied
        overlay = self.state.admin_overlay
        return JSONResponse({
            "sources": {
                "catalog": {
                    "path": os.environ.get("ROADSTEAD_MODELS_YAML", "")
                            or str(model_catalog._DEFAULT_PATH),
                    "env_var": "ROADSTEAD_MODELS_YAML",
                },
                "agents": {
                    "path": os.environ.get("LLM_PROXY_AGENTS_CONFIG", ""),
                    "env_var": "LLM_PROXY_AGENTS_CONFIG",
                },
                "api_keys": {
                    "env_var": "ROADSTEAD_API_KEYS",
                    "declared_in_env": bool(os.environ.get("ROADSTEAD_API_KEYS")),
                    "file": os.environ.get("ROADSTEAD_API_KEYS_FILE", ""),
                },
                "acl": {"env_var": "ROADSTEAD_ACL"},
                "runtime_flags": {"path": self.state.config.runtime_flags_path},
                "admin_store": {
                    "path": overlay.path,
                    "writable": overlay.writable,
                    "reason": overlay.unwritable_reason(),
                    "runtime_keys": len(overlay.keys),
                    "revocations": len(overlay.revoked),
                    "caller_overrides": len(overlay.agents),
                },
            },
            # 🚨 The gap list. Empty is the healthy state and is NOT the same as
            # "no config was loaded" — the sources block above says which files
            # were read.
            "notices": hooks.config_notices(),
        })

    # ---- keys -----------------------------------------------------------

    async def handle_admin_keys(self, request: Request) -> Response:
        """GET the key registry (redacted); POST to enrol a new credential."""
        denied = self._gate(f"{PREFIX}/keys", request)
        if denied is not None:
            return denied
        if request.method == "GET":
            return JSONResponse(self._keys_view())
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return _error("invalid_request_error", "body must be JSON", 400)
        try:
            spec = validate_key_create(body)
        except Invalid as exc:
            return _error("invalid_request_error", str(exc), 400)

        digest = spec.get("key_sha256")
        secret: str | None = None
        if digest is None:
            secret = _SECRET_PREFIX + secrets.token_urlsafe(_SECRET_BYTES)
        registry = self.state.identity.keys
        was_configured = registry.configured
        key_id = registry.register(
            agent_id=spec["agent_id"],
            secret=secret,
            key_sha256=digest,
            priority=_coerce_priority(spec.get("priority")),
            min_timeout_s=spec.get("min_timeout_s"),
            admin=bool(spec.get("admin", False)),
            key_id=spec.get("id"),
            source="runtime",
        )
        if key_id is None:
            # `register` rejects a duplicate digest and an unusable one, and says
            # why in the log. The commonest by far is re-POSTing a digest that is
            # already enrolled.
            return _error(
                "invalid_request_error",
                "the key was not registered — the digest is already enrolled or "
                "is unusable; GET this route to see what is registered", 409)

        record = {
            "id": key_id,
            "agent_id": spec["agent_id"],
            "key_sha256": digest or hashlib.sha256(
                secret.encode("utf-8")).hexdigest(),
            "priority": spec.get("priority", LLMPriority.P3_INGESTION.name),
            "min_timeout_s": spec.get("min_timeout_s"),
            "admin": bool(spec.get("admin", False)),
            "created_at": time.time(),
        }
        self.state.admin_overlay.add_key(record)
        async with self._store_lock:
            outcome = await self._persist()

        payload = {
            "key_id": key_id,
            "agent_id": spec["agent_id"],
            "priority": record["priority"],
            "min_timeout_s": record["min_timeout_s"],
            "admin": record["admin"],
            **outcome,
        }
        if secret is not None:
            # 🚨 The ONLY time this string exists outside the caller's own
            # storage. The registry keeps the digest, so it cannot be reissued,
            # re-read or recovered — losing it means revoking and enrolling
            # again, which is the correct cost.
            payload["key"] = secret
            payload["note"] = ("this secret is shown once and cannot be "
                               "recovered — store it now")
        # 🚨 A LIST, not a string. Two disclosures can be true of one action —
        # revoking an env-declared key that also happens to be the last one is
        # both "the environment will outlive this" and "the registry is empty
        # now" — and a single field means the second one silently overwrites the
        # first. That is the same shape as a config key dropped in silence.
        warnings: list[str] = []
        if not was_configured:
            # 🚨 The first key changes the identity regime for EVERY caller
            # (docs/api.md §1.5 rule 2): until now a presented key was ignored
            # and the address decided, and from now on a presented key that does
            # not resolve is a 401. An operator enrolling their first credential
            # is about to be surprised by that on some other machine.
            warnings.append(
                "this is the FIRST key: the registry is now in play, so any "
                "caller presenting an unrecognised key gets a 401 instead of "
                "being identified by its address (docs/api.md §1.5)")
        if warnings:
            payload["warnings"] = warnings
        return JSONResponse(payload, status_code=201)

    async def handle_admin_key(self, request: Request) -> Response:
        """DELETE — revoke one credential by its key id."""
        denied = self._gate(f"{PREFIX}/keys", request)
        if denied is not None:
            return denied
        key_id = request.path_params["key_id"]
        registry = self.state.identity.keys
        source = next(
            (row["source"] for row in registry.snapshot()
             if row["key_id"] == key_id),
            None,
        )
        if not registry.revoke(key_id):
            return _error("invalid_request_error",
                          f"no key with id {key_id!r} is registered", 404)
        self.state.admin_overlay.revoke_key(key_id)
        async with self._store_lock:
            outcome = await self._persist()
        payload = {"revoked": key_id, "source": source, **outcome}
        warnings: list[str] = []
        if not registry.configured:
            # 🚨 The mirror of the first-enrolment warning, and the more
            # surprising half. Revoking the LAST key empties the registry, and
            # §1.5 rule 2 then says the registry is not in play at all: a
            # presented key — including the one just revoked — is IGNORED again
            # and the source address decides. The credential confers nothing
            # either way, so nothing is escalated; what stops being true is
            # "revoked means refused", because a caller whose address is
            # enrolled keeps working under its address identity.
            #
            # Rule 2 is not narrowed to fix this. It exists so that an operator
            # who has configured no keys is not broken by the placeholder
            # Authorization header every OpenAI client sends, and making the
            # registry sticky once populated would 401 exactly the deployment
            # that has just emptied it on purpose. So the regime change is
            # DISCLOSED, the same way its opposite is.
            warnings.append(
                "that was the LAST key: the registry is no longer in play, so a "
                "presented key is ignored again and callers are identified by "
                "source address (docs/api.md §1.5 rule 2) — a caller whose "
                "address is enrolled keeps working without a credential")
        if source == "env":
            # 🚨 The tombstone survives a restart; the ENV DECLARATION does too,
            # and this layer cannot edit an environment. The revocation still
            # holds — enrol-then-revoke is the documented apply order — but an
            # operator who does not also remove the variable is left with a
            # credential that is dead for a reason nothing in their config says.
            warnings.append(
                f"key {key_id!r} is also declared in ROADSTEAD_API_KEYS; the "
                "revocation is recorded and survives a restart, but remove it "
                "from the environment too or the declaration will outlive the "
                "reason it is dead")
        if warnings:
            payload["warnings"] = warnings
        return JSONResponse(payload)

    def _keys_view(self) -> dict:
        registry = self.state.identity.keys
        rows = registry.snapshot()
        by_agent: dict[str, dict] = {}
        for row in rows:
            bucket = by_agent.setdefault(
                str(row["agent_id"]), {"keys": [], "admin": False})
            bucket["keys"].append(row["key_id"])
            bucket["admin"] = bucket["admin"] or bool(row["admin"])
        overlay = self.state.admin_overlay
        return {
            # 🚨 §1.5 rule 2: with nothing configured the registry is NOT in
            # play and a presented key is ignored entirely. An operator debugging
            # "my key does nothing" needs that stated, not inferred from a count.
            "configured": registry.configured,
            "require_key": self.state.identity.require_key,
            "keys": rows,
            # Many keys → one agent_id is how a team shares a quota: the budget
            # holder is the agent_id, not the key. See the module docstring.
            "by_agent": by_agent,
            "revoked": list(overlay.revoked),
            "store": {"path": overlay.path,
                      "writable": overlay.writable,
                      "reason": overlay.unwritable_reason()},
        }

    # ---- callers --------------------------------------------------------

    async def handle_admin_callers(self, request: Request) -> Response:
        """GET — every caller the proxy knows: identity, quota, DRR, spend."""
        denied = self._gate(f"{PREFIX}/callers", request)
        if denied is not None:
            return denied
        return JSONResponse({"callers": self._callers_view()})

    async def handle_admin_caller(self, request: Request) -> Response:
        """PATCH — edit one caller's quota. Partial: absent fields are untouched."""
        denied = self._gate(f"{PREFIX}/callers", request)
        if denied is not None:
            return denied
        agent_id = request.path_params["agent_id"]
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return _error("invalid_request_error", "body must be JSON", 400)
        try:
            fields = validate_quota_patch(body)
        except Invalid as exc:
            return _error("invalid_request_error", str(exc), 400)

        # ON THE LOOP. `config.agent_config` lazily creates, which is right: an
        # operator may set a quota for a caller that has not called yet.
        cfg = self.state.config.agent_config(agent_id)
        for name, value in fields.items():
            setattr(cfg, name, _coerce_quota(name, value))
        # And into the LIVE budget, or the edit would apply only to callers the
        # proxy has never seen — which reads as "the edit did nothing" for
        # exactly the busy caller it was aimed at.
        if "weight" in fields or "max_balance_ss" in fields:
            self.state.budget_mgr.reweight(
                agent_id,
                weight=fields.get("weight"),
                max_balance=fields.get("max_balance_ss"),
            )
        self.state.admin_overlay.set_agent(agent_id, fields)
        async with self._store_lock:
            outcome = await self._persist()
        return JSONResponse({
            "agent_id": agent_id,
            "changed": fields,
            "quota": self._quota_view(agent_id),
            **outcome,
        })

    def _known_agent_ids(self) -> list[str]:
        """Every caller worth reporting: configured, budgeted, or spending.

        Union rather than any single source — a caller with a quota stanza and
        no traffic must appear (an operator wants to see the policy took), and so
        must one that has called with no stanza at all (which is most of them).
        """
        ids: set[str] = set(self.state.config.agents)
        ids.update(self.state.budget_mgr.agents)
        ids.update(row["agent_id"] for row in self.state.spend.snapshot())
        ids.update(row["agent_id"] for row in self.state.identity.keys.snapshot())
        return sorted(ids)

    def _quota_view(self, agent_id: str) -> dict:
        cfg = self.state.config.agent_config(agent_id)
        overlay = self.state.admin_overlay
        in_force = {
            name: (getattr(cfg, name).name
                   if isinstance(getattr(cfg, name), LLMPriority)
                   else getattr(cfg, name))
            for name in EDITABLE_QUOTA_FIELDS
        }
        return {
            "in_force": in_force,
            # What the FILE said (only fields it actually set), and what this API
            # changed. Which block a value appears in is where it came from —
            # cheaper and less lossy than a per-field `source` string, and it
            # makes an override visible as a difference rather than a label.
            "declared": overlay.declared_agents.get(agent_id, {}),
            "runtime": overlay.agents.get(agent_id, {}),
        }

    def _callers_view(self) -> list[dict]:
        budgets = {row["agent_id"]: row for row in self.state.budget_mgr.snapshot()}
        spends = {row["agent_id"]: row for row in self.state.spend.snapshot()}
        keys_by_agent: dict[str, list[str]] = {}
        for row in self.state.identity.keys.snapshot():
            keys_by_agent.setdefault(str(row["agent_id"]), []).append(
                str(row["key_id"]))
        addresses = _acl_addresses(self.state.acl)

        out: list[dict] = []
        for agent_id in self._known_agent_ids():
            standing = self.state.spend_standing(agent_id)
            cfg = self.state.config.agent_config(agent_id)
            spend_row = spends.get(agent_id, {})
            out.append({
                "agent_id": agent_id,
                "identities": {
                    "keys": keys_by_agent.get(agent_id, []),
                    "addresses": addresses.get(agent_id, []),
                },
                "quota": self._quota_view(agent_id),
                "drr": budgets.get(agent_id),
                # 🚨 Two kinds of money, never summed (spend.py, §1.6): the first
                # is an invoice and the second is a saving. They are reported as
                # separate fields here for the same reason they are stored as
                # separate fields there — a total would be neither number, on the
                # surface an operator uses to decide whether a caller costs too
                # much.
                "spend": {
                    "spent_usd": spend_row.get("spent_usd", 0.0),
                    "avoided_usd": spend_row.get("avoided_usd", 0.0),
                    # The number the THRESHOLD reads. Lifetime spend above is
                    # what an operator budgets on; this is what degrades a
                    # caller today, and reporting only one of them is how the
                    # two come to be confused for each other.
                    "spent_today_usd": standing.spent_today_usd,
                    "cap_usd": standing.cap_usd,
                    "over": standing.over,
                    # What crossing the cap actually costs — one band and paid
                    # spill, never local capacity and never a refusal (§1.6).
                    "may_spill": standing.may_spill,
                    "declared_priority": cfg.default_priority.name,
                    "effective_priority":
                        standing.effective_priority(cfg.default_priority).name,
                },
                "live": self.state.scheduler.agent_snapshot(agent_id),
            })
        return out

    # ---- providers ------------------------------------------------------

    async def handle_admin_providers(self, request: Request) -> Response:
        """GET — providers and endpoints: declared, discovered, and in force."""
        denied = self._gate(f"{PREFIX}/providers", request)
        if denied is not None:
            return denied
        return JSONResponse(self._providers_view())

    def _providers_view(self) -> dict:
        cat = model_catalog.load_catalog()
        endpoints = [self._endpoint_view(name, cat)
                     for name in sorted(self.state.config.endpoints)]
        by_provider: dict[str, list[str]] = {}
        for view in endpoints:
            by_provider.setdefault(view["provider"], []).append(view["endpoint"])

        providers = []
        for name, entry in sorted(cat.providers.items()):
            descriptor = provider_for_engine(entry.engine).descriptor
            providers.append({
                "provider": name,
                "engine": entry.engine,
                # The address as configured, both forms — a local provider has
                # host/port, a remote one a base URL, and which is populated is
                # itself the answer to "is this thing local".
                "address": {
                    "host": cat.hosts.get(entry.host, entry.host),
                    "port": entry.port,
                    "base_url": entry.base_url,
                },
                # 🚨 The variable's NAME and whether it resolved. Never the
                # value: this is a read-only surface, and a surface that prints
                # a credential is a key store.
                "credential": {
                    "env_var": entry.api_key_env,
                    "present": bool(os.environ.get(entry.api_key_env))
                    if entry.api_key_env else None,
                },
                # What this KIND of backend can tell us — the declared asymmetry
                # (CLAUDE.md), so an operator reading "slots: config-seeded" can
                # see it is a property of the engine rather than a missing probe.
                "descriptor": dataclasses.asdict(descriptor),
                "endpoints": by_provider.get(name, []),
            })
        return {"providers": providers, "endpoints": endpoints}

    def _endpoint_view(self, name: str, cat: Any) -> dict:
        ep = self.state.config.endpoints[name]
        entry = cat.entry(name)
        descriptor = provider_for_engine(ep.backend_engine).descriptor
        health = self.state.endpoint_health.get(name, {})
        paused = name in self.state.paused_endpoints
        price = self.state.prices.price(name)
        return {
            "endpoint": name,
            "provider": entry.provider if entry else "",
            "kind": ep.kind,
            "status": entry.status if entry else "active",
            "role": ep.role,
            "aliases": list(entry.aliases) if entry else [],
            "capabilities": sorted(ep.capabilities),
            # 🚨 The gap, stated three times because it is three different
            # questions. `declared` is the catalog seed, `in_force` is what
            # admission actually uses, and `discovered` says whether the backend
            # was ever asked — which for a vLLM endpoint is permanently False and
            # is a property of the engine, not a fault (CLAUDE.md).
            "capacity": {
                "slots": {
                    "declared": entry.slots if entry else 0,
                    "in_force": ep.max_slots,
                    "discoverable": descriptor.publishes_slot_count,
                },
                "context_per_slot": {
                    "declared": entry.context_per_slot if entry else 0,
                    "in_force": ep.context_per_slot,
                    "discoverable": (descriptor.publishes_slot_context
                                     or descriptor.publishes_context_ceiling),
                },
            },
            "model": {
                "declared_fingerprint": ep.model_fingerprint or None,
                "serving_fingerprint": ep.discovered_model_fingerprint or None,
                "served_model_id": ep.served_model_id or None,
                "pinned_model": entry.model if entry else "",
            },
            # 🚨 `real` is what separates an invoice from an avoided cost, and it
            # is the reason a threshold reads only one of the two columns. Branch
            # on it, never on whether the endpoint looks remote (§1.6).
            "price": {**price.as_dict(),
                      "real": price.real,
                      "operator_declared": declared_price(name, ep) is not None},
            "health": {
                "healthy": bool(health.get("healthy", True)) and not paused,
                "admin_paused": paused,
                "probe_healthy": bool(health.get("healthy", True)),
            },
            "routing": {"failover_to": ep.failover_to or None,
                        "spill_to": ep.spill_to or None},
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _remote_ip(request: Request) -> str:
    client = getattr(request, "client", None)
    return getattr(client, "host", "") or ""


def _acl_addresses(acl: Any) -> dict[str, list[str]]:
    """agent_id → the addresses an operator registered for it.

    Only registrations: ``IPIdentityMap.entries`` deliberately omits the
    built-in internal nets, so a caller identified as ``internal`` shows no
    addresses here rather than a list that implies they are removable.
    """
    out: dict[str, list[str]] = {}
    for address, reg in acl.entries().items():
        out.setdefault(str(reg["agent_id"]), []).append(str(address))
    return {k: sorted(v) for k, v in out.items()}
