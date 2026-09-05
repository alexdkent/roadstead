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
(``docs/internals.md``). So **the mutation happens on the loop and only the file write
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
import ipaddress
import json
import logging
import os
import re
import secrets
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from . import hooks, model_catalog
from .config import AgentQuotaConfig, EndpointConfig, LLMPriority, env_with_legacy_prefix
from .enriched import _error
from .identity import iso_time
from .providers import known_engines, provider_for_engine
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

# ---------------------------------------------------------------------------
# The operator UI (roadmap Workstream G)
# ---------------------------------------------------------------------------
#
# 🚨 ONE static HTML file, vanilla JS, no bundler and no third-party anything.
# The dependency list is six packages on purpose (docs/internals.md); a build step, a
# node_modules or a React dependency is out, and the same judgement that keeps
# ``roadstead.client`` on httpx-and-stdlib keeps this on the platform.
#
# It ships in the wheel, so it is PUBLIC SURFACE — the same argument as
# ``roadstead.testing``. A new external reference in it is a new one for
# everybody who installs Roadstead, which is why the CSP below forbids one
# outright rather than trusting a reviewer to notice.

_UI_ENV = "ROADSTEAD_ADMIN_UI"
_UI_FILE = Path(__file__).resolve().parent / "ui" / "index.html"

#: 🚨 ``default-src 'none'`` with NO host allowed anywhere. The page may talk to
#: its own origin and load nothing at all from outside it, which is what makes
#: "the key is readable by script on the page" an acceptable trade: the only
#: script on the page is the one in the file. ``'unsafe-inline'`` is the price of
#: having no bundler — there is no build step to emit a hash or a nonce, and the
#: alternative is a toolchain. ``frame-ancestors 'none'`` closes clickjacking on
#: a page whose buttons pause backends.
_UI_CSP = (
    "default-src 'none'; "
    "script-src 'unsafe-inline'; "
    "style-src 'unsafe-inline'; "
    "connect-src 'self'; "
    "img-src data:; "
    "form-action 'none'; "
    "base-uri 'none'; "
    "frame-ancestors 'none'"
)

#: The realm a browser shows in its password box, and the key a browser caches
#: the credential under. Stable: changing it logs every operator out.
_UI_REALM = "Roadstead"


def admin_ui_enabled() -> bool:
    """Whether ``GET /rs/v1/admin/ui`` is registered at all.

    🚨 Default OFF, and OFF means the route does not exist rather than that it
    refuses — the same posture as ``ROADSTEAD_TRUSTED_PROXIES`` and
    ``ROADSTEAD_REQUIRE_API_KEY``, for the same reason: a capability that widens
    what is reachable is an operator's decision, and an HTML door on a proxy is
    reachable by things that would never send an API request on purpose.
    """
    return os.environ.get(_UI_ENV, "").strip().lower() in ("1", "true", "yes", "on")

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
    "degrade_ok", "spill_ok", "daily_spend_usd", "requests_per_minute",
)

#: Fields a ``POST /rs/v1/admin/keys`` accepts.
_KEY_CREATE_FIELDS = frozenset({
    "agent_id", "priority", "min_timeout_s", "admin", "admin_readonly",
    "expires_in_s", "bind", "may_assert", "id", "key_sha256",
})

#: Fields a ``POST /rs/v1/admin/keys/{key_id}/rotate`` accepts.
_KEY_ROTATE_FIELDS = frozenset({"id", "key_sha256", "overlap_s", "expires_in_s"})


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
``catalog``  runtime catalog stanzas — providers and endpoints created,
                 edited or deleted through the API, layered over
                 ``models.yaml`` (roadmap J2). 🚨 The stanzas persist and a
                 CREDENTIAL never does: a routing decision should be findable
                 after a restart, whereas an outbound provider key written into
                 a JSON file is a change of posture this store has never made
                 (it holds key *digests*, not secrets).
    ``audit``    who changed what, and when. See :meth:`record`.

    ``path=None`` keeps everything in memory: the changes apply, nothing
    survives a restart, and every response says so.
    """

    #: Top-level keys the file may carry. An unknown one is REPORTED rather than
    #: dropped, for the reason this whole module exists.
    _FILE_SECTIONS = frozenset({
        "version", "keys", "revoked", "retired", "agents", "catalog",
        "audit"})

    #: How many audit records to keep. Bounded because this is an append-only
    #: list on a long-lived process and the store is rewritten whole on every
    #: change — an unbounded trail turns each edit into a progressively larger
    #: synchronous file write. The oldest go first, and the read view says how
    #: many were dropped rather than presenting a truncated log as a complete
    #: one. 🚨 This is an operator-facing change trail, NOT a security audit log
    #: of record: it cannot outlive its own bound and `docs/api.md` §3.8 says so.
    _AUDIT_MAX = 500

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self.keys: list[dict] = []
        self.revoked: list[str] = []
        #: {key_id: expires_at} — retirements for keys this layer did not enrol.
        #: A softer tombstone: the key still works until the instant recorded.
        self.retired: dict[str, float] = {}
        self.agents: dict[str, dict] = {}
        #: The runtime catalog fragment: ``{"providers": {...}, "endpoints":
        #: {...}}``, each stanza a PARTIAL merged over the file's (or standing
        #: alone if the name is new), and ``None`` a tombstone. See
        #: ``model_catalog.merge_catalog_overlay`` — this is the file's own
        #: format, not a second one.
        self.catalog: dict[str, dict] = {"providers": {}, "endpoints": {}}
        #: Newest LAST, matching the file. Bounded by ``_AUDIT_MAX``.
        self.audit: list[dict] = []
        #: How many records fell off the front, in this process and in every
        #: one before it. Persisted, so the count survives the restart it
        #: describes.
        self.audit_dropped: int = 0
        #: Per-agent values the CONFIG FILE set, captured in :meth:`apply`
        #: before any override lands. What makes "declared vs in force"
        #: reportable after the two have been merged into one object.
        self.declared_agents: dict[str, dict] = {}
        # Serialises overlay writes. An asyncio.Lock, NOT a threading one: it
        # orders two coroutines' read-modify-write of the same file on one loop
        # and holds across the `to_thread` that does the I/O. It guards no
        # in-memory scheduler state, which is what docs/internals.md forbids locking.
        # 🚨 It lives HERE rather than on one handler because TWO modules now
        # persist this file — the management plane and the four control routes
        # that predate it — and a lock owned by one of them serialises only half
        # the writers.
        self._lock = asyncio.Lock()
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
        retired = raw.get("retired", {})
        if isinstance(retired, dict):
            self.retired = {}
            for k, v in retired.items():
                try:
                    self.retired[str(k)] = float(v)
                except (TypeError, ValueError):
                    logger.warning("admin store: retirement for %r has an "
                                   "unreadable expiry %r; ignoring", k, v)
        agents = raw.get("agents", {})
        if isinstance(agents, dict):
            self.agents = {str(k): dict(v) for k, v in agents.items()
                           if isinstance(v, dict)}
        # 🚨 Migrate the pre-J2 shape rather than dropping it. `endpoints:` was
        # the J1 status-override section and is exactly `catalog.endpoints` in
        # the new file — same stanzas, one level deeper. Left unmigrated it
        # would be reported as an unknown section (correct, and useless): an
        # operator would upgrade and silently find every promoted endpoint out
        # of service, which is the failure mode this store exists to prevent.
        legacy = raw.get("endpoints")
        if isinstance(legacy, dict) and legacy:
            raw.setdefault("catalog", {}).setdefault("endpoints", {}).update(legacy)
            logger.info("admin store: migrated %d endpoint override(s) from the "
                        "pre-J2 `endpoints:` section into `catalog.endpoints`",
                        len(legacy))
        catalog = raw.get("catalog", {})
        if isinstance(catalog, dict):
            for section in ("providers", "endpoints"):
                stanzas = catalog.get(section)
                if not isinstance(stanzas, dict):
                    continue
                # 🚨 `None` survives — it is a tombstone, not a malformed entry.
                self.catalog[section] = {
                    str(k): (None if v is None else dict(v))
                    for k, v in stanzas.items()
                    if v is None or isinstance(v, dict)}
        audit = raw.get("audit", {})
        if isinstance(audit, dict):
            self.audit = [e for e in audit.get("entries", []) if isinstance(e, dict)]
            try:
                self.audit_dropped = int(audit.get("dropped", 0))
            except (TypeError, ValueError):
                self.audit_dropped = 0
        logger.info("admin overlay: %d runtime key(s), %d revocation(s), "
                    "%d retirement(s), %d caller override(s), "
                    "%d provider + %d endpoint stanza(s) from %s",
                    len(self.keys), len(self.revoked), len(self.retired),
                    len(self.agents), len(self.catalog.get("providers") or {}),
                    len(self.catalog.get("endpoints") or {}), self._path)

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
                    "retired": self.retired,
                    "agents": self.agents,
                    "catalog": self.catalog,
                    "audit": {"entries": self.audit,
                              "dropped": self.audit_dropped},
                },
                indent=1, sort_keys=True,
            ))
            # 🚨 Owner-only. This file holds key DIGESTS, never a secret, but a
            # digest is still a working credential for anyone who can compute
            # one against it (§ the same reasoning `management.py`'s module
            # docstring gives for never emitting one over the wire) — the file
            # on disk deserves the same care. Set on the TEMP file, before the
            # rename, so the final path is never briefly world-readable.
            os.chmod(tmp, 0o600)
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
                admin_readonly=bool(entry.get("admin_readonly", False)),
                # 🚨 The ABSOLUTE instant, replayed as-is. A key enrolled with a
                # one-hour life and restarted after two hours must come back
                # expired, not with a fresh hour — which is what re-deriving it
                # from a stored duration would do, and it would make a restart a
                # way to extend a credential.
                expires_at=(None if entry.get("expires_at") is None
                            else float(entry["expires_at"])),
                bind=entry.get("bind") or [],
                may_assert=entry.get("may_assert") or [],
                key_id=str(entry["id"]) if entry.get("id") else None,
                source="runtime",
            )
        # 🚨 Retire BEFORE revoke, and both after enrol. A retirement is a
        # softer statement than a revocation, so a key id in both sections must
        # end up revoked — the same "a later statement wins" rule that puts
        # revoke after enrol, one step further down.
        for key_id, expires_at in self.retired.items():
            registry.set_expiry(key_id, expires_at)
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

        # 🚨 The catalog fragment LAST, and through the same function an admin
        # write uses. A fleet restored at startup must be byte-identical to the
        # one configured a minute ago, or what an operator sees after a restart
        # is not what they built.
        try:
            model_catalog.set_runtime_overlay(self.catalog)
            # Only what the fragment mentions. The rest of the routing table was
            # built from the catalog moments ago, or by whoever embedded us.
            touched = set(self.catalog.get("endpoints") or {})
            cat = model_catalog.load_catalog()
            for provider in (self.catalog.get("providers") or {}):
                touched |= {e.name for e in cat.endpoints.values()
                            if e.provider == provider}
            if touched:
                reconcile_endpoints(config, cat, names=touched, rebuild=touched)
        except Exception as exc:  # noqa: BLE001
            # A stale or unparseable fragment must not stop a boot — the same
            # rule as a mistyped `policy:` key. The file stands, and the
            # management plane reports the gap.
            logger.error("admin overlay: the runtime catalog fragment could not "
                         "be applied (%s); models.yaml stands alone", exc)
            model_catalog.set_runtime_overlay(None)

    # ---- mutation -------------------------------------------------------

    def set_catalog_stanza(self, section: str, name: str,
                           stanza: dict | None) -> None:
        """Record a runtime stanza whole. ``None`` is a tombstone (deleted)."""
        self.catalog.setdefault(section, {})[str(name)] = stanza

    def patch_catalog_stanza(self, section: str, name: str,
                             fields: dict) -> dict:
        """Merge fields into the runtime stanza, keeping what is already there.

        🚨 Merge, not replace: two edits to the same endpoint must compose. A
        `status` change that discarded an earlier `slots` override would make
        the second edit silently undo the first, which is the kind of loss an
        operator finds weeks later while reading a number they thought they set.
        A tombstone is REPLACED rather than merged into — resurrecting a deleted
        name by editing it would be a create wearing an edit's clothes.
        """
        section_map = self.catalog.setdefault(section, {})
        current = section_map.get(name)
        merged = dict(fields) if current is None else {**current, **fields}
        section_map[str(name)] = merged
        return merged

    def drop_catalog_stanza(self, section: str, name: str) -> None:
        """Forget a runtime stanza entirely — back to whatever the file says."""
        (self.catalog.get(section) or {}).pop(str(name), None)

    def add_key(self, record: dict) -> None:
        self.keys.append(record)
        self.retired.pop(str(record.get("id")), None)
        # An id being re-enrolled is no longer revoked. Without this, a key id
        # reused after a revocation would be tombstoned on the next restart and
        # work fine until then — a credential that stops working at a moment
        # unrelated to anything anyone did.
        self.revoked = [k for k in self.revoked if k != record.get("id")]

    def expire_key(self, key_id: str, expires_at: float) -> None:
        """Record a retirement DATE for a key, rather than a tombstone.

        🚨 The difference from :meth:`revoke_key` is the whole point of an
        overlap: a revoked key is gone at the next restart, while an expiring
        one must come back on restart AND come back expiring at the same
        instant. So a key already in the overlay is edited in place, and one
        that is not — an env- or file-declared predecessor this layer cannot
        edit — gets a `retired` entry carrying just the expiry, which
        :meth:`apply` replays over the declaration.
        """
        for entry in self.keys:
            if entry.get("id") == key_id:
                entry["expires_at"] = expires_at
                return
        self.retired[key_id] = expires_at

    def revoke_key(self, key_id: str) -> None:
        self.keys = [k for k in self.keys if k.get("id") != key_id]
        self.retired.pop(key_id, None)
        if key_id not in self.revoked:
            self.revoked.append(key_id)

    def set_agent(self, agent_id: str, fields: dict) -> None:
        self.agents.setdefault(agent_id, {}).update(fields)

    async def persist_async(self) -> dict:
        """Persist off-loop, serialised, and describe the outcome.

        🚨 The caller has ALREADY applied its change in memory. This only decides
        whether the change survives a restart — see the module docstring on why
        an unpersistable change is still a change.
        """
        async with self._lock:
            reason = self.unwritable_reason()
            if reason is not None:
                return {"persisted": False, "reason": reason}
            await asyncio.to_thread(self.persist)
            # Re-check: `persist` swallows its own errors so a bad disk cannot
            # break a response, which means "it ran" is not "it worked".
            return {"persisted": self.unwritable_reason() is None,
                    "store": self.path}

    def record(self, entry: dict) -> None:
        """Append one audit record. **Call on the loop.**

        🚨 In-memory only, deliberately, and this is the shape the concurrency
        invariant forces (docs/internals.md: *mutate on the loop, persist off it*). The
        record lands here synchronously — it is a list append — and reaches the
        disk on the ``persist()`` that the same handler was already going to do
        through ``asyncio.to_thread``. Writing the trail with its own file write
        would put I/O on the loop for every control action, and doing it off-loop
        separately would let a record land after the change it describes was
        already reported to the caller.

        A consequence worth naming: when the store is unwritable the trail
        applies and does not survive, exactly like the change it records. That
        is disclosed by the read view rather than fixed — a trail that refused
        to record an action the plane had already taken would make the log
        *less* truthful, not more.
        """
        self.audit.append(entry)
        if len(self.audit) > self._AUDIT_MAX:
            dropped = len(self.audit) - self._AUDIT_MAX
            del self.audit[:dropped]
            self.audit_dropped += dropped


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
    if name in ("daily_spend_usd", "requests_per_minute"):
        # Both are Optional for the same reason: null means "no threshold" and 0
        # is a real one. See AgentQuotaConfig.
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
        elif name in ("daily_spend_usd", "requests_per_minute"):
            # 🚨 `null` is not `0`. None means no threshold at all; 0.0 is a
            # real one — "no paid spend at all", "this caller should not be
            # sending" (see AgentQuotaConfig). They are one keystroke apart and
            # mean opposite things, so the JSON null survives rather than being
            # coerced through float().
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


def _binding(raw: Any) -> list[str]:
    """Coerce and CHECK a ``bind`` list. Raises :class:`Invalid`.

    🚨 Validated here rather than at use. A binding is a narrowing, and a
    narrowing that silently matches nothing because of a typo is a key that
    stops working from everywhere — while a binding that silently matched
    EVERYTHING would be worse. ``identity._address_in_any`` fails closed on an
    unparseable entry; this makes sure an operator never gets that far.
    """
    values = raw if isinstance(raw, list) else [raw]
    out: list[str] = []
    for entry in values:
        text = str(entry).strip()
        if not text:
            continue
        try:
            ipaddress.ip_network(text, strict=False)
        except ValueError as exc:
            raise Invalid(f"bind: {text!r} is not an address or CIDR ({exc})") from exc
        out.append(text)
    if not out:
        raise Invalid("bind must name at least one address or CIDR; omit it "
                      "for a key usable from anywhere")
    return out


def _may_assert(raw: Any, agent_id: str) -> list[str]:
    """Coerce ``may_assert`` — the ``agent_id``s this key may act as.

    🚨 This is the one editable field on this plane that WIDENS rather than
    narrows, and it is the exception §1.6's write boundary has to state rather
    than pretend away: every other quota knob moves a share, a band or a cap,
    and none of them can express a rejection. This one names identities a
    credential may bill. It is safe for the same reason `substitution` is safe:
    the operator grants, the caller can only ever spend inside the grant, and a
    name outside it is a 403 rather than a silent re-bill.

    A string is accepted as a one-element list (`may_assert: "chat-agent"` has one
    reading). A key listing its OWN agent_id is refused rather than trimmed —
    it reads as though the list is exhaustive, and an operator who believes that
    will later wonder why the key still works with the entry removed.
    """
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise Invalid("may_assert must be a list of agent_id strings")
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise Invalid(f"may_assert entries must be non-empty strings, "
                          f"got {item!r}")
        name = item.strip()
        if name == agent_id:
            raise Invalid(
                f"may_assert lists the key's own agent_id {name!r} — a key is "
                f"always itself, and listing it suggests the grant is needed")
        if name not in out:
            out.append(name)
    return out


def validate_key_rotate(body: Any) -> dict:
    """Coerce and check a rotation body. Raises :class:`Invalid`.

    🚨 ``overlap_s`` defaults to 0 — the predecessor is revoked NOW. The other
    default is tempting and wrong: a rotation that silently left the old
    credential alive is the one an operator believes they have completed, and
    the reason to rotate is usually that the old one should stop working. An
    overlap is a deliberate ask, and it is expressed as an EXPIRY on the
    predecessor rather than a timer, so it survives a restart.
    """
    if body is None:
        body = {}
    if not isinstance(body, dict):
        raise Invalid("expected a JSON object")
    unknown = sorted(set(body) - _KEY_ROTATE_FIELDS)
    if unknown:
        raise Invalid(f"unknown field(s) {unknown}; accepted fields are "
                      f"{sorted(_KEY_ROTATE_FIELDS)}")
    out: dict[str, Any] = {}
    if body.get("overlap_s") is not None:
        out["overlap_s"] = _positive_number(
            "overlap_s", body["overlap_s"], allow_zero=True)
    if body.get("expires_in_s") is not None:
        out["expires_in_s"] = _positive_number(
            "expires_in_s", body["expires_in_s"], allow_zero=False)
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
    if "admin_readonly" in body:
        if not isinstance(body["admin_readonly"], bool):
            raise Invalid("admin_readonly must be a JSON boolean")
        # 🚨 Refused rather than accepted-and-ignored. `admin_readonly` NARROWS
        # `admin`; on a key that has no admin scope it changes nothing, and an
        # operator who wrote it believes they have issued a safer credential
        # than they have. Same reason an unknown field is a 400 on this surface
        # rather than a silent drop.
        if body["admin_readonly"] and not out.get("admin", False):
            raise Invalid(
                "admin_readonly narrows the admin scope and grants nothing on "
                "its own — set admin: true beside it, or omit it")
        out["admin_readonly"] = body["admin_readonly"]
    if body.get("expires_in_s") is not None:
        # 🚨 A DURATION, not an instant. A caller sending an absolute time has
        # to agree with this process about the clock and the zone, and the
        # commonest way that goes wrong — a local-time string read as UTC —
        # produces a key that expires hours early or late with nothing to show
        # for it. The keys FILE takes an absolute date, because a file is
        # written once and read at every boot; an API call happens now.
        out["expires_in_s"] = _positive_number(
            "expires_in_s", body["expires_in_s"], allow_zero=False)
    if body.get("bind") is not None:
        out["bind"] = _binding(body["bind"])
    if body.get("may_assert") is not None:
        out["may_assert"] = _may_assert(body["may_assert"], out["agent_id"])
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
        #: The UI asset, read once off-loop and then held. None until first served.
        self._ui_html: str | None = None

    # ---- gate -----------------------------------------------------------

    def _gate(self, route: str, request: Request) -> Response | None:
        # 🚨 Through the resolver, never off ``request.client``: this is the
        # surface where reading the peer directly would grant admin to every
        # caller behind a front proxy at once.
        remote_ip = self.state.identity.client_ip(request)
        self._http.audit_admin_ip(route, remote_ip)
        return self._http.deny_non_admin(request, remote_ip)

    def _record(self, request: Request, action: str, target: str,
                detail: dict) -> None:
        """Write one audit record. **On the loop**, before ``_persist``.

        Ordering is load-bearing: the record is appended in memory first so it
        rides the same off-loop write as the change it describes. A trail
        persisted separately, afterwards, can be missing the last entry after a
        crash — and the last entry is the one an operator is looking for.
        """
        self.state.admin_overlay.record({
            "at": time.time(),
            "action": action,
            "target": target,
            "actor": self.state.identity.actor(request),
            "detail": detail,
        })

    async def _persist(self) -> dict:
        """Persist the overlay off-loop. The lock lives on the overlay now,
        because the control routes in ``http_handlers`` persist it too."""
        return await self.state.admin_overlay.persist_async()

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
        acl = self.state.acl
        proxies = self.state.identity.proxies
        return JSONResponse({
            # 🚨 Who you are and what that permits, from the ONE place that
            # decides it. The UI disables its write controls from this rather
            # than working it out — a page whose write affordances are live and
            # whose writes 403 is worse than one that shows them disabled.
            "you": self.state.identity.admin_scope(request),
            "sources": {
                "catalog": {
                    "path": os.environ.get("ROADSTEAD_MODELS_YAML", "")
                            or str(model_catalog._DEFAULT_PATH),
                    "env_var": "ROADSTEAD_MODELS_YAML",
                },
                "agents": {
                    # 🚨 Reported under the CURRENT spelling, and read through
                    # the dual-read helper. This view is where an operator comes
                    # to learn how to point a source — so naming the pre-rename
                    # variable here sent them to the one spelling that, until
                    # 2026-09-02, was the only one that worked.
                    "path": env_with_legacy_prefix("AGENTS_CONFIG", ""),
                    "env_var": "ROADSTEAD_AGENTS_CONFIG",
                },
                "api_keys": {
                    "env_var": "ROADSTEAD_API_KEYS",
                    "declared_in_env": bool(os.environ.get("ROADSTEAD_API_KEYS")),
                    "file": os.environ.get("ROADSTEAD_API_KEYS_FILE", ""),
                },
                "acl": {"env_var": "ROADSTEAD_ACL"},
                # 🚨 The two halves of "which address is this request from, and
                # what does that address get". Reported together because the
                # first silently changes the second: configuring a trusted proxy
                # withdraws the BUILT-IN loopback/docker admin grant from any
                # forwarded request, so an operator whose only admin path was
                # "curl from the box" needs to see that here rather than
                # discover it as a 403.
                "trusted_proxies": {
                    "env_var": "ROADSTEAD_TRUSTED_PROXIES",
                    "networks": proxies.networks(),
                    "forwarded_headers_honoured": bool(proxies),
                    "builtin_admin_nets_apply_to_forwarded": False,
                },
                "admin_nets": {
                    "env_var": "ROADSTEAD_ADMIN_NETS",
                    # 🚨 THE REACH SET, which is the question this block is
                    # about, and which this view did not report until
                    # 2026-09-01. `acl.reach_nets()` was written for the
                    # management plane — its docstring says so — and only the
                    # STARTUP LOG was calling it. So the log told the truth and
                    # this view did not: the two lists below are the
                    # identity-grant nets (`_internal_nets`), a different
                    # question, and `builtin` reported `172.16.0.0/12` as
                    # applying even after naming an operator net had dropped it.
                    # Advertising a grant that is not in force, on the surface
                    # whose entire purpose is the gap between the two.
                    "in_force": acl.reach_nets(),
                    # 🚨 Stated rather than left to be inferred from two lists.
                    # Naming any operator net drops the docker-internal default,
                    # and a containerised deployment that loses its own admin
                    # path finds out as a 403 from an address nothing in its
                    # config mentions.
                    "docker_default_dropped": bool(acl.operator_admin_nets()),
                    "builtin": acl.builtin_admin_nets(),
                    "operator": acl.operator_admin_nets(),
                    # 🚨 A subset of `operator`, and reported rather than left
                    # to be inferred from its absence. A narrowed net looks
                    # exactly like a full one in the list above, and the whole
                    # §3.5 argument is that two sources agreeing most of the
                    # time is where the expensive failures live: an operator
                    # auditing "who can change things" must not have to go back
                    # to the environment string to find out.
                    "readonly": acl.readonly_admin_nets,
                },
                "runtime_flags": {"path": self.state.config.runtime_flags_path},
                "admin_store": {
                    "path": overlay.path,
                    "writable": overlay.writable,
                    "reason": overlay.unwritable_reason(),
                    "runtime_keys": len(overlay.keys),
                    "revocations": len(overlay.revoked),
                    "caller_overrides": len(overlay.agents),
                    # J1: endpoints promoted or taken out of service at runtime.
                    "endpoint_overrides": len(overlay.catalog.get("endpoints") or {}),
                    "provider_overrides": len(overlay.catalog.get("providers") or {}),
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
        # A DURATION on the wire, an instant in the store. Resolved once, here,
        # so the record and the response agree to the second.
        expires_at = (time.time() + spec["expires_in_s"]
                      if spec.get("expires_in_s") is not None else None)
        registry = self.state.identity.keys
        was_configured = registry.configured
        key_id = registry.register(
            agent_id=spec["agent_id"],
            secret=secret,
            key_sha256=digest,
            priority=_coerce_priority(spec.get("priority")),
            min_timeout_s=spec.get("min_timeout_s"),
            admin=bool(spec.get("admin", False)),
            admin_readonly=bool(spec.get("admin_readonly", False)),
            expires_at=expires_at,
            bind=spec.get("bind"),
            may_assert=spec.get("may_assert"),
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
            "admin_readonly": bool(spec.get("admin_readonly", False)),
            "expires_at": expires_at,
            "bind": spec.get("bind", []),
            # 🚨 Load-bearing. Without it the grant lives only in memory: the
            # key keeps working after a restart, its delegation does not, and a
            # delegated caller is then billed to the CREDENTIAL rather than
            # refused — because a key with no grant IGNORES a declared agent_id
            # (§1.5 rule 3). Silent re-billing, arriving at a restart unrelated
            # to anything anyone changed.
            "may_assert": spec.get("may_assert", []),
            "created_at": time.time(),
        }
        self.state.admin_overlay.add_key(record)
        # 🚨 The record carries the key's POLICY and never its `key_sha256`.
        # The overlay stores the digest because it has to replay the enrolment
        # on restart; the audit trail has no such need, and a digest is a
        # working credential to anyone who can compute one.
        self._record(request, "key.enrol", key_id, {
            "agent_id": spec["agent_id"],
            "priority": record["priority"],
            "min_timeout_s": record["min_timeout_s"],
            "admin": record["admin"],
            "admin_readonly": record["admin_readonly"],
            "expires_at": record["expires_at"],
            "bind": record["bind"],
            # The widening field belongs in the trail more than any of the
            # others: it is the one that says which callers this credential may
            # bill.
            "may_assert": record["may_assert"],
            "secret_generated": secret is not None,
        })
        outcome = await self._persist()

        payload = {
            "key_id": key_id,
            "agent_id": spec["agent_id"],
            "priority": record["priority"],
            "min_timeout_s": record["min_timeout_s"],
            "admin": record["admin"],
            "admin_readonly": record["admin_readonly"],
            "expires_at": record["expires_at"],
            "bind": record["bind"],
            # Echoed because it WIDENS. An operator granting a credential the
            # right to bill other callers should see the grant confirmed by the
            # call that made it, not have to go and read a list.
            "may_assert": record["may_assert"],
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
        self._record(request, "key.revoke", key_id, {
            # Where the key was DECLARED, which is what decides whether the
            # revocation survives a restart on its own.
            "declared_in": source,
            "registry_now_empty": not registry.configured,
        })
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

    async def handle_admin_key_rotate(self, request: Request) -> Response:
        """POST — issue a successor and retire the predecessor, as ONE action.

        🚨 This exists because doing it by hand is two calls in an order that
        matters, and both orders are wrong. Enrol-then-revoke leaves a window
        where the successor is live and unknown to the caller; revoke-then-enrol
        leaves one where NOTHING works. Rotation is the operation an operator
        actually performs, so it is the operation the API offers, and it applies
        in memory as a unit before anything is persisted.

        ``overlap_s`` expresses the window an operator wants: the predecessor is
        given an EXPIRY rather than being revoked, so it keeps working while the
        successor is deployed and then stops on its own. It defaults to **0** —
        revoke now. The other default is tempting and wrong: the usual reason to
        rotate is that the old credential should stop working, and a rotation
        that silently left it alive is the one an operator believes they have
        completed.

        The successor INHERITS the predecessor's policy — agent_id, priority,
        deadline floor, admin scope, binding — because a rotation is a new
        secret for the same identity. Changing policy at the same time would
        make one call do two things, and the one it did silently would be the
        one nobody reviewed.
        """
        denied = self._gate(f"{PREFIX}/keys", request)
        if denied is not None:
            return denied
        key_id = request.path_params["key_id"]
        registry = self.state.identity.keys
        row = next((r for r in registry.snapshot() if r["key_id"] == key_id), None)
        if row is None:
            return _error("invalid_request_error",
                          f"no key with id {key_id!r} is registered", 404)
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 — an empty body is the common case
            body = {}
        try:
            spec = validate_key_rotate(body)
        except Invalid as exc:
            return _error("invalid_request_error", str(exc), 400)

        digest = spec.get("key_sha256")
        secret: str | None = None
        if digest is None:
            secret = _SECRET_PREFIX + secrets.token_urlsafe(_SECRET_BYTES)
        successor_id = spec.get("id") or f"{key_id}-r{int(time.time())}"
        expires_at = (time.time() + spec["expires_in_s"]
                      if spec.get("expires_in_s") is not None else None)

        new_id = registry.register(
            agent_id=row["agent_id"],
            secret=secret,
            key_sha256=digest,
            priority=_coerce_priority(row["priority"]),
            min_timeout_s=row["min_timeout_s"],
            admin=bool(row["admin"]),
            admin_readonly=bool(row["admin_readonly"]),
            expires_at=expires_at,
            bind=row["bind"],
            # A successor inherits the delegation grant, unlike the expiry: the
            # grant is a POLICY the operator made about this identity, and a
            # rotation that silently dropped it would break every delegated
            # caller at the moment the key changed. (The expiry is an absolute
            # instant, which is why THAT one cannot be inherited — see below.)
            may_assert=row.get("may_assert") or [],
            key_id=successor_id,
            source="runtime",
        )
        if new_id is None:
            # 🚨 The predecessor is UNTOUCHED. A rotation that revoked the old
            # key and then failed to mint the new one is an outage, and the
            # commonest cause — re-POSTing a digest already enrolled — is
            # entirely recoverable while nothing has been taken away.
            return _error(
                "invalid_request_error",
                "the successor was not registered — the digest is already "
                "enrolled or is unusable; the predecessor is UNCHANGED and "
                "still works", 409)

        record = {
            "id": new_id,
            "agent_id": row["agent_id"],
            "key_sha256": digest or hashlib.sha256(
                secret.encode("utf-8")).hexdigest(),
            "priority": row["priority"],
            "min_timeout_s": row["min_timeout_s"],
            "admin": bool(row["admin"]),
            "admin_readonly": bool(row["admin_readonly"]),
            "expires_at": expires_at,
            "bind": list(row["bind"]),
            # Inherited, unlike the expiry: the grant is a policy the operator
            # made about this identity, and a rotation that dropped it would
            # break every delegated caller at the moment the key changed.
            "may_assert": list(row.get("may_assert") or ()),
            "created_at": time.time(),
            "rotated_from": key_id,
        }
        self.state.admin_overlay.add_key(record)

        overlap_s = float(spec.get("overlap_s") or 0.0)
        warnings: list[str] = []
        if overlap_s > 0:
            retires_at = time.time() + overlap_s
            # 🚨 An overlap may SHORTEN a predecessor's life and must never
            # lengthen it. `overlap_s: 86400` on a key the operator gave two
            # hours would otherwise push its expiry a day out — a rotation
            # quietly extending a credential, which is the same widening this
            # repo refuses everywhere else (a narrowing another statement can
            # cancel is not a narrowing). Truncated to the earlier instant, and
            # said out loud, because the operator asked for a window they are
            # not getting.
            if row["expires_at"] is not None and retires_at > float(row["expires_at"]):
                retires_at = float(row["expires_at"])
                warnings.append(
                    f"the requested overlap would have extended {key_id!r} past "
                    f"its own expiry; it retires at {iso_time(retires_at)} as "
                    f"already declared. A rotation never lengthens a credential")
            registry.set_expiry(key_id, retires_at)
            self.state.admin_overlay.expire_key(key_id, retires_at)
            predecessor = {"key_id": key_id, "retired": False,
                           "expires_at": retires_at}
            if row["source"] == "env":
                warnings.append(
                    f"key {key_id!r} is declared in ROADSTEAD_API_KEYS; the "
                    "expiry is recorded and survives a restart, but the "
                    "declaration does too — remove it once the overlap has "
                    "passed")
        else:
            registry.revoke(key_id)
            self.state.admin_overlay.revoke_key(key_id)
            predecessor = {"key_id": key_id, "retired": True, "expires_at": None}
            if row["source"] == "env":
                warnings.append(
                    f"key {key_id!r} is also declared in ROADSTEAD_API_KEYS; "
                    "the revocation is recorded and survives a restart, but "
                    "remove it from the environment too or the declaration "
                    "will outlive the reason it is dead")

        self._record(request, "key.rotate", key_id, {
            "successor": new_id,
            "overlap_s": overlap_s,
            "predecessor_retired": predecessor["retired"],
            "expires_at": expires_at,
        })
        outcome = await self._persist()
        # 🚨 Read back from the REGISTRY, not echoed from the predecessor's row.
        # Reporting what we meant to register makes the response agree with the
        # intent by construction and unable to disagree with the outcome — the
        # same argument that keeps `roadstead.client` from importing the
        # server's constants. A mutation that registered the successor with
        # default policy left this response still claiming the inherited one.
        landed = next((r for r in registry.snapshot() if r["key_id"] == new_id), {})
        payload = {
            "key_id": new_id,
            "rotated_from": key_id,
            "agent_id": landed.get("agent_id"),
            "priority": landed.get("priority"),
            "min_timeout_s": landed.get("min_timeout_s"),
            "admin": landed.get("admin"),
            "admin_readonly": landed.get("admin_readonly"),
            "expires_at": landed.get("expires_at"),
            "bind": landed.get("bind", []),
            "may_assert": landed.get("may_assert", []),
            "predecessor": predecessor,
            **outcome,
        }
        if secret is not None:
            payload["key"] = secret
            payload["note"] = ("this secret is shown once and cannot be "
                               "recovered — store it now")
        if row["expires_at"] is not None and expires_at is None:
            # 🚨 Found by rotating a key with a two-hour life and reading the
            # response: the successor inherits the POLICY but not the EXPIRY,
            # and comes out permanent. That is the right default — inheriting an
            # absolute instant would mint a successor that expired at the
            # predecessor's moment, possibly seconds later — but it silently
            # weakens a control the operator deliberately set, which is the one
            # thing this plane must never do quietly. The duration is not
            # recoverable either: `created_at` on an env- or file-declared key
            # is process start, not enrolment, so deriving it would be a guess
            # dressed as an inheritance. So it is DISCLOSED.
            warnings.append(
                f"key {key_id!r} expires at {iso_time(row['expires_at'])}; the "
                f"successor has NO expiry. A rotation cannot inherit an "
                f"absolute instant — pass `expires_in_s` to give the successor "
                f"a life of its own")
        if overlap_s <= 0:
            # 🚨 Said out loud. The default is the safe one and it is also the
            # one that breaks a running caller the instant it is chosen.
            warnings.append(
                f"the predecessor was revoked immediately (overlap_s 0): any "
                f"caller still presenting {key_id!r} is refused NOW. Re-run "
                f"with overlap_s to give a deployment window instead")
        if warnings:
            payload["warnings"] = warnings
        return JSONResponse(payload, status_code=201)

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
            "retired": dict(overlay.retired),
            # 🚨 What a per-key BINDING is actually worth here, reported rather
            # than left to be worked out. A binding is checked against the
            # RESOLVED address, which is a forwarded one as soon as an operator
            # configures a trusted proxy — so the same `bind: [10.0.0.0/8]` is a
            # network-level fact on a direct deployment and a statement about
            # what a front proxy vouches for behind one. The §3.5 rule applied
            # to a security control: two sources that agree most of the time are
            # where the expensive failures live.
            "binding": {
                "checked_against": ("the forwarded caller address"
                                    if self.state.identity.proxies
                                    else "the peer address"),
                "forwarded_headers_honoured": bool(self.state.identity.proxies),
                "trusted_proxies": self.state.identity.proxies.networks(),
                "note": (
                    "a binding is only as trustworthy as ROADSTEAD_TRUSTED_PROXIES: "
                    "with no trusted proxy the address is the TCP peer and cannot "
                    "be spoofed; with one it is whatever that proxy reported"
                    if self.state.identity.proxies else
                    "no trusted proxy is configured, so a binding is checked "
                    "against the TCP peer address and cannot be spoofed by a "
                    "caller"),
            },
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
        self._record(request, "caller.quota", agent_id, dict(fields))
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
        # 🚨 Objects, not bare ids, since 2026-09-02. `declared_priority` below
        # reports the AGENT's configured band — step 3 of the precedence in
        # `docs/api.md` §1.1 — and a credential that names its own band (step 2)
        # beats it. Many keys to one `agent_id` is a documented shape, so "the
        # band in force for this caller" HAS NO SINGLE VALUE, and a scalar field
        # claiming otherwise is the operator surface stating something untrue on
        # the plane whose whole purpose is `declared` beside `in_force`. So the
        # agent's default stays where it is and every key says what it does.
        keys_snapshot = self.state.identity.keys.snapshot()
        keys_by_agent: dict[str, list[dict]] = {}
        for row in keys_snapshot:
            keys_by_agent.setdefault(str(row["agent_id"]), []).append(row)
        addresses = _acl_addresses(self.state.acl)

        out: list[dict] = []
        for agent_id in self._known_agent_ids():
            standing = self.state.spend_standing(agent_id)
            rate = self.state.rate_standing(agent_id)
            cfg = self.state.config.agent_config(agent_id)
            spend_row = spends.get(agent_id, {})
            out.append({
                "agent_id": agent_id,
                "identities": {
                    "keys": [
                        {
                            "key_id": k["key_id"],
                            # None when the key names no band — which is not the
                            # same statement as naming P3_INGESTION, and is
                            # exactly what `priority_declared` exists to keep
                            # separable. A null here means "this key defers to
                            # the agent's configured default".
                            "priority": (k["priority"] if k.get("priority_declared")
                                         else None),
                            "overrides_agent_default": bool(
                                k.get("priority_declared")
                                and k["priority"] != cfg.default_priority.name),
                            # The delegation grant, beside the band, because
                            # both answer "what can this credential do that the
                            # agent config does not say".
                            "may_assert": k.get("may_assert", []),
                        }
                        for k in keys_by_agent.get(agent_id, [])
                    ],
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
                    # 🚨 The COMBINED answer, from ProxyState — a caller can be
                    # demoted by its rate rather than its spend, and reporting
                    # the spend standing's own view here would show a caller at
                    # its declared band while it was actually running one below.
                    "effective_priority": self.state.effective_priority(
                        agent_id, cfg.default_priority).name,
                },
                # The abuse control DRR is not (rate.py). Reported beside spend
                # because it is the same threshold shape with the same
                # consequence, and an operator asking "why is this caller slow"
                # needs both answers in one place.
                "rate": rate.as_dict() | {
                    "declared_priority": cfg.default_priority.name,
                    "effective_priority": rate.effective_priority(
                        cfg.default_priority).name,
                },
                "live": self.state.scheduler.agent_snapshot(agent_id),
            })
        return out

    # ---- the operator UI (roadmap Workstream G) --------------------------

    def _gate_ui(self, request: Request) -> Response | None:
        """The admin gate, refusing in a shape a BROWSER can act on.

        🚨 Deliberately different from ``_gate`` in exactly one way: every
        refusal is a **401 carrying ``WWW-Authenticate: Basic``**, where the JSON
        plane distinguishes 401 (a credential that did not resolve) from 403 (a
        resolved identity without the scope).

        That split is right for an API client, which reads the two differently
        and can act on both. It is a dead end for a browser: a 403 produces no
        password box, so an operator arriving at this URL with no credential —
        which is *every* operator, the first time — would see a refusal with no
        way to answer it. A 401 is the one status that makes the browser ask.

        The refusal is otherwise unchanged: it still refuses, and the challenge
        is identical whether or not any key is configured, so it discloses
        nothing about the deployment. The routes the page then calls keep the
        ordinary 401/403 split — this applies to the door, not to the plane
        behind it.
        """
        denied = self._gate(f"{PREFIX}/ui", request)
        if denied is None:
            return None
        return Response(
            "Roadstead management — an admin API key is required. Present it as "
            "the PASSWORD; the username is ignored.\n",
            status_code=401,
            media_type="text/plain; charset=utf-8",
            headers={"WWW-Authenticate": f'Basic realm="{_UI_REALM}"',
                     "Cache-Control": "no-store"},
        )

    async def handle_admin_ui(self, request: Request) -> Response:
        """GET — the operator UI: one static page, served from the wheel.

        The file read runs **off the loop**. It is small and it is cached after
        the first request, so this is not about throughput — it is the invariant:
        a blocking read on the loop thread is a blocking read on the loop thread,
        and the exception a dashboard makes for itself is the one that is still
        there when somebody serves a bigger asset from the same handler.
        """
        denied = self._gate_ui(request)
        if denied is not None:
            return denied
        if self._ui_html is None:
            try:
                html = await asyncio.to_thread(_UI_FILE.read_text, "utf-8")
            except OSError as exc:
                # Shipped in the wheel, so this means a broken install rather
                # than a misconfiguration — say which, since the operator's next
                # move differs completely.
                logger.error("admin UI asset missing at %s: %s", _UI_FILE, exc)
                return _error("invalid_request_error",
                              f"the UI asset is missing from this install ({_UI_FILE})",
                              500)
            # Assigned ON the loop; only the read went off it. Same shape as the
            # overlay's "mutate on the loop, persist off it", and idempotent, so
            # two concurrent first-requests cannot disagree about the result.
            self._ui_html = html
        return HTMLResponse(self._ui_html, headers={
            "Content-Security-Policy": _UI_CSP,
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "X-Frame-Options": "DENY",
            "Cache-Control": "no-store",
        })

    # ---- providers ------------------------------------------------------

    async def handle_admin_audit(self, request: Request) -> Response:
        """GET — who changed what, and when.

        The fifth reporting seam, and the one that answers the question the other
        four cannot: they all report a *state*, and a state cannot say who put it
        there. `/rs/v1/admin/config` shows a quota that is not what the file
        says; only this says which credential moved it and at what time.

        🚨 It reports its own LIMITS as data, not in prose an operator has to
        know to look for. ``persisted`` says whether the trail survives a
        restart — an in-memory-only trail is the default, because no admin store
        is configured by default — and ``dropped`` says how many records the
        bound has discarded. A trail that presented itself as complete while
        being neither durable nor unbounded would be the `finish_reason` repair
        again: a thing that looks like an answer and silences the question.

        Newest FIRST here, oldest first on disk. The file is append-only and the
        reader wants the most recent change.
        """
        denied = self._gate(f"{PREFIX}/audit", request)
        if denied is not None:
            return denied
        overlay = self.state.admin_overlay
        try:
            limit = int(request.query_params.get("limit", "100"))
        except (TypeError, ValueError):
            limit = 100
        limit = min(max(limit, 1), overlay._AUDIT_MAX)
        entries = list(reversed(overlay.audit))[:limit]
        return JSONResponse({
            "entries": entries,
            "count": len(entries),
            "held": len(overlay.audit),
            "capacity": overlay._AUDIT_MAX,
            # 🚨 Both halves of "you cannot rely on this as a log of record".
            "dropped": overlay.audit_dropped,
            "persisted": overlay.writable,
            "store": overlay.path,
            "reason": overlay.unwritable_reason(),
        })

    async def handle_admin_providers(self, request: Request) -> Response:
        """GET — providers and endpoints: declared, discovered, and in force."""
        denied = self._gate(f"{PREFIX}/providers", request)
        if denied is not None:
            return denied
        return JSONResponse(self._providers_view())

    # ---- J1: writes on the providers plane ------------------------------

    async def handle_admin_provider_credential(self, request: Request) -> Response:
        """POST — supply the value for a provider's ``api_key_env``.

        🚨 WRITE-ONLY, and the emit doctrine does not move: this sets the
        variable, and no surface ever reads it back. The response says the
        variable's NAME and that it now resolves — never a prefix, never a
        digest, never a length, because each of those narrows a search.

        🚨 NOT PERSISTED, deliberately, and the response says so rather than
        leaving an operator to discover it at the next restart. The overlay
        holds key *digests* and has never held a secret; writing an outbound
        provider key into a JSON file on disk is a change of posture that
        should be argued on its own merits instead of arriving as a side effect
        of a convenience. The durable path is unchanged and is named in the
        response: export the variable, or put it in the unit file.

        It takes effect immediately because `openrouter._api_key` reads
        `os.environ` at CALL time rather than at startup — so there is no reload
        to trigger and no window where the endpoint is live and unauthenticated.
        """
        denied = self._gate(f"{PREFIX}/providers", request)
        if denied is not None:
            return denied
        provider = request.path_params["provider"]
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return _error("invalid_request_error", "body must be JSON", 400)
        if not isinstance(body, dict):
            return _error("invalid_request_error", "body must be an object", 400)
        unknown = sorted(set(body) - {"value"})
        if unknown:
            # Never a silent drop, on the surface whose purpose is exposing them.
            return _error("invalid_request_error",
                          f"unknown field(s) {unknown}; the only field is "
                          f"'value'", 400)
        value = body.get("value")
        if not isinstance(value, str) or not value.strip():
            return _error("invalid_request_error",
                          "'value' must be a non-empty string", 400)

        cat = model_catalog.load_catalog()
        entry = cat.providers.get(provider)
        if entry is None:
            return _error("unknown_endpoint",
                          f"no provider {provider!r} in the catalog", 404)
        env_var = (entry.api_key_env or "").strip()
        if not env_var:
            # Refused rather than invented: without `api_key_env` there is no
            # variable to set, and choosing one here would put the name in a
            # second place — the provider reads only what the catalog names.
            return _error("invalid_request_error",
                          f"provider {provider!r} declares no api_key_env, so "
                          f"there is no variable to set; add one to models.yaml",
                          400)

        os.environ[env_var] = value.strip()
        # 🚨 The VARIABLE, never the value. An audit record is read by more
        # people than the plane is.
        self._record(request, "provider.credential", provider,
                     {"env_var": env_var})
        return JSONResponse({
            "provider": provider,
            "credential": {"env_var": env_var, "present": True},
            "persisted": False,
            "warnings": [
                f"${env_var} is set for this process only and will NOT survive "
                f"a restart — export it in the environment to make it durable",
            ],
            # The endpoints this just unblocked, so the next step is visible
            # rather than something to go and look for.
            "endpoints": sorted(
                n for n, e in cat.endpoints.items() if e.provider == provider),
        })

    async def handle_admin_endpoint_status(self, request: Request) -> Response:
        """POST — bring a declared endpoint into service, or take it out.

        See `install_endpoint` for why this is safe on a live process, and
        roadmap J1 for why it is a much smaller claim than hot-adding one.
        """
        denied = self._gate(f"{PREFIX}/providers", request)
        if denied is not None:
            return denied
        name = request.path_params["endpoint"]
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return _error("invalid_request_error", "body must be JSON", 400)
        if not isinstance(body, dict):
            return _error("invalid_request_error", "body must be an object", 400)
        unknown = sorted(set(body) - {"status"})
        if unknown:
            return _error("invalid_request_error",
                          f"unknown field(s) {unknown}; the only field is "
                          f"'status'", 400)
        status = body.get("status")
        if status not in ("active", "planned"):
            return _error("invalid_request_error",
                          "'status' must be 'active' or 'planned'. "
                          "'on_demand' and 'retired' are catalog-only: one "
                          "changes how health probes, the other is a record "
                          "that a name is gone, and neither is a routing "
                          "decision to make from here.", 400)

        cat = model_catalog.load_catalog()
        entry = cat.endpoints.get(name)
        if entry is None:
            return _error("unknown_endpoint",
                          f"no endpoint {name!r} in the catalog. Adding one "
                          f"that is not declared is not something this plane "
                          f"can do — it is a models.yaml edit.", 404)

        warnings: list[str] = []
        if status == "active":
            env_var = endpoint_credential_gap(name, cat)
            if env_var:
                # 🚨 THE load-bearing refusal. models.yaml says in its own words
                # what `planned` is for: "a deployment that has not set
                # $OPENROUTER_API_KEY should not have an endpoint in its routing
                # table that cannot serve". Promoting without the credential
                # would put exactly that into the routing table, from the
                # surface whose whole purpose is reporting the gap.
                return _error(
                    "invalid_request_error",
                    f"endpoint {name!r} needs ${env_var}, which is not set. "
                    f"Promoting it would put an endpoint that cannot serve into "
                    f"the routing table — which is what its 'planned' status is "
                    f"there to prevent. Set the credential on provider "
                    f"{entry.provider!r} first.", 400)
            install_endpoint(self.state.config, name, cat)
        else:
            in_flight = int((self.state.scheduler.endpoint_snapshot(name)
                             or {}).get("in_flight") or 0)
            if in_flight:
                # 🚨 Removal is the dangerous direction: the request path reads
                # `config.endpoints.get(...)` at several points AFTER dispatch,
                # so pulling the entry from under work in flight turns a live
                # request into a None dereference. Pausing already drains, so
                # the safe order exists — this refuses rather than inventing a
                # second drain that would duplicate it.
                return _error(
                    "invalid_request_error",
                    f"endpoint {name!r} has {in_flight} request(s) in flight. "
                    f"Pause it first — that drains — then take it out of "
                    f"service.", 400)
            remove_endpoint(self.state.config, name)

        return await self._commit_catalog(
            request, "endpoints", name, {"status": status},
            action="endpoint.status",
            extra={"status": status, "declared_status": entry.status},
            warnings=warnings)

    # ---- J2: the catalog is writable ------------------------------------

    async def _commit_catalog(self, request: Request, section: str, name: str,
                              fields: dict | None, *, action: str,
                              extra: dict | None = None,
                              warnings: list[str] | None = None,
                              replace: bool = False) -> Response:
        """Validate a catalog write BY BUILDING IT, then install it.

        🚨 There is no second validator. The candidate fragment is merged into
        `models.yaml`'s raw dict and coerced by the same code that parses the
        file — capability vocabulary, `policy:` passthrough, duplicate aliases,
        the lot. If that reports a problem about this stanza, the write is
        REFUSED. The file loader treats the same complaint as non-fatal on
        purpose (a typo must not stop a fleet booting); arriving from a request
        it is a 400, because there is an operator on the other end who can fix
        it now. Same check, two consequences, chosen by who is asking.
        """
        overlay = self.state.admin_overlay
        candidate = {sec: dict(stanzas) for sec, stanzas
                     in (overlay.catalog or {}).items()}
        candidate.setdefault(section, {})
        if fields is None:
            candidate[section][name] = None                    # tombstone
        elif replace:
            candidate[section][name] = dict(fields)
        else:
            current = candidate[section].get(name)
            candidate[section][name] = (dict(fields) if current is None
                                        else {**current, **fields})

        subject_prefix = f"{section}.{name}"
        before = len(hooks.config_notices())
        try:
            cat = model_catalog.load_catalog(overlay=candidate)
            # 🚨 And BUILD THE KWARGS. Loading the catalog is only half of what
            # installing does, and the other half is where several checks live —
            # the `policy:` allowlist among them. Validating the load alone
            # accepted a stanza with a mistyped policy key and then emitted the
            # complaint during reconcile, AFTER the write had been agreed. If
            # "validate by building what you would install" is the rule, it has
            # to build all of it.
            affected = ([name] if section == "endpoints"
                        else [e.name for e in cat.endpoints.values()
                              if e.provider == name])
            entries = [cat.endpoints[n] for n in affected if n in cat.endpoints]
            if entries:
                model_catalog.build_endpoint_kwargs(cat, entries)
        except Exception as exc:  # noqa: BLE001 — a bad stanza is a 400
            return _error("invalid_request_error",
                          f"that would not load: {exc}", 400)
        complaints = [n for n in hooks.config_notices()[before:]
                      if str(n.get("subject", "")).startswith(subject_prefix)]
        if complaints:
            return _error(
                "invalid_request_error",
                "; ".join(str(c.get("detail")) for c in complaints), 400)

        # Install: mutate on the loop, persist off it.
        overlay.catalog = candidate
        model_catalog.set_runtime_overlay(candidate)
        if fields is None and section == "endpoints":
            # A tombstone leaves the catalog with no entry for this name, so
            # `reconcile_endpoints` cannot tell it apart from an endpoint some
            # host application configured in code — which it must not delete.
            # The handler knows, so the handler does it.
            remove_endpoint(self.state.config, name)
        # 🚨 Name what changed, or an EDIT to an endpoint already in the table
        # is accepted and does nothing: reconcile leaves existing entries alone
        # so discovery's corrections survive, so nothing would rebuild it.
        touched = ({name} if section == "endpoints"
                   else {e.name for e in cat.endpoints.values()
                         if e.provider == name})
        changed = reconcile_endpoints(self.state.config, cat,
                                      names=touched, rebuild=touched)

        self._record(request, action, name, dict(fields or {}) or {"deleted": True})
        outcome = await self._persist()
        return JSONResponse({
            section[:-1]: name,
            "routed": name in self.state.config.endpoints,
            "reconciled": changed,
            # 🚨 Said on EVERY catalog write, not just a status change. The
            # overlay being the only writer is the doctrine that makes a bad
            # save survivable, and an operator who believes the UI edited their
            # file will go looking for a change that is not there.
            "warnings": (warnings or []) + [
                "models.yaml is unchanged — this is a runtime overlay layered "
                "over it, and the Configuration view shows both",
            ] + [
                f"$" + gap + f" is not set, so {n!r} stays out of the routing "
                f"table until it is" for n, gap in
                [(x, endpoint_credential_gap(x, cat)) for x in changed["blocked"]]
            ],
            **(extra or {}),
            **outcome,
        })

    async def handle_admin_provider_models(self, request: Request) -> Response:
        """GET — the models this provider could serve, for a chooser.

        🚨 Built from the PROVIDER stanza, not from one of its endpoints. The
        moment an operator most needs this list is while creating the first
        endpoint on a new provider, when there is no endpoint to borrow a
        connection from. So a throwaway `EndpointConfig` carries the address and
        the credential — the same two fields `build_endpoint_kwargs` copies off
        a provider — and nothing else.
        """
        denied = self._gate(f"{PREFIX}/providers", request)
        if denied is not None:
            return denied
        name = request.path_params["provider"]
        cat = model_catalog.load_catalog()
        entry = cat.providers.get(name)
        if entry is None:
            return _error("unknown_endpoint",
                          f"no provider {name!r} in the catalog", 404)
        provider = provider_for_engine(entry.engine)
        if not provider.descriptor.lists_available_models:
            # Refused rather than empty: an empty list reads as "this provider
            # has no models", and the truth is that this KIND of backend serves
            # one model and names it rather than offering a catalogue.
            return _error("invalid_request_error",
                          f"{entry.engine} serves one model and names it — "
                          f"there is no catalogue to list. The endpoint's model "
                          f"is discovered, not chosen.", 400)
        gap = (entry.api_key_env or "").strip()
        if gap and not os.environ.get(gap):
            return _error("invalid_request_error",
                          f"${gap} is not set, so the catalogue cannot be "
                          f"fetched. Set the credential first.", 400)
        kwargs = {"endpoint_class": f"{name}-catalogue", "role": f"{name}-catalogue"}
        if entry.base_url:
            kwargs["base_url"] = entry.base_url
        else:
            kwargs["host"] = cat.hosts.get(entry.host, entry.host)
            kwargs["port"] = entry.port
        if entry.api_key_env:
            kwargs["api_key_env"] = entry.api_key_env
        kwargs["backend_engine"] = entry.engine
        probe = EndpointConfig(**kwargs)
        try:
            models = await provider.list_available_models(self.state.backend, probe)
        except Exception as exc:  # noqa: BLE001 — an upstream failure is a 502
            return _error("backend_error",
                          f"could not read the catalogue: {exc}", 502)
        return JSONResponse({"provider": name, "engine": entry.engine,
                             "models": models, "count": len(models)})

    async def handle_admin_catalog_entry(self, request: Request) -> Response:
        """PUT / PATCH / DELETE one provider or endpoint stanza (roadmap J2).

        PUT replaces the runtime stanza, PATCH merges into it, DELETE tombstones
        the name so the catalog no longer has it. All three go through
        ``_commit_catalog``, so all three are validated by building the catalog
        they would install.
        """
        denied = self._gate(f"{PREFIX}/providers", request)
        if denied is not None:
            return denied
        section = ("providers" if "provider" in request.path_params
                   else "endpoints")
        name = request.path_params.get("provider") or request.path_params["endpoint"]
        method = request.method.upper()

        if not _CATALOG_NAME.match(name):
            return _error("invalid_request_error",
                          f"{name!r} is not a usable name: letters, digits, "
                          f"'-', '_' and '.', up to 64 characters", 400)

        if method == "DELETE":
            refusal = self._may_delete(section, name)
            if refusal is not None:
                return refusal
            return await self._commit_catalog(
                request, section, name, None, action=f"{section[:-1]}.delete")

        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return _error("invalid_request_error", "body must be JSON", 400)
        if not isinstance(body, dict):
            return _error("invalid_request_error", "body must be an object", 400)
        allowed = (_PROVIDER_FIELDS if section == "providers"
                   else _ENDPOINT_FIELDS)
        unknown = sorted(set(body) - allowed)
        if unknown:
            # 🚨 Never a silent drop, on the surface whose purpose is exposing
            # them. The file loader reports and continues; a request is refused.
            return _error("invalid_request_error",
                          f"unknown field(s) {unknown} for a {section[:-1]}; "
                          f"known: {sorted(allowed)}", 400)
        if "api_key" in body or "key" in body:
            return _error("invalid_request_error",
                          "a catalog stanza names api_key_env — the NAME of an "
                          "environment variable — and never a key. Set the "
                          "value with POST .../credential.", 400)
        env_var = str(body.get("api_key_env") or "").strip()
        if env_var and not _ENV_VAR_NAME.match(env_var):
            # 🚨 Refused WITHOUT echoing it. The value may be the secret itself,
            # and a 400 that quotes what you typed puts it in the response body,
            # the access log and the browser's history.
            return _error(
                "invalid_request_error",
                "api_key_env must be the NAME of an environment variable "
                "(letters, digits and underscore, e.g. OPENROUTER_API_KEY) — "
                "not the key. What you sent is not a valid variable name, which "
                "usually means a credential was pasted here. It has not been "
                "stored. Set the variable's VALUE with "
                "POST /rs/v1/admin/providers/{provider}/credential, which is "
                "write-only and never persisted.", 400)
        if section == "providers":
            refusal = self._provider_address_refusal(name, body, method)
            if refusal is not None:
                return refusal

        return await self._commit_catalog(
            request, section, name, body,
            action=f"{section[:-1]}.{'replace' if method == 'PUT' else 'edit'}",
            replace=(method == "PUT"))

    def _provider_address_refusal(self, name: str, body: dict,
                                  method: str) -> Response | None:
        """Refuse a provider stanza that cannot reach anything.

        🚨 Off the DESCRIPTOR, never off the engine name. "OpenRouter needs a
        base URL and a key" is a fact about a kind of backend, and the engine
        string is the thing this repo has an AST guard against branching on. A
        fourth remote provider gets these refusals by declaring two booleans.

        Checked here rather than left to first dispatch: without it, a provider
        saved with the wrong address fails as `ProviderMisconfigured` on a real
        request — a config gap surfacing as a runtime fault, which is what this
        plane exists to catch earlier.
        """
        cat = model_catalog.load_catalog()
        existing = cat.providers.get(name)
        engine = str(body.get("engine")
                     or (existing.engine if existing else "")).strip()
        if not engine:
            return _error("invalid_request_error",
                          f"a provider needs an engine; known: "
                          f"{sorted(known_engines())}", 400)
        if engine.lower() not in known_engines():
            # 🚨 A refusal here and NOT in the file loader, which resolves an
            # unknown engine to llama.cpp so a typo degrades rather than taking
            # an endpoint offline. A typo in a file must not stop a fleet
            # booting; a typo in a form has an operator who can fix it now.
            return _error("invalid_request_error",
                          f"unknown engine {engine!r}; known: "
                          f"{sorted(known_engines())}", 400)
        d = provider_for_engine(engine).descriptor
        # On a PATCH the absent fields keep their current values.
        merged = ({} if method == "PUT" else {
            "host": existing.host if existing else "",
            "port": existing.port if existing else 0,
            "base_url": existing.base_url if existing else "",
            "api_key_env": existing.api_key_env if existing else "",
        }) | {k: v for k, v in body.items() if k != "engine"}

        if d.addressed_by_base_url and not str(merged.get("base_url") or "").strip():
            return _error("invalid_request_error",
                          f"{engine} is reached at a base_url, not host/port"
                          + (f" — try {d.default_base_url}" if d.default_base_url
                             else ""), 400)
        if not d.addressed_by_base_url and not str(merged.get("base_url") or "").strip():
            if not str(merged.get("host") or "").strip() or not merged.get("port"):
                return _error("invalid_request_error",
                              f"{engine} is reached at host and port; give both "
                              f"(or a base_url if it sits behind a gateway)", 400)
        if d.requires_credential and not str(merged.get("api_key_env") or "").strip():
            return _error("invalid_request_error",
                          f"{engine} refuses to serve without a credential, so "
                          f"this stanza needs api_key_env — the NAME of the "
                          f"environment variable holding the key, never the key",
                          400)
        return None

    def _may_delete(self, section: str, name: str) -> Response | None:
        """The two refusals deletion has to make."""
        if section == "endpoints":
            in_flight = int((self.state.scheduler.endpoint_snapshot(name)
                             or {}).get("in_flight") or 0)
            if in_flight:
                # 🚨 J1's rule. The request path reads
                # `config.endpoints.get(...)` after dispatch, so removing an
                # entry from under live work is a null dereference.
                return _error("invalid_request_error",
                              f"endpoint {name!r} has {in_flight} request(s) in "
                              f"flight. Pause it first — that drains — then "
                              f"delete it.", 400)
            return None
        # 🚨 A provider with endpoints still naming it would leave a routing
        # table pointing at a connection that no longer exists. Named, not
        # merely refused: "which ones" is the operator's next question.
        cat = model_catalog.load_catalog()
        users = sorted(e.name for e in cat.endpoints.values()
                       if e.provider == name)
        if users:
            return _error("invalid_request_error",
                          f"provider {name!r} still serves {users}. Delete or "
                          f"repoint them first.", 400)
        return None

    def _providers_view(self) -> dict:
        cat = model_catalog.load_catalog()
        # 🚨 Every endpoint the CATALOG declares, not only the ones in force.
        # `self.state.config.endpoints` is built from `cat.routed()`, so a
        # `planned` or `retired` stanza was invisible here — and a `planned`
        # endpoint is the purest example of this plane's one question: written
        # down, parsed, validated, deliberately not serving. An operator could
        # not see that `spill-chat` exists, or that the only thing between it
        # and service is an unset environment variable. Reported since
        # 2026-09-01; see roadmap J1, which cannot promote what nobody can see.
        endpoints = [self._endpoint_view(name, cat)
                     for name in sorted(set(self.state.config.endpoints)
                                        | set(cat.endpoints))]
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
                # (docs/internals.md), so an operator reading "slots: config-seeded" can
                # see it is a property of the engine rather than a missing probe.
                "descriptor": dataclasses.asdict(descriptor),
                "endpoints": by_provider.get(name, []),
            })
        return {"providers": providers, "endpoints": endpoints,
                # 🚨 Every engine this build knows, with what it publishes and
                # how it is reached. The operator UI drives its "add a provider"
                # form off this rather than off a list of its own — so a fourth
                # engine gets a correct form without the page being edited, and
                # nobody has to remember that a page exists.
                "engines": {name: dataclasses.asdict(p.descriptor)
                            for name, p in sorted(known_engines().items())}}

    def _endpoint_view(self, name: str, cat: Any) -> dict:
        ep = self.state.config.endpoints.get(name)
        entry = cat.entry(name)
        if ep is None:
            return self._unrouted_endpoint_view(name, entry, cat)
        descriptor = provider_for_engine(ep.backend_engine).descriptor
        health = self.state.endpoint_health.get(name, {})
        paused = name in self.state.paused_endpoints
        price = self.state.prices.price(name)
        return {
            "endpoint": name,
            "provider": entry.provider if entry else "",
            "kind": ep.kind,
            "status": entry.status if entry else "active",
            "routed": True,
            "role": ep.role,
            "aliases": list(entry.aliases) if entry else [],
            "capabilities": sorted(ep.capabilities),
            # 🚨 The gap, stated three times because it is three different
            # questions. `declared` is the catalog seed, `in_force` is what
            # admission actually uses, and `discovered` says whether the backend
            # was ever asked — which for a vLLM endpoint is permanently False and
            # is a property of the engine, not a fault (docs/internals.md).
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

    def _why_not_in_force(self, name: str, entry: Any, env_var: str) -> str:
        if entry is None:
            return "not in the catalog"
        requested = ((self.state.admin_overlay.catalog.get("endpoints") or {})
                     .get(name) or {}).get("status")
        if requested == "active":
            if env_var and not os.environ.get(env_var):
                return (f"promoted here, but ${env_var} is not set — a routed "
                        f"endpoint that cannot serve is what 'planned' prevents. "
                        f"Set the credential and promote again.")
            return ("promoted here, but not installed — see the startup log")
        if requested == "planned":
            return "taken out of service here"
        return f"the catalog declares status {entry.status!r}"

    def _unrouted_endpoint_view(self, name: str, entry: Any, cat: Any) -> dict:
        """An endpoint the catalog declares that nothing is serving.

        🚨 Every "in force" field is **null**, never zero. A `planned` endpoint
        has not been discovered, probed or priced, and reporting `slots: 0`
        would say the backend was asked and answered nothing — which is the one
        confusion this whole view exists to prevent, in the same words the
        capacity block uses: "discovery agreed" and "discovery never ran" must
        not be indistinguishable. Null means nobody asked.

        The credential IS resolved here, because for a `planned` remote endpoint
        it is usually the only thing standing between the stanza and service,
        and it is the field an operator has come to this view to check. Its
        NAME and whether it resolved — never the value, on the same rule as
        everywhere else.
        """
        provider = entry.provider if entry else ""
        pentry = cat.providers.get(provider)
        env_var = pentry.api_key_env if pentry else ""
        return {
            "endpoint": name,
            "provider": provider,
            "kind": entry.kind if entry else "chat",
            "status": entry.status if entry else "unknown",
            "routed": False,
            # Why it is not serving, in the plane's own idiom — and 🚨 the
            # OVERRIDE outranks the catalog in this sentence. An operator who
            # promoted this endpoint and then restarted without exporting the
            # key is told that, not "status is 'planned'": the second is true of
            # the file and answers a question they did not ask, and it is the
            # difference between "I never turned this on" and "I turned it on
            # and something is missing".
            "not_in_force": {
                "reason": self._why_not_in_force(name, entry, env_var),
                "declared_status": entry.status if entry else None,
                "requested_status": (
                    ((self.state.admin_overlay.catalog.get("endpoints") or {})
                     .get(name) or {}).get("status")),
                "credential": {
                    "env_var": env_var or None,
                    "present": bool(os.environ.get(env_var)) if env_var else None,
                },
            },
            "role": entry.role if entry else "",
            "aliases": list(entry.aliases) if entry else [],
            "capabilities": sorted(k for k, v in (entry.capabilities or {}).items()
                                   if v) if entry else [],
            "capacity": {
                "slots": {"declared": entry.slots if entry else 0,
                          "in_force": None, "discoverable": None},
                "context_per_slot": {
                    "declared": entry.context_per_slot if entry else 0,
                    "in_force": None, "discoverable": None},
            },
            "model": {"declared_fingerprint": None, "serving_fingerprint": None,
                      "served_model_id": None,
                      "pinned_model": entry.model if entry else ""},
            "price": None,
            "health": {"healthy": None, "admin_paused": False,
                       "probe_healthy": None},
            "routing": {"failover_to": None, "spill_to": None},
        }


#: What a runtime stanza may carry. 🚨 Deliberately the catalog's OWN field
#: names — this is `models.yaml`'s format, not a second one, so an operator can
#: read a stanza here and paste it into the file. `api_key_env` names a variable
#: and never a key; there is no field for a secret because the store holds none.
_PROVIDER_FIELDS = frozenset({
    "engine", "host", "port", "base_url", "api_key_env", "notes",
})
_ENDPOINT_FIELDS = frozenset({
    "provider", "kind", "status", "model", "role", "aliases", "slots",
    "context_per_slot", "timeout_floor_s", "timeout_ceiling_s",
    "stream_hard_cap_s", "capabilities", "failover_to", "spill_to",
    "degrade_ok", "spill_ok", "policy", "notes",
})

#: A catalog name is used as a dict key, a URL segment, a DRR budget key and a
#: metrics label. Bounded and boring on purpose.
_CATALOG_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

#: 🚨 What an environment variable is ALLOWED to be called — POSIX, and not a
#: heuristic. `api_key_env` names a variable and never a key, and this is the
#: constraint that makes the difference checkable rather than guessed at: no
#: real credential format is a valid identifier, because they all carry `-`,
#: `.` or `/`. An operator pasted a live OpenRouter key into this field on
#: 2026-09-01 and the plane stored it, wrote it to the overlay on disk, put it
#: in the audit trail and echoed it back from the read view — every one of which
#: the "never emit a credential" rule was supposed to prevent, defeated by a
#: field that merely *looked* like it wanted a secret. The rule was right and
#: nothing enforced its precondition.
_ENV_VAR_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


# ---------------------------------------------------------------------------
# Endpoint status — bringing a declared endpoint into service, and out again
# ---------------------------------------------------------------------------
#
# 🚨 Roadmap J1. `planned` -> `active` is NOT hot-adding an endpoint: the stanza
# already declares provider, model, slots, context, floors, capabilities and
# failover, and it has already been parsed and validated by the catalog loader.
# The only thing that changes is membership of the routing table. Inventing an
# endpoint that is not in `models.yaml` at all is J2 and a much larger question
# — these two functions deliberately cannot do it, because they resolve the
# name through the catalog and refuse what is not there.
#
# Why this is safe to do on a live process, established by reading the code
# rather than by hoping:
#   * `Scheduler` creates an endpoint's queue LAZILY (`if ep not in self._queues`),
#     so a name it has never seen needs no registration.
#   * `health` re-reads `config.endpoints` every poll, so discovery and probing
#     pick the endpoint up on the next cycle rather than needing a restart.
#   * everything else reads `config.endpoints.get(name)`, which is why removal
#     is the dangerous direction and is gated on in-flight work below.

def endpoint_credential_gap(name: str, cat: Any = None) -> str | None:
    """The env var this endpoint needs and does not have, or None.

    🚨 ONE rule, consulted from both places that promote: the runtime handler
    and the startup replay of a persisted promotion. They were written a
    function apart and the second one is the easy one to forget — a status that
    persists while the credential deliberately does not means a restart is
    exactly when an endpoint would come back routed and unable to serve, which
    is the condition `planned` exists to prevent. Same rule, same moment,
    whichever path arrives at it.
    """
    cat = cat or model_catalog.load_catalog()
    entry = cat.endpoints.get(name)
    if entry is None:
        return None
    provider = cat.providers.get(entry.provider)
    env_var = (provider.api_key_env or "").strip() if provider else ""
    if env_var and not os.environ.get(env_var):
        return env_var
    return None


def reconcile_endpoints(config: "ProxyConfig", cat: Any = None, *,
                        names: "frozenset[str] | set[str]",
                        rebuild: "frozenset[str] | set[str]" = frozenset(),
                        ) -> dict:
    """Bring the NAMED endpoints into line with the catalog. What changed.

    🚨 ``names`` is required, and narrow on purpose. An earlier version imposed
    the whole catalog — every routed endpoint in, everything else out — and that
    silently destroyed endpoints a host application had configured **in code**:
    the shipped catalog declares `spill-chat` as `planned`, a caller had added a
    routed one to `config.endpoints` itself, and startup deleted it. "Embed
    Roadstead and configure it yourself" is a supported arrangement, and a
    reconcile that treats the catalog as the only possible author of the routing
    table breaks it without a word.

    So this touches exactly what its caller says changed, and nothing else.

    🚨 The ONE function that decides what is routed, used by an admin write and
    by the startup replay alike. Two code paths that both "bring the fleet up to
    date" is the shape that makes a restart differ from a running edit.

    🚨 **The table is REPLACED, never mutated in place.** `config.endpoints` is
    iterated by loops that `await` between items — the capacity poller probes
    each backend — and mutating it under one of those is
    `RuntimeError: dictionary changed size during iteration`, which is exactly
    what happened the first time an endpoint was created at runtime. Rebinding
    the attribute lets an in-flight iteration finish over the table it started
    with, which is also the more honest semantics: a reconcile is one atomic
    change of the fleet, not a sequence of adds and removes that a reader can
    observe halfway through. It is the alternative to auditing 27 call sites for
    an `await` and being wrong about one of them.

    ``rebuild`` names endpoints whose stanza has been EDITED. An endpoint
    already in the table is otherwise left alone, deliberately: its
    `EndpointConfig` carries discovered state — the slot count `/props`
    corrected, the model fingerprint — and rebuilding it wholesale would discard
    that until the next poll. So the caller says what it changed; nothing else
    is disturbed.

    An endpoint the catalog routes but whose provider needs a credential that is
    not set is left OUT, loudly — `models.yaml` says in its own words that a
    deployment without the key "should not have an endpoint in its routing table
    that cannot serve", and a restart is exactly when that rule would otherwise
    be bypassed, because the status persists and the credential does not.
    """
    cat = cat or model_catalog.load_catalog()
    routed = {e.name for e in cat.routed()}
    table = dict(config.endpoints)
    added, removed, rebuilt, blocked = [], [], [], []
    for name in sorted(names):
        gap = endpoint_credential_gap(name, cat) if name in routed else None
        if name not in routed or gap:
            if gap:
                blocked.append((name, gap))
            if table.pop(name, None) is not None:
                removed.append(name)
            continue
        if name not in table:
            table[name] = build_endpoint(name, cat)
            added.append(name)
        elif name in rebuild:
            table[name] = build_endpoint(name, cat)
            rebuilt.append(name)
    config.endpoints = table          # one atomic swap
    for name, gap in blocked:
        logger.warning(
            "endpoint %r is routed by the catalog but $%s is not set, so it "
            "stays OUT of the routing table — a routed endpoint that cannot "
            "serve is what a 'planned' status prevents. Set the credential.",
            name, gap)
    return {"added": added, "removed": removed, "rebuilt": rebuilt,
            "blocked": [n for n, _ in blocked]}


def build_endpoint(name: str, cat: Any = None) -> EndpointConfig:
    """The EndpointConfig a declared endpoint gets — and nothing else.

    Split out of `install_endpoint` so `reconcile_endpoints` can assemble a new
    table without touching the live one.
    """
    cat = cat or model_catalog.load_catalog()
    entry = cat.endpoints.get(name)
    if entry is None:
        raise KeyError(f"{name!r} is not declared in the catalog")
    return EndpointConfig(**model_catalog.build_endpoint_kwargs(cat, [entry])[name])


def install_endpoint(config: "ProxyConfig", name: str,
                     cat: Any = None) -> EndpointConfig:
    """Put a declared endpoint into the routing table. Idempotent.

    Built through `model_catalog.build_endpoint_kwargs`, the same function a
    restart uses, so a promoted endpoint is configured identically to one that
    booted active. A second builder here would be two places deciding the same
    thing, and the drift would show up as an endpoint that behaves subtly
    differently depending on how it entered service.
    """
    ep = build_endpoint(name, cat)
    # Replaced, not mutated — see `reconcile_endpoints`.
    config.endpoints = {**config.endpoints, name: ep}
    return ep


def remove_endpoint(config: "ProxyConfig", name: str) -> None:
    """Take an endpoint out of the routing table. Idempotent."""
    if name in config.endpoints:
        config.endpoints = {k: v for k, v in config.endpoints.items()
                            if k != name}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
