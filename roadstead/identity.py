"""Caller identity — an API key first, the source address as a second factor.

A caller's identity is the single most load-bearing string in this package. It
is the **DRR fair-share key** (``agent_id`` on every queued request), the quota
holder, the budget holder, and — once Workstream D lands — whoever is billed for
remote spill. Until 2026-09-01 it came from one of two places, and neither was
an authentication:

* the **source IP**, via ``acl.IPIdentityMap`` — which identifies a *host*, not
  a caller, cannot survive DHCP, and shipped with one private fleet's addresses
  compiled into it (scrub item S1);
* the **request body**, on a bare ``/v1/submit``, where ``agent_id`` was simply
  whatever the caller wrote. That door had no access control at all, so any
  caller could claim any identity — including one with a better DRR weight. (It
  was gated by this module in 2026-09-01's Workstream B and **removed entirely**
  in Workstream C; the enriched ``/rs/v1`` door reads no identity from the body
  at all, which is the end state this was heading for.)

This module makes an **API key** the identity, and demotes the address to an
optional second factor. Concretely:

    key  →  Principal(agent_id, priority, min_timeout_s, admin)

and that principal is authoritative: it overrides the body's ``agent_id``, it
supplies the default priority band, it carries the per-identity deadline floor,
and it is what grants (or withholds) the admin surfaces.

Three rules are worth stating outright, because each of them is a decision that
looks arbitrary until it bites:

🚨 **A presented key that does not resolve is a 401 — it never falls back to the
address.** Falling back would mean a caller with a wrong or revoked credential
silently becomes a *different, weaker* identity that still works. That is the
same failure shape as the ``finish_reason`` repair that became a silencer
(``CLAUDE.md``): a correction indistinguishable, from the outside, from the
thing being correct. The error message names the situation and says what to do
about it, so an operator is never left guessing.

🚨 **With no keys configured, the registry is not in play at all.** A presented
key is ignored and the address decides. Every OpenAI client sends an
``Authorization`` header whether or not anybody meant it to — ``EMPTY``,
``sk-no-key-required``, whatever the SDK insisted on — so treating a key as
significant before an operator has configured any would 401 the entire existing
world on upgrade, for a credential nobody chose.

🚨 **A key is an authenticated identity; an address is a weak hint.** So a key
OVERRIDES a body-declared ``agent_id`` and an address only fills in one that was
omitted. Anything else would let the body launder a claim past the credential.

Keys are held as SHA-256 digests and compared by digest, so the plaintext exists
only for as long as it takes to load the config. A key in a config file is a key
in a git history, which is why ``key_sha256:`` is the documented form and
``key:`` is the convenience.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import hooks
from .config import LLMPriority, PriorityBand, priority_to_band
from .constants import _INTERACTIVE_CEILING_S

if TYPE_CHECKING:  # pragma: no cover — typing only
    from .acl import IPIdentityMap

logger = logging.getLogger(__name__)


#: Where a caller may present its key. ``Authorization: Bearer <key>`` is first
#: because it is what every OpenAI client already sends — an existing client
#: needs its ``api_key`` set and nothing else. ``X-API-Key`` exists for callers
#: that are not speaking the OpenAI dialect at all (the enriched ``/rs/v1``).
_BEARER_PREFIX = "bearer "


# ---------------------------------------------------------------------------
# The identity spec grammar — shared by keys and by the address ACL
# ---------------------------------------------------------------------------

def parse_identity_spec(
    spec: str,
    *,
    default_priority: LLMPriority = LLMPriority.P3_INGESTION,
) -> tuple[str, LLMPriority, float | None, bool]:
    """Parse ``agent_id[:priority][:min_timeout_s][:admin]`` in any order.

    ONE grammar for both registries, deliberately: an operator configuring
    ``ROADSTEAD_ACL`` and ``ROADSTEAD_API_KEYS`` in the same compose file should
    not have to learn two spellings of the same four facts. It is also
    backwards-compatible with the two forms the ACL has always accepted
    (``agent_id`` and ``agent_id:PRIORITY``).

    Segments after the id are identified by SHAPE rather than position, so the
    order does not matter and a missing one is simply absent:

    * ``admin``                → the admin scope
    * anything numeric         → ``min_timeout_s`` (the per-identity deadline floor)
    * anything else            → a priority, by NAME

    🚨 A priority must be spelled by name (``P1_TURN_SUPPORT``), not by ordinal.
    ``LLMPriority.coerce`` accepts an int, but a bare ``3`` here is
    indistinguishable from a three-second deadline floor, and guessing between
    those two is how a caller ends up in the wrong band with nothing logged.

    An unparseable segment is dropped with a WARNING rather than raising: this
    runs at startup over operator-supplied strings, and one bad character in one
    entry must not take the proxy down.
    """
    parts = [p.strip() for p in spec.split(":")]
    agent_id = parts[0]
    priority = default_priority
    min_timeout_s: float | None = None
    admin = False
    for segment in parts[1:]:
        if not segment:
            continue
        if segment.lower() == "admin":
            admin = True
            continue
        try:
            min_timeout_s = float(segment)
            continue
        except ValueError:
            pass
        try:
            priority = LLMPriority.coerce(segment)
        except ValueError:
            logger.warning(
                "identity spec %r: ignoring unrecognised segment %r (expected a "
                "priority NAME, a numeric min_timeout_s, or 'admin')",
                spec, segment)
    return agent_id, priority, min_timeout_s, admin


# ---------------------------------------------------------------------------
# The principal
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Principal:
    """One resolved caller.

    ``agent_id`` is the fair-share key — the string DRR budgets, quotas, queue
    rows and completion records are all keyed on. Everything else on this record
    is policy attached to that identity.
    """

    agent_id: str
    #: Band the caller lands in when it declares no priority of its own.
    priority: LLMPriority = LLMPriority.P3_INGESTION
    #: Per-identity MINIMUM deadline, applied ONLY to a deadline the proxy chose
    #: (see ``lifecycle.handle_submit``). ``None`` means no floor.
    min_timeout_s: float | None = None
    #: Whether this identity may reach the admin/control surfaces.
    admin: bool = False
    #: How this identity was established: ``api_key`` (authenticated) or ``ip``
    #: (a weak second factor). The two are NOT interchangeable — see the module
    #: docstring's third rule; ``authenticated`` is the property to branch on.
    source: str = "ip"
    #: A public label for the key that authenticated, safe to log. NEVER the key.
    key_id: str | None = None

    @property
    def authenticated(self) -> bool:
        """True when a credential was presented and verified.

        Branch on THIS, not on ``source == "api_key"`` — the string is a label
        for humans reading logs, and a third identity source (mTLS, a signed
        header from a trusted front proxy) should not have to be added at every
        comparison site. Same reasoning as the provider descriptors: a
        capability, never a name.
        """
        return self.source == "api_key"


@dataclass(frozen=True)
class Denial:
    """A refusal, in the shape the HTTP layer needs to render it."""

    code: str
    status: int
    message: str

    @property
    def openai_type(self) -> str:
        """The OpenAI envelope's ``error.type`` for this refusal.

        ``access_denied`` keeps its own name because that pair — type AND code
        both ``access_denied`` — is what the door has always emitted and
        ``docs/api.md`` §2 makes the error surface stable even when the
        behaviour is unchanged. A 401 is new, so it takes OpenAI's own
        convention for a rejected credential: type ``invalid_request_error``,
        code ``invalid_api_key``, which is what an OpenAI client already knows
        how to read.
        """
        return "access_denied" if self.code == "access_denied" else "invalid_request_error"


@dataclass(frozen=True)
class Resolution:
    """The outcome of identifying one request: exactly one half is set."""

    principal: Principal | None = None
    denial: Denial | None = None

    @property
    def ok(self) -> bool:
        return self.principal is not None


# ---------------------------------------------------------------------------
# The key registry
# ---------------------------------------------------------------------------

def _digest(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


class KeyRegistry:
    """API key → :class:`Principal`, keyed by SHA-256 digest.

    The digest is the storage form AND the lookup key, which buys two things at
    once: the plaintext never outlives the load, and the comparison is a dict
    hit on a fixed-width hex string rather than a byte-by-byte compare on the
    secret itself.
    """

    def __init__(self) -> None:
        self._by_digest: dict[str, Principal] = {}
        # Provenance, kept OUT of the hot-path dict on purpose: ``resolve`` runs
        # on every request and must stay a single dict hit returning the
        # principal itself. Where a key came from is an operator question asked
        # by the management plane a few times a day, not a request-path one.
        # {digest: {"source": "env" | "file" | "runtime", "created_at": float}}
        self._provenance: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self._by_digest)

    @property
    def configured(self) -> bool:
        """True once ANY key exists. Until then the registry is not in play at
        all and a presented key is ignored — see the module docstring."""
        return bool(self._by_digest)

    def register(
        self,
        *,
        agent_id: str,
        secret: str | None = None,
        key_sha256: str | None = None,
        priority: LLMPriority = LLMPriority.P3_INGESTION,
        min_timeout_s: float | None = None,
        admin: bool = False,
        key_id: str | None = None,
        source: str = "file",
    ) -> str | None:
        """Register one key. Returns its ``key_id``, or None if it was rejected.

        Exactly one of ``secret`` (plaintext, hashed here and dropped) or
        ``key_sha256`` (a digest the operator computed) must be given.

        ``source`` records where the registration came from — ``env``, ``file``
        or ``runtime`` — which the management plane needs in order to tell an
        operator that a key it can revoke *now* will come back on restart
        because it is also declared in the environment.
        """
        if bool(secret) == bool(key_sha256):
            logger.warning(
                "api key for %r: give exactly one of key/key_sha256; entry skipped",
                agent_id)
            return None
        if secret is not None:
            digest = _digest(secret)
        else:
            digest = str(key_sha256).strip().lower()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                logger.warning(
                    "api key for %r: key_sha256 is not 64 hex characters; entry "
                    "skipped", agent_id)
                return None
        if not agent_id:
            logger.warning("api key with digest %s… has no agent_id; entry skipped",
                           digest[:8])
            return None
        label = key_id or digest[:8]
        if digest in self._by_digest:
            # Two identities on one secret is not a merge, it is a
            # misconfiguration where one of them silently never applies.
            logger.warning(
                "api key %s is registered twice (%r then %r) — keeping the first",
                label, self._by_digest[digest].agent_id, agent_id)
            return None
        _warn_if_floor_exceeds_ceiling(label, agent_id, priority, min_timeout_s)
        self._by_digest[digest] = Principal(
            agent_id=agent_id,
            priority=priority,
            min_timeout_s=min_timeout_s,
            admin=admin,
            source="api_key",
            key_id=label,
        )
        self._provenance[digest] = {"source": source, "created_at": time.time()}
        return label

    def revoke(self, key_id: str) -> bool:
        """Remove the key labelled ``key_id``. True if one was removed.

        🚨 **Revocation works regardless of where the key was declared.** A key
        that leaked has to stop working NOW, and refusing because it came from
        the environment rather than the runtime store would be a correctness
        argument answered, in the moment, by a breach. What the environment
        decides is whether the revocation SURVIVES A RESTART — which the
        management plane reports rather than silently getting wrong (see
        ``management.AdminOverlay``: an env-declared key is tombstoned in the
        overlay and the read plane says to remove it from the environment too).
        """
        for digest, principal in list(self._by_digest.items()):
            if principal.key_id == key_id:
                del self._by_digest[digest]
                self._provenance.pop(digest, None)
                return True
        return False

    def resolve(self, presented: str) -> Principal | None:
        """The principal behind ``presented``, or None if no key matches."""
        if not presented:
            return None
        return self._by_digest.get(_digest(presented))

    def snapshot(self) -> list[dict]:
        """Serialisable registry state for observability.

        🚨 Carries ``key_id`` and never the digest, let alone the key. A digest
        is not a secret in the cryptographic sense, but it IS a working
        credential for anyone who can compute one, and an admin surface that
        prints it turns a read-only endpoint into a key store.
        """
        return sorted(
            (
                {
                    "key_id": p.key_id,
                    "agent_id": p.agent_id,
                    "priority": p.priority.name,
                    "min_timeout_s": p.min_timeout_s,
                    "admin": p.admin,
                    # Where the registration came from, so an operator can tell
                    # a runtime enrolment from a line in the environment.
                    "source": self._provenance.get(d, {}).get("source", "file"),
                    "created_at": self._provenance.get(d, {}).get("created_at"),
                }
                for d, p in self._by_digest.items()
            ),
            key=lambda row: (str(row["agent_id"]), str(row["key_id"])),
        )

    # -- loading ----------------------------------------------------------

    @classmethod
    def from_env(cls) -> "KeyRegistry":
        """Build from ``ROADSTEAD_API_KEYS`` and ``ROADSTEAD_API_KEYS_FILE``.

        Both are read when both are set; the file is the documented path for
        anything real (it can carry digests rather than secrets), the env var is
        for the one-key container case where a file would be ceremony.

        No keys anywhere is the DEFAULT and is a supported deployment: a
        loopback-only proxy on a laptop needs no credential, and demanding one
        would make the local-first case worse to no benefit.
        """
        reg = cls()
        reg._load_env(os.environ.get("ROADSTEAD_API_KEYS", ""))
        path = os.environ.get("ROADSTEAD_API_KEYS_FILE", "").strip()
        if path:
            reg._load_file(path)
        if reg.configured:
            logger.info("loaded %d API key(s); a presented key now authenticates "
                        "and overrides address-based identity", len(reg))
        return reg

    def _load_env(self, raw: str) -> None:
        """``<secret>=<agent_id>[:priority][:min_timeout_s][:admin]``, comma-separated.

        The right-hand side is the shared grammar (``parse_identity_spec``), so
        this is the ACL's format with a secret where the address goes.
        """
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry or "=" not in entry:
                if entry:
                    logger.warning(
                        "ROADSTEAD_API_KEYS: skipping malformed entry (expected "
                        "<key>=<agent_id>[:priority][:min_timeout_s][:admin])")
                continue
            secret, spec = entry.split("=", 1)
            agent_id, priority, floor, admin = parse_identity_spec(spec.strip())
            self.register(secret=secret.strip(), agent_id=agent_id,
                          priority=priority, min_timeout_s=floor, admin=admin,
                          source="env")

    #: Fields a keys-file entry may set. 🚨 A key NOT in this set is reported,
    #: not dropped in silence: an unreachable knob looks exactly like a policy
    #: decision that the caller is not opted in, which is the failure
    #: ``model_catalog._POLICY_PASSTHROUGH`` and ``load_agent_configs`` have both
    #: already had once each.
    _FILE_FIELDS = frozenset({
        "id", "agent_id", "key", "key_sha256", "priority", "min_timeout_s", "admin",
    })

    def _load_file(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            logger.warning("ROADSTEAD_API_KEYS_FILE=%s does not exist; no keys "
                           "loaded from it", p)
            return
        try:
            import yaml  # lazily, exactly as load_agent_configs does
            raw = yaml.safe_load(p.read_text()) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("api keys file load failed at %s: %s", p, exc)
            return
        entries = raw.get("keys") if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            logger.warning("api keys file at %s: expected a `keys:` list, got %s",
                           p, type(entries))
            return
        loaded = 0
        for entry in entries:
            if not isinstance(entry, dict):
                logger.warning("api keys file: skipping non-mapping entry %r", entry)
                continue
            unknown = sorted(set(entry) - self._FILE_FIELDS)
            if unknown:
                hooks.config_notice(
                    source=str(p),
                    subject=str(entry.get("id") or entry.get("agent_id") or "?"),
                    problem="unknown_key",
                    detail=(f"key entry field(s) {unknown} are not read and have "
                            f"NO effect on this credential"),
                    keys=unknown,
                    known=sorted(self._FILE_FIELDS),
                )
            priority = LLMPriority.P3_INGESTION
            if entry.get("priority") is not None:
                try:
                    priority = LLMPriority.coerce(entry["priority"])
                except ValueError:
                    logger.warning(
                        "api keys file: entry %r has unparseable priority %r; "
                        "using %s", entry.get("id"), entry["priority"],
                        priority.name)
            floor = entry.get("min_timeout_s")
            if self.register(
                agent_id=str(entry.get("agent_id") or ""),
                secret=(str(entry["key"]) if entry.get("key") else None),
                key_sha256=(str(entry["key_sha256"]) if entry.get("key_sha256") else None),
                priority=priority,
                min_timeout_s=(float(floor) if floor is not None else None),
                admin=bool(entry.get("admin", False)),
                key_id=(str(entry["id"]) if entry.get("id") else None),
            ):
                loaded += 1
        logger.info("loaded %d API key(s) from %s", loaded, p)


# ---------------------------------------------------------------------------
# Registration coherence — the floor-above-its-own-ceiling shape
# ---------------------------------------------------------------------------

def _warn_if_floor_exceeds_ceiling(
    label: str,
    agent_id: str,
    priority: LLMPriority,
    min_timeout_s: float | None,
) -> None:
    """Report an INTERACTIVE identity whose deadline floor is above its ceiling.

    This is not a hypothetical. A caller registered in the background band with
    the background deadline floor (1800s) was later promoted to the interactive
    band, whose ceiling is 600s — leaving a floor three times its own ceiling,
    which is incoherent in a way nothing failed on. It was live for a day, and
    the reason it was caught at all is that somebody went looking.

    A registration is operator data, so this REPORTS rather than raising or
    clamping: refusing to start over a policy typo is worse than serving with a
    loud line, and silently clamping would hide the mistake being made. The
    guard is on the SHAPE, not on any particular caller, so it fires for the
    next one too.
    """
    if min_timeout_s is None:
        return
    if priority_to_band(priority) is not PriorityBand.INTERACTIVE:
        return
    if min_timeout_s <= _INTERACTIVE_CEILING_S:
        return
    hooks.degradation(
        component="identity",
        reason="interactive identity has a deadline floor above its own ceiling",
        impact=("the floor can never be honoured — the interactive ceiling clamps "
                "first, so this caller's deadline is silently the ceiling"),
        identity=label,
        agent_id=agent_id,
        priority=priority.name,
        min_timeout_s=min_timeout_s,
        interactive_ceiling_s=_INTERACTIVE_CEILING_S,
    )


# ---------------------------------------------------------------------------
# The resolver — one place that decides who a request is from
# ---------------------------------------------------------------------------

def _header(request: Any, name: str) -> str:
    """Read a header off a Starlette request OR a plain-dict test double.

    Starlette's ``Headers`` is already case-insensitive; a bare ``dict`` (which
    is what most of this suite's request doubles carry) is not, so try the
    spellings a hand-written double is likely to use rather than making every
    call site guess.
    """
    headers = getattr(request, "headers", None)
    if headers is None:
        return ""
    for spelling in (name, name.lower(), name.title(), name.upper()):
        try:
            value = headers.get(spelling)
        except Exception:  # noqa: BLE001 — a double with a hostile .get
            return ""
        if value:
            return str(value)
    return ""


def presented_key(request: Any) -> str:
    """The API key on this request, or ``""``.

    ``Authorization: Bearer <key>`` first (what an OpenAI client sends), then
    ``X-API-Key``. A non-Bearer ``Authorization`` scheme is ignored rather than
    treated as a key: it is somebody else's auth, not a malformed one of ours.
    """
    auth = _header(request, "Authorization").strip()
    if auth.lower().startswith(_BEARER_PREFIX):
        return auth[len(_BEARER_PREFIX):].strip()
    return _header(request, "X-API-Key").strip()


def remote_ip(request: Any) -> str:
    """Source address of ``request``, or ``"unknown"`` when it has none."""
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    return str(host) if host else "unknown"


class IdentityResolver:
    """Resolves a request to a :class:`Principal`, or to a :class:`Denial`.

    The one place that knows the precedence between a credential and an address.
    Everything else — the two OpenAI doors, the three ``/rs/v1`` routes, the
    admin surfaces, the deadline floor — asks this and reads the answer off the
    principal, so a future third factor lands here and nowhere else.
    """

    def __init__(
        self,
        acl: "IPIdentityMap",
        keys: KeyRegistry | None = None,
        *,
        require_key: bool | None = None,
    ) -> None:
        self.acl = acl
        self.keys = keys if keys is not None else KeyRegistry()
        self.require_key = (
            _require_key_from_env() if require_key is None else require_key
        )
        if self.require_key and not self.keys.configured:
            # Fail LOUD at startup rather than 401-ing every caller in
            # production: this combination cannot serve anybody, so it is
            # certainly a misconfiguration and never a policy.
            logger.error(
                "ROADSTEAD_REQUIRE_API_KEY is set but NO keys are configured — "
                "every request will be refused. Set ROADSTEAD_API_KEYS or "
                "ROADSTEAD_API_KEYS_FILE, or unset the requirement.")

    # -- resolution -------------------------------------------------------

    def resolve(self, request: Any) -> Resolution:
        """Identify ``request``. See the module docstring for the three rules."""
        key = presented_key(request)

        # Rule 2: with no keys configured the registry is not in play, so a
        # placeholder Authorization header from an OpenAI client is invisible.
        if key and self.keys.configured:
            principal = self.keys.resolve(key)
            if principal is not None:
                return Resolution(principal=principal)
            # Rule 1: never fall back to the address. Say so in the message —
            # a caller whose key was revoked and whose address happens to be
            # enrolled must not discover the difference in a latency graph.
            return Resolution(denial=Denial(
                code="invalid_api_key",
                status=401,
                message=("invalid API key — the presented credential is not "
                         "registered; it is NOT ignored in favour of the source "
                         "address, so remove the header to be identified by "
                         "address instead"),
            ))

        if self.require_key:
            return Resolution(denial=Denial(
                code="invalid_api_key",
                status=401,
                message=("an API key is required — present it as "
                         "'Authorization: Bearer <key>' or 'X-API-Key: <key>'"),
            ))

        ip = remote_ip(request)
        identity = self.acl.identify(ip)
        if identity is None:
            return Resolution(denial=Denial(
                code="access_denied",
                status=403,
                message=f"access denied for {ip}",
            ))
        agent_id, priority = identity
        return Resolution(principal=Principal(
            agent_id=agent_id,
            priority=priority,
            min_timeout_s=self.acl.min_timeout_s(ip),
            admin=self.acl.is_admin(ip),
            source="ip",
        ))

    def min_timeout_s(self, request: Any) -> float | None:
        """This caller's registered deadline floor, or None.

        Never raises and never denies: it runs on the request path AFTER the
        deadline is already resolved and valid, so anything that goes wrong here
        must cost the floor, not the request.
        """
        try:
            res = self.resolve(request)
            return res.principal.min_timeout_s if res.principal else None
        except Exception:  # noqa: BLE001 — a floor lookup must never 500 a call
            logger.debug("identity floor lookup failed", exc_info=True)
            return None

    def is_admin(self, request: Any) -> bool:
        """Whether ``request`` may reach the admin/control surfaces.

        🚨 An authenticated non-admin identity is NOT admin, even from a host in
        the admin nets. Once a caller says who it is, its privileges are that
        identity's — inheriting the host's would mean a key could only ever
        widen access, never narrow it, which makes a scoped key worthless on the
        machine it runs on.
        """
        res = self.resolve(request)
        return bool(res.principal and res.principal.admin)


def _require_key_from_env() -> bool:
    """``ROADSTEAD_REQUIRE_API_KEY`` — refuse any request without a key.

    Default OFF. On is the right setting for a proxy reachable from anything
    wider than a host, and the wrong one for the laptop case this is local-first
    for, so it is a decision an operator makes rather than one taken for them.
    """
    return os.environ.get("ROADSTEAD_REQUIRE_API_KEY", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
