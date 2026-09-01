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

🚨 **A forwarded address is believed only from a trusted proxy.**
``X-Forwarded-For`` is a caller-supplied string. Honouring one unconditionally
would let any caller assert any source address — the self-asserted ``agent_id``
bug in its third costume — so it is read ONLY when the peer is in
``ROADSTEAD_TRUSTED_PROXIES``, which is empty by default. Until an operator opts
in, the peer address decides exactly as it always has.

Keys are held as SHA-256 digests and compared by digest, so the plaintext exists
only for as long as it takes to load the config. A key in a config file is a key
in a git history, which is why ``key_sha256:`` is the documented form and
``key:`` is the convenience.
"""

from __future__ import annotations

import base64
import binascii
import calendar
import hashlib
import ipaddress
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

#: 🚨 ``Authorization: Basic <base64(user:key)>`` — the KEY goes in the password
#: half and the username is ignored entirely.
#:
#: Added 2026-09-01 for the management UI (Workstream G), and the important part
#: is what it deliberately is NOT: a password. A browser cannot attach a bearer
#: token to a navigation, and ``EventSource`` cannot set a header at all, so a
#: browser-facing surface needs a scheme the browser itself carries. Basic is
#: that scheme — but minting a password to go with it would be a SECOND kind of
#: credential, with its own store, its own rotation and its own revocation,
#: parallel to a key registry that already does all three. So the credential
#: stays the API key and only its wrapper changes.
#:
#: The username is ignored rather than checked against ``key_id``: the label is
#: public, so requiring it adds a second string for an operator to remember and
#: buys nothing an attacker who has the key does not already have.
_BASIC_PREFIX = "basic "


# ---------------------------------------------------------------------------
# The identity spec grammar — shared by keys and by the address ACL
# ---------------------------------------------------------------------------

def parse_identity_spec(
    spec: str,
    *,
    default_priority: LLMPriority = LLMPriority.P3_INGESTION,
) -> tuple[str, LLMPriority, float | None, bool, bool]:
    """Parse ``agent_id[:priority][:min_timeout_s][:admin][:readonly]`` in any order.

    ONE grammar for both registries, deliberately: an operator configuring
    ``ROADSTEAD_ACL`` and ``ROADSTEAD_API_KEYS`` in the same compose file should
    not have to learn two spellings of the same four facts. It is also
    backwards-compatible with the two forms the ACL has always accepted
    (``agent_id`` and ``agent_id:PRIORITY``).

    Segments after the id are identified by SHAPE rather than position, so the
    order does not matter and a missing one is simply absent:

    * ``admin``                → the admin scope
    * ``readonly``             → NARROWS that scope to reads (see below)
    * anything numeric         → ``min_timeout_s`` (the per-identity deadline floor)
    * anything else            → a priority, by NAME

    🚨 ``readonly`` only ever NARROWS. It is meaningless without ``admin`` —
    a non-admin identity cannot reach an admin surface to read it either — and
    it is warned about rather than dropped in silence, because a scope segment
    that quietly does nothing is exactly the shape of an operator believing a
    credential is safer than it is.

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
    readonly = False
    for segment in parts[1:]:
        if not segment:
            continue
        if segment.lower() == "admin":
            admin = True
            continue
        if segment.lower() == "readonly":
            readonly = True
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
                "priority NAME, a numeric min_timeout_s, 'admin' or 'readonly')",
                spec, segment)
    if readonly and not admin:
        logger.warning(
            "identity spec %r: 'readonly' has no effect without 'admin' — it "
            "NARROWS the admin scope to reads and does not grant anything",
            spec)
    return agent_id, priority, min_timeout_s, admin, readonly


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
    #: 🚨 NARROWS ``admin`` to reads. Never widens: an identity without ``admin``
    #: is not granted anything by this being False, and an identity with it can
    #: only ever lose the mutating half. The asymmetry is the same one
    #: ``substitution`` follows and the same one per-key IP binding will — a
    #: scope that could grant what the operator withheld is the self-asserted
    #: ``agent_id`` bug in another costume.
    admin_readonly: bool = False
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

    @property
    def may_admin_write(self) -> bool:
        """Whether this identity may MUTATE through an admin surface.

        🚨 Branch on this, never on ``admin and not admin_readonly`` spelled out
        at a call site — that is the two-copies-of-a-predicate shape that
        ``cost_model.context_fit`` exists to answer, and this one decides an
        authorization. It is also why the field is a narrowing: an identity that
        is not ``admin`` at all can never reach True here however
        ``admin_readonly`` is set.
        """
        return self.admin and not self.admin_readonly


def iso_time(epoch: float | None) -> str:
    """An epoch as something an operator can read in a log line.

    Public because ``management.py`` renders the same instants back to an
    operator, and two spellings of "when did this credential die" is the shape
    ``cost_model.context_fit`` exists to answer one layer down.
    """
    if epoch is None:
        return "an unknown time"
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(epoch)))
    except (TypeError, ValueError, OSError):
        return "an unknown time"


def _address_in_any(ip: str, cidrs: "tuple[str, ...] | list[str]") -> bool:
    """Whether ``ip`` falls in any of ``cidrs``.

    🚨 Fails CLOSED on anything it cannot parse — an unreadable address or an
    unreadable CIDR is not a match. The alternative direction (a malformed entry
    that matches everything) turns a typo in a binding into no binding at all,
    silently, on the one field whose whole purpose is to narrow.
    """
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    for raw in cidrs:
        try:
            if addr in ipaddress.ip_network(str(raw), strict=False):
                return True
        except ValueError:
            logger.warning("api key binding: %r is not an address or CIDR; it "
                           "matches nothing", raw)
    return False


#: HTTP methods that only READ. Anything else is treated as a mutation by the
#: admin gate — DEFAULT-DENY, so a method nobody anticipated is refused to a
#: read-only identity rather than waved through. Derived from the METHOD rather
#: than from a list of write routes on purpose: a route list is a second thing
#: to keep in step with ``routes.py``, and the failure when it falls behind is
#: silent and in the widening direction.
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


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
class KeyLookup:
    """The outcome of matching one presented secret against the registry.

    Three states, and the third is the one this type exists for: no match, a
    match, and a match that is no longer usable. Collapsing the last two into
    ``None`` is what made an expired credential indistinguishable from a typo.
    """

    #: The identity, when the key matched AND is still usable.
    principal: Principal | None = None
    #: Set when the key matched but has passed its expiry.
    expired_at: float | None = None
    #: The matched key's public label, set even when it expired — an operator
    #: needs to know WHICH key it was.
    key_id: str | None = None
    #: CIDRs the key may be presented from. Empty means anywhere.
    bind: tuple[str, ...] = ()

    @property
    def expired(self) -> bool:
        return self.expired_at is not None


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
        # Facts about the CREDENTIAL, as opposed to the identity it confers.
        # Kept out of the Principal because a Principal is what a caller IS and
        # these are conditions on the key being usable at all — one of them
        # (``bind``) cannot even be evaluated without the request.
        #
        #   source      "env" | "file" | "runtime"
        #   created_at  epoch seconds
        #   expires_at  epoch seconds, or None for "until revoked"
        #   bind        list of CIDR strings the key may be presented from
        #
        # 🚨 ``expires_at`` and ``bind`` ARE read on the request path, which the
        # older comment here (which called this dict pure provenance) said they
        # were not. ``lookup`` stays one dict hit for the principal and one for
        # this; binding is checked by ``IdentityResolver``, which is the only
        # thing that has an address to check it against.
        self._credential: dict[str, dict] = {}

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
        admin_readonly: bool = False,
        expires_at: float | None = None,
        bind: "list[str] | tuple[str, ...] | None" = None,
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
        if admin_readonly and not admin:
            # Same disclosure as the spec grammar's: the field did not grant
            # anything and did not take anything away, and an operator who
            # believes otherwise has a wrong idea of what this credential can do.
            logger.warning(
                "api key %s for %r sets admin_readonly without admin — it "
                "NARROWS the admin scope and grants nothing on its own",
                label, agent_id)
        self._by_digest[digest] = Principal(
            agent_id=agent_id,
            priority=priority,
            min_timeout_s=min_timeout_s,
            admin=admin,
            admin_readonly=admin_readonly,
            source="api_key",
            key_id=label,
        )
        self._credential[digest] = {
            "source": source,
            "created_at": time.time(),
            "expires_at": expires_at,
            "bind": list(bind or ()),
        }
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
                self._credential.pop(digest, None)
                return True
        return False

    def is_expired(self, digest: str, *, now: float | None = None) -> bool:
        """Whether the key stored under ``digest`` has passed its expiry.

        A key with no ``expires_at`` never expires — which is what every key was
        before 2026-09-01, so an existing registry behaves exactly as it did.
        """
        expires_at = self._credential.get(digest, {}).get("expires_at")
        if expires_at is None:
            return False
        return (time.time() if now is None else now) >= float(expires_at)

    def binding(self, key_id: str) -> list[str]:
        """The CIDRs ``key_id`` may be presented from. Empty means anywhere."""
        for digest, principal in self._by_digest.items():
            if principal.key_id == key_id:
                return list(self._credential.get(digest, {}).get("bind", ()))
        return []

    def lookup(self, presented: str) -> "KeyLookup":
        """Resolve ``presented``, distinguishing WHY it failed.

        🚨 An expired key and an unknown one are both refused and must not be
        refused with the same words. "not registered" sends an operator to check
        whether they pasted the right string; "expired at T" sends them to
        enrol a successor. Both are 401 and neither falls back to the address
        (§1.5 rule 1) — what differs is the sentence, which is the part a human
        acts on. Same argument as the 401/403 split on the admin gate.

        The binding is NOT checked here: it needs the request's address, and
        ``identity.py`` resolves an address in exactly one place, which is
        ``IdentityResolver``.
        """
        if not presented:
            return KeyLookup()
        digest = _digest(presented)
        principal = self._by_digest.get(digest)
        if principal is None:
            return KeyLookup()
        if self.is_expired(digest):
            return KeyLookup(
                expired_at=self._credential.get(digest, {}).get("expires_at"),
                key_id=principal.key_id)
        return KeyLookup(principal=principal, key_id=principal.key_id,
                         bind=tuple(self._credential.get(digest, {}).get("bind", ())))

    def set_expiry(self, key_id: str, expires_at: float | None) -> bool:
        """Give ``key_id`` an expiry (or clear one). True if a key was found.

        The half of a rotation that RETIRES rather than revokes. Same argument
        as :meth:`revoke`: it works regardless of where the key was declared,
        because a credential being wound down has to actually wind down; what
        the declaration decides is whether the change survives a restart, which
        the management plane reports rather than silently getting wrong.
        """
        for digest, principal in self._by_digest.items():
            if principal.key_id == key_id:
                self._credential.setdefault(digest, {})["expires_at"] = expires_at
                return True
        return False

    def resolve(self, presented: str) -> Principal | None:
        """The principal behind ``presented``, or None if no key matches.

        The plain accessor. It reports an expired key as no key at all, which is
        correct for a caller asking "does this work"; a caller that needs to
        say WHY it does not wants :meth:`lookup`.
        """
        return self.lookup(presented).principal

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
                    # Reported beside `admin` rather than folded into it: an
                    # operator auditing who can change things needs to see WHICH
                    # of the admin credentials can, and one collapsed field
                    # ("admin: read") would make a full-admin key and a
                    # read-only one indistinguishable at a glance in the UI.
                    "admin_readonly": p.admin_readonly,
                    "may_write": p.may_admin_write,
                    # Where the registration came from, so an operator can tell
                    # a runtime enrolment from a line in the environment.
                    "source": self._credential.get(d, {}).get("source", "file"),
                    "created_at": self._credential.get(d, {}).get("created_at"),
                    # 🚨 Lifecycle, reported because nothing else can report it.
                    # A key that expires in an hour and one that never expires
                    # are indistinguishable everywhere else, and the difference
                    # is a caller that stops working at a moment unrelated to
                    # anything anyone did.
                    "expires_at": self._credential.get(d, {}).get("expires_at"),
                    "expired": self.is_expired(d),
                    # The addresses it may be presented from. Empty = anywhere,
                    # which is what every key was before 2026-09-01.
                    "bind": list(self._credential.get(d, {}).get("bind", ())),
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
                        "<key>=<agent_id>[:priority][:min_timeout_s][:admin]"
                        "[:readonly])")
                continue
            secret, spec = entry.split("=", 1)
            agent_id, priority, floor, admin, readonly = parse_identity_spec(
                spec.strip())
            self.register(secret=secret.strip(), agent_id=agent_id,
                          priority=priority, min_timeout_s=floor, admin=admin,
                          admin_readonly=readonly, source="env")

    #: Fields a keys-file entry may set. 🚨 A key NOT in this set is reported,
    #: not dropped in silence: an unreachable knob looks exactly like a policy
    #: decision that the caller is not opted in, which is the failure
    #: ``model_catalog._POLICY_PASSTHROUGH`` and ``load_agent_configs`` have both
    #: already had once each.
    _FILE_FIELDS = frozenset({
        "id", "agent_id", "key", "key_sha256", "priority", "min_timeout_s",
        "admin", "admin_readonly", "expires_at", "bind",
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
                admin_readonly=bool(entry.get("admin_readonly", False)),
                expires_at=_expiry_from_file(entry),
                bind=_binding_from_file(entry),
                key_id=(str(entry["id"]) if entry.get("id") else None),
            ):
                loaded += 1
        logger.info("loaded %d API key(s) from %s", loaded, p)


def _expiry_from_file(entry: dict) -> float | None:
    """``expires_at`` from a keys-file entry: an epoch, or an ISO-8601 date.

    Both spellings, because an operator writing a keys file by hand writes
    ``2027-01-01`` and an operator generating one writes an epoch. An
    unparseable value is a WARNING and NO expiry rather than an immediate one:
    this runs at startup over operator-supplied strings, and a typo that quietly
    killed a credential would look exactly like a revocation nobody made.
    """
    raw = entry.get("expires_at")
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    text = str(raw).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return calendar.timegm(time.strptime(text, fmt))
        except ValueError:
            continue
    try:
        return float(text)
    except ValueError:
        logger.warning(
            "api keys file: entry %r has unparseable expires_at %r — the key is "
            "loaded WITHOUT an expiry rather than as already expired",
            entry.get("id"), raw)
        return None


def _binding_from_file(entry: dict) -> list[str]:
    """``bind`` from a keys-file entry: one CIDR or a list of them."""
    raw = entry.get("bind")
    if raw is None:
        return []
    values = raw if isinstance(raw, list) else [raw]
    return [str(v).strip() for v in values if str(v).strip()]


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


def _basic_password(blob: str) -> str:
    """The password half of a base64 ``user:password``, or ``""``.

    An undecodable blob yields ``""`` — i.e. NO credential was presented, so the
    address decides. That is the right reading: a header we cannot parse is not
    a credential that failed, and 401-ing it would refuse a caller over bytes
    nobody meant as ours. A *decodable* one with a password is a presented
    credential and takes rule 1 in full.
    """
    try:
        decoded = base64.b64decode(blob, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return ""
    _user, sep, password = decoded.partition(":")
    return password.strip() if sep else ""


def presented_key(request: Any) -> str:
    """The API key on this request, or ``""``.

    Three places, in order: ``Authorization: Bearer <key>`` (what an OpenAI
    client sends), ``Authorization: Basic <base64(anything:key)>`` (what a
    BROWSER can send — see ``_BASIC_PREFIX``), then ``X-API-Key`` for callers
    speaking neither dialect.

    🚨 A scheme this does not recognise is still ignored rather than treated as
    a malformed key of ours — it is somebody else's auth. What changed on
    2026-09-01 is only that ``Basic`` moved from that category into ours, which
    is a wire-contract change and is recorded as one.
    """
    auth = _header(request, "Authorization").strip()
    if auth.lower().startswith(_BEARER_PREFIX):
        return auth[len(_BEARER_PREFIX):].strip()
    if auth.lower().startswith(_BASIC_PREFIX):
        return _basic_password(auth[len(_BASIC_PREFIX):].strip())
    return _header(request, "X-API-Key").strip()


def remote_ip(request: Any) -> str:
    """The PEER address — the far end of the socket — or ``"unknown"``.

    This is what the transport observed and is never caller-supplied, so it is
    the address the trust decision below is *made about*. It is NOT necessarily
    the caller: behind a reverse proxy it is the proxy. Use
    :meth:`IdentityResolver.client_ip`, which is the resolved answer.
    """
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    return str(host) if host else _UNKNOWN_ADDRESS


# ---------------------------------------------------------------------------
# Forwarded addresses — the reverse-proxy case
# ---------------------------------------------------------------------------
#
# 🚨 The whole of this section exists to answer one question safely: when the
# peer is a reverse proxy, WHICH of the addresses it forwarded is the caller?
#
# Get it wrong in one direction and every caller collapses into the proxy's
# address — the IP layer becomes a single identity, ``ROADSTEAD_ACL`` stops
# distinguishing anybody, and if the proxy sits in the admin nets (loopback and
# docker-internal are there by DEFAULT, and a sidecar proxy usually is one of
# them) the control plane is granted to whoever can reach the proxy.
#
# Get it wrong in the other direction — believe ``X-Forwarded-For`` from anyone,
# or take its LEFTMOST element — and a caller asserts its own source address,
# which is the self-asserted ``agent_id`` bug wearing a different hat.

#: The address reported when there is no usable one. Deliberately not an IP:
#: ``IPIdentityMap`` cannot match it, so it is refused rather than admitted, and
#: ``is_admin`` is False for it. Fail-closed by construction.
_UNKNOWN_ADDRESS = "unknown"

_FORWARDED_FOR = "X-Forwarded-For"

#: Cap on hops examined. The walk below stops at the first UNTRUSTED hop, so a
#: caller cannot lengthen it — except by repeating a trusted proxy's address
#: thousands of times, which is the only reason this exists. A chain longer than
#: this is not parsed at all; it is a hostile header, not a deployment.
_MAX_FORWARDED_HOPS = 32


class TrustedProxies:
    """The addresses whose ``X-Forwarded-For`` header may be believed.

    🚨 **Empty by default, and an empty set means the header is never read.**
    Trusting a forwarded address is an operator statement about their own
    topology — that a specific box sits in front and rewrites this header — and
    nothing about the request itself can supply that statement.
    """

    def __init__(self, nets: Any = ()) -> None:
        self._nets: list[Any] = list(nets)

    def __bool__(self) -> bool:
        return bool(self._nets)

    def __len__(self) -> int:
        return len(self._nets)

    def trusts(self, address: str) -> bool:
        """Whether ``address`` is one of the configured proxies.

        A non-address (``"unknown"``, a hostname, junk from a header) is not
        trusted — never raises, because this runs on the request path.
        """
        if not self._nets or not address:
            return False
        try:
            addr = ipaddress.ip_address(address)
        except ValueError:
            return False
        return any(addr in net for net in self._nets)

    def networks(self) -> list[str]:
        """The configured entries, for the management plane's config view."""
        return [str(net) for net in self._nets]

    @classmethod
    def parse(cls, raw: str, *, source: str = "ROADSTEAD_TRUSTED_PROXIES") -> "TrustedProxies":
        """Comma-separated addresses or CIDRs. An unparseable entry is DROPPED.

        Dropping is the safe direction — a typo removes trust rather than
        granting it — but a dropped entry is exactly the shape of failure
        ``hooks.config_notice`` exists for: the operator believes they are
        behind a proxy, the header is being ignored, and every caller is
        collapsing into one identity with nothing failing.
        """
        nets = []
        for entry in (raw or "").split(","):
            entry = entry.strip()
            if not entry:
                continue
            try:
                nets.append(ipaddress.ip_network(entry, strict=False))
            except ValueError:
                logger.warning(
                    "%s: %r is not an address or CIDR; it is NOT trusted",
                    source, entry)
                hooks.config_notice(
                    source=source,
                    subject=entry,
                    problem="unparseable",
                    detail=("not an address or CIDR — this proxy is NOT trusted, "
                            "so X-Forwarded-For from it is ignored and its "
                            "callers all resolve to its own address"),
                )
        return cls(nets)

    @classmethod
    def from_env(cls) -> "TrustedProxies":
        """``ROADSTEAD_TRUSTED_PROXIES`` — empty by default."""
        proxies = cls.parse(os.environ.get("ROADSTEAD_TRUSTED_PROXIES", ""))
        if proxies:
            logger.info(
                "trusting X-Forwarded-For from %d proxy network(s): %s. The "
                "built-in loopback/docker admin grant no longer applies to a "
                "FORWARDED request — grant admin with an admin API key or "
                "ROADSTEAD_ADMIN_NETS.",
                len(proxies), ", ".join(proxies.networks()))
        return proxies


@dataclass(frozen=True)
class ClientAddress:
    """The address a request is FROM, and how sure we are of it."""

    #: The caller's address, or ``"unknown"``.
    ip: str
    #: True when this came out of ``X-Forwarded-For`` on a connection from a
    #: trusted proxy. 🚨 Branch on THIS rather than on whether the ip differs
    #: from the peer — a proxy forwarding its own address is still forwarded,
    #: and the point of the flag is what the transport can vouch for.
    forwarded: bool = False


def _forwarded_hop(raw: str) -> str | None:
    """One ``X-Forwarded-For`` element as a bare address, or None.

    Proxies vary: bare addresses, ``ip:port`` for v4, ``[v6]:port``. Anything
    that is not an address after that is not guessed at.
    """
    text = raw.strip()
    if text.startswith("["):                    # [2001:db8::1]:443
        end = text.find("]")
        if end == -1:
            return None
        text = text[1:end]
    elif text.count(":") == 1:                  # 192.0.2.4:51234 — never bare v6
        text = text.split(":", 1)[0]
    try:
        return str(ipaddress.ip_address(text))
    except ValueError:
        return None


def client_address(request: Any, trusted: TrustedProxies | None = None) -> ClientAddress:
    """Resolve ``request`` to the address of the caller.

    🚨 **The hop is chosen by walking the chain from the RIGHT, stopping at the
    first address that is not a trusted proxy.** ``X-Forwarded-For`` is appended
    to left-to-right, so the rightmost element is what the closest proxy
    observed and the leftmost is whatever the original caller sent — which is to
    say, caller-controlled. Taking the leftmost re-introduces the spoof this
    exists to close; a caller that prepends ten fake hops simply has them
    ignored, because the walk stops before it ever reaches them.

    The walk is used rather than counting ``N`` trusted proxies and taking the
    ``(N+1)``th from the right: the trusted set is expressed as CIDRs, so its
    *width* is not its depth, and a chain that is one hop shorter than expected
    would silently return a proxy's address as a caller's.
    """
    peer = remote_ip(request)
    if not trusted or not trusted.trusts(peer):
        # The overwhelmingly common case, and the default: no proxy is trusted,
        # or this connection did not come from one. The header is not read at
        # all — it is not "validated and rejected", it is never consulted.
        return ClientAddress(ip=peer, forwarded=False)

    chain = [part for part in _header(request, _FORWARDED_FOR).split(",") if part.strip()]
    if not chain:
        # A trusted proxy that forwards nothing has told us nothing. The peer is
        # all we have, and it is the proxy — so this is exactly the collapse
        # case, and marking it forwarded is what withdraws the built-in admin
        # grant from it. A misconfigured front proxy must not be an admin.
        return ClientAddress(ip=peer, forwarded=True)
    if len(chain) > _MAX_FORWARDED_HOPS:
        logger.warning("X-Forwarded-For from %s has %d hops (cap %d); ignoring it",
                       peer, len(chain), _MAX_FORWARDED_HOPS)
        return ClientAddress(ip=_UNKNOWN_ADDRESS, forwarded=True)

    hop = None
    for raw in reversed(chain):
        hop = _forwarded_hop(raw)
        if hop is None:
            # 🚨 Unparseable, and REACHED by the walk — so it is standing where
            # a caller's address should be. Resolving to the peer here would
            # hand the proxy's identity (and its admin grant) to anyone who
            # sends junk, so this fails closed instead.
            logger.warning("X-Forwarded-For from %s contains an unparseable hop; "
                           "the caller cannot be identified", peer)
            return ClientAddress(ip=_UNKNOWN_ADDRESS, forwarded=True)
        if not trusted.trusts(hop):
            return ClientAddress(ip=hop, forwarded=True)

    # Every hop was itself a trusted proxy. The caller is further left than the
    # chain goes, so the leftmost is the closest thing to an answer we have.
    return ClientAddress(ip=str(hop), forwarded=True)


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
        trusted_proxies: TrustedProxies | None = None,
    ) -> None:
        self.acl = acl
        self.keys = keys if keys is not None else KeyRegistry()
        self.proxies = (
            TrustedProxies.from_env() if trusted_proxies is None else trusted_proxies
        )
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

    # -- addressing -------------------------------------------------------

    def client_address(self, request: Any) -> ClientAddress:
        """Who this request is FROM, honouring a trusted proxy's forwarding.

        🚨 The ONE place an address is resolved. ``http_handlers`` and
        ``management`` call this rather than reading ``request.client``: a call
        site that reads the peer directly is a call site where a reverse proxy
        collapses every caller into one identity, and there is nothing about
        such a bug that fails loudly.
        """
        return client_address(request, self.proxies)

    def client_ip(self, request: Any) -> str:
        """:meth:`client_address` when only the string is wanted (audit, logs)."""
        return self.client_address(request).ip

    # -- resolution -------------------------------------------------------

    def resolve(self, request: Any) -> Resolution:
        """Identify ``request``. See the module docstring for the three rules."""
        key = presented_key(request)

        # Rule 2: with no keys configured the registry is not in play, so a
        # placeholder Authorization header from an OpenAI client is invisible.
        if key and self.keys.configured:
            found = self.keys.lookup(key)
            if found.principal is not None:
                bound = self._binding_denial(request, found)
                return Resolution(denial=bound) if bound else Resolution(
                    principal=found.principal)
            if found.expired:
                # 🚨 Its OWN sentence. "Not registered" would send an operator
                # to check whether they pasted the right string; the key is
                # exactly right and simply out of time, and the fix is a
                # successor rather than a correction. Same code and same status
                # as below — a caller can do nothing differently either way, and
                # §2.1 mints no code it does not need.
                return Resolution(denial=Denial(
                    code="invalid_api_key",
                    status=401,
                    message=(f"expired API key — credential {found.key_id!r} "
                             f"expired at {iso_time(found.expired_at)} and is no "
                             f"longer accepted. It is NOT ignored in favour of "
                             f"the source address; rotate it "
                             f"(POST /rs/v1/admin/keys/{found.key_id}/rotate) "
                             f"or enrol a successor"),
                ))
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

        address = self.client_address(request)
        ip = address.ip
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
            # 🚨 A FORWARDED address does not inherit the BUILT-IN admin nets.
            # Those nets are loopback and docker-internal, and the whole
            # justification for auto-granting admin to them is that reaching
            # them meant already being on the box. A front proxy negates that
            # exactly: "arrived on loopback" now means "came in the front door".
            # An operator who genuinely wants a forwarded address to be admin
            # says so in ROADSTEAD_ADMIN_NETS — or, better, issues an admin key,
            # which works from anywhere and is revocable.
            admin=self.acl.is_admin(ip, trust_builtin_nets=not address.forwarded),
            # A narrowing the operator wrote on the SAME entry that granted the
            # scope (`=ops:admin:readonly`). It is not gated on `forwarded`: the
            # forwarding rule above withdraws a grant, and withdrawing a
            # narrowing would widen one.
            admin_readonly=self.acl.is_admin_readonly(ip),
            source="ip",
        ))

    def _binding_denial(self, request: Any, found: "KeyLookup") -> Denial | None:
        """Whether this key may be presented from THIS address. None to proceed.

        🚨 An ADDITIONAL constraint on a credential, never a way for one to
        widen what an address grants. A key with no binding is unconstrained,
        which is what every key was before 2026-09-01; a bound key can only ever
        be refused somewhere it would otherwise have worked. The identity it
        confers is untouched — a bound key does not *become* an address
        identity, and a caller inside the binding gets exactly the principal the
        key always carried.

        🚨 It is checked against the RESOLVED address, which may be a forwarded
        one. A binding written for a deployment behind a reverse proxy is
        therefore only as trustworthy as ``ROADSTEAD_TRUSTED_PROXIES``: if that
        is empty the address is the peer and cannot be spoofed, and if it is
        configured the binding inherits whatever that list vouches for. The
        management plane says so where an operator reads it (``docs/api.md``
        §3.3) rather than leaving it to be worked out — this is the §3.5 rule
        applied to a security control.

        Refused with the same code and status as every other bad credential.
        A caller cannot act on the distinction; the operator reading the log
        can, and the message is where that goes.
        """
        if not found.bind:
            return None
        address = self.client_address(request)
        if _address_in_any(address.ip, found.bind):
            return None
        return Denial(
            code="invalid_api_key",
            status=401,
            message=(f"API key {found.key_id!r} is bound to "
                     f"{', '.join(found.bind)} and was presented from "
                     f"{address.ip}"
                     + (" (a FORWARDED address — the binding is only as "
                        "trustworthy as ROADSTEAD_TRUSTED_PROXIES)"
                        if address.forwarded else "")),
        )

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

        Says nothing about whether the request may CHANGE anything — see
        :meth:`admin_denial`, which is what a gate should call.
        """
        res = self.resolve(request)
        return bool(res.principal and res.principal.admin)

    def actor(self, request: Any) -> dict:
        """Who is acting, in a form safe to write to an audit record.

        🚨 NEVER a credential — not the key, not the digest, on the same rule
        that governs every other management readout. ``key_id`` is the public
        label the registry already publishes, and publishing it is the point: a
        trail that could not name the credential would have to be keyed on the
        address instead, which is the factor this design treats as the weak one.

        BOTH halves are recorded, always. ``key_id`` is None for an
        address-derived admin, and a record showing an address and no key is a
        meaningful — and slightly alarming — thing for an operator to find.
        Collapsing them into one "actor" string would hide which factor actually
        authorized the change.

        Lives here because "who is this" is an identity fact and because the
        address must come from :meth:`client_ip`. A caller assembling its own
        actor dict is how ``management.py`` grew a second ``_remote_ip``.
        """
        principal = self.resolve(request).principal
        return {
            "key_id": principal.key_id if principal else None,
            "agent_id": principal.agent_id if principal else None,
            "source": principal.source if principal else "unknown",
            "address": self.client_ip(request),
        }

    def admin_scope(self, request: Any) -> dict:
        """What this request's admin scope permits, as a readout.

        Exists so nothing outside this module has to read ``admin_readonly`` or
        ``may_admin_write`` to find out — the management plane publishes this
        verbatim and the UI disables its write controls from it, and both would
        otherwise be re-deciding what a scope permits. ``tests/
        test_admin_audit.py`` fails if either field is read anywhere else.
        """
        principal = self.resolve(request).principal
        return {
            "key_id": principal.key_id if principal else None,
            "agent_id": principal.agent_id if principal else None,
            "source": principal.source if principal else "unknown",
            "admin": bool(principal and principal.admin),
            "may_write": bool(principal and principal.may_admin_write),
        }

    def admin_denial(self, request: Any) -> Denial | None:
        """THE admin authorization decision. ``None`` to proceed.

        🚨 This module owns what a scope permits, and this is the method that
        says so. A gate that resolved the principal and then decided for itself
        whether a read-only identity may POST would be a second place deciding —
        which is the thing ``identity.py`` exists to prevent, and the reason
        ``management.py``'s own ``_remote_ip`` had to be removed.

        THREE refusals, and they mean three different things:

        * a presented credential that does not resolve — 401, from
          :meth:`resolve`. Conflating it with the 403 below would tell an
          operator whose key was revoked that their *host* was not allowed,
          sending them to fix the wrong file.
        * a resolved identity without the admin scope — 403, and its message is
          byte-identical to what this surface has always returned.
        * an admin identity narrowed to reads, asking to write — 403, with its
          OWN message. It must not reuse the one above: "access denied for
          198.51.100.4" tells an operator holding a deliberately read-only
          credential to go and change an ACL, which is both wrong and, if they
          succeed, the narrowing undone.

        The read/write split is taken from the HTTP METHOD (:data:`SAFE_METHODS`)
        rather than from a list of mutating routes. A route list is a second
        thing to keep in step with ``routes.py``, and when it falls behind the
        failure is silent and in the widening direction.
        """
        resolved = self.resolve(request)
        if not resolved.ok:
            return resolved.denial
        principal = resolved.principal
        if not principal.admin:
            return Denial(
                code="access_denied",
                status=403,
                message=f"access denied for {self.client_ip(request)}",
            )
        method = str(getattr(request, "method", "GET") or "GET").upper()
        if method not in SAFE_METHODS and not principal.may_admin_write:
            return Denial(
                code="access_denied",
                status=403,
                message=(
                    f"this credential has read-only admin scope: {method} is a "
                    f"mutation and is refused, while every GET on this surface "
                    f"is allowed. The scope is a NARROWING written on the "
                    f"identity itself — widening it means issuing a different "
                    f"credential, not changing an address allowlist"),
            )
        return None


def _require_key_from_env() -> bool:
    """``ROADSTEAD_REQUIRE_API_KEY`` — refuse any request without a key.

    Default OFF. On is the right setting for a proxy reachable from anything
    wider than a host, and the wrong one for the laptop case this is local-first
    for, so it is a decision an operator makes rather than one taken for them.
    """
    return os.environ.get("ROADSTEAD_REQUIRE_API_KEY", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
