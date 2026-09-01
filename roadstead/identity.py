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
