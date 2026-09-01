"""Address-based access control — the SECOND factor, behind API keys.

Maps a source IP to a caller identity. This was the proxy's only identity
mechanism until 2026-09-01, when ``identity.py`` made an API key the primary
one; an address now fills in an identity that no credential established.

🚨 **It ships no addresses.** Until 2026-09-01 this file carried a private
fleet's LAN registrations — ten hosts by address, role and purpose, compiled
into the package (scrub item S1, ``docs/corpus_and_scrub_plan.md``). They are
gone, and the environment is now the ONLY way to register one. Two spellings for
one registration is how they come to disagree, and a shipped default that
happens to match somebody's LAN is worse than no default at all: it hands an
identity — and a DRR share — to whatever answers at an address we guessed.

What that means for a fresh install: **loopback and docker-internal are allowed;
everything else is refused until enrolled.** Default-deny is the right posture
for a door that hands out capacity, and the local-first case (a proxy on the
machine that calls it) works with no configuration at all.

The registrations that were here are not lost — the *reasoning* in them was
about shapes, not addresses, and the two that generalise now live where they can
apply to anybody's config: the deadline floor for a caller that supplies no
deadline of its own (``Lifecycle.handle_submit``), and the refusal to floor an
interactive caller above its own ceiling (``identity._warn_if_floor_exceeds_ceiling``).
"""

from __future__ import annotations

import ipaddress
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from .config import LLMPriority
from .identity import parse_identity_spec


@dataclass(frozen=True)
class _Registration:
    """One ACL entry. ``min_timeout_s`` is the per-identity MINIMUM deadline
    floor (seconds) applied ONLY when the caller supplied no deadline of its
    own — see ``IPIdentityMap.min_timeout_s`` and ``Lifecycle.handle_submit``.
    ``None`` (the default) means "no floor", i.e. behaviour unchanged."""

    agent_id: str
    priority: LLMPriority
    min_timeout_s: float | None = None


class IPIdentityMap:
    """Maps source IPs to (agent_id, default_priority) for LAN consumers.

    Registrations are stored as ``_Registration`` records, but ``identify()``
    deliberately still returns the 2-tuple ``(agent_id, priority)``: widening
    it would churn every call site and every existing assertion for a field
    only ONE code path cares about. The floor is read through its own
    ``min_timeout_s()`` lookup instead, and the whole record through
    ``identity.IdentityResolver``, which is what the request path now uses.
    """

    def __init__(self) -> None:
        self._exact: dict[str, _Registration] = {}
        self._subnets: list[tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, _Registration]] = []

        # Docker/loopback always allowed as internal
        self._internal_nets = [
            ipaddress.ip_network("127.0.0.0/8"),
            ipaddress.ip_network("172.16.0.0/12"),
            ipaddress.ip_network("::1/128"),
        ]
        # Admin surfaces (drain/pause, maintenance, flags, calls-log ingest) accept
        # the internal nets PLUS whatever ``ROADSTEAD_ADMIN_NETS`` adds —
        # deliberately SEPARATE from _internal_nets so granting admin never
        # changes a host's inference PRIORITY (identify() matches _internal_nets
        # first, before subnet entries — see is_admin).
        #
        # The list ships with the internal nets only. It previously carried one
        # fleet host by address, granted for a control-plane call and then kept
        # because a health-telemetry path turned out to depend on it too — a good
        # illustration of why an admin grant is an operator decision and not a
        # package default: nobody could tell, from here, what would break by
        # removing it. An API key with the ``admin`` scope is now the better
        # answer; this stays for the pre-key deployment shape.
        self._admin_nets = list(self._internal_nets)
        # 🚨 What the OPERATOR added, kept apart from the built-ins above.
        # The two are the same list until somebody configures a reverse proxy,
        # at which point they stop meaning the same thing: "came from loopback"
        # is a statement about the box when the connection is direct and a
        # statement about nothing at all when a front proxy made it. A forwarded
        # address is checked against THIS list only — see
        # ``identity.IdentityResolver.resolve``.
        self._operator_admin_nets: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        # 🚨 Nets whose admin grant is READ-ONLY. A subset of the two lists
        # above, never a third source of grants: an address reaches this list
        # only by having been registered as admin in the same breath, so
        # membership here can only ever take the mutating half away. See
        # :meth:`is_admin_readonly` for why matching it WINS over a full grant.
        self._readonly_admin_nets: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        # 🚨 The admin-plane REACH set — see :meth:`may_reach_admin`. Loopback
        # is unconditional (lockout), docker-internal is a default that naming
        # any operator net drops.
        self._loopback_nets = [ipaddress.ip_network("127.0.0.0/8"),
                               ipaddress.ip_network("::1/128")]
        self._docker_nets = [ipaddress.ip_network("172.16.0.0/12")]
        self._reach_nets = list(self._loopback_nets) + list(self._docker_nets)

    def _rebuild_reach(self) -> None:
        """Called whenever an operator net is registered. Naming one drops the
        docker default; loopback survives everything."""
        nets = list(self._loopback_nets)
        if self._operator_admin_nets:
            nets += [n for n in self._operator_admin_nets if n not in self._loopback_nets]
        else:
            nets += self._docker_nets
        self._reach_nets = nets

    def register(
        self,
        ip_or_subnet: str,
        agent_id: str,
        priority: LLMPriority = LLMPriority.P3_INGESTION,
        min_timeout_s: float | None = None,
    ) -> None:
        reg = _Registration(agent_id, priority, min_timeout_s)
        try:
            net = ipaddress.ip_network(ip_or_subnet, strict=False)
            if net.prefixlen == net.max_prefixlen:
                self._exact[str(net.network_address)] = reg
            else:
                self._subnets.append((net, reg))
        except ValueError:
            logger.warning("invalid IP/subnet in ACL: %s", ip_or_subnet)

    def register_admin_net(self, ip_or_subnet: str, *, readonly: bool = False) -> None:
        """Extend the admin allow-list. Identity and PRIORITY are untouched —
        granting admin to a host must never quietly promote its inference
        traffic into a better band.

        ``readonly=True`` grants the admin scope and immediately narrows it to
        reads. It is one call rather than two because the two facts arrive
        together, from one ``=ops:admin:readonly`` entry, and a host that ended
        up in the readonly list without being in an admin list at all would be
        a grant nobody wrote.
        """
        try:
            net = ipaddress.ip_network(ip_or_subnet, strict=False)
        except ValueError:
            logger.warning("invalid IP/subnet in admin nets: %s", ip_or_subnet)
            return
        self._admin_nets.append(net)
        self._operator_admin_nets.append(net)
        if readonly:
            self._readonly_admin_nets.append(net)
        # The reach set is derived, never appended to directly: naming an
        # operator net drops the docker default, and that only works if every
        # registration recomputes rather than accumulates.
        self._rebuild_reach()

    def _lookup(self, remote_ip: str) -> _Registration | None:
        """Resolve an IP to its registration, or None if unregistered.
        Match order is exact → internal nets → subnets (see ``is_admin``:
        the internal nets are matched BEFORE subnet entries deliberately)."""
        # Check exact matches first
        match = self._exact.get(remote_ip)
        if match:
            return match

        # Check if internal (docker/loopback) — these are container-local agents
        try:
            addr = ipaddress.ip_address(remote_ip)
        except ValueError:
            return None

        for net in self._internal_nets:
            if addr in net:
                return _Registration("internal", LLMPriority.P1_TURN_SUPPORT)

        # Check subnet matches
        for net, reg in self._subnets:
            if addr in net:
                return reg

        return None

    def entries(self) -> dict[str, dict]:
        """Every OPERATOR REGISTRATION, address → what it resolves to.

        For the management plane's per-caller view (roadmap E), which needs to
        show that a caller is reachable by address as well as by key.

        🚨 Reports only what an operator REGISTERED. The built-in internal nets
        are deliberately absent: they resolve to the ``internal`` identity in
        ``identify()`` without appearing here, and listing them as though they
        were registrations would tell an operator that removing a line from
        ``ROADSTEAD_ACL`` closes a door that is in fact built in.
        """
        out: dict[str, dict] = {}
        for address, reg in self._exact.items():
            out[address] = {"agent_id": reg.agent_id,
                            "priority": reg.priority.name,
                            "min_timeout_s": reg.min_timeout_s}
        for net, reg in self._subnets:
            out[str(net)] = {"agent_id": reg.agent_id,
                             "priority": reg.priority.name,
                             "min_timeout_s": reg.min_timeout_s}
        return out

    def identify(self, remote_ip: str) -> tuple[str, LLMPriority] | None:
        """Returns (agent_id, default_priority) for the given IP, or None
        if the IP is not registered."""
        reg = self._lookup(remote_ip)
        return None if reg is None else (reg.agent_id, reg.priority)

    def min_timeout_s(self, remote_ip: str) -> float | None:
        """Per-identity MINIMUM deadline floor in seconds, or None when this
        identity has no floor configured (the common case — an unregistered or
        floor-less IP must behave exactly as it did before this existed).

        Applied ONLY to a deadline the proxy chose itself: a caller that
        supplies its own ``timeout_s``/``X-Timeout-S`` stays authoritative.
        See ``Lifecycle.handle_submit``."""
        reg = self._lookup(remote_ip)
        return None if reg is None else reg.min_timeout_s

    def is_allowed(self, remote_ip: str) -> bool:
        return self.identify(remote_ip) is not None

    def builtin_admin_nets(self) -> list[str]:
        """The nets admin is granted to WITHOUT an operator saying so.

        Reported by the management plane beside :meth:`operator_admin_nets`,
        because which of the two a grant came from decides whether it survives a
        reverse proxy being put in front (see ``is_admin``).
        """
        return [str(net) for net in self._internal_nets]

    def operator_admin_nets(self) -> list[str]:
        """The nets an operator added, via ``ROADSTEAD_ADMIN_NETS`` or ``:admin``."""
        return [str(net) for net in self._operator_admin_nets]

    @property
    def readonly_admin_nets(self) -> list[str]:
        """The subset of those whose grant is narrowed to reads."""
        return [str(net) for net in self._readonly_admin_nets]

    def may_reach_admin(self, remote_ip: str, *, trust_builtin_nets: bool = True) -> bool:
        """🚨 THE NETWORK GATE on the admin plane. An address may REACH it; no
        address GRANTS it. A credential is required regardless of this answer.

        This is the half of the 2026-09-01 change that is easy to get backwards.
        Until then, ``is_admin`` below answered "is this address an admin?" and
        the answer alone was enough to reconfigure the fleet — so a request from
        loopback with no credential at all could flip a runtime flag, and the
        audit trail recorded it as ``key_id: null``. That default was written
        for the INFERENCE door, where "already on the box" is a fair proxy for
        "allowed", and it was inherited by the CONTROL door when the admin plane
        and then the UI were added on the same port. Nobody re-asked whether
        being on the box should also mean being allowed to read every caller's
        traffic and change policy.

        So the two questions are now separate and BOTH must pass:

        * *may this address reach the admin plane at all* — here, and
        * *does this credential carry the admin scope* — the key registry, via
          ``identity.IdentityResolver.admin_denial``.

        Loopback is ALWAYS in the reach set and cannot be configured out. It is
        where the startup-minted bootstrap key is usable, so removing it is a
        lockout with no recovery that does not involve editing the environment
        and restarting. Docker-internal is in the set by DEFAULT — a
        containerised deployment reaches its own admin plane over the bridge —
        but naming any net in ``ROADSTEAD_ADMIN_NETS`` drops it, because a /12
        is a weak gate and an operator who has named their own nets has said
        what they want.
        """
        try:
            addr = ipaddress.ip_address(remote_ip)
        except ValueError:
            return False
        nets = self._reach_nets if trust_builtin_nets else (
            self._operator_admin_nets + self._loopback_nets)
        return any(addr in net for net in nets)

    def reach_nets(self) -> list[str]:
        """The effective admin-plane reach set, for the management plane and the
        startup log. Reported rather than inferred: an operator who cannot see
        it has no way to tell a 403 from a typo in a CIDR."""
        return [str(net) for net in self._reach_nets]

    def is_admin(self, remote_ip: str, *, trust_builtin_nets: bool = True) -> bool:
        """Admin surfaces (drain/pause, maintenance windows, runtime flags,
        calls-log ingest) accept loopback + docker-internal sources, plus
        anything ``ROADSTEAD_ADMIN_NETS`` adds.

        🚨 ``trust_builtin_nets=False`` drops the loopback/docker grant and
        honours only what the operator registered. The request path passes it
        for an address that arrived via ``X-Forwarded-For``: the built-in grant
        assumes reaching loopback meant already being on the machine, and a
        reverse proxy is precisely the thing that makes that untrue. Narrowing
        it is safe to do unconditionally *because* it is gated on trusted-proxy
        configuration, which is empty until an operator opts in — so no existing
        deployment can lose an admin grant it has today.

        Deliberately NARROWER than ``identify``: an operator who enrols a whole
        LAN subnet for inference has said nothing about who may pause a backend
        fleet-wide, and the two decisions must not be the same decision. In the
        origin deployment the admin port was container-internal and every
        legitimate caller arrived on loopback; an audit of real admin source IPs
        before the surface was tightened found exactly that.

        🚨 This is the address answer to a question API keys answer better. When
        a key is presented, ``identity.IdentityResolver.is_admin`` reads the
        key's ``admin`` scope and does NOT consult this at all — see its
        docstring for why an authenticated non-admin must not inherit its host's
        privileges."""
        try:
            addr = ipaddress.ip_address(remote_ip)
        except ValueError:
            return False
        nets = self._admin_nets if trust_builtin_nets else self._operator_admin_nets
        return any(addr in net for net in nets)

    def is_admin_readonly(self, remote_ip: str) -> bool:
        """Whether this address's admin grant is narrowed to READS.

        🚨 Matching a read-only net WINS over also matching a full-admin one,
        including a built-in. The alternative — full wins — makes
        ``127.0.0.1=ops:admin:readonly`` silently a full grant, because loopback
        is a built-in admin net and every operator writing that line is on it.
        A narrowing that the widest overlapping grant can cancel is not a
        narrowing; it is a comment.

        Unlike :meth:`is_admin` this takes no ``trust_builtin_nets``: the
        built-ins grant admin and never narrow it, so there is nothing here for
        a forwarded address to lose.
        """
        try:
            addr = ipaddress.ip_address(remote_ip)
        except ValueError:
            return False
        return any(addr in net for net in self._readonly_admin_nets)

    @classmethod
    def from_env(cls) -> "IPIdentityMap":
        """Build from the environment. Ships no registrations of its own.

        ``ROADSTEAD_ACL`` entries are comma-separated
        ``ip_or_subnet=agent_id[:priority][:min_timeout_s][:admin][:readonly]``, e.g.::

            ROADSTEAD_ACL=192.0.2.9=ingest-worker:P3_INGESTION,192.0.2.0/24=lan:P3_INGESTION:1800

        The right-hand side is the same grammar API keys use
        (``identity.parse_identity_spec``) — segments are recognised by shape,
        so their order does not matter and any of them may be omitted. ``admin``
        on an entry ALSO adds that address to the admin nets, which is the one
        place the two decisions are made together, because an operator writing
        ``=ops:admin`` plainly means both. ``:readonly`` beside it narrows that
        grant to reads.

        ``ROADSTEAD_ADMIN_NETS`` is a comma-separated CIDR list for hosts that
        need the control plane but no inference identity.

        ``LLM_PROXY_ACL`` is still read, for the pre-rename deployments. It is
        the ONE legacy spelling kept here, and only because it configures access
        — a proxy that silently stops recognising its callers on upgrade fails
        closed in the most confusing possible way. Both are read when both are
        set; ``ROADSTEAD_ACL`` is applied second and therefore wins a collision.
        """
        acl = cls()
        for var in ("LLM_PROXY_ACL", "ROADSTEAD_ACL"):
            raw = os.environ.get(var, "")
            if raw and var == "LLM_PROXY_ACL":
                logger.warning(
                    "LLM_PROXY_ACL is the pre-rename spelling; it is still "
                    "honoured, but rename it to ROADSTEAD_ACL.")
            for entry in raw.split(","):
                entry = entry.strip()
                if not entry or "=" not in entry:
                    if entry:
                        logger.warning(
                            "%s: skipping malformed entry %r (expected "
                            "<ip-or-subnet>=<agent_id>[:priority]"
                            "[:min_timeout_s][:admin][:readonly])",
                            var, entry)
                    continue
                ip_part, spec = entry.split("=", 1)
                agent_id, priority, floor, admin, readonly = parse_identity_spec(
                    spec.strip())
                ip_part = ip_part.strip()
                acl.register(ip_part, agent_id, priority, min_timeout_s=floor)
                if admin:
                    acl.register_admin_net(ip_part, readonly=readonly)

        for net in os.environ.get("ROADSTEAD_ADMIN_NETS", "").split(","):
            net = net.strip()
            if net:
                acl.register_admin_net(net)

        return acl
