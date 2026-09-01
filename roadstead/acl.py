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

    def register_admin_net(self, ip_or_subnet: str) -> None:
        """Extend the admin allow-list. Identity and PRIORITY are untouched —
        granting admin to a host must never quietly promote its inference
        traffic into a better band."""
        try:
            self._admin_nets.append(ipaddress.ip_network(ip_or_subnet, strict=False))
        except ValueError:
            logger.warning("invalid IP/subnet in admin nets: %s", ip_or_subnet)

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

    def is_admin(self, remote_ip: str) -> bool:
        """Admin surfaces (drain/pause, maintenance windows, runtime flags,
        calls-log ingest) accept loopback + docker-internal sources, plus
        anything ``ROADSTEAD_ADMIN_NETS`` adds.

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
        return any(addr in net for net in self._admin_nets)

    @classmethod
    def from_env(cls) -> "IPIdentityMap":
        """Build from the environment. Ships no registrations of its own.

        ``ROADSTEAD_ACL`` entries are comma-separated
        ``ip_or_subnet=agent_id[:priority][:min_timeout_s][:admin]``, e.g.::

            ROADSTEAD_ACL=192.0.2.9=tideway:P3_INGESTION,192.0.2.0/24=lan:P3_INGESTION:1800

        The right-hand side is the same grammar API keys use
        (``identity.parse_identity_spec``) — segments are recognised by shape,
        so their order does not matter and any of them may be omitted. ``admin``
        on an entry ALSO adds that address to the admin nets, which is the one
        place the two decisions are made together, because an operator writing
        ``=ops:admin`` plainly means both.

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
                            "<ip-or-subnet>=<agent_id>[:priority][:min_timeout_s][:admin])",
                            var, entry)
                    continue
                ip_part, spec = entry.split("=", 1)
                agent_id, priority, floor, admin = parse_identity_spec(spec.strip())
                ip_part = ip_part.strip()
                acl.register(ip_part, agent_id, priority, min_timeout_s=floor)
                if admin:
                    acl.register_admin_net(ip_part)

        for net in os.environ.get("ROADSTEAD_ADMIN_NETS", "").split(","):
            net = net.strip()
            if net:
                acl.register_admin_net(net)

        return acl
