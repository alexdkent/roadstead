"""IP-based access control for LAN consumers.

Maps source IPs to agent identities for external services that use
the proxy's OpenAI-compatible endpoints.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from .config import LLMPriority
from .constants import _SMART_DEFAULT_CAP_S


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
    ``min_timeout_s()`` lookup instead.
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
        # the internal nets PLUS an explicit allow-list — deliberately SEPARATE from
        # _internal_nets so granting admin never changes a host's inference
        # PRIORITY (identify() matches _internal_nets first, before subnet
        # entries — see is_admin). anvil (10.0.0.3) is granted admin: originally
        # (2026-07-09) for the classify-evict pause/resume, but anvil also depends
        # on it for its HEALTH TELEMETRY path — a brief 2026-07-11 removal (after
        # classify re-homed off anvil) broke anvil health telemetry on the
        # Systems page, so it is KEPT. DO NOT remove even though classify eviction is
        # now inert: anvil has other admin-gated proxy dependencies.
        self._admin_nets = list(self._internal_nets) + [
            ipaddress.ip_network("10.0.0.3/32"),
        ]

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
        calls-log ingest) accept ONLY loopback + docker-internal sources.

        The generic ``identify`` ACL passes the whole LAN (the
        ``10.0.0.0/24 → lan-generic`` entry), which is fine for INFERENCE but
        let any LAN device pause a backend fleet-wide. :42161 is container-
        internal (port not published) and every legitimate admin caller goes
        through ``docker exec curl localhost`` (restart_llm.sh) or the gateway
        on loopback — verified by the admin-audit IP log before tightening
        (2026-06-10: 127.0.0.1 only). 2026-07-09: anvil (10.0.0.3) added to
        _admin_nets — needed for classify-evict AND the anvil health-telemetry
        path (removing it broke anvil telemetry on the Systems page); KEEP it
        (see __init__)."""
        try:
            addr = ipaddress.ip_address(remote_ip)
        except ValueError:
            return False
        return any(addr in net for net in self._admin_nets)

    @classmethod
    def from_env(cls) -> "IPIdentityMap":
        """Build from environment variables.

        LLM_PROXY_ACL entries are comma-separated: ``ip=agent_id[:priority]``
        e.g. ``10.0.0.9=tideway:P3_INGESTION,10.0.0.0/24=lan:P3_INGESTION``
        """
        acl = cls()
        raw = os.environ.get("LLM_PROXY_ACL", "")
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry or "=" not in entry:
                continue
            ip_part, id_part = entry.split("=", 1)
            if ":" in id_part:
                agent_id, pri_str = id_part.rsplit(":", 1)
                try:
                    priority = LLMPriority.coerce(pri_str)
                except ValueError:
                    priority = LLMPriority.P3_INGESTION
            else:
                agent_id = id_part
                priority = LLMPriority.P3_INGESTION
            acl.register(ip_part.strip(), agent_id.strip(), priority)

        # Always register known infrastructure
        acl.register("10.0.0.9", "tideway", LLMPriority.P3_INGESTION)
        acl.register("10.0.0.6", "nexus-local", LLMPriority.P3_INGESTION)
        acl.register("10.0.0.3", "anvil-local", LLMPriority.P3_INGESTION)
        # recipe-runner (Kestrel CTnnn, static 10.0.0.14). The Goose CLI's own LLM
        # provider hits the OpenAI-compat door (NEXUS_URL=…:42161/v1,
        # GOOSE_MODEL=llama-thinker) as a plain OpenAI client — no identity
        # header — so without this it fell through to `lan-generic`, hiding the
        # fleet's single highest-volume OpenAI-door caller (~9k calls, mostly
        # thinker) behind the catch-all. Identity-only fix: same P3 tier it
        # already got via the subnet default, so QoS is unchanged. (2026-07-12)
        acl.register("10.0.0.14", "recipe-runner", LLMPriority.P3_INGESTION)
        # pool-observer (Kestrel CTnnn, static 10.0.0.17). Identical situation to
        # goose above: the `pool` CLI is a plain OpenAI-compat client with no
        # identity header, so without this it lands in `lan-generic` and its
        # traffic is invisible in `proxy_completions`. Registered in CODE, not
        # via LLM_PROXY_ACL — container env is baked at `docker run`, so an env
        # change would force a permission-gated full-fleet re-run for what is a
        # one-line identity fix. Same P3 tier the subnet default already gave
        # it, so QoS is unchanged; the point is attribution. tier3 peaks at
        # 11/20 slots, so pool must stay deprioritizable. (2026-08-02)
        #
        # min_timeout_s=1800: pool supplies NO deadline, so it gets the smart
        # default — floor(thinker 180s) x surge x size_stretch, and the stretch
        # CLAMPS at 3.0, giving a flat 540s wall for any tier3 prompt above
        # ~82K tokens. Measured 2026-08-03: three consecutive kills at
        # elapsed_s=539.999 against applied_timeout_s=540.0 on a 123,466-token
        # prompt. A `timeout_ceiling_s` in models.yaml cannot fix this — the
        # stretch clamp binds long before any ceiling is reached, so the lever
        # has to be a FLOOR. 1800s == _SMART_DEFAULT_CAP_S, the cap the smart
        # default is already allowed to reach, so this raises pool to the
        # existing ceiling rather than inventing a new bound. Extend-only and
        # default-only: an explicit caller deadline still wins. (2026-08-03)
        acl.register("10.0.0.17", "pool-observer", LLMPriority.P3_INGESTION,
                     min_timeout_s=_SMART_DEFAULT_CAP_S)
        # lan-generic carries the SAME floor, and that is a deliberate blunt
        # instrument, not an oversight: the mac dev host runs `pool` too and
        # lands here (only the Kestrel CT has a static registration), so flooring
        # only 10.0.0.17 would leave the dev host strangled at 540s. This is the
        # fast unblock pending the real fix in the adaptive timeout model (the
        # size_stretch clamp). Whoever narrows this later: the correct end state
        # is that the model stops emitting a deadline shorter than the work
        # takes, at which point this subnet-wide floor should go away entirely —
        # it currently hands every un-registered LAN client a 30-minute deadline
        # when it omits timeout_s. (2026-08-03)
        acl.register("10.0.0.0/24", "lan-generic", LLMPriority.P3_INGESTION,
                     min_timeout_s=_SMART_DEFAULT_CAP_S)

        return acl
