"""IP-based access control for LAN consumers.

Maps source IPs to agent identities for external services that use
the proxy's OpenAI-compatible endpoints.
"""

from __future__ import annotations

import ipaddress
import logging
import os

logger = logging.getLogger(__name__)

from .config import LLMPriority


class IPIdentityMap:
    """Maps source IPs to (agent_id, default_priority) for LAN consumers."""

    def __init__(self) -> None:
        self._exact: dict[str, tuple[str, LLMPriority]] = {}
        self._subnets: list[tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, str, LLMPriority]] = []

        # Docker/loopback always allowed as internal
        self._internal_nets = [
            ipaddress.ip_network("127.0.0.0/8"),
            ipaddress.ip_network("172.16.0.0/12"),
            ipaddress.ip_network("::1/128"),
        ]

    def register(
        self,
        ip_or_subnet: str,
        agent_id: str,
        priority: LLMPriority = LLMPriority.P3_INGESTION,
    ) -> None:
        try:
            net = ipaddress.ip_network(ip_or_subnet, strict=False)
            if net.prefixlen == net.max_prefixlen:
                self._exact[str(net.network_address)] = (agent_id, priority)
            else:
                self._subnets.append((net, agent_id, priority))
        except ValueError:
            logger.warning("invalid IP/subnet in ACL: %s", ip_or_subnet)

    def identify(self, remote_ip: str) -> tuple[str, LLMPriority] | None:
        """Returns (agent_id, default_priority) for the given IP, or None
        if the IP is not registered."""
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
                return ("internal", LLMPriority.P1_TURN_SUPPORT)

        # Check subnet matches
        for net, agent_id, priority in self._subnets:
            if addr in net:
                return (agent_id, priority)

        return None

    def is_allowed(self, remote_ip: str) -> bool:
        return self.identify(remote_ip) is not None

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
        acl.register("10.0.0.0/24", "lan-generic", LLMPriority.P3_INGESTION)

        return acl
