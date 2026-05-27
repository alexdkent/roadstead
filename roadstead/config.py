"""Configuration dataclasses for the LLM proxy.

All proxy behaviour is driven by these structures.  Slot counts and
context sizes come from backend ``/props`` discovery at runtime — the
config only carries *policy* knobs (weights, floors, timeouts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


# ---------------------------------------------------------------------------
# Priority — re-exported here so proxy internals don't import framework
# ---------------------------------------------------------------------------

class LLMPriority(IntEnum):
    P0_REALTIME = 0
    P1_TURN_SUPPORT = 1
    P2_POST_TURN = 2
    P3_INGESTION = 3
    P4_HYGIENE = 4

    @classmethod
    def coerce(cls, value: "LLMPriority | str | int | None") -> "LLMPriority":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.P1_TURN_SUPPORT
        if isinstance(value, int):
            return cls(value)
        text = str(value).strip()
        if text in cls.__members__:
            return cls[text]
        lowered = text.lower()
        aliases = {
            "realtime": cls.P0_REALTIME,
            "turn": cls.P1_TURN_SUPPORT,
            "turn_support": cls.P1_TURN_SUPPORT,
            "post_turn": cls.P2_POST_TURN,
            "ingestion": cls.P3_INGESTION,
            "hygiene": cls.P4_HYGIENE,
            "background": cls.P3_INGESTION,
        }
        if lowered in aliases:
            return aliases[lowered]
        raise ValueError(f"unknown LLM priority {value!r}")


class PriorityBand(IntEnum):
    INTERACTIVE = 0   # P0_REALTIME, P1_TURN_SUPPORT
    FOREGROUND = 1    # P2_POST_TURN
    BACKGROUND = 2    # P3_INGESTION, P4_HYGIENE


def priority_to_band(p: LLMPriority) -> PriorityBand:
    if p <= LLMPriority.P1_TURN_SUPPORT:
        return PriorityBand.INTERACTIVE
    if p <= LLMPriority.P2_POST_TURN:
        return PriorityBand.FOREGROUND
    return PriorityBand.BACKGROUND


# ---------------------------------------------------------------------------
# Role → endpoint-class mapping (mirrors framework/llm_qos.py)
# ---------------------------------------------------------------------------

ROLE_TO_CLASS: dict[str, str] = {
    "qwen-analyst":  "chat",
    "qwen-composer": "companion",
    "gemma-router":  "gemma",
    "gemma-greeter": "gemma-hot",
    "bge-reranker":  "rerank",
    "bge-m3-embed":  "embed",
    "llama-thinker": "thinker",
}

CLASS_TO_ROLE: dict[str, str] = {v: k for k, v in ROLE_TO_CLASS.items()}


def normalize_endpoint(endpoint: str) -> str:
    """Collapse a role name to its QoS endpoint class."""
    text = endpoint.strip().lower()
    if text.startswith("nexus-"):
        text = text.split("-", 1)[1]
    if text == "nexus":
        text = "chat"
    if text in ROLE_TO_CLASS:
        text = ROLE_TO_CLASS[text]
    return text


# ---------------------------------------------------------------------------
# Endpoint configuration
# ---------------------------------------------------------------------------

@dataclass
class EndpointConfig:
    """Per-endpoint-class operational config.

    ``max_slots`` and ``context_per_slot`` are populated at runtime from
    backend ``/props`` discovery.  Everything else is policy.
    """
    endpoint_class: str
    role: str

    # --- discovered at runtime (mutable) ---
    max_slots: int = 0
    context_per_slot: int = 0

    # --- policy knobs ---
    min_expected_slots: int = 1
    background_floor_pct: float = 0.20
    default_timeout_s: float = 180.0

    # --- backend connection ---
    host: str = ""
    port: int = 0

    @property
    def background_floor_slots(self) -> int:
        return max(1, int(self.max_slots * self.background_floor_pct))


# ---------------------------------------------------------------------------
# Agent quota configuration
# ---------------------------------------------------------------------------

@dataclass
class AgentQuotaConfig:
    """Per-agent DRR parameters."""
    agent_id: str
    weight: float = 1.0
    max_balance_ss: float = 60.0
    default_priority: LLMPriority = LLMPriority.P1_TURN_SUPPORT


# ---------------------------------------------------------------------------
# Top-level proxy config
# ---------------------------------------------------------------------------

DEFAULT_ENDPOINTS: dict[str, EndpointConfig] = {
    "chat": EndpointConfig(
        endpoint_class="chat", role="qwen-analyst",
        max_slots=4, context_per_slot=4096,
        host="10.0.0.6", port=8080,
    ),
    "companion": EndpointConfig(
        endpoint_class="companion", role="qwen-composer",
        max_slots=2, context_per_slot=65536,
        host="10.0.0.3", port=9082,
    ),
    "gemma": EndpointConfig(
        endpoint_class="gemma", role="gemma-router",
        max_slots=2, context_per_slot=8192,
        host="10.0.0.3", port=9091,
    ),
    "gemma-hot": EndpointConfig(
        endpoint_class="gemma-hot", role="gemma-greeter",
        max_slots=2, context_per_slot=2048,
        host="10.0.0.3", port=9090,
    ),
    "rerank": EndpointConfig(
        endpoint_class="rerank", role="bge-reranker",
        max_slots=1, context_per_slot=512,
        background_floor_pct=0.0,
        host="10.0.0.3", port=9084,
    ),
    "embed": EndpointConfig(
        endpoint_class="embed", role="bge-m3-embed",
        max_slots=4, context_per_slot=8192,
        host="10.0.0.3", port=9087,
    ),
    "thinker": EndpointConfig(
        endpoint_class="thinker", role="llama-thinker",
        max_slots=4, context_per_slot=32768,
        host="10.0.0.6", port=8084,
    ),
}


@dataclass
class ProxyConfig:
    """Top-level proxy configuration."""
    port: int = 42161
    endpoints: dict[str, EndpointConfig] = field(default_factory=lambda: dict(DEFAULT_ENDPOINTS))
    agents: dict[str, AgentQuotaConfig] = field(default_factory=dict)
    starvation_timeout_s: float = 30.0
    drr_tick_interval_s: float = 0.01
    queue_db_path: str = ""
    stats_db_path: str = ""
    request_log_path: str = ""

    @property
    def total_fleet_slots(self) -> int:
        return sum(ep.max_slots for ep in self.endpoints.values())

    def agent_config(self, agent_id: str) -> AgentQuotaConfig:
        if agent_id not in self.agents:
            self.agents[agent_id] = AgentQuotaConfig(agent_id=agent_id)
        return self.agents[agent_id]
