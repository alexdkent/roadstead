"""Configuration dataclasses for the LLM proxy.

All proxy behaviour is driven by these structures.  Slot counts and
context sizes come from backend ``/props`` discovery at runtime — the
config only carries *policy* knobs (weights, floors, timeouts).

``DEFAULT_ENDPOINTS`` below is THE canonical role→host:port map for the
whole system. The LLM proxy is the single front door to all LLM traffic
(10.0.0.3 anvil / 10.0.0.6 nexus); every agent reaches a backend via
``make_nexus_client(role)`` → ``ProxyLLMClient`` → ``:42161`` → here.
There is no other routing table. The ``infra/inference/profiles/*.yaml``
files mirror this for the inferctl/profile-transition *operational*
tooling only — when they disagree, this file wins.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
    # Slots held back from the BACKGROUND band so an occasional interactive /
    # fast-path call always has an open slot (no preemption exists). When set,
    # background may use ``max_slots - fast_path_reserve_slots`` concurrently —
    # scaling automatically if max_slots grows (6→8→10). 0 → legacy behavior
    # (background capped at its floor). Intra-band fairness across agents is
    # handled separately by the DRR scheduler, not this cap.
    fast_path_reserve_slots: int = 0

    # Hard ceiling on CONCURRENT dispatch to this endpoint, independent of the
    # discovered physical slot count. 0 -> use max_slots (no extra cap). Set
    # BELOW max_slots for a concurrency-fragile backend to leave crash-headroom:
    # the companion 80B (Strix Halo Vulkan, patched-cache) aborted under full
    # 4-slot pressure with a llama.cpp KV-seq-removal assertion (2026-06-01, G1),
    # so it runs at 3 to keep one slot of headroom and lower the peak
    # seq-management concurrency. Caps DISPATCH only; capacity discovery and
    # reporting still use max_slots.
    dispatch_concurrency_cap: int = 0

    # --- backend connection ---
    host: str = ""
    port: int = 0

    # --- served model id (the name the backend answers to in the `model`
    # field). Discovered from /v1/models at runtime; empty until then.
    # vLLM validates this field and 404s on a mismatch, so the proxy sets
    # it to ``effective_model_id`` before dispatching to a vLLM backend. ---
    served_model_id: str = ""

    # --- shadow backend (A/B testing) ---
    shadow_host: str = ""
    shadow_port: int = 0

    # --- backend engine: "llama.cpp" (top-level `grammar`) or "vllm"
    # (structured_outputs.grammar). Controls proxy-side payload normalization. ---
    backend_engine: str = "llama.cpp"

    # When True, the proxy injects ``id_slot`` into chat_completion payloads
    # for INTERACTIVE-band requests, keyed by a deterministic hash of
    # session_id. This pins each conversation to one llama.cpp slot across
    # turns so the patched KV-cache checkpoint is reused instead of
    # re-prefilling from scratch. Only useful for multi-slot llama.cpp
    # backends (not vLLM). Off by default; enabled on companion.
    slot_affinity: bool = False

    @property
    def effective_max_slots(self) -> int:
        """Concurrency ceiling the scheduler dispatches against: max_slots,
        clamped to dispatch_concurrency_cap when that is set (>0). Capacity
        reporting/discovery keep using max_slots; only admission is capped."""
        if self.dispatch_concurrency_cap > 0:
            return min(self.max_slots, self.dispatch_concurrency_cap)
        return self.max_slots

    @property
    def background_floor_slots(self) -> int:
        return max(1, int(self.max_slots * self.background_floor_pct))

    @property
    def background_cap_slots(self) -> int:
        """Max concurrent background-band requests. With fast_path_reserve_slots
        set, background may use all but that many slots (leaving headroom for
        the occasional fast-path call) — scaling automatically with max_slots.
        Legacy default (reserve 0): the floor doubles as the cap. Never below
        the floor. Computed against effective_max_slots so a concurrency cap
        (G1) shrinks the background ceiling too, preserving the interactive
        reserve (e.g. companion effective=3, reserve=1 -> background cap 2)."""
        if self.fast_path_reserve_slots > 0 and self.effective_max_slots > 0:
            cap = self.effective_max_slots - self.fast_path_reserve_slots
        else:
            cap = self.background_floor_slots
        return max(self.background_floor_slots, cap)

    @property
    def effective_model_id(self) -> str:
        """The model name to send to the backend: the discovered served id,
        falling back to ``role`` (which equals the served id for the current
        vLLM thinker, so this is correct even before discovery runs)."""
        return self.served_model_id or self.role


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
        # nexus llama-vision — Qwen3-VL-30B-A3B (served id chat.gguf) on :8080.
        # Live: --ctx-size 65536 --parallel 2 → 32768/slot, 2 slots. Serves the
        # qwen-analyst role (classify / situation reports) + vision.
        endpoint_class="chat", role="qwen-analyst",
        max_slots=2, context_per_slot=32768,
        host="10.0.0.6", port=8080,
    ),
    "companion": EndpointConfig(
        # nexus llama-companion — Huihui Qwen3-Next-80B-A3B abliterated
        # (served id companion.gguf) on :8081. Live: --ctx-size 393216
        # --parallel 4 → 98304/slot, 4 slots. The fleet's high-capability
        # summarizer/composer (qwen-composer role). /props discovery is
        # unreliable on the 80B, so this seed is load-bearing — keep it exact.
        endpoint_class="companion", role="qwen-composer",
        max_slots=4, context_per_slot=98304,
        # G1 (2026-06-01): cap concurrent dispatch at 3 (of 4 physical slots).
        # The patched-cache 80B aborted under full 4-slot pressure with a
        # llama.cpp KV-seq-removal assertion; one slot of headroom lowers the
        # peak seq-management concurrency that triggers it. Capacity discovery
        # still sees 4 slots. Override via this field if the backend is patched.
        dispatch_concurrency_cap=3,
        # Reserve 1 slot for interactive/chat rounds; background (composer
        # summaries + knowledge ingestion) uses the other 3. Mirrors the
        # thinker's fast_path_reserve pattern. background_floor_pct left at the
        # default (0.20 → floor 1) so the floor never overrides the reserve:
        # background_cap = max(floor, max_slots - reserve) = max(1, 3) = 3.
        fast_path_reserve_slots=1,
        slot_affinity=True,
        host="10.0.0.6", port=8081,
    ),
    "gemma": EndpointConfig(
        # anvil Gemma-4-E4B (cold classifier + vision) :9091 — n_ctx
        # 16384/slot, 2 slots.
        endpoint_class="gemma", role="gemma-router",
        max_slots=2, context_per_slot=16384,
        host="10.0.0.3", port=9091,
    ),
    "gemma-hot": EndpointConfig(
        # anvil Gemma-4-E2B (greeter / tier-1.5 router) :9090 — n_ctx
        # 4096/slot, 2 slots.
        endpoint_class="gemma-hot", role="gemma-greeter",
        max_slots=2, context_per_slot=4096,
        host="10.0.0.3", port=9090,
    ),
    "rerank": EndpointConfig(
        # anvil bge-reranker shim :9084 (infinity backend behind it on :9085)
        # — n_ctx 8192/slot, 1 slot.
        endpoint_class="rerank", role="bge-reranker",
        max_slots=1, context_per_slot=8192,
        background_floor_pct=0.0,
        host="10.0.0.3", port=9084,
    ),
    "embed": EndpointConfig(
        # anvil bge-m3 embed :9087 — n_ctx 8192/slot, 4 slots (CANONICAL
        # fleet embed; nexus's local copy on :8091 is being removed).
        endpoint_class="embed", role="bge-m3-embed",
        max_slots=4, context_per_slot=8192,
        host="10.0.0.3", port=9087,
    ),
    "thinker": EndpointConfig(
        # vLLM-NVFP4 on GB10 (Qwen3.6-27B-Text-NVFP4-MTP since 2026-05-30).
        # max_slots=32 MATCHES vLLM's --max-num-seqs 32 — the throughput knee
        # from the autonomous tuning sweep at backend gpu-memory-utilization 0.50
        # (0 preemptions to 32; KV ~24-66%; >32 has sharply diminishing aggregate
        # for much worse per-stream latency). Decode is memory-bandwidth-bound, so
        # the backend runs MTP speculative decoding (qwen3_5_mtp n=3) which is the
        # real per-stream lever (+34-85%, composes with the guidance grammar path).
        # Remeasure (vllm:num_preemptions_total, kv_cache_usage_perc) and pull back
        # max-num-seqs/max_slots together if thrashing appears.
        endpoint_class="thinker", role="llama-thinker",
        # context_per_slot is the STARTUP SEED only — the capacity poller
        # auto-discovers the live value from vLLM /v1/models max_model_len
        # (service.py _update_endpoint_health), so this can't get ahead of the
        # backend. 2026-05-31: raised 32768 → 131072 to match the vLLM
        # --max-model-len 128K bump (opencode large-context coding). Zero extra
        # memory — the fp8 KV pool (~647K tokens @ gpu-mem 0.50) is util-driven,
        # not max-model-len-driven; 128K is well within the model's 256K native
        # context (no rope-scaling). Fleet chunkers stay at 32K on purpose
        # (framework _DEFAULT_CONTEXT_WINDOW) to keep extraction chunks small.
        max_slots=32, context_per_slot=131072,
        # ~95% of thinker load is background (knowledge ingestion, forum-agent
        # proposals, hygiene), so let the background band use all but one slot
        # (max_slots - 1); DRR keeps that fair across agents. The single
        # reserved slot leaves room for the occasional fast-path call. Scales
        # automatically if max_slots is raised (8→7 background, 10→9, etc.).
        #
        # background_floor_pct PINNED to 0.0 (→ floor 1). The 0.20 default was
        # calibrated for the old 6-slot thinker (int(6·0.20)=1); at 32 slots it
        # silently became int(32·0.20)=6, walling 6 slots off from rare
        # interactive bursts (interactive ceiling 32-6=26) for no benefit on a
        # background-dominated endpoint. The floor's only role is the
        # interactive-reservation ceiling — keep it minimal so an interactive
        # burst can use all 31 non-reserved slots. Cap stays 31 via the reserve.
        background_floor_pct=0.0,
        fast_path_reserve_slots=1,
        host="10.0.0.3", port=9083,
        backend_engine="vllm",
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

    # --- timeout-advice model (see timeout_model.py) ---
    # ``recommended = max(p99 * margin, floor)``.  Margin is the one
    # policy knob; window/min_samples govern the empirical distribution.
    timeout_advice_margin: float = 1.5
    timeout_advice_window_s: float = 7 * 86400.0
    timeout_advice_min_samples: int = 30

    @property
    def total_fleet_slots(self) -> int:
        return sum(ep.max_slots for ep in self.endpoints.values())

    def agent_config(self, agent_id: str) -> AgentQuotaConfig:
        if agent_id not in self.agents:
            self.agents[agent_id] = AgentQuotaConfig(agent_id=agent_id)
        return self.agents[agent_id]


# ---------------------------------------------------------------------------
# Per-agent quota config loader
# ---------------------------------------------------------------------------

_DEFAULT_AGENTS_CONFIG_PATH = Path(__file__).resolve().parent / "agents.yaml"


def load_agent_configs(path: str | Path | None = None) -> dict[str, AgentQuotaConfig]:
    """Load per-agent DRR quota config from a YAML file.

    Each top-level key is an agent_id (matching what the
    ProxyLLMClient / ProxyScheduler infers from the process cmdline).
    Values may set any subset of:
        weight (float),
        max_balance_ss (float),
        default_priority (str enum name — e.g. "P3_INGESTION").
    Missing keys fall back to the AgentQuotaConfig dataclass defaults.

    When ``path`` is None, looks for ``LLM_PROXY_AGENTS_CONFIG`` env
    var, else falls back to ``<package>/agents.yaml``. A missing file
    is non-fatal: returns ``{}`` and the proxy lazy-creates per-agent
    configs at default values.
    """
    if path is None:
        path = os.environ.get("LLM_PROXY_AGENTS_CONFIG") or _DEFAULT_AGENTS_CONFIG_PATH
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import yaml  # imported lazily so the package boots without yaml in
                    # purely-Python-stdlib test environments
        raw = yaml.safe_load(p.read_text()) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("agents config load failed at %s: %s", p, exc)
        return {}
    if not isinstance(raw, dict):
        logger.warning("agents config at %s: expected mapping, got %s", p, type(raw))
        return {}
    out: dict[str, AgentQuotaConfig] = {}
    for agent_id, cfg in raw.items():
        if not isinstance(cfg, dict):
            logger.warning(
                "agents config: skipping %r — expected mapping, got %s",
                agent_id, type(cfg),
            )
            continue
        kwargs: dict[str, Any] = {"agent_id": str(agent_id)}
        if "weight" in cfg:
            kwargs["weight"] = float(cfg["weight"])
        if "max_balance_ss" in cfg:
            kwargs["max_balance_ss"] = float(cfg["max_balance_ss"])
        if "default_priority" in cfg:
            kwargs["default_priority"] = LLMPriority.coerce(cfg["default_priority"])
        out[str(agent_id)] = AgentQuotaConfig(**kwargs)
    logger.info("loaded %d agent quota config(s) from %s", len(out), p)
    return out
