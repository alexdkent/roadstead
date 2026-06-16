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
import re
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
    def coerce(
        cls,
        value: "LLMPriority | str | int | None",
        *,
        default: "LLMPriority | None" = None,
    ) -> "LLMPriority":
        """Map a priority value to a member — three tiers of robustness.

        1. **Deterministically correct** a known-correctable class: exact and
           case-insensitive member names, band names (interactive / foreground /
           background), the legacy aliases, and *any* ``P<n>_<suffix>`` label by
           its numeric prefix — so a non-canonical ``P3_BACKGROUND`` resolves to
           ``P3_INGESTION`` instead of erroring. Out-of-range ints are clamped.
        2. **Soft-default** when a ``default`` is supplied: log a WARNING and
           return it instead of raising. The request path (``QueuedRequest``)
           passes ``default=P1_TURN_SUPPORT`` so a malformed ``priority`` field
           never escapes as an unhandled 500 / fails the caller's LLM call.
        3. **Hard-raise** only when no ``default`` is given (config-load
           strictness — a misconfigured quota file should fail loud).
        """
        if isinstance(value, cls):
            return value
        if value is None:
            return default if default is not None else cls.P1_TURN_SUPPORT
        if isinstance(value, bool):
            # bool is an int subclass; a JSON `true`/`false` is not a priority.
            value = str(value)
        if isinstance(value, int):
            if value in cls._value2member_map_:
                return cls(value)
            clamped = max(0, min(int(value), 4))
            logger.warning("out-of-range LLM priority int %r -> %s",
                           value, cls(clamped).name)
            return cls(clamped)
        text = str(value).strip()
        if text in cls.__members__:
            return cls[text]
        upper = text.upper()
        if upper in cls.__members__:          # case-insensitive member name
            return cls[upper]
        lowered = text.lower()
        aliases = {
            "realtime": cls.P0_REALTIME,
            "turn": cls.P1_TURN_SUPPORT,
            "turn_support": cls.P1_TURN_SUPPORT,
            "post_turn": cls.P2_POST_TURN,
            "ingestion": cls.P3_INGESTION,
            "hygiene": cls.P4_HYGIENE,
            "background": cls.P3_INGESTION,
            # band names → the representative priority of that band
            "interactive": cls.P1_TURN_SUPPORT,
            "foreground": cls.P2_POST_TURN,
        }
        if lowered in aliases:
            return aliases[lowered]
        # Deterministic-correct any "P<n>_<suffix>" label by its numeric prefix
        # (covers non-canonical band-suffixed names like P3_BACKGROUND → P3).
        m = re.match(r"^p(\d+)(?:[_-]|$)", lowered)
        if m and 0 <= int(m.group(1)) <= 4:
            corrected = cls(int(m.group(1)))
            logger.warning("non-canonical LLM priority %r -> %s",
                           value, corrected.name)
            return corrected
        if default is not None:
            logger.warning("unknown LLM priority %r; defaulting to %s",
                           value, default.name)
            return default
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
    # 2026-06-08: gemma-greeter consolidated onto the E4B "gemma" backend (:9091);
    # the E2B-hot unit (:9090, llama-gemma-hot) was DECOMMISSIONED to reclaim ~2GB of
    # anvil's unified pool. It served ~1.2K tokens over 2.4d — the Tier-1.5 LLM
    # router/classifier fallback rarely fires (keyword routers handle the common path).
    # The role NAME is kept so the ~8 agent callers (health-verifier / mail-agent /
    # homeassistant / infrastructure / knowledge / sidekick / orchestrator / temporal) are
    # untouched — their make_nexus_client("gemma-greeter") calls now land on E4B.
    # Listed before gemma-router so CLASS_TO_ROLE["gemma"] stays "gemma-router".
    "gemma-greeter": "gemma",
    "gemma-router":  "gemma",
    "bge-reranker":  "rerank",
    "bge-m3-embed":  "embed",
    "llama-thinker": "thinker",
    # 2026-06-11: DECKARD-31B creative/counterpoint model (Gemma-4 NVFP4-AWQ on the
    # AEON DGX-Spark vLLM image, anvil :9095). Different-family generative voice to
    # the thinker; gemma4 reasoning parser + guidance structured-outputs (tools off).
    "creative":      "creative",
}

CLASS_TO_ROLE: dict[str, str] = {v: k for k, v in ROLE_TO_CLASS.items()}


def normalize_endpoint(endpoint: str) -> str:
    """Collapse a role name to its QoS endpoint class."""
    text = str(endpoint).strip().lower()
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

    # --- on-demand lifecycle (see on_demand.OnDemandManager) ---
    # When True, this endpoint's model is NOT always-resident: before a request
    # dispatches, the proxy acquires the anvil GPU-slot dispatcher lease
    # (``dispatcher_capability``), which loads the model (FIFO behind imagegen /
    # diarize / etc.) and idle-unloads it when quiet — freeing the GPU pool.
    # Pairs with a high timeout floor (cold load takes minutes).
    on_demand: bool = False
    dispatcher_capability: str = ""

    # When True, the proxy injects ``id_slot`` into chat_completion payloads
    # for INTERACTIVE-band requests, keyed by a deterministic hash of
    # session_id. This pins each conversation to one llama.cpp slot across
    # turns so the patched KV-cache checkpoint is reused instead of
    # re-prefilling from scratch. Only useful for multi-slot llama.cpp
    # backends (not vLLM). Off by default; enabled on companion.
    slot_affinity: bool = False

    # When True, the capacity poller probes ONLY /health for this endpoint —
    # no /props, /v1/models, or vLLM capacity discovery. For non-OpenAI
    # FastAPI shims (embed :9087, rerank :9084) that have neither route:
    # without this they 404 both discovery probes every 10s forever (httpx
    # log spam + a consecutive_failures counter that cycles 1-2-3-reset and
    # never means anything). max_slots/context stay config-seeded.
    skip_discovery: bool = False

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
        # (served id companion.gguf) on :8081. Live (2026-06-05): --ctx-size
        # 98304 --parallel 3 → 32768/slot, 3 slots — downsized from 4×98304 to
        # free nexus memory for the Chatterbox Turbo TTS. The fleet's
        # high-capability summarizer/composer (qwen-composer role). /props
        # discovery is unreliable on the 80B, so this seed is load-bearing —
        # keep it exact.
        endpoint_class="companion", role="qwen-composer",
        max_slots=3, context_per_slot=32768,
        # Served --parallel 3 (== max_slots). 2026-06-16: migrated nexus off the
        # self-patched 0be84685b-cached build to CLEAN upstream llama.cpp b9309
        # (6d57c26), which fixes the recurrent/hybrid seq_rm path natively
        # (common_context_can_seq_rm → checkpoint-restore fallback). That ended
        # the recurring GGML_ABORT "failed to remove sequence" core-dumps (Jun
        # 1/5/13/14) the old patched build hit on partial KV trims, while keeping
        # checkpoint reuse (validated: 14 restores, 0 reprefill, 0 abort on
        # multi-turn 4.6K-ctx). The 3-slot cap stays as a conservative ceiling.
        dispatch_concurrency_cap=3,
        # Reserve 1 slot for interactive/chat rounds; background (composer
        # summaries + knowledge ingestion) uses the other 2. Mirrors the
        # thinker's fast_path_reserve pattern. background_floor_pct left at the
        # default (0.20 → floor 1): background_cap = max(floor, max_slots -
        # reserve) = max(1, 2) = 2.
        fast_path_reserve_slots=1,
        slot_affinity=True,
        host="10.0.0.6", port=8081,
    ),
    "gemma": EndpointConfig(
        # anvil Gemma-4-E4B (cold classifier + vision) :9091 — n_ctx
        # 16384/slot, 2 slots. Since 2026-06-08 this backend ALSO serves the
        # gemma-greeter role (Tier-1.5 router/classifier fallback) — the E2B-hot
        # :9090 backend was decommissioned (see ROLE_TO_CLASS note above). E4B is
        # lightly loaded (~177K tok/day, n_busy≈1.0 of 2 slots) so it absorbs the
        # rare greeter fallbacks with headroom; the only cost is slightly slower
        # first-token (4B vs 2B) on those infrequent calls.
        endpoint_class="gemma", role="gemma-router",
        max_slots=2, context_per_slot=16384,
        host="10.0.0.3", port=9091,
    ),
    # NOTE: the "gemma-hot" endpoint class (anvil Gemma-4-E2B :9090) was REMOVED
    # 2026-06-08. The llama-gemma-hot.service unit is stopped+disabled on anvil.
    # gemma-greeter now routes to the "gemma" class above. Do NOT re-add a probe of
    # :9090 — it is dark by design.
    "rerank": EndpointConfig(
        # anvil bge-reranker shim :9084 (infinity backend behind it on :9085)
        # — n_ctx 8192/slot, 1 slot.
        endpoint_class="rerank", role="bge-reranker",
        max_slots=1, context_per_slot=8192,
        background_floor_pct=0.0,
        host="10.0.0.3", port=9084,
        # FastAPI shim — no /props or /v1/models; health-probe only.
        skip_discovery=True,
    ),
    "embed": EndpointConfig(
        # anvil bge-m3 embed :9087 — n_ctx 8192/slot, 4 slots (the CANONICAL
        # fleet embed; the nexus CPU copy on :8091 was removed 2026-06-05).
        endpoint_class="embed", role="bge-m3-embed",
        max_slots=4, context_per_slot=8192,
        host="10.0.0.3", port=9087,
        # FastAPI shim — no /props or /v1/models; health-probe only.
        skip_discovery=True,
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
    "creative": EndpointConfig(
        # anvil DECKARD-31B (Gemma-4-31B NVFP4-AWQ, AEON DGX-Spark image) :9095 —
        # the `creative` role: a different-family creative/counterpoint voice to the
        # thinker. Dense 31B, bandwidth-bound on Spark (~7 tok/s @1, ~42 agg @4) →
        # low-QPS by design. --max-num-seqs 6 → max_slots 6; --max-model-len 131072;
        # gpu-mem-util 0.21 (~26GB: ~18.4GB weights + ~7.4GB fp8 KV ≈ exactly one
        # full 131072-ctx request, 2026-06-12). Reasoning model (gemma4 <think>);
        # guidance structured-outputs available, auto-tool-choice OFF.
        endpoint_class="creative", role="creative",
        max_slots=6, context_per_slot=131072,
        # Background-band slot policy mirrors the thinker (see above): the
        # creative role is ~all background (sidekick song authoring at P4_HYGIENE —
        # craft/theme-pick/hook-polish/self-echo). Left at the dataclass
        # defaults (floor_pct 0.20 → floor 1, reserve 0 → cap == floor) the
        # background band was pinned to ONE slot, serializing batch authoring
        # at 1-wide while the backend is sized for --max-num-seqs 4. DECKARD is
        # bandwidth-bound (~7 tok/s @1 vs ~42 agg @4) so the extra concurrency
        # is a large aggregate-throughput win and costs no extra anvil memory
        # (KV inside the already-allocated gpu-mem 0.28). floor_pct 0.0 → floor
        # 1; reserve 1 → background cap = max_slots-1 = 3, with one slot held
        # back for an occasional interactive/fast-path creative call.
        background_floor_pct=0.0,
        fast_path_reserve_slots=1,
        host="10.0.0.3", port=9095,
        backend_engine="vllm",
        # 2026-06-12: PINNED ALWAYS-ON — no longer on-demand. Creative became key
        # to multiple flows so it now mirrors the thinker: a boot-started,
        # always-resident vLLM unit (vllm-deckard.service WantedBy=multi-user.target;
        # anvil dispatcher registry manage_process=False → never idle-unloaded).
        # The proxy therefore dispatches directly with NO dispatcher lease gating,
        # removing the "dispatcher down → creative 503 even though DECKARD is up"
        # failure mode. whisper+diarize moving to Nasbox frees the headroom for
        # permanent residency. (Slot is no longer shared/evicted; the 900s timeout
        # floor is now moot for warm calls but harmless.)
    ),
}


# NOTE: the proxy-centralized "reason-then-constrain" reason-injection layer
# (StructMode/StructPolicy registry + grammar.inject_reason_field) was REMOVED
# 2026-06-14. The 2026-06-05 offline A/B measured it neutral-to-modestly-worse
# on the only real-data call_site and recommended dropping it; it shipped OFF
# and never activated. The egress silent-drop DETECTOR below
# (shadow_egress_detect_enabled / grammar.verify_conformance) is independent and
# stays — it's the piece the A/B said was worth keeping.


# --- Thinking option (per-request opt-in native reasoning) ---------------------
# Callers opt in per request with ``thinking: true``; default OFF is fully
# transparent (backend.py forces enable_thinking=False), so existing traffic is
# unaffected. On a vLLM (reasoning-parser) backend the proxy then enables native
# <think>, adds a GENEROUS reasoning budget to max_tokens, and normalizes the
# structured response (deterministic recovery of the bounded stray-brace artifact
# that vLLM emits at the reason→JSON boundary under MTP spec-decode — see
# grammar.recover_structured_object + vLLM #34650/PR #44142).

def thinking_enabled() -> bool:
    """Feature kill-switch (default ON). Per-request opt-in is the real gate;
    nothing reasons until a caller sets ``thinking: true``, so this only exists
    to disable the path fleet-wide in an incident."""
    return os.environ.get("COLLECTIVE_PROXY_THINKING", "1").strip().lower() not in (
        "0", "false", "no", "off", "",
    )


def shadow_egress_detect_enabled() -> bool:
    """WS-4 (2026-06-05): run ``verify_conformance`` on EVERY grammar-bearing
    structured response in SHADOW — log + count silent grammar-drops (parse
    fail / markdown-fence / non-conformance, i.e. llama.cpp dropped the grammar
    and ran free-form), but NEVER mutate the response. Zero caller risk, so
    default ON: the whole point is to collect a baseline silent-drop rate per
    call_site from live traffic. Env kill-switch ``COLLECTIVE_PROXY_SHADOW_EGRESS``."""
    return os.environ.get("COLLECTIVE_PROXY_SHADOW_EGRESS", "1").strip().lower() not in (
        "0", "false", "no", "off", "",
    )


def degeneration_guard_enabled() -> bool:
    """Egress repetition-LOOP guard (2026-06-10): detect a degenerate response —
    the same long n-gram repeated many times ('… the song of ## … the song of ##
    …'), a model failure mode any caller can hit on a long prompt — and RE-DISPATCH
    with an anti-repetition penalty. A 200-with-garbage is invisible to the
    transient-error and grammar checks, so this is the layer that catches it.
    Default ON. Env kill-switch ``COLLECTIVE_PROXY_DEGENERATION_GUARD``."""
    return os.environ.get("COLLECTIVE_PROXY_DEGENERATION_GUARD", "1").strip().lower() not in (
        "0", "false", "no", "off", "",
    )


def degeneration_shadow_only() -> bool:
    """When set, the degeneration guard only DETECTS + logs + counts (no
    re-dispatch) — the shadow-measure phase to confirm it never false-flags
    legitimate repetition (a song chorus) before it acts. Default OFF (the guard
    actively corrects). Env ``COLLECTIVE_PROXY_DEGENERATION_SHADOW``."""
    return os.environ.get("COLLECTIVE_PROXY_DEGENERATION_SHADOW", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def thinking_reasoning_budget() -> int:
    """Tokens of reasoning headroom ADDED to a thinking request's max_tokens.
    Reasoning is generated <think> output and counts against max_tokens, so too
    small a budget truncates mid-reasoning (finish=length). Operator directive
    (2026-06-05): prefer short-term slowness over cutoff failures — start
    generous, watch the truncation metric, tune DOWN over time. Env override
    ``COLLECTIVE_PROXY_THINKING_BUDGET``."""
    try:
        return max(0, int(os.environ.get("COLLECTIVE_PROXY_THINKING_BUDGET", "8000")))
    except ValueError:
        return 8000


def _inflight_stream_interval_s() -> float:
    """Cadence of the SSE ``inflight`` reconcile frame that powers the live
    in-flight board. The instant ``call.dispatched``/``call.completed`` events do
    the real work; this just refreshes elapsed/queue/occupancy and recovers any
    missed event. Gated on connected clients (free when nobody's watching). Env
    override ``COLLECTIVE_PROXY_INFLIGHT_INTERVAL_S`` (clamped ≥0.25s)."""
    try:
        return max(0.25, float(os.environ.get("COLLECTIVE_PROXY_INFLIGHT_INTERVAL_S", "1.5")))
    except ValueError:
        return 1.5


@dataclass
class ProxyConfig:
    """Top-level proxy configuration."""
    port: int = 42161
    endpoints: dict[str, EndpointConfig] = field(default_factory=lambda: dict(DEFAULT_ENDPOINTS))
    agents: dict[str, AgentQuotaConfig] = field(default_factory=dict)
    starvation_timeout_s: float = 30.0
    drr_tick_interval_s: float = 0.01
    # Live in-flight SSE frame cadence (see _inflight_stream_interval_s).
    inflight_stream_interval_s: float = field(default_factory=_inflight_stream_interval_s)
    queue_db_path: str = ""
    stats_db_path: str = ""
    request_log_path: str = ""
    # Runtime-mutable feature flags (flags.py): persisted JSON, mutated via
    # POST /v1/admin/flags. Empty path → in-memory defaults (tests).
    runtime_flags_path: str = ""
    # Capacity-poller cadence. Production default 10s; tests shrink it so
    # poller-loop behaviour is observable without 10s waits.
    poller_interval_s: float = 10.0

    # --- queue.db maintenance (persistence cleanup) ---
    # Payload bodies (payload_json/response_json on proxy_completions) are only
    # read by recent-data consumers (llm_health_sweep last 100-300 rows, the
    # replay/AB harness --hours 4); NULL them after this window while completion
    # METADATA keeps its full retention. Sheds the bulk of the DB weight.
    payload_retention_s: float = 48 * 3600.0
    # Completion METADATA retention (rows, not payload bodies). Now that
    # proxy_completions is the whole-fleet call-metrics store (Phase 1: LLM
    # native + non-LLM pushed), keep 30 days to back the fleet usage/savings
    # rollups (the host daemon's `calls` table kept 90d — we trade history
    # depth for the proxy's much higher row volume). Override via env.
    completions_retention_s: float = 30 * 86400.0
    # TRUNCATE-checkpoint the WAL on this cadence so the -wal sidecar can't camp
    # at a burst high-water mark (auto-checkpoint only resets it for reuse, never
    # shrinks the file).
    wal_checkpoint_interval_s: float = 300.0
    # Return freed pages (deletes + payload NULLs) to the OS gradually via
    # incremental_vacuum — only effective once auto_vacuum=INCREMENTAL is
    # committed by the one-time startup VACUUM.
    incremental_vacuum_interval_s: float = 600.0
    incremental_vacuum_pages: int = 4000
    # One-time full VACUUM at startup (pre-writer, single-threaded) when the
    # freelist exceeds this — reclaims dead space AND commits the auto_vacuum mode
    # change. Normally a no-op on a healthy DB.
    startup_vacuum_freelist_threshold_bytes: int = 200 * 1024 * 1024

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
