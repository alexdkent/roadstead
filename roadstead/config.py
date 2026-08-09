"""Configuration dataclasses for the LLM proxy.

All proxy behaviour is driven by these structures.  Slot counts and
context sizes come from backend ``/props`` discovery at runtime — the
config only carries *policy* knobs (weights, floors, timeouts).

``DEFAULT_ENDPOINTS`` below is THE canonical role→host:port map for the
whole system. The LLM proxy is the single front door to all LLM traffic
(10.0.0.3 anvil / 10.0.0.6 nexus); every agent reaches a backend via
``make_nexus_client(role)`` → ``ProxyLLMClient`` → ``:42161`` → here.
There is no other routing table. (The ``infra/inference/profiles/*.yaml``
shadow map and its ``inferctl`` tooling were DELETED 2026-08-09 — they
maintained a parallel role→URL table the proxy never read.)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

from . import model_catalog

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

# Derived from the model authority (llmproxy/models.yaml). To add/move/rename a
# role, edit THAT file — these maps follow automatically. (gemma-greeter shares
# the E4B "gemma" backend with gemma-router; the proxy-endpoint OWNER of "gemma"
# is gemma-router, so CLASS_TO_ROLE["gemma"] == "gemma-router".)
ROLE_TO_CLASS: dict[str, str] = model_catalog.build_role_to_class()

CLASS_TO_ROLE: dict[str, str] = model_catalog.build_class_to_role()


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
    # The backend's REAL launch-time concurrency ceiling for a vLLM endpoint
    # (``--max-num-seqs``), mirrored from the tracked serve script. vLLM does NOT
    # expose this over its API (unlike llama.cpp /props), so ``max_slots`` stays
    # config-seeded and can silently drift from the backend's actual cap (the
    # reasoner 32-vs-20 class, reconciled in Step 2b). 0 = not tracked (llama.cpp /
    # shims). The Step-4c shadow reconciler warns when ``max_slots`` != this value.
    documented_max_num_seqs: int = 0
    # The backend was launched with structured-output whitespace BANNED
    # (vLLM ``--structured-outputs-config '{"disable_any_whitespace":true}'``),
    # mirrored from the tracked serve script — the proxy cannot introspect a
    # backend launch flag, so this is a DECLARATION, pinned against the vendored
    # script by test_tier3_serve_script_doctrine.py.
    #
    # The flag is load-bearing (it stops the whitespace runaway; see the reasoner
    # notes in models.yaml) but it has ONE toxic interaction: with whitespace
    # banned, a BARE ``response_format:{"type":"json_object"}`` makes the literal
    # two-character document ``{}`` a legal, complete, zero-whitespace object —
    # and therefore the greedy path. Every such caller silently received ``{}``
    # with finish_reason=stop from 2026-07-31 15:06 UTC. ``Correction
    # .apply_json_object_guard`` uses this flag to strip bare json_object on
    # these endpoints. False everywhere the launch flag is absent.
    disable_any_whitespace: bool = False
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
    # BELOW max_slots for a concurrency-fragile backend to leave crash-headroom.
    # Historical example: the FORMER companion, Qwen3-Next-80B (Strix Halo Vulkan,
    # patched-cache), aborted under full 4-slot pressure with a llama.cpp
    # KV-seq-removal assertion (2026-06-01, G1), so it ran at 3 for headroom. That
    # 80B was retired 2026-07-03 — companion is now Qwen3.5-122B-A10B on nexus,
    # which re-validated safe at cap 4 (see models.yaml composer.policy). Caps
    # DISPATCH only; capacity discovery and reporting still use max_slots.
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

    # --- the model ALWAYS emits an un-disable-able reasoning trace (mirrored from
    # models.yaml ``capabilities.reasoning``; e.g. creative/Trinity-Mini). The CoT
    # counts against max_tokens, so the submit path reserves reasoning headroom
    # (forced_reasoning_budget) on top of the caller's answer cap — otherwise a
    # small cap is consumed mid-reasoning → empty/truncated completion. Harmless on
    # endpoints where thinking is disablable (the proxy forces enable_thinking=False
    # by default, so no reasoning is emitted and the extra cap is never reached). ---
    forces_reasoning: bool = False

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

# THE canonical role→host:port routing table — DERIVED from the model authority
# (llmproxy/models.yaml). Each `proxy_endpoint: true` role produces one entry,
# keyed by its endpoint class. Slots/context are STARTUP SEEDS — the capacity
# poller overwrites max_slots/context_per_slot from backend /props at runtime.
# To add/move/retune a routed model, edit models.yaml (+ the box's systemd unit);
# the per-class policy knobs (slot_affinity, fast_path_reserve_slots,
# dispatch_concurrency_cap, background_floor_pct, skip_discovery) live in its
# `policy:` block. The deep operational rationale for each knob is in the model's
# `notes:` field in models.yaml. NOTE: the "gemma-hot" class (E2B :9090) is
# decommissioned — do NOT re-add a probe of :9090.
DEFAULT_ENDPOINTS: dict[str, EndpointConfig] = {
    cls: EndpointConfig(**kwargs)
    for cls, kwargs in model_catalog.build_endpoint_kwargs().items()
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


def uniform_correction_enabled() -> bool:
    """Step 4a (2026-07-01): route the STREAMING + internal ``/v1/submit`` response
    paths through the uniform correction layer, closing the two path-dependence
    gaps the sync path never had — (1) sanitize vLLM ``qwen3_xml`` tool-call streams
    on BOTH doors (today the ``_ToolCallStreamSanitizer`` runs only for the OpenAI
    door; internal ``/v1/submit`` streams emit raw), and (2) run truncation +
    degeneration DETECTION over a stream's reassembled content (a stream can't
    un-send, but it records the same tallies the sync guards do, so streaming is no
    longer a correction blind spot). Default OFF == byte-identical (only OpenAI
    streams sanitized, no stream-side detection). Env
    ``COLLECTIVE_PROXY_UNIFORM_CORRECTION``."""
    return os.environ.get("COLLECTIVE_PROXY_UNIFORM_CORRECTION", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def max_slots_reconcile_enabled() -> bool:
    """Step 4c (2026-07-01): SHADOW reconciler for the vLLM ``max_slots`` desync.
    vLLM's ``--max-num-seqs`` is not API-discoverable, so ``max_slots`` is
    config-seeded and can silently drift from the backend's real launch cap (the
    reasoner 32-vs-20 bug, Step 2b). This standing guard compares each vLLM
    endpoint's ``max_slots`` against its ``documented_max_num_seqs`` (mirrored from
    the serve script) and raises a ``max_slots_drift`` WARNING alert on mismatch.
    OBSERVABILITY ONLY — it never changes admission (zero caller-visible effect,
    like the shadow-egress detector), so default ON. Env kill-switch
    ``COLLECTIVE_PROXY_MAX_SLOTS_RECONCILE``."""
    return os.environ.get("COLLECTIVE_PROXY_MAX_SLOTS_RECONCILE", "1").strip().lower() not in (
        "0", "false", "no", "off", "",
    )


def cache_drift_alarm_enabled() -> bool:
    """Tier-2 step 3 (2026-07-01): the periodic prefix-cache DRIFT alarm. Each
    ``compute_cache_stats`` cycle, detect call_sites whose front-loaded-prefix
    share (LCP%) collapsed vs their own trailing baseline (a prompt edit broke
    the cacheable leading block) and raise a ``CACHE_DRIFT_ALERT`` log marker +
    a store-less ``llmproxy_cache_drift`` security event (dedup'd, re-fires at
    most every ``CACHE_DRIFT_REALERT_S``). OBSERVABILITY ONLY — it never changes
    routing/admission/output (zero caller-visible effect, like the max_slots
    reconciler + shadow-egress detector), so default ON. Env kill-switch
    ``COLLECTIVE_PROXY_CACHE_DRIFT_ALARM``. See
    ``docs/llmproxy_prefix_cache_observability.md`` (Tier-2 step 3)."""
    return os.environ.get("COLLECTIVE_PROXY_CACHE_DRIFT_ALARM", "1").strip().lower() not in (
        "0", "false", "no", "off", "",
    )


def endpoint_cooldown_enabled() -> bool:
    """Step 4b ENFORCE: after ``cooldown_allowed_fails`` BACKEND-FAULT failures
    (5xx / timeout / unavailable) within ``cooldown_window_s``, briefly mark an
    endpoint unhealthy so the scheduler defers its traffic; auto-recovers after
    ``cooldown_duration_s``. Catches a FLAKY backend (intermittent 5xx that never
    strings ``health_fail_threshold`` in a row — which the consecutive-fail circuit
    misses). Only DEFERS/DEGRADES — never reroutes across backends (that's Phase 4),
    so a cache-sensitive interactive conversation is not cache-busted. Default OFF
    == byte-identical. Env ``COLLECTIVE_PROXY_ENDPOINT_COOLDOWN``."""
    return os.environ.get("COLLECTIVE_PROXY_ENDPOINT_COOLDOWN", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def structured_validity_guard_enabled() -> bool:
    """Operator-mandated ALWAYS-ON floor (2026-07-11): a structured request must
    never see a clean success whose content is not parseable JSON. Two caller-
    visible enforcement points, both under this one kill-switch:

      * sync: a 200 whose content fails ``json.loads`` on a JSON-implying
        structured request (``request_expects_json``) flips to the established
        status=error → 502 (code ``structured_invalid_json``);
      * streaming: a structured stream that TRUNCATED (finish_reason=length) or
        whose reassembled content fails ``json.loads`` terminates with the
        established error frame instead of a clean ``done``.

    Runs AFTER the flag-gated repair layers (thinking-finalize / degeneration /
    schema-backstop), so anything they recover passes; this is the last-resort
    parse-only floor when they're off or exhausted. Schema CONFORMANCE is
    deliberately NOT checked here (the backends enforce grammar; the backstop
    owns schema validation) — this catches truncation/malformation only. The
    LLMPROXY_TRUNCATION / LLMPROXY_STRUCTURED_INVALID observability (ERROR log
    + per-(model, caller) tallies) is NOT gated by this switch — it's read-only.
    Default ON. Env kill-switch ``COLLECTIVE_PROXY_STRUCTURED_VALIDITY``."""
    return os.environ.get("COLLECTIVE_PROXY_STRUCTURED_VALIDITY", "1").strip().lower() not in (
        "0", "false", "no", "off", "",
    )


def schema_backstop_enabled() -> bool:
    """Phase 3 (2026-07-01): the structured-output/tool-call reliability backstop.
    On a structured/tool SYNC response, run
    ``json-repair → jsonschema-validate against the caller's declared schema →
    one bounded (deadline+concurrency) retry with the error fed back → fail-loud
    deferrable``, closing the parseable-but-schema-invalid / trailing-prose /
    fenced / malformed-``tool_calls.arguments`` gap (today only empty / degenerate
    / truncated / thinking-noise are rescued). Default OFF == byte-identical (the
    guard early-returns before any observable effect). Env
    ``COLLECTIVE_PROXY_SCHEMA_BACKSTOP``. See
    ``docs/llmproxy_phase3_schema_backstop_contract.md``."""
    return os.environ.get("COLLECTIVE_PROXY_SCHEMA_BACKSTOP", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def schema_backstop_shadow() -> bool:
    """When set (and the backstop enabled), the backstop only DETECTS + attempts
    repair in memory + logs what it WOULD return + counts — but returns the
    ORIGINAL response untouched (no swap, no retry re-dispatch, no fail-loud). The
    review window: measure how often the backstop fires and whether repair succeeds
    before it mutates live responses. Default OFF (the guard actively corrects when
    enabled). Env ``COLLECTIVE_PROXY_SCHEMA_BACKSTOP_SHADOW``."""
    return os.environ.get("COLLECTIVE_PROXY_SCHEMA_BACKSTOP_SHADOW", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def endpoint_cooldown_shadow() -> bool:
    """Step 4b SHADOW (review this first): count backend-fault failures + log a
    'would cool' + surface the trip on /v1/status, but DON'T actually pull the
    endpoint. Confirm the thresholds don't false-trip on this fleet's traffic, then
    flip enforce. Default OFF. Env ``COLLECTIVE_PROXY_ENDPOINT_COOLDOWN_SHADOW``."""
    return os.environ.get("COLLECTIVE_PROXY_ENDPOINT_COOLDOWN_SHADOW", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def cooldown_allowed_fails() -> int:
    """Backend-fault failures within the window that trip a cooldown (default 4 —
    above the 3-consecutive circuit so this catches the INTERMITTENT case).
    Env ``COLLECTIVE_PROXY_COOLDOWN_ALLOWED_FAILS``."""
    try:
        return max(1, int(os.environ.get("COLLECTIVE_PROXY_COOLDOWN_ALLOWED_FAILS", "4")))
    except ValueError:
        return 4


def cooldown_window_s() -> float:
    """Sliding window over which ``cooldown_allowed_fails`` is counted (default 60s).
    Env ``COLLECTIVE_PROXY_COOLDOWN_WINDOW_S``."""
    try:
        return max(1.0, float(os.environ.get("COLLECTIVE_PROXY_COOLDOWN_WINDOW_S", "60")))
    except ValueError:
        return 60.0


def cooldown_duration_s() -> float:
    """How long a tripped endpoint stays cooled before auto-recovery (default 30s).
    Env ``COLLECTIVE_PROXY_COOLDOWN_DURATION_S``."""
    try:
        return max(1.0, float(os.environ.get("COLLECTIVE_PROXY_COOLDOWN_DURATION_S", "30")))
    except ValueError:
        return 30.0


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


def forced_reasoning_budget() -> int:
    """Tokens of reasoning headroom ADDED to max_tokens for an endpoint whose model
    ALWAYS emits an un-disable-able reasoning trace (``capabilities.reasoning=true``
    — e.g. ``creative``/Trinity-Mini). The CoT is generated output and counts against
    max_tokens, so a small caller cap (dj-crew speaker turns run ~240-360) truncates
    mid-reasoning → empty/partial content. This reserves room for the answer AFTER
    the reasoning. Deliberately smaller than ``thinking_reasoning_budget`` (that's for
    explicit vLLM long-form thinking); creative turns reason ~350-500 tokens, so the
    default gives ~3x headroom without inflating tiny caps into runaway generations.
    Env override ``COLLECTIVE_PROXY_FORCED_REASONING_BUDGET``."""
    try:
        return max(0, int(os.environ.get("COLLECTIVE_PROXY_FORCED_REASONING_BUDGET", "1536")))
    except ValueError:
        return 1536


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
    # Prefix-cache observability cadence (piggybacks the poller): scrape backend
    # /metrics + run the cache-ability screen, persist a snapshot per chat
    # endpoint. 30min keeps it cheap; first run fires on the first poll.
    cache_stats_interval_s: float = 1800.0

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
    # Adaptive (load × size) uplift on top of the empirical recommendation, so a
    # deadline widens with LIVE contention and prompt size instead of failing a
    # merely-busy/slow call (2026-07-05). ``surge = 1 + k_load*min(over, max)``
    # where ``over`` is backlog measured in units of endpoint capacity;
    # ``size_stretch`` smooths the coarse-bucket cliff past the 16K top edge.
    # Bounded by the per-caller-class ceiling (interactive vs background band +
    # per-role ``timeout_ceiling_s`` override).
    timeout_surge_k: float = 0.5
    timeout_surge_max: float = 3.0
    timeout_size_k: float = 0.5
    timeout_size_max: float = 4.0
    timeout_ceiling_interactive_s: float = 600.0
    timeout_ceiling_background_s: float = 1800.0

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
