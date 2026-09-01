"""Configuration dataclasses for the LLM proxy.

All proxy behaviour is driven by these structures.  Slot counts and
context sizes come from backend ``/props`` discovery at runtime — the
config only carries *policy* knobs (weights, floors, timeouts).

``DEFAULT_ENDPOINTS`` below is THE routing table, derived from the catalog
(``models.yaml``): one entry per routed endpoint class, its connection taken
from the provider it names. There is no second one, and that is deliberate —
the origin deployment kept a parallel role→URL map in its own tooling that the
proxy never read, and the two disagreed for as long as both existed.
"""

from __future__ import annotations

import logging
import os
import re
import dataclasses
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

from . import model_catalog
from .constants import _INTERACTIVE_CEILING_S

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
    #: Fraction of a request's ``max_tokens`` to give the model for REASONING, when
    #: this backend supports a reasoning budget at all. 0.0 = unsupported, inject
    #: nothing — and that is the safe default for every endpoint.
    #:
    #: 🚨 This is a DECLARATION of a launch flag, like ``disable_any_whitespace``
    #: above: vLLM only honours ``thinking_token_budget`` when the server was started
    #: with ``--reasoning-config``, and it 400s the whole request when it was not
    #: ("thinking_token_budget is set but reasoning_config is not configured").
    #: The proxy cannot introspect that flag, so it is mirrored here and pinned
    #: against the vendored serve script by
    #: test_tier3_serve_script_doctrine.py::test_thinking_budget_ratio_matches_the_script.
    #: Setting this on an endpoint whose server lacks the flag does not degrade it —
    #: it breaks every thinking request to it outright.
    #:
    #: WHY A RATIO AND NOT A FIXED NUMBER. Measured 2026-08-23 on tier3: with no
    #: budget, reasoning consumed the ENTIRE completion allowance and the caller got
    #: ``content: ""`` — twice out of twice, after 8 and 13 minutes. The failure is
    #: not slowness, it is that reasoning leaves no room for an answer. So the number
    #: that matters is answer HEADROOM, which is a fraction of ``max_tokens`` — not
    #: prompt size, and not a constant that is generous for one caller and starving
    #: for the next.
    thinking_budget_ratio: float = 0.0
    #: The chat-template variable(s) that switch REASONING on/off for the model
    #: this endpoint serves, from the stanza's ``policy.thinking_kwargs``.
    #:
    #: 🚨 THIS IS A PROPERTY OF THE MODEL'S CHAT TEMPLATE, NOT OF THE ENGINE.
    #: It was hardcoded to Qwen's ``enable_thinking`` in ``backend.py`` until
    #: 2026-08-24, which meant the 2026-08-23 tier3 swap from Qwen3.6 to
    #: DeepSeek-V4-Flash left the proxy driving a switch it had no reason to
    #: believe the new template read. Measured live per family (probe table in
    #: ``backend.py``'s injection block): DeepSeek-V4 answers to BOTH
    #: ``thinking`` and ``enable_thinking``; Qwen3.8/Qwen3.6 answer ONLY to
    #: ``enable_thinking``. Declaring it here is what makes the next model swap
    #: a one-line edit next to the model name instead of a silent no-op.
    #:
    #: Empty tuple = undeclared, and the proxy then injects NOTHING — the safe
    #: default for an endpoint whose template we have not measured. Detection of
    #: a caller's own pin is deliberately NOT gated on this: see
    #: ``backend._THINKING_KWARG_NAMES``.
    thinking_kwargs: tuple[str, ...] = ()
    #: The expected identity of the WEIGHTS behind this endpoint, from the
    #: stanza's ``policy.model_fingerprint``. vLLM reports `/v1/models[0].root`
    #: (a weights path); llama.cpp reports a `meta` block folded to
    #: ``params=N;vocab=N;ftype=X``. See ``backend.probe_model_fingerprint``.
    #:
    #: 🚨 This exists because ``served_model_id`` CANNOT detect a model swap.
    #: tier3 is served as `--served-model-name llama-thinker`, so its id was
    #: unchanged across the 2026-08-23 Qwen3.6 -> DeepSeek-V4-Flash cutover
    #: while every model-dependent declaration on the stanza (thinking_kwargs,
    #: thinking_budget_ratio, disable_any_whitespace, documented_max_num_seqs)
    #: silently became a claim about a model that was no longer there.
    #:
    #: Empty = undeclared, and the drift alert stays silent. Comparison is
    #: EQUALITY on an opaque string — never parse a fingerprint.
    model_fingerprint: str = ""
    # --- discovered at runtime (mutable), for the drift + canary alerts ------
    #: What the backend actually reports now (None/"" = could not tell).
    discovered_model_fingerprint: str = ""
    #: Last thinking-canary verdict: "" = not yet probed / could not tell,
    #: "ok" = the declared switch really turned reasoning on, otherwise a short
    #: human-readable failure detail. Set by the poller, read by the alert
    #: builder — the poller owns the I/O, the alert builder owns the judgement.
    thinking_canary_state: str = ""
    #: Monotonic timestamp of the last canary attempt (0 = never). The canary
    #: costs a real generation, so it runs on a slow cadence, not every poll.
    thinking_canary_checked_at: float = 0.0
    # --- tier3 failover (§ 9 of the anvil2/V4-Flash plan) ------------------
    # The endpoint CLASS this one degrades to while it is unhealthy, derived
    # from the model stanza's `fallback:` in models.yaml. Empty = no failover
    # (today's behaviour: a clean 503).
    #
    # 🚨 `fallback:` was advisory metadata that nothing consumed for MONTHS
    # (models.yaml said so in its own comment). This field is what makes it
    # load-bearing, which is exactly why it now owes a doctrine test that it
    # resolves to a live, active proxy endpoint — otherwise it rots back into a
    # comment and the failover target silently becomes nothing.
    # Guard: test_failover.py::test_declared_fallbacks_resolve_to_live_endpoints.
    failover_to: str = ""
    # Minimum time (s) spent in degraded mode before returning to this endpoint,
    # even once it is healthy again and the degraded cohort has drained. NOT
    # redundant with the drain: without it a backend flapping every 30s produces
    # a request-boundary model flip every 30s. From the stanza's
    # `policy.failover_dwell_s`.
    failover_dwell_s: float = 120.0
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

    # --- backend connection, the general form ---
    #
    # A full base URL, which SUPERSEDES host/port when set. host:port is the
    # shape a local engine on the LAN has, and it cannot express a remote
    # provider: no scheme (OpenRouter is https), no base path (its routes hang
    # off /api/v1), no way to reach anything that is not a bare origin. Rather
    # than overload `host` with a URL and leave every reader guessing which it
    # holds, the two are separate fields and ``backend_url`` below is the one
    # thing the transport reads.
    base_url: str = ""

    # --- Name of the environment variable holding this backend's API key, for
    # a provider that needs one. THE NAME, NEVER THE KEY: a key in a config file
    # is a key in a git history, and this repo is heading for public
    # (`docs/corpus_and_scrub_plan.md`). Empty for every local backend — they
    # take no auth, which is itself a reason local capacity is the design
    # centre. Resolved at request time by the provider, so rotating the secret
    # does not need a restart. ---
    api_key_env: str = ""

    # --- served model id (the name the backend answers to in the `model`
    # field). Discovered from /v1/models at runtime; empty until then.
    # vLLM validates this field and 404s on a mismatch, so the proxy sets
    # it to ``effective_model_id`` before dispatching to a vLLM backend.
    #
    # A REMOTE provider seeds it from config instead: OpenRouter serves
    # hundreds of models behind one base URL, so "the model this endpoint is"
    # is a choice we make (`openai/gpt-oss-120b`), not a fact to discover, and
    # its provider declares `publishes_served_model_id=False` so the poller
    # does not try. ---
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

    # --- the backend can actually SEE an image (mirrored from models.yaml
    # ``capabilities.vision``; true where an mmproj / vision tower is loaded).
    # Read by the submit-path vision gate in `lifecycle.handle_submit`. 🚨
    # DEFAULTS FALSE ON PURPOSE, and that makes this field load-bearing in one
    # direction only: it is a SHADOW COUNTER by default, never a rejection, so
    # a stanza that forgets to declare vision produces a warning, not an
    # outage. Do NOT arm `vision_capability_enforce` until the counter is
    # clean, or a missing declaration becomes a 400 on a working caller. ---
    vision: bool = False

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

    # This endpoint backs the fleet's CONVERSATIONAL lane, so `/readyz` fails
    # CLOSED when it is unhealthy or paused (tier2 split plan §8.0 req 4:
    # "the proxy stops dispatching rather than queueing to a dead box").
    #
    # Deliberately a DECLARATION rather than `kind: chat`, which spans tier1
    # router through tier3 reasoner — 503-ing fleet readiness because a
    # long-form authoring tier was down would be a different and wrong claim.
    # ⚠️ It marks a ROLE, so it MOVES with that role: at the Phase 3 cutover it
    # leaves `tier2-analyst` and lands on `tier2-chat`.
    readiness_critical: bool = False

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
    def backend_url(self) -> str:
        """Base URL the HTTP client is built against. ``base_url`` when set,
        otherwise the historic ``http://host:port``. This is the ONLY thing the
        transport should read — ``host``/``port`` remain for telemetry labels
        and for config that has not been written as a URL."""
        if self.base_url:
            return self.base_url.rstrip("/")
        return f"http://{self.host}:{self.port}"

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
    # Tier-3 failover opt-IN (§9.5 of the anvil2/V4-Flash plan). When an
    # endpoint with a declared `failover_to` is unhealthy, ONLY agents that set
    # this may be rerouted to the smaller model; everyone else gets the same
    # clean 503 they get today. Absent = False on purpose: a degraded answer is
    # a WORSE answer, and which callers can absorb one is a judgement made per
    # agent in the audit cycle (audit_dimensions.md § E9), never a default.
    #
    # Granularity is per-AGENT by operator decision. An agent that does both
    # conversational and extraction/authoring work is in or out as a whole, and
    # the extraction path governs — so it stays OUT.
    degrade_ok: bool = False


# ---------------------------------------------------------------------------
# Top-level proxy config
# ---------------------------------------------------------------------------

# THE routing table, DERIVED from the catalog (``models.yaml``). Each ROUTED
# endpoint produces one entry, keyed by its endpoint class; its connection comes
# from the provider it names. Slots/context here are STARTUP SEEDS — the capacity
# poller overwrites them from the backend where the engine publishes them.
#
# To add/move/retune a routed model, edit models.yaml; the per-class policy knobs
# (slot_affinity, fast_path_reserve_slots, dispatch_concurrency_cap,
# background_floor_pct, skip_discovery) live in its `policy:` block, and a key
# missing from `model_catalog._POLICY_PASSTHROUGH` is dropped in silence.
#
# 🚨 The shipped catalog is an EXAMPLE (`tier1`/`tier2`/`tier3`/`embed`/`rerank`,
# RFC 5737 addresses). Comments throughout this package cite measurements taken
# on a real fleet under ITS names — those are records of what was measured, not
# references to classes that exist here.
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
    return os.environ.get("ROADSTEAD_PROXY_THINKING", "1").strip().lower() not in (
        "0", "false", "no", "off", "",
    )


def shadow_egress_detect_enabled() -> bool:
    """WS-4 (2026-06-05): run ``verify_conformance`` on EVERY grammar-bearing
    structured response in SHADOW — log + count silent grammar-drops (parse
    fail / markdown-fence / non-conformance, i.e. llama.cpp dropped the grammar
    and ran free-form), but NEVER mutate the response. Zero caller risk, so
    default ON: the whole point is to collect a baseline silent-drop rate per
    call_site from live traffic. Env kill-switch ``ROADSTEAD_PROXY_SHADOW_EGRESS``."""
    return os.environ.get("ROADSTEAD_PROXY_SHADOW_EGRESS", "1").strip().lower() not in (
        "0", "false", "no", "off", "",
    )


def degeneration_guard_enabled() -> bool:
    """Egress repetition-LOOP guard (2026-06-10): detect a degenerate response —
    the same long n-gram repeated many times ('… the song of ## … the song of ##
    …'), a model failure mode any caller can hit on a long prompt — and RE-DISPATCH
    with an anti-repetition penalty. A 200-with-garbage is invisible to the
    transient-error and grammar checks, so this is the layer that catches it.
    Default ON. Env kill-switch ``ROADSTEAD_PROXY_DEGENERATION_GUARD``."""
    return os.environ.get("ROADSTEAD_PROXY_DEGENERATION_GUARD", "1").strip().lower() not in (
        "0", "false", "no", "off", "",
    )


def degeneration_shadow_only() -> bool:
    """When set, the degeneration guard only DETECTS + logs + counts (no
    re-dispatch) — the shadow-measure phase to confirm it never false-flags
    legitimate repetition (a song chorus) before it acts. Default OFF (the guard
    actively corrects). Env ``ROADSTEAD_PROXY_DEGENERATION_SHADOW``."""
    return os.environ.get("ROADSTEAD_PROXY_DEGENERATION_SHADOW", "0").strip().lower() in (
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
    ``ROADSTEAD_PROXY_UNIFORM_CORRECTION``."""
    return os.environ.get("ROADSTEAD_PROXY_UNIFORM_CORRECTION", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


#: How often the thinking canary makes its one real call, per endpoint.
#:
#: Every other probe in the poller is a cheap GET; this one is a GENERATION, so
#: the cadence is set by what it costs, not by how fast we want to notice. One
#: hour against a defect that went unnoticed for a DAY is already two orders of
#: magnitude better, and at ~1s/200 tokens per endpoint per hour the cost is
#: not worth optimising further. A code constant, not an env var: this is
#: tuning, and tuning belongs in code where it can be read next to its reason.
_THINKING_CANARY_INTERVAL_S = 3600.0


def thinking_canary_interval_s() -> float:
    return _THINKING_CANARY_INTERVAL_S


def thinking_canary_enabled() -> bool:
    """Standing guard (2026-08-24, ledger `tier3-reasoning-parser-default-
    mismatch`): prove each endpoint's DECLARED thinking switch still switches
    reasoning on the model actually loaded, with one real call.

    OBSERVABILITY ONLY — it raises a `thinking_switch_broken` WARNING and never
    changes routing, admission or any caller's payload, so default ON for the
    same reason the shadow-egress detector is. Env kill-switch
    ``ROADSTEAD_PROXY_THINKING_CANARY=0`` for the case where a backend must not
    be touched at all (a benchmark run wanting a quiet endpoint)."""
    return os.environ.get("ROADSTEAD_PROXY_THINKING_CANARY", "1") != "0"


def max_slots_reconcile_enabled() -> bool:
    """Step 4c (2026-07-01): SHADOW reconciler for the vLLM ``max_slots`` desync.
    vLLM's ``--max-num-seqs`` is not API-discoverable, so ``max_slots`` is
    config-seeded and can silently drift from the backend's real launch cap (the
    reasoner 32-vs-20 bug, Step 2b). This standing guard compares each vLLM
    endpoint's ``max_slots`` against its ``documented_max_num_seqs`` (mirrored from
    the serve script) and raises a ``max_slots_drift`` WARNING alert on mismatch.
    OBSERVABILITY ONLY — it never changes admission (zero caller-visible effect,
    like the shadow-egress detector), so default ON. Env kill-switch
    ``ROADSTEAD_PROXY_MAX_SLOTS_RECONCILE``."""
    return os.environ.get("ROADSTEAD_PROXY_MAX_SLOTS_RECONCILE", "1").strip().lower() not in (
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
    ``ROADSTEAD_PROXY_CACHE_DRIFT_ALARM``. See
    ``docs/llmproxy_prefix_cache_observability.md`` (Tier-2 step 3)."""
    return os.environ.get("ROADSTEAD_PROXY_CACHE_DRIFT_ALARM", "1").strip().lower() not in (
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
    == byte-identical. Env ``ROADSTEAD_PROXY_ENDPOINT_COOLDOWN``."""
    return os.environ.get("ROADSTEAD_PROXY_ENDPOINT_COOLDOWN", "0").strip().lower() in (
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
    Default ON. Env kill-switch ``ROADSTEAD_PROXY_STRUCTURED_VALIDITY``."""
    return os.environ.get("ROADSTEAD_PROXY_STRUCTURED_VALIDITY", "1").strip().lower() not in (
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
    ``ROADSTEAD_PROXY_SCHEMA_BACKSTOP``. See
    ``docs/llmproxy_phase3_schema_backstop_contract.md``."""
    return os.environ.get("ROADSTEAD_PROXY_SCHEMA_BACKSTOP", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def schema_backstop_shadow() -> bool:
    """When set (and the backstop enabled), the backstop only DETECTS + attempts
    repair in memory + logs what it WOULD return + counts — but returns the
    ORIGINAL response untouched (no swap, no retry re-dispatch, no fail-loud). The
    review window: measure how often the backstop fires and whether repair succeeds
    before it mutates live responses. Default OFF (the guard actively corrects when
    enabled). Env ``ROADSTEAD_PROXY_SCHEMA_BACKSTOP_SHADOW``."""
    return os.environ.get("ROADSTEAD_PROXY_SCHEMA_BACKSTOP_SHADOW", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def endpoint_cooldown_shadow() -> bool:
    """Step 4b SHADOW (review this first): count backend-fault failures + log a
    'would cool' + surface the trip on /v1/status, but DON'T actually pull the
    endpoint. Confirm the thresholds don't false-trip on this fleet's traffic, then
    flip enforce. Default OFF. Env ``ROADSTEAD_PROXY_ENDPOINT_COOLDOWN_SHADOW``."""
    return os.environ.get("ROADSTEAD_PROXY_ENDPOINT_COOLDOWN_SHADOW", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def cooldown_allowed_fails() -> int:
    """Backend-fault failures within the window that trip a cooldown (default 4 —
    above the 3-consecutive circuit so this catches the INTERMITTENT case).
    Env ``ROADSTEAD_PROXY_COOLDOWN_ALLOWED_FAILS``."""
    try:
        return max(1, int(os.environ.get("ROADSTEAD_PROXY_COOLDOWN_ALLOWED_FAILS", "4")))
    except ValueError:
        return 4


def cooldown_window_s() -> float:
    """Sliding window over which ``cooldown_allowed_fails`` is counted (default 60s).
    Env ``ROADSTEAD_PROXY_COOLDOWN_WINDOW_S``."""
    try:
        return max(1.0, float(os.environ.get("ROADSTEAD_PROXY_COOLDOWN_WINDOW_S", "60")))
    except ValueError:
        return 60.0


def cooldown_duration_s() -> float:
    """How long a tripped endpoint stays cooled before auto-recovery (default 30s).
    Env ``ROADSTEAD_PROXY_COOLDOWN_DURATION_S``."""
    try:
        return max(1.0, float(os.environ.get("ROADSTEAD_PROXY_COOLDOWN_DURATION_S", "30")))
    except ValueError:
        return 30.0


def thinking_reasoning_budget() -> int:
    """Tokens of reasoning headroom ADDED to a thinking request's max_tokens.
    Reasoning is generated <think> output and counts against max_tokens, so too
    small a budget truncates mid-reasoning (finish=length). Operator directive
    (2026-06-05): prefer short-term slowness over cutoff failures — start
    generous, watch the truncation metric, tune DOWN over time. Env override
    ``ROADSTEAD_PROXY_THINKING_BUDGET``."""
    try:
        return max(0, int(os.environ.get("ROADSTEAD_PROXY_THINKING_BUDGET", "8000")))
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
    Env override ``ROADSTEAD_PROXY_FORCED_REASONING_BUDGET``."""
    try:
        return max(0, int(os.environ.get("ROADSTEAD_PROXY_FORCED_REASONING_BUDGET", "1536")))
    except ValueError:
        return 1536


def _inflight_stream_interval_s() -> float:
    """Cadence of the SSE ``inflight`` reconcile frame that powers the live
    in-flight board. The instant ``call.dispatched``/``call.completed`` events do
    the real work; this just refreshes elapsed/queue/occupancy and recovers any
    missed event. Gated on connected clients (free when nobody's watching). Env
    override ``ROADSTEAD_PROXY_INFLIGHT_INTERVAL_S`` (clamped ≥0.25s)."""
    try:
        return max(0.25, float(os.environ.get("ROADSTEAD_PROXY_INFLIGHT_INTERVAL_S", "1.5")))
    except ValueError:
        return 1.5


@dataclass
class ProxyConfig:
    """Top-level proxy configuration."""
    port: int = 42161
    # 🚨 A COPY PER INSTANCE, not `dict(DEFAULT_ENDPOINTS)`. That was a shallow
    # copy of the dict around the SAME EndpointConfig objects, so every
    # ProxyConfig in a process shared them — and they are mutated at runtime:
    # capacity discovery writes max_slots and context_per_slot, and the poller
    # writes served_model_id. One service's discovery therefore reached into
    # another's config. It never mattered in production, where there is exactly
    # one, which is why it survived; in the suite it means one test's
    # `endpoints["tier2"].context_per_slot = 0` silently governs every test that
    # runs after it, and the failure surfaces somewhere unrelated.
    endpoints: dict[str, EndpointConfig] = field(
        default_factory=lambda: {k: dataclasses.replace(v)
                                 for k, v in DEFAULT_ENDPOINTS.items()})
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
    timeout_ceiling_interactive_s: float = _INTERACTIVE_CEILING_S
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
        default_priority (str enum name — e.g. "P3_INGESTION"),
        degrade_ok (bool — tier3 failover opt-in, § 9.5).
    Missing keys fall back to the AgentQuotaConfig dataclass defaults.
    ⚠️ A key not parsed below is SILENTLY IGNORED — adding a knob to
    AgentQuotaConfig is not enough to make it operator-reachable. Guarded by
    test_failover.py::test_degrade_ok_reaches_agent_config.

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
        if "degrade_ok" in cfg:
            kwargs["degrade_ok"] = bool(cfg["degrade_ok"])
        out[str(agent_id)] = AgentQuotaConfig(**kwargs)
    logger.info("loaded %d agent quota config(s) from %s", len(out), p)
    return out
