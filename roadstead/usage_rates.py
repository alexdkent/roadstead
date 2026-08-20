"""Cloud-equivalent rate model for the fleet-savings metric.

Every fleet endpoint maps to a REALISTIC cloud-rental price — the published cost
of renting the SAME (or closest hostable) open model on a real provider — so the
"savings" figure reflects what we'd actually pay to rent instead of self-host,
NOT a frontier-model (Opus/Sonnet/Haiku) fantasy. Rates researched 2026-07-12;
full sourcing + methodology in `originfleet/docs/fleet_cost_model.md`.

Two cost regimes:

  1. TOKEN-NATIVE LLM classes (chat / embed / rerank) — (in, out) USD per 1M
     tokens. Keyed by the proxy endpoint-CLASS (creative / companion / thinker /
     gemma / embed / rerank); every legacy role/alias resolves to one of
     those via `_ENDPOINT_CLASS`.

  2. PER-UNIT capabilities (speech / audio) — the producer packs its NATIVE
     billable unit into the token columns (a documented CONTRACT with the
     wrappers), decoded here:
       * STT / diarize / stem / lyrics:  input_tokens = audio_seconds * 100
         (nexus_stream/telemetry.py, wrappers/lyrics_service.py,
          wrappers/diarize_gpu_server.py — verified)
       * TTS:                            input_tokens = characters
         (orpheus_tts_server.py, wrappers/sesame_tts_server.py — verified)
     so cost is computed from audio-hours / characters, not a token rate.

  3. MEDIA generation (imagegen / video / musicgen) is per-image / per-second /
     per-generation and is NOT yet metered — those jobs don't push a
     `proxy_completions` row — so they contribute $0 today. The published
     per-unit rates are recorded in the comments below so wiring a job-count push
     later is a one-place change.

The single entry point is `cloud_cost_usd(endpoint, in_tokens, out_tokens)`.
`cloud_rate(endpoint)` is kept for backward-compat (token (in,out) rate only).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Token-native LLM classes — (input, output) USD per 1M tokens.
#    Anchor = closest hostable open model, mid-of-market (DeepInfra / OpenRouter
#    / Together / Groq / Fireworks). See docs/fleet_cost_model.md for the table.
# ---------------------------------------------------------------------------
_RATES_BY_CLASS: dict[str, tuple[float, float]] = {
    # 2026-07-31 tier migration: `composer`/`companion`/`thinker`/`reasoner`/`tier3` ALL resolve
    # to the tier3 endpoint now (class `thinker`), which runs **Laguna S 2.1 INT4** — a ~120B
    # sparse MoE (256 experts, top-10) with a 700K window, NOT the dense Qwen3.6-27B this row was
    # anchored to. Re-anchored to the heavier analog that the moved traffic was ALREADY priced
    # against as `companion`, because it is the same workload on a bigger model; keeping the
    # 32B anchor made the "$ saved" figure understate every heavy-tier call.
    "thinker":   (0.15, 0.60),   # Laguna S 2.1 INT4 (~120B MoE) → Qwen3-235B-A22B-Instruct-2507
    # ----------------------------------------------------------------------
    # ⏸ PENDING CUTOVER — B2 of docs/anvil2_tier3_deepseek_v4_flash_plan_2026-08.md
    #   (drafted 2026-08-20). NOT LIVE: Laguna is serving and the hardware has not
    #   arrived. Apply by deleting the `#⏸ ` marker and the live line above.
    #
    #⏸ "thinker":   (0.14, 0.28),   # DeepSeek-V4-Flash-DSpark → its OWN published API rate
    #
    #   UNITS — CHECKED, NOT ASSUMED. This dict is documented at the top of the file
    #   as "(input, output) USD per 1M tokens" and `cloud_cost_usd` computes
    #   `(ti/1e6)*rate[0] + (to/1e6)*rate[1]` (line ~168). So the plan's
    #   "$0.14/$0.28 per 1M" maps 1:1 with no conversion. (The per-UNIT capabilities
    #   further down this file pack a different quantity into the same columns —
    #   audio-seconds*100, characters — which is exactly why this needed checking.)
    #
    #   ANCHOR CHANGE, not just a number change. Every other row here is a PROXY:
    #   "the closest hostable open model, mid-of-market". This row stops being a proxy
    #   — V4 Flash is served by its own vendor at a published rate, so tier3 becomes
    #   the only EXACT-model anchor in the table. That is strictly better sourcing and
    #   worth stating in the comment when it lands.
    #
    #   🚨 TWO THINGS FOR THE REVIEWER, neither of which is a defect:
    #
    #   1. THE SAVINGS FIGURE STEPS DOWN ON CUTOVER DAY. Output falls 0.60 → 0.28, so
    #      the cloud-equivalent cost of a tier3 completion drops by more than half on
    #      the output leg. A 1M-in/1M-out call goes $0.75 → $0.42. Nothing is broken;
    #      the fleet just stops being credited for renting a 235B when it is running a
    #      284B-total/13B-active model that is cheap to rent. Say so, or someone
    #      "fixes" it back.
    #
    #   2. IT REPRICES HISTORY RETROACTIVELY. Unlike the 2026-08-19 tier2 split, this
    #      cutover does NOT change the endpoint CLASS — pre- and post-cutover rows in
    #      `proxy_completions` both carry `thinker`, so an in-place edit prices every
    #      Laguna row that ever ran at DeepSeek's rate. The tier2 split avoided this by
    #      minting a NEW class (`tier2-chat`) so both sides keep pricing.
    #      RECOMMENDATION: edit IN PLACE anyway and accept it. Minting a class here
    #      would mean changing `endpoint_class` in models.yaml, i.e. changing ROUTING
    #      and DRR accounting to fix a reporting artifact — and an endpoint class that
    #      is also an alias is the `endpoint-class-alias-collision` shape. Record the
    #      cutover DATE in the comment instead, so the discontinuity in the savings
    #      series is documented rather than mysterious.
    #
    #   TODO(confirm before applying) — the ONE thing not verifiable from the repo:
    #   whether $0.14 input is the CACHE-MISS rate or a blended figure. DeepSeek have
    #   historically published split cache-hit/cache-miss input pricing, and
    #   `proxy_completions` logs raw input tokens with no cache-hit split, so a blended
    #   number would understate. Confirm against the live price page at cutover; if it
    #   is split, use cache-miss (the conservative leg) and note why.
    #
    #   COUPLED TEST EDIT — MUST land in the same commit or the suite goes RED:
    #   tests/llmproxy/test_fleet_metrics.py hardcodes this rate five times —
    #   lines 121-124 (`thinker`/`llama-thinker`/`composer`/`companion` == (0.15,0.60))
    #   and line 138 (`cloud_cost_usd("thinker", 1e6, 1e6) == 0.75` → 0.42).
    #   B2's four-file list does NOT mention it. That file is outside this draft's scope.
    # ----------------------------------------------------------------------
    # HISTORICAL ONLY as of 2026-08-02: `companion` is no longer an endpoint class at all (the
    # 122B backup stopped being a proxy endpoint — its class name collided with the `companion`
    # alias and silently rerouted the backup to tier3; see models.yaml + the
    # `endpoint-class-alias-collision` ledger entry). No NEW row can carry this class. Kept only
    # so pre-cutover rows in the usage tables still price instead of falling to a default.
    "companion": (0.15, 0.60),   # historical: Qwen3.5-122B-A10B → Qwen3-235B-A22B-Instruct-2507
    "creative":  (0.15, 0.55),   # Qwen3.6-35B-A3B+vis → Qwen3-VL-30B-A3B-Instruct (vision billed as input tokens, no premium)
    # 2026-08-19 tier2 split: the CHAT half of `creative` moved to its own class on
    # jetty. Same weights, same quant, text-only (no mmproj on that box), so the same
    # anchor — the row exists because the CLASS is what proxy_completions records and
    # a class with no rate silently bills $0, which reads as "this lane is free".
    "tier2-chat": (0.15, 0.55),  # Qwen3.6-35B-A3B (text) → Qwen3-VL-30B-A3B-Instruct
    "gemma":     (0.04, 0.08),   # Gemma-4-E4B         → Gemma-3-4B-it (DeepInfra)
    "embed":     (0.01, 0.0),    # BGE-M3 (EXACT model on DeepInfra) — market floor
    "rerank":    (0.02, 0.02),   # BGE-Reranker-v2-m3  → Voyage/Jina lite (per-token)
}

# ---------------------------------------------------------------------------
# 2a. Per-audio-hour capabilities.  input_tokens = audio_seconds * 100
#     (the wrappers' contract), so hours = input_tokens / (100 * 3600).
# ---------------------------------------------------------------------------
_AUDIO_HOUR_RATE: dict[str, float] = {
    "stt":     0.111,   # Whisper-large-v3 (EXACT model on Groq): $0.111/hr
    "diarize": 0.12,    # Deepgram diarization add-on ($0.002/min); often bundled-free
    "stem":    0.75,    # HTDemucs (Replicate ~$0.045/track) as a per-audio-hour proxy
}
_HOURS_PER_INPUT_TOKEN = 1.0 / (100.0 * 3600.0)

# ---------------------------------------------------------------------------
# 2b. Per-character capability.  input_tokens = characters.
# ---------------------------------------------------------------------------
_TTS_CHAR_RATE = 15.0  # $/1M chars — tts-1 / Deepgram Aura-1 tier (no hosted Orpheus)

# ---------------------------------------------------------------------------
# 3. Media generation — declared for when a job-count push lands; $0 today.
#    imagegen $0.02/image (Fal FLUX.1-dev) · video $0.06/s (Fal LTX-2.3, exact) ·
#    musicgen $0.07/gen (Replicate, exact). Not metered → mapped to None below.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Endpoint / role / unit name  ->  rate key (a class above, or None for $0).
# `proxy_completions.endpoint` may carry the QoS class, a role name, or (for
# non-LLM units pushed via /v1/calls/log) the unit name — so we cover all three.
# ---------------------------------------------------------------------------
_ENDPOINT_CLASS: dict[str, str | None] = {
    # --- token-native ---
    "thinker": "thinker", "reasoner": "thinker", "llama-thinker": "thinker",
    # 2026-08-02: these five pointed at a `companion` CLASS that no longer
    # exists — and did not agree with the router even before it was removed.
    # `normalize_endpoint` has resolved companion/composer/qwen-composer/
    # nexus-companion to `thinker` since the 2026-07-30 tier migration, so this
    # map was attributing tier3 traffic to a class nothing was routed to. The $
    # figure happened to match (both rows were priced 0.15/0.60), which is
    # exactly why it went unnoticed. Point them where the traffic actually goes.
    "companion": "thinker", "composer": "thinker", "qwen-composer": "thinker",
    "nexus-companion": "thinker",
    # The 122B backup is no longer a proxy endpoint, and as of 2026-08-20 its
    # models.yaml stanza is gone entirely — it has no class to bill to.
    # `llama-companion` is its systemd UNIT name, never a routable alias.
    # DELIBERATELY KEPT despite the stanza's removal: historical `proxy_completions`
    # rows logged while it was `proxy_endpoint: true` (pre-2026-08-02) still carry
    # these strings, and this map must keep resolving them or billing/usage queries
    # over that older data start raising on an unknown endpoint.
    "llama-companion": None, "tier3-backup": None, "llama-companion-122b": None,
    "creative": "creative", "deckard": "creative", "deckard-31b": "creative",
    # 2026-08-19 tier2 split: these three left `creative` for `tier2-chat` on jetty.
    # Rows already in proxy_completions carry the endpoint string they were logged
    # with, so both sides must keep pricing — the pre-cutover ones are `creative`
    # traffic and the post-cutover ones are not, and mapping them all to one class
    # would make the split invisible in the cost tables.
    "tier2-chat": "tier2-chat",
    "companion-lite": "tier2-chat",
    "chat": "tier2-chat", "nexus-chat": "tier2-chat",
    "classify": "creative", "qwen-classify": "creative",
    "analyst": "creative", "qwen-analyst": "creative",
    "analyst-vision": "creative", "qwen-vision-8b": "creative",
    "qwen-vision-9b": "creative", "vision-8b": "creative", "vision9b": "creative",
    "gemma": "gemma", "gemma-router": "gemma", "nexus-gemma": "gemma",
    "router": "gemma", "classifier": "gemma",
    "gemma-hot": "gemma", "gemma-greeter": "gemma", "nexus-gemma-hot": "gemma",
    "embed": "embed", "bge-m3-embed": "embed",
    "rerank": "rerank", "bge-reranker": "rerank", "nexus-rerank": "rerank",
    # --- per-unit: speech / audio (input_tokens = audio_seconds*100) ---
    "stt": "stt", "nasbox-whisper": "stt", "whisper-1": "stt",
    "nasbox-whisper-lyrics": "stt", "anvil-lyrics": "stt", "lyrics": "stt",
    "diarize": "diarize", "nasbox-diarize": "diarize", "diarize-gpu": "diarize",
    "nexus-diarize": "diarize", "nexus-diarize-offline": "diarize",
    "stem": "stem", "nasbox-htdemucs": "stem", "htdemucs": "stem",
    # --- per-unit: TTS (input_tokens = characters) ---
    "tts": "tts", "orpheus-tts": "tts", "sesame-tts": "tts", "chatterbox-tts": "tts",
    # --- explicit $0 (no clean cloud analog, or would double-count) ---
    "stream": None,          # streaming audio mux; transcription counted under stt
    "ocr": None, "got-ocr": None,   # folded into creative-vision; no separate volume
    "audio-variation": None, "vampnet": None,   # no hosted analog
    "musicgen": None, "imagegen": None, "video": None,  # media: not yet metered
    "probe": None, "kb-metrics": None,           # infra/no-op
}


def _capability(endpoint: str) -> str | None:
    """Resolve an endpoint/role/unit name to its rate key (or None for $0)."""
    return _ENDPOINT_CLASS.get((endpoint or "").strip().lower(), None)


def cloud_cost_usd(endpoint: str, input_tokens: int, output_tokens: int = 0) -> float:
    """Cloud-equivalent USD this completion would have cost to rent externally.

    Handles BOTH regimes: token-native LLM classes (rate x tokens) and per-unit
    capabilities (audio-hours / characters decoded from the packed token columns).
    Unknown / no-analog / not-yet-metered endpoints return 0.0."""
    cap = _capability(endpoint)
    if cap is None:
        return 0.0
    ti = int(input_tokens or 0)
    to = int(output_tokens or 0)
    rate = _RATES_BY_CLASS.get(cap)
    if rate is not None:
        return (ti / 1e6) * rate[0] + (to / 1e6) * rate[1]
    hour_rate = _AUDIO_HOUR_RATE.get(cap)
    if hour_rate is not None:
        # input_tokens = audio_seconds * 100  ->  hours
        return ti * _HOURS_PER_INPUT_TOKEN * hour_rate
    if cap == "tts":
        # input_tokens = characters
        return (ti / 1e6) * _TTS_CHAR_RATE
    return 0.0


def cloud_rate(endpoint: str) -> tuple[float, float]:
    """Backward-compat: (in_rate, out_rate) USD/1M tokens for TOKEN-NATIVE
    classes; (0.0, 0.0) for per-unit / no-analog endpoints (whose real cost is
    computed by `cloud_cost_usd`, not a token rate)."""
    cap = _capability(endpoint)
    if cap and cap in _RATES_BY_CLASS:
        return _RATES_BY_CLASS[cap]
    return (0.0, 0.0)
