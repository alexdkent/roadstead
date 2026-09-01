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
`cloud_rate(endpoint)` is kept for backward-compat (token (in,out) rate only), and is also what
`spend.PriceBook` reads for its imputed fallback.

🚨 **Nothing here is money somebody owes.** Every number in this file is what renting the same class
of model WOULD have cost — a saving, not a bill. `spend.py` keeps that distinction as a property of
the price itself, because both kinds are USD per million tokens and nothing else would.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Token-native LLM classes — (input, output) USD per 1M tokens.
#    Anchor = closest hostable open model, mid-of-market (DeepInfra / OpenRouter
#    / Together / Groq / Fireworks). See docs/fleet_cost_model.md for the table.
# ---------------------------------------------------------------------------
_RATES_BY_CLASS: dict[str, tuple[float, float]] = {
    # Anchored to the class of model an endpoint serves, not to a specific one:
    # the rate is what renting THAT CLASS costs mid-market, so a model swap
    # inside a tier does not silently reprice history.
    #
    # 🚨 These are for the AVOIDED-cost metric, which only makes sense for
    # capacity you own. A REMOTE provider's endpoint costs actual money and must
    # not be priced from this table — as of Workstream D (2026-09-01) it is
    # metered from what the provider publishes, through `spend.PriceBook`, which
    # falls back HERE only for an endpoint nobody has priced. A remote class
    # therefore still maps to None below, and now for a second reason: a remote
    # endpoint that reached this table would book a SAVING for a call made over
    # the internet.
    "tier3": (0.14, 0.28),   # large sparse-MoE reasoner (~250B total / ~15B active)
    "tier2": (0.15, 0.55),   # mid MoE with vision (~30B total / ~3B active).
                             # Vision bills as input tokens; no premium.
    "tier1": (0.04, 0.08),   # small dense instruct model (~4B)
    "embed": (0.01, 0.0),    # a market-floor embedding endpoint
    "rerank": (0.02, 0.02),  # a per-token reranker
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
    # --- token-native: the endpoint classes, and every alias that resolves to
    # one. A class with no rate silently bills $0, which reads as "this lane is
    # free" rather than "nobody priced it" — so a new class belongs here on the
    # day it is added.
    "tier3": "tier3", "reasoner": "tier3", "thinker": "tier3", "composer": "tier3",
    "creative": "tier3", "vision": "tier3",
    "tier2": "tier2", "chat": "tier2", "analyst": "tier2",
    "tier1": "tier1", "router": "tier1", "small": "tier1", "fast-chat": "tier1",
    "classify": "tier1", "classifier": "tier1",
    "embed": "embed", "embeddings": "embed",
    "rerank": "rerank",
    # --- remote spill: real money, and not this table's business. Explicitly
    # None so a stray lookup reads as unpriced rather than free — `spend.py`
    # owns these, from the provider's published prices.
    "spill-chat": None, "spill-reasoning": None,
    # --- per-unit: speech / audio (input_tokens = audio_seconds*100). Roadstead
    # does not serve these itself; they arrive through /v1/calls/log from
    # whatever else a deployment runs, which is why the names here are
    # CAPABILITIES rather than endpoints.
    "stt": "stt", "whisper-1": "stt", "transcribe": "stt",
    "diarize": "diarize",
    "stem": "stem",
    # --- per-unit: TTS (input_tokens = characters) ---
    "tts": "tts", "speak": "tts",
    # --- explicit $0 (no clean cloud analog, or would double-count) ---
    "stream": None,          # streaming audio mux; transcription counted under stt
    "ocr": None,             # folded into vision; no separate volume
    "musicgen": None, "imagegen": None, "video": None,  # media: not yet metered
    "probe": None,           # infra/no-op
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
