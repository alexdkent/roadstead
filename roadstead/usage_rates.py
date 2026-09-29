"""Cloud-equivalent rate model for the fleet-savings metric.

Every fleet endpoint maps to a REALISTIC cloud-rental price — the published cost
of renting the SAME (or closest hostable) open model on a real provider — so the
"savings" figure reflects what we'd actually pay to rent instead of self-host,
NOT a frontier-model (Opus/Sonnet/Haiku) fantasy.

METHODOLOGY, so the numbers can be re-derived rather than trusted. Each rate is
the MID-OF-MARKET published price, on 2026-07-12, for renting the closest
hostable OPEN model to the one an endpoint actually serves — surveyed across
DeepInfra, OpenRouter, Together, Groq and Fireworks, taking the middle rather
than the cheapest so a single provider's loss-leader cannot inflate the saving.
Anchoring on the CLASS (parameter count and architecture) rather than on a named
model is deliberate: swapping the model behind an endpoint for another of the
same size must not silently reprice history. The exact anchor for each class is
in the comment beside it below.

🚨 These are STALE-BY-DESIGN and nothing recomputes them. Inference prices fall;
a rate left alone for a year overstates the saving. Re-survey before quoting the
figure anywhere it matters, and treat every number here as an order-of-magnitude
claim rather than an accounting one.

Two cost regimes:

  1. TOKEN-NATIVE LLM classes (chat / embed / rerank) — (in, out) USD per 1M
     tokens. Keyed by the rate CLASS (tier3 / tier2 / tier1 / embed / rerank);
     every endpoint name, role and legacy alias resolves to one of those via
     `_ENDPOINT_CLASS`.

  2. PER-UNIT capabilities (speech / audio) — Roadstead does not serve these; a
     deployment's own wrapper pushes them through `/v1/calls/log`, and packs its
     NATIVE billable unit into the token columns. That packing is a CONTRACT
     between the producer and this table, not an accident of the schema, and it
     is decoded here:
       * STT / diarize / stem / lyrics:  input_tokens = audio_seconds * 100
       * TTS:                            input_tokens = characters
     so cost is computed from audio-hours / characters, not a token rate. 🚨 A
     wrapper that pushes real token counts for one of these units will be priced
     as though its tokens were seconds — the unit name in `_ENDPOINT_CLASS`
     below is what selects the decoder, so a new audio capability must be added
     there and taught the packing, in that order.

  3. MEDIA generation (imagegen / comfyui / meshgen / videogen / musicgen) is
     per-image / per-second / per-generation and is NOT yet metered — those jobs
     don't push a `proxy_completions` row — so they contribute $0 today, by a
     DECLARED None rather than by omission. The published per-unit rates are
     recorded in the comments below so wiring a job-count push later is a
     one-place change.

Every name that reaches this table is either priced or deliberately zeroed. A
name that is neither books $0 without saying so, which is why the rollup
(`queue.savings_summary`) reports the ones it sees as `undeclared` and this
module logs each once — see :func:`is_declared_endpoint` and
:func:`warn_if_undeclared`.

The single entry point is `cloud_cost_usd(endpoint, in_tokens, out_tokens)`.
`cloud_rate(endpoint)` is kept for backward-compat (token (in,out) rate only), and is also what
`spend.PriceBook` reads for its imputed fallback.

🚨 **Nothing here is money somebody owes.** Every number in this file is what renting the same class
of model WOULD have cost — a saving, not a bill. `spend.py` keeps that distinction as a property of
the price itself, because both kinds are USD per million tokens and nothing else would.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Token-native LLM classes — (input, output) USD per 1M tokens.
#    Anchor = closest hostable open model, mid-of-market (DeepInfra / OpenRouter
#    / Together / Groq / Fireworks), surveyed 2026-07-12. See the module
#    docstring for how the anchor is chosen and why it is a class, not a model.
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
    # 🚨 THE SPLIT TIER-2 ENDPOINTS. A deployment that splits a tier across boxes
    # gets NEW endpoint names, and the bare `tier2` above stops matching any of
    # them — every call then books $0 while reading as a priced lane. Measured on
    # the reference fleet 2026-09-14: `tier2-chat` (150.1M in / 2.5M out) and
    # `tier2-analyst` (42.2M / 3.3M) had been booking nothing since the 2026-08-19
    # split, ~$32 of the ~$119 total that was missing.
    "tier2-chat": "tier2", "tier2-analyst": "tier2",
    # `tier2-flash` is a genuinely different model from the other two (a
    # vision+reasoning MoE on its own silicon), not a second copy of them.
    # ⚠️ ANCHORED AT tier2 CONSERVATIVELY AND IT MAY UNDER-PRICE: its stanza
    # deliberately publishes no parameter count (the build's /health does not
    # expose one), so there is no measured basis for the higher `tier3` anchor.
    # Under-claiming a saving is the safer error. Re-anchor if a parameter count
    # is ever measured — do not re-anchor on vibes about checkpoint size.
    "tier2-flash": "tier2", "halogen": "tier2", "flash": "tier2",
    "tier1": "tier1", "router": "tier1", "small": "tier1", "fast-chat": "tier1",
    "classify": "tier1", "classifier": "tier1",
    # `gemma` is the tier1 router's own endpoint key and carries GENUINE LLM
    # tokens — verified three ways on the reference fleet: it is a declared chat
    # alias in the catalog, `/v1/calls/log` REFUSES a push under a configured
    # endpoint name (409), and the push client drops proxy-native providers
    # before queueing. So nothing but real router traffic can land here.
    "gemma": "tier1", "gemma-router": "tier1", "gemma-greeter": "tier1",
    # `usersim` is a small DENSE model (~8B) serving a simulated user for
    # conversation drills, so its traffic is SYNTHETIC — and that does not make
    # it free: it occupies the same silicon a cloud rental would bill for, and
    # this table does not price by caller anywhere. Eval and drill traffic on
    # the tier endpoints is booked at the tier rate like any other call, so a
    # zero here would be the one lane treated differently for being a test.
    # Anchored at tier1 (small dense) rather than tier2 (a ~30B sparse MoE):
    # by the class rule — parameter count and architecture, not the name — an
    # 8B dense model is nearest the small dense class, and tier1's rate sits at
    # the low end of the mid-market 8B rentals, so it can only under-claim.
    "usersim": "tier1",
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
    # 🚨 HOST-QUALIFIED PUSH NAMES. A producer pushes `<host>-<unit>` as the
    # provider, and `config.normalize_endpoint` no longer strips a host prefix
    # (deliberately — that rule was one deployment's naming hardcoded into the
    # public code, and the catalog's `aliases:` is the only rewriting mechanism).
    # The rollup therefore books the name AS PUSHED, and a bare unit key above
    # never matches it: these lanes booked $0 while reading as priced. They are
    # spelled out rather than matched by pattern on purpose — a prefix rule would
    # price any `<x>-tts` somebody pushes, whatever it packs, and this file's
    # contract is that each name was verified at its producer first. Verified
    # 2026-09-28:
    #   cortex-diarize / cortex-diarize-offline / diarize-gpu:
    #       input_tokens = int(audio_seconds*100), output_tokens = segment count
    #   cortex-orpheus-tts / cortex-chatterbox-tts:
    #       input_tokens = chars_in, output_tokens = int(audio_seconds_out*100)
    "cortex-diarize": "diarize", "cortex-diarize-offline": "diarize",
    "diarize-gpu": "diarize",
    # 🚨 The UNIT names a deployment actually pushes, which are not the
    # capability names above. Each was verified at its producer's packing site
    # before being priced here — the contract is `input_tokens = seconds*100`
    # and a producer that pushed real tokens would be billed as though tokens
    # were seconds, so this list is evidence-backed, not pattern-matched:
    #   unraid-whisper / -lyrics : int(audio_seconds*100), out = transcript chars
    #   unraid-diarize           : int(audio_seconds*100), out = segment count
    #   unraid-htdemucs          : int(duration_s*100),    out = 4 (stem count)
    # Cross-checked against real deployment totals: transcript-chars-per-
    # audio-second lands in the expected range and differs between spoken and
    # sung input, and stem output is a fixed count per track rather than a
    # length-proportional one.
    "unraid-whisper": "stt", "unraid-whisper-lyrics": "stt",
    "unraid-diarize": "diarize",
    "unraid-htdemucs": "stem",
    # --- per-unit: TTS (input_tokens = characters) ---
    "tts": "tts", "speak": "tts",
    # 🚨 Both TTS producers pack `input_tokens = chars_in` and
    # `output_tokens = int(audio_seconds_out*100)` — output is AUDIO
    # CENTISECONDS, not tokens. The `tts` branch of cloud_cost_usd() ignores
    # output entirely, which is what keeps that honest; do NOT give the tts
    # class an output rate, or an ordinary backlog of synthesised speech gets
    # priced as tens of millions of output tokens.
    "orpheus-tts": "tts", "chatterbox-tts": "tts",
    "cortex-orpheus-tts": "tts", "cortex-chatterbox-tts": "tts",
    # --- explicit $0 (no clean cloud analog, or would double-count) ---
    "stream": None,          # streaming audio mux; transcription counted under stt
    "cortex-stream": None,   # the name `stream` is actually pushed under; same reason
    "ocr": None,             # folded into vision; no separate volume
    # Media: NOT METERED. No producer pushes a count for any of these — the
    # generators are dispatcher capabilities that run a job and return an
    # artifact, and no `push_call`/`calls/log` site exists for them (swept
    # 2026-09-28), so a rate here would multiply nothing. Declared rather than
    # omitted so the day one starts pushing is a decision, not a silent $0.
    "musicgen": None, "imagegen": None, "video": None,
    "comfyui": None, "meshgen": None, "videogen": None,
    "probe": None,           # infra/no-op
    # Deliberate $0 with a MEASURED reason, so nobody "fixes" them into a rate:
    #  · `stream` re-counts audio already billed under the whisper units — two of
    #    its call sites read `duration` straight off the whisper response.
    #  · `asr` pushes NO token fields at all (rows exist only so the Inference
    #    page's ASR card is not blank), so any rate on it still yields $0. The
    #    producer must be taught to push seconds before a rate means anything.
    #  · `got-ocr`'s producer was retired and its packing is UNKNOWN — unpriced
    #    rather than guessed.
    "asr": None, "cortex-asr": None,
    "got-ocr": None,
}

#: Every key above is a DECIDED endpoint: priced, or zero for a stated reason.
#: A name that is not here at all has been decided by nobody — see
#: :func:`is_declared_endpoint`.
_DECLARED_ENDPOINTS = frozenset(_ENDPOINT_CLASS)


def is_declared_endpoint(endpoint: str) -> bool:
    """Has this endpoint name been PRICED OR DELIBERATELY ZEROED?

    🚨 THIS IS THE DISTINCTION THE TABLE COULD NOT MAKE, AND ITS ABSENCE COST
    REAL MONEY. ``cloud_cost_usd`` returns 0.0 both for "this lane is genuinely
    free" and for "nobody has ever priced this name", and those two are
    indistinguishable in the output — a plausible number instead of a failure on
    a key nothing recognises. So a rename or a host move silently zeroes a lane
    and the total keeps looking reasonable.

    Measured on the reference fleet 2026-09-14, years after the keys drifted:
    ~$119 of ~$678 lifetime cloud-equivalent savings (17.5%) was unbooked across
    ten endpoint names, the largest being a TTS unit at $61.64. Nothing alerted,
    because every one of them returned a number rather than an error.

    Callers that summarise savings should check this against the endpoints they
    actually observe and report the misses LOUDLY. Pricing cannot be made to
    fail closed here — a hard raise inside ``cloud_cost_usd`` would take down
    the rollup for a typo — so the check belongs at the surface that can see the
    whole endpoint set at once.
    """
    return (endpoint or "").strip().lower() in _DECLARED_ENDPOINTS


#: Names already warned about, so a name that books $0 is announced ONCE per
#: process rather than once per rollup poll (a dashboard asks every few seconds).
#: Bounded, because the name is caller-supplied text — `/v1/calls/log` accepts
#: any endpoint string from an admin-scoped pusher — and an unbounded set is a
#: leak a misbehaving pusher can grow. A restart re-warns, which is the point of
#: a warning nobody may have been watching for the first one.
_WARNED_UNDECLARED: set[str] = set()
_WARNED_UNDECLARED_CAP = 256
_cap_notice_sent = False


def warn_if_undeclared(endpoint: str) -> bool:
    """True when ``endpoint`` has no pricing decision; WARNs once per name.

    Detection belongs at the surface that sees the whole endpoint set at once
    (see :func:`is_declared_endpoint`), and pricing itself stays fail-open: this
    only ever logs and returns a bool, so a rollup that calls it cannot be taken
    down by a name nobody recognises.
    """
    if is_declared_endpoint(endpoint):
        return False
    name = (endpoint or "").strip().lower()
    if name in _WARNED_UNDECLARED:
        return True
    if len(_WARNED_UNDECLARED) >= _WARNED_UNDECLARED_CAP:
        global _cap_notice_sent
        if not _cap_notice_sent:
            _cap_notice_sent = True
            logger.warning(
                "usage_rates: %d undeclared endpoint names already reported; "
                "further ones are counted in /v1/fleet/savings `undeclared` but "
                "no longer logged", _WARNED_UNDECLARED_CAP)
        return True
    _WARNED_UNDECLARED.add(name)
    logger.warning(
        "usage_rates: endpoint %r has NO pricing decision, so it books $0 in the "
        "savings total. Declare it in _ENDPOINT_CLASS — priced, or None with the "
        "reason. Its volume is listed under `undeclared` in /v1/fleet/savings.",
        name)
    return True


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
