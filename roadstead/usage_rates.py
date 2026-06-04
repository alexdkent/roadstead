"""Cloud-equivalent rate table for the fleet-savings metric.

Ported from the host telemetry daemon's ``fleet_savings`` (which keyed on
canonical unit names). The proxy records ``endpoint`` as the QoS class
(chat/companion/…); non-LLM units pushed via ``/v1/calls/log`` record
their unit name (whisper-1/…) as the endpoint. We key on BOTH so whatever
lands in ``proxy_completions.endpoint`` resolves to the right rate.

Rates are USD per MILLION tokens (in, out) of the closed-source model the
local role *would* have hit if we'd gone external. Roles with no clean
per-token cloud analog (TTS, OCR, diarize, stream) contribute $0 — token
totals are still counted, only the cost is zero.
"""

from __future__ import annotations

# (in_rate, out_rate) USD per 1M tokens. Keyed by proxy endpoint-class.
_RATES_BY_CLASS: dict[str, tuple[float, float]] = {
    "thinker":    (15.0, 75.0),   # llama-thinker  → Opus-equivalent
    "companion":  (5.0,  25.0),   # qwen-composer (80B) long-context synthesis
    "chat":       (3.0,  15.0),   # qwen-analyst (VL-30B) → Sonnet-equivalent
    "gemma":      (1.0,   5.0),   # gemma-router  → Haiku-equivalent
    "gemma-hot":  (1.0,   5.0),   # gemma-greeter → Haiku-equivalent
    "embed":      (0.13,  0.0),   # bge-m3-embed  → text-embedding-3-large
    "rerank":     (2.0,   2.0),   # bge-reranker  → Cohere rerank
}

# Non-LLM units (pushed via /v1/calls/log) keyed by their unit name.
_RATES_BY_UNIT: dict[str, tuple[float, float]] = {
    "whisper-1":  (0.36, 0.0),    # ASR → OpenAI Whisper
    # diarize-gpu / chatterbox-tts / got-ocr / stream → $0 (no clean analog)
}

# Role-name aliases for the same rate (the proxy stores classes, but a row
# could carry a role name on legacy data / direct ingest).
_ROLE_ALIASES: dict[str, str] = {
    "llama-thinker": "thinker",
    "qwen-composer": "companion",
    "qwen-analyst":  "chat",
    "gemma-router":  "gemma",
    "gemma-greeter": "gemma-hot",
    "bge-m3-embed":  "embed",
    "bge-reranker":  "rerank",
}


def cloud_rate(endpoint: str) -> tuple[float, float]:
    """Return (in_rate, out_rate) USD/1M tokens for an endpoint/unit name,
    or (0.0, 0.0) when there is no cloud-equivalent rate."""
    ep = (endpoint or "").strip().lower()
    if ep in _RATES_BY_CLASS:
        return _RATES_BY_CLASS[ep]
    if ep in _RATES_BY_UNIT:
        return _RATES_BY_UNIT[ep]
    alias = _ROLE_ALIASES.get(ep)
    if alias:
        return _RATES_BY_CLASS.get(alias, (0.0, 0.0))
    return (0.0, 0.0)
