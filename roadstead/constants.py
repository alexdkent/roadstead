"""Shared numeric constants for the LLMProxy service (de-monolith Step 3).

Extracted so both ``service.py`` (Lifecycle retry path) and ``correction.py``
(degeneration re-dispatch) can reference them without a circular import.
"""

# Phase 1.3 — bounded in-proxy retry for transient backend failures (defer,
# don't drop). Only retry if at least this much of the caller's deadline remains
# after a short backoff, so a retry never starts work the caller will abandon.
_MIN_RETRY_BUDGET_S = 5.0
_RETRY_BACKOFF_S = 0.5

# Phase 3.1 — single server-side submit-timeout default (was a 180.0 literal in
# three places). Client-side extend-only advice still applies on top per role/
# tier; this is only the floor when a caller supplies no timeout.
_DEFAULT_TIMEOUT_S = 180.0

# Phase 5C — time-to-first-token watchdog for streaming. Data (2026-05-31): the
# companion 80B sometimes produces ZERO tokens on a large-context synth and burns
# the FULL deadline (180s), uselessly holding a scarce slot. If no first token
# arrives within this bound, abort + free the slot + return a deferrable error so
# the caller defers instead of the slot being dead for minutes. Capped to the
# caller's own deadline so a legitimately short request isn't over-waited.
_STREAM_TTFT_DEADLINE_S = 30.0

# Inter-token no-progress watchdog (stall resilience, 2026-06-06). The TTFT
# bound above guards only the FIRST token; a backend that streams a few tokens
# then STALLS mid-generation (observed during the thinker contention episode:
# generation throughput collapsing to ~0 tok/s with the request resident) would
# still burn the rest of the SLA. Each token resets this gap deadline (bounded
# by the caller's remaining SLA), so a mid-stream stall aborts in ~this many
# seconds and frees the slot with a deferrable error instead of hanging to 180s.
_STREAM_INTERTOKEN_GAP_S = 30.0

# Map the scheduler's payload_type to the completion `kind` tag (so the unified
# Inference page can group LLM sub-kinds; non-LLM calls pushed via /v1/calls/log
# carry their own kind — audio/imagegen/ocr/translate).
_PAYLOAD_KIND = {
    "chat_completion": "chat",
    "embedding": "embed",
    "rerank": "rerank",
}
