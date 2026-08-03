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

# Phase 5a — upper bound on the server-side SMART default deadline (used only
# when the smart_default_timeout flag is on, for callers that omit timeout_s).
# Mirrors the client's COLLECTIVE_TIMEOUT_ADVICE_CAP_S (1800s) so server + client
# agree, and guards against a heavy-tailed background cell (recommended =
# p99*margin) yielding a pathological multi-hour deadline.
_SMART_DEFAULT_CAP_S = 1800.0

# Phase 5C — time-to-first-token watchdog for streaming. Data (2026-05-31): the
# then-companion Qwen3-Next-80B (retired 2026-07-03; companion is now the
# Qwen3.5-122B-A10B on nexus) sometimes produced ZERO tokens on a large-context
# synth and burned the FULL deadline (180s), uselessly holding a scarce slot. The
# watchdog still applies to the 122B. If no first token
# arrives within this bound, abort + free the slot + return a deferrable error so
# the caller defers instead of the slot being dead for minutes. Capped to the
# caller's own deadline so a legitimately short request isn't over-waited.
_STREAM_TTFT_DEADLINE_S = 30.0

# Prefill-rate FLOOR (tokens/s) used to size the TTFT allowance for a large
# prompt. The old sizing hardcoded 1000 tok/s (+1s per 1k est input tokens) and
# killed legitimate prefills: live 2026-08-03, endpoint=thinker, a 123,466-token
# prompt aborted at elapsed_s=145.078 against exactly 30 + 115077/1000.
#
# Derivation (all measured; prefill rates recorded in models.yaml's tier3 stanza):
#   * tier3 prefill is SUPERLINEAR in length — 1,426 tok/s at 213K falls to
#     729 tok/s at 578K. The floor must be the slow end, not the fast one.
#   * halve it for contention/variance headroom             -> ~365 tok/s
#   * divide by the RESIDUAL est_input_tokens undercount, because the numerator
#     we divide is smaller than the real prompt        -> 365 / 1.53 = ~239 tok/s
# Rounded to 240 tok/s. Sanity: a real 578K-token request measures ~794s
# end-to-end; at the est_in that corresponds to (~378K) this yields a ~1,605s
# TTFT allowance — comfortably above it, and still bounded by the hard cap below.
#
# ⚠️ THE 1.53 IS COUPLED TO cost_model.estimate_input_tokens — RE-MEASURE IF IT
# CHANGES. This constant is not independent: it compensates for how far that
# estimator falls short of real tokenization, so "improving" the estimator
# without revisiting this number silently inflates every TTFT allowance.
# Measured 2026-08-03 against five live `pool` payloads (tool-heavy code
# transcripts) with the pre-fix estimator as a symmetric control:
#     actual 226,014 tok | pre-D4 est 119,460 (1.89x) | post-D4 est 147,735 (1.53x)
# The D4 fix (counting tool_calls / tool_use / tool_result) closed 1.89 -> 1.53;
# it did NOT close the gap, and the remainder is NOT a missing field. It is the
# global `4 chars ~= 1 token` divisor: these payloads actually tokenize at
# ~2.6 chars/token because JSON and code are denser than prose. That divisor is
# right for prose and is calibrated elsewhere, so it is deliberately NOT changed
# here — the compensation lives in this constant instead, where it is visible.
# `test_ttft_prefill_floor_matches_measured_undercount` pins the coupling.
#
# A generous TTFT allowance is only safe BECAUSE the inter-token gap watchdog
# takes over the instant the first token lands: a genuinely dead backend is
# still caught in ~_STREAM_INTERTOKEN_GAP_S, and a backend that never speaks at
# all is bounded by the hard cap. This number buys patience for a legitimate
# prefill, not tolerance for a hang.
_STREAM_PREFILL_FLOOR_TOK_S = 240.0

# The measured residual undercount of ``cost_model.estimate_input_tokens`` on
# tool-heavy payloads, AFTER the D4 fix (see the derivation above). Named so the
# coupling between the estimator and the TTFT floor is greppable and testable
# rather than buried in a comment.
_EST_IN_RESIDUAL_UNDERCOUNT = 1.53

# Absolute upper bound on a PROXY-CHOSEN streaming deadline that keeps getting
# extended because the stream is demonstrably making progress (see
# Lifecycle._stream_hard_cap_s). This is the backstop that keeps "a call that is
# emitting tokens is not hung" from becoming "a slot can be held forever".
#
# Only ever applies to a deadline the PROXY chose (the default/adaptive path).
# An explicit caller deadline (body timeout_s / X-Timeout-S) is a contract and
# is honoured as a hard wall exactly as before — it is never extended and never
# consults this cap.
#
# Background (P3/P4) is sized to clear the measured worst case with room: a
# 578,400-token tier3 request measures 794s end-to-end, so 3600s leaves ~4.5x.
# Interactive (P0-P2) is deliberately far tighter — a user-facing turn that has
# been streaming for 15 minutes has already failed its purpose, whatever the
# tokens say. Per-endpoint-class overrides come from models.yaml
# `stream_hard_cap_s` (build_class_stream_hard_caps), mirroring how
# timeout_floor_s / timeout_ceiling_s already flow.
_STREAM_HARD_CAP_BACKGROUND_S = 3600.0
_STREAM_HARD_CAP_INTERACTIVE_S = 900.0

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
