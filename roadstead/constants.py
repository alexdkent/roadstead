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
# Mirrors the client's ROADSTEAD_TIMEOUT_ADVICE_CAP_S (1800s) so server + client
# agree, and guards against a heavy-tailed background cell (recommended =
# p99*margin) yielding a pathological multi-hour deadline.
_SMART_DEFAULT_CAP_S = 1800.0

# The INTERACTIVE band's deadline ceiling. Single source of truth: `config`'s
# `TimeoutConfig.timeout_ceiling_interactive_s` defaults to this, and it is the
# right `min_timeout_s` floor for an interactive caller that supplies no
# deadline of its own. 🚨 Do NOT floor an interactive caller at
# `_SMART_DEFAULT_CAP_S` (1800s) — that is the BACKGROUND cap and is three
# times this, i.e. a floor above its own ceiling. That mistake was live on one
# registration for a day in the origin fleet, when a caller was promoted from
# the background band and its floor was left behind. `identity.py` now reports
# that shape at load for ANY registration, in either registry.
_INTERACTIVE_CEILING_S = 600.0

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
# 🚨 THIS FIRED ON WORK THAT WOULD HAVE SUCCEEDED, and raising the number was
# never the fix — C6, BUILT 2026-08-23. A silent wire does NOT mean a dead
# backend, and this gap alone cannot tell the two apart. TWO measured causes,
# one signature:
#   1. JIT. tier3 (DeepSeek-V4-Flash, TP=2) compiles kernels mid-request
#      (TileLang `mhc_pre_big_fuse_broadcast_*`, Triton
#      `_build_c128a_topk_metadata_kernel`; vLLM logs them at jit_monitor.py:135).
#      Compilation is CPU work, so the GPU sits at 0% and no token appears.
#      Measured during the incident: +1,972 prompt and +12 generation tokens in
#      the 30 s the client saw nothing.
#   2. TOOL-CALL BATCHING. vLLM's tool-call parser withholds argument deltas
#      until it can emit a complete `tool_call`. MEASURED 2026-08-23 against the
#      live tier3, same 2,500-token generation, engine otherwise idle:
#        no tools -> 530 frames, max wire gap 0.09 s
#        + tools  ->  29 frames, max wire gap 14.23 s
#      At 6,000 tokens: 48-58 frames, gaps 8.8-15.1 s. Those are IDLE numbers;
#      real load ran decode 21-27 tok/s against ~40 idle, which is what pushes a
#      15 s gap past 30 s. This is why every stalling caller was a tool-caller
#      (cli-write, cli-read, pool-observer) and why the stalls arrived in bursts
#      of the SAME prompt retried: 17 stalls/24 h on a healthy engine that was
#      generating 16-68 tok/s throughout.
# Ledger: `a-jit-compile-and-a-wedge-look-identical-to-an-inter-token-watchdog`.

#: Base gap for a request that CARRIES `tools`. Not a bigger guess — it is the
#: measured burst width (8-15 s idle) with headroom for the load multiplier
#: above, so the common bursty case never reaches the progress probe at all.
#: A tool-less stream keeps the tighter 30 s: it has no reason to be bursty.
_STREAM_GAP_TOOLS_S = 60.0

#: How many times the progress probe may push the deadline out before it stops
#: arguing. THE BOUND IS THE POINT: a genuinely slow-but-alive backend still
#: cannot burn the whole SLA, and a wedge that somehow keeps a global counter
#: moving (another request on the same engine) dies after at most this many
#: extensions instead of hanging forever. Worst case added latency for a truly
#: dead stream = _STREAM_GAP_MAX_EXTENSIONS x the gap in force.
_STREAM_GAP_MAX_EXTENSIONS = 4

#: Probe cadence, as a fraction of the gap in force. Must be < 0.5 so that TWO
#: samples (a baseline and a comparison) both land BEFORE the deadline fires —
#: at 0.3 x 30 s the baseline is taken at 9 s and the verdict at 18 s, leaving
#: 12 s of margin. Sampling only on expiry cannot work: `asyncio.timeout`
#: cancels the `async for`, so by the time it fires the stream is already dead
#: and there is nothing left to extend.
_STREAM_GAP_PROBE_FRACTION = 0.3

# Map the scheduler's payload_type to the completion `kind` tag (so the unified
# Inference page can group LLM sub-kinds; non-LLM calls pushed via /v1/calls/log
# carry their own kind — audio/imagegen/ocr/translate).
_PAYLOAD_KIND = {
    "chat_completion": "chat",
    "embedding": "embed",
    "rerank": "rerank",
}
