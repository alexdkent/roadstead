"""Lifecycle — admission -> scheduling -> dispatch -> response for the LLMProxy.

The request hot path: endpoint resolution, the `handle_submit` admission
sequence, the sync/stream response paths, the scheduler loop, backend dispatch
(`execute_*`), and completion/timeout recording. A behavior object over the
shared :class:`ProxyState`; it calls the Correction and Health collaborators by
reference. ProxyService keeps identical-signature delegators (contract §2).

`_openai_error` is a module function here (not a method) so both Lifecycle and
the HTTP handlers can build the OpenAI-shaped error envelope without a cycle.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import math
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from .backend import (
    BackendError,
    BackendResponse,
    BackendTimeout,
    BackendUnavailable,
    coerce_token_count,
    extract_cached_tokens,
)
from .config import (
    LLMPriority,
    PriorityBand,
    normalize_endpoint,
    structured_validity_guard_enabled,
    uniform_correction_enabled,
)
from .constants import (
    _DEFAULT_TIMEOUT_S,
    _MIN_RETRY_BUDGET_S,
    _PAYLOAD_KIND,
    _RETRY_BACKOFF_S,
    _SMART_DEFAULT_CAP_S,
    _STREAM_HARD_CAP_BACKGROUND_S,
    _STREAM_HARD_CAP_INTERACTIVE_S,
    _STREAM_GAP_MAX_EXTENSIONS,
    _STREAM_GAP_PROBE_FRACTION,
    _STREAM_GAP_TOOLS_S,
    _STREAM_INTERTOKEN_GAP_S,
    _STREAM_PREFILL_FLOOR_TOK_S,
    _STREAM_TTFT_DEADLINE_S,
)
from .correction import _EMPTY_RESCUE_MIN_TOKENS, _ToolCallStreamSanitizer
from .cost_model import context_fit, estimate_input_tokens
from .enriched import (
    WIRE_ENRICHED,
    WIRE_OPENAI,
    attribution,
    cost_block,
    enrichment_headers,
    timing_block,
)
from .observability import MetricsSample, RequestLogRecord
from .on_demand import OnDemandUnavailable
from .scheduler import CompletionRecord, DispatchDecision, QueuedRequest
from .sse_hub import DROP_SENTINEL

if TYPE_CHECKING:
    from .config import EndpointConfig  # noqa: F401
    from .correction import Correction
    from .health import Health
    from .state import ProxyState

logger = logging.getLogger(__name__)


# A timeout whose caller-APPLIED deadline is below this fraction of the model's
# size-aware recommended time is a best-effort sub-floor give-up (gemma greeter
# advisory ~0.9s / sidekick.extract ~3s vs the 60s gemma floor), NOT reliable
# backend-stall evidence. Tagged on the timeout metric (best_effort) so the
# endpoint_stalled heuristic can exclude it — including PROXY-initiated aborts
# (TTFT watchdog / endpoint-paused fast-fail) that `premature` can't flag. See
# observability._ENDPOINT_STALL_MIN_TIMEOUTS (2026-07-05 gemma stall-burst fix).
_STALL_BEST_EFFORT_RATIO = 0.5

# The COOLDOWN exclusion uses a DELIBERATELY TIGHTER ratio than the stall
# heuristic above, and the difference is load-bearing. `recommended_ms` is
# floored by timeout_model.FLOOR_S, which is 60-120s on the chat tiers — so at
# 0.5 nearly EVERY interactive caller (whose deadline is single-digit seconds)
# would read as best-effort and the cooldown would go inert for timeouts, which
# is a safety mechanism silently disarmed rather than a bug fixed.
#
# 0.1 excludes only egregious under-budgeting — the case this exists for is
# `orchestrator.gemma_greeter_advisory` at 0.9-1.6s against tier1's 60s floor,
# a ratio of ~0.015-0.027, two orders below the bar. A caller that gave the
# backend even a tenth of the recommended time still cools it on a genuine
# stall. Raising this toward 0.5 re-disarms the cooldown; do not.
_COOLDOWN_BEST_EFFORT_RATIO = 0.1


#: Content-part type tags that carry an IMAGE. Both wire shapes appear here:
#: OpenAI multipart (``{"type": "image_url", ...}``) and Anthropic typed blocks
#: (``{"type": "image", "source": {...}}``) — the proxy accepts both and
#: rewrites Anthropic→OpenAI in ``backend.py``, so a check that knew only one
#: shape would be blind to half the fleet's callers.
_IMAGE_PART_TYPES = frozenset({"image", "image_url"})


def _carries_image(body: dict) -> bool:
    """True if this request puts an image on the wire.

    🚨 LOOKS IN BOTH PLACES ON PURPOSE. Every real caller reaches this function
    through the SUBMIT ENVELOPE — `{"agent_id", "endpoint", "payload": {...}}` —
    with the actual chat request nested under ``payload``: the OpenAI front door
    builds that wrapper in `http_handlers.handle_openai_chat`, and
    `framework/llm_proxy_client` builds the same one. A first version of this
    read only top-level ``messages``, passed its unit tests against a
    hand-written body, and then fired on NOTHING in production — the counter sat
    at {} through a live 500 while the gate believed no image had ever been
    sent. Top-level is kept as well because it costs one `.get` and makes the
    function safe to call on either shape.

    Deliberately CHEAP and shallow: walks message ``content`` lists looking at
    the ``type`` tag only, never touching the base64 payload — this runs on the
    hot admission path for every request, including the text-only majority.
    Returns False on any unexpected shape rather than raising; a telemetry gate
    must never be able to reject a valid request.
    """
    try:
        payload = body.get("payload")
        for source in (body, payload if isinstance(payload, dict) else None):
            if source is None:
                continue
            messages = source.get("messages")
            if not isinstance(messages, list):
                continue
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if not isinstance(content, list):
                    continue      # a plain string cannot carry an image
                for part in content:
                    if (isinstance(part, dict)
                            and part.get("type") in _IMAGE_PART_TYPES):
                        return True
    except Exception:  # noqa: BLE001 — telemetry must not break admission
        return False
    return False


def _timeout_below_recommended(
    applied_timeout_s, recommended_ms, ratio: float = _STALL_BEST_EFFORT_RATIO,
) -> bool:
    """True if the applied deadline is far under the recommended time — a
    best-effort caller that never gave the backend a fair chance. Cheap + total:
    a missing/zero recommended or applied returns False (counts as genuine).

    ``ratio`` defaults to the stall heuristic's 0.5; the cooldown passes the
    much tighter ``_COOLDOWN_BEST_EFFORT_RATIO`` — see its comment for why the
    two consumers must NOT share one threshold."""
    applied_ms = float(applied_timeout_s or 0.0) * 1000.0
    return bool(
        recommended_ms
        and applied_ms
        and applied_ms < float(recommended_ms) * ratio
    )


# OpenAI-shaped error envelope — shared by Lifecycle + the HTTP front door.
def _openai_error(
    message: str, err_type: str, status_code: int, code: str | None = None,
) -> JSONResponse:
    """OpenAI-shaped error envelope for the /v1/chat/completions front door
    (goose-cli + any OpenAI client expects ``{"error": {...}}``). ``code``
    is the machine-readable taxonomy field (M2) — additive; the message
    substrings the fleet's deferral classifier sniffs are unchanged."""
    err: dict = {"message": str(message), "type": err_type}
    if code:
        err["code"] = code
    return JSONResponse({"error": err}, status_code=status_code)


def _split_coalesced_finish_chunk(parsed: dict) -> tuple[dict, dict] | None:
    """Split a chunk that carries BOTH `delta.content` and a non-null
    `finish_reason` into (content chunk, empty-delta terminal chunk).

    Returns `None` when the chunk is not coalesced and must pass through
    byte-identically — which is the overwhelming majority.

    ## Why this exists

    vLLM stamps `finish_reason` onto whatever delta the same engine iteration
    produced, so when its producer gets ahead of the SSE consumer the final
    content and the finish are MERGED into one chunk
    (`RequestOutputCollector.add(..., aggregate=True)`; its docstring says so
    outright). Measured on this fleet 2026-08-24: **30 of 34** responses.

    That is schema-legal — OpenAI's OpenAPI spec makes `finish_reason` a
    required, nullable field on EVERY choice and never ties it to `delta`
    being empty — but it is not what OpenAI's own service emits, and a client
    that only ever saw the reference implementation may not handle it.

    One does not: the Beacon agent runtime skips its own `finish_reason`
    capture when a content chunk's text trips an SSE-lookalike heuristic
    (`_provider_stream_text_may_be_sse` -> `continue`, jumping over the
    capture at the bottom of its loop). It then reads the turn as a
    mid-stream drop, stamps it `length`, and spends a SECOND full model call
    "continuing" an answer that was already complete. Root-caused by
    instrumenting the running container; upstream issue #91373 is open with a
    different (and measurably wrong) diagnosis, so there is no version to
    upgrade to. Ledger:
    `a-streams-finish_reason-rides-alone-on-a-chunk-nobody-misses`.

    Splitting removes ONE of the two conditions that failure needs, from the
    only side we control. It is a normalisation toward the canonical shape,
    not a workaround pointed at one client: any consumer that handles
    OpenAI's own output already handles what this produces.

    🚨 Deliberately narrow. Tool-call deltas carry `tool_calls`, not
    `content`, so a tool-call finish chunk is NOT split and
    `_ToolCallStreamSanitizer`'s finalize-on-finish path is untouched.
    `usage` stays on the content chunk so the synthetic terminal one can
    never be mistaken for the usage chunk (`choices == []` is the documented
    test for that, and this chunk has choices).
    """
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    split_needed = False
    for ch in choices:
        if not isinstance(ch, dict):
            return None
        delta = ch.get("delta")
        content = delta.get("content") if isinstance(delta, dict) else None
        if ch.get("finish_reason") is not None and content:
            split_needed = True
    if not split_needed:
        return None

    head = copy.deepcopy(parsed)
    tail = copy.deepcopy(parsed)
    for ch in head.get("choices", []):
        ch["finish_reason"] = None
    for ch in tail.get("choices", []):
        # Mirror the shape vLLM itself emits when it has a finish with no
        # text: an empty delta object, not a dropped key.
        ch["delta"] = {}
    tail.pop("usage", None)
    return head, tail


def _enriched_done(state, req, event: dict) -> dict:
    """The producer's flat ``done`` frame, reshaped onto the enriched wire.

    The producer (``execute_streaming``) emits what it MEASURED — queue wait,
    backend latency, TTFT, token counts — in one flat frame, and knows nothing
    about wires. Everything below is derivation: the same four blocks the
    non-streaming envelope carries, so a caller that handles one handles both.
    """
    usage = event.get("usage") or {}
    in_tok = int(usage.get("prompt_tokens") or 0)
    out_tok = int(usage.get("completion_tokens") or 0)
    attrib = attribution(state, req)
    attrib["cost"] = cost_block(state, req.endpoint, in_tok, out_tok)
    return {
        "type": "done",
        "request_id": req.request_id,
        # 🚨 Reported at the END as well as at the start, and it is the END one
        # that is authoritative: failover and spill both move a request AFTER
        # the `accepted` frame is on the wire. A caller that trusted the opening
        # frame's attribution would be told the endpoint we intended rather than
        # the one that answered — the exact silent-substitution failure this API
        # exists to close.
        "attribution": attrib,
        "timing": timing_block(
            req,
            queue_wait_ms=event.get("queue_wait_ms") or 0.0,
            backend_latency_ms=event.get("backend_latency_ms") or 0.0,
            ttft_ms=event.get("ttft_ms"),
        ),
        "usage": {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "slot_seconds": round(req.estimated_cost_ss, 3),
        },
    }


class Lifecycle:
    """The request hot path over the shared ProxyState."""

    def __init__(self, state: "ProxyState", correction: "Correction",
                 health: "Health") -> None:
        self.state = state
        self.correction = correction
        self.health = health

    def resolve_endpoint(self, body: dict) -> str:
        """Pick the routing endpoint, honoring the requested ``model`` over
        the caller's client role (OpenAI-consistent). The client role is the
        default when no model is given.

        Only overrides for chat completions when the payload's ``model``
        maps to a *known* endpoint that differs from the submit endpoint —
        so embeddings (no model) and rerank (model='bge', not an endpoint)
        fall through untouched. Logs every reconciliation loudly so a
        mis-wired client (role != requested model) stays visible.
        """
        submit_ep = normalize_endpoint(body.get("endpoint", "chat"))
        if body.get("payload_type", "chat_completion") != "chat_completion":
            return submit_ep
        model = (body.get("payload") or {}).get("model")
        if not model:
            return submit_ep
        model_ep = normalize_endpoint(str(model))
        if model_ep in self.state.config.endpoints and model_ep != submit_ep:
            logger.warning(
                "route reconcile: client endpoint=%s but model=%s -> routing to %s "
                "(caller=%s call_site=%s)",
                submit_ep, model, model_ep,
                body.get("caller_id"), body.get("call_site"),
            )
            return model_ep
        return submit_ep
    async def handle_submit(
        self, body: dict, request: Request, *, wire: str = WIRE_ENRICHED,
    ) -> Response:
        # 🚨 ``wire`` varies ONLY the response serialization — never a routing,
        # admission, correction or accounting decision. Two shapes:
        #
        #   WIRE_OPENAI    the bare OpenAI chat.completion / chat.completion.chunk
        #                  + [DONE] stream, set by /v1/chat/completions and
        #                  /v1/embeddings. Enrichment rides in headers, never in
        #                  the body — see ``enriched.ENRICHMENT_HEADERS``.
        #   WIRE_ENRICHED  the Roadstead envelope, with its attribution and
        #                  timing blocks. What /rs/v1/chat serves.
        #
        # The enqueue / scheduler / grammar / cache / DRR / telemetry path is
        # identical for both, and keeping it that way is the whole reason this
        # is one method with a serialization flag rather than two doors: a
        # second hot path is a second set of admission bugs.
        if self.state.draining.is_set():
            # Phase 2.1: refuse new work while draining for shutdown so it defers
            # to the (about-to-restart) next instance instead of being dropped.
            # Phase 5C: include the "backpressure" marker so the body is
            # classified deferrable (is_deferrable_llm_error) by BOTH the sync
            # and streaming clients — without it, a streaming turn caught mid-
            # SIGTERM surfaced a hard error instead of deferring cleanly.
            err = "proxy draining for shutdown — backpressure"
            if wire == WIRE_OPENAI:
                return _openai_error(err, "backpressure", 503, code="draining")
            return JSONResponse(
                {"status": "error", "error": err, "code": "draining"},
                status_code=503)

        # Identity, BEFORE anything reads the body. Until 2026-09-01 this door
        # had no access control at all — the OpenAI doors were ACL-gated and
        # `/v1/submit` was not, on the same port — so a caller could both reach
        # it unenrolled and claim any `agent_id` it liked, including one with a
        # better DRR weight. The fair-share key was self-asserted.
        #
        # The OpenAI doors resolve first and pass the answer down in the
        # envelope; resolving again here costs a dict lookup and makes this door
        # safe on its own rather than safe by virtue of who calls it. Gating
        # before endpoint resolution also stops an unenrolled caller enumerating
        # endpoint names off the 404/tally path.
        resolved = self.state.identity.resolve(request)
        if not resolved.ok:
            denial = resolved.denial
            if wire == WIRE_OPENAI:
                return _openai_error(denial.message, denial.openai_type,
                                     denial.status, code=denial.code)
            return JSONResponse(
                {"status": "error", "error": denial.message,
                 "code": denial.code},
                status_code=denial.status)
        principal = resolved.principal

        # 🚨 A KEY overrides a body-declared identity; an ADDRESS only fills in
        # one that was omitted. A verified credential is a stronger statement
        # about who is calling than anything in the body, so letting the body
        # win would launder a claim past the credential and put the traffic on
        # somebody else's DRR budget. A source address is a much weaker signal —
        # it identifies a host, not a caller, and several callers legitimately
        # share one — so there it is the body that knows better, and the
        # registration only supplies what the body left out.
        if principal.authenticated:
            agent_id = principal.agent_id
        else:
            agent_id = str(body.get("agent_id") or principal.agent_id)
        # Same rule for the band: a declared priority wins, the identity's
        # default fills in. Compared against None rather than truthiness —
        # P0_REALTIME is 0, and `or` would silently promote realtime traffic to
        # the identity default.
        declared_priority = body.get("priority")
        if declared_priority is None:
            declared_priority = principal.priority
        # 🚨 Recorded BEFORE the standing is taken, so a caller's own request
        # counts toward the rate it is judged on. The alternative — record after
        # — lets a caller sit exactly one request under its threshold forever.
        self.state.record_request(agent_id)
        # Workstream D: a caller over its daily spend cap drops one band; since
        # 2026-09-01, so does one over its request-rate threshold, and crossing
        # both still costs exactly one band (see ProxyState.spend_demote).
        # 🚨 Applied HERE, to the priority, and nowhere near the admission
        # decision — the whole doctrine is that a threshold costs a caller its
        # PLACE IN THE QUEUE and never its access to local capacity. There is
        # deliberately no branch below this line that can turn an over-threshold
        # caller into an error.
        declared_priority = self.state.spend_demote(
            agent_id, LLMPriority.coerce(declared_priority,
                                         default=LLMPriority.P1_TURN_SUPPORT))

        now = time.monotonic()

        # Unknown-endpoint gate. Without it a typo'd role enqueues into a
        # queue the dispatch loop never visits and the caller blocks its FULL
        # timeout_s before a useless 504 (the OpenAI front door already 404s).
        # Shadow (flag off): count + WARN, behaviour unchanged. Enforce: fast
        # 404 whose message deliberately carries NO deferrable marker — a
        # typo is deterministic and must surface, not defer-loop.
        endpoint = self.resolve_endpoint(body)
        if endpoint not in self.state.config.endpoints:
            caller = str(body.get("caller_id") or body.get("agent_id") or "unknown")
            tally = self.state.unknown_endpoint_submits.setdefault(
                endpoint, {"count": 0, "callers": {}})
            tally["count"] += 1
            tally["callers"][caller] = tally["callers"].get(caller, 0) + 1
            if self.state.flags.get("unknown_endpoint_enforce"):
                err = (f"unknown endpoint {endpoint!r} — no such model/role "
                       f"(known: {sorted(self.state.config.endpoints)})")
                if wire == WIRE_OPENAI:
                    return _openai_error(
                        err, "model_not_found", 404, code="unknown_endpoint")
                return JSONResponse(
                    {"status": "error", "error": err, "code": "unknown_endpoint"},
                    status_code=404)
            logger.warning(
                "unknown endpoint %r submitted by %s (call_site=%s) — SHADOW: "
                "request will wait out its full deadline; flip "
                "unknown_endpoint_enforce for a fast 404",
                endpoint, caller, body.get("call_site"))

        # Vision-capability gate. An image sent to an endpoint with no mmproj
        # gets a backend HTTP 500 "image input is not supported" — which the
        # CALLER then usually swallows into an empty string, because a vision
        # helper that returns "" is indistinguishable from one that honestly
        # could not read the picture. That is how this shipped silently for a
        # day (ledger `a-role-rename-carried-vision-to-a-text-only-box`): the
        # catalog declared `vision: false` the whole time and NOTHING READ IT.
        # This makes the declaration load-bearing.
        #
        # Shadow by default, exactly like the unknown-endpoint gate above:
        # count + WARN, behaviour unchanged. `vision_capability_enforce` turns
        # it into a fast 400 — deliberately NOT deferrable, since a text-only
        # backend will never grow an mmproj by being retried. Enforce is left
        # OFF because a WRONG `vision:` flag in models.yaml would then take a
        # working caller down instantly; the counter tells you the flag is
        # right before you arm it.
        if _carries_image(body):
            entry = self.state.config.endpoints.get(endpoint)
            if entry is not None and not getattr(entry, "vision", False):
                caller = str(body.get("caller_id") or body.get("agent_id")
                             or "unknown")
                tally = self.state.vision_capability_violations.setdefault(
                    endpoint, {"count": 0, "callers": {}})
                tally["count"] += 1
                tally["callers"][caller] = tally["callers"].get(caller, 0) + 1
                if self.state.flags.get("vision_capability_enforce"):
                    err = (f"endpoint {endpoint!r} has no vision capability — "
                           "this request carries image content and the backend "
                           "would answer 500 'image input is not supported'")
                    if wire == WIRE_OPENAI:
                        return _openai_error(
                            err, "invalid_request_error", 400,
                            code="vision_not_supported")
                    return JSONResponse(
                        {"status": "error", "error": err,
                         "code": "vision_not_supported"}, status_code=400)
                logger.warning(
                    "IMAGE sent to non-vision endpoint %r by %s (call_site=%s) "
                    "— the backend will 500 and the caller will most likely "
                    "swallow it into an empty result. Route this to a "
                    "vision-capable role.",
                    endpoint, caller, body.get("call_site"))

        # Robust timeout_s. A caller either OMITS a deadline (→ the smart/flat
        # default via resolve_default_timeout; an explicit None counts as
        # omitted) or SUPPLIES one (coerced — a malformed value must default,
        # not 500 the request). A supplied value ALWAYS wins over the default.
        # (priority is soft-defaulted inside QueuedRequest.create; payload/
        # endpoint/call_site already use safe .get defaults.)
        raw_timeout = body.get("timeout_s")
        # Tracks whether the deadline is OURS or the CALLER's — the identity
        # floor below applies only to a deadline the proxy chose.
        deadline_is_default = raw_timeout is None
        if raw_timeout is None:
            timeout_s = self.resolve_default_timeout(endpoint, body)
        else:
            try:
                timeout_s = float(raw_timeout)
                # Reject non-positive, NaN, AND non-finite (+inf slips past
                # ``>0`` and ``!=`` — a supplied +inf would otherwise stamp an
                # UNBOUNDED sync deadline: asyncio.wait_for(timeout=inf) +
                # backend.call(timeout_s=inf) never reclaims the slot on a hung
                # backend. Fall through to the default, which is finite+capped).
                if not (timeout_s > 0) or not math.isfinite(timeout_s):
                    raise ValueError("timeout_s must be a positive, finite number")
            except (TypeError, ValueError) as exc:
                logger.warning("submit: bad timeout_s %r (%s); using default",
                               raw_timeout, exc)
                timeout_s = self.resolve_default_timeout(endpoint, body)
                # A value we could not use is not a caller deadline; this
                # request is on the default and the floor applies to it.
                deadline_is_default = True

        # Per-identity MINIMUM deadline floor (2026-08-03). Some registered
        # OpenAI-door callers (the `pool` CLI) send NO deadline at all, so the
        # deadline is entirely ours — and the adaptive model's size_stretch
        # clamps at 3.0, producing a flat 540s wall on tier3 that killed real
        # work mid-stream at elapsed_s=539.999. Raise such a caller's deadline
        # to its registered floor. Deliberately NOT applied when the caller
        # supplied its own timeout_s/X-Timeout-S: an explicit caller deadline
        # stays authoritative in both directions, including shorter than the
        # floor. Independent of the smart_default_timeout flag — the flat 180s
        # default is if anything a tighter wall than the smart one.
        if deadline_is_default:
            min_s = principal.min_timeout_s
            if min_s is not None and timeout_s < min_s:
                timeout_s = min_s

        # On-demand endpoints cold-load for minutes — a caller's short timeout
        # (or the 180s default) would expire mid-load and never see a token.
        # Floor the deadline at the endpoint's timeout floor so the request can
        # wait out the load regardless of what the caller sent. (Extend-only:
        # a caller asking for MORE than the floor keeps their value.)
        if self.state.on_demand.manages(endpoint):
            floor_s = self.state.timeout_model.floor_ms(endpoint) / 1000.0
            if timeout_s < floor_s:
                timeout_s = floor_s

        req = QueuedRequest.create(
            agent_id=agent_id,
            endpoint=endpoint,
            priority=declared_priority,
            call_site=body.get("call_site", "unknown"),
            payload_type=body.get("payload_type", "chat_completion"),
            payload=body.get("payload", {}),
            timeout_s=timeout_s,
            session_id=body.get("session_id"),
            turn_id=body.get("turn_id"),
            caller_id=body.get("caller_id"),
            request_id=body.get("request_id"),
            now=now,
            # Carries the caller-vs-proxy deadline distinction resolved above
            # into the streaming path, which is the ONLY consumer: a deadline we
            # chose is a soft budget that token progress may extend, a deadline
            # the caller chose is a hard wall. Nothing else reads it.
            deadline_is_default=deadline_is_default,
            # Workstream C disclosure. The enriched door supplies `requested`
            # (an intent profile, or the endpoint the caller pinned); the OpenAI
            # doors leave it empty and `create` falls back to the endpoint name,
            # which is the truthful answer for a door with no intent vocabulary.
            requested=str(body.get("requested") or ""),
            # 🚨 NARROWING only — see QueuedRequest.allow_degrade. A body that
            # sets either True grants nothing on its own; both gates take the
            # AND with the operator's opt-in.
            allow_degrade=body.get("allow_degrade"),
            allow_spill=body.get("allow_spill"),
        )

        # Payload-shape gate (north-face hardening). A chat payload whose
        # ``messages`` is not a list of objects is unambiguously malformed: the
        # backend would 400 on it, and worse, ``estimate_input_tokens`` (context
        # gate below) and ``the provider's prepare_chat_payload`` both do
        # ``msg.get(...)`` on each element → ``AttributeError`` → a confusing
        # generic 500 instead of a clean rejection. Reject here as a typed 400.
        # No shadow phase: unlike the unknown-endpoint / context heuristics a
        # non-dict message is never legitimate, so there is no false-positive
        # risk to soak. (Guards the Phase-T fuzz repro: messages=["hi","there"].)
        if req.payload_type == "chat_completion":
            messages = req.payload.get("messages")
            if messages is not None and (
                not isinstance(messages, list)
                or any(not isinstance(m, dict) for m in messages)
            ):
                err = ("invalid request: 'messages' must be a list of "
                       "{role, content} objects")
                if wire == WIRE_OPENAI:
                    return _openai_error(
                        err, "invalid_request_error", 400,
                        code="invalid_messages")
                return JSONResponse(
                    {"status": "error", "request_id": req.request_id,
                     "error": err, "code": "invalid_messages"},
                    status_code=400)

        # Grammar authority: validate + safe-normalize any GBNF grammar
        # BEFORE enqueue. Fail loud on an invalid grammar rather than
        # dispatching it (llama-server would silently run unconstrained).
        if req.payload_type == "chat_completion":
            grammar_err = self.correction.process_grammar(req)
            if grammar_err is not None:
                if wire == WIRE_OPENAI:
                    return _openai_error(
                        grammar_err.get("detail", "invalid grammar"),
                        "invalid_request_error", 422, code="invalid_grammar",
                    )
                return JSONResponse(
                    {"status": "error", "request_id": req.request_id,
                     "code": "invalid_grammar", **grammar_err},
                    status_code=422,
                )

        # Context-window pre-admission gate (M1). The proxy KNOWS the live
        # per-slot context (poller-discovered for thinker, config-seeded
        # elsewhere) and estimates input tokens anyway — an oversized prompt
        # should fail fast with an actionable message, not queue, dispatch,
        # and die as a confusing backend 400. Shadow (default): WARN + counter
        # only. Enforce (runtime flag, flipped after the shadow window shows
        # no false positives): 422 whose message embeds the canonical
        # context-overflow marker ("exceeds the available context size") so
        # chunking callers' re-chunk handling engages exactly as it does for
        # the backend's own overflow error. Known undercounts are all in the
        # SAFE direction (false-negative): the struct-policy/thinking
        # max_tokens bumps apply after this check, and image blocks contribute
        # ~0 chars to the estimate.
        if req.payload_type == "chat_completion":
            gate_cfg = self.state.config.endpoints.get(req.endpoint)
            ctx_limit = gate_cfg.context_per_slot if gate_cfg else 0
            req.ctx_per_slot_at_admission = ctx_limit
            # ONE predicate — `cost_model.context_fit`, shared with the failover
            # gate, the spill gate and the recovery tally. 🚨 What is NOT shared
            # is everything below `if not fit.fits`: the shadow counter, the
            # durable evidence row and the enforce flag are this gate's alone,
            # and folding them into the shared predicate would arm a flag nobody
            # flipped on the two gates that refuse unconditionally.
            fit = context_fit(req.payload, req.payload_type, ctx_limit)
            if not fit.fits:
                caller = str(body.get("caller_id") or req.agent_id)
                tally = self.state.context_overflows.setdefault(
                    req.endpoint, {"count": 0, "callers": {}, "max_est_in": 0})
                tally["count"] += 1
                tally["callers"][caller] = tally["callers"].get(caller, 0) + 1
                tally["max_est_in"] = max(tally["max_est_in"], fit.est_in)
                # Durable (audit 2026-07-02): the in-memory tally resets on
                # every ship restart, so the flip-review window never
                # accumulated. Rare event → one async writer op.
                try:
                    self.state.queue_db.record_context_overflow(
                        req.endpoint, caller, fit.est_in)
                except Exception:  # noqa: BLE001 — evidence must not break admission
                    logger.debug("context-overflow persist failed", exc_info=True)
                err = (f"request {fit.overflow_detail(req.endpoint)} — chunk "
                       f"the input or route to a larger-context endpoint")
                if self.state.flags.get("context_gate_enforce"):
                    if wire == WIRE_OPENAI:
                        return _openai_error(
                            err, "invalid_request_error", 422,
                            code="context_overflow")
                    return JSONResponse(
                        {"status": "error", "request_id": req.request_id,
                         "error": err, "code": "context_overflow"},
                        status_code=422)
                logger.warning(
                    "context gate SHADOW: %s (caller=%s call_site=%s) — "
                    "request admitted; flip context_gate_enforce for a "
                    "fast 422", err, caller, req.call_site)

        # Check deterministic cache
        cache_key = self.state.cache.cache_key(req.endpoint, req.payload)
        if cache_key:
            cached = self.state.cache.get(cache_key)
            if cached:
                # Phase 4.1: count cache hits in metrics. They bypass dispatch,
                # so without this they're invisible in /v1/metrics and real
                # traffic is undercounted (the cache's own hit_rate aside).
                self.state.metrics.record(MetricsSample(
                    timestamp=now, endpoint=req.endpoint, agent_id=req.agent_id,
                    priority=req.priority.name, queue_wait_ms=0.0,
                    backend_latency_ms=0.0, status="ok", slot_seconds=0.0))
                # OpenAI consumers get the bare cached completion; internal
                # consumers get the submit envelope (unchanged).
                if wire == WIRE_OPENAI:
                    return JSONResponse(cached)
                return JSONResponse({
                    "status": "ok",
                    "request_id": req.request_id,
                    "queue_wait_ms": 0,
                    "backend_latency_ms": 0,
                    "estimated_cost_ss": 0,
                    "response": cached,
                    "cache_hit": True,
                })

        # On-demand backends (e.g. creative, Gemma-4-31B abliterated): acquire the anvil
        # GPU-slot dispatcher lease so the model is resident before dispatch.
        # Blocks (FIFO) behind any other on-demand service (imagegen/diarize/…)
        # holding the slot, and may cold-load for minutes — the 900s timeout
        # floor covers it. A dispatcher failure surfaces as a DEFERRABLE error
        # ("backpressure") so the caller retries instead of dispatching into a
        # dead backend. After the cache check so a cached reply never needlessly
        # wakes the model; before the circuit breaker (which _endpoint_healthy
        # short-circuits to True for on-demand endpoints).
        if self.state.on_demand.manages(req.endpoint):
            try:
                await self.state.on_demand.ensure_loaded(req.endpoint)
            except OnDemandUnavailable as exc:
                err = (f"on-demand backend {req.endpoint} could not be loaded "
                       f"— backpressure: {exc}")
                logger.warning("on_demand ensure_loaded failed: %s", err)
                if wire == WIRE_OPENAI:
                    return _openai_error(
                        err, "backend_unavailable", 503, code="on_demand_unavailable")
                return JSONResponse(
                    {"status": "error", "request_id": req.request_id,
                     "error": err, "code": "on_demand_unavailable"},
                    status_code=503)

        # Circuit breaker (Phase 1.2): when the backend is marked unhealthy,
        # fast-fail interactive/foreground submits immediately with a DEFERRABLE
        # error instead of queuing them to wait out their full deadline; let
        # background work queue so it defers until the backend recovers. (Cache
        # hits above are served regardless — they don't need the backend.)
        if not self.health.endpoint_healthy(req.endpoint):
            # § 9 tier3 failover. Evaluated for EVERY band, not just
            # interactive/foreground: a background request that can be served
            # now by the smaller model should be, rather than sitting deferred
            # for the whole outage. Nothing below runs while the fleet is
            # healthy — this is the branch that already meant "refuse".
            src_ep = normalize_endpoint(req.endpoint)
            self.state.failover.refresh()
            fplan = self.state.failover.plan(req)
            if fplan.rerouted:
                self.state.failover.apply(req, fplan.target)
            elif req.band != PriorityBand.BACKGROUND:
                # Phase 5F: distinguish an operator drain (planned) from an
                # auto-circuit trip (backend unreachable). Both are DEFERRABLE
                # ("circuit open" / "backpressure" are is_deferrable_llm_error
                # markers) so the caller retries; the wording just aids triage.
                #
                if src_ep in self.state.paused_endpoints:
                    err = f"backend {req.endpoint} paused for maintenance (drain) — backpressure"
                    code = "draining"
                else:
                    err = f"backend {req.endpoint} unavailable (circuit open)"
                    code = "circuit_open"
                # A failover REFUSAL (§ 9.4) APPENDS to that message and travels
                # in its own field. It must not replace either: `code` is the
                # taxonomy callers already classify on, and the message above is
                # what distinguishes a planned drain from an outage for the
                # operator — plus it carries the substrings the fleet's clients
                # sniff for deferrability.
                body_extra: dict = {}
                if fplan.refusal_code:
                    err += fplan.refusal_detail
                    body_extra["degraded_refusal"] = fplan.refusal_code
                    # Counted HERE, at the point the request is actually turned
                    # away — not inside plan(), which is pure. A BACKGROUND
                    # request that fails both gates never reaches this branch:
                    # it falls through and queues to defer until recovery,
                    # exactly as before failover existed. It was not refused,
                    # and must not inflate the counter the operator reads to
                    # decide whether the opt-in set is too small.
                    self.state.failover.record_refusal(req, fplan)
                retry_after = self.health.retry_after_s(src_ep)
                if wire == WIRE_OPENAI:
                    resp = _openai_error(err, "backend_unavailable", 503, code=code)
                else:
                    resp = JSONResponse(
                        {"status": "error", "request_id": req.request_id, "error": err,
                         "code": code, **body_extra},
                        status_code=503,
                    )
                resp.headers["Retry-After"] = str(retry_after)
                return resp

        # Load-shed / backpressure (Phase 2.4): under sustained saturation, shed
        # NON-interactive work with 429 + Retry-After so callers defer instead of
        # all queuing until their deadlines and 504ing together. Interactive is
        # never shed.
        if req.band != PriorityBand.INTERACTIVE:
            snap = self.state.scheduler.endpoint_snapshot(req.endpoint)
            band_key = req.band.name.lower()
            if snap.get("queue_by_band", {}).get(band_key, 0) >= self.state.shed_depth:
                err = f"backpressure: {req.endpoint} {band_key} queue saturated"
                retry_after = self.health.retry_after_s(req.endpoint)
                if wire == WIRE_OPENAI:
                    resp = _openai_error(err, "backpressure", 429, code="backpressure")
                else:
                    resp = JSONResponse(
                        {"status": "error", "request_id": req.request_id, "error": err,
                         "code": "backpressure"},
                        status_code=429)
                resp.headers["Retry-After"] = str(retry_after)
                return resp

        # Forced-reasoning endpoints (e.g. creative/Trinity-Mini, capabilities.reasoning
        # =true) ALWAYS spend max_tokens on an un-disable-able CoT before the answer, so
        # small caller caps truncate mid-reasoning. Reserve answer headroom for BOTH the
        # streaming and sync paths (must precede the branch — apply_thinking is sync-only).
        self.correction.apply_forced_reasoning_budget(req)

        # A backend launched with structured-output whitespace BANNED
        # (disable_any_whitespace — tier3) turns a BARE response_format
        # json_object into the literal `{}`: with no whitespace allowed, `{}` is a
        # legal COMPLETE object and greedy decoding closes immediately. Strip it.
        # Same placement rationale as the line above — this guard has to cover
        # both the streaming and the sync path.
        self.correction.apply_json_object_guard(req)

        # Thinking option: honor a per-request `thinking:true` opt-in (enable
        # native <think> on vLLM + generous budget bump + system fold). No-op
        # otherwise. Sits HERE, above the branch, alongside the two guards above:
        # it used to live inside handle_sync_submit and bail on req.stream, which
        # made the opt-in a silent no-op for streaming callers. The response-side
        # half (finalize_thinking) is still sync-only — apply_thinking skips the
        # thinking_active registration when streaming.
        self.correction.apply_thinking(req)

        # Streaming vs non-streaming
        if req.stream:
            return await self.handle_streaming_submit(req, wire=wire)
        else:
            return await self.handle_sync_submit(req, cache_key, wire=wire)
    async def handle_sync_submit(
        self, req: QueuedRequest, cache_key: str | None, *, wire: str = WIRE_ENRICHED,
    ) -> Response:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self.state.pending_futures[req.request_id] = future

        self.state.scheduler.enqueue(req)
        self.state.queue_db.persist_enqueue(req)
        self.state.dispatch_event.set()

        try:
            result = await asyncio.wait_for(
                future, timeout=req.timeout_s,
            )
        except asyncio.TimeoutError:
            self.state.scheduler.cancel(req.request_id)
            self.state.queue_db.persist_expire(req.request_id)
            self.state.pending_futures.pop(req.request_id, None)
            self.state.thinking_active.pop(req.request_id, None)
            # The caller's deadline fired — work may still be in flight.
            self.record_timeout_event(req, layer="client_wait", elapsed_s=req.timeout_s)
            if wire == WIRE_OPENAI:
                return _openai_error(
                    f"proxy timeout after {req.timeout_s:.0f}s", "proxy_timeout", 504,
                    code="proxy_timeout",
                )
            return JSONResponse(
                {"status": "error", "request_id": req.request_id,
                 "code": "proxy_timeout",
                 "error": f"proxy timeout after {req.timeout_s:.0f}s",
                 "attribution": attribution(self.state, req)},
                status_code=504,
            )
        finally:
            self.state.pending_futures.pop(req.request_id, None)

        # Admission timeout (scheduler callback) resolves the future with a
        # timeout result — already logged there; surface the same 504.
        if result.get("status") == "timeout":
            self.state.thinking_active.pop(req.request_id, None)
            if wire == WIRE_OPENAI:
                return _openai_error(
                    f"proxy timeout after {req.timeout_s:.0f}s", "proxy_timeout", 504,
                    code="proxy_timeout",
                )
            return JSONResponse(
                {"status": "error", "request_id": req.request_id,
                 "code": "proxy_timeout",
                 "error": f"proxy timeout after {req.timeout_s:.0f}s",
                 "attribution": attribution(self.state, req)},
                status_code=504,
            )

        # Uniform non-streaming correction (Step 4a): thinking-finalize →
        # degeneration-correct → shadow-egress-detect, in that load-bearing order
        # (finalizers first so the guard/detector/cache see corrected content;
        # degeneration before the detector + cache so a never-cache-degenerate flag
        # is set first). Byte-identical to the prior inline sequence — see
        # Correction.apply.
        await self.correction.apply(req, result)

        # Cache if deterministic — but NEVER cache an unrecovered degenerate
        # response (don't serve the same garbage for the cache TTL). Both internal
        # markers are POPPED here so they never leak into the returned JSON; the
        # schema-unrecoverable path also flips status to "error" (→ excluded from
        # cache anyway), but we pop it for symmetry + response hygiene.
        degen_unrecovered = result.pop("_degenerate_unrecovered", False)
        schema_unrecovered = result.pop("_schema_unrecoverable", False)
        # …and NEVER cache a truncated (finish_reason=length) body (operator
        # mandate 2026-07-11): a temperature=0 free-text truncation passes as
        # status=ok, and caching it would re-serve the cut-off text for the
        # whole cache TTL — one capped call poisoning every identical call
        # after it. Cheap, total shape probe; a non-chat/odd body reads as
        # not-truncated (cached as before).
        truncated_ok = False
        try:
            _resp_body = result.get("response")
            if isinstance(_resp_body, dict):
                _ch0 = (_resp_body.get("choices") or [{}])[0]
                truncated_ok = (
                    isinstance(_ch0, dict)
                    and _ch0.get("finish_reason") == "length")
        except (AttributeError, IndexError, TypeError):
            truncated_ok = False
        if (cache_key and result.get("status") == "ok"
                and not degen_unrecovered and not schema_unrecovered
                and not truncated_ok):
            self.state.cache.put(cache_key, result.get("response", {}))

        # OpenAI consumers get the bare chat.completion (or an OpenAI-shaped
        # error) with the enrichment in headers; enriched consumers get the
        # Roadstead envelope. Same `result` either way — only the bytes differ.
        if wire == WIRE_OPENAI:
            if result.get("status") == "ok":
                return JSONResponse(result.get("response", {}),
                                    headers=enrichment_headers(req))
            return _openai_error(
                result.get("error", "backend error"), "backend_error", 502,
                code="backend_error",
            )
        return self._enriched_response(req, result)
    def _predicted_ms(self, req: QueuedRequest) -> float | None:
        """What the timeout model would recommend for this call, in ms.

        Computed on the ENRICHED wire only, at response time. It is a percentile
        over the learned distribution — cheap, but not free, and the OpenAI hot
        path has no way to carry the answer and so must not pay for it. Fully
        guarded: a prediction is a nicety, and nothing about it may turn a
        completed request into a 500.
        """
        try:
            mt = req.payload.get("max_tokens") if isinstance(req.payload, dict) else None
            est_out = mt if isinstance(mt, int) and mt > 0 else 0
            advice = self.state.effective_timeout_advice(
                req.endpoint, int(req.priority), req.est_input_tokens, est_out)
            return float(advice.get("recommended_ms") or 0.0) or None
        except Exception:  # noqa: BLE001 — observability must not break a served call
            logger.debug("predicted_ms unavailable", exc_info=True)
            return None

    def _enriched_response(self, req: QueuedRequest, result: dict) -> Response:
        """Serialize one finished non-streaming request onto the enriched wire.

        Four blocks and nothing else: ``attribution`` (who served, and whether
        that is who was asked for), ``timing`` (both sides of the deadline),
        ``usage`` (tokens and the slot-seconds this call actually occupied), and
        the backend's own ``response``.

        🚨 No priority, no band, no queue position. ``docs/api.md`` §1.6: a
        caller cannot observe its own spend demotion in a response, and every
        one of those fields would leak it.
        """
        ok = result.get("status") == "ok"
        in_tok = int(result.get("input_tokens") or 0)
        out_tok = int(result.get("output_tokens") or 0)
        attrib = attribution(self.state, req)
        attrib["cost"] = cost_block(self.state, req.endpoint, in_tok, out_tok)
        body = {
            "status": "ok" if ok else "error",
            "request_id": req.request_id,
            "attribution": attrib,
            "timing": timing_block(
                req,
                queue_wait_ms=result.get("queue_wait_ms") or 0.0,
                backend_latency_ms=result.get("backend_latency_ms") or 0.0,
                predicted_ms=self._predicted_ms(req),
            ),
            "usage": {
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                # The unit of fairness, published. A caller that wants to
                # understand why it is being scheduled the way it is cannot do so
                # from token counts — occupancy time is what DRR charges.
                "slot_seconds": result.get("estimated_cost_ss") or 0.0,
            },
        }
        if ok:
            body["response"] = result.get("response", {})
            return JSONResponse(body, status_code=200)
        # 🚨 `code` and `error` keep the §2.1/§2.2 spellings verbatim: the
        # marker substrings a caller classifies on live in `error`, and they must
        # mean the same thing on both doors.
        body["code"] = result.get("code") or "backend_error"
        body["error"] = result.get("error", "backend error")
        # 🚨 The BACKEND's status, when the failure came from one. `code` alone
        # cannot say whether retrying is worth anything: `backend_error` covers
        # a transient 502 and a permanent 400, and §2.2 asks a client to
        # classify on the code rather than on prose — so the fact it needs has
        # to be a field rather than something to parse out of the message.
        if result.get("backend_status") is not None:
            body["backend_status"] = result["backend_status"]
        return JSONResponse(body, status_code=502)

    async def handle_streaming_submit(
        self, req: QueuedRequest, *, wire: str = WIRE_ENRICHED,
    ) -> Response:
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self.state.pending_streams[req.request_id] = queue

        # Consumer-side backstop, per EVENT (not per stream): how long this
        # generator waits for the next frame before giving up on the producer.
        # It must not be tighter than the producer's own watchdogs, or it would
        # re-impose the very wall clock _execute_stream just stopped enforcing —
        # a proxy-chosen 540s deadline would kill a stream whose TTFT allowance
        # is legitimately 688s, and the fix would be silently half-applied.
        # The producer always puts an error frame on its own abort, so this only
        # ever fires if the producer itself wedged.
        consumer_wait_s = req.timeout_s
        if req.deadline_is_default:
            consumer_wait_s = max(req.timeout_s, self._stream_hard_cap_s(req))

        self.state.scheduler.enqueue(req)
        self.state.queue_db.persist_enqueue(req)
        self.state.dispatch_event.set()

        async def stream_generator():
            # Per-request tool-call stream sanitizer (stateful across this one
            # stream). Default: OpenAI front door only; Step-4a
            # (uniform_correction_enabled) extends it to the enriched
            # /rs/v1/chat streams too — see the gate ~45 lines below.
            # See _ToolCallStreamSanitizer.
            toolcall_sanitizer = _ToolCallStreamSanitizer()
            # Enriched consumers get an opening frame naming what will serve
            # them BEFORE the first token, which is the whole reason this door
            # exists — a streaming caller has no headers left to read by then.
            # OpenAI consumers (goose-cli) get ONLY chat.completion.chunk
            # frames, so no marker is emitted at all: an OpenAI client chokes
            # parsing one.
            if wire != WIRE_OPENAI:
                yield ("data: " + json.dumps({
                    "type": "accepted",
                    "request_id": req.request_id,
                    "attribution": attribution(self.state, req),
                    "timing": timing_block(
                        req, queue_wait_ms=0.0, backend_latency_ms=0.0,
                        predicted_ms=self._predicted_ms(req)),
                }) + "\n\n")

            try:
                while True:
                    event = await asyncio.wait_for(
                        queue.get(), timeout=consumer_wait_s,
                    )
                    if wire == WIRE_OPENAI:
                        etype = event.get("type")
                        if etype == "chunk":
                            # event["data"] is the backend's raw OpenAI
                            # chat.completion.chunk line. Plain content chunks
                            # pass through byte-identical; streaming tool-call
                            # deltas are sanitized so strict clients (the Vercel
                            # AI SDK that opencode uses) don't choke on vLLM's
                            # qwen3_xml phantom/name-less openers. See
                            # _ToolCallStreamSanitizer.
                            yield f"data: {toolcall_sanitizer.feed(event['data'])}\n\n"
                            continue
                        if etype == "done":
                            yield "data: [DONE]\n\n"
                            break
                        if etype == "error":
                            yield (
                                "data: "
                                + json.dumps({"error": {
                                    "message": event.get("error", "stream error"),
                                    "type": "proxy_error",
                                }})
                                + "\n\n"
                            )
                            break
                        # queued / admitted / anything else → not an OpenAI frame.
                        continue
                    # Internal envelope path: re-emit every event. Under uniform
                    # correction (Step 4a), route tool-call CHUNK frames through the
                    # SAME sanitizer the OpenAI door uses, so enriched
                    # /rs/v1/chat consumers get the qwen3_xml phantom/
                    # truncated-arg fix too — not just the OpenAI door.
                    # Default OFF == byte-identical (internal streams emit raw).
                    if (uniform_correction_enabled()
                            and event.get("type") == "chunk" and "data" in event):
                        event = {**event, "data": toolcall_sanitizer.feed(event["data"])}
                    if event.get("type") == "done":
                        # Reshape the producer's flat done frame onto the
                        # enriched wire HERE, at the serializer boundary, rather
                        # than teaching the producer about a wire. The producer
                        # measures; this decides how to say it — which is why
                        # adding this API changed nothing in `execute_streaming`.
                        event = _enriched_done(self.state, req, event)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") in ("done", "error"):
                        break
            except asyncio.TimeoutError:
                if wire == WIRE_OPENAI:
                    yield (
                        "data: "
                        + json.dumps({"error": {
                            "message": f"proxy stream timeout after {consumer_wait_s:.0f}s",
                            "type": "proxy_timeout",
                        }})
                        + "\n\n"
                    )
                else:
                    yield f"data: {json.dumps({'type': 'error', 'error': 'timeout'})}\n\n"
                self.record_timeout_event(req, layer="stream", elapsed_s=req.timeout_s)
            finally:
                # Phase 5B.1: the SSE consumer is gone (client disconnect, our
                # own timeout, or normal completion). Cancel the producer
                # dispatch task if it's still running — otherwise a producer
                # blocked on a full stream_q.put (maxsize=256, consumer no longer
                # draining) wedges forever, holding the scheduler slot until the
                # proxy restarts. That leak cascaded all 4 companion slots into a
                # full endpoint jam (2026-05-31). The CancelledError branch in
                # _execute_dispatch records the completion → frees the slot.
                self.state.pending_streams.pop(req.request_id, None)
                producer = self.state.inflight_tasks.get(req.request_id)
                if producer is not None and not producer.done():
                    producer.cancel()

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            # 🚨 The enrichment headers are on the wire BEFORE the first token,
            # so they carry only what admission already settled — request id,
            # the endpoint chosen, the deadline and who chose it. A later
            # failover or spill cannot be reflected here, which is exactly why
            # the enriched stream repeats attribution on its `done` frame.
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                     **enrichment_headers(req)},
        )
    async def scheduler_loop(self) -> None:
        """Main scheduling loop — runs dispatch on every event or interval.

        The iteration body is guarded: one poisoned tick (a scheduler bug, a
        corrupt request) must not kill dispatching for the WHOLE fleet. On an
        escaped exception we log CRITICAL and back off 1s; CancelledError
        (shutdown) propagates."""
        while True:
            try:
                await asyncio.wait_for(
                    self.state.dispatch_event.wait(),
                    timeout=self.state.config.drr_tick_interval_s * 10,
                )
            except asyncio.TimeoutError:
                pass
            self.state.dispatch_event.clear()

            try:
                now = time.monotonic()
                decisions = self.state.scheduler.tick(now)

                for decision in decisions:
                    rid = decision.request.request_id
                    task = asyncio.create_task(self.execute_dispatch(decision))
                    self.state.inflight_tasks[rid] = task
                    task.add_done_callback(
                        lambda t, _rid=rid: self.state.inflight_tasks.pop(_rid, None)
                    )
            except Exception:  # noqa: BLE001 — keep the fleet dispatching
                logger.critical(
                    "scheduler loop iteration failed — dispatch continues after "
                    "1s backoff", exc_info=True,
                )
                await asyncio.sleep(1.0)
    async def execute_dispatch(self, decision: DispatchDecision) -> None:
        """Execute a dispatch decision: call the backend and resolve the
        caller's future/stream."""
        req = decision.request
        ep_cfg = self.state.config.endpoints.get(req.endpoint)
        if not ep_cfg:
            self.state.resolve_error(req, f"unknown endpoint {req.endpoint}")
            return

        if (
            ep_cfg.slot_affinity
            and req.payload_type == "chat_completion"
            and req.session_id
            and req.band == PriorityBand.INTERACTIVE
        ):
            slot_n = ep_cfg.dispatch_concurrency_cap or ep_cfg.max_slots or 1
            slot_id = (
                int.from_bytes(
                    hashlib.md5(req.session_id.encode()).digest()[:4], "little"
                ) % slot_n
            )
            req.payload = {**req.payload, "id_slot": slot_id}
            logger.debug(
                "slot_affinity: session=%s → id_slot=%d (n=%d) on %s",
                req.session_id, slot_id, slot_n, ep_cfg.role,
            )

        self.state.queue_db.persist_dispatch(req.request_id)

        # Real-time fan-out — a request just started executing. Lets /v1/stream
        # subscribers render the live in-flight board the instant work begins
        # (the `call.completed` event later removes it). Synchronous + no-op when
        # no clients; guarded so a fault never disturbs the dispatch path.
        try:
            self.state.sse.publish("call.dispatched", {
                "request_id": req.request_id,
                "agent": req.agent_id,
                "endpoint": req.endpoint,
                "call_site": req.call_site,
                "priority": req.priority.name,
                "band": req.band.name.lower(),
                "input_tokens": req.est_input_tokens,
                "estimated_remaining_s": round(req.estimated_cost_ss, 2),
                "queue_wait_ms": round(decision.queue_wait_ms, 1),
                "ts": time.time(),
            })
        except Exception as exc:  # noqa: BLE001
            logger.debug("sse call.dispatched publish failed for %s: %s", req.request_id, exc)

        t0 = time.monotonic()
        try:
            if req.stream:
                await self.execute_streaming(req, ep_cfg, decision)
            else:
                await self.execute_sync(req, ep_cfg, decision)
        except asyncio.CancelledError:
            # Phase 5B.2: this dispatch task was cancelled. Two callers cancel
            # it: (a) a streaming consumer disconnected and the SSE generator's
            # finally cancels us (Phase 5B.1 — otherwise a producer wedged on a
            # full stream_q.put would hold the scheduler slot forever — the live
            # companion-jam bug), or (b) the shutdown drain deadline fired.
            # Either way we MUST record the completion so the scheduler frees the
            # slot, then re-raise so cancellation propagates and the task ends.
            # CancelledError is a BaseException, so the `except Exception` below
            # would NOT catch it — without this branch the slot leaks.
            duration = time.monotonic() - t0
            self.state.slot_leak_reclaimed += 1
            logger.info(
                "dispatch %s cancelled (consumer gone / drain) after %.1fs — "
                "reclaiming slot", req.request_id, duration,
            )
            self.state.resolve_error(
                req, "llm proxy stream cancelled (consumer disconnected)")
            self.record_completion(req, decision, duration, 0, 0, "cancelled")
            raise
        except Exception as exc:
            duration = time.monotonic() - t0
            logger.error(
                "dispatch %s failed: %s", req.request_id, exc,
            )
            self.state.resolve_error(req, str(exc))
            self.record_completion(req, decision, duration, 0, 0, "error")
    async def execute_sync(
        self,
        req: QueuedRequest,
        ep_cfg: EndpointConfig,
        decision: DispatchDecision,
    ) -> None:
        attempts = 0
        # Local dispatch payload: the empty-completion rescue swaps in a COPY
        # with min_tokens on the retry — req.payload itself is corpus-persisted
        # and must stay the caller's bytes.
        dispatch_payload = req.payload
        rescue_armed = False
        while True:
            attempts += 1
            # Phase 1.5 slot-leak fix: bound each backend attempt to the
            # caller's ABSOLUTE deadline (timeout_deadline), not a fresh full
            # timeout_s. The client started its clock at enqueue and the backend
            # at dispatch (after queue_wait), so a fresh timeout_s here would let
            # an abandoned call outlive its caller and hold the slot for
            # queue_wait + timeout_s. The remaining-deadline bound frees the slot
            # at the SLA instead.
            remaining = req.timeout_deadline - time.monotonic()
            if remaining <= 0:
                # Phase 5C: the deadline passed in-queue. For BACKGROUND work,
                # surface a DEFERRABLE error ("backpressure") so the caller
                # re-queues it on a later pass instead of dead-lettering work it
                # never got to run — the "stop abandoning work at the deadline"
                # goal. Interactive/foreground still hard-fail (the turn is over).
                is_bg = int(req.priority) >= int(LLMPriority.P3_INGESTION)
                msg = (
                    f"backend {ep_cfg.role} deadline exceeded in queue — backpressure"
                    if is_bg else
                    f"backend {ep_cfg.role} deadline exceeded before dispatch"
                )
                self.state.resolve_error(req, msg)
                self.record_completion(req, decision, 0.0, 0, 0, "timeout")
                self.record_timeout_event(
                    req, layer="backend", elapsed_s=0.0,
                    queue_wait_ms=decision.queue_wait_ms, emit_metrics_and_log=False,
                )
                return

            t0 = time.monotonic()
            try:
                resp = await self.state.backend.call(
                    ep_cfg, dispatch_payload, req.payload_type,
                    req.request_id, timeout_s=max(1.0, remaining),
                )
            except BackendTimeout as exc:
                duration = time.monotonic() - t0
                self.state.resolve_error(req, str(exc))
                self.record_completion(req, decision, duration, 0, 0, "timeout")
                self.record_timeout_event(
                    req, layer="backend", elapsed_s=duration,
                    queue_wait_ms=decision.queue_wait_ms, emit_metrics_and_log=False,
                )
                # Step 4b. best_effort: a caller that never gave the backend a
                # fair chance must not cool it for everyone else (tier1 greeter
                # advisory, 2026-08-20).
                self.health.record_dispatch_failure(
                    req.endpoint, exc,
                    best_effort=self._is_best_effort_timeout(
                        req, _COOLDOWN_BEST_EFFORT_RATIO),
                )
                return
            except (BackendUnavailable, BackendError) as exc:
                duration = time.monotonic() - t0
                # Per-endpoint empty-completion tally (audit 2026-07-12, C-3):
                # count EVERY empty-completion event (each attempt), before the
                # retry decision, so the signal is complete whether or not the
                # rescue path engages. Loop-thread-only increment (single-writer).
                if (
                    req.payload_type == "chat_completion"
                    and "empty completion" in (exc.detail or "")
                ):
                    ep_key = normalize_endpoint(req.endpoint)
                    self.state.empty_completion_by_endpoint[ep_key] = (
                        self.state.empty_completion_by_endpoint.get(ep_key, 0) + 1)
                # Phase 1.3 defer-don't-drop: a transient infra failure (backend
                # unreachable/503, or an empty completion — a backend hiccup, not
                # a content error) RETRIES within the remaining deadline rather
                # than burning the call. Deterministic 4xx/other 5xx surface.
                if (
                    self.correction.is_transient_backend_error(exc)
                    and attempts <= self.state.transient_retry_max
                    and (req.timeout_deadline - time.monotonic()) > _MIN_RETRY_BUDGET_S
                    and self.health.endpoint_healthy(req.endpoint)
                ):
                    # Empty-completion rescue: a position-0-EOS degeneration is
                    # DETERMINISTIC for its prompt — re-dispatching the same
                    # bytes can't recover it. Mask EOS for the first N tokens
                    # on the retry (vLLM min_tokens; llama.cpp ignores it).
                    if (
                        req.payload_type == "chat_completion"
                        and "empty completion" in (exc.detail or "")
                    ):
                        dispatch_payload = {
                            **req.payload,
                            "min_tokens": _EMPTY_RESCUE_MIN_TOKENS,
                        }
                        rescue_armed = True
                        self.state.empty_rescue_attempts += 1
                        logger.warning(
                            "empty completion on %s (attempt %d) — retrying "
                            "with min_tokens=%d (EOS-degeneration rescue)",
                            ep_cfg.role, attempts, _EMPTY_RESCUE_MIN_TOKENS,
                        )
                    else:
                        logger.warning(
                            "transient backend error on %s (attempt %d) — retrying: %s",
                            ep_cfg.role, attempts, exc,
                        )
                    await asyncio.sleep(_RETRY_BACKOFF_S)
                    continue
                # 🚨 Reached only when the retry above declined — i.e. when this
                # proxy has judged the failure DETERMINISTIC. Carrying the
                # backend's status lets the caller reach the same conclusion,
                # instead of being handed `backend_error` and a client-side
                # default that says "retry", which is this proxy advising a
                # retry it just refused to make itself.
                self.state.resolve_error(
                    req, str(exc),
                    backend_status=getattr(exc, "status_code", None))
                self.record_completion(req, decision, duration, 0, 0, "error")
                # Step 4b: count a backend-fault (5xx/503) toward the cooldown; a
                # 4xx caller error is classified out inside record_dispatch_failure.
                self.health.record_dispatch_failure(req.endpoint, exc)
                return

            duration = time.monotonic() - t0

            if rescue_armed:
                # The min_tokens re-dispatch produced a real response where the
                # plain dispatch got 1-token EOS — the rescue worked.
                self.state.empty_rescue_recovered += 1
                logger.info(
                    "empty-completion rescue RECOVERED on %s (call_site=%s, "
                    "output_tokens=%d)", ep_cfg.role, req.call_site,
                    resp.output_tokens,
                )
                rescue_armed = False

            # Phase 1.1 truncation integrity. finish_reason=length means the
            # backend hit max_tokens mid-output. For a STRUCTURED request
            # (grammar / response_format / structured_outputs) the body is almost
            # certainly broken/unparseable JSON — fail loud with a DEFERRABLE
            # error so the caller re-chunks instead of recording garbage, and
            # never cache it (status != ok). Free-form truncation is benign.
            if resp.finish_reason == "length" and self.correction.request_is_structured(req):
                self.state.resolve_error(
                    req,
                    f"backend {ep_cfg.role} truncated structured output "
                    f"(finish_reason=length, output_tokens={resp.output_tokens})",
                )
                self.record_completion(
                    req, decision, duration,
                    resp.input_tokens, resp.output_tokens, "truncated",
                    response_body=resp.body if req.payload_type == "chat_completion" else None,
                    finish_reason=resp.finish_reason,
                    cached_tokens=resp.cached_tokens,
                )
                return

            result = {
                "request_id": req.request_id,
                "queue_wait_ms": round(decision.queue_wait_ms, 1),
                "backend_latency_ms": round(duration * 1000, 1),
                "estimated_cost_ss": round(req.estimated_cost_ss, 3),
                "response": resp.body,
                "status": "ok",
                # The BACKEND's own counts, carried so the response serializer
                # can price the call without re-deriving them from the body.
                # 🚨 The measured numbers, never the cost model's estimate: an
                # invoice built from our own guess would be marking our own
                # homework (`spend.SpendLedger.charge` makes the same point).
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
            }
            # § 9.6 — an explicit degraded marker, so an opted-in caller can
            # still choose to defer its own work rather than accept a smaller
            # model's answer. Added ONLY when degraded: the envelope shape for
            # normal traffic is unchanged.
            #
            # The served model needs no special handling — ``resp.body`` is the
            # backend's own reply and its ``model`` field is the model that
            # ACTUALLY served, because the reroute re-pointed req.endpoint
            # before the backend was chosen. Reading back resp.model is the
            # fleet's own rule for trusting a rename; degraded mode must not
            # break it, and this is why the reroute is one assignment on the
            # routing key rather than a dispatcher-side override.
            if req.degraded_from:
                result["degraded"] = True
                result["degraded_from"] = req.degraded_from

            future = self.state.pending_futures.get(req.request_id)
            if future and not future.done():
                future.set_result(result)

            capture_response = resp.body if req.payload_type == "chat_completion" else None
            self.record_completion(
                req, decision, duration,
                resp.input_tokens, resp.output_tokens, "ok",
                response_body=capture_response, finish_reason=resp.finish_reason,
                cached_tokens=resp.cached_tokens,
            )

            # Shadow backend A/B: fire-and-forget to the shadow if configured
            if ep_cfg.shadow_host and ep_cfg.shadow_port:
                asyncio.create_task(self.execute_shadow(
                    req, ep_cfg, decision, resp,
                ))
            return
    async def execute_shadow(
        self,
        req: QueuedRequest,
        ep_cfg: EndpointConfig,
        decision: DispatchDecision,
        primary_resp: BackendResponse,
    ) -> None:
        """Send the same request to the shadow backend and record the
        comparison result. Never affects the primary caller."""
        shadow_resp = await self.state.backend.call_shadow(
            ep_cfg.shadow_host, ep_cfg.shadow_port,
            req.payload, req.payload_type,
            req.request_id, timeout_s=req.timeout_s,
        )
        if shadow_resp is None:
            logger.debug("shadow dispatch %s failed", req.request_id)
            return

        self.state.queue_db.persist_complete(
            f"shadow-{req.request_id}", req.agent_id, req.endpoint,
            req.call_site, int(req.priority),
            shadow_resp.input_tokens, shadow_resp.output_tokens,
            shadow_resp.duration_s, decision.queue_wait_ms, "ok",
            payload=req.payload, response=shadow_resp.body,
            cached_tokens=shadow_resp.cached_tokens,
        )

    async def _stream_progress_probe(
        self,
        ep_cfg,
        cm: asyncio.Timeout,
        get_last_chunk_at,
        get_ttft_ms,
        gap_s: float,
        hard_limit_s: float,
        t0: float,
        req: QueuedRequest,
    ) -> None:
        """C6 — extend the inter-token deadline while the BACKEND is provably working.

        Runs as a companion task for the life of one stream. The gap watchdog on
        its own sees only its own wire, so it cannot tell a wedged backend from
        one doing legitimate work that emits no token — JIT compilation and
        vLLM's tool-call batching both look exactly like death. This asks the
        backend directly.

        Cadence, not expiry: `asyncio.timeout` firing CANCELS the `async for`, so
        a probe that waited for the deadline would arrive at a stream that is
        already dead. Two samples are taken inside each gap (see
        `_STREAM_GAP_PROBE_FRACTION`) so a verdict exists before it fires.

        The three outcomes, and why each is what it is:
          * counters ADVANCED  → backend alive; push the deadline out, at most
            `_STREAM_GAP_MAX_EXTENSIONS` times, never past the absolute cap.
          * counters FROZEN    → the true-wedge signature (`thinker-silent-hang`).
            Stop arguing and let the deadline fire — a watchdog that no longer
            fires on a real wedge is worse than the bug it fixed.
          * counters ABSENT    → cannot discriminate (llama.cpp, unreachable,
            non-200). Also stop: fall back to today's behaviour rather than
            inventing evidence of life.
        """
        extensions = 0
        prev: dict | None = None
        # The floor only guards against a pathologically small gap spinning the
        # probe; the production gaps (30 s / 60 s) give 9 s and 18 s and never
        # reach it. Keeping it low is what makes this behaviour testable at all.
        check_s = max(0.2, gap_s * _STREAM_GAP_PROBE_FRACTION)
        loop = asyncio.get_running_loop()
        while extensions < _STREAM_GAP_MAX_EXTENSIONS:
            await asyncio.sleep(check_s)
            # Before the first token the TTFT watchdog owns the stream, and it
            # is deliberately generous about prefill; don't second-guess it.
            if get_ttft_ms() is None:
                prev = None
                continue
            if (time.monotonic() - get_last_chunk_at()) < check_s:
                prev = None  # flowing — any baseline we held is stale
                continue
            counters = await self.state.backend.probe_progress_counters(ep_cfg)
            if counters is None:
                return
            if prev is None:
                prev = counters
                continue
            advanced = (counters["prompt"] > prev["prompt"]
                        or counters["generation"] > prev["generation"])
            prev = counters
            if not advanced:
                return
            remaining = hard_limit_s - (time.monotonic() - t0)
            if remaining <= 0:
                return  # the absolute cap owns it from here
            extensions += 1
            try:
                cm.reschedule(loop.time() + min(gap_s, remaining))
            except RuntimeError:
                return  # stream already finished and left the context
            self.state.stream_progress_extensions += 1
            logger.info(
                "LLMPROXY_STREAM_PROGRESS_EXTEND endpoint=%s caller=%s "
                "request_id=%s extension=%d/%d gap_s=%.0f prompt=%d generation=%d",
                req.endpoint, req.agent_id, req.request_id, extensions,
                _STREAM_GAP_MAX_EXTENSIONS, gap_s,
                counters["prompt"], counters["generation"],
            )

    async def execute_streaming(
        self,
        req: QueuedRequest,
        ep_cfg: EndpointConfig,
        decision: DispatchDecision,
    ) -> None:
        stream_q = self.state.pending_streams.get(req.request_id)
        if not stream_q:
            # No consumer for this stream (it died with a previous process, or
            # any future no-consumer path). The dispatch already claimed a
            # scheduler slot — record a completion so it's FREED instead of
            # leaking until the next proxy restart (recovery now drops queued
            # stream rows, so this is the belt-and-braces layer).
            logger.warning(
                "streaming dispatch %s has no consumer — reclaiming slot",
                req.request_id)
            self.record_completion(req, decision, 0.0, 0, 0, "cancelled")
            return

        await stream_q.put({
            "type": "admitted",
            "queue_wait_ms": round(decision.queue_wait_ms, 1),
        })

        # Streaming usage accounting (2026-06-10): without
        # stream_options.include_usage most backends emit NO usage in the
        # stream, so streaming completions recorded 0 tokens (live: 495
        # zero-token orchestrator rows/day) — undercounting every usage/savings
        # rollup and starving the cost model. Inject it into the BACKEND
        # payload on a LOCAL COPY (req.payload is corpus-persisted and must
        # stay the caller's bytes). If the CALLER didn't ask for usage, the
        # usage-only frame (usage present, empty choices) is captured and
        # DROPPED below so strict OpenAI clients see a byte-identical stream.
        # Kill-switch: runtime flag inject_stream_usage.
        payload = req.payload
        so = payload.get("stream_options") if isinstance(payload, dict) else None
        client_wants_usage = bool(isinstance(so, dict) and so.get("include_usage"))
        inject_usage = (
            req.payload_type == "chat_completion"
            and not client_wants_usage
            and self.state.flags.get("inject_stream_usage")
        )
        if inject_usage:
            so = dict(so) if isinstance(so, dict) else {}
            so["include_usage"] = True
            payload = {**payload, "stream_options": so}

        t0 = time.monotonic()
        input_tokens = 0
        output_tokens = 0
        cached_tokens: int | None = None  # Phase 2a — prefix-cache attribution
        last_finish_reason: str | None = None
        # Did the BACKEND terminate its own stream properly (`data: [DONE]`)?
        # This is the discriminator behind the terminal-chunk repair below: it
        # separates "the backend says this response is complete but forgot to
        # label it" from "the stream just stopped", which mean opposite things
        # to a client and must never be collapsed.
        saw_backend_done = False
        chunks_relayed = 0
        coalesced_splits = 0
        ttft_ms: float | None = None  # Phase 4.1 — time to first token
        # Step 4a: accumulate assistant content across chunks for end-of-stream
        # DETECTION (degeneration loop / silent grammar-drop) — gated on the flag
        # so the flag-OFF path adds zero per-chunk work and stays byte-identical.
        uniform_on = uniform_correction_enabled()
        # Operator mandate (2026-07-11): a STRUCTURED stream must terminate with
        # an error frame — never a clean 'done' — when it truncated or its
        # reassembled content is not valid JSON. Computed once per stream (cheap
        # payload inspection); ``expects_json`` additionally requires a JSON-
        # implying constraint so guided_choice / bare-token grammars are never
        # parse-gated. Structured streams pay the per-chunk content append even
        # with uniform correction off — bounded by the request's max_tokens.
        guard_on = structured_validity_guard_enabled()
        stream_structured = (
            guard_on and req.payload_type == "chat_completion"
            and self.correction.request_is_structured(req))
        stream_expects_json = (
            stream_structured and self.correction.request_expects_json(req))
        accumulate = uniform_on or stream_expects_json
        accumulated_content = ""

        # Phase 1.5: bound the stream to the caller's remaining deadline so an
        # abandoned stream can't hold its slot past the SLA.
        stream_timeout = max(1.0, req.timeout_deadline - time.monotonic())
        # Progress-governed streaming deadline (2026-08-03). A stream emitting a
        # token every 200ms for nine minutes is manifestly not hung, and the wall
        # clock killed it anyway: live P3_INGESTION evidence on thinker showed
        # three consecutive kills at elapsed_s=539.999 against
        # applied_timeout_s=540.0 on a 123,466-token prompt.
        #
        # The fix is a semantic, not a bigger number. A deadline the CALLER chose
        # (body timeout_s / X-Timeout-S) is a contract and stays a hard wall,
        # exactly as before. A deadline the PROXY chose (resolve_default_timeout)
        # is a BUDGET — while the stream demonstrably makes progress it may run
        # past it, bounded by an absolute hard cap. Under that semantic a stream
        # dies when, and only when: no first token within the TTFT allowance, OR
        # no token for the inter-token gap, OR the hard cap is reached, OR the
        # client goes away (the SSE generator's finally cancels the producer).
        #
        # ``hard_limit_s`` is the single number every bound below is expressed
        # against. For a caller deadline it IS stream_timeout, so this whole path
        # is byte-identical to the previous behaviour. max() means the cap can
        # only ever EXTEND: a proxy deadline already longer than the cap (a
        # generous adaptive one) is never shortened by it.
        soft_budget = bool(req.deadline_is_default)
        hard_limit_s = stream_timeout
        if soft_budget:
            hard_limit_s = max(stream_timeout, self._stream_hard_cap_s(req))
        # Phase 5C: time-to-first-token watchdog. Start with a SHORT deadline; on
        # the first token, reschedule to the gap deadline. A 0-token hang then
        # aborts in ~TTFT seconds (freeing the slot) instead of burning the whole
        # deadline. TTFT scales with prefill size: a 20k+-token prompt
        # legitimately takes >30s to first token under contention (observed: the
        # 06-30 stream-kill storm at est_in=7360 and 21.7k openai_compat
        # prefills, audit 2026-07-02).
        #
        # The rate used to be a hardcoded 1000 tok/s (+1s per 1k est input),
        # which is simply not what the hardware does — tier3 measures 1,426 tok/s
        # at 213K falling to 729 tok/s at 578K, and est_input_tokens undercounts
        # a tool-heavy payload ~2x on top. That combination is what aborted a
        # legitimate prefill at elapsed_s=145.078 (== 30 + 115077/1000).
        # _STREAM_PREFILL_FLOOR_TOK_S is derived from the SLOW end of the
        # measured range with headroom for both effects. Still capped by
        # hard_limit_s so a caller asking for 10s is never over-waited.
        # The INTER-token gap stays flat — once tokens flow, 30s of silence is a
        # genuine stall regardless of prompt size, and it is that guard (not the
        # wall clock) that makes the generous TTFT allowance safe.
        ttft_deadline_s = min(
            _STREAM_TTFT_DEADLINE_S
            + (req.est_input_tokens or 0) / _STREAM_PREFILL_FLOOR_TOK_S,
            hard_limit_s,
        )
        # C6 (2026-08-23): a request CARRYING tools has a legitimately bursty
        # wire — vLLM's tool-call parser withholds argument deltas until it can
        # emit a complete tool_call, measured at 14.23 s of silence mid-answer on
        # an otherwise idle tier3. Give that shape the wider measured base; a
        # tool-less stream keeps the tighter 30 s. See constants.py for the
        # frames/gap table this number comes from.
        base_gap_s = (_STREAM_GAP_TOOLS_S if payload.get("tools")
                      else _STREAM_INTERTOKEN_GAP_S)
        gap_deadline_s = min(base_gap_s, hard_limit_s)
        loop = asyncio.get_running_loop()
        last_chunk_at = t0
        hit_hard_cap = False
        # C6 progress probe: a companion task that extends the deadline while the
        # BACKEND is demonstrably working, so a silent-but-alive stream is not
        # killed. It has to run alongside the stream rather than on the abort
        # path, because `asyncio.timeout` firing cancels the `async for` — once
        # we are in the except block the stream is already gone.
        progress_probe: asyncio.Task | None = None
        try:
            async with asyncio.timeout(ttft_deadline_s) as _cm:
                progress_probe = asyncio.create_task(self._stream_progress_probe(
                    ep_cfg, _cm, lambda: last_chunk_at, lambda: ttft_ms,
                    gap_deadline_s, hard_limit_s, t0, req,
                ))
                async for event in self.state.backend.stream(
                    ep_cfg, payload, req.payload_type,
                    req.request_id, timeout_s=hard_limit_s,
                ):
                    if event.event_type == "chunk":
                        now_m = time.monotonic()
                        if ttft_ms is None:
                            ttft_ms = (now_m - t0) * 1000.0
                        last_chunk_at = now_m
                        # Per-token no-progress watchdog: each token resets a gap
                        # deadline bounded by the caller's remaining SLA. A
                        # 0-token hang aborts in ~TTFT; a MID-STREAM stall aborts
                        # in ~gap — both free the slot instead of burning 180s.
                        remaining = hard_limit_s - (now_m - t0)
                        if remaining <= 0:
                            # The absolute cap: this stream IS progressing, so
                            # neither watchdog would ever fire. Only this stops
                            # an infinitely-progressing stream from holding a
                            # scarce slot forever.
                            hit_hard_cap = True
                            raise asyncio.TimeoutError
                        _cm.reschedule(loop.time() + min(gap_deadline_s, remaining))
                        usage_only = False
                        if event.parsed:
                            usage = event.parsed.get("usage")
                            choices = event.parsed.get("choices") or []
                            if usage:
                                input_tokens = coerce_token_count(usage.get("prompt_tokens"), input_tokens)
                                output_tokens = coerce_token_count(usage.get("completion_tokens"), output_tokens)
                                ct = extract_cached_tokens(usage)
                                if ct is not None:
                                    cached_tokens = ct
                                # The synthetic usage frame (usage, no choices)
                                # exists because WE injected include_usage —
                                # capture it but never relay it to a client
                                # that didn't ask. Usage riding on a normal
                                # content/finish chunk passes through.
                                usage_only = inject_usage and not choices
                            if choices and isinstance(choices[0], dict):
                                fr = choices[0].get("finish_reason")
                                if fr:
                                    last_finish_reason = fr
                                if accumulate:
                                    delta = choices[0].get("delta")
                                    piece = delta.get("content") if isinstance(delta, dict) else None
                                    if isinstance(piece, str):
                                        accumulated_content += piece
                        if not usage_only:
                            # Normalise a coalesced finish chunk into the
                            # canonical content-then-terminal pair. `None`
                            # (the common case) relays the ORIGINAL bytes
                            # untouched — no reserialisation, no drift.
                            split = (
                                _split_coalesced_finish_chunk(event.parsed)
                                if isinstance(event.parsed, dict) else None
                            )
                            if split is None:
                                chunks_relayed += 1
                                await stream_q.put({
                                    "type": "chunk",
                                    "data": event.data,
                                })
                            else:
                                coalesced_splits += 1
                                for part in split:
                                    chunks_relayed += 1
                                    await stream_q.put({
                                        "type": "chunk",
                                        "data": json.dumps(part),
                                    })
                    elif event.event_type == "done":
                        saw_backend_done = True
                        break
        except (asyncio.TimeoutError, BackendTimeout) as exc:
            # ttft watchdog OR overall stream deadline OR backend timeout. The
            # "backpressure" marker makes it deferrable (is_deferrable_llm_error)
            # so the caller defers instead of dead-lettering. A 0-token hang is
            # the common case — name it so log_scan can see the pattern.
            # Name WHICH bound fired. All four are "the stream ended early", but
            # they mean completely different things to an operator: a TTFT abort
            # is a backend that never spoke, a stall is one that died mid-answer,
            # a hard-cap abort is a healthy stream we cut off for capacity, and a
            # caller-deadline abort is us keeping a promise the caller made. Only
            # the first three are proxy-initiated.
            watchdog = False  # proxy-initiated abort, not a client SLA
            now_m = time.monotonic()
            # Most specific first. ``hit_hard_cap`` is only set when OUR explicit
            # check wins the race; when the rescheduled asyncio.timeout fires at
            # the same instant it raises first and the flag is never set, so the
            # cap is ALSO recognised by elapsed time. Without that second arm a
            # cap abort silently reports itself as a caller deadline — which is
            # exactly what the hard-cap test caught.
            if ttft_ms is None and isinstance(exc, asyncio.TimeoutError):
                abort_reason = "ttft"
                err = ("backend produced no output within "
                       f"{ttft_deadline_s:.0f}s (ttft timeout) — backpressure")
                watchdog = True
            elif (isinstance(exc, asyncio.TimeoutError)
                  and (now_m - last_chunk_at) >= gap_deadline_s - 0.5):
                abort_reason = "stall"
                err = ("backend stalled mid-stream (no token for "
                       f"{gap_deadline_s:.0f}s) — backpressure")
                watchdog = True
            elif hit_hard_cap or (soft_budget and (now_m - t0) >= hard_limit_s):
                abort_reason = "hard_cap"
                err = (f"stream exceeded the absolute {hard_limit_s:.0f}s cap "
                       f"while still progressing — backpressure")
                watchdog = True
                self.state.stream_hard_cap_aborts += 1
            else:
                abort_reason = "caller_deadline"
                err = f"stream deadline exceeded — backpressure ({exc})"
            await stream_q.put({"type": "error", "error": err})
            duration = time.monotonic() - t0
            self._note_stream_extension(req, duration, stream_timeout, soft_budget)
            self.record_completion(req, decision, duration, input_tokens, output_tokens, "timeout")
            self.record_timeout_event(
                req, layer="stream", elapsed_s=duration,
                queue_wait_ms=decision.queue_wait_ms, emit_metrics_and_log=False,
                proxy_initiated=watchdog, abort_reason=abort_reason,
            )
            # Step 4b (BackendTimeout only) — same best-effort exclusion as the
            # non-stream path; a sub-floor stream deadline is not stall evidence.
            self.health.record_dispatch_failure(
                req.endpoint, exc,
                best_effort=self._is_best_effort_timeout(
                    req, _COOLDOWN_BEST_EFFORT_RATIO),
            )
            return
        except Exception as exc:
            await stream_q.put({"type": "error", "error": str(exc)})
            duration = time.monotonic() - t0
            self.record_completion(req, decision, duration, input_tokens, output_tokens, "error")
            self.health.record_dispatch_failure(req.endpoint, exc)  # Step 4b
            return
        finally:
            # Every exit path, including the two `return`s above: the probe holds
            # a reference to the timeout context, and rescheduling one that has
            # already exited raises. Cancel is idempotent on a finished task.
            if progress_probe is not None:
                progress_probe.cancel()

        duration = time.monotonic() - t0
        # A stream that COMPLETED past its soft budget is the whole point of the
        # progress-governed deadline — count it, or the feature is invisible.
        self._note_stream_extension(req, duration, stream_timeout, soft_budget)

        # Operator-mandated structured end-of-stream guard (2026-07-11): the
        # chunks already streamed (can't un-send), but every structured-stream
        # consumer reassembles + json-parses at 'done' — so terminating with the
        # ESTABLISHED error frame (instead of a clean 'done') is what converts
        # silent garbage into an explicit, retryable failure, mirroring the sync
        # path's truncation-integrity 502. Truncation checks any structured
        # constraint; the json.loads validity check only JSON-implying ones
        # (guided_choice / bare-token grammars are exempt) and is bounded by the
        # request's max_tokens. Kill-switch ROADSTEAD_PROXY_STRUCTURED_VALIDITY
        # (guard_on) restores the legacy clean-'done' behavior.
        stream_guard_err: str | None = None
        stream_guard_status = "ok"
        if stream_structured and last_finish_reason == "length":
            # Same message shape as the sync truncation error so existing client
            # deferral classification ("truncated structured output") engages.
            stream_guard_err = (
                f"backend {ep_cfg.role} truncated structured output "
                f"(finish_reason=length, output_tokens={output_tokens})")
            stream_guard_status = "truncated"
        elif stream_expects_json and accumulated_content.strip():
            try:
                json.loads(accumulated_content)
            except ValueError:
                self.correction.record_structured_parse_failure(
                    req, stream=True, output_tokens=output_tokens)
                stream_guard_err = (
                    f"backend {ep_cfg.role} returned invalid JSON for a "
                    f"structured request (stream reassembly does not parse; "
                    f"output_tokens={output_tokens})")
                stream_guard_status = "error"

        # --- terminal-chunk repair + per-stream observability (2026-08-24) ---
        #
        # THE DEFECT. In an OpenAI SSE stream `finish_reason` rides ALONE on a
        # final chunk whose `delta` is `{}` — it carries no content, so its
        # loss is undetectable by content alone. When a backend ends a stream
        # without ever emitting that chunk, we relayed the content and then
        # `[DONE]`, and the client saw text with no finish_reason.
        #
        # Measured cost, on Beacon (CTnnn), 2026-08-24: 47 turns. Beacon's
        # `_text_only_dropped_no_finish` guard (agent/chat_completion_helpers.py)
        # treats finish_reason-less text as a mid-stream drop, stamps the turn
        # `length`, and injects "[System: The previous response was cut off by a
        # network error mid-stream. Continue exactly where you left off.]" — so
        # it spends a SECOND full model call continuing an answer that was
        # already complete. It is not a network error and not a token budget:
        # Beacon has separate prompts for both of those causes and used them 0
        # and 0 times, while the answers themselves end in complete sentences
        # far below any cap.
        #
        # THE REPAIR, and why it is not a fabrication. We synthesize the missing
        # terminal chunk ONLY when the backend sent its own `[DONE]` — i.e. the
        # backend asserted the response is complete and merely failed to label
        # it. That is normalising a spec violation, not inventing an outcome.
        # When the stream ended WITHOUT `[DONE]` the truncation is real, we
        # synthesize NOTHING, and the client's own drop handling is correct —
        # collapsing those two cases would trade a visible bug for a silent one.
        if (last_finish_reason is None and saw_backend_done
                and chunks_relayed > 0 and not stream_guard_err):
            last_finish_reason = "stop"
            await stream_q.put({
                "type": "chunk",
                "data": json.dumps({
                    "id": f"chatcmpl-{req.request_id}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": ep_cfg.effective_model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                        "proxy_synthesized_finish": True,
                    }],
                }),
            })
            logger.warning(
                "LLMPROXY_STREAM_FINISH_REPAIRED endpoint=%s caller=%s "
                "request_id=%s chunks=%d out_tokens=%d — backend sent [DONE] "
                "with no finish_reason chunk; synthesized finish_reason=stop. "
                "Without this the client sees text with no finish_reason and "
                "may treat a COMPLETE answer as a mid-stream drop.",
                req.endpoint, req.agent_id, req.request_id,
                chunks_relayed, output_tokens,
            )
        elif last_finish_reason is None and chunks_relayed > 0:
            # Genuinely unterminated: no [DONE], no finish_reason. Say so —
            # this is the case where the client's drop handling is RIGHT.
            logger.warning(
                "LLMPROXY_STREAM_UNTERMINATED endpoint=%s caller=%s "
                "request_id=%s chunks=%d out_tokens=%d — backend stream ended "
                "with NEITHER [DONE] nor a finish_reason; relaying as-is (a "
                "real truncation, not repaired).",
                req.endpoint, req.agent_id, req.request_id,
                chunks_relayed, output_tokens,
            )

        # One line per STREAMING request. Before this there were 5 log lines
        # covering 653 beacon requests, which is why the defect above could not
        # be attributed to a layer for as long as it existed. Cheap and
        # unconditional on purpose: a stream that only logs when something
        # already went wrong cannot tell you what "normal" looked like.
        logger.info(
            "LLMPROXY_STREAM_DONE endpoint=%s caller=%s request_id=%s "
            "chunks=%d ttft_ms=%.0f duration_ms=%.0f finish_reason=%s "
            "backend_done=%s splits=%d in_tokens=%d out_tokens=%d",
            req.endpoint, req.agent_id, req.request_id, chunks_relayed,
            ttft_ms or 0.0, duration * 1000.0,
            last_finish_reason or "ABSENT", saw_backend_done, coalesced_splits,
            input_tokens, output_tokens,
        )

        if stream_guard_err is not None:
            await stream_q.put({"type": "error", "error": stream_guard_err})
        else:
            done_frame = {
                "type": "done",
                "queue_wait_ms": round(decision.queue_wait_ms, 1),
                "backend_latency_ms": round(duration * 1000, 1),
                "ttft_ms": round(ttft_ms or 0.0, 1),  # Phase 4.1
                "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
            }
            # § 9.6 — same degraded marker as the sync envelope, so a streaming
            # caller is told too. Added only when degraded.
            if req.degraded_from:
                done_frame["degraded"] = True
                done_frame["degraded_from"] = req.degraded_from
            await stream_q.put(done_frame)
        # Phase 1.1: record truncation of a structured stream so the storm is
        # visible in metrics (kept independent of the guard kill-switch — with
        # the guard off the caller still got the legacy 'done', but the
        # completion row + LLMPROXY_TRUNCATION log stay loud).
        status = stream_guard_status
        if (status == "ok" and last_finish_reason == "length"
                and self.correction.request_is_structured(req)):
            status = "truncated"
        self.record_completion(
            req, decision, duration, input_tokens, output_tokens, status,
            finish_reason=last_finish_reason, cached_tokens=cached_tokens,
        )
        # Step 4a: uniform streaming detection over the reassembled content
        # (degeneration loop / silent grammar-drop). Detect-only + fail-open +
        # self-gated on the flag; no-op when uniform correction is off.
        if uniform_on:
            self.correction.finalize_stream(req, accumulated_content, last_finish_reason)
    def _stream_hard_cap_s(self, req: QueuedRequest) -> float:
        """Absolute ceiling (seconds) on a PROXY-CHOSEN streaming deadline that
        token progress keeps extending.

        Resolution order, mirroring the timeout floors/ceilings: a per-endpoint-
        class ``stream_hard_cap_s`` from models.yaml wins; otherwise the band
        default. Background (P3/P4) is sized to clear the measured worst case
        (a 578K-token tier3 request at ~794s end-to-end) with room; interactive
        (P0-P2) is deliberately far tighter, because a user-facing turn still
        streaming after 15 minutes has already failed regardless of throughput.

        Never consulted for an explicit caller deadline. Fully guarded — a
        lookup fault must not abort a live stream, so it degrades to the
        background default rather than raising into the streaming path."""
        try:
            cap = self.state.stream_hard_caps.get(normalize_endpoint(req.endpoint))
            if cap and cap > 0:
                return float(cap)
            if req.band == PriorityBand.INTERACTIVE:
                return _STREAM_HARD_CAP_INTERACTIVE_S
            return _STREAM_HARD_CAP_BACKGROUND_S
        except Exception:  # noqa: BLE001 — never break a live stream on a lookup
            return _STREAM_HARD_CAP_BACKGROUND_S

    def _note_stream_extension(
        self, req: QueuedRequest, duration: float, budget_s: float,
        soft_budget: bool,
    ) -> None:
        """Count a stream that outlived the proxy-chosen deadline it was given.

        Extending a deadline trades a capacity guarantee for a completion, and a
        trade nobody can measure is a trade nobody can revisit — so record BOTH
        the count and the seconds granted, and log each one. Only fires on the
        soft-budget path; a caller deadline can't be extended, so there is
        nothing to count."""
        if not soft_budget:
            return
        over = duration - budget_s
        if over <= 0:
            return
        try:
            self.state.stream_deadline_extended += 1
            self.state.stream_extension_s_total += over
            logger.info(
                "stream deadline EXTENDED by progress: endpoint=%s tier=%s "
                "caller=%s budget=%.1fs ran=%.1fs (+%.1fs)",
                req.endpoint, req.priority.name,
                req.caller_id or f"{req.agent_id}/{req.call_site}",
                budget_s, duration, over,
            )
        except Exception:  # noqa: BLE001 — accounting must never break a stream
            pass

    def on_admission_timeout(self, req: QueuedRequest) -> None:
        """Scheduler callback: a request expired while still queued. Log
        it and release the caller promptly with a timeout result (instead
        of letting it wait out its own — identical — deadline)."""
        elapsed = time.monotonic() - req.enqueued_at
        self.record_timeout_event(req, layer="admission", elapsed_s=elapsed)
        future = self.state.pending_futures.get(req.request_id)
        if future and not future.done():
            future.set_result({
                "request_id": req.request_id,
                "status": "timeout",
                "error": "timeout",
            })
        stream_q = self.state.pending_streams.get(req.request_id)
        if stream_q:
            try:
                stream_q.put_nowait({"type": "error", "error": "timeout"})
            except asyncio.QueueFull:
                pass
        self.state.queue_db.persist_expire(req.request_id)
    def record_completion(
        self,
        req: QueuedRequest,
        decision: DispatchDecision,
        duration_s: float,
        input_tokens: int,
        output_tokens: int,
        status: str,
        response_body: dict | None = None,
        finish_reason: str | None = None,
        cached_tokens: int | None = None,
    ) -> None:
        now = time.monotonic()

        # Operator-mandated truncation visibility (2026-07-11): EVERY completion
        # — both response modes, every caller — passes through here with its
        # finish_reason, making this the single choke point where an output-cap
        # hit can never pass silently. finish_reason == "length" is what BOTH
        # llama-server and vLLM emit on the OpenAI-compat surface (sync body and
        # the final stream chunk) when max_tokens cut the output; the tool-call
        # stream sanitizer also relabels a truncated tool stream to "length".
        # Structured truncation already fails the request upstream; free-text
        # truncation still serves (may be a legitimate cap) — this is the loud
        # ERROR + per-(model, caller) tally either way. Synchronous +
        # allocation-light (one small dict row per (model, caller)); fail-open
        # so observability can never break completion accounting.
        if finish_reason == "length" and req.payload_type == "chat_completion":
            try:
                self.correction.record_truncation_event(
                    req, structured=self.correction.request_is_structured(req),
                    stream=req.stream, output_tokens=output_tokens, status=status)
            except Exception:  # noqa: BLE001 — observability must not break accounting
                logger.debug("truncation tally failed", exc_info=True)

        # On-demand: a request to this endpoint reached a terminal outcome
        # (ok/error/timeout/cancel) — release its in-flight hold so the idle
        # watchdog can eventually drop the dispatcher lease. No-op for always-on
        # endpoints. This is also the natural attach point for a future explicit
        # "unload when done" agent signal (release immediately on that request).
        self.state.on_demand.request_done(req.endpoint)

        # Report to scheduler
        self.state.scheduler.complete(
            CompletionRecord(
                request_id=req.request_id,
                duration_s=duration_s,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=(status == "ok"),
                occupancy_during=decision.occupancy_at_dispatch,
            ),
            now,
        )

        # Meter the caller (Workstream D). Charged from the tokens the BACKEND
        # reported, never from the estimate the cost model made — a scheduler
        # that billed its own guess would be marking its own homework, and the
        # estimate exists to reserve a slot, not to price a call.
        #
        # 🚨 Keyed on ``req.endpoint``, which is the class that ACTUALLY served:
        # a spilled request has already been re-pointed at the remote endpoint,
        # so it is priced at the remote endpoint's rate. Pricing it at the
        # endpoint it was submitted for would be the one arrangement guaranteed
        # to under-report exactly the calls that cost real money.
        #
        # Fail-open like the truncation tally above: accounting must never break
        # completion, because a request that completed and was not billed is a
        # far better outcome than a caller who never gets their answer.
        try:
            self.state.spend.charge(
                req.agent_id, req.endpoint, input_tokens, output_tokens)
        except Exception:  # noqa: BLE001 — metering must not break accounting
            logger.debug("spend metering failed", exc_info=True)

        # Persist completion (with payload + response for corpus).
        # Skip corpus capture for embedding/rerank — large vectors bloat
        # the DB and aren't useful for replay testing.
        capture = req.payload_type == "chat_completion"
        kind = _PAYLOAD_KIND.get(req.payload_type, "llm")
        self.state.queue_db.persist_complete(
            req.request_id, req.agent_id, req.endpoint,
            req.call_site, int(req.priority),
            input_tokens, output_tokens, duration_s,
            decision.queue_wait_ms, status,
            payload=req.payload if capture else None,
            response=response_body,
            session_id=req.session_id,
            turn_id=req.turn_id,
            caller_id=req.caller_id,
            finish_reason=finish_reason,
            kind=kind,
            cached_tokens=cached_tokens,
        )

        # Log
        self.state.request_logger.log(RequestLogRecord(
            ts=datetime.now(timezone.utc).isoformat(),
            request_id=req.request_id,
            agent_id=req.agent_id,
            endpoint=req.endpoint,
            call_site=req.call_site,
            priority=req.priority.name,
            band=req.band.name.lower(),
            payload_type=req.payload_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_ss=req.estimated_cost_ss,
            actual_cost_ss=duration_s,
            queue_wait_ms=decision.queue_wait_ms,
            backend_latency_ms=duration_s * 1000,
            total_latency_ms=(now - req.enqueued_at) * 1000,
            occupancy_at_dispatch=decision.occupancy_at_dispatch,
            status=status,
            estimated_input_tokens=req.est_input_tokens or estimate_input_tokens(req.payload),
            max_output_tokens=req.payload.get("max_tokens", 0),
            session_id=req.session_id,
            turn_id=req.turn_id,
            caller_id=req.caller_id,
        ))

        # A backend/stream timeout on a best-effort sub-floor caller (applied
        # deadline far under the recommended time) is not reliable stall evidence
        # — tag it so endpoint_stalled excludes it (2026-07-05 gemma bursts).
        # Only recomputes the advice on the rare timeout completion; fully guarded.
        _best_effort = (
            self._is_best_effort_timeout(req) if status == "timeout" else False)

        # Metrics
        self.state.metrics.record(MetricsSample(
            timestamp=now,
            endpoint=req.endpoint,
            agent_id=req.agent_id,
            priority=req.priority.name,
            queue_wait_ms=decision.queue_wait_ms,
            backend_latency_ms=duration_s * 1000,
            status=status,
            slot_seconds=duration_s,
            call_site=req.call_site,
            best_effort=_best_effort,
        ))

        # Timeout-advice model + shadow log (observational only — guarded
        # so a fault here never disturbs the caller or the scheduler).
        try:
            self.record_timeout_shadow(req, now, duration_s, output_tokens, status)
        except Exception as exc:  # noqa: BLE001
            logger.warning("timeout shadow record failed for %s: %s", req.request_id, exc)

        # Real-time fan-out — emit the completed call to /v1/stream subscribers.
        # Synchronous + no-op when no clients; guarded so a fault never disturbs
        # the caller or the scheduler.
        try:
            self.state.sse.publish("call.completed", {
                "request_id": req.request_id,
                "agent": req.agent_id,
                "endpoint": req.endpoint,
                "call_site": req.call_site,
                "kind": kind,
                "priority": req.priority.name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "duration_s": round(duration_s, 3),
                "queue_wait_ms": round(decision.queue_wait_ms, 1),
                "status": status,
                "ts": time.time(),
            })
        except Exception as exc:  # noqa: BLE001
            logger.debug("sse call.completed publish failed for %s: %s", req.request_id, exc)

        # Trigger scheduler (a slot freed up)
        self.state.dispatch_event.set()

    def resolve_default_timeout(self, endpoint: str, body: dict) -> float:
        """Deadline for a caller that supplied no ``timeout_s`` (the OpenAI door
        or ``/rs/v1/chat``). Caller-supplied deadlines never reach here.

        Flag OFF (``smart_default_timeout`` false — the default): the flat
        ``_DEFAULT_TIMEOUT_S`` (180s), byte-identical to the historical default.
        Flag ON: the timeout model's class-floored, capped recommendation for
        this ``(endpoint, tier, size)`` — so a caller that gives no deadline gets
        the same data-driven bound framework callers already get via
        ``apply_extend_only``, instead of a blanket 180s that is too tight for a
        cold on-demand load and absurdly loose for embed/rerank.

        A shadow tally (flat vs smart) is recorded EITHER WAY, so the flip
        decision has a direct artifact on ``/v1/status.smart_default_shadow``.
        Fully guarded — advice/estimation faults never 500 a request; on any
        error the flat default is used and applied."""
        smart_s = _DEFAULT_TIMEOUT_S
        try:
            priority = int(LLMPriority.coerce(
                body.get("priority"), default=LLMPriority.P1_TURN_SUPPORT))
            payload = body.get("payload") or {}
            if isinstance(payload, dict):
                est_in = estimate_input_tokens(payload)
                mt = payload.get("max_tokens")
                est_out = mt if isinstance(mt, int) and mt > 0 else 0
            else:
                est_in = est_out = 0
            rec = self.state.effective_timeout_advice(
                endpoint, priority, est_in, est_out)["recommended_timeout_s"]
            if rec and rec > 0:
                smart_s = min(float(rec), _SMART_DEFAULT_CAP_S)
        except Exception:  # noqa: BLE001 — default resolution must never 500 a request
            smart_s = _DEFAULT_TIMEOUT_S

        # Shadow tally (both modes) — the reviewable artifact for the flip.
        try:
            tally = self.state.smart_default_shadow.setdefault(
                normalize_endpoint(endpoint),
                {"count": 0, "flat_s": _DEFAULT_TIMEOUT_S,
                 "smart_s_min": None, "smart_s_max": None, "smart_s_sum": 0.0})
            tally["count"] += 1
            tally["smart_s_sum"] += smart_s
            tally["smart_s_min"] = (
                smart_s if tally["smart_s_min"] is None
                else min(tally["smart_s_min"], smart_s))
            tally["smart_s_max"] = (
                smart_s if tally["smart_s_max"] is None
                else max(tally["smart_s_max"], smart_s))
        except Exception:  # noqa: BLE001 — a tally fault must never disturb the caller
            pass

        return smart_s if self.state.flags.get("smart_default_timeout") else _DEFAULT_TIMEOUT_S

    def record_timeout_shadow(
        self,
        req: QueuedRequest,
        now: float,
        duration_s: float,
        output_tokens: int,
        status: str,
    ) -> None:
        """Feed the timeout model and log the counterfactual: what
        ``recommended`` would have been for this call vs. the actual
        end-to-end latency and the timeout actually applied."""
        end_to_end_ms = (now - req.enqueued_at) * 1000.0
        est_in = req.est_input_tokens or estimate_input_tokens(req.payload)
        est_out = int(req.payload.get("max_tokens", 0) or 0)
        priority = int(req.priority)

        # Advice reflects history BEFORE this sample is folded in.
        advice = self.state.timeout_model.advise(req.endpoint, priority, est_in, est_out)

        self.state.timeout_model.record(
            endpoint=req.endpoint,
            priority=priority,
            input_tokens=est_in,
            output_tokens=output_tokens,
            end_to_end_ms=end_to_end_ms,
            status=status,
            now=now,
        )

        # Only successful calls give a representative latency to compare.
        if status != "ok":
            return
        recommended_ms = advice["recommended_ms"]
        self.state.queue_db.persist_timeout_shadow(
            request_id=req.request_id,
            endpoint=normalize_endpoint(req.endpoint),
            priority=priority,
            est_in=est_in,
            est_out=est_out,
            actual_out=output_tokens,
            actual_total_ms=round(end_to_end_ms, 1),
            applied_timeout_s=req.timeout_s,
            recommended_ms=recommended_ms,
            p95_ms=advice["p95_ms"],
            median_ms=advice["median_ms"],
            min_ms=advice["min_ms"],
            source=advice["source"],
            would_timeout=(end_to_end_ms > recommended_ms),
        )
    def _is_best_effort_timeout(
        self, req: QueuedRequest, ratio: float = _STALL_BEST_EFFORT_RATIO,
    ) -> bool:
        """True when THIS request's applied deadline was a sub-floor give-up.

        One definition, three consumers: the ``best_effort`` metric tag, the
        ``endpoint_stalled`` heuristic, and (since 2026-08-20) the cooldown
        window in ``health.record_dispatch_failure``. Keeping the computation in
        one method is the point — the cooldown bug existed precisely because the
        predicate lived inline next to ONE consumer and the other never got it.

        Recomputes the size-aware advice, which is why it is called only on the
        rare timeout path. Fully guarded: any fault answers False (counts as a
        genuine backend fault), so a broken advice model can never silently
        disarm the cooldown."""
        try:
            rec = self.state.timeout_model.advise(
                req.endpoint, int(req.priority),
                req.est_input_tokens or estimate_input_tokens(req.payload),
                int(req.payload.get("max_tokens", 0) or 0),
            )["recommended_ms"]
            return _timeout_below_recommended(req.timeout_s, rec, ratio)
        except Exception:  # noqa: BLE001 — never let telemetry break dispatch
            return False

    def record_timeout_event(
        self,
        req: QueuedRequest,
        *,
        layer: str,
        elapsed_s: float,
        queue_wait_ms: float | None = None,
        emit_metrics_and_log: bool = True,
        proxy_initiated: bool = False,
        abort_reason: str | None = None,
    ) -> None:
        """Record a call that hit its timeout instead of finishing.

        Writes a queryable ``proxy_timeouts`` row with the load context at
        the moment it gave up, emits a WARNING (so log_scan/health-verifier see it),
        and — for the layers that don't otherwise flow through
        ``_record_completion`` (admission/client_wait) — a metrics sample
        and request-log line so the timeout counter and JSONL trail are
        complete. Fully guarded: a fault here never disturbs the caller.

        ``layer``: admission | client_wait | backend | stream.

        ``abort_reason`` (stream layer): which bound actually fired —
        ``ttft`` (no first token) | ``stall`` (no token for the inter-token gap)
        | ``hard_cap`` (a still-progressing stream cut off at the absolute cap)
        | ``caller_deadline`` (the caller's own explicit wall). ``proxy_initiated``
        alone can't distinguish these, and they call for opposite responses — a
        stall means fix the backend, a hard_cap means the cap is too tight or a
        caller is abusive. Surfaced per-reason by ``/v1/timeouts``.
        """
        rid = req.request_id
        if rid in self.state.timed_out_ids:
            return  # already counted this request's timeout
        self.state.timed_out_ids.add(rid)
        if len(self.state.timed_out_ids) > 8192:
            self.state.timed_out_ids.clear()  # bounded; rare duplicate after reset is harmless

        try:
            now = time.monotonic()
            snap = self.state.scheduler.endpoint_snapshot(req.endpoint)
            est_in = req.est_input_tokens or estimate_input_tokens(req.payload)
            est_out = int(req.payload.get("max_tokens", 0) or 0)
            priority = int(req.priority)
            ep_cfg = self.state.config.endpoints.get(normalize_endpoint(req.endpoint))
            # Prefer the admission-time snapshot: the poller mutates
            # context_per_slot, so the live value can differ from the
            # denominator the gate evaluated (audit 2026-07-02).
            context_window = req.ctx_per_slot_at_admission or (
                ep_cfg.context_per_slot if ep_cfg else 0)
            context_used_pct = (
                round(est_in / context_window * 100.0, 1) if context_window else None
            )
            try:
                recommended_ms = self.state.timeout_model.advise(
                    req.endpoint, priority, est_in, est_out,
                )["recommended_ms"]
            except Exception:  # noqa: BLE001
                recommended_ms = 0.0
            # `premature` means the CLIENT gave up below the recommended
            # deadline. A proxy-initiated watchdog abort (stream TTFT/stall)
            # is the proxy giving up — never tag it premature, or the rollup
            # blames callers for the proxy's own kills (audit 2026-07-02).
            under = (
                bool(recommended_ms and elapsed_s * 1000.0 <= recommended_ms)
                and not proxy_initiated
            )

            logger.warning(
                "LLM TIMEOUT layer=%s endpoint=%s tier=%s caller=%s "
                "elapsed=%.1fs applied=%.1fs in_flight=%d queued=%d est_in=%d "
                "est_out=%d ctx_used=%s%% recommended=%.0fms premature=%s "
                "proxy_watchdog=%s abort_reason=%s",
                layer, req.endpoint, req.priority.name,
                req.caller_id or f"{req.agent_id}/{req.call_site}",
                elapsed_s, req.timeout_s, snap["in_flight"], snap["queued"],
                est_in, est_out, context_used_pct, recommended_ms, under,
                proxy_initiated, abort_reason or "-",
            )

            if emit_metrics_and_log:
                # Make the /v1/metrics "timeouts" counter real for the
                # paths that never reach _record_completion.
                self.state.metrics.record(MetricsSample(
                    timestamp=now,
                    endpoint=req.endpoint,
                    agent_id=req.agent_id,
                    priority=req.priority.name,
                    queue_wait_ms=(queue_wait_ms or 0.0),
                    backend_latency_ms=0.0,
                    status="timeout",
                    slot_seconds=0.0,
                    call_site=req.call_site,
                    # `under` = fired below the proxy's recommended deadline; a
                    # client-side give-up, excluded from the endpoint_stalled
                    # backend-stall heuristic (best-effort sub-floor callers).
                    premature=under,
                    # best_effort also catches proxy-initiated aborts (fast-fail
                    # / TTFT watchdog) on sub-floor callers, which `under` forces
                    # False for — the gemma stall-burst amplifier (2026-07-05).
                    best_effort=_timeout_below_recommended(
                        req.timeout_s, recommended_ms),
                ))
                self.state.request_logger.log(RequestLogRecord(
                    ts=datetime.now(timezone.utc).isoformat(),
                    request_id=rid,
                    agent_id=req.agent_id,
                    endpoint=req.endpoint,
                    call_site=req.call_site,
                    priority=req.priority.name,
                    band=req.band.name.lower(),
                    payload_type=req.payload_type,
                    input_tokens=0,
                    output_tokens=0,
                    estimated_cost_ss=req.estimated_cost_ss,
                    actual_cost_ss=0.0,
                    queue_wait_ms=(queue_wait_ms or 0.0),
                    backend_latency_ms=0.0,
                    total_latency_ms=elapsed_s * 1000.0,
                    occupancy_at_dispatch=snap["in_flight"],
                    status="timeout",
                    estimated_input_tokens=est_in,
                    max_output_tokens=est_out,
                    session_id=req.session_id,
                    turn_id=req.turn_id,
                    caller_id=req.caller_id,
                ))

            self.state.queue_db.persist_timeout_event(
                request_id=rid,
                endpoint=normalize_endpoint(req.endpoint),
                priority=priority,
                agent_id=req.agent_id,
                call_site=req.call_site,
                layer=layer,
                elapsed_s=round(elapsed_s, 3),
                applied_timeout_s=req.timeout_s,
                queue_wait_ms=(round(queue_wait_ms, 1) if queue_wait_ms is not None else None),
                in_flight=snap["in_flight"],
                queued=snap["queued"],
                max_slots=snap["max_slots"],
                est_in=est_in,
                est_out=est_out,
                recommended_ms=recommended_ms,
                under_recommended=under,
                session_id=req.session_id,
                turn_id=req.turn_id,
                caller_id=req.caller_id,
                context_window=context_window,
                context_used_pct=context_used_pct,
                abort_reason=abort_reason,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("timeout event record failed for %s: %s", rid, exc)
