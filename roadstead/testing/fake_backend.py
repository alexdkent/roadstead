"""Programmable fake inference backend — a shipped capability, not test scaffolding.

Promoted out of ``tests/fake_backend.py`` on 2026-08-31 (see
``roadstead/testing/__init__.py`` for why). It is an ASGI (Starlette) app that
faithfully mimics **every south-face surface the proxy calls** — for both wire
shapes the proxy speaks:

  POST /v1/chat/completions   (sync JSON + streaming SSE)   — llama.cpp & vLLM
  POST /embed                 (embeddings shim)
  POST /rerank                (rerank shim)
  GET  /props                 (llama.cpp capacity discovery)
  GET  /v1/models             (served-model id + vLLM max_model_len)
  GET  /metrics               (vLLM Prometheus prefix-cache counters)
  GET  /health                (liveness)

It serves correct happy-path responses AND, on command, any **south-face
pathology** the proxy must survive: truncated/invalid JSON, empty completion,
degenerate repetition, wrong/invalid schema, phantom & truncated tool_calls,
partial/interleaved SSE frames, TTFT stall, inter-token stall, mid-stream reset,
4xx/5xx, timeout, capacity desync, slow-drain.

Fault selection, per request, in priority order:
  1. the ``X-Fault`` request header (+ optional ``X-Fault-Arg`` numeric arg) — so a
     single running server can serve a *different* fault per call (needed for the
     duplicate-storm / mixed-traffic tests);
  2. the controller's ``default_fault`` (set by a test via ``FakeBackend.set_fault``).

The app is normally driven through :class:`FakeBackendServer`, which runs uvicorn
on a real ephemeral 127.0.0.1 socket in a background thread, so the proxy's real
``httpx`` client + real SSE framing are exercised end-to-end (higher fidelity than
an in-process ``MockTransport`` for the mid-stream/reset/interleaved-frame faults).
A ``MockTransport`` handler (:func:`mock_transport_handler`) is also exported for
pure-unit cases that don't need a socket.

Kept dependency-free beyond Starlette/uvicorn/httpx — all already declared
package dependencies, so importing this adds nothing to the dependency set.
🚨 Keep it that way: this module is now public surface, and a new import here is
a new import for everyone who installs Roadstead.

Nothing in the library core imports it. It loads only when asked for by name, so
a production deployment never pays for it.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any, AsyncIterator, Callable, Deque, Dict, List, Optional, Tuple,
)

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route


# --------------------------------------------------------------------------- #
# Fault catalogue
# --------------------------------------------------------------------------- #
# Names are the single source of truth for both the fake and the meta-tests that
# assert each fault actually fires. Keep this list and the handlers in sync; the
# meta-test iterates ALL_FAULTS.

# Happy path.
FAULT_NONE = "none"

# --- non-streaming HTTP-status / body pathologies ---
FAULT_HTTP_400 = "http_400"            # 4xx  -> BackendError(400)
FAULT_HTTP_500 = "http_500"            # 5xx  -> BackendError(500)
FAULT_HTTP_503 = "http_503"            # 503  -> BackendError(503)
FAULT_TRUNCATED_JSON = "truncated_json"  # 200 with unparseable body
FAULT_EMPTY_COMPLETION = "empty_completion"  # 200, blank content, no tool_calls -> proxy 502
FAULT_NO_USAGE = "no_usage"            # 200 valid, but usage block omitted
FAULT_FINISH_LENGTH = "finish_length"  # finish_reason=length (truncation signal)
FAULT_DEGENERATE_LOOP = "degenerate_loop"  # same long n-gram repeated many times
FAULT_SCHEMA_VALID_WRONG = "schema_valid_wrong"    # valid JSON, semantically wrong
FAULT_SCHEMA_INVALID = "schema_invalid"            # parseable JSON, violates declared schema
FAULT_PHANTOM_TOOL_CALLS = "phantom_tool_calls"    # tool_calls w/ malformed args, empty content

# --- streaming (SSE) pathologies ---
FAULT_TTFT_STALL = "ttft_stall"        # delay before first chunk (arg=seconds)
FAULT_INTERTOKEN_STALL = "intertoken_stall"  # delay between chunks (arg=seconds)
FAULT_MID_STREAM_RESET = "mid_stream_reset"  # abort the connection mid-stream
FAULT_PARTIAL_SSE = "partial_sse"      # a data: frame split across writes / incomplete
FAULT_INTERLEAVED_SSE = "interleaved_sse"    # comment/keepalive lines interleaved with frames
FAULT_NO_DONE = "no_done"              # stream ends without the [DONE] sentinel
FAULT_SLOW_DRAIN = "slow_drain"        # valid stream, tokens emitted slowly (arg=seconds/token)
FAULT_TRUNCATED_TOOL_CALLS = "truncated_tool_calls"  # streamed tool_call args cut off mid-JSON

# --- transport / capacity ---
FAULT_TIMEOUT = "timeout"              # sleep past the caller deadline (arg=seconds)
FAULT_CAPACITY_DESYNC = "capacity_desync"  # accept only N concurrent, 503 beyond (props lies)

ALL_FAULTS: Tuple[str, ...] = (
    FAULT_NONE,
    FAULT_HTTP_400, FAULT_HTTP_500, FAULT_HTTP_503,
    FAULT_TRUNCATED_JSON, FAULT_EMPTY_COMPLETION, FAULT_NO_USAGE,
    FAULT_FINISH_LENGTH, FAULT_DEGENERATE_LOOP,
    FAULT_SCHEMA_VALID_WRONG, FAULT_SCHEMA_INVALID, FAULT_PHANTOM_TOOL_CALLS,
    FAULT_TTFT_STALL, FAULT_INTERTOKEN_STALL, FAULT_MID_STREAM_RESET,
    FAULT_PARTIAL_SSE, FAULT_INTERLEAVED_SSE, FAULT_NO_DONE, FAULT_SLOW_DRAIN,
    FAULT_TRUNCATED_TOOL_CALLS,
    FAULT_TIMEOUT, FAULT_CAPACITY_DESYNC,
)

# Faults meaningful only on the streaming path.
STREAM_ONLY_FAULTS: Tuple[str, ...] = (
    FAULT_TTFT_STALL, FAULT_INTERTOKEN_STALL, FAULT_MID_STREAM_RESET,
    FAULT_PARTIAL_SSE, FAULT_INTERLEAVED_SSE, FAULT_NO_DONE, FAULT_SLOW_DRAIN,
    FAULT_TRUNCATED_TOOL_CALLS,
)


# Sentinels for the adversarial usage-shape knobs (below). ``USAGE_DEFAULT``
# = "use the normal happy-path usage block"; ``OMIT_USAGE`` = "emit no usage
# block at all". Distinct objects so ``None`` remains a legal override value
# (an explicit ``usage: null``) — which is the whole point: a backend that sends
# ``"usage": null`` and one that sends no ``usage`` key are different bugs.
#
# Public names since the 2026-08-31 promotion out of tests/: a caller of a
# shipped module should not have to import underscore-prefixed sentinels to use
# a documented knob. ``_UNSET``/``_OMIT_USAGE`` remain as aliases — they are the
# spelling in this module's own history and in any monorepo copy.
USAGE_DEFAULT: Any = object()
OMIT_USAGE: Any = object()

_UNSET = USAGE_DEFAULT
_OMIT_USAGE = OMIT_USAGE


class MidStreamReset(RuntimeError):
    """Raised inside the SSE generator to abort the connection mid-body so the
    real httpx client sees a RemoteProtocolError (proxy -> BackendUnavailable)."""


@dataclass
class RecordedRequest:
    """One backend call the fake received — inspected by tests/meta-tests."""
    method: str
    path: str
    fault: str
    body: Optional[dict]
    headers: Dict[str, str]


#: How many recorded requests :class:`FakeBackend` keeps. Large enough that no
#: test in this repo notices, small enough that a multi-hour load run does not
#: turn the recorder into the thing being measured. See ``FakeBackend.requests``.
REQUEST_LOG_CAPACITY = 1000


@dataclass
class FakeBackend:
    """Mutable controller shared by the ASGI app and the test.

    A test flips ``default_fault`` (and optional ``fault_arg``) to steer every
    subsequent call, or sends a per-call ``X-Fault`` header. All received calls
    are recorded in ``requests`` for assertions.
    """
    # engine shape this endpoint imitates: "llama.cpp", "vllm", or "openrouter"
    #
    # 🚨 "openrouter" is a REMOTE shape and differs in more than a field name:
    # its routes hang off a base path (/api/v1) instead of the origin, it
    # REQUIRES a bearer token and 401s without one, and its /models is a
    # CATALOGUE carrying context_length and per-token pricing rather than one
    # served model. Those are the four things a local engine never makes a
    # gateway deal with, so a fake that only spoke the local shapes could not
    # exercise a remote provider at all.
    engine: str = "llama.cpp"
    served_model_id: str = "fake-model"
    props_n_parallel: int = 4
    props_n_ctx: int = 32768
    # Which /props SHAPE to publish. Verified against a real llama-server
    # (b5350) on 2026-08-31 — see tests/wire_fidelity/README.md.
    #
    #   "modern" (default) — what a current build ACTUALLY publishes:
    #       total_slots, and default_generation_settings.n_ctx (per-slot).
    #       It does NOT publish default_generation_settings.n_parallel, a
    #       `slots` list, or a top-level n_ctx. 🚨 That matters: capacity
    #       discovery PREFERS n_parallel and only falls back to total_slots, so
    #       with the old superset shape the fallback a real engine actually
    #       depends on was never exercised. Anyone "simplifying" that fallback
    #       away would have kept a green suite and broken real discovery.
    #
    #   "legacy" — the superset this fake published before the shape was
    #       verified: every field at once. Kept so a build that does publish
    #       n_parallel or a top-level n_ctx can still be emulated, not because
    #       any observed engine emits all of it.
    props_profile: str = "modern"
    max_model_len: int = 40960
    # --- remote ("openrouter") shape knobs --------------------------------- #
    # The token this fake accepts. A request without `Authorization: Bearer
    # <api_key>` gets a 401, so a provider that forgets its credential fails
    # here the way it would fail against the real thing rather than passing.
    api_key: str = "fake-openrouter-key"
    # The catalogue /models publishes. Deliberately MORE than one entry: picking
    # the right row out of a catalogue is the parsing step a single-model fake
    # would let a provider skip, and "took the first row" is a confidently wrong
    # context ceiling feeding the admission gate.
    catalogue_context_length: int = 131072
    catalogue_prompt_cost: str = "0.0000005"
    catalogue_completion_cost: str = "0.0000015"
    # prefix-cache counters exposed on /metrics (vLLM shape)
    prefix_cache_hits: int = 0
    prefix_cache_queries: int = 0
    # --- engine WORK counters on /metrics ---------------------------------- #
    # Drive the per-request progress probe AND the endpoint-level
    # goodput-collapse detector. ``None`` = do not publish that series at all,
    # which is the shape that must read as "cannot discriminate" rather than as a
    # zero: a real llama.cpp backend publishes no iteration counter, and a
    # detector that reads its absence as 0 concludes the engine has stopped.
    #
    # Names emitted depend on ``engine``: vLLM's ``vllm:*`` set, or llama.cpp's
    # ``llamacpp:*`` set (where there is no iteration counter to publish, so
    # ``iteration_tokens_count`` is ignored).
    prompt_tokens_total: Optional[int] = None
    generation_tokens_total: Optional[int] = None
    iteration_tokens_count: Optional[int] = None
    num_requests_running: Optional[int] = None
    # 🚨 When True (and the counter above is set), publish the OTHER members of
    # vLLM's ``vllm:iteration_tokens_total`` HISTOGRAM family — the ``_bucket``
    # lines with their ``le=`` labels, ``_sum``, ``_created`` — with values far
    # larger than ``_count``. A parser that matches the family by prefix sums
    # them into a number that rises monotonically and looks exactly like a
    # working counter. Default True so the trap is armed by default rather than
    # opted into by whoever remembered it.
    emit_iteration_histogram_siblings: bool = True

    default_fault: str = FAULT_NONE
    fault_arg: float = 0.0
    # Apply the default fault at most this many times, then serve happy — models
    # a TRANSIENT backend fault that recovers on retry (degeneration re-dispatch,
    # empty-rescue). 0 = unlimited (persistent fault). Header-driven faults are
    # never rate-limited (they are per-call by construction).
    fault_max_hits: int = 0
    # For FAULT_CAPACITY_DESYNC: accept at most this many concurrent chat calls.
    accept_limit: int = 1
    # Phase 2a — per-request prefix-cache attribution. When set, the happy-path
    # completion (sync + the streaming usage frame) emits
    # ``usage.prompt_tokens_details.cached_tokens`` (the vLLM shape); prompt_tokens
    # is fixed at 12 so a test can assert an exact hit rate (cached/12). None
    # (default) emits NO such field — the llama.cpp shape the proxy must treat as
    # n/a, distinct from a reported 0.
    cached_tokens: Optional[int] = None
    # Phase 2a ADVERSARIAL (independent track): drive arbitrary hostile usage
    # shapes the plain ``cached_tokens`` int knob can't express. When
    # ``usage_override`` is not ``_UNSET`` it REPLACES the entire happy-path
    # ``usage`` block verbatim (sync body + the streaming usage frame); set it to
    # ``_OMIT_USAGE`` to emit no usage at all. Lets a test send
    # ``prompt_tokens_details`` as a non-dict/null/list, a missing/zero
    # ``prompt_tokens`` (division-by-zero probe), an explicit null cached_tokens,
    # etc. NB: ``cached_tokens`` itself is only type-hinted ``int`` — a test may
    # assign ANY value (float/str/bool/list/NaN) to exercise the parser.
    usage_override: Any = _UNSET
    # When set (str), the SYNC happy-path chat returns this EXACT string as a 200
    # ``application/json`` body. The only way to put NaN/Infinity on the sync
    # wire: Starlette's JSONResponse hard-codes ``allow_nan=False`` and would 500
    # inside the fake, whereas ``json.loads`` (what the proxy's ``resp.json()``
    # uses) DOES accept ``NaN``/``Infinity`` — so a raw body reaches the parser.
    raw_completion_text: Optional[str] = None
    # Phase 3 schema-backstop (independent adversarial track). When set, the SYNC
    # happy-path (fault-none) completion returns this EXACT string as the
    # assistant ``content`` (instead of ``echo: …``) — the only way to steer the
    # backend's *structured* content (fenced / trailing-prose / non-JSON /
    # schema-conforming) without adding a named fault (keeps ALL_FAULTS meta
    # coverage untouched). Also serves, via ``fault_max_hits``, as the RECOVERED
    # body a retry sees.
    structured_content: Optional[str] = None
    # Phase 3 — when set, the SYNC happy-path completion carries a single
    # tool_call whose ``function.arguments`` is this EXACT string (empty content),
    # so a test can emit tool-call args that are repairable or hopeless.
    structured_tool_args: Optional[str] = None

    #: The last :data:`REQUEST_LOG_CAPACITY` requests this backend received,
    #: oldest first. Indexing, ``len`` and iteration all work as they did.
    #:
    #: 🚨 **Bounded, and it was not.** It was a plain list holding a body dict
    #: and a header dict per request, kept for the life of the process — so a
    #: sustained run accumulated one record per request forever. That is a real
    #: defect for anyone who installs this (it is shipped surface), and it had a
    #: second, worse consequence here: ``tools/soak.py`` exists to detect leaks
    #: by RSS slope, and its headline number was dominated by its own test
    #: double's recorder. A measuring instrument was reporting its own artifact.
    #:
    #: Dropping silently would be the wrong fix on this codebase's own terms, so
    #: the bound reports itself: :attr:`requests_seen` counts every request ever
    #: recorded and :attr:`requests_dropped` says how many aged out. A test that
    #: needs more than the cap can read the totals instead of the records.
    requests: Deque[RecordedRequest] = field(
        default_factory=lambda: deque(maxlen=REQUEST_LOG_CAPACITY))
    #: Total requests ever recorded, including those the bound has dropped.
    requests_seen: int = 0
    # live concurrency counter (capacity_desync)
    _inflight: int = 0
    _hits: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def requests_dropped(self) -> int:
        """How many recorded requests the bound has aged out.

        Reported rather than hidden: a caller that asserts on ``len(requests)``
        over a long run is entitled to know the list stopped being complete,
        instead of reading a number that silently means something else.
        """
        return max(0, self.requests_seen - len(self.requests))

    def set_fault(self, name: str, arg: float = 0.0, max_hits: int = 0) -> None:
        if name not in ALL_FAULTS:
            raise ValueError(f"unknown fault {name!r}")
        self.default_fault = name
        self.fault_arg = arg
        self.fault_max_hits = max_hits
        self._hits = 0

    def reset(self) -> None:
        self.default_fault = FAULT_NONE
        self.fault_arg = 0.0
        self.fault_max_hits = 0
        self.cached_tokens = None
        self.usage_override = _UNSET
        self.raw_completion_text = None
        self.structured_content = None
        self.structured_tool_args = None
        self.requests.clear()
        self.requests_seen = 0
        with self._lock:
            self._inflight = 0
            self._hits = 0

    # -- fault resolution --------------------------------------------------- #
    def _resolve_fault(self, request: Request) -> Tuple[str, float]:
        hdr = request.headers.get("x-fault")
        if hdr:
            arg_raw = request.headers.get("x-fault-arg", "")
            try:
                arg = float(arg_raw) if arg_raw else 0.0
            except ValueError:
                arg = 0.0
            return hdr, arg
        if self.default_fault != FAULT_NONE and self.fault_max_hits:
            with self._lock:
                if self._hits >= self.fault_max_hits:
                    return FAULT_NONE, 0.0
                self._hits += 1
        return self.default_fault, self.fault_arg


# --------------------------------------------------------------------------- #
# Response builders
# --------------------------------------------------------------------------- #

def _last_user_text(body: Optional[dict]) -> str:
    if not body:
        return ""
    msgs = body.get("messages") or []
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):  # OpenAI multimodal content parts
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return str(part.get("text", ""))
    return ""


def _completion_body(content: str, *, finish: str = "stop",
                     with_usage: bool = True,
                     tool_calls: Optional[list] = None,
                     cached_tokens: Any = None,
                     usage_override: Any = _UNSET,
                     model: str = "fake-model") -> dict:
    msg: Dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    body: Dict[str, Any] = {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
    }
    if usage_override is not _UNSET:
        # Adversarial: verbatim usage block (or none at all).
        if usage_override is not _OMIT_USAGE:
            body["usage"] = usage_override
        return body
    if with_usage:
        pt = 12
        ct = max(1, len(content.split()))
        usage: Dict[str, Any] = {"prompt_tokens": pt, "completion_tokens": ct,
                                 "total_tokens": pt + ct}
        if cached_tokens is not None:  # Phase 2a — vLLM prefix-cache shape
            usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
        body["usage"] = usage
    return body


def _sse(obj: dict) -> bytes:
    return ("data: " + json.dumps(obj) + "\n\n").encode()


def _chunk(delta: dict, finish: Optional[str] = None) -> dict:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #

def make_fake_app(controller: FakeBackend) -> Starlette:
    """Build the Starlette ASGI app bound to ``controller``."""

    async def _record(request: Request, fault: str, body: Optional[dict]) -> None:
        controller.requests_seen += 1
        controller.requests.append(RecordedRequest(
            method=request.method, path=request.url.path, fault=fault,
            body=body, headers={k: v for k, v in request.headers.items()},
        ))

    async def _sleep_or_disconnect(
        request: Request, seconds: float, step: float = 0.05,
    ) -> None:
        """Sleep up to ``seconds`` but return the instant the client
        disconnects. A plain (non-streaming) Response handler is NOT cancelled
        by uvicorn on client disconnect, so a naive ``asyncio.sleep(arg)`` keeps
        the fake's worker busy for the full ``arg`` even after the proxy aborts
        at its own deadline — which then blocks fixture teardown (thread-join)
        for ~arg seconds. Polling for disconnect preserves the injected-timeout
        semantics (outlast the caller deadline) while freeing the worker
        promptly. StreamingResponse handlers already self-cancel on disconnect,
        so only the sync timeout fault needs this."""
        elapsed = 0.0
        while elapsed < seconds:
            try:
                if await request.is_disconnected():
                    return
            except Exception:
                return
            await asyncio.sleep(min(step, seconds - elapsed))
            elapsed += step

    # ---- chat completions (sync + stream) -------------------------------- #
    async def chat(request: Request) -> Response:
        try:
            body = await request.json()
        except Exception:
            body = None
        fault, arg = controller._resolve_fault(request)
        await _record(request, fault, body)
        streaming = bool(body and body.get("stream"))

        # capacity desync applies to both sync/stream: 503 once over the limit.
        if fault == FAULT_CAPACITY_DESYNC:
            with controller._lock:
                controller._inflight += 1
                over = controller._inflight > controller.accept_limit
            if over:
                with controller._lock:
                    controller._inflight -= 1
                return JSONResponse({"error": "no slot available"}, status_code=503)
            try:
                await asyncio.sleep(max(arg, 0.05))
                return JSONResponse(_completion_body(
                    "echo: " + _last_user_text(body)))
            finally:
                with controller._lock:
                    controller._inflight -= 1

        # transport faults
        if fault == FAULT_HTTP_400:
            return JSONResponse({"error": "bad request (injected)"}, status_code=400)
        if fault == FAULT_HTTP_500:
            return JSONResponse({"error": "internal error (injected)"}, status_code=500)
        if fault == FAULT_HTTP_503:
            return JSONResponse({"error": "unavailable (injected)"}, status_code=503)
        if fault == FAULT_TIMEOUT:
            await _sleep_or_disconnect(request, arg if arg > 0 else 2.0)
            # If the caller hasn't already abandoned us, still answer.
            return JSONResponse(_completion_body("late: " + _last_user_text(body)))

        if streaming:
            return _stream_response(body, fault, arg)

        # non-streaming body pathologies
        if fault == FAULT_TRUNCATED_JSON:
            # 200 with an unparseable body: proxy resp.json() fails -> {"raw":..}
            # -> chat path sees no choices -> empty-completion 502.
            return PlainTextResponse(
                '{"choices":[{"message":{"content":"hel',
                status_code=200, media_type="application/json")
        if fault == FAULT_EMPTY_COMPLETION:
            return JSONResponse(_completion_body(""))
        if fault == FAULT_NO_USAGE:
            return JSONResponse(_completion_body(
                "echo: " + _last_user_text(body), with_usage=False))
        if fault == FAULT_FINISH_LENGTH:
            return JSONResponse(_completion_body(
                "truncated mid-thought", finish="length"))
        if fault == FAULT_DEGENERATE_LOOP:
            loop = ("the song of the sea " * 40).strip()
            return JSONResponse(_completion_body(loop))
        if fault == FAULT_SCHEMA_VALID_WRONG:
            # valid JSON object, but not what the (declared) schema wanted.
            return JSONResponse(_completion_body('{"unexpected": "shape", "n": 1}'))
        if fault == FAULT_SCHEMA_INVALID:
            # parseable JSON with a trailing-prose tail — the classic
            # "parseable but schema-invalid" case Phase 3 must repair.
            return JSONResponse(_completion_body(
                '{"answer": "yes"} — and here is some extra prose the schema forbids'))
        if fault == FAULT_PHANTOM_TOOL_CALLS:
            # tool_calls present (so the empty-content gate is bypassed) but the
            # arguments are malformed JSON — the Phase-3 tool backstop target.
            tc = [{
                "id": "call_0", "type": "function",
                "function": {"name": "do_thing", "arguments": '{"x": 1'},
            }]
            return JSONResponse(_completion_body("", tool_calls=tc))

        # happy path
        if controller.raw_completion_text is not None:
            # Verbatim body (NaN/Infinity etc. that JSONResponse would reject).
            return PlainTextResponse(
                controller.raw_completion_text, status_code=200,
                media_type="application/json")
        if controller.structured_tool_args is not None:
            # Phase 3: a single tool_call with caller-chosen (possibly hopeless)
            # arguments + empty content — the tool-call backstop target.
            tc = [{
                "id": "call_0", "type": "function",
                "function": {"name": "do_thing",
                             "arguments": controller.structured_tool_args},
            }]
            return JSONResponse(_completion_body("", tool_calls=tc))
        if controller.structured_content is not None:
            # Phase 3: caller-chosen structured content (fenced / prose / JSON).
            return JSONResponse(_completion_body(controller.structured_content))
        return JSONResponse(_completion_body(
            "echo: " + _last_user_text(body),
            cached_tokens=controller.cached_tokens,
            usage_override=controller.usage_override,
            model=controller.served_model_id))

    def _stream_response(body: Optional[dict], fault: str, arg: float) -> StreamingResponse:
        text = "echo: " + _last_user_text(body)
        tokens = text.split(" ")

        async def gen() -> AsyncIterator[bytes]:
            # role preamble
            if fault == FAULT_TTFT_STALL:
                await asyncio.sleep(arg if arg > 0 else 1.0)
            yield _sse(_chunk({"role": "assistant"}))

            if fault == FAULT_MID_STREAM_RESET:
                yield _sse(_chunk({"content": tokens[0] if tokens else "x"}))
                # Abort the connection mid-body: real httpx sees RemoteProtocolError.
                raise MidStreamReset("injected mid-stream reset")

            if fault == FAULT_DEGENERATE_LOOP:
                for _ in range(40):
                    yield _sse(_chunk({"content": "the song of the sea "}))
                yield _sse(_chunk({}, finish="stop"))
                yield b"data: [DONE]\n\n"
                return

            if fault == FAULT_TRUNCATED_TOOL_CALLS:
                # stream a tool_call whose argument JSON is cut off mid-object
                yield _sse(_chunk({"tool_calls": [{
                    "index": 0, "id": "call_0", "type": "function",
                    "function": {"name": "do_thing", "arguments": '{"x":'},
                }]}))
                yield _sse(_chunk({}, finish="tool_calls"))
                yield b"data: [DONE]\n\n"
                return

            if fault == FAULT_PARTIAL_SSE:
                # emit a valid frame, then a *partial* data line with no
                # terminating blank line, then finish properly.
                yield _sse(_chunk({"content": "partial "}))
                yield b'data: {"choices":[{"index":0,"delta":{"content":"cut'
                yield _sse(_chunk({}, finish="stop"))
                yield b"data: [DONE]\n\n"
                return

            for i, tok in enumerate(tokens):
                if fault == FAULT_INTERTOKEN_STALL and i == 1:
                    await asyncio.sleep(arg if arg > 0 else 1.0)
                if fault == FAULT_SLOW_DRAIN:
                    await asyncio.sleep(arg if arg > 0 else 0.02)
                if fault == FAULT_INTERLEAVED_SSE:
                    # SSE comment / keepalive lines the proxy parser must skip.
                    yield b": keepalive\n\n"
                    yield b"\n"
                yield _sse(_chunk({"content": (tok + " ")}))

            yield _sse(_chunk({}, finish="stop"))
            # usage-only frame (include_usage) then DONE — unless suppressed.
            # Adversarial usage_override wins; else the normal (optionally
            # cached_tokens-bearing) block. _sse uses json.dumps (allow_nan=True),
            # so a NaN/Infinity cached_tokens rides the stream frame verbatim.
            if controller.usage_override is not _UNSET:
                if controller.usage_override is not _OMIT_USAGE:
                    yield _sse({
                        "id": "chatcmpl-fake", "object": "chat.completion.chunk",
                        "choices": [], "usage": controller.usage_override,
                    })
            else:
                _usage: Dict[str, Any] = {
                    "prompt_tokens": 12, "completion_tokens": len(tokens),
                    "total_tokens": 12 + len(tokens)}
                if controller.cached_tokens is not None:  # Phase 2a — vLLM shape
                    _usage["prompt_tokens_details"] = {
                        "cached_tokens": controller.cached_tokens}
                yield _sse({
                    "id": "chatcmpl-fake", "object": "chat.completion.chunk",
                    "choices": [], "usage": _usage,
                })
            if fault != FAULT_NO_DONE:
                yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    # ---- embeddings shim -------------------------------------------------- #
    async def embed(request: Request) -> Response:
        try:
            body = await request.json()
        except Exception:
            body = None
        fault, arg = controller._resolve_fault(request)
        await _record(request, fault, body)
        if fault == FAULT_HTTP_500:
            return JSONResponse({"error": "embed failed (injected)"}, status_code=500)
        if fault == FAULT_TRUNCATED_JSON:
            return PlainTextResponse('{"data":[{"embedding":[0.0,0.1', status_code=200,
                                     media_type="application/json")
        inputs = []
        if body:
            raw = body.get("input")
            if isinstance(raw, list):
                inputs = raw
            elif raw is not None:
                inputs = [raw]
        if not inputs:
            inputs = [""]
        data = [{"object": "embedding", "index": i, "embedding": [0.01 * (i + 1)] * 8}
                for i in range(len(inputs))]
        return JSONResponse({
            "object": "list", "data": data, "model": controller.served_model_id,
            "usage": {"prompt_tokens": 8, "completion_tokens": 0, "total_tokens": 8},
        })

    # ---- rerank shim ------------------------------------------------------ #
    async def rerank(request: Request) -> Response:
        try:
            body = await request.json()
        except Exception:
            body = None
        fault, arg = controller._resolve_fault(request)
        await _record(request, fault, body)
        if fault == FAULT_HTTP_500:
            return JSONResponse({"error": "rerank failed (injected)"}, status_code=500)
        docs = (body or {}).get("documents") or (body or {}).get("texts") or []
        results = [{"index": i, "relevance_score": 1.0 / (i + 1)}
                   for i in range(len(docs))]
        return JSONResponse({"results": results,
                             "usage": {"prompt_tokens": 8, "completion_tokens": 0}})

    # ---- capacity / discovery probes ------------------------------------- #
    async def props(request: Request) -> Response:
        if controller.props_profile == "legacy":
            # Every field at once. NB the top-level n_ctx carries the SAME value
            # as the per-slot one, which cannot be right for both readings —
            # capacity discovery divides the top-level by the slot count. It is
            # preserved verbatim only so a test can drive the legacy branch;
            # a real engine has not been observed publishing it at all.
            return JSONResponse({
                "default_generation_settings": {
                    "n_parallel": controller.props_n_parallel,
                    "n_ctx": controller.props_n_ctx,
                },
                "total_slots": controller.props_n_parallel,
                "slots": [{} for _ in range(controller.props_n_parallel)],
                "n_ctx": controller.props_n_ctx,
            })
        # "modern": the verified b5350 shape. Deliberately NARROW — the point of
        # a fake is to be as stingy as the real thing, not as generous as the
        # reader can cope with.
        return JSONResponse({
            "default_generation_settings": {
                "n_ctx": controller.props_n_ctx,
            },
            "total_slots": controller.props_n_parallel,
            "build_info": "fake-backend",
            "model_path": f"/fake/{controller.served_model_id}.gguf",
        })

    async def models(request: Request) -> Response:
        # Per-engine, verified 2026-08-31. The two shapes differ in exactly the
        # fields the proxy reads, and emitting the union would let a probe pass
        # here that cannot pass against the real thing:
        #
        #   max_model_len — vLLM ONLY. It is the sole capacity fact vLLM
        #       publishes, which is why vLLM concurrency stays config-seeded.
        #       llama.cpp does not have it (b5350 confirmed); a fake that
        #       offered it anyway would let probe_vllm_capacity "succeed"
        #       against a llama.cpp endpoint.
        #   root          — vLLM ONLY: the weights path, which is what
        #       probe_model_fingerprint prefers because a served alias can be
        #       repointed at different weights without changing.
        #   meta          — llama.cpp ONLY: {n_params, n_vocab, ...}, the
        #       fingerprint FALLBACK. Real values below are the shape b5350
        #       returns, so the fallback is exercised rather than assumed.
        entry: Dict[str, Any] = {
            "id": controller.served_model_id, "object": "model",
        }
        if controller.engine == "vllm":
            entry["max_model_len"] = controller.max_model_len
            entry["root"] = f"/srv/models/{controller.served_model_id}"
        else:
            entry["meta"] = {
                "vocab_type": 2,
                "n_vocab": 151936,
                "n_ctx_train": 32768,
                "n_embd": 896,
                "n_params": 630167424,
                "size": 669763072,
            }
        return JSONResponse({"object": "list", "data": [entry]})

    async def metrics(request: Request) -> Response:
        lines = [
            "# HELP vllm:prefix_cache_hits_total Prefix cache hits.",
            "# TYPE vllm:prefix_cache_hits_total counter",
            'vllm:prefix_cache_hits_total{model_name="%s"} %d.0' % (
                controller.served_model_id, controller.prefix_cache_hits),
            'vllm:prefix_cache_queries_total{model_name="%s"} %d.0' % (
                controller.served_model_id, controller.prefix_cache_queries),
        ]
        model = controller.served_model_id
        if controller.engine == "vllm":
            if controller.prompt_tokens_total is not None:
                lines.append('vllm:prompt_tokens_total{engine="0",model_name="%s"} %d.0'
                             % (model, controller.prompt_tokens_total))
            if controller.generation_tokens_total is not None:
                lines.append('vllm:generation_tokens_total{engine="0",model_name="%s"} %d.0'
                             % (model, controller.generation_tokens_total))
            if controller.num_requests_running is not None:
                lines.append('vllm:num_requests_running{engine="0",model_name="%s"} %d.0'
                             % (model, controller.num_requests_running))
            if controller.iteration_tokens_count is not None:
                if controller.emit_iteration_histogram_siblings:
                    # The trap, as the real engine lays it: same family prefix,
                    # values an order of magnitude larger, emitted BEFORE the
                    # `_count` line so a prefix matcher has already been poisoned
                    # by the time the right series arrives.
                    lines.append("# TYPE vllm:iteration_tokens_total histogram")
                    for le in ("1.0", "8.0", "64.0", "+Inf"):
                        lines.append(
                            'vllm:iteration_tokens_total_bucket{engine="0",'
                            'le="%s",model_name="%s"} %d.0'
                            % (le, model, controller.iteration_tokens_count * 7))
                    lines.append('vllm:iteration_tokens_total_sum{engine="0",'
                                 'model_name="%s"} %d.0'
                                 % (model, controller.iteration_tokens_count * 113))
                    lines.append('vllm:iteration_tokens_total_created{engine="0",'
                                 'model_name="%s"} 1.757e+09' % model)
                lines.append('vllm:iteration_tokens_total_count{engine="0",'
                             'model_name="%s"} %d.0'
                             % (model, controller.iteration_tokens_count))
        else:
            # llama.cpp: unlabelled series, and NO iteration counter exists — see
            # `backend.probe_progress_counters` on why `n_decode_total` is not
            # mapped to both the generation and the iteration clause.
            if controller.prompt_tokens_total is not None:
                lines.append("llamacpp:prompt_tokens_total %d"
                             % controller.prompt_tokens_total)
            if controller.generation_tokens_total is not None:
                # `tokens_predicted_total` is published too, FROZEN at 0 — the
                # measured trap: it is credited only at request completion, so a
                # parser that takes it by name reads "no progress" throughout the
                # window the watchdog judges.
                lines.append("llamacpp:tokens_predicted_total 0")
                lines.append("llamacpp:n_decode_total %d"
                             % controller.generation_tokens_total)
            if controller.num_requests_running is not None:
                lines.append("llamacpp:requests_processing %d"
                             % controller.num_requests_running)
        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain")

    async def health(request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    # ---- remote shape ----------------------------------------------------- #
    def _unauthorized(request: Request) -> Optional[Response]:
        """A remote provider's first difference from a local one: it says no.

        Checked BEFORE anything else, including faults, because that is the
        order a real gateway hits it in — a credential problem is not one of the
        pathologies this fake injects, it is the wall in front of them."""
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {controller.api_key}":
            return JSONResponse(
                {"error": {"message": "No auth credentials found",
                           "code": 401}}, status_code=401)
        return None

    async def remote_chat(request: Request) -> Response:
        denied = _unauthorized(request)
        if denied is not None:
            return denied
        return await chat(request)

    async def remote_models(request: Request) -> Response:
        """The catalogue. Two entries, one of which is ours — see
        ``catalogue_context_length``."""
        denied = _unauthorized(request)
        if denied is not None:
            return denied
        pricing = {"prompt": controller.catalogue_prompt_cost,
                   "completion": controller.catalogue_completion_cost,
                   "image": "0", "request": "0"}
        return JSONResponse({"data": [
            {"id": "someone-else/some-other-model",
             "name": "Not the one this endpoint is pinned to",
             "context_length": 8192,
             "pricing": dict(pricing, prompt="0.09")},
            {"id": controller.served_model_id,
             "name": "The pinned model",
             "context_length": controller.catalogue_context_length,
             "pricing": pricing},
        ]})

    if controller.engine == "openrouter":
        # Deliberately NOT a superset: a remote provider gets the remote routes
        # and nothing else. No /props, no /health, no /metrics — the fake is as
        # stingy as the real thing, so a probe that only works because the fake
        # was generous fails here instead of in production.
        routes = [
            Route("/api/v1/chat/completions", remote_chat, methods=["POST"]),
            Route("/api/v1/models", remote_models, methods=["GET"]),
        ]
    else:
        routes = [
            Route("/v1/chat/completions", chat, methods=["POST"]),
            Route("/embed", embed, methods=["POST"]),
            Route("/rerank", rerank, methods=["POST"]),
            Route("/props", props, methods=["GET"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/metrics", metrics, methods=["GET"]),
            Route("/health", health, methods=["GET"]),
        ]
    app = Starlette(routes=routes)
    app.state.controller = controller
    return app


# --------------------------------------------------------------------------- #
# Real-socket server (uvicorn in a background thread)
# --------------------------------------------------------------------------- #

def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


class FakeBackendServer:
    """Runs :func:`make_fake_app` on a real 127.0.0.1 socket in a daemon thread.

    Usage::

        srv = FakeBackendServer(FakeBackend(engine="vllm"))
        srv.start()
        try:
            ...  # point an EndpointConfig at srv.host / srv.port
        finally:
            srv.stop()
    """

    def __init__(self, controller: Optional[FakeBackend] = None,
                 host: str = "127.0.0.1", port: Optional[int] = None) -> None:
        self.controller = controller or FakeBackend()
        self.host = host
        self.port = port or _free_port()
        self.app = make_fake_app(self.controller)
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def base_url(self) -> str:
        """What an ``EndpointConfig.base_url`` should be pointed at.

        Same as ``url`` for a local engine shape. The remote shape adds the
        ``/api/v1`` base path, because "the routes are not at the origin" is one
        of the things that make a remote provider different, and a test that
        papered over it would leave the base-path composition untested."""
        if self.controller.engine == "openrouter":
            return f"{self.url}/api/v1"
        return self.url

    def start(self, timeout: float = 5.0) -> "FakeBackendServer":
        config = uvicorn.Config(
            self.app, host=self.host, port=self.port,
            log_level="warning", access_log=False, lifespan="off",
        )
        self._server = uvicorn.Server(config)
        # Silence uvicorn's install_signal_handlers (only valid on main thread).
        self._server.install_signal_handlers = lambda: None
        self._thread = threading.Thread(target=self._server.run, daemon=True,
                                        name=f"fake-backend-{self.port}")
        self._thread.start()
        # wait until the socket accepts connections
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._server.started:
                return self
            try:
                with socket.create_connection((self.host, self.port), timeout=0.1):
                    return self
            except OSError:
                time.sleep(0.02)
        raise RuntimeError(f"fake backend did not start on {self.url}")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._server = None
        self._thread = None

    def __enter__(self) -> "FakeBackendServer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


# --------------------------------------------------------------------------- #
# MockTransport handler (no socket — pure-unit fallback)
# --------------------------------------------------------------------------- #

def mock_transport_handler(controller: FakeBackend) -> Callable[[Any], Any]:
    """Return an httpx.MockTransport handler serving the same happy-path shapes.

    Only covers the non-streaming happy path + discovery probes — for the
    streaming/reset/interleave faults use :class:`FakeBackendServer` (a real
    socket). Handy where a test just needs a deterministic 200 without a thread.
    """
    import httpx

    def handler(request: "httpx.Request") -> "httpx.Response":
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/props":
            return httpx.Response(200, json={
                "default_generation_settings": {
                    "n_parallel": controller.props_n_parallel,
                    "n_ctx": controller.props_n_ctx},
                "total_slots": controller.props_n_parallel})
        if path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{
                "id": controller.served_model_id,
                "max_model_len": controller.max_model_len}]})
        if path == "/embed":
            return httpx.Response(200, json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1] * 8}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 0}})
        if path == "/rerank":
            return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}]})
        # chat
        try:
            body = json.loads(request.content.decode() or "{}")
        except Exception:
            body = {}
        return httpx.Response(200, json=_completion_body("echo: " + _last_user_text(body)))

    return handler
