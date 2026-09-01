"""The async client, and a blocking wrapper over it.

``AsyncRoadsteadClient`` is the real implementation: one ``httpx.AsyncClient``,
native SSE, no threads. ``RoadsteadClient`` runs it on a private event loop in a
worker thread so a script or a sync codebase can use the same API.

🚨 **The blocking wrapper owns its own loop and its own thread**, and never
touches a loop the caller already has. A "sync" wrapper implemented as
``asyncio.run`` on a shared client is the classic way to make a library that
works in a REPL and deadlocks in an async application; if you are already async,
use ``AsyncRoadsteadClient`` directly and the wrapper will tell you so rather
than hanging.
"""

from __future__ import annotations

import asyncio
import json
import threading
import weakref
from typing import Any, AsyncIterator, Iterator

import httpx

from ._errors import AuthError, RoadsteadError, UnroutableError
from ._models import (
    Attribution,
    ChatResult,
    Enrichment,
    ModelInfo,
    Plan,
    Timing,
    Usage,
    _headers_get,
)
from ._wire import (
    CLIENT_KEEPALIVE_EXPIRY_S,
    HEADER_DEADLINE_S,
    HEADER_DEADLINE_SOURCE,
    HEADER_ENDPOINT,
    HEADER_REQUEST_ID,
    ROUTE_CHAT,
    ROUTE_MODELS,
    ROUTE_PLAN,
)

#: Matches the server's own default. A caller with no opinion should not have to
#: have one; a caller with an opinion sends ``deadline_s`` and Roadstead honours
#: it. This bounds the HTTP read, not the model — see ``chat``.
_DEFAULT_HTTP_TIMEOUT_S = 900.0

#: How long the blocking wrapper waits for an abandoned stream to unwind on the
#: worker loop. Bounded rather than infinite because this runs from a finalizer:
#: the caller is not asking for the result and may not even be on their own
#: thread, so a wedged unwind must not become a hang in unrelated code.
_STREAM_UNWIND_TIMEOUT_S = 5.0


def enrichment_from(headers: Any) -> Enrichment:
    """Read the ``X-Roadstead-*`` enrichment off an OpenAI-door response.

    For a caller that is not ready to move off ``/v1/chat/completions``. Accepts
    ``response.headers`` or a plain dict, and returns an empty
    :class:`Enrichment` (``present`` False) for a response that carries none —
    so pointing the same code at a stock OpenAI server degrades instead of
    raising.
    """
    raw = _headers_get(headers, HEADER_DEADLINE_S)
    try:
        deadline = float(raw) if raw else 0.0
    except ValueError:
        deadline = 0.0
    return Enrichment(
        request_id=_headers_get(headers, HEADER_REQUEST_ID),
        endpoint=_headers_get(headers, HEADER_ENDPOINT),
        deadline_s=deadline,
        deadline_source=_headers_get(headers, HEADER_DEADLINE_SOURCE),
    )


def _raise_for_envelope(resp: httpx.Response) -> dict:
    """Turn a non-2xx enriched envelope into the right typed exception."""
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    if resp.status_code < 400:
        return body

    code = str(body.get("code") or "")
    message = str(body.get("error") or body.get("message")
                  or f"HTTP {resp.status_code}")
    request_id = str(body.get("request_id") or "")
    kwargs = dict(code=code, status=resp.status_code,
                  request_id=request_id, body=body)
    if resp.status_code in (401, 403):
        raise AuthError(message, **kwargs)
    if code == "unknown_endpoint":
        raise UnroutableError(message, **kwargs)
    raise RoadsteadError(message, **kwargs)


def _chat_body(
    *,
    messages: list | None,
    intent: str,
    model: str,
    requires: list[str] | tuple[str, ...] | None,
    exclude: list[str] | tuple[str, ...] | None,
    kind: str,
    min_context: int,
    prefer: str,
    priority: str | int | None,
    interactive: bool | None,
    deadline_s: float | None,
    allow_degrade: bool | None,
    allow_spill: bool | None,
    call_site: str,
    session_id: str | None,
    turn_id: str | None,
    stream: bool,
    payload: dict | None,
    extra_payload: dict | None,
) -> dict:
    """Assemble the enriched envelope.

    The split is the contract: routing declarations outside, the model request
    inside ``payload``. That is what lets ``deadline_s`` exist at all without
    being forwarded to a backend that would reject the unknown field.
    """
    inner: dict = dict(payload or {})
    if messages is not None:
        inner["messages"] = messages
    if extra_payload:
        inner.update(extra_payload)
    if stream:
        inner["stream"] = True

    body: dict = {"payload": inner}
    if intent:
        body["intent"] = intent
    if model:
        body["model"] = model
    if requires:
        body["requires"] = list(requires)
    if exclude:
        # A NEGATIVE constraint, and a constraint rather than a hint: an
        # entry naming no endpoint this fleet serves is refused by the
        # server (404), because "not here" and "here, misspelled" are the
        # same bytes from its side. docs/api.md §1.7.1.
        body["exclude"] = list(exclude)
    if kind:
        body["kind"] = kind
    if min_context:
        body["min_context"] = int(min_context)
    if prefer:
        body["prefer"] = prefer
    if priority is not None:
        body["priority"] = priority
    if interactive is not None:
        body["interactive"] = bool(interactive)
    if deadline_s is not None:
        body["deadline_s"] = float(deadline_s)
    if allow_degrade is not None or allow_spill is not None:
        # 🚨 NARROWING only. Sending `true` here grants nothing the operator has
        # not already granted the identity — the server takes the AND. Sending
        # `false` is the useful direction: one confidential prompt on an
        # identity that is otherwise happy to spill.
        sub: dict = {}
        if allow_degrade is not None:
            sub["degrade"] = bool(allow_degrade)
        if allow_spill is not None:
            sub["spill"] = bool(allow_spill)
        body["substitution"] = sub
    if call_site:
        body["call_site"] = call_site
    if session_id:
        body["session_id"] = session_id
    if turn_id:
        body["turn_id"] = turn_id
    return body


class AsyncRoadsteadClient:
    """Talk to Roadstead's enriched API.

    ::

        async with AsyncRoadsteadClient("http://proxy:42100", api_key=...) as c:
            plan = await c.plan(intent="reasoning", est_in=8_000)
            result = await c.chat(intent="reasoning",
                                  messages=[{"role": "user", "content": "hi"}])
            print(result.content, result.attribution.endpoint)
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_s: float = _DEFAULT_HTTP_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            # 🚨 docs/api.md §1.3: the client must retire idle sockets FIRST.
            # The server's idle timeout is 30s; whichever side closes second can
            # hand the other a socket it has already closed, which surfaces as a
            # transport error for a request that was never attempted. This is
            # the side a client controls, so the SDK sets it rather than leaving
            # the invariant to whoever configures the pool.
            limits=httpx.Limits(keepalive_expiry=CLIENT_KEEPALIVE_EXPIRY_S),
        )

    # ---- plumbing ----

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            return {}
        # Bearer rather than X-API-Key: it is what an OpenAI client already
        # sends, so one credential works on both doors.
        return {"Authorization": f"Bearer {self._api_key}"}

    async def __aenter__(self) -> "AsyncRoadsteadClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ---- routes ----

    async def models(self) -> list[ModelInfo]:
        """Every endpoint: what it is, what it can do, what it is like now."""
        resp = await self._client.get(ROUTE_MODELS, headers=self._headers())
        body = _raise_for_envelope(resp)
        return [ModelInfo(row) for row in body.get("models") or ()]

    async def intents(self) -> list[dict]:
        """The intent vocabulary this deployment accepts.

        Published rather than documented, and worth calling rather than
        hard-coding: the table is built into the proxy today and is expected to
        become per-deployment, so a client that read our list off the source
        breaks on the day it does.
        """
        resp = await self._client.get(ROUTE_MODELS, headers=self._headers())
        return list(_raise_for_envelope(resp).get("intents") or ())

    async def plan(
        self,
        *,
        intent: str = "",
        model: str = "",
        requires: list[str] | tuple[str, ...] | None = None,
        exclude: list[str] | tuple[str, ...] | None = None,
        kind: str = "",
        min_context: int = 0,
        prefer: str = "",
        priority: str | int | None = None,
        est_in: int = 0,
        est_out: int = 0,
        payload: dict | None = None,
    ) -> Plan:
        """Where would this go, how long should I allow, what would it cost.

        Runs the same resolver the dispatching route runs, over the same facts —
        so a plan and the call that follows it agree, provided the fleet did not
        move in between.
        """
        body = _chat_body(
            messages=None, intent=intent, model=model, requires=requires,
            exclude=exclude, kind=kind, min_context=min_context, prefer=prefer,
            priority=priority, interactive=None, deadline_s=None,
            allow_degrade=None, allow_spill=None, call_site="",
            session_id=None, turn_id=None, stream=False, payload=payload,
            extra_payload=None)
        if est_in:
            body["est_in"] = int(est_in)
        if est_out:
            body["est_out"] = int(est_out)
        resp = await self._client.post(ROUTE_PLAN, json=body,
                                       headers=self._headers())
        return Plan(_raise_for_envelope(resp))

    async def chat(
        self,
        *,
        messages: list | None = None,
        intent: str = "",
        model: str = "",
        requires: list[str] | tuple[str, ...] | None = None,
        exclude: list[str] | tuple[str, ...] | None = None,
        kind: str = "",
        min_context: int = 0,
        prefer: str = "",
        priority: str | int | None = None,
        interactive: bool | None = None,
        deadline_s: float | None = None,
        allow_degrade: bool | None = None,
        allow_spill: bool | None = None,
        call_site: str = "",
        session_id: str | None = None,
        turn_id: str | None = None,
        payload: dict | None = None,
        **extra_payload: Any,
    ) -> ChatResult:
        """One enriched, non-streaming call.

        Declare an ``intent`` (Roadstead picks the model) or a ``model`` (a pin,
        treated as a constraint on routing). Everything a model understands —
        ``max_tokens``, ``tools``, ``response_format`` — goes through
        ``**extra_payload`` or ``payload``.

        🚨 ``deadline_s`` is optional and usually should be omitted: Roadstead
        computes one from the learned latency distribution for this endpoint,
        tier and size, which is a better number than a caller's guess. Supply
        one only when you have a real external constraint — and know that doing
        so turns the deadline into a hard wall, where a computed one is a budget
        the streaming path may extend while tokens are still arriving.
        """
        body = _chat_body(
            messages=messages, intent=intent, model=model, requires=requires,
            exclude=exclude, kind=kind, min_context=min_context, prefer=prefer,
            priority=priority, interactive=interactive, deadline_s=deadline_s,
            allow_degrade=allow_degrade, allow_spill=allow_spill,
            call_site=call_site, session_id=session_id, turn_id=turn_id,
            stream=False, payload=payload, extra_payload=extra_payload)
        resp = await self._client.post(ROUTE_CHAT, json=body,
                                       headers=self._headers())
        return ChatResult(_raise_for_envelope(resp))

    async def stream(
        self,
        *,
        messages: list | None = None,
        intent: str = "",
        model: str = "",
        requires: list[str] | tuple[str, ...] | None = None,
        exclude: list[str] | tuple[str, ...] | None = None,
        kind: str = "",
        min_context: int = 0,
        prefer: str = "",
        priority: str | int | None = None,
        interactive: bool | None = None,
        deadline_s: float | None = None,
        allow_degrade: bool | None = None,
        allow_spill: bool | None = None,
        call_site: str = "",
        session_id: str | None = None,
        turn_id: str | None = None,
        payload: dict | None = None,
        **extra_payload: Any,
    ) -> AsyncIterator[dict]:
        """Stream one enriched call. Yields decoded frames.

        Frame types, in order:

        ``accepted``  once, before anything else — carries ``attribution`` for
                      the endpoint admission chose. The only place a streaming
                      caller can learn that, since the headers are already sent.
        ``admitted``  a scheduling marker; safe to ignore.
        ``chunk``     ``frame["data"]`` is the backend's raw OpenAI
                      ``chat.completion.chunk`` JSON, byte-identical.
        ``done``      once, last — ``attribution``, ``timing``, ``usage``.
        ``error``     terminal instead of ``done``; raises.

        🚨 The ``done`` frame's attribution is the authoritative one. Failover
        and spill both move a request *after* ``accepted`` is on the wire, so a
        caller that trusted the opening frame would be told the endpoint we
        intended rather than the one that answered.
        """
        body = _chat_body(
            messages=messages, intent=intent, model=model, requires=requires,
            exclude=exclude, kind=kind, min_context=min_context, prefer=prefer,
            priority=priority, interactive=interactive, deadline_s=deadline_s,
            allow_degrade=allow_degrade, allow_spill=allow_spill,
            call_site=call_site, session_id=session_id, turn_id=turn_id,
            stream=True, payload=payload, extra_payload=extra_payload)
        async with self._client.stream(
                "POST", ROUTE_CHAT, json=body, headers=self._headers()) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                _raise_for_envelope(resp)
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                raw = line[len("data: "):]
                if raw == "[DONE]":
                    # The enriched wire does not emit it; tolerated so that a
                    # client pointed at the OpenAI door by mistake ends cleanly
                    # rather than hanging on a stream that will never say `done`.
                    break
                try:
                    frame = json.loads(raw)
                except ValueError:
                    continue
                if frame.get("type") == "error":
                    raise RoadsteadError(
                        str(frame.get("error") or "stream error"),
                        code=str(frame.get("code") or ""),
                        request_id=str(frame.get("request_id") or ""),
                        body=frame)
                yield frame
                if frame.get("type") == "done":
                    break

    async def text_stream(self, **kwargs: Any) -> AsyncIterator[str]:
        """``stream``, reduced to the content deltas. For the common case."""
        async for frame in self.stream(**kwargs):
            if frame.get("type") != "chunk":
                continue
            try:
                chunk = json.loads(frame["data"])
                delta = chunk["choices"][0]["delta"].get("content")
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if delta:
                yield delta


class RoadsteadClient:
    """Blocking wrapper over :class:`AsyncRoadsteadClient`.

    Owns a private event loop on a worker thread, so it composes with sync code
    that has no loop of its own and cannot deadlock one that does. If you are
    already async, use the async class — this one refuses to run inside a
    running loop rather than hanging in a way that is very hard to diagnose.
    """

    def __init__(self, base_url: str, *, api_key: str | None = None,
                 timeout_s: float = _DEFAULT_HTTP_TIMEOUT_S) -> None:
        # 🚨 Refuse BEFORE building the coroutine. `_make_async(...)` evaluates
        # first if the guard lives only inside `_run`, so the refusal a caller
        # inside a loop is supposed to get arrives trailing a
        # `RuntimeWarning: coroutine '_make_async' was never awaited` — noise on
        # the one path whose whole job is to say something clear.
        self._refuse_inside_a_running_loop()
        self._closed = False
        # Async generators handed out by `stream`, so `close` can unwind them on
        # the loop that owns them. Weak, because the ordinary end of a stream is
        # the caller dropping it and this must not be what keeps it alive.
        self._live_streams: "weakref.WeakSet" = weakref.WeakSet()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="roadstead-client",
            daemon=True)
        self._thread.start()
        self._async = self._run(_make_async(base_url, api_key, timeout_s))

    @staticmethod
    def _refuse_inside_a_running_loop() -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError(
            "RoadsteadClient is the blocking client and was called from a "
            "running event loop — use AsyncRoadsteadClient instead")

    def _run(self, coro):
        self._refuse_inside_a_running_loop()
        # 🚨 A closed client must REFUSE, not block. `close` stops the worker
        # loop and joins its thread; a coroutine submitted afterwards is
        # accepted by `run_coroutine_threadsafe` and then never runs, so
        # `.result()` waits on a future nothing will ever resolve — an
        # unkillable hang, in the wrapper whose module docstring promises it
        # "will tell you so rather than hanging". Use-after-close is a caller
        # mistake and deserves a sentence, which is the same argument §1.5
        # makes for refusing an unresolvable key instead of falling back.
        if self._closed:
            coro.close()
            raise RuntimeError(
                "RoadsteadClient is closed — build a new one; a closed client "
                "has no event loop to run on")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _unwind_stream(self, agen) -> None:
        """Close one async generator ON THE WORKER LOOP that owns it.

        🚨 Breaking out of ``for frame in client.stream(...)`` is the ordinary
        way to use a stream, not an edge case, and without this it leaks the
        connection. The async generator is left for CPython to finalize from the
        GC, on whatever thread collects it, with no running loop: anyio raises
        ``NoEventLoopError`` inside ``Exception ignored in: <async_generator>``,
        so the ``async with self._client.stream(...)`` body never unwinds and the
        response is never closed. Nobody reads "Exception ignored", and the leak
        surfaces later as a pool that has stopped handing out connections.

        Errors are swallowed because this runs from a generator finalizer, where
        raising produces exactly the unreadable "Exception ignored" this exists
        to remove. It is cleanup on a request the caller has already walked away
        from — there is no result to be wrong about.
        """
        self._live_streams.discard(agen)
        if self._closed or not self._loop.is_running():
            return
        try:
            asyncio.run_coroutine_threadsafe(agen.aclose(), self._loop).result(
                timeout=_STREAM_UNWIND_TIMEOUT_S)
        except BaseException:
            pass

    def __enter__(self) -> "RoadsteadClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return          # idempotent: `with` plus an explicit close is normal
        try:
            # Streams the caller still holds go first, while the loop that owns
            # them is alive. After the join there is nowhere left to unwind them.
            for agen in list(self._live_streams):
                self._unwind_stream(agen)
            self._run(self._async.aclose())
        finally:
            self._closed = True
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5.0)

    def models(self) -> list[ModelInfo]:
        return self._run(self._async.models())

    def intents(self) -> list[dict]:
        return self._run(self._async.intents())

    def plan(self, **kwargs: Any) -> Plan:
        return self._run(self._async.plan(**kwargs))

    def chat(self, **kwargs: Any) -> ChatResult:
        return self._run(self._async.chat(**kwargs))

    def stream(self, **kwargs: Any) -> Iterator[dict]:
        """Blocking iteration over the enriched stream frames.

        Abandoning it partway — a ``break``, an exception, or simply dropping
        it — is safe and releases the connection; see ``_unwind_stream``.
        """
        yield from self._pump(self._async.stream(**kwargs))

    def text_stream(self, **kwargs: Any) -> Iterator[str]:
        yield from self._pump(self._async.text_stream(**kwargs))

    def _pump(self, agen):
        """Drive an async generator one item at a time across the worker loop.

        One item at a time rather than collected first: a streaming API that
        buffers the whole response before yielding anything is a non-streaming
        API with extra steps. The ``finally`` is what makes abandoning it safe —
        see ``_unwind_stream``.
        """
        self._live_streams.add(agen)
        try:
            while True:
                try:
                    yield self._run(agen.__anext__())
                except StopAsyncIteration:
                    return
        finally:
            self._unwind_stream(agen)


async def _make_async(base_url: str, api_key: str | None,
                      timeout_s: float) -> AsyncRoadsteadClient:
    """Construct the async client ON the worker loop.

    httpx binds its connection pool to the loop that created it, so building the
    client on the caller's thread and using it on the worker's is a subtle
    cross-loop bug that only shows up under concurrency.
    """
    return AsyncRoadsteadClient(base_url, api_key=api_key, timeout_s=timeout_s)


__all__ = [
    "AsyncRoadsteadClient",
    "RoadsteadClient",
    "enrichment_from",
    "Attribution",
    "ChatResult",
    "Enrichment",
    "ModelInfo",
    "Plan",
    "Timing",
    "Usage",
]
