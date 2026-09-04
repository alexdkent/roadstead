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
    CallResult,
    ChatResult,
    Enrichment,
    Identity,
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
    PAYLOAD_CHAT,
    PAYLOAD_EMBEDDING,
    PAYLOAD_RERANK,
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


def _envelope(
    *,
    messages: list | None = None,
    payload_type: str = "",
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
    caller_id: str | None = None,
    request_id: str | None = None,
    agent_id: str | None = None,
    est_in: int = 0,
    est_out: int = 0,
    stream: bool = False,
    payload: dict | None = None,
    extra_payload: dict | None = None,
) -> dict:
    """Assemble the enriched envelope. **The one place a field is put on the wire.**

    The split is the contract: routing declarations outside, the model request
    inside ``payload``. That is what lets ``deadline_s`` exist at all without
    being forwarded to a backend that would reject the unknown field.

    🚨 Every dispatching method funnels through here rather than building its
    own body, and ``tests/test_enriched_envelope_coverage.py`` reads this
    function's emitted keys against the fields ``enriched.py`` actually consumes.
    A second assembly site would be a second place for a field to go missing —
    which is exactly how ``payload_type``, ``caller_id`` and ``request_id`` were
    read by the server and sent by nobody for the whole of this SDK's life.
    """
    inner: dict = dict(payload or {})
    if messages is not None:
        inner["messages"] = messages
    if extra_payload:
        inner.update(extra_payload)
    if stream:
        inner["stream"] = True

    body: dict = {"payload": inner}
    if payload_type:
        # 🚨 What SHAPE the payload is — `chat_completion` | `embedding` |
        # `rerank`. Not the same question as `kind`, which is a ROUTING
        # declaration about the endpoint. Both are needed: `kind` picks an
        # embedder, `payload_type` decides that the request goes to its `/embed`
        # route rather than `/v1/chat/completions`. Omitted rather than
        # defaulted here so `plan`, which never dispatches, sends no shape at all.
        body["payload_type"] = payload_type
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
    if caller_id:
        # The sub-identity a call is attributed to, INSIDE the authenticated
        # `agent_id` — the server defaults it to the principal. Dropping it is
        # silent and only shows up as flat `/v1/fleet/top-callers` and
        # `/v1/fleet/cache-attribution` readouts, which is a missing measurement
        # rather than a missing answer, and so is never reported by anything.
        body["caller_id"] = caller_id
    if request_id:
        # The CALLER's own correlation id, carried through the durable record so
        # a caller's log line and the proxy's completion row can be joined.
        body["request_id"] = request_id
    if agent_id:
        # 🚨 ACT AS this caller — a delegation, not a claim. It is honoured only
        # where the credential's own `may_assert` grants the name, refused 403
        # otherwise, and ignored by a key with no grant at all (docs/api.md
        # §1.5 rule 3). Sending it never widens anything: the operator writes
        # the list, this only picks from it.
        #
        # 🚨 NOT the same question as `caller_id`. This changes whose DRR
        # balance, quota and spend cap the call spends; `caller_id` changes only
        # how it is labelled in the analytics. Reach for this one when the two
        # kinds of work should not drain one balance — §1.7.2's table.
        body["agent_id"] = agent_id
    # `/rs/v1/plan` only — it prices a call that has not been written yet, so the
    # sizes are declared rather than measured off a payload. Assembled here
    # rather than bolted on by `plan` afterwards so that this function really is
    # the ONE place a field goes on the wire, which is what the coverage guard
    # calls it to find out.
    if est_in:
        body["est_in"] = int(est_in)
    if est_out:
        body["est_out"] = int(est_out)
    return body


#: Envelope fields ``embed``/``rerank``/``call`` will not accept through their
#: ``**routing`` catch-all. Each is the METHOD's own business: the payload is
#: built from the arguments those methods take, and ``payload_type`` is the whole
#: point of having a typed method rather than a `chat(payload_type=...)` call.
#: Refused loudly rather than dropped — a silently ignored ``stream=True`` would
#: hand the caller a non-streaming answer and no reason.
_METHOD_OWNED = ("payload_type", "payload", "messages", "stream",
                 "extra_payload")

#: What ``docs/api.md`` §1.7.1 requires: a request must declare at least one of
#: these, or the server refuses it.
_DECLARATIONS = ("intent", "model", "requires", "exclude")


def _routing(kwargs: dict, *, method: str, default_kind: str) -> dict:
    """Validate a typed method's ``**routing`` and supply its two defaults.

    🚨 **``kind`` is defaulted always, ``intent`` only as a fallback**, and the
    asymmetry is the point.

    ``kind`` is not a preference here: ``embed()`` is *for* an embed-kind
    endpoint, and a caller who pins one (``model="bge-m3-embed"``) with no
    ``kind`` resolves against the default ``chat`` and gets
    ``404 no endpoint satisfies model='…', kind='chat'`` — a sentence about the
    fleet for a request that was perfectly clear. So the method states the kind
    its payload type implies, and a caller who passes an explicit ``kind`` keeps
    it.

    ``intent`` is different: it is one of the four declarations §1.7.1 requires,
    and supplying one beside a caller's own pin would be the SDK making a routing
    declaration on their behalf. So it fills in only when the caller declared
    nothing at all — without which ``rs.embed(texts=[…])`` would be a 400 about a
    field the caller never mentioned.
    """
    owned = [k for k in _METHOD_OWNED if k in kwargs]
    if owned:
        raise TypeError(
            f"{method}() builds its own payload — {', '.join(owned)} "
            f"{'is' if len(owned) == 1 else 'are'} not accepted here; pass the "
            f"model request through `payload=` (or use `call()` for a payload "
            f"type this SDK has no typed method for)")
    if not kwargs.get("kind"):
        kwargs["kind"] = default_kind
    if not any(kwargs.get(k) for k in _DECLARATIONS):
        kwargs["intent"] = default_kind
    return kwargs


class AsyncRoadsteadClient:
    """Talk to Roadstead's enriched API.

    ::

        async with AsyncRoadsteadClient("http://proxy:42161", api_key=...) as c:
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

    async def _post(self, body: dict) -> dict:
        """One non-streaming enriched dispatch. ONE route, for every payload
        type — ``/rs/v1/chat`` is the enriched door, not the chat door, and
        rerank has had no route at all since ``/v1/submit`` was removed."""
        resp = await self._client.post(ROUTE_CHAT, json=body,
                                       headers=self._headers())
        return _raise_for_envelope(resp)

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
        # No `payload_type`: this route resolves and prices, it never
        # dispatches, so the shape of the payload is not one of its questions.
        body = _envelope(
            intent=intent, model=model, requires=requires, exclude=exclude,
            kind=kind, min_context=min_context, prefer=prefer,
            priority=priority, payload=payload,
            est_in=est_in, est_out=est_out)
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
        caller_id: str | None = None,
        request_id: str | None = None,
        agent_id: str | None = None,
        payload: dict | None = None,
        **extra_payload: Any,
    ) -> CallResult:
        """One enriched, non-streaming CHAT call.

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
        body = _envelope(
            messages=messages, payload_type=PAYLOAD_CHAT, intent=intent,
            model=model, requires=requires,
            exclude=exclude, kind=kind, min_context=min_context, prefer=prefer,
            priority=priority, interactive=interactive, deadline_s=deadline_s,
            allow_degrade=allow_degrade, allow_spill=allow_spill,
            call_site=call_site, session_id=session_id, turn_id=turn_id,
            caller_id=caller_id, request_id=request_id, agent_id=agent_id,
            stream=False, payload=payload, extra_payload=extra_payload)
        return CallResult(await self._post(body))

    async def embed(
        self,
        *,
        texts: str | list[str],
        payload: dict | None = None,
        **routing: Any,
    ) -> CallResult:
        """Embed one string or a list of them.

        ::

            r = await rs.embed(texts=["a chunk", "another"])
            r.response["dense"]      # and `sparse`, and `colbert`, if the
                                     # backend produces them

        🚨 **This is the lossless embedding path, and ``/v1/embeddings`` is not.**
        The OpenAI door must answer in OpenAI's ``{object, data, usage}``, which
        has nowhere to put a hybrid embedder's sparse and colbert halves, so it
        translates and drops them — deliberately, because ``/v1/*`` is
        OpenAI-compatible and strictly so (``docs/api.md`` §1.1). Here the
        backend's body arrives whole under :attr:`CallResult.response`.

        ``**routing`` takes the same declarations :meth:`chat` does — ``model``,
        ``requires``, ``priority``, ``deadline_s``, ``caller_id`` and the rest.
        ``kind="embed"`` is supplied unless you pass one, and ``intent="embed"``
        only if you declared nothing at all — see :func:`_routing`.
        """
        one = [texts] if isinstance(texts, str) else list(texts)
        body = _envelope(
            payload_type=PAYLOAD_EMBEDDING,
            # BOTH dialects, for the reason `http_handlers.handle_openai_embeddings`
            # gives: the hybrid shim reads `texts` and ignores unknown keys, an
            # OpenAI-shaped embeddings server reads `input` and SIZES ITS REPLY
            # from it. Sending only `texts` to the latter returns one vector for
            # an N-input request — a well-formed list of the wrong length, which
            # is worse than an error. The same normalized list goes in both, so
            # they cannot disagree about content.
            payload={"texts": one, "input": one, **(payload or {})},
            **_routing(routing, method="embed", default_kind="embed"))
        return CallResult(await self._post(body))

    async def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        payload: dict | None = None,
        **routing: Any,
    ) -> CallResult:
        """Score ``documents`` against ``query`` with a cross-encoder.

        ::

            r = await rs.rerank(query="why is the queue deep?",
                                documents=[c.text for c in candidates])
            r.response["results"]    # [{"index": .., "relevance_score": ..}, ..]

        🚨 **Rerank had NO route from this SDK, and none from anywhere, between
        the removal of ``/v1/submit`` and 2026-09-02.** There is deliberately no
        ``/v1/rerank`` door to add: OpenAI has no rerank shape to be compatible
        with, so a new ``/v1/*`` route would be Roadstead's own API wearing
        OpenAI's version number. This is the route.

        ``**routing`` as in :meth:`embed`, with ``kind``/``intent`` defaulting
        to ``rerank``.
        """
        inner: dict = {"query": query, "documents": list(documents)}
        if top_n is not None:
            inner["top_n"] = int(top_n)
        inner.update(payload or {})
        body = _envelope(
            payload_type=PAYLOAD_RERANK, payload=inner,
            **_routing(routing, method="rerank", default_kind="rerank"))
        return CallResult(await self._post(body))

    async def call(
        self,
        *,
        payload_type: str,
        payload: dict,
        **routing: Any,
    ) -> CallResult:
        """One enriched dispatch of an arbitrary payload type. The escape hatch.

        🚨 Here so that a proxy newer than this SDK is usable rather than
        gated — the same argument every typed view makes for keeping ``.raw``.
        ``payload_type`` is sent verbatim and is not checked against
        :data:`PAYLOAD_TYPES`; an unknown one is the server's to refuse, with a
        sentence, rather than this SDK's to pre-empt with a stale literal.

        A declaration is required (``intent``, ``model``, ``requires`` or
        ``exclude``): there is no sensible default intent for a payload type this
        SDK has never heard of, and guessing one would route the call somewhere.

        🚨 Pass ``kind`` too when pinning a non-chat endpoint. Unlike
        :meth:`embed` and :meth:`rerank`, this method supplies no ``kind`` — it
        cannot know one — and the resolver's default is ``chat``, so a pin at an
        embedder alone is refused with ``kind='chat'`` in the message.
        """
        owned = [k for k in _METHOD_OWNED if k in routing]
        if owned:
            raise TypeError(
                f"call() takes `payload_type` and `payload` as named arguments "
                f"— {', '.join(owned)} cannot also be passed through routing")
        return CallResult(await self._post(_envelope(
            payload_type=payload_type, payload=payload, **routing)))

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
        caller_id: str | None = None,
        request_id: str | None = None,
        agent_id: str | None = None,
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
        body = _envelope(
            messages=messages, payload_type=PAYLOAD_CHAT, intent=intent,
            model=model, requires=requires,
            exclude=exclude, kind=kind, min_context=min_context, prefer=prefer,
            priority=priority, interactive=interactive, deadline_s=deadline_s,
            allow_degrade=allow_degrade, allow_spill=allow_spill,
            call_site=call_site, session_id=session_id, turn_id=turn_id,
            caller_id=caller_id, request_id=request_id, agent_id=agent_id,
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

    def chat(self, **kwargs: Any) -> CallResult:
        return self._run(self._async.chat(**kwargs))

    def embed(self, **kwargs: Any) -> CallResult:
        return self._run(self._async.embed(**kwargs))

    def rerank(self, **kwargs: Any) -> CallResult:
        return self._run(self._async.rerank(**kwargs))

    def call(self, **kwargs: Any) -> CallResult:
        return self._run(self._async.call(**kwargs))

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
    "CallResult",
    "ChatResult",
    "Enrichment",
    "Identity",
    "ModelInfo",
    "Plan",
    "Timing",
    "Usage",
]
