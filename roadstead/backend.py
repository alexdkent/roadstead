"""Backend HTTP client for nexus/anvil LLM endpoints.

Maintains persistent httpx.AsyncClient pools per host.  Handles
streaming relay, X-Request-ID injection, and health probing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from .config import EndpointConfig

logger = logging.getLogger(__name__)


def _has_anthropic_image_block(messages: Any) -> bool:
    """True when any message carries an Anthropic-shaped image content block
    (``{"type": "image", "source": {...}}``) that needs OAI translation."""
    if not isinstance(messages, list):
        return False
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    return True
    return False


def _translate_anthropic_image_blocks(messages: Any) -> list:
    """Rewrite Anthropic image blocks → OpenAI ``image_url`` for the backend.

    ``ProxyLLMClient`` forwards vision messages verbatim in Anthropic shape
    (``{"type": "image", "source": {"type": "base64", "media_type", "data"}}``)
    — the same shape the knowledge image-ingest path builds. Neither llama.cpp
    nor vLLM understands that block type; they want
    ``{"type": "image_url", "image_url": {"url": "data:<mt>;base64,<data>"}}``
    and otherwise reject the request with ``400 unsupported content[].type``,
    silently dropping every image upload's description + OCR. Mirrors
    ``framework.nexus_translate._translate_content_block`` (kept local so the
    proxy package stays self-contained). Builds new dicts — never mutates the
    caller's message objects (corpus capture stores ``req.payload``). A block
    already in ``image_url`` shape, or any non-image block, passes through.
    """
    out: list = []
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            out.append(msg)
            continue
        new_content = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                source = block.get("source") or {}
                if source.get("type") == "base64":
                    mt = source.get("media_type", "image/jpeg")
                    url = f"data:{mt};base64,{source.get('data', '')}"
                else:
                    url = source.get("url", "")
                new_content.append({"type": "image_url", "image_url": {"url": url}})
            else:
                new_content.append(block)
        out.append({**msg, "content": new_content})
    return out


def _has_consecutive_role(messages: Any, role: str) -> bool:
    """True if ``messages`` contains two or more ADJACENT entries of ``role`` —
    the shape that trips strict chat templates (Qwen rejects >1 system; Mistral
    rejects consecutive user, requiring strict user/assistant alternation)."""
    prev = False
    for msg in messages or []:
        is_role = isinstance(msg, dict) and msg.get("role") == role
        if is_role and prev:
            return True
        prev = is_role
    return False


def _coalesce_role(messages: Any, role: str) -> list:
    """Collapse each run of ADJACENT ``role`` messages into one, joining their
    string ``content`` with ``"\\n\\n"``. Content-preserving (order kept), and
    since coalescing happens in the message body it never disturbs a byte-stable
    leading system prefix used for prefix-cache reuse. Builds new dicts — never
    mutates the caller's objects. Non-string content (e.g. a multimodal block
    list) starts a new run so vision blocks are never mangled.
    """
    out: list = []
    for msg in messages or []:
        is_role = isinstance(msg, dict) and msg.get("role") == role
        if (
            is_role
            and out
            and out[-1].get("role") == role
            and isinstance(out[-1].get("content"), str)
            and isinstance(msg.get("content"), str)
        ):
            prev = out[-1]
            out[-1] = {**prev, "content": f"{prev['content']}\n\n{msg['content']}"}
        else:
            out.append(msg)
    return out


def _has_consecutive_system_messages(messages: Any) -> bool:
    """True if ``messages`` has adjacent ``role: system`` entries.

    The Qwen3.5-122B (composer) template raises ``System message must be at the
    beginning`` on >1 system message; the dj crew and the inner-loop split a
    stable leading system block from a dynamic one. Back-compat wrapper.
    """
    return _has_consecutive_role(messages, "system")


def _coalesce_system_messages(messages: Any) -> list:
    """Collapse adjacent ``role: system`` runs into one (back-compat wrapper)."""
    return _coalesce_role(messages, "system")


def _normalize_chat_payload(
    payload: dict, vllm: bool = False, model_id: str | None = None,
) -> dict:
    """Make an Anthropic/extra_body-shaped chat payload wire-correct for the
    backend.

    The pre-proxy path built requests with ``call_nexus`` +
    ``openai.OpenAI``: the former inlined a top-level ``system`` field as
    the first ``messages`` entry, the latter merged ``extra_body`` keys
    (e.g. GBNF ``grammar``) into the top-level request body. ``ProxyLLMClient``
    does neither, so without this normalization llama-server silently ignores
    both — structured-output call sites (knowledge extract/dedup/relationships,
    temporal, health-verifier) lose their system prompt AND grammar and fall back to
    free-form output that fails JSON parsing.

    For a vLLM backend (``vllm=True``), a top-level ``grammar`` (llama.cpp's
    field) is silently ignored — vLLM enforces GBNF only via
    ``structured_outputs.grammar``. We move it there so the thinker (vLLM-NVFP4)
    actually enforces grammar instead of emitting free-form output.

    Also for vLLM, when ``model_id`` is given we force ``payload["model"]`` to
    it: vLLM validates the model field and 404s on any other name (a role
    alias, an endpoint class, or a different model the caller asked for and
    that the proxy routed here). llama.cpp ignores the field, so we only touch
    it for vLLM.

    Finally, for vLLM we default ``chat_template_kwargs.enable_thinking`` to
    False. The thinker (Qwen3.6) emits chain-of-thought as PROSE ("Here's a
    thinking process: ...") — not ``<think>`` tags — and no reasoning parser is
    configured, so with thinking on the reasoning leaks into
    ``message.content`` and breaks every structured/section parser downstream
    (song theme JSON, craft scorecards, knowledge extract, etc.). Nothing reads
    the reasoning today, so it is pure pollution. A caller that genuinely wants
    reasoning can set ``chat_template_kwargs.enable_thinking`` itself and we
    leave it untouched. llama.cpp ignores the field, so this is vLLM-only.
    """
    if not isinstance(payload, dict):
        return payload
    needs_model_set = bool(vllm and model_id and payload.get("model") != model_id)
    needs_thinking_default = bool(vllm and not _has_enable_thinking(payload))
    needs_vision_xlate = _has_anthropic_image_block(payload.get("messages"))
    # llama.cpp only: >1 adjacent system message 400s the composer 122B template.
    needs_system_coalesce = (not vllm) and _has_consecutive_system_messages(
        payload.get("messages"))
    # llama.cpp only: consecutive user messages 500 strict templates (Mistral/
    # Ministral require strict user/assistant alternation after one optional
    # system). The inner loop emits [system, user, user] when a history turn has
    # an empty assistant response — fine for Qwen, fatal for Mistral. Coalescing
    # is content-preserving and cache-safe (happens after the leading prefix).
    needs_user_coalesce = (not vllm) and _has_consecutive_role(
        payload.get("messages"), "user")
    if (
        "system" not in payload
        and "extra_body" not in payload
        and not (vllm and "grammar" in payload)
        and not needs_model_set
        and not needs_thinking_default
        and not needs_vision_xlate
        and not needs_system_coalesce
        and not needs_user_coalesce
    ):
        return payload
    p = dict(payload)
    if vllm and model_id:
        p["model"] = model_id
    system = p.pop("system", None)
    if system:
        content = system if isinstance(system, str) else str(system)
        p["messages"] = [{"role": "system", "content": content}, *(p.get("messages") or [])]
    # Anthropic vision blocks → OAI image_url (both backends reject the
    # Anthropic shape with 400 unsupported content[].type). After the system
    # inline so the prepended system message is walked too (it's a no-op there).
    if needs_vision_xlate:
        p["messages"] = _translate_anthropic_image_blocks(p.get("messages"))
    # llama.cpp only: merge adjacent system messages (the composer 122B template
    # rejects >1). Runs AFTER the system-inline above so a top-level `system`
    # prepended in front of a messages list that already starts with a system
    # message is collapsed too. No-op when there is no adjacency.
    if not vllm and _has_consecutive_system_messages(p.get("messages")):
        p["messages"] = _coalesce_system_messages(p.get("messages"))
    # llama.cpp only: merge consecutive user messages so strict alternation
    # templates (Mistral/Ministral) don't 500 on the loop's [system, user, user]
    # shape. Runs after the system coalesce; no-op when users aren't adjacent.
    if not vllm and _has_consecutive_role(p.get("messages"), "user"):
        p["messages"] = _coalesce_role(p.get("messages"), "user")
    extra_body = p.pop("extra_body", None)
    if isinstance(extra_body, dict):
        p.update(extra_body)
    if vllm and isinstance(p.get("grammar"), str) and p["grammar"].strip():
        so = p.get("structured_outputs")
        so = dict(so) if isinstance(so, dict) else {}
        so.setdefault("grammar", p.pop("grammar"))
        p["structured_outputs"] = so
    # Default thinking off for vLLM (checked AFTER the extra_body merge so a
    # caller's chat_template_kwargs nested in extra_body still wins).
    if vllm and not _has_enable_thinking(p):
        ck = p.get("chat_template_kwargs")
        ck = dict(ck) if isinstance(ck, dict) else {}
        ck["enable_thinking"] = False
        p["chat_template_kwargs"] = ck
    return p


def _has_enable_thinking(payload: dict) -> bool:
    """True when the caller has already pinned chat_template_kwargs.enable_thinking
    (top-level or nested in extra_body) — in which case we don't override it."""
    for container in (payload, payload.get("extra_body")):
        if isinstance(container, dict):
            ck = container.get("chat_template_kwargs")
            if isinstance(ck, dict) and "enable_thinking" in ck:
                return True
    return False


def extract_cached_tokens(usage: Any) -> int | None:
    """Pull the per-request prefix-cache hit count out of an OpenAI ``usage``
    block, defensively. Phase 2a attribution.

    vLLM (V1, prefix caching on) emits ``usage.prompt_tokens_details.cached_tokens``
    — the number of prompt tokens served from the KV prefix cache. llama.cpp does
    NOT emit any such field. The distinction matters: a backend that reports ``0``
    is a real *cold* miss (attributable), whereas an *absent* field means the
    backend can't tell us (llama.cpp) — which must persist as NULL so the rollup
    marks it ``n/a`` instead of dragging the hit rate toward zero.

    Returns the int (including 0) when present and numeric, else ``None``.
    Never raises on a malformed/hostile usage shape (south-face safety)."""
    if not isinstance(usage, dict):
        return None
    details = usage.get("prompt_tokens_details")
    candidate: Any = None
    if isinstance(details, dict) and "cached_tokens" in details:
        candidate = details.get("cached_tokens")
    elif "cached_tokens" in usage:  # some shims flatten it to the top level
        candidate = usage.get("cached_tokens")
    else:
        return None
    if isinstance(candidate, bool):  # bool is an int subclass — reject it
        return None
    if isinstance(candidate, (int, float)):
        # NaN/±Infinity reach here over the wire — JSON accepts them by default
        # (both httpx .json() and our SSE-frame json.loads), and int(nan)/int(inf)
        # RAISE (ValueError/OverflowError). An observability field must never
        # convert a good completion into a caller-facing error → reject non-finite.
        if isinstance(candidate, float) and not math.isfinite(candidate):
            return None
        val = int(candidate)
        return val if val >= 0 else None
    return None


def coerce_token_count(candidate: Any, default: int = 0) -> int:
    """Coerce a usage token count (``prompt_tokens``/``completion_tokens``) to a
    safe non-negative int. Same defensive contract as ``extract_cached_tokens``:
    ``.get(k, 0)`` returns present-but-``null`` as ``None`` (not the default),
    and NaN/±Infinity/bool/str over the wire must never reach cost arithmetic,
    MetricsSample, or the INTEGER telemetry columns (NaN poisons SUM rollups)."""
    if isinstance(candidate, bool):
        return default
    if isinstance(candidate, (int, float)):
        if isinstance(candidate, float) and not math.isfinite(candidate):
            return default
        val = int(candidate)
        return val if val >= 0 else default
    return default


@dataclass
class BackendResponse:
    """Result of a backend call (non-streaming)."""
    status_code: int
    body: dict
    duration_s: float
    input_tokens: int
    output_tokens: int
    finish_reason: str | None = None
    # Phase 2a: prompt tokens served from the backend prefix cache, when the
    # backend reports it (vLLM). None == backend didn't say (llama.cpp) → NULL
    # in telemetry so the per-caller rollup marks it n/a, not a 0% hit.
    cached_tokens: int | None = None


@dataclass
class BackendStreamEvent:
    """One SSE event from a streaming backend call."""
    event_type: str   # "chunk" | "done" | "error"
    data: str         # raw SSE data line
    parsed: dict | None = None  # parsed JSON if applicable


class BackendError(Exception):
    """Backend returned a non-2xx response."""
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"backend error {status_code}: {detail}")


class BackendTimeout(BackendError):
    def __init__(self, detail: str = "timeout") -> None:
        super().__init__(504, detail)


class BackendUnavailable(BackendError):
    def __init__(self, detail: str = "unavailable") -> None:
        super().__init__(503, detail)


class BackendClientPool:
    """Manages httpx.AsyncClient instances for backend connections."""

    def __init__(self) -> None:
        # key -> (client, max_connections it was built with)
        self._clients: dict[str, tuple[httpx.AsyncClient, int]] = {}
        # Clients superseded by a larger pool (slot discovery raised an
        # endpoint's concurrency after first use). They stay open so their
        # in-flight requests finish unharmed; closed at shutdown.
        self._retired: list[httpx.AsyncClient] = []

    def _client_for(
        self, host: str, port: int, min_pool: int = 0,
    ) -> httpx.AsyncClient:
        """Connection-pooled client for a backend, sized to its concurrency.

        ``min_pool`` is the caller's concurrency requirement — the dispatch
        path passes ``effective_max_slots + headroom`` so the pool can never
        be smaller than the scheduler's admission ceiling. The historic flat
        20 starved the 32-slot thinker: dispatches 21+ queued on the httpx
        pool (pool=5.0s) and failed as PoolTimeout even though the backend
        had free slots. If a later call needs a BIGGER pool than the cached
        client has (slot discovery raised max_slots), the old client is
        retired (not closed — in-flight requests finish on it) and replaced.
        """
        key = f"{host}:{port}"
        want = max(20, min_pool)
        cur = self._clients.get(key)
        if cur is not None:
            client, built_with = cur
            if built_with >= want:
                return client
            self._retired.append(client)
        client = httpx.AsyncClient(
            base_url=f"http://{host}:{port}",
            timeout=httpx.Timeout(connect=5.0, read=600.0, write=10.0, pool=5.0),
            limits=httpx.Limits(
                max_connections=want,
                max_keepalive_connections=max(10, want // 2),
            ),
        )
        self._clients[key] = (client, want)
        return client

    async def close(self) -> None:
        for client, _size in self._clients.values():
            try:
                await client.aclose()
            except Exception:
                pass
        self._clients.clear()
        for client in self._retired:
            try:
                await client.aclose()
            except Exception:
                pass
        self._retired.clear()

    async def call(
        self,
        ep_cfg: EndpointConfig,
        payload: dict,
        payload_type: str,
        request_id: str,
        timeout_s: float = 180.0,
    ) -> BackendResponse:
        """Make a non-streaming backend call."""
        client = self._client_for(
            ep_cfg.host, ep_cfg.port, ep_cfg.effective_max_slots + 4)
        path = self._path_for(payload_type)
        headers = {"X-Request-ID": request_id}
        if payload_type == "chat_completion":
            payload = _normalize_chat_payload(
                payload, vllm=(ep_cfg.backend_engine == "vllm"),
                model_id=ep_cfg.effective_model_id)

        t0 = time.monotonic()
        try:
            resp = await asyncio.wait_for(
                client.post(path, json=payload, headers=headers),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            raise BackendTimeout(f"backend {ep_cfg.role} timeout after {timeout_s}s")
        except httpx.ConnectError as exc:
            raise BackendUnavailable(f"backend {ep_cfg.role} unreachable: {exc}")
        except httpx.PoolTimeout as exc:
            # Local connection-pool exhaustion, NOT a backend fault. With the
            # slot-sized pool above this should be unreachable; if it ever
            # fires, it's transient by nature → BackendUnavailable so the
            # in-proxy retry + caller deferral engage instead of a hard 502.
            raise BackendUnavailable(
                f"backend {ep_cfg.role} connection pool exhausted: {exc}")
        except httpx.RemoteProtocolError as exc:
            # Backend dropped the connection before responding — a server-closed
            # keep-alive socket reused from the pool, or a crash. Transient by
            # nature → BackendUnavailable so the in-proxy retry engages, parity
            # with the stream() path (must precede the HTTPError catch below —
            # RemoteProtocolError is an httpx.HTTPError subclass).
            raise BackendUnavailable(f"backend {ep_cfg.role} disconnected: {exc}")
        except httpx.HTTPError as exc:
            raise BackendError(502, f"backend {ep_cfg.role} http error: {exc}")

        duration = time.monotonic() - t0

        if resp.status_code >= 400:
            raise BackendError(resp.status_code, resp.text[:500])

        try:
            body = resp.json()
        except (json.JSONDecodeError, ValueError):
            body = {"raw": resp.text[:2000]}

        input_tokens = 0
        output_tokens = 0
        usage = body.get("usage") or {}
        cached_tokens = extract_cached_tokens(usage)
        if usage:
            input_tokens = coerce_token_count(usage.get("prompt_tokens", 0))
            output_tokens = coerce_token_count(usage.get("completion_tokens", 0))

        # Empty-completion gate (fail-loud). A 2xx with no generated content is a
        # silent failure — backend hiccup, grammar over-constraint that masks all
        # tokens, or an immediate EOS. It must NEVER pass as a valid (empty)
        # result: callers (e.g. knowledge.extract_entities) would record "no
        # entities" and silently drop data. Surface it as an error so the caller
        # retries/handles and WS2 monitoring sees it. Content-based (not token-
        # count) so it holds even when a backend omits usage; tool-call responses
        # legitimately have empty content, so they're exempt.
        finish_reason: str | None = None
        if payload_type == "chat_completion":
            choice0 = (body.get("choices") or [{}])[0] or {}
            finish_reason = choice0.get("finish_reason")
            msg = choice0.get("message") or {}
            if not (msg.get("content") or "").strip() and not msg.get("tool_calls"):
                raise BackendError(
                    502,
                    f"backend {ep_cfg.role} returned empty completion "
                    f"(no content, output_tokens={output_tokens})",
                )

        return BackendResponse(
            status_code=resp.status_code,
            body=body,
            duration_s=duration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            cached_tokens=cached_tokens,
        )

    async def stream(
        self,
        ep_cfg: EndpointConfig,
        payload: dict,
        payload_type: str,
        request_id: str,
        timeout_s: float = 180.0,
    ) -> AsyncIterator[BackendStreamEvent]:
        """Make a streaming backend call.  Yields SSE events."""
        client = self._client_for(
            ep_cfg.host, ep_cfg.port, ep_cfg.effective_max_slots + 4)
        path = self._path_for(payload_type)
        headers = {"X-Request-ID": request_id}
        if payload_type == "chat_completion":
            payload = _normalize_chat_payload(
                payload, vllm=(ep_cfg.backend_engine == "vllm"),
                model_id=ep_cfg.effective_model_id)

        try:
            async with client.stream(
                "POST", path, json=payload, headers=headers,
                timeout=httpx.Timeout(connect=5.0, read=timeout_s, write=10.0, pool=5.0),
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise BackendError(response.status_code, body.decode("utf-8", errors="replace")[:500])

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            yield BackendStreamEvent(event_type="done", data=data)
                            break
                        try:
                            parsed = json.loads(data)
                        except json.JSONDecodeError:
                            parsed = None
                        yield BackendStreamEvent(
                            event_type="chunk", data=data, parsed=parsed,
                        )
        except httpx.ConnectError as exc:
            raise BackendUnavailable(f"backend {ep_cfg.role} unreachable: {exc}")
        except httpx.RemoteProtocolError as exc:
            # Backend dropped the connection — a server-closed keep-alive socket
            # reused from the pool, or a crash mid-stream. httpx normally
            # recovers from the keep-alive case transparently, but if it does
            # surface, map it to a clean BackendUnavailable instead of letting
            # the raw httpx error propagate (parity with call()).
            raise BackendUnavailable(f"backend {ep_cfg.role} disconnected mid-stream: {exc}")
        except httpx.PoolTimeout as exc:
            # Local pool exhaustion (see call()) — transient, NOT a stream
            # timeout; must precede the TimeoutException catch below.
            raise BackendUnavailable(
                f"backend {ep_cfg.role} connection pool exhausted: {exc}")
        except (httpx.TimeoutException, asyncio.TimeoutError):
            # stream() bounds reads via httpx.Timeout, which raises
            # httpx.ReadTimeout (a TimeoutException) — not asyncio.TimeoutError.
            raise BackendTimeout(f"backend {ep_cfg.role} stream timeout after {timeout_s}s")
        except httpx.HTTPError as exc:
            raise BackendError(502, f"backend {ep_cfg.role} stream http error: {exc}")

    async def probe_props(self, ep_cfg: EndpointConfig) -> dict | None:
        """Probe backend /props for capacity discovery."""
        client = self._client_for(ep_cfg.host, ep_cfg.port)
        try:
            resp = await asyncio.wait_for(
                client.get("/props"),
                timeout=5.0,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    async def probe_models(self, ep_cfg: EndpointConfig) -> str | None:
        """Probe backend /v1/models for the served model id (the name the
        backend answers to in the `model` field). Returns the first model
        id, or None on any failure."""
        client = self._client_for(ep_cfg.host, ep_cfg.port)
        try:
            resp = await asyncio.wait_for(client.get("/v1/models"), timeout=5.0)
            if resp.status_code == 200:
                data = resp.json().get("data") or []
                if data and isinstance(data[0], dict):
                    model_id = data[0].get("id")
                    if isinstance(model_id, str) and model_id:
                        return model_id
        except Exception:
            pass
        return None

    async def probe_vllm_capacity(self, ep_cfg: EndpointConfig) -> dict | None:
        """Capacity discovery for a vLLM backend. vLLM has no llama.cpp /props
        or /slots; the per-request context ceiling comes from /v1/models
        `max_model_len`. Concurrency (--max-num-seqs) is NOT exposed over the
        API, so max_slots stays config-driven. Returns
        ``{"max_model_len": int}`` or None on any failure."""
        client = self._client_for(ep_cfg.host, ep_cfg.port)
        try:
            resp = await asyncio.wait_for(client.get("/v1/models"), timeout=5.0)
            if resp.status_code == 200:
                data = resp.json().get("data") or []
                if data and isinstance(data[0], dict):
                    mlen = data[0].get("max_model_len")
                    if isinstance(mlen, int) and mlen > 0:
                        return {"max_model_len": mlen}
        except Exception:
            pass
        return None

    async def probe_health(self, ep_cfg: EndpointConfig) -> bool:
        """Simple health check."""
        client = self._client_for(ep_cfg.host, ep_cfg.port)
        try:
            resp = await asyncio.wait_for(
                client.get("/health"),
                timeout=3.0,
            )
            return resp.status_code == 200
        except Exception:
            return False

    async def probe_prefix_cache(self, ep_cfg: EndpointConfig) -> dict | None:
        """Scrape a backend's Prometheus `/metrics` for prefix-cache counters.

        Returns ``{"hits": int, "queries": int}`` (cumulative-since-backend-boot,
        token-weighted) or ``None`` when the backend doesn't expose them — vLLM
        publishes ``vllm:prefix_cache_hits_total`` / ``vllm:prefix_cache_queries_total``;
        llama.cpp has no equivalent, so its endpoints read as actual-rate ``n/a``
        (the cache-ability screen still covers them). Best-effort: any error → None.
        """
        client = self._client_for(ep_cfg.host, ep_cfg.port)
        try:
            resp = await asyncio.wait_for(client.get("/metrics"), timeout=5.0)
            if resp.status_code != 200:
                return None
            hits = queries = None
            for line in resp.text.splitlines():
                if line.startswith("#") or "prefix_cache" not in line:
                    continue
                # "vllm:prefix_cache_hits_total{...} 58112.0"
                try:
                    name, val = line.rsplit(" ", 1)
                    v = int(float(val))
                except ValueError:
                    continue
                if name.startswith("vllm:prefix_cache_hits_total"):
                    hits = v
                elif name.startswith("vllm:prefix_cache_queries_total"):
                    queries = v
            if hits is not None and queries is not None:
                return {"hits": hits, "queries": queries}
        except Exception:
            pass
        return None

    async def call_shadow(
        self,
        shadow_host: str,
        shadow_port: int,
        payload: dict,
        payload_type: str,
        request_id: str,
        timeout_s: float = 300.0,
    ) -> BackendResponse | None:
        """Fire-and-forget shadow call for A/B testing.

        Returns the response on success, None on any failure.
        Never raises — shadow failures must not affect the primary path.
        """
        client = self._client_for(shadow_host, shadow_port)
        path = self._path_for(payload_type)
        headers = {"X-Request-ID": f"shadow-{request_id}"}
        t0 = time.monotonic()
        try:
            resp = await asyncio.wait_for(
                client.post(path, json=payload, headers=headers),
                timeout=timeout_s,
            )
            duration = time.monotonic() - t0
            if resp.status_code >= 400:
                return None
            body = resp.json()
            usage = body.get("usage") or {}
            return BackendResponse(
                status_code=resp.status_code,
                body=body,
                duration_s=duration,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                cached_tokens=extract_cached_tokens(usage),
            )
        except Exception:
            return None

    @staticmethod
    def _path_for(payload_type: str) -> str:
        if payload_type == "embedding":
            return "/embed"
        if payload_type == "rerank":
            return "/rerank"
        return "/v1/chat/completions"
