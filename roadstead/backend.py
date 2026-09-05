"""Backend HTTP client — the transport half of the south face.

Maintains persistent httpx.AsyncClient pools per host. Handles streaming relay,
X-Request-ID injection, deadlines, the error taxonomy, and health probing.

What a particular ENGINE requires of a request, and what it will tell us about
itself, lives in `providers/` — this module is the part that is the same for
every backend. Providers are handed this pool and ask it to probe rather than
opening sockets of their own, which is what keeps connection pooling (and the
unit suite's network isolation, which stubs `probe_*` by name here) in one
place.
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

from .config import EndpointConfig, max_response_bytes_from_env
from .providers import DEFAULT_PROVIDER, ProviderError, provider_for

logger = logging.getLogger(__name__)


def empty_completion_error(role: str, msg: dict, finish_reason: str | None,
                           output_tokens: int) -> "BackendError | None":
    """The empty-completion decision, as a pure function so it can be TESTED.

    Extracted 2026-07-31 rather than left inline: a test that re-implements this
    logic is a copy that drifts, and the whole point of the change is that the
    message must stay precise. See
    `tests/test_empty_completion_names_its_cause.py`.

    Returns the error to raise, or None when the response is fine.
    """
    if (msg.get("content") or "").strip() or msg.get("tool_calls"):
        return None
    reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "")
    if reasoning.strip() and finish_reason == "length":
        return BackendError(
            502,
            f"backend {role} spent its ENTIRE {output_tokens}-token budget on "
            f"REASONING and never reached content ({len(reasoning)} chars of "
            f"reasoning, finish_reason=length). This is a token-budget problem, "
            f"NOT a broken model: raise max_tokens (reasoning is additive to the "
            f"answer), or use the proxy's `thinking: <tokens>` opt-in which adds "
            f"that much reasoning budget for you, or turn reasoning off with the "
            f"chat_template_kwargs key THIS model reads — `thinking` for "
            f"DeepSeek-V4, `enable_thinking` for Qwen (models.yaml "
            f"policy.thinking_kwargs); the other family's key is a silent no-op.")
    return BackendError(
        502,
        f"backend {role} returned empty completion "
        f"(no content, output_tokens={output_tokens}, "
        f"finish_reason={finish_reason!r})")

def extract_cached_tokens(usage: Any) -> int | None:
    """Pull the per-request prefix-cache hit count out of an OpenAI ``usage``
    block, defensively. Phase 2a attribution.

    ``usage.prompt_tokens_details.cached_tokens`` is the number of prompt tokens
    served from the KV prefix cache. The distinction that matters: a backend
    reporting ``0`` is a real *cold* miss (attributable), whereas an *absent*
    field means the backend can't tell us — which must persist as NULL so the
    rollup marks it ``n/a`` instead of dragging the hit rate toward zero.

    🚨 CORRECTED 2026-08-20 — this docstring said "llama.cpp does NOT emit any
    such field". IT DOES, on every build we run. Measured directly against all
    three llama.cpp backends (tier1, tier2-analyst, tier2-chat): each returns ``prompt_tokens_details: {"cached_tokens": N}``, and
    the live ``/v1/fleet/cache-attribution`` rollup attributes them 321/321,
    37/37 and 5/7 respectively. The function was always correct — it keys on the
    SHAPE, not the backend — but the comment would have talked a reader out of
    trusting a real number. It matters because tier2's genuinely near-zero reuse
    (18.3% analyst / 5.2% chat vs tier1's 74.0%) is a MEASURED defect, and
    "llama.cpp can't report it" is exactly the sentence that would file it as an
    artifact. The backend that truly reports nothing here is vLLM, and only
    because ``--enable-prompt-tokens-details`` is unset (see ``health.py``).

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


#: Slack between the transport read deadline and the request's own deadline, so
#: `asyncio.wait_for` is always the layer that fires and the failure is typed as
#: a TIMEOUT (retryable, deferrable) rather than as a transport 502.
_TRANSPORT_READ_MARGIN_S = 30.0


def _transport_timeout(timeout_s: float) -> "httpx.Timeout":
    """Per-request transport deadline for a non-streaming backend call.

    🚨 WHY THIS EXISTS. The pooled client is built with a FLAT ``read=600.0``
    (see ``_client_for``). That constant silently OVERRODE every caller deadline
    above it: ``asyncio.wait_for`` was handed the real ``timeout_s``, but httpx
    gave up reading at 600s first, and a ``ReadTimeout`` is an ``httpx.HTTPError``
    — so the call surfaced as ``BackendError(502)``, not as a timeout. Two
    consequences, both bad: a legitimately long generation could never exceed
    600s no matter what it declared, and the failure was mistyped so the caller
    saw an infrastructure fault instead of a deadline it could act on.

    Measured 2026-08-24 on tier3 (DeepSeek-V4-Flash-0731): a native-reasoning
    song-authoring call — ~15k prompt, 12k max_tokens after the proxy's reasoning
    budget — is dispatched with a ~1230s deadline and died at exactly 600.0s with
    ``status=error``, twice, while the scheduler's own clock still had 10 minutes
    left on it.

    The request's ``timeout_s`` stays the authority: the transport gets a small
    margin ON TOP so `wait_for` fires first. The floor keeps short-deadline calls
    on the historic behaviour (a tiny caller budget must not shorten the read
    below what the pool was built for).
    """
    read = max(600.0, float(timeout_s) + _TRANSPORT_READ_MARGIN_S)
    return httpx.Timeout(connect=5.0, read=read, write=10.0, pool=5.0)


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
        self, base_url: str, min_pool: int = 0,
    ) -> httpx.AsyncClient:
        """Connection-pooled client for a backend, sized to its concurrency.

        ``base_url`` is the full origin (plus any base path) — ``ep_cfg.
        backend_url``, which is ``http://host:port`` for a local engine and a
        real URL for a remote provider. It doubles as the pool key, so two
        endpoints on one origin share connections exactly as they did when the
        key was ``host:port``, and a remote provider on https gets its own.

        ``min_pool`` is the caller's concurrency requirement — the dispatch
        path passes ``effective_max_slots + headroom`` so the pool can never
        be smaller than the scheduler's admission ceiling. The historic flat
        20 starved the 32-slot thinker: dispatches 21+ queued on the httpx
        pool (pool=5.0s) and failed as PoolTimeout even though the backend
        had free slots. If a later call needs a BIGGER pool than the cached
        client has (slot discovery raised max_slots), the old client is
        retired (not closed — in-flight requests finish on it) and replaced.
        """
        key = base_url
        want = max(20, min_pool)
        cur = self._clients.get(key)
        if cur is not None:
            client, built_with = cur
            if built_with >= want:
                return client
            self._retired.append(client)
        client = httpx.AsyncClient(
            base_url=base_url,
            # Default deadline for callers that pass none. The NON-STREAMING
            # path overrides this per request (`_transport_timeout`) so a
            # declared deadline above 600s is honoured instead of being cut here
            # and mistyped as a 502; the streaming path keeps this as an
            # inter-chunk read gap, where 600s is generous.
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
            ep_cfg.backend_url, ep_cfg.effective_max_slots + 4)
        provider = provider_for(ep_cfg)
        # 🚨 A provider that cannot serve this request says so HERE, before a
        # socket is used, and the refusal is typed as a 400 rather than dressed
        # up as a backend fault: the backend is fine, the pairing is wrong. See
        # providers/base.ProviderError — a dropped constraint would be
        # indistinguishable, to the caller, from a model that answered badly.
        try:
            path = provider.path_for(payload_type)
            headers = provider.request_headers(ep_cfg, request_id)
            if payload_type == "chat_completion":
                payload = provider.prepare_chat_payload(
                    payload,
                    model_id=ep_cfg.effective_model_id,
                    thinking_budget_ratio=ep_cfg.thinking_budget_ratio,
                    thinking_kwargs=ep_cfg.thinking_kwargs)
        except ProviderError as exc:
            raise BackendError(
                400, f"backend {ep_cfg.role} ({provider.name}): {exc}")

        t0 = time.monotonic()
        try:
            resp = await asyncio.wait_for(
                client.post(path, json=payload, headers=headers,
                            timeout=_transport_timeout(timeout_s)),
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
            err = empty_completion_error(
                ep_cfg.role, msg, finish_reason, output_tokens)
            if err is not None:
                raise err

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
            ep_cfg.backend_url, ep_cfg.effective_max_slots + 4)
        provider = provider_for(ep_cfg)
        # 🚨 A provider that cannot serve this request says so HERE, before a
        # socket is used, and the refusal is typed as a 400 rather than dressed
        # up as a backend fault: the backend is fine, the pairing is wrong. See
        # providers/base.ProviderError — a dropped constraint would be
        # indistinguishable, to the caller, from a model that answered badly.
        try:
            path = provider.path_for(payload_type)
            headers = provider.request_headers(ep_cfg, request_id)
            if payload_type == "chat_completion":
                payload = provider.prepare_chat_payload(
                    payload,
                    model_id=ep_cfg.effective_model_id,
                    thinking_budget_ratio=ep_cfg.thinking_budget_ratio,
                    thinking_kwargs=ep_cfg.thinking_kwargs)
        except ProviderError as exc:
            raise BackendError(
                400, f"backend {ep_cfg.role} ({provider.name}): {exc}")

        try:
            async with client.stream(
                "POST", path, json=payload, headers=headers,
                timeout=httpx.Timeout(connect=5.0, read=timeout_s, write=10.0, pool=5.0),
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise BackendError(response.status_code, body.decode("utf-8", errors="replace")[:500])

                # 🚨 A running cap on the streamed response, checked BEFORE
                # each line is buffered rather than after. `aiter_lines()`
                # already holds the whole line in memory before yielding it,
                # so a wedged or adversarial backend that never stops talking
                # (no `[DONE]`, no chunk boundary) can otherwise grow the
                # proxy's own memory without bound — the streaming counterpart
                # to the request-body cap in `__main__.RequestSizeLimitMiddleware`.
                # Not classified transient (`Correction.is_transient_backend_error`
                # only special-cases `BackendUnavailable` and an "empty
                # completion" detail): the same request would very likely
                # reproduce the same oversized response, so the proxy's own
                # defer/retry loop must not spend a slot retrying it.
                max_response_bytes = max_response_bytes_from_env()
                seen_bytes = 0
                async for line in response.aiter_lines():
                    seen_bytes += len(line.encode("utf-8", errors="ignore")) + 1
                    if seen_bytes > max_response_bytes:
                        raise BackendError(
                            502,
                            f"backend {ep_cfg.role} streamed response exceeded "
                            f"the {max_response_bytes}-byte cap "
                            f"(ROADSTEAD_MAX_RESPONSE_BYTES) — aborted rather "
                            f"than buffered without bound")
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
        client = self._client_for(ep_cfg.backend_url)
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
        client = self._client_for(ep_cfg.backend_url)
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
        client = self._client_for(ep_cfg.backend_url)
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

    async def probe_model_fingerprint(self, ep_cfg: EndpointConfig) -> str | None:
        """Probe /v1/models for a string that IDENTIFIES THE WEIGHTS, not the
        alias the backend answers to.

        🚨 WHY NOT `served_model_id`. That is what `probe_models` returns, and it
        is USELESS for detecting a model swap: tier3 serves
        `--served-model-name llama-thinker`, so its `id` stayed `llama-thinker`
        straight through the 2026-08-23 Qwen3.6 -> DeepSeek-V4-Flash cutover. An
        alert keyed on it could never have fired. Both engines do expose the real
        thing, in different places (measured on the live backends 2026-08-24):

          vLLM       data[0].root      -> "/srv/models/deepseek-v4-flash-0731"
          llama.cpp  data[0].meta      -> {n_params, n_vocab, ftype, ...}

        For llama.cpp we fold the meta into a short stable string. `n_params` is
        the field that actually carries the signal — the boxa's 2026-08-19 swap
        went 35B-A3B MoE -> dense 27B, i.e. 34,660,610,688 -> 27,320,697,856.
        `ftype` is absent on some builds (tier1 has no such key), so every field
        is optional and only the ones present are folded in; a fingerprint is
        compared for EQUALITY against a declaration, never parsed.

        Returns the fingerprint string, or None on any failure (a missing
        fingerprint must read as "cannot tell", never as "changed" — see the
        drift alert, which stays silent while this is None)."""
        client = self._client_for(ep_cfg.backend_url)
        try:
            resp = await asyncio.wait_for(client.get("/v1/models"), timeout=5.0)
            if resp.status_code != 200:
                return None
            data = resp.json().get("data") or []
            if not data or not isinstance(data[0], dict):
                return None
            entry = data[0]
            root = entry.get("root")
            if isinstance(root, str) and root.strip():
                return root.strip()
            meta = entry.get("meta")
            if isinstance(meta, dict):
                parts = []
                for field, label in (("n_params", "params"),
                                     ("n_vocab", "vocab"),
                                     ("ftype", "ftype")):
                    val = meta.get(field)
                    if isinstance(val, bool):
                        continue
                    if isinstance(val, (int, float)) and val > 0:
                        parts.append(f"{label}={int(val)}")
                    elif isinstance(val, str) and val.strip():
                        parts.append(f"{label}={val.strip()}")
                if parts:
                    return ";".join(parts)
        except Exception:
            pass
        return None

    async def probe_thinking_switch(
        self, ep_cfg: EndpointConfig, key: str, nonce: str,
    ) -> dict | None:
        """Send ONE real chat call and report whether `key` actually switched
        reasoning on. The canary behind the `thinking_switch_broken` alert.

        🚨 THIS IS THE POINT: `policy.thinking_kwargs` is a DECLARATION, and a
        declaration cannot notice that the model underneath it changed. That is
        exactly how the 2026-08-23 tier3 swap went unnoticed for a day — the
        proxy asserted a switch nobody had ever made one real call to verify.
        The house rule is that a green suite does not mean a capability works;
        one real call at the size you will actually send does.

        Deliberately NOT routed through `call()`:
          * it must not consume a DRR slot or queue behind live traffic — this
            follows the same direct-client pattern as every other probe here;
          * `call()` raises `empty_completion_error` when a response has no
            content, and an ON-arm probe that spends its small budget entirely
            on reasoning is a PASS for our purposes, not a backend fault.

        `nonce` is folded into the prompt because a fixed probe payload would be
        answered by the prefix cache rather than the model, and a cached answer
        proves nothing about today's template. Returns
        ``{"reasoning_chars": int, "content_chars": int}`` or None if the call
        did not complete (unreachable/busy → we say nothing, never "broken")."""
        client = self._client_for(ep_cfg.backend_url)
        payload = {
            "model": ep_cfg.effective_model_id,
            "messages": [{"role": "user", "content":
                          f"In one short sentence, why is the sky blue? (ref {nonce})"}],
            "max_tokens": 200,
            "temperature": 0.0,
            "chat_template_kwargs": {key: True},
        }
        try:
            resp = await asyncio.wait_for(
                client.post("/v1/chat/completions", json=payload), timeout=60.0)
            if resp.status_code != 200:
                return None
            choices = resp.json().get("choices") or []
            if not choices or not isinstance(choices[0], dict):
                return None
            msg = choices[0].get("message") or {}
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
            content = msg.get("content") or ""
            return {"reasoning_chars": len(reasoning),
                    "content_chars": len(content)}
        except Exception:
            return None

    async def probe_health(self, ep_cfg: EndpointConfig) -> bool:
        """Simple health check."""
        client = self._client_for(ep_cfg.backend_url)
        try:
            resp = await asyncio.wait_for(
                client.get("/health"),
                timeout=3.0,
            )
            return resp.status_code == 200
        except Exception:
            return False

    async def probe_json(
        self, ep_cfg: EndpointConfig, path: str,
        headers: dict[str, str] | None = None, timeout_s: float = 5.0,
    ) -> dict | None:
        """Generic GET-and-parse, for a provider that needs a probe this class
        does not have a named method for.

        The named probes above encode ENGINE knowledge (which route, which
        field) and predate the provider split; this one encodes none, so a
        provider can own the route and the parsing while the transport keeps
        owning the connection pool and the deadline. New providers should use
        it rather than growing another `probe_<engine>_<thing>` here.

        Returns the decoded object, or None on any failure — a probe that
        cannot answer must say "cannot tell", never raise into the poller.
        """
        client = self._client_for(ep_cfg.backend_url)
        try:
            resp = await asyncio.wait_for(
                client.get(path, headers=headers or {}), timeout=timeout_s)
            if resp.status_code == 200:
                body = resp.json()
                return body if isinstance(body, dict) else None
        except Exception:
            pass
        return None

    async def probe_prefix_cache(self, ep_cfg: EndpointConfig) -> dict | None:
        """Scrape a backend's Prometheus `/metrics` for prefix-cache counters.

        Returns ``{"hits": int, "queries": int}`` (cumulative-since-backend-boot,
        token-weighted) or ``None`` when the backend doesn't expose them — vLLM
        publishes ``vllm:prefix_cache_hits_total`` / ``vllm:prefix_cache_queries_total``;
        llama.cpp has no equivalent, so its endpoints read as actual-rate ``n/a``
        (the cache-ability screen still covers them). Best-effort: any error → None.
        """
        client = self._client_for(ep_cfg.backend_url)
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

    async def probe_progress_counters(self, ep_cfg: EndpointConfig) -> dict | None:
        """Scrape a backend's `/metrics` for CUMULATIVE work counters (C6).

        Returns ``{"prompt": int, "generation": int}``, or ``None`` when the
        backend exposes neither (an unreachable backend, a non-200, an engine
        with no `/metrics`). ``None`` means "cannot discriminate", and every
        caller must fall back to today's behaviour on it rather than treating it
        as "no progress"; that distinction is the whole safety property of the
        progress-aware watchdog.

        BOTH engine families are covered, because the stall this exists for is
        not vLLM-specific — `creative`/`tier2`/the classify family are llama.cpp
        and were stalling too.

        🚨 **The llama.cpp generation counter is `n_decode_total`, NOT
        `tokens_predicted_total`.** The same-sounding name is the trap: MEASURED
        2026-08-23 against the live boxa during a 400-token generation, sampling
        every 3 s —
            tokens_predicted_total  +0, +0, +0, then +400 AT COMPLETION
            n_decode_total         +24, +66, +66, +65   (live, ~22/s)
        `tokens_predicted_total` is credited when the request FINISHES, so it
        reads frozen for exactly the window the watchdog is judging. Mapping it
        to vLLM's `generation_tokens_total` by name would have made this probe
        return "no progress" on every llama.cpp endpoint — a fix that runs,
        reports green, and protects nothing.

        Precision note: llama.cpp prints 6 significant figures, so a counter
        past 1e6 quantises (observed `prompt_tokens_total` stepping 1905110 →
        1905130). `n_decode_total` is the smaller counter and stays exact for
        far longer, and a backend emitting fewer than ~10 tokens per probe
        interval is not one we want to call healthy anyway.

        These are ENGINE-WIDE, not per-request: on a busy engine another
        request's tokens also advance them, so this can only ever prove the
        BACKEND is alive, never that MY stream is. That is why the watchdog
        bounds its extensions instead of trusting this indefinitely.

        Best-effort and short-timeout by construction: it runs alongside a
        stream that is already unhappy, so it must never become the thing that
        hangs. Any error → None.
        """
        client = self._client_for(ep_cfg.backend_url)
        try:
            resp = await asyncio.wait_for(client.get("/metrics"), timeout=3.0)
            if resp.status_code != 200:
                return None
            prompt = generation = None
            for line in resp.text.splitlines():
                if line.startswith("#"):
                    continue
                # 'vllm:prompt_tokens_total{engine="0",model_name="x"} 7815245.0'
                # 'llamacpp:n_decode_total 196408'
                try:
                    name, val = line.rsplit(" ", 1)
                    v = int(float(val))
                except ValueError:
                    continue
                # Summed, not replaced: a data-parallel backend publishes one
                # series per engine and taking the last would silently track
                # only whichever engine sorted last.
                if name.startswith("vllm:prompt_tokens_total"):
                    prompt = v if prompt is None else prompt + v
                elif name.startswith("vllm:generation_tokens_total"):
                    generation = v if generation is None else generation + v
                elif name.startswith("llamacpp:prompt_tokens_total"):
                    prompt = v if prompt is None else prompt + v
                elif name.startswith("llamacpp:n_decode_total"):
                    generation = v if generation is None else generation + v
            if prompt is not None or generation is not None:
                return {"prompt": prompt or 0, "generation": generation or 0}
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
        client = self._client_for(f"http://{shadow_host}:{shadow_port}")
        # The shadow target is a bare host:port with no EndpointConfig behind
        # it, so there is no provider to resolve. Routing is identical across
        # the local engines today; when a provider appears whose routes differ,
        # the shadow config has to name its provider rather than assume one.
        path = DEFAULT_PROVIDER.path_for(payload_type)
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
                # Same defensive coercion as call() — a null/NaN/bool shadow
                # usage count must never reach the INTEGER telemetry columns
                # (audit 2026-07-12, D-4; parity with the primary path).
                input_tokens=coerce_token_count(usage.get("prompt_tokens", 0)),
                output_tokens=coerce_token_count(usage.get("completion_tokens", 0)),
                cached_tokens=extract_cached_tokens(usage),
            )
        except Exception:
            return None
