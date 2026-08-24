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


def _first_nonsystem_is_assistant(messages: Any) -> bool:
    """True if the first non-system message is an assistant. Strict-alternation
    templates (Mistral/Ministral) require a USER turn to follow the optional
    leading system — an assistant there 500s ('After the optional system…')."""
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        r = msg.get("role")
        if r == "system":
            continue
        return r == "assistant"
    return False


def _needs_alternation_fix(messages: Any) -> bool:
    """Any shape a strict-alternation template rejects: consecutive same-role
    (system/user/assistant) or a leading assistant after the optional system."""
    return (
        _has_consecutive_role(messages, "system")
        or _has_consecutive_role(messages, "user")
        or _has_consecutive_role(messages, "assistant")
        or _first_nonsystem_is_assistant(messages)
    )


def _normalize_strict_alternation(messages: Any) -> list:
    """Make ``messages`` valid for strict user/assistant alternation templates
    (Mistral/Ministral): at most one leading system, then user/assistant/user…
    starting with user. Qwen tolerates loose shapes; Mistral 500s on ALL of:
    consecutive system, consecutive user, consecutive assistant, and an
    assistant right after the system.

    Two steps:
      1. Coalesce consecutive same-role runs (any role), joining string content
         with ``"\\n\\n"`` (non-string content starts a new run — vision blocks
         are never mangled). After this the sequence strictly alternates.
      2. Drop assistant messages that precede the first user turn (orphan history
         fragments with no preceding user), keeping a leading system.

    Cache-safe: the leading system's bytes are never modified and it stays at
    index 0, so the warm prefix-cache prefix is preserved. Content-preserving
    except the deliberate drop of orphan leading assistants (a broken history
    fragment; better dropped than a 500). Builds new dicts — never mutates
    the caller's objects.
    """
    out: list = []
    for msg in messages or []:
        role = msg.get("role") if isinstance(msg, dict) else None
        if (
            out
            and role is not None
            # Never merge consecutive tool messages: each carries its own
            # ``tool_call_id`` that pairs it to a specific assistant tool_call,
            # and ``{**prev, "content": ...}`` would keep only the FIRST id —
            # silently dropping the second tool result's linkage (audit
            # 2026-07-12, D-3a). Strict-alternation templates want user/assistant
            # alternation anyway; tool turns are the backend's concern, not ours
            # to coalesce.
            and role != "tool"
            and out[-1].get("role") == role
            and isinstance(out[-1].get("content"), str)
            and isinstance(msg.get("content"), str)
        ):
            prev = out[-1]
            out[-1] = {**prev, "content": f"{prev['content']}\n\n{msg['content']}"}
        else:
            out.append(msg)
    lead: list = []
    rest = out
    if rest and isinstance(rest[0], dict) and rest[0].get("role") == "system":
        lead = [rest[0]]
        rest = rest[1:]
    while rest and isinstance(rest[0], dict) and rest[0].get("role") == "assistant":
        rest = rest[1:]
    return lead + rest


def _normalize_chat_payload(
    payload: dict, vllm: bool = False, model_id: str | None = None,
    thinking_budget_ratio: float = 0.0,
    thinking_kwargs: tuple[str, ...] = (),
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

    Finally, for vLLM we default the endpoint's DECLARED thinking switch(es) to
    False — see the block at the injection site for which key and why.
    ``thinking_kwargs`` comes from the endpoint's ``policy.thinking_kwargs`` in
    ``models.yaml``; empty means "we do not know this template's switch", and we
    inject nothing rather than guess. A caller that pins ANY known thinking
    switch is left completely untouched.
    """
    if not isinstance(payload, dict):
        return payload
    needs_model_set = bool(vllm and model_id and payload.get("model") != model_id)
    needs_thinking_default = bool(
        vllm and thinking_kwargs and not _has_thinking_kwarg(payload))
    needs_vision_xlate = _has_anthropic_image_block(payload.get("messages"))
    # llama.cpp only: strict-alternation templates (Mistral/Ministral) 500 on
    # consecutive system/user/assistant AND on an assistant right after the
    # system. The inner loop emits these from empty-assistant history turns and
    # tool-observation injections — fine for Qwen's loose template, fatal for
    # Mistral. Normalize to valid alternation (content-preserving + cache-safe:
    # the leading system prefix is never touched). Subsumes the old system-only
    # coalesce (the composer 122B's >1-adjacent-system case is one instance).
    needs_alternation = (not vllm) and _needs_alternation_fix(
        payload.get("messages"))
    if (
        "system" not in payload
        and "extra_body" not in payload
        and not (vllm and "grammar" in payload)
        and not needs_model_set
        and not needs_thinking_default
        and not needs_vision_xlate
        and not needs_alternation
        and not (thinking_budget_ratio > 0)
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
    # llama.cpp only: normalize to strict user/assistant alternation (Mistral/
    # Ministral). Runs AFTER the system-inline above so a top-level `system`
    # prepended in front of a messages list is folded in too. No-op when the
    # sequence is already valid (the common case). Also collapses the composer
    # 122B's >1-adjacent-system shape (a consecutive-system run).
    if not vllm and _needs_alternation_fix(p.get("messages")):
        p["messages"] = _normalize_strict_alternation(p.get("messages"))
    extra_body = p.pop("extra_body", None)
    if isinstance(extra_body, dict):
        p.update(extra_body)
    if vllm and isinstance(p.get("grammar"), str) and p["grammar"].strip():
        so = p.get("structured_outputs")
        so = dict(so) if isinstance(so, dict) else {}
        so.setdefault("grammar", p.pop("grammar"))
        p["structured_outputs"] = so
    # Default thinking OFF for vLLM (checked AFTER the extra_body merge so a
    # caller's chat_template_kwargs nested in extra_body still wins).
    #
    # 🔑 WHICH KEY — this is MODEL-FAMILY-SPECIFIC and it has already bitten us.
    # The switch is a chat-TEMPLATE variable, so its spelling belongs to the
    # model, not to vLLM. Measured live through the proxy 2026-08-24, one probe
    # per cell, `chat_template_kwargs` sent verbatim:
    #
    #   endpoint       model                 `thinking`   `enable_thinking`
    #   tier3          DeepSeek-V4-Flash     ON (556ch)   ON (518ch)
    #   tier2-analyst  Qwen3.8-27B           NO-OP (0ch)  ON (2913ch)
    #   tier2-chat     Qwen3.6-35B           NO-OP (0ch)  ON (2572ch)
    #
    # So `enable_thinking` happens to be understood by BOTH families today and
    # `thinking` only by DeepSeek — which is why hardcoding the Qwen key
    # survived the 2026-08-23 tier3 model swap without an alarm. It survived on
    # luck: V4's template ORs the two names, and the serve script separately
    # pins `--default-chat-template-kwargs '{"thinking":false}'`, so the two
    # agreed. Driving it off the endpoint's declaration instead means the next
    # swap onto a template that reads only its OWN key cannot repeat this.
    #
    # ⚠️ WHY IT IS OFF BY DEFAULT, correctly stated (the previous version of
    # this comment was STALE and would have sent the next reader down the wrong
    # path). It is NOT that "no reasoning parser is configured" and reasoning
    # therefore corrupts structured output — tier3 runs `--reasoning-parser
    # deepseek_v4` and the split is CLEAN. Measured the same day, thinking ON:
    # the JSON-extraction probe returned `{"artist": "Miles Davis", "year":
    # 1959}` in `content` with 119 chars in a SEPARATE `reasoning` field, and
    # the judge probe returned `10`. Structured callers are not the problem.
    #
    # The default stays OFF because reasoning tokens are ADDITIVE to the answer
    # and essentially every caller sizes max_tokens for the answer alone, so a
    # fleet-wide flip exposes every tier3 call to the recorded bimodal tail
    # (reasoning past 12k tokens → `content: ""` + finish=length → a 502). That
    # is a availability risk taken on behalf of callers who did not ask for it.
    # Thinking is therefore OPT-IN per call site (`thinking: true`, handled in
    # `Correction.apply_thinking`), which is also what lets a caller size its
    # own budget. What the opt-in COSTS is small when the budget is not
    # inflated: same open-ended prompt, 11.0s/384 completion tokens with
    # thinking ON vs 21.9s/738 with it OFF — ON was FASTER and its `content`
    # was 996 chars of answer instead of 2,772 chars of answer-with-
    # deliberation-inline.
    if vllm and thinking_kwargs and not _has_thinking_kwarg(p):
        ck = p.get("chat_template_kwargs")
        ck = dict(ck) if isinstance(ck, dict) else {}
        for key in thinking_kwargs:
            ck[key] = False
        p["chat_template_kwargs"] = ck
    _apply_thinking_token_budget(p, thinking_budget_ratio)
    return p


#: Floor and ceiling on an injected reasoning cap.
#:
#: FLOOR: vLLM #44676 reports forced reasoning-end tokens landing INSIDE tool-call
#: JSON at ~256 tokens, ~75% of runs. Anthropic's published minimum thinking budget is
#: 1024 and is the only floor-shaped number anyone documents. 2000 sits above both.
#: Measured here 2026-08-23: at 2000 the model answered but rambled to the ceiling
#: every time, so this is a floor to stay ABOVE, not a target.
#: CEILING: past ~12k, published accuracy curves are falling, not rising
#: (arXiv 2604.10739 peaks at 10-12k and crosses into net harm around 7k), and our own
#: worst observed block was 16,562 tokens — well past the peak.
_THINKING_BUDGET_FLOOR = 2000
_THINKING_BUDGET_CEILING = 16000
#: Below this `max_tokens` there is no sane split: the floor would eat the whole
#: allowance and leave nothing for an answer, which is the failure we are preventing.
_THINKING_BUDGET_MIN_MAX_TOKENS = 3000


def _apply_thinking_token_budget(p: dict, ratio: float) -> None:
    """Cap REASONING at a fraction of `max_tokens`, so an answer always has room.

    Complements — never replaces — `Correction.apply_thinking`, which ADDS
    `thinking_reasoning_budget()` (8000) of headroom to `max_tokens` on the proxy's
    own `thinking: true` opt-in. That mechanism makes the pie bigger so reasoning does
    not truncate the answer; this one slices the pie so reasoning cannot eat all of it.
    Both are needed, and they act on different callers: the opt-in only fires for
    callers that send the top-level control field, while this fires for anyone who has
    thinking on by any route (dsh sets `chat_template_kwargs` directly and never
    touches the opt-in).

    Measured on tier3 2026-08-23, N=2, interleaved, streaming: with no cap the model's
    natural reasoning length is BIMODAL — sometimes ~7-8k tokens and a clean answer,
    sometimes past a 12,000-token ceiling with `content: ""` and finish=length. The cap
    exists to bound that TAIL, not to shorten the median, which is why the ratio is
    generous rather than tight.

    Silent no-op unless the endpoint DECLARES support (`thinking_budget_ratio` > 0,
    mirrored from its `--reasoning-config` launch flag). vLLM 400s the entire request
    when the parameter arrives at a server without that flag, so a wrong declaration
    does not degrade an endpoint, it breaks every thinking call to it.
    """
    if not ratio or ratio <= 0:
        return
    if "thinking_token_budget" in p:
        return                      # caller declared intent; never override it
    ck = p.get("chat_template_kwargs")
    if not isinstance(ck, dict):
        return
    if not (ck.get("thinking") or ck.get("enable_thinking")):
        return                      # thinking is off — a budget would be meaningless
    mt = p.get("max_tokens")
    if not isinstance(mt, int) or mt < _THINKING_BUDGET_MIN_MAX_TOKENS:
        return
    budget = int(mt * ratio)
    budget = max(_THINKING_BUDGET_FLOOR, min(budget, _THINKING_BUDGET_CEILING))
    if budget >= mt:
        return                      # nothing left for an answer; leave it alone
    p["thinking_token_budget"] = budget



def empty_completion_error(role: str, msg: dict, finish_reason: str | None,
                           output_tokens: int) -> "BackendError | None":
    """The empty-completion decision, as a pure function so it can be TESTED.

    Extracted 2026-07-31 rather than left inline: a test that re-implements this
    logic is a copy that drifts, and the whole point of the change is that the
    message must stay precise. See
    `tests/llmproxy/test_empty_completion_names_its_cause.py`.

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


#: Every spelling of the thinking switch any fleet chat template understands.
#:
#: DETECTION uses the whole union; INJECTION uses only the endpoint's declared
#: `policy.thinking_kwargs`. The asymmetry is deliberate and is the actual bug
#: fixed here: a caller pinning DeepSeek's `thinking` was invisible to a guard
#: that only knew Qwen's `enable_thinking`, so the proxy appended a
#: contradictory `enable_thinking: False` to a payload the caller had already
#: made up its mind about. That was survivable only because V4's template ORs
#: the two names (measured: `{"thinking": true}` + an injected
#: `enable_thinking: false` still reasoned, 556 chars). A template that instead
#: ANDs them, or lets the second name win, would have turned every opt-in into
#: a SILENT no-op — an answer with no reasoning reads as "the model didn't
#: reason today", not as a proxy bug. Respect a pin, whatever it is called.
#:
#: Add a name here when a new family arrives; that costs nothing, whereas
#: MISSING a name silently overrides callers.
_THINKING_KWARG_NAMES = frozenset({"thinking", "enable_thinking"})


def _has_thinking_kwarg(payload: dict) -> bool:
    """True when the caller has already pinned ANY known thinking switch in
    chat_template_kwargs (top-level or nested in extra_body) — in which case we
    leave the payload alone, whichever name and whichever value they chose."""
    for container in (payload, payload.get("extra_body")):
        if isinstance(container, dict):
            ck = container.get("chat_template_kwargs")
            if isinstance(ck, dict) and not _THINKING_KWARG_NAMES.isdisjoint(ck):
                return True
    return False


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
    three llama.cpp backends (tier1 :9091, tier2-analyst :9196, tier2-chat
    :30000): each returns ``prompt_tokens_details: {"cached_tokens": N}``, and
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
            ep_cfg.host, ep_cfg.port, ep_cfg.effective_max_slots + 4)
        path = self._path_for(payload_type)
        headers = {"X-Request-ID": request_id}
        if payload_type == "chat_completion":
            payload = _normalize_chat_payload(
                payload, vllm=(ep_cfg.backend_engine == "vllm"),
                model_id=ep_cfg.effective_model_id,
                thinking_budget_ratio=ep_cfg.thinking_budget_ratio,
                thinking_kwargs=ep_cfg.thinking_kwargs)

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
            ep_cfg.host, ep_cfg.port, ep_cfg.effective_max_slots + 4)
        path = self._path_for(payload_type)
        headers = {"X-Request-ID": request_id}
        if payload_type == "chat_completion":
            payload = _normalize_chat_payload(
                payload, vllm=(ep_cfg.backend_engine == "vllm"),
                model_id=ep_cfg.effective_model_id,
                thinking_budget_ratio=ep_cfg.thinking_budget_ratio,
                thinking_kwargs=ep_cfg.thinking_kwargs)

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
        client = self._client_for(ep_cfg.host, ep_cfg.port)
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
                # Same defensive coercion as call() — a null/NaN/bool shadow
                # usage count must never reach the INTEGER telemetry columns
                # (audit 2026-07-12, D-4; parity with the primary path).
                input_tokens=coerce_token_count(usage.get("prompt_tokens", 0)),
                output_tokens=coerce_token_count(usage.get("completion_tokens", 0)),
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
