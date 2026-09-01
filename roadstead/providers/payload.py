"""Engine-neutral payload surgery shared by the local providers.

Everything here was measured against a real backend, and every comment names
the failure it prevents. It lives beside the providers rather than inside one
because both llama.cpp and vLLM need the same repairs to a caller's chat
payload — an Anthropic image block, a top-level ``system``, an ``extra_body``
the OpenAI SDK used to merge for us — while differing in what they do NEXT.

Moved out of ``backend.py`` on 2026-08-31 with the provider split, otherwise
unchanged, so ``git log -L`` on any function here still reaches the commit that
explains it.
"""

from __future__ import annotations

from typing import Any


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

