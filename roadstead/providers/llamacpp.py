"""llama.cpp — the backend that tells the truth about its own capacity.

This is the only engine Roadstead talks to that publishes real slot counts and
real per-slot context (``/props``), and that measurement is what makes admission
control *accurate* rather than assumed. Everything else here is repair work for
chat templates.

Also the DEFAULT provider: an endpoint whose ``backend_engine`` is unset, or
spells something the registry does not know (``shim``, ``llama.cpp (Vulkan)``),
lands here. That is deliberate and it is the historic behaviour — every branch
this file replaces was written as ``== "vllm"``, so everything else already took
this path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import CapacityReport, Provider, ProviderDescriptor
from .payload import (
    _apply_thinking_token_budget,
    _has_anthropic_image_block,
    _needs_alternation_fix,
    _normalize_strict_alternation,
    _translate_anthropic_image_blocks,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import EndpointConfig


class LlamaCppProvider(Provider):
    descriptor = ProviderDescriptor(
        name="llama.cpp",
        kind="local",
        # /props: `default_generation_settings.n_parallel` (or `total_slots`,
        # or the length of `slots`) and a per-slot `n_ctx`. The reason
        # capacity discovery exists at all.
        publishes_slot_count=True,
        publishes_slot_context=True,
        # It reports context PER SLOT, which is the useful unit; there is no
        # separate whole-request ceiling to discover.
        publishes_context_ceiling=False,
        # One model per server, named on /v1/models — which is what makes the
        # served-id and weights-fingerprint probes worth running.
        publishes_served_model_id=True,
        # No `/metrics` prefix-cache counters and no per-request cached-token
        # field, so its cache hit rate reads as n/a — NOT as 0%. The absent-is-
        # not-zero contract in queue.py depends on this staying honest.
        publishes_prefix_cache_metrics=False,
        publishes_cached_tokens=False,
        # Nothing local charges money.
        publishes_token_costs=False,
        # Ignores the `model` field entirely, so the proxy leaves whatever the
        # caller sent in place.
        validates_model_field=False,
        grammar_field="grammar",
        # Mistral/Ministral 500 on consecutive same-role turns and on an
        # assistant right after the system.
        strict_alternation_templates=True,
        # No proxy-side kill switch: a llama.cpp reasoning model emits its
        # CoT unconditionally, which is why model_catalog sets
        # `forces_reasoning` here and the submit path reserves answer headroom.
        reasoning_is_switchable=False,
        # Labels a truncated tool call `length`, correctly.
        # Reached at host:port on a machine you run; no credential of its own.
        # Stated rather than defaulted — see test_provider_interface.
        addressed_by_base_url=False,
        requires_credential=False,
        default_base_url="",
        # Serves one model and names it; there is no catalogue to list.
        lists_available_models=False,
        mislabels_truncated_tool_calls=False,
    )

    def prepare_chat_payload(
        self,
        payload: dict,
        *,
        model_id: str | None = None,
        thinking_budget_ratio: float = 0.0,
        thinking_kwargs: tuple[str, ...] = (),
    ) -> dict:
        """Make an Anthropic/extra_body-shaped chat payload wire-correct for
        llama-server.

        The pre-proxy path built requests with ``call_nexus`` +
        ``openai.OpenAI``: the former inlined a top-level ``system`` field as
        the first ``messages`` entry, the latter merged ``extra_body`` keys
        (e.g. GBNF ``grammar``) into the top-level request body. The proxy's
        client does neither, so without this normalization llama-server silently
        ignores both — structured-output call sites lose their system prompt AND
        their grammar and fall back to free-form output that fails JSON parsing.

        ``model_id`` is accepted and ignored: llama.cpp does not validate the
        ``model`` field, so overwriting a caller's value would be churn with no
        wire effect. ``thinking_kwargs`` is likewise not injected here — see
        ``honours_thinking_kwargs`` in the descriptor.
        """
        if not isinstance(payload, dict):
            return payload
        needs_vision_xlate = _has_anthropic_image_block(payload.get("messages"))
        # Strict-alternation templates (Mistral/Ministral) 500 on consecutive
        # system/user/assistant AND on an assistant right after the system. An
        # inner agent loop emits these from empty-assistant history turns and
        # tool-observation injections — fine for Qwen's loose template, fatal
        # for Mistral. Normalize to valid alternation (content-preserving +
        # cache-safe: the leading system prefix is never touched). Subsumes the
        # old system-only coalesce (a >1-adjacent-system payload is one case).
        needs_alternation = _needs_alternation_fix(payload.get("messages"))
        if (
            "system" not in payload
            and "extra_body" not in payload
            and not needs_vision_xlate
            and not needs_alternation
            and not (thinking_budget_ratio > 0)
        ):
            return payload
        p = dict(payload)
        system = p.pop("system", None)
        if system:
            content = system if isinstance(system, str) else str(system)
            p["messages"] = [{"role": "system", "content": content},
                             *(p.get("messages") or [])]
        # Anthropic vision blocks → OAI image_url (both engines reject the
        # Anthropic shape with 400 unsupported content[].type). After the system
        # inline so the prepended system message is walked too (a no-op there).
        if needs_vision_xlate:
            p["messages"] = _translate_anthropic_image_blocks(p.get("messages"))
        # Runs AFTER the system-inline above so a top-level `system` prepended
        # in front of a messages list is folded in too. No-op when the sequence
        # is already valid (the common case).
        if _needs_alternation_fix(p.get("messages")):
            p["messages"] = _normalize_strict_alternation(p.get("messages"))
        extra_body = p.pop("extra_body", None)
        if isinstance(extra_body, dict):
            p.update(extra_body)
        # Applied by BOTH providers, exactly as the single pre-split function
        # did. The ratio mirrors a vLLM `--reasoning-config` launch flag and is
        # 0 on every llama.cpp stanza today, so this is a no-op here — kept
        # rather than dropped because "no-op today" is not "unreachable", and a
        # silent behaviour change in a payload path is the one thing the
        # provider split must not introduce.
        _apply_thinking_token_budget(p, thinking_budget_ratio)
        return p

    async def discover_capacity(
        self, pool: Any, ep_cfg: "EndpointConfig",
    ) -> CapacityReport | None:
        props = await pool.probe_props(ep_cfg)
        if not props:
            return None
        return self.parse_capacity(props)

    def parse_capacity(self, raw: dict) -> CapacityReport:
        """Parse ``/props`` into slots + per-slot context.

        🚨 ``default_generation_settings.n_ctx`` is ALREADY per-slot in current
        llama.cpp builds (confirmed live: ``--ctx-size 131072 --parallel 4``
        reports ``n_ctx=32768`` there, not 131072) — dividing it by n_parallel
        again silently quartered every multi-slot endpoint's discovered
        ``context_per_slot`` (32768 → 8192 for a 4-slot unit), which feeds the
        context-gate admission check and could wrongly REJECT requests that
        actually fit. Only the top-level ``props["n_ctx"]`` fallback
        (older/different builds, and it is unconfirmed whether anything ever
        populates it as an aggregate) still gets divided.
        """
        gen_settings = raw.get("default_generation_settings", {}) or {}
        n_parallel = gen_settings.get("n_parallel")
        if n_parallel is None:
            n_parallel = raw.get("total_slots")
        if n_parallel is None:
            slots = raw.get("slots")
            if isinstance(slots, list):
                n_parallel = len(slots)

        context_per_slot: int | None = None
        gen_n_ctx = gen_settings.get("n_ctx")
        if gen_n_ctx:
            context_per_slot = gen_n_ctx
        else:
            top_n_ctx = raw.get("n_ctx")
            if top_n_ctx and n_parallel:
                context_per_slot = top_n_ctx // n_parallel

        return CapacityReport(
            source="discovered",
            slots=n_parallel or None,
            context_per_slot=context_per_slot,
        )


#: The stateless singleton. Import this, never instantiate.
LLAMACPP = LlamaCppProvider()
