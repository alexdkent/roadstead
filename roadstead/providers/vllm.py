"""vLLM — elastic decode, and almost nothing it will admit to.

The mirror image of llama.cpp: it is the faster, larger-batch engine and it
publishes ``max_model_len`` and nothing else useful about capacity.
``--max-num-seqs`` is not exposed over the API at all, so an endpoint's
concurrency stays config-seeded here with a drift alert on top. That asymmetry
is a property of the engine, not an oversight, and the descriptor states it
rather than averaging it away.

It is also the strict one about request shape — it validates the ``model``
field, reads GBNF only from ``structured_outputs``, and 400s the WHOLE request
over a ``thinking_token_budget`` sent to a server launched without
``--reasoning-config``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import CapacityReport, Provider, ProviderDescriptor
from .payload import (
    _apply_thinking_token_budget,
    _has_anthropic_image_block,
    _has_thinking_kwarg,
    _translate_anthropic_image_blocks,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import EndpointConfig


class VLLMProvider(Provider):
    descriptor = ProviderDescriptor(
        name="vllm",
        kind="local",
        # No /props, no /slots, and --max-num-seqs is not on the API. All it
        # will tell us is the per-request context ceiling.
        publishes_slot_count=False,
        publishes_slot_context=False,
        publishes_context_ceiling=True,
        # One model per server. Its `id` is the operator's --served-model-name
        # and stays put across a weights swap, which is exactly why the
        # fingerprint probe reads `root` instead.
        publishes_served_model_id=True,
        # Prometheus vllm:prefix_cache_{hits,queries}_total — the only real
        # per-endpoint cache hit rate either engine gives us.
        publishes_prefix_cache_metrics=True,
        # CAPABLE of per-request cached_tokens, which is not the same as
        # reporting them: the field is gated behind
        # `--enable-prompt-tokens-details` at launch and reads NULL without it
        # (see health.compute_cache_stats). A capability, not a promise.
        publishes_cached_tokens=True,
        # Nothing local charges money. The field is here for the remote
        # providers Workstream D needs to meter.
        publishes_token_costs=False,
        # 404s on any `model` it is not serving — a role alias, an endpoint
        # class, or a different model the caller asked for and the proxy routed
        # here. The proxy overwrites the field with the discovered served id.
        validates_model_field=True,
        grammar_field="structured_outputs",
        strict_alternation_templates=False,
        # Reasoning is OFF by default and opt-in per call — the proxy
        # defaults the declared switch(es) in chat_template_kwargs.
        reasoning_is_switchable=True,
        # 🚨 Labels a tool call cut mid-JSON `finish_reason=tool_calls`, so
        # truncation is indistinguishable from completion without parsing the
        # arguments. correction.py's structured-validity guard exists for this
        # and is gated on it.
        # Reached at host:port on a machine you run; no credential of its own.
        # Stated rather than defaulted — see test_provider_interface.
        addressed_by_base_url=False,
        requires_credential=False,
        default_base_url="",
        # Serves one model and names it; there is no catalogue to list.
        lists_available_models=False,
        mislabels_truncated_tool_calls=True,
    )

    def prepare_chat_payload(
        self,
        payload: dict,
        *,
        model_id: str | None = None,
        thinking_budget_ratio: float = 0.0,
        thinking_kwargs: tuple[str, ...] = (),
    ) -> dict:
        """Make an Anthropic/extra_body-shaped chat payload wire-correct for vLLM.

        Four vLLM-specific repairs on top of the shared ones (system inline,
        vision translation, extra_body merge):

        * a top-level ``grammar`` (llama.cpp's field) is SILENTLY IGNORED here
          — vLLM enforces GBNF only via ``structured_outputs.grammar``, so we
          move it there and the endpoint actually constrains its output instead
          of emitting free-form text;
        * ``payload["model"]`` is forced to ``model_id`` when given, because
          vLLM validates the field and 404s on any other name;
        * the endpoint's DECLARED thinking switch(es) default to False — see
          the block at the injection site for which key and why;
        * ``thinking_token_budget`` is capped as a fraction of ``max_tokens``
          when the endpoint declares a ratio.

        ``thinking_kwargs`` comes from the endpoint's ``policy.thinking_kwargs``
        in ``models.yaml``; empty means "we do not know this template's switch",
        and we inject nothing rather than guess. A caller that pins ANY known
        thinking switch is left completely untouched.
        """
        if not isinstance(payload, dict):
            return payload
        needs_model_set = bool(model_id and payload.get("model") != model_id)
        needs_thinking_default = bool(
            thinking_kwargs and not _has_thinking_kwarg(payload))
        needs_vision_xlate = _has_anthropic_image_block(payload.get("messages"))
        if (
            "system" not in payload
            and "extra_body" not in payload
            and "grammar" not in payload
            and not needs_model_set
            and not needs_thinking_default
            and not needs_vision_xlate
            and not (thinking_budget_ratio > 0)
        ):
            return payload
        p = dict(payload)
        if model_id:
            p["model"] = model_id
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
        extra_body = p.pop("extra_body", None)
        if isinstance(extra_body, dict):
            p.update(extra_body)
        if isinstance(p.get("grammar"), str) and p["grammar"].strip():
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
        if thinking_kwargs and not _has_thinking_kwarg(p):
            ck = p.get("chat_template_kwargs")
            ck = dict(ck) if isinstance(ck, dict) else {}
            for key in thinking_kwargs:
                ck[key] = False
            p["chat_template_kwargs"] = ck
        _apply_thinking_token_budget(p, thinking_budget_ratio)
        return p

    async def discover_capacity(
        self, pool: Any, ep_cfg: "EndpointConfig",
    ) -> CapacityReport | None:
        cap = await pool.probe_vllm_capacity(ep_cfg)
        if not cap:
            return None
        return self.parse_capacity(cap)

    def parse_capacity(self, raw: dict) -> CapacityReport:
        """``max_model_len`` is the per-request context window DIRECTLY — not a
        fleet-wide n_ctx to divide by slots. Slots stay absent: vLLM does not
        expose ``--max-num-seqs`` over the API, and the proxy's admission cap is
        a deliberate policy knob rather than a discovered value, so reporting a
        made-up number here would be worse than reporting none."""
        mlen = raw.get("max_model_len")
        return CapacityReport(
            source="vLLM max_model_len",
            slots=None,
            context_per_slot=mlen if isinstance(mlen, int) and mlen > 0 else None,
        )


#: The stateless singleton. Import this, never instantiate.
VLLM = VLLMProvider()
