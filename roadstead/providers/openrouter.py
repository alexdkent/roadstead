"""OpenRouter — the first backend Roadstead does not own.

Everything the local providers take for granted is different here, which is the
point of it being the first remote one: it is reached over TLS at a base path
rather than at ``host:port``, it needs a credential, it fronts a CATALOGUE of
models rather than serving one, it publishes prices instead of slots, and it
cannot enforce a GBNF grammar at all.

🚨 **Remote capacity is not local capacity, and this file does not pretend
otherwise.** Nothing here reports slots, because there are none to report: the
scarce thing a remote provider sells is money and rate limit, not occupancy.
Roadstead's fairness unit is the slot-second precisely because LOCAL slots are
finite, so an endpoint pointed here still carries a config-seeded concurrency
cap — a policy knob we choose, not a discovered capacity. Making remote capacity
an *outcome of the same admission decision* (dispatch locally / spill / defer)
is Workstream D; this provider is what D will dispatch through.

**What it will tell us.** ``GET /models`` returns the whole catalogue, and the
entry for one model carries ``context_length`` and a ``pricing`` block in
dollars per token. That makes this the only provider so far whose descriptor
says ``publishes_token_costs`` — the reader for it arrives with D's cost
accounting, and the flag is what D will look for.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from .base import (
    CapacityReport,
    Provider,
    ProviderDescriptor,
    ProviderMisconfigured,
    UnsupportedRequest,
)
from .payload import _has_anthropic_image_block, _translate_anthropic_image_blocks

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import EndpointConfig

logger = logging.getLogger(__name__)

#: Fields the local engines need and OpenRouter has no idea about. They are
#: HINTS, not constraints — which is why dropping them is correct and dropping
#: a grammar is not (see `prepare_chat_payload`). Each one is a specific engine
#: instruction: a llama.cpp KV-cache slot to pin the conversation to, a chat
#: TEMPLATE variable that only exists because we run the template ourselves, and
#: a vLLM reasoning cap that is a mirror of a launch flag on a server we own.
_ENGINE_LOCAL_KEYS = ("id_slot", "chat_template_kwargs", "thinking_token_budget")


class OpenRouterProvider(Provider):
    descriptor = ProviderDescriptor(
        name="openrouter",
        kind="remote",
        # Reached at a base URL with a base path, needs a key, and knows its own
        # address. See `ProviderDescriptor` — each of these has a reader.
        addressed_by_base_url=True,
        requires_credential=True,
        default_base_url="https://openrouter.ai/api/v1",
        lists_available_models=True,
        # There is no occupancy to discover. A remote provider's concurrency
        # limit is a rate limit on OUR account, not a slot count, and it is not
        # published — so an endpoint here stays config-capped.
        publishes_slot_count=False,
        publishes_slot_context=False,
        # GET /models -> data[].context_length, per model.
        publishes_context_ceiling=True,
        # It fronts a catalogue. "Which model" is our routing choice, seeded
        # into served_model_id from config, so the served-id and fingerprint
        # probes have nothing to discover and the poller skips them.
        publishes_served_model_id=False,
        # No Prometheus endpoint, and no cache to report on.
        publishes_prefix_cache_metrics=False,
        # Upstreams that report cached prompt tokens do surface them in the
        # OpenAI-shaped usage block, so `extract_cached_tokens` already reads
        # them where they appear. It varies by upstream, which is exactly the
        # kind of thing a descriptor should not overclaim.
        publishes_cached_tokens=True,
        # 🔑 THE ONE THAT IS NEW. Prices per token, published per model, and
        # real money rather than the cloud-equivalent cost `usage_rates.py`
        # currently imputes. Workstream D reads this.
        publishes_token_costs=True,
        # It routes on the model slug and errors on one it does not know.
        validates_model_field=True,
        # 🚨 No GBNF, at all. See prepare_chat_payload — this provider REFUSES
        # a grammar rather than dropping it.
        grammar_field=None,
        # Templates are applied upstream; message shape is the upstream's
        # problem and it accepts ordinary OpenAI conversations.
        strict_alternation_templates=False,
        # Reasoning is a per-model upstream concern controlled by OpenRouter's
        # own `reasoning` parameter, not by a chat-template switch we inject.
        # True here means model_catalog must NOT add the forced-CoT headroom
        # reserve, which exists for a llama.cpp model with no kill switch.
        reasoning_is_switchable=True,
        # No evidence either way, and a defect declaration must be earned by
        # measurement rather than assumed by analogy with vLLM.
        mislabels_truncated_tool_calls=False,
    )

    # --- request shaping -----------------------------------------------------

    def path_for(self, payload_type: str) -> str:
        """Routes hang off the base path (``https://openrouter.ai/api/v1``), so
        they are relative where a local engine's are absolute — which is the
        whole reason ``base_url`` exists as a config field.

        Embedding and rerank raise: OpenRouter serves neither, and an endpoint
        configured to send them here is a routing mistake that should surface
        now rather than as a 404 from a stranger's server.
        """
        if payload_type in ("embedding", "rerank"):
            raise UnsupportedRequest(
                f"openrouter has no {payload_type} route — route this payload "
                f"type to a local backend")
        return "/chat/completions"

    def request_headers(
        self, ep_cfg: "EndpointConfig", request_id: str,
    ) -> dict[str, str]:
        return {
            "X-Request-ID": request_id,
            "Authorization": f"Bearer {self._api_key(ep_cfg)}",
        }

    @staticmethod
    def _api_key(ep_cfg: "EndpointConfig") -> str:
        """Read the key from the environment, by the name the endpoint declares.

        Read at REQUEST time rather than cached at load: a rotated secret then
        takes effect on the next call instead of the next restart, and nothing
        holds the plaintext anywhere it could be logged or persisted. Raising
        beats sending an unauthenticated request — that would come back as a
        401 attributed to the backend, which is a true statement about the wrong
        component.
        """
        name = (ep_cfg.api_key_env or "").strip()
        if not name:
            raise ProviderMisconfigured(
                f"endpoint {ep_cfg.role}: openrouter needs api_key_env naming "
                f"the environment variable that holds its API key")
        key = os.environ.get(name, "").strip()
        if not key:
            raise ProviderMisconfigured(
                f"endpoint {ep_cfg.role}: ${name} is unset or empty")
        return key

    def prepare_chat_payload(
        self,
        payload: dict,
        *,
        model_id: str | None = None,
        thinking_budget_ratio: float = 0.0,
        thinking_kwargs: tuple[str, ...] = (),
    ) -> dict:
        """Reduce a payload shaped for our own engines to plain OpenAI chat.

        Two kinds of removal, and the difference between them is the whole
        argument:

        * **Engine hints are dropped.** ``id_slot`` names a llama.cpp KV-cache
          slot, ``chat_template_kwargs`` sets a variable in a template we are
          not the ones applying, ``thinking_token_budget`` mirrors a vLLM launch
          flag. None of them is something a caller asked for; they are how we
          talk to hardware we own. Silence is correct.

        * **A grammar is REFUSED.** 🚨 ``grammar`` / ``structured_outputs`` is a
          caller CONSTRAINT, and this provider cannot enforce it. Dropping it
          would return free-form text where the caller required conforming
          output, and — the part that matters — the caller could not tell the
          difference from a model that simply did badly. That is the exact shape
          of the `finish_reason` bug recorded in CLAUDE.md, where a repair
          became a silencer. So it raises, and the request fails with a reason.

        (The correction layer's schema backstop is a repair for output that came
        back wrong, not a licence to route a constrained call somewhere it cannot
        be constrained. When the enriched API grows `response_format` handling,
        the honest translation is json_schema → the upstream's own structured
        output, per model — not GBNF, which nothing remote speaks.)
        """
        if not isinstance(payload, dict):
            return payload
        if self._declares_grammar(payload):
            raise UnsupportedRequest(
                "openrouter cannot enforce a GBNF grammar; refusing rather "
                "than dropping the constraint silently")
        needs_vision_xlate = _has_anthropic_image_block(payload.get("messages"))
        needs_model_set = bool(model_id and payload.get("model") != model_id)
        has_engine_keys = any(k in payload for k in _ENGINE_LOCAL_KEYS)
        if (
            "system" not in payload
            and "extra_body" not in payload
            and not needs_model_set
            and not needs_vision_xlate
            and not has_engine_keys
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
        if needs_vision_xlate:
            p["messages"] = _translate_anthropic_image_blocks(p.get("messages"))
        extra_body = p.pop("extra_body", None)
        if isinstance(extra_body, dict):
            p.update(extra_body)
            # The merge can reintroduce either kind — a caller's extra_body is
            # exactly where a grammar has always travelled.
            if self._declares_grammar(p):
                raise UnsupportedRequest(
                    "openrouter cannot enforce a GBNF grammar (arrived via "
                    "extra_body); refusing rather than dropping it silently")
        for key in _ENGINE_LOCAL_KEYS:
            p.pop(key, None)
        return p

    @staticmethod
    def _declares_grammar(payload: dict) -> bool:
        if isinstance(payload.get("grammar"), str) and payload["grammar"].strip():
            return True
        so = payload.get("structured_outputs")
        return isinstance(so, dict) and bool(so.get("grammar"))

    # --- capacity discovery --------------------------------------------------

    async def discover_capacity(
        self, pool: Any, ep_cfg: "EndpointConfig",
    ) -> CapacityReport | None:
        """Read this endpoint's model out of the catalogue.

        Returns None on anything unexpected, INCLUDING a missing credential:
        the poller reads None as "cannot tell", which trips the circuit breaker
        after its usual consecutive failures and stops dispatch to an endpoint
        that could not have served a request anyway. That is a better answer
        than 502-ing live traffic one call at a time to discover the same thing.
        """
        try:
            headers = self.request_headers(ep_cfg, "capacity-probe")
        except ProviderMisconfigured as exc:
            logger.warning("openrouter endpoint %s not usable: %s",
                           ep_cfg.role, exc)
            return None
        body = await pool.probe_json(ep_cfg, "/models", headers=headers)
        if not body:
            return None
        return self.parse_capacity({"models": body, "model_id": ep_cfg.effective_model_id})

    async def list_available_models(self, pool: Any,
                                    ep_cfg: "EndpointConfig") -> list[dict]:
        """The catalogue, flattened for a chooser.

        Same fetch `discover_capacity` already makes — `/models` — so this adds
        no new way of talking to the service, only a second reader of the answer.
        Sorted by id so the list is stable between calls; an operator scanning
        for a slug should not have to re-find it because the upstream reordered.
        """
        headers = self.request_headers(ep_cfg, "model-list")
        return self.parse_models(await pool.probe_json(
            ep_cfg, "/models", headers=headers))

    def parse_models(self, body: Any) -> list[dict]:
        """Flatten a ``GET /models`` body into rows to choose from.

        Split from the fetch for the reason `parse_capacity` is: the shaping is
        the part worth testing and the network is the part that makes testing it
        awkward. Sorted by id so the list is stable between calls — an operator
        scanning for a slug should not have to re-find it because the upstream
        reordered.
        """
        data = (body or {}).get("data")
        if not isinstance(data, list):
            return []
        out: list[dict] = []
        for entry in data:
            if not isinstance(entry, dict) or not entry.get("id"):
                continue
            price_in, price_out = self._parse_pricing(entry.get("pricing"))
            out.append({
                "id": str(entry["id"]),
                "name": str(entry.get("name") or entry["id"]),
                "context_length": entry.get("context_length"),
                "input_usd_per_mtok": price_in,
                "output_usd_per_mtok": price_out,
            })
        return sorted(out, key=lambda m: m["id"])

    def parse_capacity(self, raw: dict) -> CapacityReport | None:
        """Pick one model out of a catalogue response.

        ``raw`` is ``{"models": <GET /models body>, "model_id": <slug>}``. The
        catalogue is hundreds of entries and only ours is meaningful — a
        context ceiling taken from the wrong row would be a confidently wrong
        number feeding the admission context gate, so an id that is not in the
        catalogue reports nothing rather than the first row it finds.
        """
        model_id = raw.get("model_id")
        data = (raw.get("models") or {}).get("data")
        if not isinstance(data, list) or not model_id:
            return None
        for entry in data:
            if not isinstance(entry, dict) or entry.get("id") != model_id:
                continue
            ctx = entry.get("context_length")
            price_in, price_out = self._parse_pricing(entry.get("pricing"))
            return CapacityReport(
                source="openrouter context_length + pricing",
                slots=None,
                context_per_slot=ctx if isinstance(ctx, int) and ctx > 0 else None,
                input_usd_per_mtok=price_in,
                output_usd_per_mtok=price_out,
            )
        logger.warning(
            "openrouter catalogue has no model %r — capacity unknown", model_id)
        return None

    @staticmethod
    def _parse_pricing(pricing: Any) -> tuple[float | None, float | None]:
        """``{"prompt": "0.0000005", "completion": "0.0000015"}`` -> USD/Mtok.

        Two things about this shape that a reader will otherwise get wrong.

        **The values are STRINGS, and per single token.** They are strings
        because a price like ``0.0000005`` is exactly the magnitude where a JSON
        float starts losing digits, and per-token because that is the unit the
        upstream bills in. Both are multiplied up here, once, so nothing
        downstream has to remember the exponent — ``spend.py`` deals in USD per
        million tokens throughout because that is how every price is *quoted*.

        🚨 **A price of zero is a real price and is kept.** A free model on a
        remote provider is still a remote model, and reporting "no price" for it
        would push it onto the imputed avoided-cost fallback — which would then
        credit the deployment with money SAVED for a call it made over the
        internet. Only a missing or unparseable field reports ``None``, and the
        two halves are parsed independently: a provider that publishes a prompt
        price and no completion price has told us half of something true.
        """
        if not isinstance(pricing, dict):
            return (None, None)

        def _one(key: str) -> float | None:
            raw = pricing.get(key)
            if raw is None or isinstance(raw, bool):
                return None
            try:
                per_token = float(raw)
            except (TypeError, ValueError):
                logger.warning(
                    "openrouter pricing.%s is not a number (%r) — treating the "
                    "price as unpublished rather than as free", key, raw)
                return None
            if per_token < 0:
                return None
            return per_token * 1e6

        return (_one("prompt"), _one("completion"))


#: The stateless singleton. Import this, never instantiate.
OPENROUTER = OpenRouterProvider()
