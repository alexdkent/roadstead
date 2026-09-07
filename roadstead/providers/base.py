"""The provider interface — Roadstead's south face, made explicit.

A *provider* is the adapter for one kind of inference backend. It owns the two
things engines genuinely disagree about, and nothing else:

1. **What a request has to look like to be accepted** (:meth:`Provider.path_for`,
   :meth:`Provider.prepare_chat_payload`). vLLM validates the ``model`` field and
   enforces GBNF only via ``structured_outputs.grammar``; llama.cpp ignores the
   first and reads the second at the top level. Neither is a preference — send
   the wrong shape and the call 404s, 400s, or silently drops the constraint.

2. **What it can tell us about itself** (:meth:`Provider.discover_capacity` and
   :class:`ProviderDescriptor`). This is deliberately ASYMMETRIC, and the
   asymmetry is a property of the engines rather than an oversight: llama.cpp
   ``/props`` publishes real slot counts and per-slot context, while vLLM
   publishes only ``max_model_len`` and keeps ``--max-num-seqs`` off the API
   entirely, so vLLM concurrency stays config-seeded with a drift alert. The
   descriptor states which is which, so a caller branches on a CAPABILITY it
   needs instead of on an engine name it recognises.

**Transport is not a provider concern.** Connection pools, deadlines, the error
taxonomy and the SSE relay live in ``backend.py`` and are the same for every
backend; a provider that opened its own sockets would fork the single-loop
concurrency invariant (``docs/internals.md``). Providers are handed the pool and ask it
to probe, which is also what keeps the unit suite's network isolation working —
it stubs probes by name on ``BackendClientPool``.

🚨 **Providers are stateless singletons and must stay that way.** One instance
is shared by every endpoint of its engine on the single event loop. Per-endpoint
state belongs on ``EndpointConfig``; anything mutable here is a data race that
no test in this repo would catch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import EndpointConfig


class ProviderError(Exception):
    """A provider cannot serve this request, and saying so is the point.

    🚨 The alternative — quietly dropping what a provider cannot honour — is
    the failure mode this codebase treats as worst: a caller that asked for a
    grammar and got free-form text back has no way to tell that its constraint
    was discarded, and will read the result as a bad model rather than a bad
    route. A provider that must drop a CONSTRAINT raises; a provider dropping
    an engine HINT (a slot id, a chat-template switch) just drops it.
    """


class UnsupportedRequest(ProviderError):
    """This provider cannot do what the request asks — an enforced grammar, a
    payload type it has no route for. A property of the pairing, not a fault:
    the same request to a different provider is fine."""


class ProviderMisconfigured(ProviderError):
    """The endpoint's own configuration is incomplete — most often a declared
    API key that is not in the environment. Distinct from
    :class:`UnsupportedRequest` because nothing the caller does can fix it, and
    because an endpoint in this state should fail its health probe and stop
    being dispatched to rather than 502 live traffic one call at a time."""


@dataclass(frozen=True)
class ProviderDescriptor:
    """What this kind of backend publishes, requires, and gets wrong.

    Every field exists because some call site used to ask ``backend_engine ==
    "vllm"`` and mean one of these instead. Keep it that way: a field with no
    reader is a claim nobody checks. The one deliberate exception is
    ``publishes_token_costs``, which is False everywhere today and is the seam
    Workstream D (spill + costing) needs from a remote provider.
    """

    #: Registry key, and what ``models.yaml`` spells in ``backend_engine``.
    name: str
    #: ``"local"`` capacity is the scarce thing DRR fair-shares in slot-seconds;
    #: ``"remote"`` capacity is elastic and governed by cost instead. No remote
    #: provider exists yet — the distinction is stated now because admission
    #: (local / spill / defer) is a single decision, not two systems.
    kind: str = "local"

    # --- what it can TELL us -------------------------------------------------
    #: Publishes its real concurrency (llama.cpp ``/props``). When False the
    #: endpoint's ``max_slots`` stays config-seeded and only a drift alert can
    #: notice it is wrong.
    publishes_slot_count: bool = False
    #: Publishes per-slot context (llama.cpp ``default_generation_settings.n_ctx``).
    publishes_slot_context: bool = False
    #: Publishes a whole-request context ceiling (vLLM ``max_model_len``).
    publishes_context_ceiling: bool = False
    #: Serves ONE model and will name it — so the poller can discover what this
    #: endpoint answers to, and notice the weights changing under it. False for
    #: a provider fronting a catalogue of models, where "which model" is a
    #: routing choice we make rather than a fact to read off the backend.
    publishes_served_model_id: bool = False
    #: Publishes Prometheus prefix-cache counters on ``/metrics``. Gates the
    #: cache-stats scrape; llama.cpp has no equivalent, so its endpoints read
    #: as ``n/a`` rather than as a 0% hit rate.
    publishes_prefix_cache_metrics: bool = False
    #: Can report per-request cached prompt tokens in ``usage``. Note this is
    #: an engine CAPABILITY, not a promise: vLLM only fills it when launched
    #: with ``--enable-prompt-tokens-details`` (see ``health.compute_cache_stats``).
    publishes_cached_tokens: bool = False
    #: Reports what a call actually cost in money. Nothing local does.
    publishes_token_costs: bool = False

    # --- how it is REACHED ---------------------------------------------------
    #: Reached at a ``base_url`` carrying a scheme and a base path, rather than
    #: at ``host``/``port``. Two readers: the management plane refuses a stanza
    #: that gives the wrong one, and the operator UI shows the field that
    #: applies instead of all three and letting somebody fill in the wrong two.
    #:
    #: 🚨 Not the same question as ``kind``. "Remote" is about whose capacity it
    #: is and who bills for it; this is about what an address looks like. They
    #: coincide today and conflating them is how a local engine behind a
    #: gateway becomes unconfigurable.
    addressed_by_base_url: bool = False
    #: Refuses to serve without a credential, so a provider stanza that names no
    #: ``api_key_env`` is misconfigured. Declared rather than discovered at first
    #: dispatch: without it the failure is a `ProviderMisconfigured` on the first
    #: real request, which is the config gap surfacing as a runtime fault —
    #: exactly what the management plane exists to catch earlier.
    requires_credential: bool = False
    #: The service's own address, for a form to prefill. Empty for anything
    #: whose address is a property of the deployment rather than of the engine.
    default_base_url: str = ""
    #: Fronts a CATALOGUE and will enumerate it, so "which model" is a choice an
    #: operator makes from a list rather than a slug they have to know. Two
    #: readers: `GET /rs/v1/admin/providers/{p}/models` refuses a provider that
    #: cannot, and the operator UI turns its model field into a picker when it
    #: can. 🚨 Related to `publishes_served_model_id` and not the same: that one
    #: says the backend will name the ONE model it is serving, this one says it
    #: will list the many it could serve. A provider that does the first cannot
    #: do the second, which is why they are two fields and not one.
    lists_available_models: bool = False

    # --- what it REQUIRES of a request ---------------------------------------
    #: 404s unless the ``model`` field names what it is serving, so the proxy
    #: must overwrite the caller's alias with the discovered served id.
    validates_model_field: bool = False
    #: Where a GBNF grammar has to sit to be enforced: ``"grammar"`` (top level,
    #: llama.cpp) or ``"structured_outputs"`` (vLLM). ``None`` = no grammar
    #: support. Put it in the wrong place and the backend accepts the request
    #: and ignores the constraint, which is the failure that looks like a bad
    #: model rather than a bad request.
    grammar_field: str | None = None
    #: Its chat templates require strict user/assistant alternation and 500 on
    #: consecutive same-role turns (Mistral/Ministral).
    strict_alternation_templates: bool = False
    #: Reasoning can be turned OFF from the proxy side, by defaulting the
    #: endpoint's declared switch in ``chat_template_kwargs``. Two readers, one
    #: fact: the vLLM provider injects that default, and ``model_catalog`` sets
    #: ``forces_reasoning`` on a reasoning model where this is False — its CoT
    #: is unconditional, so the submit path has to reserve answer headroom on
    #: top of the caller's cap. The switch's SPELLING is per model family and
    #: lives in the catalog, never here.
    reasoning_is_switchable: bool = False
    #: Wire name this engine reads for a REASONING TOKEN CAP, or None where it
    #: has none. Same rule as ``grammar_field`` directly above: the proxy knows
    #: one concept and each engine names it differently, so branch on the
    #: declaration rather than on ``backend_engine`` at the call site.
    #:
    #: 🚨 THE TWO SPELLINGS ARE NOT INTERCHANGEABLE, AND THE DIFFERENCE IS
    #: SILENT. vLLM reads ``thinking_token_budget`` and only when the server was
    #: launched with ``--reasoning-config``; llama.cpp reads
    #: ``reasoning_budget_tokens`` and needs no launch flag. Sending vLLM's
    #: spelling to llama.cpp is not an error — the field is simply ignored, so a
    #: declared cap reads as applied and bounds nothing. Measured 2026-09-07:
    #: llama.cpp b10488 honours ``reasoning_budget_tokens`` (reasoning 8,595 ->
    #: 1,858 chars at a 512 cap, 2/2), while every llama.cpp stanza's
    #: ``thinking_budget_ratio`` had been a documented no-op for exactly this
    #: reason.
    reasoning_budget_field: str | None = None

    # --- characterised DEFECTS -----------------------------------------------
    #: Labels a tool-call response ``finish_reason=tool_calls`` even when the
    #: arguments were cut mid-JSON. llama.cpp labels that truncation ``length``
    #: correctly, so the structured-validity guard is engine-specific.
    mislabels_truncated_tool_calls: bool = False


@dataclass(frozen=True)
class CapacityReport:
    """One discovery pass, parsed. ``None`` means *this backend cannot tell us*
    — never zero, and never "unchanged". Both callers must be able to keep a
    config-seeded value rather than overwrite it with a guess."""

    #: Where the numbers came from, for the discovery log line.
    source: str
    #: Real concurrency, when the backend publishes it.
    slots: int | None = None
    #: Context available to ONE request. Already per-slot — the divide-by-
    #: n_parallel decision belongs to the parser that knows the engine's units
    #: (see ``LlamaCppProvider.parse_capacity``).
    context_per_slot: int | None = None
    #: Published price, USD per MILLION tokens. Only a provider whose descriptor
    #: says ``publishes_token_costs`` ever sets these; ``None`` means the backend
    #: did not say, which is not the same as free.
    #:
    #: 🚨 Plain floats rather than a ``spend.TokenPrice`` deliberately. This
    #: package's import graph runs ``config`` -> ``model_catalog`` -> ``providers``,
    #: so a provider importing ``spend`` (which needs ``config`` for the priority
    #: enum) would close a cycle. It is also the right split on its own terms: a
    #: report says what the BACKEND published, and whether that counts as money
    #: somebody owes is a policy question ``spend.PriceBook`` answers.
    input_usd_per_mtok: float | None = None
    output_usd_per_mtok: float | None = None

    @property
    def publishes_prices(self) -> bool:
        return (self.input_usd_per_mtok is not None
                or self.output_usd_per_mtok is not None)


class Provider(ABC):
    """Adapter for one kind of backend. Stateless; see the module docstring."""

    #: Immutable declaration. Read it instead of testing ``isinstance``.
    descriptor: ProviderDescriptor

    @property
    def name(self) -> str:
        return self.descriptor.name

    # --- request shaping -----------------------------------------------------

    def path_for(self, payload_type: str) -> str:
        """Route for a payload type. Shared by both local engines today because
        both speak the OpenAI chat route and both sit behind the same embed /
        rerank shims; a remote provider will override it."""
        if payload_type == "embedding":
            return "/embed"
        if payload_type == "rerank":
            return "/rerank"
        return "/v1/chat/completions"

    def request_headers(
        self, ep_cfg: "EndpointConfig", request_id: str,
    ) -> dict[str, str]:
        """Headers for one backend call. The local engines take no auth; a
        remote provider adds its own here rather than having the transport
        learn about credentials.

        May raise :class:`ProviderMisconfigured` — an endpoint that cannot
        authenticate must not send the request anyway.
        """
        return {"X-Request-ID": request_id}

    @abstractmethod
    def prepare_chat_payload(
        self,
        payload: dict,
        *,
        model_id: str | None = None,
        thinking_budget_ratio: float = 0.0,
        thinking_kwargs: tuple[str, ...] = (),
        reasoning_budget_tokens: int = 0,
    ) -> dict:
        """Make a caller's chat payload wire-correct for this backend.

        Must never mutate ``payload`` — corpus capture stores ``req.payload``
        and the retry path re-sends it. Returns the original object unchanged
        when there is nothing to do, so the common case allocates nothing.
        """

    # --- capacity discovery --------------------------------------------------

    @abstractmethod
    async def discover_capacity(
        self, pool: Any, ep_cfg: "EndpointConfig",
    ) -> CapacityReport | None:
        """Probe this backend and report what it will admit to.

        ``pool`` is the ``BackendClientPool`` — providers borrow its connection
        pools and its probe methods rather than opening sockets of their own.
        ``None`` on any failure: an unreachable backend must read as "cannot
        tell", which is what leaves the configured capacity standing.
        """

    async def list_available_models(self, pool: Any,
                                    ep_cfg: "EndpointConfig") -> list[dict]:
        """The models this provider could serve, for an operator to choose from.

        Only meaningful when the descriptor says ``lists_available_models``.
        Each entry is ``{"id", "name", "context_length", "input_usd_per_mtok",
        "output_usd_per_mtok"}`` — enough to choose with, and priced, because
        "which model" and "what will it cost" are the same question when
        somebody is picking one.

        Default: nothing. An engine that serves one model has no catalogue, and
        returning its own id here would make a list of one look like a choice.
        """
        return []

    def parse_capacity(self, raw: dict) -> CapacityReport | None:
        """Pure parse of a probe body, split out from the I/O so the engine
        knowledge is testable without a socket."""
        raise NotImplementedError
