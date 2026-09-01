"""Typed views over the enriched envelope.

Thin on purpose. Each class wraps the raw dict and names the fields
``docs/api.md`` publishes; unknown keys stay reachable through ``.raw`` so a
proxy newer than this SDK is usable rather than lossy. 🚨 That last part is the
whole design rule here: a client that discards what it does not recognise turns
every server-side addition into an invisible loss for its callers, and the
person who eventually notices has no way to tell it apart from the server not
sending it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Price:
    input_usd_per_mtok: float = 0.0
    output_usd_per_mtok: float = 0.0
    #: 🚨 True = an INVOICE, False = a cost AVOIDED by running locally. Both are
    #: USD per million tokens and nothing else distinguishes them. Never add a
    #: real price to an avoided one — that is the single bug ``spend.py`` exists
    #: to prevent, and it is just as available to a client.
    real: bool = False
    source: str = ""
    detail: str = ""

    @classmethod
    def of(cls, raw: dict | None) -> "Price":
        raw = raw or {}
        return cls(
            input_usd_per_mtok=float(raw.get("input_usd_per_mtok") or 0.0),
            output_usd_per_mtok=float(raw.get("output_usd_per_mtok") or 0.0),
            real=bool(raw.get("real")),
            source=str(raw.get("source") or ""),
            detail=str(raw.get("detail") or ""),
        )


class _Wrapped:
    """Attribute access over a dict, with the raw form always reachable."""

    __slots__ = ("raw",)

    def __init__(self, raw: dict | None) -> None:
        self.raw: dict = dict(raw or {})

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"{type(self).__name__}({self.raw!r})"


class Attribution(_Wrapped):
    """Who served the request, and whether that is who was asked for."""

    @property
    def requested(self) -> str:
        """The caller's own words — the intent profile or the pin, as sent."""
        return str(self.raw.get("requested") or "")

    @property
    def resolved(self) -> str:
        """The endpoint intent resolution chose, before any substitution."""
        return str(self.raw.get("resolved") or "")

    @property
    def endpoint(self) -> str:
        """The endpoint that actually served."""
        return str(self.raw.get("endpoint") or "")

    @property
    def substituted(self) -> bool:
        """🚨 True when the request was moved AFTER the endpoint was chosen.

        Not true merely because an intent resolved to some endpoint the caller
        did not name — that is Roadstead doing the job it was asked to do.
        """
        return bool(self.raw.get("substituted"))

    @property
    def substitution(self) -> str:
        """``failover`` | ``spill`` | ``""``.

        🚨 Two different events and a caller reacts to them differently:
        ``failover`` means the backend was DOWN and a smaller model answered —
        a worse answer, possibly worth redoing later. ``spill`` means it was
        FULL and somebody else was paid to answer — the answer is fine and the
        thing that changed is that it cost money and left the machine.
        """
        return str(self.raw.get("substitution") or "")

    @property
    def provider(self) -> str:
        return str(self.raw.get("provider") or "")

    @property
    def engine(self) -> str:
        return str(self.raw.get("engine") or "")

    @property
    def model(self) -> str:
        """The model the backend is actually serving, as discovery found it."""
        return str(self.raw.get("model") or "")

    @property
    def spent_usd(self) -> float:
        """Real money. Non-zero only for a remote provider's published price."""
        return float((self.raw.get("cost") or {}).get("spent_usd") or 0.0)

    @property
    def avoided_usd(self) -> float:
        """Cloud-equivalent cost NOT spent. 🚨 Never add this to ``spent_usd``."""
        return float((self.raw.get("cost") or {}).get("avoided_usd") or 0.0)

    @property
    def price(self) -> Price:
        return Price.of((self.raw.get("cost") or {}).get("price"))


class Timing(_Wrapped):
    """When things happened, against what was predicted."""

    @property
    def queue_wait_ms(self) -> float:
        return float(self.raw.get("queue_wait_ms") or 0.0)

    @property
    def backend_latency_ms(self) -> float:
        return float(self.raw.get("backend_latency_ms") or 0.0)

    @property
    def ttft_ms(self) -> float | None:
        """Time to first token. ``None`` for a non-streaming call — there is no
        first token to time, and a zero would read as an instantaneous one."""
        val = self.raw.get("ttft_ms")
        return None if val is None else float(val)

    @property
    def total_ms(self) -> float:
        return float(self.raw.get("total_ms") or 0.0)

    @property
    def deadline_s(self) -> float:
        return float(self.raw.get("deadline_s") or 0.0)

    @property
    def deadline_source(self) -> str:
        """``caller`` when you supplied ``deadline_s``, ``computed`` when
        Roadstead chose it from the learned distribution.

        🚨 It is also a behaviour difference, not a label: a deadline Roadstead
        chose is a soft budget the streaming path may extend while tokens are
        demonstrably still arriving, and one you supplied is a hard wall.
        """
        return str(self.raw.get("deadline_source") or "")

    @property
    def predicted_ms(self) -> float | None:
        """What the timeout model expected this call to take. ``None`` when
        there is not enough evidence yet."""
        val = self.raw.get("predicted_ms")
        return None if val is None else float(val)


class Usage(_Wrapped):
    @property
    def input_tokens(self) -> int:
        return int(self.raw.get("input_tokens") or 0)

    @property
    def output_tokens(self) -> int:
        return int(self.raw.get("output_tokens") or 0)

    @property
    def slot_seconds(self) -> float:
        """Backend occupancy time — **the unit of DRR fairness**, and the number
        that explains scheduling in a way token counts cannot."""
        return float(self.raw.get("slot_seconds") or 0.0)


class ChatResult(_Wrapped):
    """One completed enriched call."""

    @property
    def request_id(self) -> str:
        return str(self.raw.get("request_id") or "")

    @property
    def response(self) -> dict:
        """The backend's own OpenAI-shaped body, untouched.

        Nested rather than merged, so a caller never has to tell Roadstead's
        fields from the model's.
        """
        return dict(self.raw.get("response") or {})

    @property
    def content(self) -> str:
        """The first choice's message content, or ``""``. A convenience — read
        ``response`` for anything real (tool calls, multiple choices)."""
        try:
            return self.response["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return ""

    @property
    def attribution(self) -> Attribution:
        return Attribution(self.raw.get("attribution"))

    @property
    def timing(self) -> Timing:
        return Timing(self.raw.get("timing"))

    @property
    def usage(self) -> Usage:
        return Usage(self.raw.get("usage"))


class ModelInfo(_Wrapped):
    """One row of ``GET /rs/v1/models``: what an endpoint is, and what it is
    currently like."""

    @property
    def endpoint(self) -> str:
        return str(self.raw.get("endpoint") or "")

    @property
    def kind(self) -> str:
        return str(self.raw.get("kind") or "")

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset(self.raw.get("capabilities") or ())

    @property
    def context(self) -> int:
        return int(self.raw.get("context") or 0)

    @property
    def aliases(self) -> tuple[str, ...]:
        return tuple(self.raw.get("aliases") or ())

    @property
    def routed(self) -> bool:
        return bool(self.raw.get("routed"))

    @property
    def healthy(self) -> bool:
        return bool(self.raw.get("healthy"))

    @property
    def free_slots(self) -> int:
        return int(self.raw.get("free_slots") or 0)

    @property
    def queued(self) -> int:
        return int(self.raw.get("queued") or 0)

    @property
    def typical_ms(self) -> float | None:
        """Learned median latency. ``None`` = never measured — 🚨 not fast."""
        val = self.raw.get("typical_ms")
        return None if val is None else float(val)

    @property
    def price(self) -> Price:
        return Price.of(self.raw.get("price"))


class Plan(_Wrapped):
    """The answer to "where would this go, and how long should I allow"."""

    @property
    def endpoint(self) -> str:
        return str(self.raw.get("endpoint") or "")

    @property
    def alternatives(self) -> tuple[str, ...]:
        """The runners-up, in the order the router would fall through them."""
        return tuple(self.raw.get("alternatives") or ())

    @property
    def recommended_deadline_s(self) -> float:
        return float((self.raw.get("timing") or {})
                     .get("recommended_deadline_s") or 0.0)

    @property
    def timing(self) -> dict:
        return dict(self.raw.get("timing") or {})

    @property
    def estimated_usd(self) -> float:
        return float((self.raw.get("cost") or {}).get("estimated_usd") or 0.0)

    @property
    def model(self) -> ModelInfo:
        return ModelInfo(self.raw.get("model"))

    @property
    def may_degrade(self) -> bool:
        """Whether a smaller model may answer if this backend goes DOWN."""
        return bool(((self.raw.get("substitution") or {})
                     .get("degrade") or {}).get("allowed"))

    @property
    def may_spill(self) -> bool:
        """Whether paid remote capacity may answer if this backend is FULL.

        🚨 A different question from :attr:`may_degrade` and neither implies the
        other — one is about answer quality, the other about money and
        confidentiality.
        """
        return bool(((self.raw.get("substitution") or {})
                     .get("spill") or {}).get("allowed"))


@dataclass(frozen=True)
class Enrichment:
    """What the ``X-Roadstead-*`` headers on the OPENAI door carry.

    🚨 Deliberately thinner than :class:`Attribution`. Response headers are on
    the wire before the body, so this is what admission had settled — a later
    failover or spill cannot be reflected in it. A caller that needs to know
    what actually served has to use the enriched API; advertising a substitution
    here that a stream might contradict would be worse than advertising nothing.
    """

    request_id: str = ""
    endpoint: str = ""
    deadline_s: float = 0.0
    deadline_source: str = ""

    @property
    def present(self) -> bool:
        """False when the response came from something that is not Roadstead —
        a plain OpenAI-compatible server, or a proxy in front of one."""
        return bool(self.request_id or self.endpoint)


def _headers_get(headers: Any, name: str) -> str:
    """Case-insensitive lookup that works on httpx.Headers, a plain dict, or
    anything else with ``get``."""
    try:
        val = headers.get(name)
    except AttributeError:
        return ""
    if val is None:
        # httpx.Headers is already case-insensitive; a plain dict is not, and a
        # caller passing `dict(response.headers)` is the common shape.
        lowered = {str(k).lower(): v for k, v in dict(headers).items()}
        val = lowered.get(name.lower())
    return "" if val is None else str(val)
