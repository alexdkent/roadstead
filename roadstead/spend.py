"""What a call costs, who owes it, and what a threshold does about it.

Pure computation — no I/O, no framework imports. It joins ``scheduler.py``,
``cost_model.py`` and ``timeout_model.py`` as a module whose entire behaviour is
a function of its arguments, and it should stay that way: every clock reading is
passed in, exactly as ``BudgetManager`` takes its ``now``.

Roadmap Workstream D. The descriptor half of this existed before the reader did
— ``ProviderDescriptor.publishes_token_costs`` and OpenRouter's catalogue prices
have been sitting there with nothing looking at them.

---

🚨 **There are TWO kinds of money here and adding them together is the bug this
module exists to make impossible.**

``usage_rates.py`` prices a LOCAL endpoint at what renting the same class of
model would have cost. That is a real, useful number — it is what the local-first
case is *worth* — but it is money **avoided**, and nobody is billed for it. A
remote provider's published price is money **spent**: an invoice arrives.

Both are USD per million tokens, both are floats, and nothing in the type system
would stop one being summed into the other. So a :class:`TokenPrice` carries
which kind it is, :class:`SpendLedger` keeps the two totals in separate fields
that are never added, and a threshold reads **only** the spent one.

That last part is not fastidiousness. A threshold that counted avoided cost
would throttle a caller for using capacity that is free and already paid for —
the exact inversion of what "local-first" means, and it would arrive as a
mysterious latency regression on the machine the user owns.

---

🚨 **Thresholds DEGRADE. They never reject.** Crossing a spend cap costs a
caller two things:

* its **priority band** — one step down, floored at BACKGROUND;
* its access to **paid remote spill**.

It never costs it access to local capacity, and it never turns a request into an
error. Two reasons, both from ``docs/roadmap.md``:

1. **Admission control is about CAPACITY, not billing.** The question the
   scheduler answers is "is there room, and whose turn is it" — a question that
   has the same answer whether or not somebody has spent their allowance. Wiring
   money into it would mean a budget could make a *free local slot* go unused
   while a request waits, which serves nobody.
2. **A misconfigured quota must not be able to take a caller offline.** Somebody
   will typo a cap, and the failure mode of that typo has to be "this caller
   waits behind the others" rather than "this caller is down". A runaway caller
   is already bounded by what is free and by DRR fairness — that is what DRR is
   for.

A degraded caller is still served, still fairly, just last. That is the whole
design.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import LLMPriority

#: What a price is a statement about. The distinction is the module's reason for
#: existing — see the docstring.
SOURCE_PROVIDER = "provider"   # the backend published it. REAL money.
SOURCE_CONFIG = "config"       # an operator declared it. REAL money.
SOURCE_IMPUTED = "imputed"     # usage_rates.py's avoided-cost model. NOT money.

#: Sources that mean an invoice arrives. Membership here — not the endpoint's
#: name, not whether its provider is remote — is what makes spend count against a
#: threshold. A local endpoint an operator has genuinely declared a price for
#: (electricity, an internal chargeback) is real money and is treated as such.
_REAL_SOURCES = frozenset({SOURCE_PROVIDER, SOURCE_CONFIG})

_SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class TokenPrice:
    """USD per MILLION tokens, in and out, plus what kind of money that is.

    Per million rather than per token because that is how every provider
    publishes it and how ``usage_rates.py`` already stores it; converting once
    at the edge beats carrying 1e-7 floats around and rediscovering the
    conversion at each site.
    """

    input_usd_per_mtok: float = 0.0
    output_usd_per_mtok: float = 0.0
    #: One of the ``SOURCE_*`` constants above.
    source: str = SOURCE_IMPUTED
    #: Free-text provenance for the readout ("openrouter pricing", "usage_rates
    #: class tier3"). Never parsed.
    detail: str = ""

    @property
    def real(self) -> bool:
        """True when this price means an invoice. See the module docstring."""
        return self.source in _REAL_SOURCES

    def cost_usd(self, input_tokens: int, output_tokens: int = 0) -> float:
        ti = max(0, int(input_tokens or 0))
        to = max(0, int(output_tokens or 0))
        return (ti / 1e6) * self.input_usd_per_mtok + (to / 1e6) * self.output_usd_per_mtok

    def as_dict(self) -> dict:
        return {
            "input_usd_per_mtok": self.input_usd_per_mtok,
            "output_usd_per_mtok": self.output_usd_per_mtok,
            "source": self.source,
            "real": self.real,
            "detail": self.detail,
        }


class PriceBook:
    """Endpoint class → :class:`TokenPrice`.

    Three ways in, in strict precedence, because they differ in how much they
    know rather than in how recently they were written:

    1. **What the provider published**, via ``discover_capacity`` on a provider
       whose descriptor says ``publishes_token_costs``. Authoritative: it is the
       rate we will actually be billed at, read from the backend that will bill
       us.
    2. **What an operator declared** in the catalog. For a provider that
       publishes nothing, or to override one that publishes something wrong.
    3. **What ``usage_rates.py`` imputes** for a local class. The fallback, and
       the only one that is not money.

    🚨 A discovered price never overwrites a declared one. An operator who wrote
    a number down has said something the catalogue cannot: usually that they are
    on a negotiated rate, or that the published price is not what this deployment
    pays. The reverse precedence would make the declaration unreachable, which is
    the ``_POLICY_PASSTHROUGH`` failure in a different costume — a knob that is
    settable and has no effect.
    """

    def __init__(self) -> None:
        self._declared: dict[str, TokenPrice] = {}
        self._discovered: dict[str, TokenPrice] = {}

    def declare(self, endpoint: str, price: TokenPrice) -> None:
        """Record an operator-declared price. Wins over discovery."""
        self._declared[endpoint] = price

    def observe(self, endpoint: str, price: TokenPrice) -> None:
        """Record a provider-published price. Loses to a declaration."""
        self._discovered[endpoint] = price

    def price(self, endpoint: str) -> TokenPrice:
        """This endpoint's price. Never None — an endpoint nobody priced falls
        back to the imputed table, and an endpoint that table does not know
        prices at zero, which is what it has always done."""
        ep = (endpoint or "").strip()
        if ep in self._declared:
            return self._declared[ep]
        if ep in self._discovered:
            return self._discovered[ep]
        return _imputed_price(ep)

    def cost_usd(self, endpoint: str, input_tokens: int, output_tokens: int = 0) -> float:
        return self.price(endpoint).cost_usd(input_tokens, output_tokens)

    def snapshot(self) -> dict[str, dict]:
        keys = set(self._declared) | set(self._discovered)
        return {ep: self.price(ep).as_dict() for ep in sorted(keys)}


def _imputed_price(endpoint: str) -> TokenPrice:
    """The avoided-cost fallback, read off ``usage_rates.py``.

    Imported lazily and locally so this module keeps no import of the rate table
    at module scope: ``usage_rates`` is data that changes on a different clock
    from this logic, and a deployment that replaces it should not have to think
    about import order.
    """
    from .usage_rates import cloud_rate

    rate_in, rate_out = cloud_rate(endpoint)
    return TokenPrice(
        input_usd_per_mtok=rate_in,
        output_usd_per_mtok=rate_out,
        source=SOURCE_IMPUTED,
        detail=f"usage_rates avoided-cost for {endpoint!r}",
    )


def declared_price(endpoint: str, ep_cfg: object) -> TokenPrice | None:
    """The operator-declared price on an endpoint config, or None.

    Reads ``input_usd_per_mtok`` / ``output_usd_per_mtok`` by ``getattr`` rather
    than typing the parameter as ``EndpointConfig``: this module is imported by
    ``state`` and ``health``, and importing the config dataclass here for one
    attribute lookup would tie a pure-computation module to the shape of the
    routing table.

    🚨 Half a declaration is a whole declaration. An operator who writes only an
    input price has said something true and specific, and filling the other half
    from the imputed table would silently mix a real rate with an avoided-cost
    one inside a single price — which is the exact confusion this module is built
    to prevent. The undeclared half is zero, and the price is real.
    """
    price_in = getattr(ep_cfg, "input_usd_per_mtok", None)
    price_out = getattr(ep_cfg, "output_usd_per_mtok", None)
    if price_in is None and price_out is None:
        return None
    return TokenPrice(
        input_usd_per_mtok=float(price_in or 0.0),
        output_usd_per_mtok=float(price_out or 0.0),
        source=SOURCE_CONFIG,
        detail=f"declared in the catalog for {endpoint!r}",
    )


@dataclass
class AgentSpend:
    """One caller's running account.

    ``spent_usd`` and ``avoided_usd`` are deliberately two fields with no
    combined accessor. Anything that wants a total can add them and will have
    written down that it meant to; nothing here does it by accident.
    """

    agent_id: str
    tokens_in: int = 0
    tokens_out: int = 0
    #: REAL money, lifetime. Only prices whose ``source`` is real land here.
    spent_usd: float = 0.0
    #: Cloud-equivalent cost AVOIDED, lifetime. Never a bill, never a threshold.
    avoided_usd: float = 0.0
    requests: int = 0
    #: The day bucket ``day_spent_usd`` refers to (epoch days, UTC).
    day: int = 0
    #: REAL money spent inside ``day``. This is the number a threshold reads.
    day_spent_usd: float = 0.0
    #: Per-endpoint real spend, lifetime — so "where did it go" is answerable
    #: without a database round trip on the loop.
    by_endpoint: dict[str, float] = field(default_factory=dict)

    def as_dict(self, *, day: int | None = None) -> dict:
        """Serialisable form.

        ``day`` is the CURRENT epoch day. Pass it and ``day_spent_usd`` is
        reported as 0 once the window has rolled over — the same read-side
        roll-over ``SpendLedger.spent_today`` does, and for the same reason: a
        caller that spent yesterday and has not called since would otherwise
        show yesterday's number on a status page as though it were today's.
        """
        stale = day is not None and day != self.day
        return {
            "agent_id": self.agent_id,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "spent_usd": round(self.spent_usd, 6),
            "avoided_usd": round(self.avoided_usd, 6),
            "day_spent_usd": 0.0 if stale else round(self.day_spent_usd, 6),
            "requests": self.requests,
            "by_endpoint": {k: round(v, 6) for k, v in sorted(self.by_endpoint.items())},
        }


class SpendLedger:
    """Per-caller token and cost accounting, live on the loop.

    Single-threaded like everything else in this package's in-memory state —
    ``CLAUDE.md``'s concurrency invariant applies here in full. There is no lock
    because there is one writer, and a second thread touching this would be the
    same data race as a second thread touching the DRR budgets.

    The durable record stays where it already is: ``queue.py`` writes a
    completion row per call and ``/v1/fleet/*`` aggregates it off-loop. This is
    the LIVE view — what a threshold decision needs to be able to read without
    blocking the loop on a query. The two can disagree for as long as a restart
    takes, and that is fine for what it governs: the ledger is what says whether
    a caller is degraded *right now*, not what says what it owes.
    """

    def __init__(self, prices: PriceBook | None = None) -> None:
        self.prices = prices if prices is not None else PriceBook()
        self._agents: dict[str, AgentSpend] = {}

    def __len__(self) -> int:
        return len(self._agents)

    def get(self, agent_id: str) -> AgentSpend | None:
        return self._agents.get(agent_id)

    def charge(
        self,
        agent_id: str,
        endpoint: str,
        input_tokens: int,
        output_tokens: int,
        *,
        now: float | None = None,
        requests: int = 1,
    ) -> float:
        """Record one completion. Returns the REAL USD it cost (0.0 if free).

        Called from the completion path, where the token counts are the ones the
        backend reported rather than the ones we estimated — a cost model that
        billed its own estimate would be marking its own homework.

        ``requests`` is how many completions these totals represent, and exists
        for exactly one caller: startup recovery, replaying a day already on
        disk as one aggregated row per caller-endpoint pair. It is a parameter
        rather than a second `recover()` method so there is ONE implementation
        of the pricing and day-rollover rules — a second copy is what
        `cost_model.context_fit` was created to undo, and that copy had already
        gone silently dead.
        """
        now = time.time() if now is None else now
        acct = self._agents.get(agent_id)
        if acct is None:
            acct = self._agents[agent_id] = AgentSpend(agent_id=agent_id)
        ti = max(0, int(input_tokens or 0))
        to = max(0, int(output_tokens or 0))
        acct.tokens_in += ti
        acct.tokens_out += to
        acct.requests += max(0, int(requests))

        price = self.prices.price(endpoint)
        amount = price.cost_usd(ti, to)
        if not price.real:
            acct.avoided_usd += amount
            return 0.0

        acct.spent_usd += amount
        acct.by_endpoint[endpoint] = acct.by_endpoint.get(endpoint, 0.0) + amount
        day = day_bucket(now)
        if acct.day != day:
            # A new day resets the window. Deliberately lazy rather than swept on
            # a timer: a caller that makes no calls has no spend to roll over,
            # and a sweep would be a second thing to keep in step with the clock.
            acct.day = day
            acct.day_spent_usd = 0.0
        acct.day_spent_usd += amount
        return amount

    def spent_today(self, agent_id: str, *, now: float | None = None) -> float:
        """REAL spend inside the current day bucket. 0.0 for an unknown caller.

        Reads the bucket rather than trusting the stored one, so a caller that
        crossed its cap yesterday and has not called since is not still degraded
        today — the roll-over has to happen on READ as well as on write, or a
        cap becomes permanent for exactly the callers that stopped calling.
        """
        acct = self._agents.get(agent_id)
        if acct is None:
            return 0.0
        now = time.time() if now is None else now
        return acct.day_spent_usd if acct.day == day_bucket(now) else 0.0

    def snapshot(self, *, now: float | None = None) -> list[dict]:
        today = day_bucket(time.time() if now is None else now)
        return [a.as_dict(day=today) for a in sorted(self._agents.values(),
                                                     key=lambda a: a.agent_id)]


def day_bucket(now: float) -> int:
    """Epoch day, UTC.

    A cap is per-day and days are the same length for everyone; a local-timezone
    bucket would move under a DST transition and make one day of the year 23
    hours of allowance.

    Public because the over-cap REPORT is deduplicated per day as well as the
    spend itself, and two modules deriving "which day is it" separately is how
    they come to disagree at a boundary.
    """
    return int(now // _SECONDS_PER_DAY)


def day_start(now: float) -> float:
    """Epoch SECONDS at which the current day bucket began, UTC.

    Derived from :func:`day_bucket` rather than computed alongside it, so the
    startup query that reloads today's spend and the threshold that reads it
    cannot disagree about where a day begins. The two would drift apart at
    exactly one instant a day, which is the hardest possible time to notice.
    """
    return float(day_bucket(now) * _SECONDS_PER_DAY)


@dataclass(frozen=True)
class SpendStanding:
    """Where one caller stands against its cap, and what that costs it."""

    agent_id: str
    spent_today_usd: float
    #: The cap in force, or None when the caller has none (the default).
    cap_usd: float | None = None

    @property
    def over(self) -> bool:
        """True when a cap exists and has been crossed.

        A cap of 0.0 is a real cap meaning "no paid spend at all", not an absent
        one — hence the explicit None check rather than a truthiness test. That
        distinction is the whole difference between a caller that may not spill
        and a caller nobody has configured.
        """
        return self.cap_usd is not None and self.spent_today_usd >= self.cap_usd

    @property
    def may_spill(self) -> bool:
        """Whether this caller may be served by PAID remote capacity."""
        return not self.over

    def effective_priority(self, declared: LLMPriority) -> LLMPriority:
        """The band this caller actually gets.

        🚨 One step down, floored at the lowest band — never a refusal, and never
        further than one step however far over the cap the caller is. A
        proportional penalty would make the degradation unbounded, and an
        unbounded penalty on a billing signal is a rejection wearing a different
        hat. One step is enough to put a caller behind everyone who is inside
        their allowance, which is all this is for.
        """
        if not self.over:
            return declared
        return LLMPriority.coerce(min(int(declared) + 1, int(LLMPriority.P4_HYGIENE)))

    def as_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "spent_today_usd": round(self.spent_today_usd, 6),
            "cap_usd": self.cap_usd,
            "over": self.over,
            "may_spill": self.may_spill,
        }


def standing(
    ledger: SpendLedger,
    agent_id: str,
    cap_usd: float | None,
    *,
    now: float | None = None,
) -> SpendStanding:
    """Assemble one caller's standing. The single place a cap is compared.

    Separate from :class:`SpendLedger` on purpose: the ledger knows what has been
    spent and the agent config knows what the cap is, and the thing that joins
    them is policy. Keeping the join in a function means the ledger can be
    snapshotted, replayed or tested with no notion of a cap at all.
    """
    return SpendStanding(
        agent_id=agent_id,
        spent_today_usd=ledger.spent_today(agent_id, now=now),
        cap_usd=cap_usd,
    )
