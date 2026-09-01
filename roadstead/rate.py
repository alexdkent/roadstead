"""Per-caller request rate: the abuse control DRR is not.

Pure computation — no I/O, no framework imports. State lives in
:class:`RateLedger`; recording and the joins are the caller's job.

🚨 **Why this exists, and why it is not a rate limiter in the usual sense.**

DRR is fairness *under contention*. A caller alone on a quiet fleet is
unthrottled by design, and that is correct: with nobody to be unfair to, there
is nothing for fairness to do. It is also, exactly, why DRR is not an abuse
control. A caller stuck in a loop at three in the morning contends with nobody
and is bounded by nothing until somebody else shows up.

So the gap is real. What closes it is *not* a rejection:

🚨 **Thresholds DEGRADE, they never reject, and there is NO ERROR CODE for one**
(``docs/api.md`` §1.6). A rate limit that answered 429 would be the fourth
spelling of "no" that three workstreams have now declined to mint, and it would
put a misconfigured threshold in a position to take a caller offline — the same
argument that keeps ``daily_spend_usd`` from ever refusing anything. Admission
control is about *capacity*. This is about *manners*.

Crossing a rate threshold therefore costs a caller exactly what crossing a spend
threshold costs it: **one priority band, floored at the lowest, and access to
paid spill.** It never costs local capacity. A runaway caller ends up behind
every caller behaving itself, which is the entire point and is enough.

🚨 **Degradations do not STACK.** A caller over both its spend cap and its rate
threshold drops **one** band, not two. Two independent one-band penalties would
mean that adding a second threshold silently doubled the first one's, and the
unbounded-penalty argument in ``spend.SpendStanding.effective_priority`` applies
with more force to two of them than to one. ``state.ProxyState.demote`` is where
that is enforced, because it is the one place both standings are known.

🚨 **Keyed on the ``agent_id``, not on the key.** Every other quota is, the
budget holder is, and the consequence — a band — is denominated per caller. A
per-key threshold whose penalty landed on the agent's band would punish a team
for one credential's behaviour anyway, so keying it per key would be a
distinction with no difference in the outcome. The answer to *one* credential
misbehaving is to revoke it, which is instant and already exists.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .config import LLMPriority

#: The window a rate is measured over. One minute, because that is the unit an
#: operator writes a threshold in and a shorter one would make a burst of five
#: requests look like a sustained 300/min.
WINDOW_S = 60.0

#: Per-caller cap on remembered timestamps. A caller at 100k requests/min would
#: otherwise hold 100k floats; the window is trimmed on every read anyway, so
#: this only bounds the space between two reads. 🚨 When the cap bites the
#: observed rate is an UNDERCOUNT, which is the safe direction: it can only fail
#: to degrade a caller, never degrade one that was behaving.
_MAX_SAMPLES = 4096


@dataclass
class RateLedger:
    """Sliding-window request counts per caller.

    A deque of timestamps per ``agent_id``, trimmed on access. Deliberately not
    a token bucket: a bucket answers "may this request proceed", which is the
    question this module refuses to be asked. This answers "how fast is this
    caller going", which is a measurement, and the policy is applied elsewhere.
    """

    windows: dict[str, deque] = field(default_factory=dict)

    def record(self, agent_id: str, now: float) -> None:
        """Note one request. Call once per admitted request, on the loop."""
        window = self.windows.get(agent_id)
        if window is None:
            window = self.windows[agent_id] = deque()
        window.append(now)
        if len(window) > _MAX_SAMPLES:
            window.popleft()
        self._trim(window, now)

    def observed(self, agent_id: str, now: float) -> float:
        """Requests per minute over the trailing window.

        Extrapolated from the window rather than from the process lifetime: a
        caller that sent 600 requests in its first second and nothing since is
        not going at 600/min, and treating it as though it were would degrade a
        caller for something it has already stopped doing.
        """
        window = self.windows.get(agent_id)
        if not window:
            return 0.0
        self._trim(window, now)
        return len(window) * (60.0 / WINDOW_S)

    def forget(self, agent_id: str) -> None:
        self.windows.pop(agent_id, None)

    def prune(self, now: float) -> int:
        """Drop callers with nothing in the window. Returns how many went.

        Called from the maintenance tick: without it the dict grows one entry
        per ``agent_id`` ever seen, and an ``agent_id`` is a caller-supplied
        string on the address path.
        """
        stale = [a for a, w in self.windows.items()
                 if not (self._trim(w, now) or w)]
        for agent_id in stale:
            del self.windows[agent_id]
        return len(stale)

    @staticmethod
    def _trim(window: deque, now: float) -> None:
        cutoff = now - WINDOW_S
        while window and window[0] < cutoff:
            window.popleft()

    def snapshot(self, now: float) -> list[dict]:
        return sorted(
            ({"agent_id": a, "observed_per_min": round(self.observed(a, now), 2)}
             for a in list(self.windows)),
            key=lambda row: str(row["agent_id"]),
        )


@dataclass(frozen=True)
class RateStanding:
    """Where one caller stands against its rate threshold, and what it costs.

    Deliberately the same shape as :class:`spend.SpendStanding` — ``over``,
    ``may_spill``, ``effective_priority`` — because the consequence is the same
    consequence. Two threshold types with two different penalties would be two
    things for an operator to learn and a second place for the "never reject"
    rule to be got wrong.
    """

    agent_id: str
    observed_per_min: float
    #: The threshold in force, or None when the caller has none (the default).
    limit_per_min: float | None = None

    @property
    def over(self) -> bool:
        """True when a threshold exists and has been crossed.

        A threshold of 0.0 is a real one meaning "this caller should not be
        sending at all", not an absent one — the same explicit ``None`` check
        and the same reason as ``spend.SpendStanding.over``.
        """
        return (self.limit_per_min is not None
                and self.observed_per_min > self.limit_per_min)

    @property
    def may_spill(self) -> bool:
        """Whether this caller may be served by PAID remote capacity.

        A caller going far too fast is the last one whose overflow should be
        turned into an invoice on somebody else's hardware.
        """
        return not self.over

    def effective_priority(self, declared: LLMPriority) -> LLMPriority:
        """The band this caller actually gets. One step down, floored."""
        if not self.over:
            return declared
        return LLMPriority.coerce(min(int(declared) + 1, int(LLMPriority.P4_HYGIENE)))

    def as_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "observed_per_min": round(self.observed_per_min, 2),
            "limit_per_min": self.limit_per_min,
            "over": self.over,
            "may_spill": self.may_spill,
        }


def standing(
    ledger: RateLedger,
    agent_id: str,
    limit_per_min: float | None,
    *,
    now: float,
) -> RateStanding:
    """Assemble one caller's standing. The single place a threshold is compared.

    Separate from :class:`RateLedger` for the same reason ``spend.standing`` is
    separate from its ledger: the ledger knows the rate, the agent config knows
    the threshold, and the thing that joins them is policy.
    """
    return RateStanding(
        agent_id=agent_id,
        observed_per_min=ledger.observed(agent_id, now),
        limit_per_min=limit_per_min,
    )
