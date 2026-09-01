"""Typed errors, and the deferrable classifier ``docs/api.md`` §2.2 asks for.

The single most important thing a caller does with a Roadstead error is decide
whether to **retry it later** or **fix something and stop**. Historically that
decision lived in the caller and matched substrings of the error prose, which is
why §2.2 says rewording an error message is a breaking API change even when the
machine-readable ``code`` is untouched.

This module is the migration off that. It classifies on ``code`` first, falls
back to the legacy markers, and gives a caller one property to read.
"""

from __future__ import annotations

from ._wire import (
    CONTEXT_OVERFLOW_MARKER,
    DEFERRABLE_CODES,
    DEFERRABLE_MARKERS,
)


class RoadsteadError(Exception):
    """A refusal from the proxy, with its taxonomy attached.

    🚨 Raised for a proxy-level refusal only. A *model* that answers badly is a
    successful call and comes back as a result, not an exception — a client that
    raised on both would make "the gateway is full" and "the model was unhelpful"
    the same event, and no caller can react to those the same way.
    """

    def __init__(self, message: str, *, code: str = "", status: int = 0,
                 request_id: str = "", body: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        #: The machine-readable code from ``docs/api.md`` §2.1. Empty when the
        #: proxy could not be reached at all, which is a different situation and
        #: deliberately not given a code we invented.
        self.code = code
        self.status = status
        self.request_id = request_id
        self.body = body or {}

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return (f"RoadsteadError(code={self.code!r}, status={self.status}, "
                f"message={self.message!r})")

    @property
    def deferrable(self) -> bool:
        """Whether retrying later is the right response.

        Code first (§2.2's ask of a shipped client), prose second. The prose
        fallback matters for one real case: this SDK pointed at an older proxy
        whose taxonomy predates a code we now know. Dropping it would silently
        turn a deferrable backpressure error into a hard failure on exactly the
        deployment least able to absorb one.
        """
        if self.code:
            return self.code in DEFERRABLE_CODES
        low = self.message.lower()
        return any(m in low for m in DEFERRABLE_MARKERS)

    @property
    def context_overflow(self) -> bool:
        """Whether the prompt did not fit, so re-chunking is the fix.

        Matches the verbatim marker as well as the code, because the same
        condition arrives two ways: Roadstead's own pre-admission context gate
        (code ``context_overflow``) and a backend's own overflow error relayed
        through, which carries the marker inside a ``backend_error``.
        """
        return (self.code == "context_overflow"
                or CONTEXT_OVERFLOW_MARKER in self.message)


class AuthError(RoadsteadError):
    """401/403. 🚨 Two codes, and they are NOT interchangeable — see §2.1.

    ``invalid_api_key`` means *this credential is wrong*; ``access_denied``
    means *this source is not enrolled*. They send an operator to different
    files, so they get one class with a readable ``code`` rather than being
    collapsed into one meaning.
    """


class UnroutableError(RoadsteadError):
    """No endpoint can serve this request: an unknown pin, or an intent nothing
    in the fleet satisfies. Deterministic, so never deferrable — retrying will
    get the same answer until the catalog or the request changes.

    ``considered`` carries the near-misses the proxy weighed, which is what
    turns "no endpoint matched" into something a caller can act on.
    """

    @property
    def considered(self) -> list[dict]:
        return list(self.body.get("considered") or [])
