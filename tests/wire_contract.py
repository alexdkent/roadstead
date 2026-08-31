"""The wire contract, restated as literals — the server side of a two-ended pin.

**What this is for.** Some of Roadstead's guarantees are not checkable from
inside Roadstead. Whether an error is *deferrable* is decided by the CALLER,
which matches substrings of the error message; the classifier historically
lives in the host application's ``framework/nexus_errors.py`` and is not
importable here. Several tests used to import it and use it as an oracle, which
is what coupled them to the monorepo (``tests/_pending/README.md``, Group A).

Importing Roadstead's own copy of the rule instead would be worse than useless:
asserting that one source agrees with itself proves nothing. That is the
tautology trap the quarantine README names.

So the markers live here as literals transcribed from ``docs/api.md`` — the
shared boundary object, which both sides read and neither side owns. Roadstead's
tests pin what it EMITS against them; the monorepo's integration tests pin what
its classifier MATCHES against them. Drift on either side fails on that side.

**The one thing that could rot** is this file itself: a literal transcribed by
hand goes stale the day someone edits the contract without editing the copy.
``test_wire_contract.py`` reads ``docs/api.md`` back and fails if it no longer
says what these constants claim it says.

🚨 Rewording an error message that carries one of these markers is a BREAKING
API change even when the machine-readable ``code`` is untouched.
"""
from __future__ import annotations

from pathlib import Path

#: ``docs/api.md`` §2.2 — "Matched markers include ``circuit open`` and
#: ``backpressure``". Non-exhaustive by the contract's own wording: the client
#: also matches prefixes it constructs itself (``LLM proxy error <status>``),
#: which the proxy never emits and this side therefore cannot assert.
DEFERRABLE_MARKERS = ("backpressure", "circuit open")

#: ``docs/api.md`` §2.2 — "The context-overflow marker is VERBATIM ... Do not
#: reword it." Load-bearing at both ends: the proxy emits it, and a client-side
#: chunker matches it to decide whether to re-chunk and retry.
CONTEXT_OVERFLOW_MARKER = "exceeds the available context size"

#: ``docs/api.md`` §1.3 — the caller's HTTP pool ``keepalive_expiry``, in the
#: host's ``_CLIENT_KEEPALIVE_EXPIRY_S``. Roadstead cannot import it, but its
#: own idle timeout is only correct *relative to* this number.
CLIENT_KEEPALIVE_EXPIRY_S = 4.5

#: ``docs/api.md`` §1.3 — the server's idle timeout must exceed the client's by
#: at least this much: enough to cover clock skew and RTT jitter, not merely a
#: positive difference. A margin of zero is what caused the original race.
KEEPALIVE_MIN_MARGIN_S = 5.0

#: ``docs/api.md`` §1.4 — the floor applied to an endpoint class the timeout
#: model has never heard of. A client that mirrors the floor table rather than
#: asking ``/v1/timeout-advice`` needs this exact number, and a drifted copy
#: makes its honour-a-sub-floor-deadline decision disagree with the server.
TIMEOUT_FALLBACK_FLOOR_S = 60.0

#: ``docs/api.md`` §1.4 — the per-caller-class ceilings on the adaptive
#: recommendation. Published because a client sizing its own retry budget needs
#: to know the recommendation cannot exceed them.
TIMEOUT_CEILING_INTERACTIVE_S = 600.0
TIMEOUT_CEILING_BACKGROUND_S = 1800.0

#: The contract document these constants are transcribed from.
API_DOC = Path(__file__).resolve().parents[1] / "docs" / "api.md"


def carries_deferral_marker(*parts: str) -> bool:
    """True when this envelope text carries a marker a caller defers on.

    Accepts several fields (``error``, ``detail``, …) because an envelope can
    put the human-readable half in any of them, and a caller matching on text
    sees them concatenated.
    """
    text = " ".join(p for p in parts if p).lower()
    return any(marker in text for marker in DEFERRABLE_MARKERS)
