"""Operator-granted "don't store my content" — and what it still records.

Roadstead keeps the prompts and responses it serves (``proxy_completions``'s
``payload_json``/``response_json`` for ``ROADSTEAD_PAYLOAD_RETENTION_S``, the
restart WAL, a few log lines). That is the right default: it is what the replay
harness, the health sweeps and an operator debugging a bad answer read. A few
callers carry content that must not sit in a database at all, and this module is
the one door through which a request can opt out.

The request signal is a header, ``X-Roadstead-No-Store: content``, on any door
(the OpenAI-compatible one cannot carry a body field, and one signal that works
everywhere beats two that differ). The header alone grants nothing:

🚨 **A request ASKS; the operator GRANTS.** Honoured only when ALL of these hold,
and each is a refusal reason of its own so an operator can tell them apart:

* ``not_authenticated`` — the caller's identity was established by an API key.
  An address registration is a weak hint about a HOST (``acl.py``), shared by
  everything on it; granting a content privilege to whatever happens to answer
  at an address is the self-asserted ``agent_id`` bug in another costume.
* ``shared_identity`` — the resolved ``agent_id`` is not one an address
  registration also maps to (the built-in ``internal`` identity included). A key
  minted FOR a shared label would let every holder of that label opt out.
* ``not_granted`` — ``agents.yaml`` carries a valid grant for that ``agent_id``
  (``config.load_agent_configs``: ``content_no_store: allowed`` plus
  ``approved_by: operator`` and an ISO ``approved_on``). Absent ⇒ off, and the
  grant is FILE-ONLY: the admin plane's PATCH refuses the field, because a
  credential that can edit quotas must not be able to edit this.

A refused attempt is never silent (:func:`note_refused`): the request is stored
IN FULL, a counter increments, a WARNING names the caller and the reason, the
completion row says ``refused:<reason>`` (``/v1/status`` counters reset at a
restart and ``docker logs`` does not survive a redeploy — the row does), and the
response says so in ``X-Roadstead-No-Store``.

What no-store DROPS is content: the request messages / tool definitions / tool
results and the response text, from the completion row, the restart WAL, the
response cache, the reasoning-replay cache, the prefix keep-alive capture, the
shadow comparison, and every log line that would have quoted them
(:func:`redact`). What it KEEPS is everything else: identity, endpoint, timings,
token counts, status, ``finish_reason``, sizes of what was dropped, message
count and roles, tool NAMES, and a flag that it was honoured
(:func:`content_summary`).

🚨 **No content hash.** A hash would give cross-request correlation, and both
available kinds are wrong here. A plain digest of a short, low-entropy prompt is
reversible by a dictionary (the caller's whole vocabulary is a few hundred
phrases); a keyed one needs a key that either lives beside the database it is
meant to protect or dies at every restart, which makes it useless for the one
thing correlation is for. ``request_id``, ``session_id``, ``turn_id`` and
``call_site`` already correlate every row a caller can usefully correlate, and
the sizes tell an operator when two requests were the same shape.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

#: The request signal. One header, on every door.
HEADER = "X-Roadstead-No-Store"

#: The only value the header takes today. A scope word rather than a boolean so
#: a later "metadata too" can be added without redefining ``true``.
SCOPE_CONTENT = "content"

#: Values of ``QueuedRequest.no_store_outcome`` / ``proxy_completions.no_store``
#: and of the response header. ``""``/NULL = the caller asked for nothing.
OUTCOME_HONOURED = "honoured"
OUTCOME_REFUSED_PREFIX = "refused:"

REASON_NOT_AUTHENTICATED = "not_authenticated"
REASON_SHARED_IDENTITY = "shared_identity"
REASON_NOT_GRANTED = "not_granted"

#: What a redacted log field says instead of the content.
WITHHELD = "<withheld: no-store>"

#: The grant spellings, shared with ``config.load_agent_configs``.
GRANT_VALUE = "allowed"
APPROVER = "operator"


class InvalidNoStoreHeader(ValueError):
    """The header was present with a value this version does not define."""


def read_header(request: Any) -> str | None:
    """The header's raw value off a Starlette request OR a plain-dict test double
    (``identity._header``'s tolerance: most of this suite's doubles carry a bare
    dict, which is case-sensitive)."""
    headers = getattr(request, "headers", None)
    if headers is None:
        return None
    for spelling in (HEADER, HEADER.lower(), HEADER.upper()):
        try:
            value = headers.get(spelling)
        except Exception:  # noqa: BLE001 — a double with a hostile .get
            return None
        if value is not None:
            return str(value)
    return None


def parse_header(raw: str | None) -> bool:
    """Whether the request asked for no-store. Raises on a value we don't define.

    🚨 An unrecognised value is a 400, not a shrug. The failure the other
    reading produces — the caller wrote ``true``, was ignored, and believes its
    content is not being kept — is the exact silent outcome this feature exists
    to rule out.
    """
    if raw is None:
        return False
    value = raw.strip().lower()
    if not value:
        return False
    if value == SCOPE_CONTENT:
        return True
    raise InvalidNoStoreHeader(
        f"{HEADER} must be {SCOPE_CONTENT!r} (got {raw[:40]!r}). The header is "
        f"a request, not a switch: it only takes effect for a caller its "
        f"operator has granted, and a value this proxy does not define is "
        f"refused rather than ignored so nobody believes they opted out.")


@dataclass(frozen=True)
class Decision:
    """The outcome for one request. ``outcome`` is ``""`` when nothing was asked."""

    outcome: str = ""
    reason: str = ""

    @property
    def requested(self) -> bool:
        return bool(self.outcome)

    @property
    def honoured(self) -> bool:
        return self.outcome == OUTCOME_HONOURED


NOT_REQUESTED = Decision()


def decide(
    requested: bool, *, authenticated: bool, agent_id: str,
    granted: bool, shared_identities: "frozenset[str] | set[str]",
) -> Decision:
    """Pure policy: the three refusal reasons, in the order an operator fixes them."""
    if not requested:
        return NOT_REQUESTED
    if not authenticated:
        return _refused(REASON_NOT_AUTHENTICATED)
    if agent_id in shared_identities:
        return _refused(REASON_SHARED_IDENTITY)
    if not granted:
        return _refused(REASON_NOT_GRANTED)
    return Decision(OUTCOME_HONOURED)


def _refused(reason: str) -> Decision:
    return Decision(OUTCOME_REFUSED_PREFIX + reason, reason)


def outcome_of(req: Any) -> str:
    """``req``'s no-store outcome, ``""`` for anything that carries none.

    The one reader the stores and log sites use, so a request-shaped stub (a
    test double, a recovered WAL row) is "never asked" rather than an
    ``AttributeError`` in the middle of a completion — and so a truthy mock
    attribute can never read as a grant.
    """
    value = getattr(req, "no_store_outcome", "")
    return value if isinstance(value, str) else ""


def engaged(req: Any) -> bool:
    """True when ``req``'s content must reach no store and no log line."""
    return outcome_of(req) == OUTCOME_HONOURED


def redact(no_store: bool, text: Any) -> Any:
    """``text`` unless the request is no-store, in which case :data:`WITHHELD`.

    For a log field that may quote content: a backend error body, a validator's
    message about the offending value, an excerpt of the response. The caller
    that holds the request passes ``req.no_store``.
    """
    return WITHHELD if no_store else text


# ---------------------------------------------------------------------------
# What is still recorded
# ---------------------------------------------------------------------------

#: Roles are caller-supplied strings; only the protocol's own survive, so a role
#: field cannot be used to smuggle text past the drop.
_ROLES = frozenset({"system", "developer", "user", "assistant", "tool", "function"})

_NAME_OK = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_MODEL_OK = re.compile(r"^[A-Za-z0-9_.:/@+-]{1,128}$")
_MAX_NAMES = 64


def _clean_name(name: Any) -> str:
    return name if isinstance(name, str) and _NAME_OK.match(name) else "other"


def _names(items: Any, *, path: tuple[str, ...]) -> list[str]:
    """Sorted unique function names out of a ``tools`` / ``tool_calls`` list."""
    out: set[str] = set()
    for item in items if isinstance(items, list) else []:
        node: Any = item
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        if node is not None:
            out.add(_clean_name(node))
    return sorted(out)[:_MAX_NAMES]


def _size(obj: Any) -> int:
    try:
        return len(json.dumps(obj, separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError):
        return 0


def content_summary(payload: dict | None, response: dict | None) -> dict:
    """The shape of a request/response pair, with none of its content.

    Sizes are of what was DROPPED (the JSON the completion row would have
    held), so an operator can still see that a call was huge. Computed here, at
    the one choke point, from the same dicts that would have been stored.
    """
    out: dict[str, Any] = {"v": 1}
    if isinstance(payload, dict):
        out["payload_bytes"] = _size(payload)
        messages = payload.get("messages")
        if isinstance(messages, list):
            roles: dict[str, int] = {}
            for m in messages:
                role = m.get("role") if isinstance(m, dict) else None
                key = role if role in _ROLES else "other"
                roles[key] = roles.get(key, 0) + 1
            out["message_count"] = len(messages)
            out["roles"] = roles
        tools = _names(payload.get("tools"), path=("function", "name"))
        if tools:
            out["tools_declared"] = tools
        for key in ("input", "texts"):
            seq = payload.get(key)
            if isinstance(seq, list):
                out["input_count"] = len(seq)
    if isinstance(response, dict):
        # The model the BACKEND says served (a streamed row's whole captured
        # "response" is this provenance pair, `Lifecycle._stream_model_body`) —
        # a fact about the backend, not about the caller's content.
        for key in ("model", "model_source"):
            val = response.get(key)
            if isinstance(val, str) and _MODEL_OK.match(val):
                out[key] = val
        choices = response.get("choices")
        if isinstance(choices, list):
            out["response_bytes"] = _size(response)
            out["choice_count"] = len(choices)
            called: set[str] = set()
            for ch in choices:
                msg = ch.get("message") if isinstance(ch, dict) else None
                calls = msg.get("tool_calls") if isinstance(msg, dict) else None
                called.update(_names(calls, path=("function", "name")))
            if called:
                out["tool_calls"] = sorted(called)[:_MAX_NAMES]
    return out
