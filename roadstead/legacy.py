"""``POST /v1/submit`` — the door Workstream C removed, restored behind a flag.

🚨 **This is a compatibility surface with an expiry date, not a third north
face.** ``docs/compatibility.md`` allows a deliberate, recorded break; removing
this door was one, and it is still the right decision. What it did not allow for
was a fleet with a dozen callers already speaking the old envelope and no window
in which to move them all at once. So the shape comes back *unchanged* —
byte-for-byte what ``docs/api.md`` §1.9.1 publishes — and it comes back OFF:
without ``ROADSTEAD_LEGACY_SUBMIT`` the route is not registered, so a request
gets the same 404 any unknown path gets. Turning it on is an operator's
decision, exactly as ``ROADSTEAD_ADMIN_UI`` is.

**It is not a second hot path.** Everything below translates into
``Lifecycle.handle_submit`` and varies only how the answer is serialized —
``WIRE_LEGACY`` beside ``WIRE_OPENAI`` and ``WIRE_ENRICHED``. A second admission
path would be a second set of admission bugs, and the reason the ``wire``
parameter exists at all is so a new shape never needs one.

**The one deliberate deviation from §1.5, and its blast radius.** On this door,
a caller inside the built-in internal nets (loopback and docker-internal, as
``acl.IPIdentityMap`` defines them) is identified by the ``agent_id`` in its own
body, unchecked. That is what the old door did, it is what every fleet caller
depends on for its DRR share, and it is the one thing about this envelope that
cannot be reproduced without saying so out loud:

* it applies **only** on this route and **only** under the flag;
* it applies **only** to an address the transport itself vouches for — an
  address that arrived through ``X-Forwarded-For`` is refused it, on the same
  reasoning ``acl.is_admin(trust_builtin_nets=False)`` uses: "already on the
  box" stops being true the moment a front proxy is in the way;
* a caller with a **registered** address keeps its registration, because an
  operator who wrote an ACL entry said who that host is;
* a caller presenting a **key** keeps key semantics — ``may_assert`` decides,
  through ``IdentityResolver.delegate``, exactly as everywhere else;
* and it grants **nothing but a name**. An address still never grants admin, a
  band, a deadline floor or a quota.

**Removing this door again** is gated on one fact and not on a date: that the
inventory of callers still sending this envelope is empty. Until then the
deprecation notice below (one WARNING per ``agent_id`` per UTC day, plus a
counter on ``/v1/status``) is how that inventory gets taken.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

from starlette.responses import JSONResponse, Response

if TYPE_CHECKING:
    from starlette.requests import Request

    from .lifecycle import Lifecycle
    from .scheduler import QueuedRequest
    from .state import ProxyState

logger = logging.getLogger(__name__)

#: The route, at its original spelling. Restoring it anywhere else would be a
#: new door wearing an old name — every caller this exists for has the literal
#: string in its config.
ROUTE = "/v1/submit"

#: The flag. Default OFF, and OFF means the route does not EXIST rather than
#: that it refuses — the posture ``ROADSTEAD_ADMIN_UI`` established, for the
#: same reason: re-opening a surface somebody deliberately closed is an
#: operator's decision, and a door that 403s still tells a scanner it is there.
ENV = "ROADSTEAD_LEGACY_SUBMIT"

#: The third value ``Lifecycle.handle_submit``'s ``wire`` takes. It varies the
#: response bytes and nothing else — no routing, admission, correction or
#: accounting decision reads it.
WIRE_LEGACY = "legacy"

#: How many distinct ``agent_id``s the deprecation notice and the ``/v1/status``
#: counter track before they stop growing. The name is caller-asserted on this
#: door (that is the whole exception above), so its cardinality is not ours to
#: trust — an unbounded map keyed on it is a caller-controlled allocation. Past
#: the cap the notice logs on EVERY call instead of once a day: louder is the
#: right failure for a surface that exists to be migrated off.
_MAX_TRACKED_CALLERS = 256


def enabled() -> bool:
    """Whether ``POST /v1/submit`` is registered at all."""
    return os.environ.get(ENV, "").strip().lower() in ("1", "true", "yes", "on")


def may_self_declare(state: "ProxyState", request: Any) -> bool:
    """Whether this request may name its own ``agent_id`` (the §1.9.2 exception).

    True only for an address-derived caller inside the built-in internal nets,
    on a connection the transport vouches for. A registered address, a forwarded
    address and an unknown one all answer False, and an authenticated caller
    never reaches here — ``IdentityResolver.delegate`` owns that half.

    🚨 The address is resolved through ``IdentityResolver`` and never off
    ``request.client``. A call site that reads the peer directly is one where a
    reverse proxy collapses every caller into one identity; the AST guard in
    ``tests/test_trusted_proxies.py`` fails if a second one appears.
    """
    address = state.identity.client_address(request)
    if address.forwarded:
        return False
    return state.acl.is_internal(address.ip)


# --------------------------------------------------------------------------- #
# Serialization — the published shapes, and nothing else
# --------------------------------------------------------------------------- #

def sync_response(req: "QueuedRequest", result: dict) -> Response:
    """One finished non-streaming request, on the legacy wire (§1.9.3).

    🚨 The ok envelope is **exactly six keys**, plus ``cache_hit`` on a cache
    hit and ``degraded``/``degraded_from`` when a failover served a smaller
    model. Nothing else may be added to it: the fleet's own contract test
    asserts the key SET, so an additive field there is a breaking change while
    the same field on an error envelope is not — which is why the error branch
    below is allowed to carry ``backend_status`` and this one is not.
    """
    if result.get("status") != "ok":
        body = {
            "status": "error",
            "request_id": req.request_id,
            "error": result.get("error", "backend error"),
            "code": result.get("code") or "backend_error",
        }
        # §2.2's permanent-vs-transient fact. Additive on an ERROR envelope,
        # where no caller pins the key set, and the alternative is this door
        # advising a retry the proxy itself declined to make.
        if result.get("backend_status") is not None:
            body["backend_status"] = result["backend_status"]
        return JSONResponse(body, status_code=502)

    body = {
        "status": "ok",
        "request_id": req.request_id,
        "queue_wait_ms": result.get("queue_wait_ms") or 0.0,
        "backend_latency_ms": result.get("backend_latency_ms") or 0.0,
        "estimated_cost_ss": result.get("estimated_cost_ss") or 0.0,
        "response": result.get("response", {}),
    }
    if result.get("degraded"):
        body["degraded"] = True
        body["degraded_from"] = result.get("degraded_from")
    return JSONResponse(body)


def timeout_response(req: "QueuedRequest") -> Response:
    """The 504, exactly as published — and with one field added on purpose.

    🚨 ``error`` is the bare string ``"timeout"``, not the enriched wire's
    ``"proxy timeout after 180s"``. The elapsed number is in the caller's own
    clock, in ``/v1/timeouts`` and in the log line; what a caller of THIS door
    has is a string comparison written years ago. ``status`` is the one
    addition: the published body omitted it, every caller reads
    ``.get("status") != "ok"``, and supplying it can only make more of them
    agree with the rest of the taxonomy.
    """
    return JSONResponse(
        {"status": "error", "request_id": req.request_id,
         "error": "timeout", "code": "proxy_timeout"},
        status_code=504)


def queued_frame(req: "QueuedRequest") -> dict:
    """The opening SSE frame. The enriched wire's ``accepted`` carries
    attribution and timing; this one carries what the old door carried, which is
    the request id and nothing more."""
    return {"type": "queued", "request_id": req.request_id}


# --------------------------------------------------------------------------- #
# The door
# --------------------------------------------------------------------------- #

class LegacySubmitDoor:
    """``POST /v1/submit`` over the shared ProxyState. A translator and a warning."""

    def __init__(self, state: "ProxyState", lifecycle: "Lifecycle") -> None:
        self.state = state
        self.lifecycle = lifecycle
        #: ``agent_id`` → the UTC date its deprecation notice was last logged.
        self._notified: dict[str, str] = {}

    async def handle(self, body: dict, request: "Request") -> Response:
        """Translate the legacy envelope onto the one hot path.

        🚨 The envelope is copied field by field rather than forwarded whole.
        ``Lifecycle.handle_submit`` reads four keys this door never published
        (``declared_agent_id``, ``requested``, ``allow_degrade``,
        ``allow_spill``), and forwarding the caller's dict verbatim would let a
        legacy caller reach them — a door quietly wider than the contract it
        exists to reproduce.
        """
        if not isinstance(body, dict):
            return JSONResponse(
                {"status": "error",
                 "error": "invalid request: the body must be a JSON object",
                 "code": "invalid_request_error"},
                status_code=400)

        declared = str(body.get("agent_id") or "")
        call_site = str(body.get("call_site") or "unknown")
        self._note_use(declared or "unknown", call_site)

        submit: dict = {
            "agent_id": declared,
            # The caller's own word, carried separately because `agent_id`
            # above is what the identity path will resolve INTO — the two
            # differing is the thing `identity.honoured` discloses.
            "declared_agent_id": declared,
            "endpoint": str(body.get("endpoint") or "chat"),
            # Passed through raw: the four-step precedence in
            # `Lifecycle.resolve_declared_priority` treats None as "declared
            # nothing", and every malformed value soft-defaults inside
            # `LLMPriority.coerce` rather than reaching the caller as a 500.
            "priority": body.get("priority"),
            "call_site": call_site,
            "payload_type": str(body.get("payload_type") or "chat_completion"),
            "payload": body.get("payload") or {},
            # Likewise raw. Absent (or an explicit null) means the proxy's own
            # computed default applies, per-identity floor included; anything
            # else is the caller's deadline and wins.
            "timeout_s": body.get("timeout_s"),
            "session_id": body.get("session_id"),
            "turn_id": body.get("turn_id"),
            "caller_id": body.get("caller_id"),
            "request_id": body.get("request_id"),
        }
        return await self.lifecycle.handle_submit(
            submit, request, wire=WIRE_LEGACY)

    def _note_use(self, agent_id: str, call_site: str) -> None:
        """One WARNING per caller per UTC day, and a counter beside it.

        Per DAY rather than per process: a proxy that restarts nightly would
        otherwise report the same inventory every morning, and one that runs for
        a month would report it once and then look migrated. Per CALLER because
        the inventory this is taken for is a list of callers, and a single
        fleet-wide line names none of them.
        """
        tally = self.state.legacy_submits
        tally["count"] = tally.get("count", 0) + 1
        callers = tally.setdefault("callers", {})
        if agent_id in callers or len(callers) < _MAX_TRACKED_CALLERS:
            callers[agent_id] = callers.get(agent_id, 0) + 1

        today = time.strftime("%Y-%m-%d", time.gmtime())
        if self._notified.get(agent_id) == today:
            return
        if len(self._notified) >= _MAX_TRACKED_CALLERS:
            # Over the cap the map is dropped rather than grown. What that costs
            # is a repeat notice for callers whose day-mark went with it, which
            # is the harmless direction — and a caller minting names still gets
            # a line each, because an unseen name has no mark to hit.
            self._notified.clear()
        self._notified[agent_id] = today
        logger.warning(
            "legacy /v1/submit used by agent_id=%s (call_site=%s) — migrate to "
            "/rs/v1/chat (docs/api.md §1.9)", agent_id, call_site)
