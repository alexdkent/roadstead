"""The Roadstead client SDK — the enriched API, typed, for Python callers.

    from roadstead.client import AsyncRoadsteadClient

    async with AsyncRoadsteadClient("http://proxy:42100", api_key=KEY) as rs:
        result = await rs.chat(
            intent="reasoning",
            messages=[{"role": "user", "content": "why is the queue deep?"}],
        )
        print(result.content)
        print(result.attribution.endpoint, result.timing.queue_wait_ms)

---

## What this is for

Roadstead's OpenAI-compatible door is a drop-in target and needs no SDK — point
any OpenAI client at it. This package is for the **other** door, the one that
carries what an OpenAI shape cannot: declare an *intent* instead of a model,
get a recommended deadline before you call, and be told afterwards what actually
served you, whether that differed from what you asked for, and what it cost.

## Three payload types, one route

``/rs/v1/chat`` is the ENRICHED door, not the chat door: ``chat``, ``embed`` and
``rerank`` all dispatch through it and differ only by the ``payload_type`` on the
envelope. ``call`` sends any other payload type a newer proxy may accept.

🚨 **``embed`` is the LOSSLESS embedding path.** The OpenAI door,
``/v1/embeddings``, must answer in OpenAI's ``{object, data, usage}``, which has
nowhere to put a hybrid embedder's sparse and colbert halves — so it translates
and drops them, deliberately and permanently, because ``/v1/*`` is
OpenAI-compatible and strictly so. Here the backend's body arrives whole under
``result.response``. And ``rerank`` has no OpenAI spelling at all: this is its
only route.

## Three things it does that a hand-rolled ``httpx.post`` would not

🚨 **It classifies errors on the CODE.** ``docs/api.md`` §2.2: the substrings in
Roadstead's error *prose* are what callers historically matched on, which made
rewording a message a breaking API change. ``RoadsteadError.deferrable`` reads
the machine-readable ``code`` first and falls back to the legacy markers only
for a proxy older than this SDK. Moving callers onto this is the migration §2.2
asks for.

🚨 **It sets the client keepalive correctly.** §1.3: the ordering between the
two idle timeouts is load-bearing, and getting it backwards produces
``RemoteProtocolError`` for requests that were never attempted. The SDK sets the
side a client controls rather than leaving it to whoever configures the pool.

🚨 **It keeps unknown fields.** Every typed view exposes ``.raw``, so a proxy
newer than this SDK is usable rather than lossy.

## The boundary

**httpx and the stdlib.** Nothing in here imports the Roadstead server, and it
must stay that way for two reasons: a consumer installing the client should not
be installing Starlette, uvicorn, PyYAML and jsonschema to send an HTTP request;
and a client that reads the server's own constants agrees with the server *by
construction* and can never catch a drift. The contract lives in ``_wire.py`` as
literals transcribed from ``docs/api.md``, and ``tests/test_client_sdk.py``
reads that document back — the same two-ended pin ``tests/wire_contract.py``
uses from the server side.

## Migrating off ``/v1/submit``

That door was removed with Workstream C. The field-by-field map is in
``docs/api.md`` §1.9; the short version::

    agent_id   -> gone. Identity is the API key (or the source address).
    endpoint   -> model= (a pin), or intent= (let Roadstead choose)
    payload    -> payload= (unchanged)
    timeout_s  -> deadline_s= (and usually: omit it)
    priority   -> priority=, or interactive=True/False
"""

from __future__ import annotations

from ._client import (
    AsyncRoadsteadClient,
    RoadsteadClient,
    enrichment_from,
)
from ._errors import AuthError, RoadsteadError, UnroutableError
from ._models import (
    Attribution,
    CallResult,
    ChatResult,
    Enrichment,
    Identity,
    ModelInfo,
    Plan,
    Price,
    Timing,
    Usage,
)
from ._wire import (
    CONTEXT_OVERFLOW_MARKER,
    DEFERRABLE_CODES,
    ERROR_CODES,
    PAYLOAD_CHAT,
    PAYLOAD_EMBEDDING,
    PAYLOAD_RERANK,
    PAYLOAD_TYPES,
    PREFIX,
)

__all__ = [
    # clients
    "AsyncRoadsteadClient",
    "RoadsteadClient",
    "enrichment_from",
    # errors
    "RoadsteadError",
    "AuthError",
    "UnroutableError",
    # typed views
    "Attribution",
    "CallResult",
    "ChatResult",
    "Enrichment",
    "Identity",
    "ModelInfo",
    "Plan",
    "Price",
    "Timing",
    "Usage",
    # contract literals
    "CONTEXT_OVERFLOW_MARKER",
    "DEFERRABLE_CODES",
    "ERROR_CODES",
    "PAYLOAD_CHAT",
    "PAYLOAD_EMBEDDING",
    "PAYLOAD_RERANK",
    "PAYLOAD_TYPES",
    "PREFIX",
]
