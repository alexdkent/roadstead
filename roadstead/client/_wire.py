"""The wire contract, restated as literals — the CLIENT side of a two-ended pin.

Everything in this file is transcribed by hand from ``docs/api.md``. That is
deliberate and it is the same arrangement as ``tests/wire_contract.py``, which
does it from the server side: two independent transcriptions of one published
document, each checked against the document rather than against the other.

🚨 **The SDK must not import the server to learn the contract.** It would be the
easiest thing in the world for ``roadstead.client`` to do ``from
roadstead.enriched import ENRICHMENT_HEADERS`` and be permanently, trivially
correct — and completely useless as a check, because a client that reads the
server's constants agrees with the server by construction. It would also make
every consumer install Starlette, uvicorn, PyYAML and jsonschema to send an HTTP
request. So: literals here, and ``tests/test_client_sdk.py`` reads
``docs/api.md`` back and fails when this file no longer says what the contract
says.

**Dependencies: httpx and the stdlib.** Nothing else, ever. This package is what
another project installs to talk to Roadstead, and a client that drags a server
in is not a client.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Routes — docs/api.md §1.7
# ---------------------------------------------------------------------------

#: The enriched API's own version prefix. Separate from ``/v1/*``, which is
#: versioned by OpenAI.
PREFIX = "/rs/v1"

ROUTE_MODELS = f"{PREFIX}/models"
ROUTE_PLAN = f"{PREFIX}/plan"
ROUTE_CHAT = f"{PREFIX}/chat"

#: The OpenAI-compatible door, for ``enrichment_from`` users who are not ready
#: to move.
ROUTE_OPENAI_CHAT = "/v1/chat/completions"


# ---------------------------------------------------------------------------
# Payload types — docs/api.md §1.7.2
# ---------------------------------------------------------------------------

#: What SHAPE the ``payload`` is, and therefore which route on the backend
#: serves it: the chat completion route, the embedder's ``/embed`` or the
#: reranker's ``/rerank``.
#:
#: 🚨 **Not the same field as ``kind``, and neither defaults from the other.**
#: ``kind`` is a ROUTING declaration — what sort of endpoint may serve this, the
#: thing an intent profile also sets. ``payload_type`` is a declaration about the
#: BODY. Sending ``kind: "embed"`` alone routes an embedding request to an
#: embedder and then posts it to that backend's chat route; sending
#: ``payload_type`` alone asks a chat model to answer an embedding body. Both are
#: on the envelope because both questions are real.
PAYLOAD_CHAT = "chat_completion"
PAYLOAD_EMBEDDING = "embedding"
PAYLOAD_RERANK = "rerank"

#: The three the proxy knows today. The SDK does not enforce it — ``call()``
#: sends whatever it is given, so a proxy newer than this SDK is usable rather
#: than gated by a literal transcribed here (the same argument every typed view
#: makes for keeping ``.raw``).
PAYLOAD_TYPES: frozenset[str] = frozenset({
    PAYLOAD_CHAT,
    PAYLOAD_EMBEDDING,
    PAYLOAD_RERANK,
})


# ---------------------------------------------------------------------------
# Error codes — docs/api.md §2.1
# ---------------------------------------------------------------------------

#: Every code the proxy emits. Seventeen (a common under-count is eight, and
#: this file itself under-counted at fifteen until 2026-09-05: the correction
#: layer's two codes reach the wire through a passthrough in ``lifecycle.py``
#: rather than a handler that names them, so no reading of the server's error
#: HANDLERS finds them. ``tests/test_error_codes_published.py`` walks the whole
#: package for assignments instead, which is the only reading that can).
ERROR_CODES: frozenset[str] = frozenset({
    "backpressure",
    "circuit_open",
    "draining",
    "unknown_endpoint",
    "invalid_grammar",
    "proxy_timeout",
    "backend_error",
    "context_overflow",
    "access_denied",
    "invalid_api_key",
    "invalid_messages",
    "invalid_request_error",
    "vision_not_supported",
    "on_demand_unavailable",
    "structured_invalid_json",
    "schema_invalid",
    "toolcall_truncated",
})

#: Codes a caller should DEFER on: retry later, or hand the work to a queue.
#:
#: 🚨 Classified on the CODE, which is what ``docs/api.md`` §2.2 asks a shipped
#: client library to do — the historical classifier matched substrings of the
#: error prose, which made rewording a message a breaking change. The markers
#: below stay as a fallback for exactly as long as it takes callers to migrate;
#: they are not the primary rule any more.
DEFERRABLE_CODES: frozenset[str] = frozenset({
    "backpressure",
    "circuit_open",
    "draining",
    "proxy_timeout",
    "backend_error",
    "on_demand_unavailable",
    # 🚨 A truncation, not a refusal — §2.1's row says raise the output budget
    # and retry, and §2.2 has said so of the same fault under its marker name
    # since before this code existed. It was absent from ERROR_CODES entirely
    # until 2026-09-05, and `deferrable` reads an unknown code as non-deferrable
    # without consulting the prose, so this SDK threw away a tool call the next
    # attempt would have completed.
    "toolcall_truncated",
    # `schema_invalid` is deliberately NOT here: the proxy repaired, then
    # retried once with the validation error fed back, and the model failed the
    # schema anyway. §2.1 — the caller must change the schema, not the clock.
    # Its sibling `structured_invalid_json` is out for the same reason.
})

#: §2.2 — the legacy marker substrings, matched case-insensitively against the
#: message. Kept because a caller upgrading to this SDK may still be pointed at
#: an older proxy, and because §2.2 says a shipped client must keep recognising
#: them until every existing caller has migrated.
DEFERRABLE_MARKERS: tuple[str, ...] = ("backpressure", "circuit open")

#: §2.2 — load-bearing at both ends and quoted VERBATIM. The proxy emits it; a
#: chunking caller matches it to decide whether to re-chunk and retry. Do not
#: reword it on either side.
CONTEXT_OVERFLOW_MARKER = "exceeds the available context size"


# ---------------------------------------------------------------------------
# Enrichment headers on the OpenAI door — docs/api.md §1.8
# ---------------------------------------------------------------------------

HEADER_REQUEST_ID = "X-Roadstead-Request-Id"
HEADER_ENDPOINT = "X-Roadstead-Endpoint"
HEADER_DEADLINE_S = "X-Roadstead-Deadline-S"
HEADER_DEADLINE_SOURCE = "X-Roadstead-Deadline-Source"


# ---------------------------------------------------------------------------
# Keepalive — docs/api.md §1.3
# ---------------------------------------------------------------------------

#: 🚨 The ordering between the two idle timeouts is load-bearing, and this is
#: the side a client controls. The server's is 30s; whichever side expires an
#: idle socket first is the side that closes it cleanly, and if the SERVER wins
#: the race a POST lands on a socket it has already closed and fails as a
#: transport error for a request that was never attempted. The contract requires
#: client < server with at least 5s of margin.
CLIENT_KEEPALIVE_EXPIRY_S = 4.5
