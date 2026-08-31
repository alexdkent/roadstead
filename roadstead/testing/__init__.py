"""Test doubles Roadstead ships as part of the product, not as scaffolding.

`FakeBackend` is a programmable inference backend: a real ASGI app on a real
socket that speaks both engine wire shapes and, on command, any of twenty-odd
south-face pathologies. It was `tests/fake_backend.py` until 2026-08-31 and is
promoted here deliberately.

**Why it belongs in the package.** For a gateway whose thesis is capacity-aware
admission control, *"here is a backend that lies about its capacity on demand"*
is a capability, not test furniture. `FAULT_CAPACITY_DESYNC` accepts N
concurrent and 503s beyond while `/props` reports something else entirely —
which is precisely the condition Roadstead exists to handle, and precisely what
nobody can reproduce against a real GPU on demand. Anyone integrating against
Roadstead, or building a gateway of their own, needs that more than they need
another mock.

It is also the honest way to state the boundary: this is what Roadstead assumes
of a backend, executable. `docs/api.md` §4 says the same thing in prose.

**Dependencies.** Starlette and uvicorn only, both already required by the
package — importing this module adds nothing to the dependency set. Nothing in
the library core imports it; it is loaded only when you ask for it by name.

    from roadstead.testing import FakeBackend, FakeBackendServer, FAULT_CAPACITY_DESYNC

    server = FakeBackendServer(FakeBackend(engine="vllm")).start()
    try:
        server.controller.set_fault(FAULT_CAPACITY_DESYNC, 2)
        ...  # point an EndpointConfig at server.host / server.port
    finally:
        server.stop()
"""
from __future__ import annotations

from .fake_backend import (
    ALL_FAULTS,
    FAULT_CAPACITY_DESYNC,
    FAULT_DEGENERATE_LOOP,
    FAULT_EMPTY_COMPLETION,
    FAULT_FINISH_LENGTH,
    FAULT_HTTP_400,
    FAULT_HTTP_500,
    FAULT_HTTP_503,
    FAULT_INTERLEAVED_SSE,
    FAULT_INTERTOKEN_STALL,
    FAULT_MID_STREAM_RESET,
    FAULT_NO_DONE,
    FAULT_NO_USAGE,
    FAULT_NONE,
    FAULT_PARTIAL_SSE,
    FAULT_PHANTOM_TOOL_CALLS,
    FAULT_SCHEMA_INVALID,
    FAULT_SCHEMA_VALID_WRONG,
    FAULT_SLOW_DRAIN,
    FAULT_TIMEOUT,
    FAULT_TRUNCATED_JSON,
    FAULT_TRUNCATED_TOOL_CALLS,
    FAULT_TTFT_STALL,
    OMIT_USAGE,
    STREAM_ONLY_FAULTS,
    USAGE_DEFAULT,
    FakeBackend,
    FakeBackendServer,
    MidStreamReset,
    RecordedRequest,
    make_fake_app,
    mock_transport_handler,
)

__all__ = [
    # the two things most callers need
    "FakeBackend",
    "FakeBackendServer",
    # the app + transport, for callers wiring it themselves
    "make_fake_app",
    "mock_transport_handler",
    # introspection
    "RecordedRequest",
    "MidStreamReset",
    # usage-shape knobs (see FakeBackend.usage_override)
    "USAGE_DEFAULT",
    "OMIT_USAGE",
    # the fault library — ALL_FAULTS is the authoritative list
    "ALL_FAULTS",
    "STREAM_ONLY_FAULTS",
    "FAULT_NONE",
    "FAULT_HTTP_400",
    "FAULT_HTTP_500",
    "FAULT_HTTP_503",
    "FAULT_TRUNCATED_JSON",
    "FAULT_EMPTY_COMPLETION",
    "FAULT_NO_USAGE",
    "FAULT_FINISH_LENGTH",
    "FAULT_DEGENERATE_LOOP",
    "FAULT_SCHEMA_VALID_WRONG",
    "FAULT_SCHEMA_INVALID",
    "FAULT_PHANTOM_TOOL_CALLS",
    "FAULT_TTFT_STALL",
    "FAULT_INTERTOKEN_STALL",
    "FAULT_MID_STREAM_RESET",
    "FAULT_PARTIAL_SSE",
    "FAULT_INTERLEAVED_SSE",
    "FAULT_NO_DONE",
    "FAULT_SLOW_DRAIN",
    "FAULT_TRUNCATED_TOOL_CALLS",
    "FAULT_TIMEOUT",
    "FAULT_CAPACITY_DESYNC",
]
