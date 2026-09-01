"""The fake backend's request log is bounded, and says when it drops.

Found by running `tools/soak.py` for 25 minutes: RSS climbed linearly from
118MB to 1867MB (+72.8 MB/min) over 697,229 requests and never plateaued —
which is precisely the shape that tool's own docstring calls unhealthy. The
growth was `FakeBackend.requests`, a plain list holding a body dict and a header
dict per request for the life of the process.

🚨 **Two separate defects, and the second is the worse one.**

1. `roadstead.testing` is SHIPPED public surface, so an unbounded recorder is a
   real leak for anyone who installs it and drives load through it.
2. `tools/soak.py` exists to detect leaks by RSS slope, and its headline number
   was dominated by its own test double. A measuring instrument was reporting
   its own artifact — and reporting it in the field the tool tells you is "the
   number to look at".

The bound reports itself rather than dropping in silence, which is the same
answer the audit trail gives (`persisted` / `dropped` / `capacity`).

Every guard below was observed going red by mutating the code it guards.
"""
from __future__ import annotations

from collections import deque

import pytest

from roadstead.testing import REQUEST_LOG_CAPACITY, FakeBackend, RecordedRequest
from roadstead.testing.fake_backend import make_fake_app


def _record_n(backend: FakeBackend, n: int) -> None:
    """Append through the same field the app appends through."""
    for i in range(n):
        backend.requests_seen += 1
        backend.requests.append(RecordedRequest(
            method="POST", path="/v1/chat/completions", fault="none",
            body={"i": i}, headers={},
        ))


def test_the_log_is_bounded():
    """Mutation: `default_factory=list`. The deque cap disappears, the log grows
    without bound, and this fails by assertion rather than by running out of
    memory an hour later."""
    b = FakeBackend()
    assert isinstance(b.requests, deque)
    assert b.requests.maxlen == REQUEST_LOG_CAPACITY

    _record_n(b, REQUEST_LOG_CAPACITY * 3)
    assert len(b.requests) == REQUEST_LOG_CAPACITY


def test_the_cap_keeps_the_MOST_RECENT_requests():
    """A test that inspects `requests[-1]` is the overwhelmingly common use —
    eight files here do it — so the end that survives must be the new one.

    🚨 Driven through the REAL app, with the cap shrunk on the instance. The
    first version of this test used the local helper and SURVIVED the mutation
    that matters (`appendleft` in `make_fake_app._record`): the helper appends
    the way the test thinks the app does, so it was asserting this file's model
    of the recorder against itself. `len` stays capped under that mutation and
    the leak stays fixed, so a size-only check passes while every `[-1]`
    assertion in the suite silently starts reading the oldest request.
    """
    from starlette.testclient import TestClient

    b = FakeBackend()
    b.requests = deque(maxlen=3)          # same semantics, cheap to overflow
    with TestClient(make_fake_app(b)) as client:
        for i in range(5):
            client.post("/v1/chat/completions", json={
                "model": "m",
                "messages": [{"role": "user", "content": f"msg-{i}"}]})

    assert len(b.requests) == 3
    assert b.requests[-1].body["messages"][0]["content"] == "msg-4"
    assert b.requests[0].body["messages"][0]["content"] == "msg-2"
    assert b.requests_seen == 5 and b.requests_dropped == 2


def test_it_says_how_many_it_dropped():
    """🚨 Silently truncating is the failure this codebase names most often. A
    caller asserting on `len(requests)` over a long run is entitled to learn the
    list stopped being complete instead of reading a smaller number that means
    something else.

    Mutation: return 0 from `requests_dropped`. The bound still works and the
    leak is still fixed — and the disclosure that makes it honest is gone.
    """
    b = FakeBackend()
    _record_n(b, 10)
    assert (b.requests_seen, b.requests_dropped) == (10, 0)

    _record_n(b, REQUEST_LOG_CAPACITY)
    assert b.requests_seen == REQUEST_LOG_CAPACITY + 10
    assert b.requests_dropped == 10
    assert b.requests_dropped == b.requests_seen - len(b.requests)


def test_reset_clears_the_total_as_well_as_the_records():
    """Mutation: leave `requests_seen` alone in `reset()`. A reused controller
    reports a drop count inherited from the previous test, which is a phantom
    failure in whichever test happens to run next."""
    b = FakeBackend()
    _record_n(b, REQUEST_LOG_CAPACITY + 20)
    b.reset()
    assert (len(b.requests), b.requests_seen, b.requests_dropped) == (0, 0, 0)


def test_the_recorder_still_records_through_the_real_app():
    """The bound must not have cost the feature. Driven through the ASGI app so
    the field the handler actually appends to is the one under test."""
    from starlette.testclient import TestClient

    b = FakeBackend()
    with TestClient(make_fake_app(b)) as client:
        client.post("/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user",
                                                      "content": "hi"}]})
    assert b.requests_seen == 1
    assert b.requests[-1].path == "/v1/chat/completions"
    assert b.requests[-1].body["messages"][0]["content"] == "hi"
    assert b.requests_dropped == 0


def test_the_capacity_is_public():
    """It is shipped surface: a consumer that needs to reason about the bound
    must be able to read it rather than transcribe 1000 from our source."""
    import roadstead.testing as testing

    assert "REQUEST_LOG_CAPACITY" in testing.__all__
    assert testing.REQUEST_LOG_CAPACITY == REQUEST_LOG_CAPACITY
    # Big enough that no test here trips it by accident.
    assert REQUEST_LOG_CAPACITY >= 500
