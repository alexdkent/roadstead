"""A straggler's caller gets the proxy's envelope, not a raw 500.

`docs/ledger.md` carried this as "Open, and ours to fix". At
`timeout_graceful_shutdown` uvicorn calls `task.cancel()` on every in-flight
handler; `asyncio.CancelledError` derives from **BaseException**, not
`Exception`, so it sails past both `build_app`'s `exception_handlers` and
Starlette's own `ServerErrorMiddleware`. The caller received
`500 Internal Server Error` — plain text, no `code`, no marker — for a condition
that is retryable and entirely our doing.

Measured before and after with `tools/sigterm_drain_probe.py` S3:

    before  {'status': 500, 'body': 'Internal Server Error',      'elapsed_s': 48.77}
    after   {'status': 503, 'body': '{"code": "draining", ...}',  'elapsed_s': 48.76}

The probe takes ~80s and is off the default path; these run in milliseconds by
cancelling the inner app directly, which is the same exception from the same
place. Every guard was observed going red by mutating the code it guards.
"""
from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path

import pytest

from roadstead import __main__ as entry
from roadstead.__main__ import ShutdownEnvelopeMiddleware


class _Sent(list):
    """Collects ASGI messages, and answers the questions the tests ask."""

    @property
    def start(self) -> dict | None:
        return next((m for m in self if m["type"] == "http.response.start"), None)

    @property
    def body(self) -> bytes:
        return b"".join(m.get("body", b"") for m in self
                        if m["type"] == "http.response.body")


async def _drive(inner, *, scope_type="http") -> tuple[_Sent, BaseException | None]:
    sent = _Sent()

    async def send(message):
        sent.append(message)

    async def receive():                       # pragma: no cover - unused
        return {"type": "http.request"}

    app = ShutdownEnvelopeMiddleware(inner)
    raised: BaseException | None = None
    try:
        await app({"type": scope_type}, receive, send)
    except BaseException as exc:               # noqa: BLE001 — that is the point
        raised = exc
    return sent, raised


# ---------------------------------------------------------------------------
# Nothing sent yet — the common case, and the one the probe measured
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_cancelled_handler_answers_503_draining():
    async def inner(scope, receive, send):
        raise asyncio.CancelledError()

    sent, raised = await _drive(inner)
    # Asserted before indexing. The mutation that matters most here — catching
    # `Exception` instead of `CancelledError`, i.e. the original bug — sends
    # nothing at all, and without this the guard failed with a bare
    # `TypeError: 'NoneType' object is not subscriptable`, which reports a crash
    # rather than the rule.
    assert sent.start is not None, (
        "nothing was sent — the cancellation escaped without an envelope, which "
        "is the raw-500 behaviour this file closes")
    assert sent.start["status"] == 503, "a raw 500 is the bug this file closes"
    envelope = json.loads(sent.body)
    assert envelope["code"] == "draining"
    assert envelope["status"] == "error"


@pytest.mark.asyncio
async def test_the_message_carries_the_deferrable_MARKER():
    """🚨 §2.2: callers classify deferrable-vs-not on a substring of the message,
    not on `code`. A shutdown that reads as a hard failure sends every
    un-migrated client down the wrong branch — and the whole point of answering
    at all is that the caller should retry against the next instance.

    Mutation: drop the word `backpressure`. The status and the code are still
    right and the envelope still looks correct, which is exactly why this is
    asserted separately.
    """
    async def inner(scope, receive, send):
        raise asyncio.CancelledError()

    sent, _ = await _drive(inner)
    assert "backpressure" in json.loads(sent.body)["error"]


@pytest.mark.asyncio
async def test_the_cancellation_is_ALWAYS_re_raised():
    """🚨 The rule that keeps this from becoming a worse bug than the one it
    fixes. Swallowing a cancellation breaks the asyncio contract and leaves
    uvicorn waiting on a task that has declined to die — turning the bounded
    shutdown into the unbounded one `SHUTDOWN_DEADLINE_S` exists to prevent.

    Mutation: `return` instead of `raise`. The envelope is still delivered and
    every other test here still passes.
    """
    async def inner(scope, receive, send):
        raise asyncio.CancelledError()

    _, raised = await _drive(inner)
    assert isinstance(raised, asyncio.CancelledError)


@pytest.mark.asyncio
async def test_a_failure_to_deliver_the_envelope_never_masks_the_cancellation():
    """The send can fail — the transport may already be gone. When it does, the
    caller loses an explanation; if it also lost the cancellation, shutdown
    would hang on this connection.

    Mutation: drop the inner `except BaseException`. The OSError escapes in
    place of the CancelledError.
    """
    async def inner(scope, receive, send):
        raise asyncio.CancelledError()

    async def send(message):
        raise OSError("transport gone")

    async def receive():                       # pragma: no cover
        return {"type": "http.request"}

    app = ShutdownEnvelopeMiddleware(inner)
    with pytest.raises(asyncio.CancelledError):
        await app({"type": "http"}, receive, send)


# ---------------------------------------------------------------------------
# Headers already on the wire
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_cancelled_STREAM_gets_an_error_frame_and_NO_done():
    """🚨 The `[DONE]` rule, in the one place it would be tempting to break.

    `[DONE]` is the backend asserting completeness. This stream is truncated, so
    emitting one would tell the caller a cancelled answer was a finished
    one — the exact collapse `correction.py`'s `finish_reason` repair refuses to
    make, and the reason that repair fires ONLY when the backend sent its own
    `[DONE]`.

    Mutation: append `data: [DONE]`. The caller is told the truncated stream
    completed normally, and nothing else in the suite notices.
    """
    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")]})
        await send({"type": "http.response.body", "body": b"data: {}\n\n",
                    "more_body": True})
        raise asyncio.CancelledError()

    sent, raised = await _drive(inner)
    assert isinstance(raised, asyncio.CancelledError)
    assert sent.start["status"] == 200          # headers were already committed
    tail = sent.body.decode()
    assert "draining" in tail and "backpressure" in tail
    assert "[DONE]" not in tail


@pytest.mark.asyncio
async def test_a_cancelled_NON_stream_mid_body_is_closed_not_left_hanging():
    """Headers are out and it is not a stream, so there is no frame to explain
    in. Close the body rather than leave the caller waiting on a response that
    will never continue.

    Mutation: `more_body: True`. The connection is left open and the caller
    blocks until its own timeout, which is the failure mode a shutdown is
    supposed to avoid.
    """
    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b'{"par', "more_body": True})
        raise asyncio.CancelledError()

    sent, _ = await _drive(inner)
    assert sent[-1]["type"] == "http.response.body"
    assert sent[-1].get("more_body") is False


@pytest.mark.asyncio
async def test_a_normal_response_passes_through_untouched():
    """The middleware is on every request. If it altered a healthy one, it would
    be a far larger bug than the one it fixes."""
    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b'{"ok": true}'})

    sent, raised = await _drive(inner)
    assert raised is None
    assert sent.start["status"] == 200
    assert json.loads(sent.body) == {"ok": True}


@pytest.mark.asyncio
async def test_a_non_http_scope_is_passed_straight_through():
    """`lifespan` runs through the same stack, and its shutdown phase is
    cancelled routinely. Answering an HTTP envelope into a lifespan channel
    would be a protocol violation.

    Mutation: drop the scope check. The lifespan cancellation gets an
    `http.response.start` sent into it.
    """
    seen = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])
        raise asyncio.CancelledError()

    sent, raised = await _drive(inner, scope_type="lifespan")
    assert seen == ["lifespan"]
    assert isinstance(raised, asyncio.CancelledError)
    assert sent == [], "nothing may be written into a non-HTTP scope"


# ---------------------------------------------------------------------------
# It is actually installed
# ---------------------------------------------------------------------------

def test_build_app_installs_it():
    """A source read, because the failure is an ABSENT registration — the app
    behaves identically until the one moment this exists for, which is 48
    seconds into a shutdown and is not on any test's path.

    An AST walk rather than a grep: the class name appears in prose in its own
    docstring and in this file's imports.
    """
    tree = ast.parse(Path(entry.__file__).read_text())
    build = next(n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "build_app")
    names = {n.id for n in ast.walk(build) if isinstance(n, ast.Name)}
    assert "ShutdownEnvelopeMiddleware" in names, (
        "build_app no longer installs ShutdownEnvelopeMiddleware — a cancelled "
        "straggler goes back to a raw 500")


def test_the_code_it_emits_is_one_docs_api_md_publishes():
    """§2.1 is a closed list of fifteen. Minting a sixteenth for this would be a
    new error code for a condition that already has one — `lifecycle` emits
    exactly this triple for work REFUSED while draining, and the caller's
    situation is identical."""
    from roadstead.__main__ import _SHUTDOWN_CANCELLED_BODY

    section = Path("docs/api.md").read_text().split("### 2.1 Codes")[1]
    published = section.split("### 2.2")[0]
    assert f"`{_SHUTDOWN_CANCELLED_BODY['code']}`" in published
