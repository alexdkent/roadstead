"""The BLOCKING client, over a real socket — the half nothing had ever run.

`RoadsteadClient` is exported from `roadstead.client.__all__` and documented as
the way a sync codebase uses the SDK, and before 2026-09-01 not one test
constructed it. The async class had two files' worth of coverage; its wrapper
had none, and the wrapper is where the genuinely hard part lives — a private
event loop on a worker thread, with async generators pumped across it one item
at a time.

Running it found two defects, both on paths a caller reaches by doing the
obvious thing:

* **Abandoning a stream leaked the connection.** `break`-ing out of
  `for frame in client.stream(...)` left the async generator for CPython to
  finalize from the GC, on a thread with no running loop. anyio raised
  `NoEventLoopError` inside `Exception ignored in: <async_generator ...>`, so
  the `async with self._client.stream(...)` body never unwound and the response
  was never closed. Silent — nobody reads "Exception ignored" — and the damage
  is a pooled connection that never comes back.
* **Any call after `close()` hung forever.** `close` stops the worker loop and
  joins its thread; `run_coroutine_threadsafe` then accepts a coroutine that
  will never run, and `.result()` waits on it with no timeout. In the wrapper
  whose module docstring promises it "will tell you so rather than hanging".

🚨 The stream test asserts the CONSEQUENCE, not the absence of the warning.
"No `Exception ignored` on stderr" would pass against any code that swallowed
it while still leaking the socket. The pool is what decides: a released
connection is reused by the next request, a leaked one is still busy and forces
a second.
"""
from __future__ import annotations

import threading
from typing import Iterator

import pytest

from roadstead.client import RoadsteadClient

from .test_client_sdk_socket import _ProxyServer, live_proxy  # noqa: F401

_MSG = [{"role": "user", "content": "count to twenty"}]


@pytest.fixture
def sdk(live_proxy) -> Iterator[RoadsteadClient]:  # noqa: F811
    client = RoadsteadClient(live_proxy.url)
    try:
        yield client
    finally:
        client.close()


def _pool(sdk: RoadsteadClient):
    """The live httpx connection pool behind the wrapper's async client.

    Reaching in deliberately, for the reason `test_client_sdk_socket._pool`
    gives: the thing under test IS the transport's state, and every version of
    this assertion that does not reach in is a tautology. Asserts the attribute
    exists rather than skipping, so an httpx upgrade re-points the guard instead
    of blinding it.
    """
    transport = sdk._async._client._transport
    assert hasattr(transport, "_pool"), (
        "httpx moved the connection pool — re-point this guard, do not delete it")
    return transport._pool


def test_every_shipped_method_works_through_the_blocking_wrapper(sdk):
    """All six, sync, over real TCP. None of them had ever been called."""
    assert {m.endpoint for m in sdk.models()} >= {"tier1", "tier2", "tier3"}
    assert {i["name"] for i in sdk.intents()}

    plan = sdk.plan(intent="reasoning", est_in=2000, est_out=200)
    assert plan.endpoint

    result = sdk.chat(intent="fast-chat", messages=_MSG, max_tokens=16)
    assert result.content and result.attribution.endpoint

    frames = list(sdk.stream(intent="fast-chat", messages=_MSG, max_tokens=16))
    assert "done" in [f.get("type") for f in frames]

    assert "".join(sdk.text_stream(intent="fast-chat", messages=_MSG,
                                   max_tokens=16))


def test_abandoning_a_stream_unwinds_it_on_the_loop_that_owns_it(sdk):
    """🚨 `break` is how streams end in real code, and it used to leak.

    🚨 The witness is `sys.unraisablehook`, and the first version of this test
    used the connection POOL instead — it passed against the bug. Reverting the
    fix reproduced the `NoEventLoopError` in full view and the pool still held
    exactly one connection, because httpx discards a connection it finds broken
    and opens a replacement; the count is identical either way. A guard that
    cannot tell the two apart is not a guard.

    What actually decides is whether the generator unwound on a loop. When it
    does not, the failure is raised inside a finalizer, where CPython prints
    `Exception ignored in: <async_generator ...>` and continues — no exception
    reaches anyone. `sys.unraisablehook` is the one place that becomes a value.
    The pool assertion stays as a second, weaker witness: it would catch a
    unwind that ran but left the connection leased.
    """
    import gc
    import sys

    sdk.models()
    assert len(list(_pool(sdk).connections)) == 1

    unraisable: list = []
    previous = sys.unraisablehook
    sys.unraisablehook = unraisable.append
    try:
        for frame in sdk.stream(intent="fast-chat", messages=_MSG,
                                max_tokens=64):
            assert frame.get("type")
            break               # the ordinary way a caller stops reading
        gc.collect()            # finalize it here, not in some later test
    finally:
        sys.unraisablehook = previous

    assert not unraisable, (
        "the abandoned stream was left for the GC to finalize with no loop to "
        f"run on, so its response was never closed: "
        f"{[u.exc_value for u in unraisable]}")

    sdk.chat(intent="fast-chat", messages=_MSG, max_tokens=16)
    after = list(_pool(sdk).connections)
    assert len(after) == 1, (
        f"the abandoned stream never released its connection: {after}")


def test_closing_with_a_live_stream_unwinds_it_on_the_loop_that_owns_it(live_proxy):  # noqa: F811
    """`close()` while the caller still holds a half-read stream.

    🚨 The generator's own `finally` cannot save this one: by the time the GC
    reaches it the worker loop is stopped, so there is nowhere left to unwind.
    `close` therefore drains the streams it handed out first, while the loop is
    still alive.

    Asserted through `sys.unraisablehook`, because after `close()` the pool is
    gone and the leak has no other witness. A finalizer that fails does not
    raise — CPython prints `Exception ignored in: <async_generator ...>` and
    carries on, which is exactly why this went unnoticed. The hook is the only
    place that failure is a value a test can hold.
    """
    import gc
    import sys

    unraisable: list = []
    previous = sys.unraisablehook
    sys.unraisablehook = unraisable.append
    try:
        client = RoadsteadClient(live_proxy.url)
        frames = client.stream(intent="fast-chat", messages=_MSG, max_tokens=64)
        assert next(frames).get("type")
        client.close()
        del frames
        gc.collect()
    finally:
        sys.unraisablehook = previous

    assert not unraisable, (
        "closing with a live stream left an async generator to be finalized "
        f"with no loop to run on: {[u.exc_value for u in unraisable]}")


def test_a_call_after_close_refuses_instead_of_hanging(live_proxy):  # noqa: F811
    """🚨 Driven from a thread so a REGRESSION FAILS rather than wedging.

    The bug this pins is an unbounded wait on a future nothing will resolve.
    Calling `models()` inline would make a regression hang the whole suite until
    pytest's timeout kills it, which reports as a timeout rather than as this
    test failing — half a guard. The join bounds it here instead.
    """
    client = RoadsteadClient(live_proxy.url)
    client.models()
    client.close()

    box: list = []
    def _call():
        try:
            box.append(("returned", client.models()))
        except BaseException as exc:      # noqa: BLE001 — the point is the type
            box.append(("raised", exc))

    worker = threading.Thread(target=_call, daemon=True)
    worker.start()
    worker.join(timeout=10.0)

    assert not worker.is_alive(), (
        "a call after close() blocked — this is the unbounded "
        "run_coroutine_threadsafe wait on a stopped loop")
    kind, payload = box[0]
    assert kind == "raised", f"expected a refusal, got {payload!r}"
    assert isinstance(payload, RuntimeError), payload
    assert "closed" in str(payload)


def test_close_is_idempotent(live_proxy):  # noqa: F811
    """`with` plus an explicit `close()` is ordinary, not an error."""
    client = RoadsteadClient(live_proxy.url)
    client.models()
    client.close()
    client.close()


async def test_the_blocking_client_refuses_to_be_built_inside_a_running_loop():
    """The documented refusal — and it must arrive CLEAN.

    An `async def` test so there is a real running loop to be refused from.
    The guard used to live inside `_run`, which meant `_make_async(...)` had
    already been evaluated by the time it fired: the caller got the right
    exception trailing `RuntimeWarning: coroutine '_make_async' was never
    awaited`. Promoting that warning to an error pins the refusal as clean,
    not merely correct.
    """
    import gc
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="AsyncRoadsteadClient"):
            RoadsteadClient("http://127.0.0.1:1")
        # 🚨 Recorded and inspected, not promoted to an error. "never awaited"
        # is emitted when the GC destroys the orphaned coroutine, which is after
        # the `raises` block and on no particular thread — `simplefilter(
        # "error")` there raises inside a finalizer, where it becomes an
        # unraisable that pytest downgrades to a warning and the test PASSES.
        # That is the version of this guard that did not bite.
        gc.collect()

    orphans = [w for w in caught if "never awaited" in str(w.message)]
    assert not orphans, (
        "the refusal arrived after `_make_async(...)` had already been built: "
        f"{[str(w.message) for w in orphans]}")
