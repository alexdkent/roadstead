"""Progress-governed STREAMING deadlines (2026-08-03).

A streaming call that is actively emitting tokens is not hung, and must not be
killed by a wall clock. Live evidence that it was: agent_id=lan-generic,
endpoint=tier3, layer=stream, P3_INGESTION — three consecutive kills at
elapsed_s=539.999 against applied_timeout_s=540.0 on a 123,466-token prompt,
plus one at elapsed_s=145.078 where 30 + 115077/1000 = 145.08 (the TTFT
watchdog firing during a legitimate prefill).

The semantic under test:

  * a deadline the CALLER supplied (body ``timeout_s`` / ``X-Timeout-S``) is a
    CONTRACT — a hard wall, never extended;
  * a deadline the PROXY chose (``resolve_default_timeout``) is a BUDGET — while
    the stream demonstrably makes progress it may run past it, bounded by an
    absolute hard cap.

So a stream dies when, and only when: no first token within the TTFT allowance,
OR no token for the inter-token gap, OR the absolute hard cap, OR the client
goes away. Each of those four is pinned below, plus the two things that make the
change safe to operate: the extension is COUNTED, and ``req.timeout_deadline``
keeps its old meaning everywhere outside the streaming path.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from roadstead import lifecycle as lifecycle_mod
from roadstead.backend import BackendStreamEvent
from roadstead.config import ProxyConfig
from roadstead.enriched import WIRE_ENRICHED
from roadstead.scheduler import QueuedRequest
from roadstead.service import ProxyService


class _FakeRequest:
    class _Client:
        host = "172.16.0.5"

    client = _Client()
    headers: dict = {}


async def _none(*a, **k):
    return None


def _stub_probes(svc: ProxyService) -> None:
    svc._backend.probe_vllm_capacity = _none
    svc._backend.probe_props = _none
    svc._backend.probe_models = _none


def _body(*, timeout_s: float | None = None, priority: str = "P3_INGESTION"):
    """A streaming submit. Omitting ``timeout_s`` is what makes the deadline
    PROXY-chosen (``deadline_is_default``) — that omission is the whole axis
    these tests turn on, so it is never incidental."""
    body = {
        "agent_id": "a", "endpoint": "tier3", "priority": priority,
        "call_site": "t", "payload_type": "chat_completion",
        "payload": {"messages": [{"role": "user", "content": "x"}], "stream": True},
    }
    if timeout_s is not None:
        body["timeout_s"] = timeout_s
    return body


# _execute_stream floors its budget at max(1.0, deadline - now), so ANY
# sub-second pinned deadline is really 1.0s. Tests that need a stream to
# outlive its budget must therefore beat 1.0s, not the number they pinned —
# otherwise they pass on a margin of milliseconds and flake on a loaded box.
_BUDGET_FLOOR_S = 1.0


def _pin_default_timeout(svc: ProxyService, seconds: float) -> None:
    """Force the PROXY-chosen deadline to ``seconds``.

    Patched at the resolver so the request still travels the real
    ``deadline_is_default`` path in handle_submit (including the sibling
    identity floor); only the NUMBER is pinned, so the test does not depend on
    the adaptive model's live tuning.
    """
    svc._lifecycle.resolve_default_timeout = lambda endpoint, body: seconds


def _chunk_event() -> BackendStreamEvent:
    return BackendStreamEvent(
        "chunk", '{"choices":[{"delta":{"content":"x"}}]}',
        {"choices": [{"delta": {"content": "x"}}]})


def _emit_every(interval_s: float, count: int | None, *, then_done: bool = True):
    """A fake backend stream emitting a token every ``interval_s``.

    ``count=None`` streams forever — the "manifestly progressing, never stops"
    shape that only the absolute hard cap can terminate.
    """
    async def stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        n = 0
        while count is None or n < count:
            await asyncio.sleep(interval_s)
            yield _chunk_event()
            n += 1
        if then_done:
            yield BackendStreamEvent("done", "[DONE]")
    return stream


async def _drain(resp, timeout_s: float = 20.0) -> str:
    frames: list[str] = []

    async def go():
        async for ch in resp.body_iterator:
            frames.append(ch.decode() if isinstance(ch, (bytes, bytearray)) else ch)
    await asyncio.wait_for(go(), timeout=timeout_s)
    return "".join(frames)


def _capture_streaming_requests(svc: ProxyService) -> list[QueuedRequest]:
    """Tap the QueuedRequest handle_submit actually built, without changing what
    it then does with it — the flag under test is set inside handle_submit, so
    the only honest place to read it is on the way past."""
    captured: list[QueuedRequest] = []
    orig = svc._lifecycle.handle_streaming_submit

    async def tap(req, *, wire=WIRE_ENRICHED):
        captured.append(req)
        return await orig(req, wire=wire)

    svc._lifecycle.handle_streaming_submit = tap
    return captured


async def _wait_slot_freed(svc: ProxyService, timeout_s: float = 5.0) -> None:
    async def go():
        while svc._scheduler.endpoint_snapshot("tier3")["in_flight"] > 0:
            await asyncio.sleep(0.02)
    await asyncio.wait_for(go(), timeout=timeout_s)


# --- 1. a progressing stream SURVIVES the proxy-chosen deadline --------------

@pytest.mark.asyncio
async def test_progressing_stream_survives_proxy_chosen_deadline():
    """The core fix. A stream emitting a token every 50ms runs ~2s against a
    proxy-chosen budget of 1s and must COMPLETE — the exact shape that produced
    the elapsed_s=539.999 kills, scaled down.

    Before the fix this aborted at the budget with a 'stream deadline exceeded'
    frame, because both watchdogs were bounded by ``stream_timeout``.
    """
    svc = ProxyService(ProxyConfig())
    svc._backend.stream = _emit_every(0.05, 40)     # ~2.0s of steady progress
    _stub_probes(svc)
    await svc.startup()
    try:
        _pin_default_timeout(svc, 0.3)              # floored to 1.0s
        resp = await svc.handle_submit(_body(), _FakeRequest())
        t0 = time.monotonic()
        out = await _drain(resp)
        elapsed = time.monotonic() - t0

        assert '"done"' in out, out[-400:]
        assert "backpressure" not in out and "deadline exceeded" not in out, out[-400:]
        # It genuinely outlived the (floored) budget rather than finishing
        # inside it — otherwise this would pass without exercising anything.
        assert elapsed > _BUDGET_FLOOR_S * 1.5, (
            f"test did not exercise the extension (ran {elapsed:.2f}s)")
        await _wait_slot_freed(svc)
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_extension_past_the_soft_budget_is_counted():
    """An extension nobody can measure is a capacity sink nobody can revisit.
    The surviving stream above must leave a countable trace (surfaced by
    /v1/timeouts as ``stream_extensions``)."""
    svc = ProxyService(ProxyConfig())
    svc._backend.stream = _emit_every(0.05, 40)     # ~2.0s vs a 1.0s floor
    _stub_probes(svc)
    await svc.startup()
    try:
        _pin_default_timeout(svc, 0.3)
        assert svc._state.stream_deadline_extended == 0
        resp = await svc.handle_submit(_body(), _FakeRequest())
        await _drain(resp)

        assert svc._state.stream_deadline_extended == 1
        # Not merely >0: the seconds granted must be REAL, or an operator
        # reading /v1/timeouts cannot size the capacity this is spending.
        assert svc._state.stream_extension_s_total > 0.5, (
            f"extension seconds not credited "
            f"({svc._state.stream_extension_s_total:.3f}s)")
    finally:
        await svc.shutdown()


# --- 2. the stall guard must NOT regress ------------------------------------

@pytest.mark.asyncio
async def test_mid_stream_stall_still_killed_in_gap_seconds(monkeypatch):
    """Removing the wall clock makes the gap watchdog load-bearing: it is now
    the ONLY thing between a backend that dies mid-answer and the hard cap. A
    stream that emits two tokens then goes silent must still die in ~gap
    seconds, on the SOFT-budget path where the wall clock no longer helps."""
    monkeypatch.setattr(lifecycle_mod, "_STREAM_INTERTOKEN_GAP_S", 0.3)
    svc = ProxyService(ProxyConfig())

    async def stall_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        for _ in range(2):
            await asyncio.sleep(0.05)
            yield _chunk_event()
        await asyncio.sleep(60)          # dies mid-answer, never speaks again
        yield BackendStreamEvent("done", "[DONE]")  # unreachable

    svc._backend.stream = stall_stream
    _stub_probes(svc)
    await svc.startup()
    try:
        # A deliberately GENEROUS budget: if the kill still happens it is the
        # gap watchdog doing it, not a wall clock.
        _pin_default_timeout(svc, 30.0)
        resp = await svc.handle_submit(_body(), _FakeRequest())
        t0 = time.monotonic()
        out = await _drain(resp, timeout_s=10.0)
        elapsed = time.monotonic() - t0

        assert "stalled mid-stream" in out, out[-400:]
        assert elapsed < 5.0, f"stall guard regressed — took {elapsed:.2f}s"
        await _wait_slot_freed(svc)
    finally:
        await svc.shutdown()


# --- 3. the TTFT guard must NOT regress -------------------------------------

@pytest.mark.asyncio
async def test_zero_token_backend_still_killed_in_ttft_seconds(monkeypatch):
    """Same argument for the first token: on the soft-budget path the TTFT
    watchdog is all that stands between a backend that never speaks and the
    hard cap. It must still fire, and free the slot."""
    monkeypatch.setattr(lifecycle_mod, "_STREAM_TTFT_DEADLINE_S", 0.3)
    svc = ProxyService(ProxyConfig())

    async def hang_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        await asyncio.sleep(60)          # never produces a first token
        yield BackendStreamEvent("done", "[DONE]")  # unreachable

    svc._backend.stream = hang_stream
    _stub_probes(svc)
    await svc.startup()
    try:
        _pin_default_timeout(svc, 30.0)
        resp = await svc.handle_submit(_body(), _FakeRequest())
        t0 = time.monotonic()
        out = await _drain(resp, timeout_s=10.0)
        elapsed = time.monotonic() - t0

        assert "ttft timeout" in out, out[-400:]
        assert elapsed < 5.0, f"ttft guard regressed — took {elapsed:.2f}s"
        await _wait_slot_freed(svc)
    finally:
        await svc.shutdown()


def test_ttft_allowance_is_sized_against_measured_prefill_rates():
    """The allowance used to assume 1000 tok/s prefill. tier3 measures 1,426
    tok/s at 213K falling to 729 tok/s at 578K (superlinear in length), and
    est_input_tokens undercounts a tool-heavy payload ~2x on top — which is why
    a legitimate prefill died at elapsed_s=145.078 == 30 + 115077/1000.

    Pin the property, not the constant: the allowance for the request that
    actually died must now comfortably exceed what that prefill really costs at
    the SLOW end of the measured range.
    """
    est_in = 115_077                       # the est_in on the real 145.078s kill
    allowance = (lifecycle_mod._STREAM_TTFT_DEADLINE_S
                 + est_in / lifecycle_mod._STREAM_PREFILL_FLOOR_TOK_S)
    assert allowance > 145.078, "the allowance still kills the observed prefill"
    # est_in undercounts ~2x, and the true rate at that size is at best the
    # 729 tok/s slow end — so the real prefill can cost this much:
    worst_case_prefill_s = (est_in * 2) / 729.0
    assert allowance > worst_case_prefill_s, (
        f"allowance {allowance:.0f}s < worst-case prefill {worst_case_prefill_s:.0f}s")


# --- 4. an EXPLICIT caller deadline stays a hard wall -----------------------

@pytest.mark.asyncio
async def test_explicit_caller_deadline_is_not_extended():
    """The contract boundary. A caller that named its own deadline gets it
    honoured to the letter, however healthily the stream is progressing — the
    soft budget must never leak onto the caller-supplied path.

    This is the guard against the fix over-applying: it fails the moment the
    soft-budget semantic is made unconditional.
    """
    svc = ProxyService(ProxyConfig())
    svc._backend.stream = _emit_every(0.05, None)   # progresses forever
    _stub_probes(svc)
    await svc.startup()
    try:
        resp = await svc.handle_submit(_body(timeout_s=0.5), _FakeRequest())
        t0 = time.monotonic()
        out = await _drain(resp, timeout_s=10.0)
        elapsed = time.monotonic() - t0

        assert "deadline exceeded" in out, out[-400:]
        assert elapsed < 3.0, (
            f"caller deadline was EXTENDED — ran {elapsed:.2f}s past a 0.5s wall")
        assert svc._state.stream_deadline_extended == 0, (
            "a caller deadline must never be counted as an extension")
        await _wait_slot_freed(svc)
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_caller_deadline_request_is_not_flagged_default():
    """The flag itself, at the seam: an explicit body timeout_s must produce a
    request the streaming path treats as a hard wall."""
    svc = ProxyService(ProxyConfig())
    svc._backend.stream = _emit_every(0.01, 2)
    _stub_probes(svc)
    await svc.startup()
    try:
        _pin_default_timeout(svc, 12.0)
        captured = _capture_streaming_requests(svc)

        await _drain(await svc.handle_submit(_body(timeout_s=12.0), _FakeRequest()))
        await _drain(await svc.handle_submit(_body(), _FakeRequest()))

        explicit, implicit = captured
        assert explicit.deadline_is_default is False, (
            "an explicit caller timeout_s must NOT be treated as a proxy default")
        assert implicit.deadline_is_default is True, (
            "an omitted timeout_s must be treated as a proxy-chosen budget")
    finally:
        await svc.shutdown()


# --- 5. the absolute hard cap ------------------------------------------------

@pytest.mark.asyncio
async def test_hard_cap_terminates_an_infinitely_progressing_stream(monkeypatch):
    """Extending on progress means a stream that ALWAYS progresses would hold a
    scarce slot forever. The absolute cap is the only bound that stops it —
    neither watchdog ever fires on a healthy stream."""
    # Above the 1.0s budget floor, so the cap is unambiguously what kills it.
    monkeypatch.setattr(lifecycle_mod, "_STREAM_HARD_CAP_BACKGROUND_S", 1.8)
    svc = ProxyService(ProxyConfig())
    svc._backend.stream = _emit_every(0.05, None)   # progresses forever
    _stub_probes(svc)
    await svc.startup()
    try:
        _pin_default_timeout(svc, 0.3)              # floored to 1.0s
        resp = await svc.handle_submit(_body(), _FakeRequest())
        t0 = time.monotonic()
        out = await _drain(resp, timeout_s=15.0)
        elapsed = time.monotonic() - t0

        assert "absolute" in out and "cap" in out, out[-400:]
        assert svc._state.stream_hard_cap_aborts == 1
        # It ran PAST the 1.0s floored budget (the extension happened) and was
        # stopped at the 1.8s cap — so the cap, not the budget, is the bound.
        assert 1.7 <= elapsed < 8.0, f"cap did not bound the stream ({elapsed:.2f}s)"
        await _wait_slot_freed(svc)
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_hard_cap_is_configurable_from_models_yaml(monkeypatch):
    """The cap is real config on the models.yaml → model_catalog → state path
    the timeout floors/ceilings already use, not a bare literal. A per-class
    ``stream_hard_cap_s`` must win over the band default."""
    monkeypatch.setattr(lifecycle_mod, "_STREAM_HARD_CAP_BACKGROUND_S", 3600.0)
    svc = ProxyService(ProxyConfig())
    _stub_probes(svc)
    await svc.startup()
    try:
        req = QueuedRequest.create(
            agent_id="a", endpoint="tier3", priority="P3_INGESTION",
            call_site="t", payload_type="chat_completion",
            payload={"messages": [{"role": "user", "content": "x"}], "stream": True},
            timeout_s=30.0, deadline_is_default=True,
        )
        assert svc._lifecycle._stream_hard_cap_s(req) == 3600.0
        svc._state.stream_hard_caps["tier3"] = 1234.0
        assert svc._lifecycle._stream_hard_cap_s(req) == 1234.0
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_interactive_band_gets_the_tighter_cap():
    """A user-facing turn still streaming after 15 minutes has already failed,
    whatever the token rate says — interactive must not inherit the background
    hour."""
    svc = ProxyService(ProxyConfig())
    _stub_probes(svc)
    await svc.startup()
    try:
        def _req(priority):
            return QueuedRequest.create(
                agent_id="a", endpoint="tier3", priority=priority,
                call_site="t", payload_type="chat_completion",
                payload={"messages": [{"role": "user", "content": "x"}],
                         "stream": True},
                timeout_s=30.0, deadline_is_default=True)

        interactive = svc._lifecycle._stream_hard_cap_s(_req("P1_TURN_SUPPORT"))
        background = svc._lifecycle._stream_hard_cap_s(_req("P3_INGESTION"))
        assert interactive < background
        # Background must clear the measured worst case: a 578,400-token tier3
        # request is 794s end-to-end (models.yaml, tier3 stanza).
        assert background > 794.0 * 2
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_hard_cap_never_shortens_an_already_longer_deadline(monkeypatch):
    """The cap only ever EXTENDS. A proxy-chosen deadline that is already more
    generous than the cap (a big adaptive one) must not be clipped down to it —
    that would turn a fix into a regression for the longest calls."""
    monkeypatch.setattr(lifecycle_mod, "_STREAM_HARD_CAP_BACKGROUND_S", 0.5)
    svc = ProxyService(ProxyConfig())
    svc._backend.stream = _emit_every(0.05, 20)
    _stub_probes(svc)
    await svc.startup()
    try:
        _pin_default_timeout(svc, 30.0)   # already far above the 0.5s cap
        out = await _drain(await svc.handle_submit(_body(), _FakeRequest()))
        assert '"done"' in out, out[-400:]
        assert svc._state.stream_hard_cap_aborts == 0
    finally:
        await svc.shutdown()


# --- 6. client disconnect aborts the producer and frees the slot ------------

@pytest.mark.asyncio
async def test_client_disconnect_aborts_producer_and_frees_slot():
    """Load-bearing in a way it was not before. The wall clock used to
    incidentally bound how long the proxy held a slot for a client that had
    hung up; with the deadline extended on progress it no longer does, so the
    disconnect path IS the bound.

    Drives the real ASGI ``__call__`` with an ``http.disconnect`` message —
    not ``aclose()`` — because that is what uvicorn actually delivers, and it
    is the branch (Starlette's ``listen_for_disconnect`` task group) that has
    to cancel the producer.
    """
    svc = ProxyService(ProxyConfig())
    svc._backend.stream = _emit_every(0.02, None)   # would stream forever
    _stub_probes(svc)
    await svc.startup()
    try:
        # Soft budget with a huge cap: nothing but the disconnect can stop this.
        _pin_default_timeout(svc, 3000.0)
        resp = await svc.handle_submit(_body(), _FakeRequest())

        seen_chunk = asyncio.Event()
        bodies: list[bytes] = []

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                bodies.append(message["body"])
                if b"delta" in message["body"]:
                    seen_chunk.set()

        async def receive():
            # Client hangs up once tokens are genuinely flowing.
            await seen_chunk.wait()
            return {"type": "http.disconnect"}

        scope = {
            "type": "http", "method": "POST", "path": "/v1/submit",
            # uvicorn's HTTP protocols advertise spec_version 2.3, which is the
            # branch that runs listen_for_disconnect. Pinned so this test can
            # never silently exercise the OTHER branch.
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "headers": [],
        }
        await asyncio.wait_for(resp(scope, receive, send), timeout=10.0)

        assert seen_chunk.is_set(), "never got a chunk — test proved nothing"
        # The producer was cancelled and its slot reclaimed, despite the stream
        # still having 'work' to do and no deadline anywhere near expiring.
        await _wait_slot_freed(svc)
        assert svc._scheduler.endpoint_snapshot("tier3")["in_flight"] == 0
        assert svc._slot_leak_reclaimed >= 1
    finally:
        await svc.shutdown()


# --- 7. the soft budget is scoped to the streaming path only ----------------

@pytest.mark.asyncio
async def test_soft_budget_does_not_mutate_timeout_deadline():
    """``req.timeout_deadline`` is also read by the in-proxy retry budget, the
    sync dispatch bound and admission. The streaming extension must be local to
    the streaming path — if it moved the deadline itself, it would silently
    widen all three."""
    svc = ProxyService(ProxyConfig())
    svc._backend.stream = _emit_every(0.05, 40)     # ~2.0s vs a 1.0s floor
    _stub_probes(svc)
    await svc.startup()
    try:
        _pin_default_timeout(svc, 0.3)
        captured = _capture_streaming_requests(svc)
        out = await _drain(await svc.handle_submit(_body(), _FakeRequest()))
        assert '"done"' in out, out[-400:]           # it DID get extended

        req = captured[0]
        # The deadline is still exactly what admission and the retry budget were
        # told it was — the extension lives in the streaming path's own local
        # bound, never in the shared field.
        assert req.timeout_s == 0.3
        assert req.timeout_deadline == pytest.approx(req.enqueued_at + 0.3), (
            "the streaming path moved the shared deadline")
    finally:
        await svc.shutdown()
