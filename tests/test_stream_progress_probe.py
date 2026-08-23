"""C6 — the inter-token watchdog asks the BACKEND before it kills a stream (2026-08-23).

`_STREAM_INTERTOKEN_GAP_S` observes only its own wire, and a silent wire does
not mean a dead backend. Two measured causes produced the same silence on a
demonstrably healthy tier3:

  * JIT — the engine compiles kernels mid-request on CPU, GPU at 0%, no token
    emitted. Measured during the incident: +1,972 prompt and +12 generation
    tokens in the 30 s the client saw nothing.
  * TOOL-CALL BATCHING — vLLM withholds argument deltas until it can emit a
    complete tool_call. Measured live, same 2,500-token generation, idle engine:
    530 frames / 0.09 s max gap without tools vs 29 frames / 14.23 s WITH them.

Both killed work that would have finished: 17 stalls in 24 h, every one of them
a tool-caller, arriving in bursts of the same prompt retried, while the engine
generated 16-68 tok/s throughout.

So the watchdog now has a discriminator — the backend's own cumulative token
counters — and this file pins the three outcomes that make it safe:

  advanced → extend (BOUNDED) · frozen → abort exactly as before · absent →
  abort exactly as before.

The middle one is the one that matters most. A watchdog that no longer fires on
a real wedge is worse than the bug it fixed, so `test_frozen_counters_*` and
`test_extensions_are_bounded` are the load-bearing tests here, not the happy
path — they are what stops this change from being a way to hang forever.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from originfleet.llmproxy import lifecycle as lifecycle_mod
from originfleet.llmproxy.backend import BackendStreamEvent
from originfleet.llmproxy.config import ProxyConfig
from originfleet.llmproxy.service import ProxyService


async def _none(*a, **k):
    return None


def _stub_probes(svc: ProxyService) -> None:
    svc._backend.probe_vllm_capacity = _none
    svc._backend.probe_props = _none
    svc._backend.probe_models = _none


def _body(*, tools: bool = False, priority: str = "P3_INGESTION"):
    """A streaming submit with a PROXY-chosen deadline (no ``timeout_s``)."""
    payload = {"messages": [{"role": "user", "content": "x"}], "stream": True}
    if tools:
        payload["tools"] = [{
            "type": "function",
            "function": {"name": "write_file", "parameters": {"type": "object"}},
        }]
    return {
        "agent_id": "a", "endpoint": "llama-thinker", "priority": priority,
        "call_site": "t", "payload_type": "chat_completion", "payload": payload,
    }


class _FakeRequest:
    class _Client:
        host = "172.16.0.5"

    client = _Client()
    headers: dict = {}


def _pin_default_timeout(svc: ProxyService, seconds: float) -> None:
    svc._lifecycle.resolve_default_timeout = lambda endpoint, body: seconds


def _chunk_event() -> BackendStreamEvent:
    return BackendStreamEvent(
        "chunk", '{"choices":[{"delta":{"content":"x"}}]}',
        {"choices": [{"delta": {"content": "x"}}]})


def _speaks_then_silent(silence_s: float, *, then_done: bool = True):
    """Two tokens, then a silence — the exact shape of every stall we saw.

    ``silence_s`` is what the watchdog has to make a judgement about; whether it
    survives is decided ONLY by what the counters say, which is the point.
    """
    async def stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        for _ in range(2):
            await asyncio.sleep(0.05)
            yield _chunk_event()
        await asyncio.sleep(silence_s)
        if then_done:
            yield _chunk_event()
            yield BackendStreamEvent("done", "[DONE]")
    return stream


def _counters(svc: ProxyService, mode: str) -> list[int]:
    """Install a fake `/metrics` progress scrape. Returns the call log so a test
    can assert the probe actually RAN — a probe that never fired would make
    every assertion below pass for the wrong reason."""
    calls: list[int] = []

    async def probe(ep_cfg):
        calls.append(1)
        if mode == "absent":
            return None
        if mode == "frozen":
            return {"prompt": 1000, "generation": 500}
        # advancing: generation climbs, which is the "alive and working" shape
        return {"prompt": 1000, "generation": 500 + len(calls)}

    svc._backend.probe_progress_counters = probe
    return calls


async def _drain(resp, timeout_s: float = 20.0) -> str:
    frames: list[str] = []

    async def go():
        async for ch in resp.body_iterator:
            frames.append(ch.decode() if isinstance(ch, (bytes, bytearray)) else ch)
    await asyncio.wait_for(go(), timeout=timeout_s)
    return "".join(frames)


# --- 1. the wedge must still die -------------------------------------------

@pytest.mark.asyncio
async def test_frozen_counters_still_abort(monkeypatch):
    """The true-wedge signature: BOTH counters frozen while the wire is silent.

    This is the test that keeps the fix honest. If it ever goes green by
    hanging instead of aborting, C6 has become a way to never notice a dead
    backend — strictly worse than the false positives it was built to stop.
    """
    monkeypatch.setattr(lifecycle_mod, "_STREAM_INTERTOKEN_GAP_S", 1.5)
    svc = ProxyService(ProxyConfig())
    svc._backend.stream = _speaks_then_silent(60.0, then_done=False)
    _stub_probes(svc)
    calls = _counters(svc, "frozen")
    await svc.startup()
    try:
        _pin_default_timeout(svc, 30.0)     # generous: only the gap can kill it
        resp = await svc.handle_submit(_body(), _FakeRequest())
        t0 = time.monotonic()
        out = await _drain(resp, timeout_s=15.0)
        elapsed = time.monotonic() - t0

        assert "stalled mid-stream" in out, out[-400:]
        assert elapsed < 6.0, f"wedge detection regressed — took {elapsed:.2f}s"
        assert calls, "the progress probe never ran; test proves nothing"
        assert svc._state.stream_progress_extensions == 0
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_absent_counters_fall_back_to_aborting(monkeypatch):
    """A backend that exposes no counters (llama.cpp, unreachable, non-200)
    must behave EXACTLY as it did before C6. `None` means "cannot
    discriminate", and reading it as evidence of life would silently disable
    the stall guard for every non-vLLM endpoint in the fleet."""
    monkeypatch.setattr(lifecycle_mod, "_STREAM_INTERTOKEN_GAP_S", 1.5)
    svc = ProxyService(ProxyConfig())
    svc._backend.stream = _speaks_then_silent(60.0, then_done=False)
    _stub_probes(svc)
    calls = _counters(svc, "absent")
    await svc.startup()
    try:
        _pin_default_timeout(svc, 30.0)
        resp = await svc.handle_submit(_body(), _FakeRequest())
        t0 = time.monotonic()
        out = await _drain(resp, timeout_s=15.0)
        elapsed = time.monotonic() - t0

        assert "stalled mid-stream" in out, out[-400:]
        assert elapsed < 6.0, f"took {elapsed:.2f}s"
        assert calls, "the progress probe never ran; test proves nothing"
        assert svc._state.stream_progress_extensions == 0
    finally:
        await svc.shutdown()


# --- 2. legitimate work must survive ----------------------------------------

@pytest.mark.asyncio
async def test_advancing_counters_extend_the_deadline(monkeypatch):
    """The dsh turn this was built for: silent for longer than the gap, but the
    backend is provably working, so it must finish rather than be killed."""
    monkeypatch.setattr(lifecycle_mod, "_STREAM_INTERTOKEN_GAP_S", 1.5)
    svc = ProxyService(ProxyConfig())
    # 4.0s of silence against a 1.5s gap: without the probe this is three gap
    # deadlines' worth of death, and the pre-C6 code killed it every time.
    svc._backend.stream = _speaks_then_silent(4.0)
    _stub_probes(svc)
    calls = _counters(svc, "advancing")
    await svc.startup()
    try:
        _pin_default_timeout(svc, 30.0)
        resp = await svc.handle_submit(_body(), _FakeRequest())
        out = await _drain(resp, timeout_s=20.0)

        assert '"done"' in out, out[-400:]
        assert "stalled mid-stream" not in out, out[-400:]
        assert calls, "the progress probe never ran; test proves nothing"
        # Counted, or the fix is unobservable in production.
        assert svc._state.stream_progress_extensions > 0
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_extensions_are_bounded(monkeypatch):
    """Advancing counters buy a BOUNDED reprieve, never an open one.

    The counters are engine-wide, so on a busy engine another request's tokens
    keep them moving while MY stream is wedged. That is precisely the case this
    bound exists for: the stream must still die.
    """
    monkeypatch.setattr(lifecycle_mod, "_STREAM_INTERTOKEN_GAP_S", 1.0)
    monkeypatch.setattr(lifecycle_mod, "_STREAM_GAP_MAX_EXTENSIONS", 2)
    svc = ProxyService(ProxyConfig())
    svc._backend.stream = _speaks_then_silent(120.0, then_done=False)
    _stub_probes(svc)
    _counters(svc, "advancing")          # never stops "proving" liveness
    await svc.startup()
    try:
        _pin_default_timeout(svc, 60.0)
        resp = await svc.handle_submit(_body(), _FakeRequest())
        t0 = time.monotonic()
        out = await _drain(resp, timeout_s=25.0)
        elapsed = time.monotonic() - t0

        assert "stalled mid-stream" in out, out[-400:]
        # 2 extensions x a 1s gap, plus the original gap and probe cadence.
        assert elapsed < 12.0, f"extensions were not bounded — {elapsed:.2f}s"
        assert svc._state.stream_progress_extensions == 2
    finally:
        await svc.shutdown()


# --- 3. the tools-aware base gap --------------------------------------------

@pytest.mark.asyncio
async def test_tool_carrying_request_gets_the_wider_gap(monkeypatch):
    """A request carrying `tools` has a legitimately bursty wire (measured:
    14.23 s of silence mid-answer on an idle engine), so it starts from the
    wider base gap.

    Both halves run with counters ABSENT, which pins the gap itself rather than
    the progress probe: the same silence must kill the tool-less stream and
    spare the tool-carrying one.
    """
    monkeypatch.setattr(lifecycle_mod, "_STREAM_INTERTOKEN_GAP_S", 1.0)
    monkeypatch.setattr(lifecycle_mod, "_STREAM_GAP_TOOLS_S", 8.0)

    async def run(tools: bool) -> str:
        svc = ProxyService(ProxyConfig())
        svc._backend.stream = _speaks_then_silent(3.0)   # between the two gaps
        _stub_probes(svc)
        _counters(svc, "absent")
        await svc.startup()
        try:
            _pin_default_timeout(svc, 30.0)
            resp = await svc.handle_submit(_body(tools=tools), _FakeRequest())
            return await _drain(resp, timeout_s=20.0)
        finally:
            await svc.shutdown()

    without = await run(False)
    assert "stalled mid-stream" in without, without[-400:]

    with_tools = await run(True)
    assert '"done"' in with_tools, with_tools[-400:]
    assert "stalled mid-stream" not in with_tools, with_tools[-400:]
