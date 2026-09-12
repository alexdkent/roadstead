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

from roadstead import lifecycle as lifecycle_mod
from roadstead.backend import BackendClientPool, BackendStreamEvent
from roadstead.config import EndpointConfig, ProxyConfig
from roadstead.service import ProxyService


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
        "agent_id": "a", "endpoint": "tier3", "priority": priority,
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
    """A backend that exposes no counters (unreachable, non-200, an engine with
    no `/metrics`) must behave EXACTLY as it did before C6. `None` means "cannot
    discriminate", and reading it as evidence of life would silently disable the
    stall guard wherever the probe cannot see. (llama.cpp is NOT such a backend
    — it publishes `llamacpp:n_decode_total`; see section 4.)"""
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


# --- 3b. the call-site COALESCE, driven directly ------------------------------
#
# 🚨 THIS SECTION EXISTS BECAUSE DELETING THE PRODUCTION LINE LEFT THE ENTIRE
# SUITE GREEN. `probe_progress_counters` returns `None` for an absent counter
# (`goodput.py` must be able to tell a missing instrument from a stopped engine),
# so THIS caller — which wants the opposite default — coalesces at its own call
# site: `counters = {k: (v or 0) for k, v in raw.items()}`. Nothing held it there.
#
# Neither existing candidate could see it. `test_absent_counters_fall_back_to_
# aborting` uses the whole-dict `None`, which returns at `if raw is None` BEFORE
# the coalesce — a different branch. And a test in the wiring file re-implemented
# the coalesce in its own body, so the production line was unreachable from it.
#
# ⚠️ AND A NAIVE END-TO-END TEST CANNOT SEE IT EITHER, which is why these drive
# `_stream_progress_probe` DIRECTLY. The probe runs in `asyncio.create_task`, so
# with the coalesce deleted the real `TypeError: '>' not supported between
# instances of 'NoneType' and 'NoneType'` is SWALLOWED by the task: the gap
# deadline fires, the stream aborts, and the observable behaviour is unchanged and
# safe. What is lost is SILENT — the watchdog task is dead, so every LEGITIMATE
# extension is gone on any endpoint publishing a partial counter set. So the
# assertion has to be "the probe returned cleanly", not "the stream did the right
# thing".

class _RecordingTimeout:
    """Stands in for the `asyncio.Timeout` the probe reschedules.

    Counting reschedules is the whole measurement: a probe that DIED cannot have
    made one, and a probe that ran to completion on frozen counters must not.
    """

    def __init__(self) -> None:
        self.reschedules = 0

    def reschedule(self, when: float) -> None:
        self.reschedules += 1


async def _run_probe(samples, *, gap_s: float = 0.4):
    """Drive `_stream_progress_probe` directly and report how it ended.

    Returns ``(exception_or_None, reschedule_count, probe_call_count)``. The
    exception is CAPTURED rather than allowed to vanish into a task, which is the
    entire reason this helper exists.
    """
    svc = ProxyService(ProxyConfig())
    _stub_probes(svc)
    calls: list[int] = []

    async def probe(ep_cfg):
        calls.append(1)
        return samples[min(len(calls) - 1, len(samples) - 1)]

    svc._backend.probe_progress_counters = probe
    cm = _RecordingTimeout()
    ep_cfg = next(iter(svc._config.endpoints.values()))
    t0 = time.monotonic()

    class _Req:
        endpoint = "tier3"
        agent_id = "a"
        request_id = "r"

    err: BaseException | None = None
    try:
        await asyncio.wait_for(
            svc._lifecycle._stream_progress_probe(
                # `get_last_chunk_at` returns t0 forever = the wire has been
                # silent since the start, which is the state the probe judges.
                # `get_ttft_ms` is non-None so the TTFT watchdog has handed over.
                ep_cfg, cm, lambda: t0, lambda: 12.0,
                gap_s, 600.0, t0, _Req(),
            ),
            timeout=gap_s * 8,
        )
    except asyncio.TimeoutError:
        pass          # the probe is a loop; running out the clock is normal
    except BaseException as exc:  # noqa: BLE001 — capturing it IS the test
        err = exc
    return err, cm.reschedules, len(calls)


@pytest.mark.asyncio
async def test_the_probe_returns_cleanly_when_a_counter_is_absent(monkeypatch):
    """🚨 THE GUARD FOR THE CALL-SITE COALESCE. Delete it and this goes red.

    A backend publishing occupancy but no token counters — the partial shape
    `probe_progress_counters` now reports as `None` per key. Without the coalesce
    the comparison is `None > None`, a `TypeError`, and the probe dies silently
    inside its task. Here the exception has nowhere to hide.

    Absent-reads-as-0-twice is the "cannot discriminate" case, so the probe must
    also decline to extend: a clean return with ZERO reschedules.
    """
    monkeypatch.setattr(lifecycle_mod, "_STREAM_INTERTOKEN_GAP_S", 0.4)
    partial = {"prompt": None, "generation": None,
               "iterations": None, "running": 6}
    err, reschedules, calls = await _run_probe([partial] * 4)

    assert err is None, (
        f"the progress probe raised {err!r} on a PARTIAL counter set. Without the "
        f"`or 0` coalesce at its call site this is `None > None` — and in "
        f"production it is swallowed by asyncio.create_task, so the watchdog task "
        f"dies silently and every legitimate extension is lost on any endpoint "
        f"that publishes a partial counter set.")
    assert calls >= 2, "the probe never sampled twice; the test proves nothing"
    assert reschedules == 0, (
        "absent counters are 'cannot discriminate' — the probe must decline to "
        "extend, not invent evidence of life")


@pytest.mark.asyncio
async def test_the_probe_still_extends_on_a_fully_published_advance(monkeypatch):
    """🚨 THE POSITIVE CONTROL for the test above.

    Without it, that test passes on a probe that returns immediately and can never
    reschedule anything — and "zero reschedules" would then mean nothing. Same
    driver, complete counters that ADVANCE: this one must reschedule.
    """
    monkeypatch.setattr(lifecycle_mod, "_STREAM_INTERTOKEN_GAP_S", 0.4)
    advancing = [{"prompt": 100 + i * 50, "generation": 10 + i * 5,
                  "iterations": 7 + i, "running": 6} for i in range(8)]
    err, reschedules, calls = await _run_probe(advancing)

    assert err is None, f"the probe raised {err!r} on a complete counter set"
    assert calls >= 2
    assert reschedules > 0, "advancing counters must buy an extension"


@pytest.mark.asyncio
async def test_the_probe_tolerates_a_mixed_partial_set(monkeypatch):
    """The realistic partial shape: prefill advancing, decode counter absent.

    An endpoint publishing `prompt` and not `generation` must still be usable by
    the per-request watchdog — the coalesce makes the absent key a constant 0, so
    the OTHER key's advance is what decides. A `TypeError` here would take the
    watchdog down on a backend that is provably working.
    """
    monkeypatch.setattr(lifecycle_mod, "_STREAM_INTERTOKEN_GAP_S", 0.4)
    mixed = [{"prompt": 100 + i * 50, "generation": None,
              "iterations": None, "running": 6} for i in range(8)]
    err, reschedules, calls = await _run_probe(mixed)

    assert err is None, f"the probe raised {err!r} on a mixed partial set"
    assert calls >= 2
    assert reschedules > 0, (
        "prefill IS advancing — the stream is doing real work and must not lose "
        "its extension because a second counter is unpublished")


# --- 4. WHICH counter, per engine family -------------------------------------
#
# The discriminator is only as good as the counter it reads, and llama.cpp has
# a same-sounding counter that is WRONG for this purpose. These pin the choice.

def _probe_returning(body: str):
    """A BackendClientPool whose `/metrics` returns ``body``."""
    pool = BackendClientPool.__new__(BackendClientPool)

    class _Resp:
        status_code = 200
        text = body

    class _Client:
        async def get(self, path):
            assert path == "/metrics", path
            return _Resp()

    pool._client_for = lambda base_url, min_pool=0: _Client()
    return pool


#: 🚨 Captured UNBOUND at import, BEFORE the autouse `_no_network_probes`
#: fixture replaces it on the class. `probe_progress_counters` moved into
#: `STUBBED_PROBES` on 2026-09-12 (the poller reaches it now — see
#: `test_model_swap_guards.py::test_every_backend_probe_is_stubbed`), so a bound
#: `pool.probe_progress_counters(...)` in this file would exercise the STUB and
#: assert nothing about the parser. This is the pattern that note points at.
_REAL_COUNTERS = BackendClientPool.probe_progress_counters

_EP = EndpointConfig(endpoint_class="x", role="x")

_VLLM = """# HELP vllm:prompt_tokens_total x
vllm:prompt_tokens_total{engine="0",model_name="m"} 7815245.0
vllm:generation_tokens_total{engine="0",model_name="m"} 358688.0
"""

# Real shape, copied from a live llama.cpp backend. Note `n_decode_total`
# carries no labels and no `tokens_total` substring — an earlier cut of this
# parser filtered on `"tokens_total" in line` and would have dropped it.
_LLAMACPP = """# HELP llamacpp:prompt_tokens_total x
llamacpp:prompt_tokens_total 1.90511e+06
llamacpp:tokens_predicted_total 420506
llamacpp:n_decode_total 196408
"""


@pytest.mark.asyncio
async def test_vllm_counters_are_read():
    """The two C6 counters, with every ABSENT key as `None`.

    🚨 **ONE RULE, NO ASYMMETRY: the parser returns absence as `None` for EVERY
    counter.** A missing series and a real zero are different facts, and
    `goodput.py` reaches opposite conclusions from them — "the instrument is
    missing" must never read as "the engine has stopped", because only the second
    sheds traffic.

    THIS caller wants the opposite default, and gets it by coalescing at its own
    call site (`lifecycle.py`): for a SINGLE request, an unpublished counter is not
    evidence of life, so it counts as no progress and the stall abort proceeds as
    it did before C6. Two consumers, one number, opposite defaults — each stated
    where it is applied.

    ⚠️ This docstring described the opposite contract for a day — "the two original
    keys coalesce to 0 … while the two new ones must be `None`" — which is the
    defect, not the design. The `or 0` in the parser made a missing `prompt` or
    `generation` satisfy two of the detector's four clauses, latching the breaker on
    an endpoint nobody had measured. Prose that documents a removed bug as the
    contract will talk the next reader into restoring it.
    """
    got = await _REAL_COUNTERS(_probe_returning(_VLLM), _EP)
    assert got == {"prompt": 7815245, "generation": 358688,
                   "iterations": None, "running": None}


@pytest.mark.asyncio
async def test_llamacpp_generation_reads_n_decode_not_tokens_predicted():
    """🚨 The trap this test exists for.

    MEASURED on the live boxa during a 400-token generation, sampled every 3 s:
    `tokens_predicted_total` went +0, +0, +0 and then +400 AT COMPLETION, while
    `n_decode_total` went +24, +66, +66, +65. `tokens_predicted_total` is
    credited when the request finishes, so it reads FROZEN for exactly the
    window the watchdog judges — mapping it to vLLM's `generation_tokens_total`
    by name would make the probe report "no progress" on every llama.cpp
    endpoint, and C6 would protect nothing while looking green.
    """
    got = await _REAL_COUNTERS(_probe_returning(_LLAMACPP), _EP)
    assert got is not None, "llama.cpp must not read as 'cannot discriminate'"
    assert got["generation"] == 196408, (
        f"generation must come from n_decode_total; got {got['generation']} "
        f"(420506 means it read tokens_predicted_total, which is frozen "
        f"mid-generation)")
    assert got["prompt"] == 1905110      # 6-sig-fig float parsed, not dropped


@pytest.mark.asyncio
async def test_backend_without_counters_reads_as_cannot_discriminate():
    """Absence must stay distinguishable from zero — the watchdog falls back to
    aborting on None, and a 0 would look like a frozen backend instead."""
    got = await _REAL_COUNTERS(_probe_returning("# nothing useful here\nfoo_bar 1\n"
                                 ), _EP)
    assert got is None
