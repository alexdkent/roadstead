"""The thinking canary must never block the capacity poller (2026-09-26).

`Health._maybe_run_thinking_canary` makes a REAL generation call and used to be
AWAITED INLINE inside `poll_endpoint_once`, which `poller_iteration` calls
sequentially for every endpoint in one pass. A canary running past the goodput
detector's `_RATE_WINDOW_S` (25s, `goodput.py`) delayed every LATER endpoint's
own `sample_goodput` call past its window too — read UNKNOWN
(`too_few_samples`) for an endpoint the canary never touched, and every other
poller-derived state (capacity discovery, the circuit breaker, WAL/retention
chores) went stale for the same stretch. Observed live 08:32:53Z and
09:33:08Z.

The fix (`Health._schedule_thinking_canary`) is to run it as its own task
rather than block the pass that every other endpoint depends on. These tests
pin three things: scheduling does not block the caller, at most one canary is
ever in flight per endpoint, and a canary that raises is contained — none of
which the pre-existing canary tests in `test_model_swap_guards.py` (which call
`_maybe_run_thinking_canary` directly, never through the poller) could catch.
"""

from __future__ import annotations

import asyncio
import time
import types

import pytest

from roadstead.config import EndpointConfig
from roadstead.flags import RuntimeFlags
from roadstead.goodput import GoodputMonitor
from roadstead.health import Health


def _state(**over):
    """Same shape as `test_goodput_wiring.py`'s `_state()` — everything
    `endpoint_healthy`/`sample_goodput`/`poll_endpoint_once` read, nothing
    production doesn't have. Duplicated rather than imported: the two files
    test different concerns and neither should have to change because the
    other's fixture grew a field.
    """
    state = types.SimpleNamespace(
        endpoint_failure_times={}, endpoint_cooldown_until={},
        endpoint_cooldown_trips={}, cooldown_best_effort_skips={},
        paused_endpoints=set(), endpoint_health={},
        collapsed_endpoints=set(), goodput=GoodputMonitor(),
        goodput_verdicts={}, flags=RuntimeFlags(None),
        on_demand=types.SimpleNamespace(manages=lambda ep: False,
                                        is_loaded=lambda ep: True),
        scheduler=types.SimpleNamespace(queued_requests=lambda ep, bands: []),
        dispatch_event=types.SimpleNamespace(set=lambda: None),
        failover=None,
        # The dict this fix adds — `_schedule_thinking_canary` reads/writes it.
        thinking_canary_tasks={},
    )
    for k, v in over.items():
        setattr(state, k, v)
    return state


def _ep(**over) -> EndpointConfig:
    kw = dict(endpoint_class="tier3", role="reasoner",
              thinking_kwargs=("thinking",),
              # Non-zero: skips the "never probe on the first poll after
              # startup" arm-the-clock branch, so a due call actually fires.
              thinking_canary_checked_at=1.0)
    kw.update(over)
    return EndpointConfig(**kw)


class _SlowCanaryBackend:
    """`probe_thinking_switch` sleeps `sleep_s` before answering — standing in
    for a real generation slow enough to cross the goodput rate window."""

    def __init__(self, sleep_s: float) -> None:
        self.sleep_s = sleep_s
        self.thinking_calls = 0
        self.goodput_calls = 0

    async def probe_health(self, ep_cfg):
        return True

    async def probe_thinking_switch(self, ep_cfg, key, nonce):
        self.thinking_calls += 1
        await asyncio.sleep(self.sleep_s)
        return {"reasoning_chars": 10, "content_chars": 0}

    async def probe_progress_counters(self, ep_cfg):
        self.goodput_calls += 1
        return {"iterations": None, "generation": None, "prompt": None,
                "running": 3}


class _RaisingCanaryBackend:
    async def probe_health(self, ep_cfg):
        return True

    async def probe_thinking_switch(self, ep_cfg, key, nonce):
        raise RuntimeError("backend exploded mid-generation")


# --------------------------------------------------------------------------- #
# 1. Scheduling returns immediately — the caller never waits on the probe.
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_scheduling_does_not_block_the_caller():
    backend = _SlowCanaryBackend(sleep_s=0.3)
    state = _state(backend=backend)
    health = Health(state)

    t0 = time.monotonic()
    health._schedule_thinking_canary("tier3", _ep())
    elapsed = time.monotonic() - t0

    assert elapsed < 0.05, (
        f"_schedule_thinking_canary blocked for {elapsed:.3f}s — it must return "
        f"the instant a task is created, never wait on the probe")
    assert backend.thinking_calls == 0, (
        "the probe must not have even started yet — create_task does not run "
        "the coroutine until the caller yields control")

    task = state.thinking_canary_tasks["tier3"]
    await asyncio.wait_for(task, timeout=1.0)
    assert backend.thinking_calls == 1, "the scheduled task must still have run"


@pytest.mark.asyncio
async def test_a_slow_canary_never_delays_a_later_endpoints_goodput_sample():
    """🚨 THE REGRESSION ITSELF, reproduced through the real poller entry point.

    tier3's canary is due and slow; tier2 is a goodput-armed endpoint that must
    still be sampled promptly, exactly as `poller_iteration`'s sequential
    per-endpoint loop drives it — `poll_endpoint_once(tier3)` immediately
    followed by `poll_endpoint_once(tier2)`, with nothing awaited in between
    but the scheduling call itself.
    """
    backend = _SlowCanaryBackend(sleep_s=0.3)
    state = _state(backend=backend)
    health = Health(state)

    tier3 = _ep(endpoint_class="tier3", skip_discovery=True, kind="embed")
    tier2 = _ep(endpoint_class="tier2", role="chat", skip_discovery=True,
               kind="embed", thinking_kwargs=(),
               goodput_min_running=1, goodput_sustain_evaluations=1,
               goodput_max_iteration_rate=1.0)

    t0 = time.monotonic()
    await health.poll_endpoint_once("tier3", tier3)
    await health.poll_endpoint_once("tier2", tier2)
    elapsed = time.monotonic() - t0

    assert elapsed < 0.05, (
        f"both polls together took {elapsed:.3f}s — tier2's poll waited on "
        f"tier3's canary, which is the exact bug this fix removes")
    assert backend.goodput_calls == 1, "tier2 must have been sampled"
    assert "tier2" in state.goodput_verdicts, (
        "tier2's sample never landed — this is 'too_few_samples' reproduced")

    # The canary itself still completes, off to the side.
    task = state.thinking_canary_tasks["tier3"]
    await asyncio.wait_for(task, timeout=1.0)
    assert backend.thinking_calls == 1


# --------------------------------------------------------------------------- #
# 2. At most one canary in flight per endpoint.
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_a_second_schedule_call_does_not_start_a_second_task_in_flight():
    backend = _SlowCanaryBackend(sleep_s=0.3)
    state = _state(backend=backend)
    health = Health(state)
    ep = _ep()

    health._schedule_thinking_canary("tier3", ep)
    first_task = state.thinking_canary_tasks["tier3"]
    # The interval hasn't elapsed again (checked_at was just stamped), but even
    # if it had, a scheduling call while one is in flight must be a no-op —
    # simulate that directly by resetting the interval gate and re-scheduling.
    ep.thinking_canary_checked_at = 1.0
    health._schedule_thinking_canary("tier3", ep)

    assert state.thinking_canary_tasks["tier3"] is first_task, (
        "a second in-flight task replaced the first — at most one per "
        "endpoint was supposed to be scheduled")

    await asyncio.wait_for(first_task, timeout=1.0)
    assert backend.thinking_calls == 1, (
        f"expected exactly one probe call, got {backend.thinking_calls} — the "
        f"in-flight guard let a duplicate through")


@pytest.mark.asyncio
async def test_a_new_canary_can_be_scheduled_once_the_previous_one_finished():
    backend = _SlowCanaryBackend(sleep_s=0.0)
    state = _state(backend=backend)
    health = Health(state)
    ep = _ep()

    health._schedule_thinking_canary("tier3", ep)
    await asyncio.wait_for(state.thinking_canary_tasks["tier3"], timeout=1.0)

    ep.thinking_canary_checked_at = 1.0  # due again
    health._schedule_thinking_canary("tier3", ep)
    await asyncio.wait_for(state.thinking_canary_tasks["tier3"], timeout=1.0)

    assert backend.thinking_calls == 2, (
        "a finished canary must not block the NEXT one from ever being "
        "scheduled — only an IN-FLIGHT one should")


@pytest.mark.asyncio
async def test_a_second_endpoints_canary_is_independent():
    backend = _SlowCanaryBackend(sleep_s=0.3)
    state = _state(backend=backend)
    health = Health(state)

    health._schedule_thinking_canary("tier3", _ep())
    health._schedule_thinking_canary("tier2", _ep(endpoint_class="tier2"))

    assert set(state.thinking_canary_tasks) == {"tier3", "tier2"}, (
        "the in-flight guard is per endpoint, not global")
    await asyncio.wait_for(
        asyncio.gather(*state.thinking_canary_tasks.values()), timeout=1.0)
    assert backend.thinking_calls == 2


# --------------------------------------------------------------------------- #
# 3. Exceptions are contained — a bare background task must never escape.
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_a_raising_probe_does_not_escape_the_background_task():
    state = _state(backend=_RaisingCanaryBackend())
    health = Health(state)

    health._schedule_thinking_canary("tier3", _ep())
    task = state.thinking_canary_tasks["tier3"]
    # If the exception escaped the task, awaiting it here would re-raise it.
    await asyncio.wait_for(task, timeout=1.0)
    assert task.exception() is None, (
        "the canary's own try/except must have caught the RuntimeError — a "
        "bare background task that raises is an 'unhandled exception in a "
        "Task' warning nothing else in this fix guards against")


@pytest.mark.asyncio
async def test_done_callback_prunes_the_task_whether_it_failed_or_not():
    state = _state(backend=_RaisingCanaryBackend())
    health = Health(state)

    health._schedule_thinking_canary("tier3", _ep())
    task = state.thinking_canary_tasks["tier3"]
    await asyncio.wait_for(task, timeout=1.0)
    # add_done_callback runs on the next loop iteration after completion.
    await asyncio.sleep(0)

    assert "tier3" not in state.thinking_canary_tasks, (
        "a finished (even failed) task must be pruned, or the in-flight guard "
        "would wrongly believe one is still running forever")
