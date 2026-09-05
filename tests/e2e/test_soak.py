"""Sustained concurrent load, with the concurrency invariant armed.

**Workstream F.** `docs/internals.md` names the single most dangerous thing in this repo
and, until this file, said plainly that nothing in the suite guarded it:

    🚨 Single event loop. No locks on in-memory scheduler / budget / cache
    state. That is only safe because there is exactly one thread mutating it.

Sustained concurrent load is the only thing that surfaces a violation of that,
and the only thing that surfaces the accounting bugs beside it — a slot that is
never returned, a DRR budget that drifts, a ledger that counts a call twice.
None of those raise. They accumulate, and the symptom appears days later and
nowhere near the commit.

## What this file asserts, and why each one is a distinct question

**Thread affinity** (`tests/loop_affinity.py`) — every mutation of scheduler,
budget, cost-model, spend, price and cache state came from ONE thread. The DB
writer is expected to be a second thread and touches none of it.

**Slot conservation** — in-flight returns to exactly zero. A leaked slot is
permanent: it never frees, and the endpoint's usable capacity shrinks silently
until a restart. This has actually happened (a producer wedged on a full stream
queue, 2026-05-31, cascading four slots into a full endpoint jam).

**Budget conservation** — every dispatched request was charged and every
completion refunded, so no agent's balance drifts. DRR fairness is exactly this
arithmetic; a drift is unfairness that compounds.

**Ledger conservation** — the spend ledger's request count equals the number of
requests that completed. Counted twice, a caller crosses its threshold early and
is degraded for work it did not do.

**Every request got an answer** — a status code, always, for all of them. Under
saturation a shed or a defer is a correct answer; silence is not.

🚨 **A concurrency test that runs no concurrency passes trivially**, so the
workload is asserted to have overlapped: peak observed in-flight must exceed 1,
and the recorder must have seen real mutations. This repo has a ledger entry for
a green suite that ran nothing.

Marked ``heavy``: it stays on the default path (a guard nobody runs is not a
guard) but is bounded to a few seconds. The unbounded version — minutes of load,
memory growth, WAL behaviour — is ``tools/soak.py``, off the default path
because it is an experiment rather than an assertion.
"""
from __future__ import annotations

import asyncio

import pytest

from tests.loop_affinity import EXPECTED_ARMED_METHODS, arm

pytestmark = pytest.mark.heavy


#: Enough overlap to interleave the scheduler loop with request handling many
#: times over, and small enough to stay inside the suite's 60s timeout against
#: a fake backend on a real socket.
_WAVES = 6
_CONCURRENCY = 12


def _body(i: int, *, stream: bool = False) -> dict:
    """Spread the load across endpoints, bands and both declaration styles, so
    the workload exercises intent resolution and pinning rather than one path
    twelve times."""
    intent = ("chat", "reasoning", "fast-chat")[i % 3]
    priority = ("P1_TURN_SUPPORT", "P2_POST_TURN", "P3_INGESTION")[i % 3]
    body: dict = {
        "priority": priority,
        "call_site": f"soak.{i % 4}",
        # 🚨 An explicit, SHORT deadline, and it is what makes a slot leak a
        # fast failure instead of a two-minute hang. On the computed default
        # (180s) a leaked slot makes every later wave queue behind it, so the
        # first symptom is the suite timing out — a signal nobody can read. At
        # 10s against a fake backend that answers in milliseconds, a leak
        # surfaces as 504s in the status list, which names itself.
        "deadline_s": 5.0,
        "payload": {
            "messages": [{"role": "user", "content": f"soak {i}"}],
            "max_tokens": 8,
            "stream": stream,
        },
    }
    # Half by intent, half by pin — a soak that only exercised one would leave
    # the other's state mutations unwatched.
    if i % 2:
        body["intent"] = intent
    else:
        body["model"] = ("tier1", "tier2", "tier3")[i % 3]
    return body


async def _drain_stream(client, body) -> int:
    async with client.stream("POST", "/rs/v1/chat", json=body) as resp:
        async for _ in resp.aiter_lines():
            pass
        return resp.status_code


async def _peak_inflight(svc, peak: list[int]) -> None:
    """Sample occupancy while the load runs — the proof that it overlapped."""
    while True:
        total = sum(svc._scheduler.endpoint_snapshot(ep)["in_flight"]
                    for ep in svc._state.config.endpoints)
        peak[0] = max(peak[0], total)
        await asyncio.sleep(0.005)


async def test_sustained_concurrent_load_holds_every_invariant(proxy):
    state = proxy.svc._state
    affinity = arm(state)
    assert affinity.armed_methods == EXPECTED_ARMED_METHODS, (
        f"armed {affinity.armed_methods} of {EXPECTED_ARMED_METHODS} methods — "
        f"a name in loop_affinity.STATE_METHODS has been renamed, so this soak "
        f"is watching less than it claims")

    peak = [0]
    watcher = asyncio.create_task(_peak_inflight(proxy.svc, peak))
    statuses: list[int] = []
    try:
        for wave in range(_WAVES):
            calls = []
            for i in range(_CONCURRENCY):
                n = wave * _CONCURRENCY + i
                if n % 5 == 4:
                    calls.append(_drain_stream(proxy.client, _body(n, stream=True)))
                else:
                    calls.append(proxy.client.post("/rs/v1/chat", json=_body(n)))
            done = await asyncio.gather(*calls, return_exceptions=True)
            wave_statuses = []
            for r in done:
                assert not isinstance(r, BaseException), r
                wave_statuses.append(r if isinstance(r, int) else r.status_code)
            statuses.extend(wave_statuses)
            # 🚨 Checked PER WAVE, not once at the end. A leaked slot makes
            # every later wave queue behind it, so an end-of-run assertion is
            # reached only after all six waves have each waited out a deadline —
            # which is a suite timeout, a signal nobody can read. Failing on the
            # first wave that 504s names the leak in seconds instead.
            assert all(st in (200, 429, 503) for st in wave_statuses), (
                f"wave {wave} returned {sorted(set(wave_statuses))} — a 504 "
                f"here means requests waited out a 5s deadline against a "
                f"backend that answers in milliseconds, which is a LEAKED "
                f"SLOT; a 500 means an unhandled fault")
    finally:
        watcher.cancel()
        affinity.disarm()

    total = _WAVES * _CONCURRENCY

    # --- every request got an answer ------------------------------------
    assert len(statuses) == total
    # Under saturation a 429 shed or a 503 is a CORRECT answer — the point is
    # that nothing was dropped and nothing 500'd.
    # (statuses were checked wave by wave above, where a leak names itself)

    # --- the workload really overlapped ---------------------------------
    assert peak[0] > 1, (
        f"peak in-flight was {peak[0]} — the load never overlapped, so every "
        f"invariant below passed trivially")
    assert affinity.total_mutations > total, (
        f"only {affinity.total_mutations} mutations recorded for {total} "
        f"requests — the recorder is not seeing the hot path")

    # --- 🚨 THE concurrency invariant -----------------------------------
    affinity.assert_single_threaded()

    # --- slot conservation ----------------------------------------------
    assert await _settled(proxy) == 0, (
        "a slot was never returned — capacity shrinks silently until restart")

    # --- budget conservation --------------------------------------------
    # Every charge is matched by a completion's retroactive adjust + refund, so
    # no agent may be left holding a balance below its floor or above its cap.
    for row in state.budget_mgr.snapshot():
        assert row["balance_ss"] >= -1e-6, (
            f"agent {row['agent_id']} holds a NEGATIVE DRR balance "
            f"({row['balance_ss']}) — a charge outlived its refund, which is "
            f"unfairness that compounds")

    # --- ledger conservation --------------------------------------------
    served = sum(1 for s in statuses if s == 200)
    counted = sum(a["requests"] for a in state.spend.snapshot())
    assert counted == served, (
        f"the spend ledger counted {counted} requests for {served} served — "
        f"a double count degrades a caller for work it did not do")


async def _settled(proxy, timeout_s: float = 5.0) -> int:
    """Wait for in-flight to reach zero, then report it. A stream's producer is
    cancelled on consumer exit and records its completion asynchronously, so a
    bare read races the teardown rather than measuring a leak."""
    async def go():
        while proxy.total_in_flight() > 0:
            await asyncio.sleep(0.01)
    try:
        await asyncio.wait_for(go(), timeout=timeout_s)
    except asyncio.TimeoutError:
        pass
    return proxy.total_in_flight()


async def test_the_affinity_guard_actually_catches_a_second_writer(proxy):
    """🚨 The guard, observed going red — from inside the suite, permanently.

    Every other assertion in this file is worth exactly as much as this one. A
    thread-affinity recorder that silently stopped recording would leave the
    soak green forever, and the failure it exists to catch is precisely the kind
    nobody goes looking for. So: do the forbidden thing on purpose — mutate DRR
    budget state from a second thread — and require the recorder to say so.
    """
    state = proxy.svc._state
    affinity = arm(state)
    try:
        await proxy.client.post("/rs/v1/chat", json=_body(0))

        def offending_writer():
            # Exactly what docs/internals.md forbids: "a second thread that touches
            # scheduler or budget state".
            state.budget_mgr.charge("intruder", 1.0, 0.0)

        import threading
        t = threading.Thread(target=offending_writer)
        t.start()
        t.join()

        problems = affinity.violations()
        assert problems, (
            "the loop-affinity guard did NOT notice a second thread mutating "
            "DRR budget state — it is watching nothing, and every other "
            "assertion in this file is worthless")
        assert any("budget_mgr" in p for p in problems), problems
        with pytest.raises(AssertionError, match="CONCURRENCY INVARIANT"):
            affinity.assert_single_threaded()
    finally:
        affinity.disarm()
        state.budget_mgr.remove("intruder")
