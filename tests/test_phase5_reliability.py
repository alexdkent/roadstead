"""Phase 5B — proxy reliability hardening.

  5B.1 streaming slot-leak: when the SSE consumer disconnects, the producer
       dispatch task is cancelled so it can't wedge on a full stream_q.put and
       hold the scheduler slot forever (the live companion-jam bug, 2026-05-31).
  5B.2 drain: a straggler that exceeds the drain deadline is cancelled AND its
       CancelledError handler frees the slot (records the completion).
  5B.4 budget restore: a configured agent's cap/weight win over the persisted
       row on restart.
"""

from __future__ import annotations

import asyncio

import pytest

from originfleet.llmproxy import service as service_mod
from originfleet.llmproxy.backend import BackendStreamEvent
from originfleet.llmproxy.config import AgentQuotaConfig, ProxyConfig
from originfleet.llmproxy.queue import PersistentQueue
from originfleet.llmproxy.service import ProxyService


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


def _body(stream=False, timeout_s=10.0):
    return {
        "agent_id": "a", "endpoint": "llama-thinker", "priority": "P3_INGESTION",
        "call_site": "t", "payload_type": "chat_completion",
        "payload": {"messages": [{"role": "user", "content": "x"}], "stream": stream},
        "timeout_s": timeout_s,
    }


# --- 5B.1 streaming slot-leak ----------------------------------------------

@pytest.mark.asyncio
async def test_consumer_disconnect_cancels_producer_and_frees_slot():
    svc = ProxyService(ProxyConfig())

    async def slow_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        # Produces indefinitely so the producer is still streaming when the
        # consumer disconnects (and would wedge on a full stream_q without the fix).
        for i in range(100000):
            await asyncio.sleep(0.005)
            yield BackendStreamEvent(
                "chunk", '{"choices":[{"delta":{"content":"x"}}]}',
                {"choices": [{"delta": {"content": "x"}}]})

    svc._backend.stream = slow_stream
    _stub_probes(svc)
    await svc.startup()
    try:
        resp = await svc.handle_submit(_body(stream=True), _FakeRequest())
        it = resp.body_iterator

        # Read frames until a chunk arrives → the producer is dispatched, live,
        # and holding a scheduler slot.
        async def read_until_chunk():
            async for ch in it:
                t = ch.decode() if isinstance(ch, (bytes, bytearray)) else ch
                if '"chunk"' in t or '"delta"' in t:
                    return
        await asyncio.wait_for(read_until_chunk(), timeout=5.0)
        assert svc._scheduler.endpoint_snapshot("thinker")["in_flight"] >= 1

        # Consumer disconnects: close the SSE generator → finally cancels the
        # producer.
        await it.aclose()

        # Let the cancellation + CancelledError handler propagate.
        async def wait_freed():
            while svc._scheduler.endpoint_snapshot("thinker")["in_flight"] > 0:
                await asyncio.sleep(0.02)
        await asyncio.wait_for(wait_freed(), timeout=5.0)

        assert svc._slot_leak_reclaimed >= 1, "producer cancel must reclaim the slot"
        assert svc._scheduler.endpoint_snapshot("thinker")["in_flight"] == 0
    finally:
        await svc.shutdown()


# --- 5B.2 drain straggler cancel frees the slot -----------------------------

@pytest.mark.asyncio
async def test_drain_cancels_straggler_and_counts(monkeypatch):
    monkeypatch.setattr(service_mod, "_DRAIN_DEADLINE_S", 0.1)
    svc = ProxyService(ProxyConfig())
    _stub_probes(svc)
    await svc.startup()

    started = asyncio.Event()

    async def slow():
        started.set()
        await asyncio.sleep(30)  # far exceeds the 0.1s drain deadline

    task = asyncio.create_task(slow())
    svc._inflight_tasks["straggler"] = task
    task.add_done_callback(lambda t: svc._inflight_tasks.pop("straggler", None))
    await asyncio.wait_for(started.wait(), timeout=2.0)

    await svc.shutdown()  # drain deadline fires → cancel + awaited gather
    assert svc._drain_straggler_cancelled == 1
    assert task.cancelled() or task.done()


# --- 5B.4 budget restore honors configured cap/weight -----------------------

@pytest.mark.asyncio
async def test_budget_restore_uses_configured_cap(tmp_path):
    db = str(tmp_path / "q.db")
    # Persist a balance ABOVE the default 60.0 cap with a stale weight.
    pq = PersistentQueue(db)
    pq.save_budgets([{
        "agent_id": "agentX", "weight": 9.0,
        "balance_ss": 500.0, "total_consumed_ss": 12.0,
    }])
    pq.close()

    cfg = ProxyConfig(queue_db_path=db)
    cfg.agents["agentX"] = AgentQuotaConfig(
        agent_id="agentX", weight=2.0, max_balance_ss=300.0)
    svc = ProxyService(cfg)
    _stub_probes(svc)
    await svc.startup()
    try:
        b = svc._budget_mgr.get_or_create("agentX")
        assert b.max_balance == 300.0, "configured cap must win over default 60"
        assert b.weight == 2.0, "configured weight must win over persisted 9.0"
        assert b.balance == 300.0, "balance clamped to the configured cap (was 500)"
    finally:
        await svc.shutdown()
